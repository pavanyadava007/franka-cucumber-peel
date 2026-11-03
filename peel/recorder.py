"""Episode recording and conversion to RDP's zarr layout.

Raw episodes are .npz files (one per episode) with RDP key names. make_rdp_zarr() merges them into
<out>/replay_buffer.zarr with data/<key> arrays and meta/episode_ends, the layout RDP's ReplayBuffer reads
(see third_party/rdp/README.md "Data Postprocessing").

Two datasets come from the same recordings:
  * "e2e"    : whole peel, both arms, action = 20-D [l_xyz, l_rot6d, r_xyz, r_rot6d, l_grip, r_grip]
  * "stroke" : one clip per stroke (approach, descend, stroke, lift), peeler arm only,
               action = 10-D [r_xyz, r_rot6d, r_grip]. The left arm holds still during these phases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PHASES = ["start", "roll", "approach", "descend", "stroke", "lift", "return", "policy"]
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
LOWDIM_KEYS = [
    f"{s}_robot_{k}"
    for s in ("left", "right")
    for k in ("tcp_pose", "tcp_vel", "tcp_wrench", "gripper_width", "gripper_force", "joint_pos")
] + ["cucumber_roll", "coverage", "gouge", "blade_fn"]
IMAGE_KEYS = ["external_img", "right_wrist_img"]
STROKE_PHASES = {PHASE_ID[p] for p in ("approach", "descend", "stroke", "lift")}


class EpisodeRecorder:
    def __init__(self):
        self.buf: dict[str, list] = {}

    def add(self, obs: dict, action20: np.ndarray, phase: str, stroke_id: int, gello_cmd: np.ndarray | None):
        row = {k: obs[k] for k in LOWDIM_KEYS + IMAGE_KEYS if k in obs}
        row["action"] = np.asarray(action20, np.float32)
        row["target"] = row["action"]
        row["phase"] = np.array([PHASE_ID[phase]], np.int8)
        row["stroke_id"] = np.array([stroke_id], np.int16)
        row["gello_cmd"] = np.zeros(16, np.float32) if gello_cmd is None else np.asarray(gello_cmd, np.float32)
        for k, v in row.items():
            self.buf.setdefault(k, []).append(v)

    def __len__(self):
        return len(self.buf.get("action", []))

    def save(self, path: Path, meta: dict):
        arrs = {k: np.stack(v) for k, v in self.buf.items()}
        n = len(arrs["action"])
        arrs["timestamp"] = (np.arange(n) / 24.0).astype(np.float32)
        for k, v in meta.items():
            arrs[f"meta_{k}"] = np.asarray(v)
        np.savez_compressed(path, **arrs)


def action20_from_env(env) -> np.ndarray:
    """The commanded TCP targets of both arms (what the controller is tracking), RDP layout."""
    from .env import rot6d

    L, R = env.left, env.right
    return np.concatenate([L.tgt_p, rot6d(L.tgt_R), R.tgt_p, rot6d(R.tgt_R), [L.grip_w, R.grip_w]]).astype(np.float32)


def make_rdp_zarr(raw_files: list[Path], out_dir: Path, kind: str = "e2e", success_only: bool = True) -> dict:
    """Merge raw episodes into RDP's replay_buffer.zarr. Returns a summary dict."""
    import zarr
    from numcodecs import Blosc

    assert kind in ("e2e", "stroke")
    out_dir.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out_dir / "replay_buffer.zarr"), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    comp = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    ends, total, kept, skipped = [], 0, 0, 0
    arrays = {}
    for f in raw_files:
        ep = np.load(f)
        if success_only and not bool(ep["meta_success"]):
            skipped += 1
            continue
        segs = [np.arange(len(ep["action"]))]
        if kind == "stroke":
            ph, sid = ep["phase"][:, 0], ep["stroke_id"][:, 0]
            segs = []
            for s in np.unique(sid):
                idx = np.nonzero((sid == s) & np.isin(ph, list(STROKE_PHASES)))[0]
                if len(idx) > 20:
                    segs.append(idx)
        for idx in segs:
            chunk = {
                k: ep[k][idx]
                for k in LOWDIM_KEYS + IMAGE_KEYS + ["phase", "stroke_id", "gello_cmd", "timestamp"]
                if k in ep
            }
            a = ep["action"][idx]
            if kind == "stroke":
                a = np.concatenate([a[:, 9:18], a[:, 19:20]], axis=1)  # right arm: xyz, rot6d, grip
            chunk["action"] = a
            chunk["target"] = a
            for k, v in chunk.items():
                if k not in arrays:
                    is_img = v.ndim == 4
                    chunks = (1,) + v.shape[1:] if is_img else (4096,) + v.shape[1:]
                    arrays[k] = data.zeros(k, shape=(0,) + v.shape[1:], chunks=chunks, dtype=v.dtype, compressor=comp)
                arrays[k].append(v)
            total += len(idx)
            ends.append(total)
        kept += 1
    meta.array("episode_ends", np.array(ends, dtype=np.int64), dtype=np.int64, overwrite=True)
    return {"kind": kind, "episodes_used": kept, "episodes_skipped": skipped, "clips": len(ends), "frames": total}
