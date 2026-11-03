"""Run trained RDP / Diffusion Policy checkpoints in the simulation, the same way RDP's real-robot runner does.

Mirrors third_party/rdp/reactive_diffusion_policy/env_runner/real_runner.py, synchronously (no threads):
  * every `steps_per_inference` control steps (24 Hz / 6 Hz = 4) the slow policy runs on the last n_obs_steps frames
    (spaced by obs_temporal_downsample_ratio) and returns either an action chunk (DP) or a latent action (RDP)
  * every control step, RDP's asymmetric-tokenizer decoder turns the latent action into the current action using the
    wrench history since the chunk's first observation frame (the fast, force-reactive loop)
  * relative actions are converted back to absolute TCP targets with the pose at inference time

Episode structure (orchestrated skill execution): the orchestrator picks the side to peel and rolls the cucumber,
the learned stroke skill does approach -> touch -> press -> slide -> lift, then a scripted return to the park pose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RDP_DIR = ROOT / "third_party" / "rdp"


def _rdp_imports():
    for p in (str(ROOT / "rdp_shims"), str(RDP_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


class SimPolicy:
    """Loads an RDP-repo checkpoint (latent diffusion = RDP, or plain diffusion = DP)."""

    def __init__(self, ckpt_path: str, device: str = "cuda", num_inference_steps: int = 8):
        _rdp_imports()
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        cwd = os.getcwd()
        os.chdir(RDP_DIR)  # the workspace resolves some paths relative to the RDP repo
        try:
            ws = cls(cfg, output_dir=str(Path(ckpt_path).parents[1]))
            ws.load_payload(payload, exclude_keys=None, include_keys=None)
        finally:
            os.chdir(cwd)
        policy = ws.ema_model if cfg.training.use_ema else ws.model
        self.latent = "latent" in cfg.name
        if self.latent:
            policy.at.set_normalizer(policy.normalizer)
        self.device = torch.device(device)
        policy.eval().to(self.device)
        policy.num_inference_steps = num_inference_steps
        self.policy = policy
        self.torch = torch
        self.cfg = cfg
        self.shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.obs_ds = int(cfg.get("obs_temporal_downsample_ratio", 1))
        self.dataset_obs_ds = int(cfg.get("dataset_obs_temporal_downsample_ratio", 1))
        self.n_action_steps = int(cfg.n_action_steps)
        self.relative = bool(cfg.task.dataset.relative_action)
        self.To_full = self.n_obs_steps * self.dataset_obs_ds  # frames (at 24 Hz) covered by the observation window
        self.name = "RDP (force)" if self.latent else "DP (vision only)"


class StrokeSkillRunner:
    """Runs one learned stroke (approach..lift) for the peeler arm. Action = 10-D [r_xyz, r_rot6d, r_grip]."""

    def __init__(self, sp: SimPolicy, control_fps: int = 24, inference_fps: int = 6):
        _rdp_imports()
        from reactive_diffusion_policy.common.action_utils import (
            absolute_actions_to_relative_actions,
            relative_actions_to_absolute_actions,
        )
        from reactive_diffusion_policy.real_world.real_inference_util import get_real_obs_dict

        self.sp = sp
        self.spi = control_fps // inference_fps
        self.abs2rel = absolute_actions_to_relative_actions
        self.rel2abs = relative_actions_to_absolute_actions
        self.get_real_obs_dict = get_real_obs_dict

    def _window(self, hist: list, key: str, n: int, ds: int):
        idx = [max(len(hist) - 1 - ds * i, 0) for i in reversed(range(n))]
        return np.stack([hist[i][key] for i in idx])

    def _obs_dict(self, hist):
        sp = self.sp
        raw = {k: self._window(hist, k, sp.n_obs_steps, sp.obs_ds) for k in sp.shape_meta["obs"]}
        np_obs = self.get_real_obs_dict(env_obs=raw, shape_meta=sp.shape_meta)
        base = np.concatenate(
            [raw[k][-1] for k in ("left_robot_tcp_pose", "right_robot_tcp_pose") if k in sp.shape_meta["obs"]]
        )
        if sp.relative:
            for k in np_obs:
                if "robot_tcp_pose" in k:
                    np_obs[k] = self.abs2rel(np_obs[k], base_absolute_action=np_obs[k][-1])
        t = {k: sp.torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(sp.device) for k, v in np_obs.items()}
        return t, base

    def run(self, env, left_hold, max_steps: int = 160, on_step=None):
        """Execute one stroke. left_hold = (pos, R) target for the holding arm. Returns number of steps."""
        sp, torch = self.sp, self.sp.torch
        sp.policy.reset()
        hist: list[dict] = []
        chunk = None
        z_end = env.cfg.cucumber.peel_zone[1]
        o = env.obs()
        for k in range(max_steps):
            hist.append(o)
            if k % self.spi == 0:
                obs_t, base = self._obs_dict(hist)
                with torch.no_grad():
                    if sp.latent:
                        ad = sp.policy.predict_action(
                            obs_t, dataset_obs_temporal_downsample_ratio=sp.dataset_obs_ds, return_latent_action=True
                        )
                    else:
                        ad = sp.policy.predict_action(obs_t)
                acts = ad["action"][0].detach().cpu().numpy()
                chunk = {"acts": acts, "base": base, "start": k, "hist_start": max(len(hist) - sp.To_full, 0)}
            i = k - chunk["start"]
            if sp.latent:
                ext_step = sp.To_full + i
                wr = [h["right_robot_tcp_wrench"] for h in hist[chunk["hist_start"] :]]
                wr = [wr[0]] * max(0, ext_step - len(wr)) + wr  # pad at the start of the episode
                ext = {
                    "right_robot_tcp_wrench": torch.from_numpy(np.stack(wr[-ext_step:]).astype(np.float32)).unsqueeze(0)
                }
                lat = torch.from_numpy(chunk["acts"][min(i, len(chunk["acts"]) - 1)].astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    out = sp.policy.predict_from_latent_action(lat, ext, ext_step, sp.dataset_obs_ds)
                a = out["action"][0].detach().cpu().numpy()[-1]
            else:
                a = chunk["acts"][min(i, len(chunk["acts"]) - 1)]
            if sp.relative:
                a = self.rel2abs(a[None], chunk["base"])[0]
            from .env import rot6d_to_mat

            o = env.step_tcp(
                left=left_hold,
                right=(a[:3], rot6d_to_mat(a[3:9])),
                grip=(env.left.grip_w, float(np.clip(a[9], 0.0, 0.08))),
            )
            if on_step is not None:
                on_step(env, "policy", k)
            # skill termination: blade past the tip end and lifted off the skin
            pc, Rc = env.cucumber_frame()
            bp = env.blade_pos()
            bl = Rc.T @ (bp - pc)
            top, n, _ = env.surface_top(float(np.clip(bl[2], *env.shape.z_cyl)))
            height = float((bp - top) @ n)
            if k > 24 and bl[2] > z_end - 0.004 and height > 0.012:
                return k + 1
        return max_steps


def run_policy_episode(
    seed: int,
    policy: SimPolicy | None,
    shape_class: str | None = None,
    video=None,
    max_time: float = 150.0,
    max_strokes: int = 18,
):
    """Orchestrated episode: scripted roll/return + learned stroke skill. policy=None runs the scripted expert
    stroke instead (for checking the harness)."""
    from .env import EpisodeConfig, PeelEnv
    from .expert import ExpertParams, PeelExpert
    from .rollout import VideoSink, is_success

    cfg = EpisodeConfig.sample(seed, render_obs=True, shape_class=shape_class)
    env = PeelEnv(cfg)
    rng = np.random.default_rng(seed)
    ex = PeelExpert(env, ExpertParams.sample(rng), rng, gello=False)
    ex.orch.max_strokes = max_strokes
    runner = StrokeSkillRunner(policy) if policy is not None else None
    vid = (
        VideoSink(
            video,
            f"{policy.name if policy else 'expert'} stroke skill, {cfg.cucumber.shape_class} cucumber, seed {seed}",
            policy.name if policy else "expert",
        )
        if video
        else None
    )
    on_step = (lambda e, ph, k: vid.add(e, ph, ex.stroke_id)) if vid else None
    gen = ex.plan()
    try:
        phase, sid, left, right = next(gen)
    except StopIteration:
        phase = None
    while phase is not None and env.t < max_time:
        if phase in ("approach", "descend", "stroke", "lift") and runner is not None:
            # hand the stroke to the learned skill, then skip the expert's version of this stroke
            runner.run(env, left, on_step=on_step)
            ex._skip_stroke = True
            while phase in ("approach", "descend", "stroke", "lift"):
                try:
                    phase, sid, left, right = gen.send(None)
                except StopIteration:
                    phase = None
                    break
            continue
        a = np.concatenate([left[0], _r6(left[1]), right[0], _r6(right[1]), [2 * cfg.cucumber.shape.r_grip, 0.018]])
        env.step_action(a)
        if vid:
            vid.add(env, phase, sid)
        try:
            phase, sid, left, right = next(gen)
        except StopIteration:
            phase = None
    m = env.metrics()
    m["success"] = is_success(m)
    m["strokes"] = ex.orch.strokes
    m["shape_class"] = cfg.cucumber.shape_class
    if vid:
        vid.close()
    env.close()
    return m


def _r6(R):
    from .env import rot6d

    return rot6d(R)
