"""Inspect an RDP replay_buffer.zarr (read-only). Usage: python -I rdp_zarr_inspect.py <path/to/replay_buffer.zarr>"""
import sys
import numpy as np
import zarr

root = zarr.open(sys.argv[1], mode="r")
print("groups:", list(root.group_keys()))
for g in ["data", "meta"]:
    grp = root[g]
    print(f"[{g}] attrs={dict(grp.attrs)}")
    for k in sorted(grp.array_keys()):
        a = grp[k]
        print(f"  {g}/{k}: shape={a.shape} dtype={a.dtype} chunks={a.chunks} compressor={a.compressor}")
ends = root["meta/episode_ends"][:]
print("episode_ends:", ends.tolist())
starts = np.concatenate([[0], ends[:-1]])
print("episode lengths:", (ends - starts).tolist())
ts = root["data/timestamp"][:]
for s, e in zip(starts, ends):
    d = np.diff(ts[s:e])
    print(f"  ep [{s},{e}) t0={ts[s]:.4f} t_end={ts[e-1]:.4f} dur={ts[e-1]-ts[s]:.3f}s dt mean={d.mean():.4f} median={np.median(d):.4f} min={d.min():.4f} max={d.max():.4f} -> fps~{1/np.median(d):.2f}")
np.set_printoptions(precision=4, suppress=True, linewidth=200)
small = [k for k in root["data"].array_keys() if "img" not in k]
for k in sorted(small):
    a = root["data"][k][:]
    a2 = a.reshape(a.shape[0], -1)
    print(f"\n{k} {a.shape}: min={a2.min(0)}\n   max={a2.max(0)}\n   mean={a2.mean(0)}")
a = root["data/action"][:]
t = root["data/target"][:]
lp = root["data/left_robot_tcp_pose"][:]
rp = root["data/right_robot_tcp_pose"][:]
lw = root["data/left_robot_gripper_width"][:]
rw = root["data/right_robot_gripper_width"][:]
print("\nfirst 3 rows action:\n", a[:3])
print("first 3 rows target:\n", t[:3])
print("first 3 rows left tcp pose:\n", lp[:3])
print("first 3 rows right tcp pose:\n", rp[:3])
print("first 3 rows left/right gripper width:", lw[:3].ravel(), rw[:3].ravel())
print("action == target everywhere?", np.allclose(a, t))
# compare action xyz against tcp xyz (both arms), next-step
for name, p in [("left", lp), ("right", rp)]:
    for lag in [0, 1, 2, 4]:
        n = len(a) - lag
        err = np.abs(a[:n, :3] - p[lag:lag + n, :3]).mean()
        print(f"  mean |action[:3]-{name}_tcp[:3]| lag {lag}: {err:.5f}")
print("corr action[:,3] vs left width:", np.corrcoef(a[:, 3], lw.ravel())[0, 1], " vs right width:", np.corrcoef(a[:, 3], rw.ravel())[0, 1])
