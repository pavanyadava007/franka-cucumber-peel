"""Dump frames from RDP zarr image keys. Usage: python -I rdp_frames.py <replay_buffer.zarr> <out_dir>"""
import sys, os
import numpy as np, zarr
from PIL import Image

r = zarr.open(sys.argv[1], "r")
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
ends = r["meta/episode_ends"][:]
starts = np.concatenate([[0], ends[:-1]])
keys = [k for k in r["data"].array_keys() if k.endswith("_img")]
for ei, (s, e) in enumerate(zip(starts, ends)):
    idxs = [s, s + (e - s) // 4, s + (e - s) // 2, s + 3 * (e - s) // 4, e - 1]
    for k in keys:
        frames = [r["data"][k][i] for i in idxs]
        row = np.concatenate(frames, axis=1)
        Image.fromarray(row).save(os.path.join(out, f"ep{ei}_{k}.png"))
        print("saved", f"ep{ei}_{k}.png", "frames", [int(i) for i in idxs])
