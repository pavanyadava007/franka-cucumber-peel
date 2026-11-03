"""Render saved point-cloud frames (camera frame xyz + rgb) to PNGs with a simple pinhole z-buffer.
Usage: python -I render_pc.py <frames.npz> <out_prefix>"""
import sys
import numpy as np
from PIL import Image

d = np.load(sys.argv[1], allow_pickle=False)
pre = sys.argv[2]
W, H = 480, 360
for i in d["idxs"]:
    xyz = d[f"xyz_{i}"].astype(np.float64)
    rgb = d[f"rgb_{i}"].astype(np.float64)
    if rgb.max() <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    m = np.isfinite(xyz).all(1) & (xyz[:, 2] > 1e-3)
    xyz, rgb = xyz[m], rgb[m]
    z = xyz[:, 2]
    u0, v0 = xyz[:, 0] / z, xyz[:, 1] / z
    # fit the projected cloud into the image
    lo, hi = np.percentile(u0, 1), np.percentile(u0, 99)
    lo2, hi2 = np.percentile(v0, 1), np.percentile(v0, 99)
    s = min((W - 20) / max(hi - lo, 1e-6), (H - 20) / max(hi2 - lo2, 1e-6))
    u = ((u0 - lo) * s + 10).astype(int)
    v = ((v0 - lo2) * s + 10).astype(int)
    img = np.full((H, W, 3), 255, np.uint8)
    zbuf = np.full((H, W), np.inf)
    order = np.argsort(-z)
    for k in order:
        for du in (0, 1, 2):
            for dv in (0, 1, 2):
                uu, vv = u[k] + du, v[k] + dv
                if 0 <= uu < W and 0 <= vv < H and z[k] < zbuf[vv, uu]:
                    zbuf[vv, uu] = z[k]
                    img[vv, uu] = np.clip(rgb[k], 0, 255)
    Image.fromarray(img).save(f"{pre}_{int(i):04d}.png")
    print("frame", int(i), "points", len(z), "z range", round(float(z.min()), 3), round(float(z.max()), 3))
fs = d["pyft_fs"]
print("pyft_fs", fs.shape, "|F| percentiles 50/90/99/max:", np.round(np.percentile(np.linalg.norm(fs, axis=1), [50, 90, 99, 100]), 2))
print("angler_widths range", float(d["angler_widths"].min()), float(d["angler_widths"].max()))
print("a", d["a"].shape, d["a"].dtype, "first row", d["a"][0]); print("o", d["o"].shape, "first", d["o"][:5].ravel())
print("n_demos", int(d["n_demos"]))
