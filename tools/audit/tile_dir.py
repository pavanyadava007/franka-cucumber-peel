"""Tile frames <key>_<idx>.png from a dir into a grid (rows = keys, cols = idx). Usage: python -I tile_dir.py <dir> <out.png> [tile_w]"""
import sys, os, glob, re
from collections import defaultdict
from PIL import Image

d, out = sys.argv[1], sys.argv[2]
tw = int(sys.argv[3]) if len(sys.argv) > 3 else 320
groups = defaultdict(dict)
for f in glob.glob(os.path.join(d, "*.png")):
    m = re.match(r"(.+)_(\d+)\.png$", os.path.basename(f))
    if m and not os.path.basename(f).startswith(("strip", "grid", "wrist_")):
        groups[m.group(1)][int(m.group(2))] = f
keys = sorted(groups)
cols = sorted({i for k in keys for i in groups[k]})
th = None
tiles = {}
for k in keys:
    for i, f in groups[k].items():
        im = Image.open(f).convert("RGB")
        h = int(im.height * tw / im.width)
        th = max(th or 0, h)
        tiles[(k, i)] = im.resize((tw, h))
W = Image.new("RGB", (tw * len(cols), th * len(keys)), (255, 255, 255))
for r, k in enumerate(keys):
    for c, i in enumerate(cols):
        if (k, i) in tiles:
            W.paste(tiles[(k, i)], (c * tw, r * th))
W.save(out)
print("rows:", keys, "cols:", cols)
