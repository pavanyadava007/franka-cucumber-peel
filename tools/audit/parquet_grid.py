"""Grid of chosen rows/cols from an image parquet. Usage: python -I parquet_grid.py <file.parquet> <out.png> <col1,col2> <i1,i2,...> [scale]"""
import sys, io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

path, out = sys.argv[1], sys.argv[2]
cols = sys.argv[3].split(",")
idxs = [int(x) for x in sys.argv[4].split(",")]
scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
pf = pq.ParquetFile(path)
md = pf.metadata
st = np.cumsum([0] + [md.row_group(i).num_rows for i in range(md.num_row_groups)])
rows = []
for c in cols:
    ims = []
    cache = {}
    for i in idxs:
        g = int(np.searchsorted(st, i, side="right") - 1)
        if g not in cache:
            cache = {g: pf.read_row_group(g, columns=[c])}
        cell = cache[g].column(0)[i - st[g]].as_py()
        b = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(b)).convert("RGB")
        ims.append(np.asarray(im))
    rows.append(np.concatenate(ims, 1))
g = np.concatenate(rows, 0)
im = Image.fromarray(g)
if scale != 1.0:
    im = im.resize((int(im.width * scale), int(im.height * scale)))
im.save(out)
print("saved", out, im.size)
