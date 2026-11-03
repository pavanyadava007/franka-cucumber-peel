"""Save chosen frames from an image parquet. Usage: python -I parquet_pick.py <file.parquet> <out_dir> <prefix> col:idx [col:idx ...]"""
import sys, io, os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

path, out, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
pf = pq.ParquetFile(path)
md = pf.metadata
st = np.cumsum([0] + [md.row_group(i).num_rows for i in range(md.num_row_groups)])
for spec in sys.argv[4:]:
    c, i = spec.rsplit(":", 1)
    i = int(i)
    g = int(np.searchsorted(st, i, side="right") - 1)
    cell = pf.read_row_group(g, columns=[c]).column(0)[i - st[g]].as_py()
    b = cell["bytes"] if isinstance(cell, dict) else cell
    fn = os.path.join(out, f"{prefix}_{c.replace('.', '_')}_{i:05d}.png")
    Image.open(io.BytesIO(b)).convert("RGB").save(fn)
    print(fn)
