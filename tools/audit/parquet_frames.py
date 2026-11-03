"""Inspect a LeRobot data parquet with embedded images, dump frames and low-dim stats.
Usage: python -I parquet_frames.py <file.parquet> <out_dir> <image_col1,image_col2,...> [n_frames]
Reads row groups lazily; images expected as struct{bytes, path} (PNG/JPEG)."""
import sys, os, io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

path, out = sys.argv[1], sys.argv[2]
img_cols = [c for c in sys.argv[3].split(",") if c]
nf = int(sys.argv[4]) if len(sys.argv) > 4 else 5
os.makedirs(out, exist_ok=True)
pf = pq.ParquetFile(path)
print("schema:")
print(pf.schema_arrow)
md = pf.metadata
print("rows:", md.num_rows, "row groups:", md.num_row_groups)
low = [n for n in pf.schema_arrow.names if n not in img_cols]
tab = pf.read(columns=low).to_pandas()
np.set_printoptions(precision=4, suppress=True, linewidth=220)
for c in low:
    v = tab[c].iloc[0]
    if isinstance(v, (np.ndarray, list)):
        arr = np.stack(tab[c].to_numpy())
        print(f"{c}: shape {arr.shape}\n  min {arr.min(0)}\n  max {arr.max(0)}\n  mean {arr.mean(0)}\n  row0 {arr[0]}")
    else:
        print(f"{c}: first {tab[c].iloc[0]} last {tab[c].iloc[-1]} unique {tab[c].nunique()}")
n = md.num_rows
idxs = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, nf)))
# map global row -> row group
rg_starts = np.cumsum([0] + [md.row_group(i).num_rows for i in range(md.num_row_groups)])
for c in img_cols:
    for i in idxs:
        g = int(np.searchsorted(rg_starts, i, side="right") - 1)
        t = pf.read_row_group(g, columns=[c])
        cell = t.column(c)[i - rg_starts[g]].as_py()
        b = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(b))
        fmt = im.format
        im = im.convert("RGB")
        fn = os.path.join(out, f"{c.replace('.', '_')}_{i:05d}.png")
        im.save(fn)
        print("saved", fn, fmt, im.size)
