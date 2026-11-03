"""Inspect a remote HDF5 via HTTP range requests (no full download), with a transfer cap.
Usage: python -I http_h5.py <url> tree
       python -I http_h5.py <url> frames <out_dir> <dataset_path> <i1,i2,...>
       python -I http_h5.py <url> stats <dataset_path> [max_rows]"""
import io, os, sys, urllib.request
import numpy as np
import h5py

MAX_TOTAL = 60 * 1024 * 1024


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        self.fetched += n
        if self.fetched > MAX_TOTAL:
            raise RuntimeError("transfer cap reached")
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[: len(d)] = d; return len(d)


url, cmd = sys.argv[1], sys.argv[2]
rf = RangeFile(url)
f = h5py.File(io.BufferedReader(rf, buffer_size=1 << 14), "r")
print("file size", rf.size)
if cmd == "keys":
    def show(g, depth, maxd, maxk):
        ks = list(g.keys())
        print("  " * depth + f"[{g.name}] {len(ks)} keys: {ks[:maxk]}")
        for k in ks[:1] if depth < maxd else []:
            o = g[k]
            if isinstance(o, h5py.Group):
                show(o, depth + 1, maxd, maxk)
        if depth == maxd or True:
            for k in ks[:maxk]:
                o = g.get(k, getclass=True)
                if o is h5py.Dataset:
                    d = g[k]
                    print("  " * (depth + 1) + f"D {k} shape={d.shape} dtype={d.dtype} chunks={d.chunks} compression={d.compression}")
    show(f, 0, int(sys.argv[3]) if len(sys.argv) > 3 else 3, 40)
elif cmd == "pc":
    # pc <out.npz> <demo_group> <i1,i2,...> : save point cloud frames + small arrays of one demo
    out, grp, idxs = sys.argv[3], sys.argv[4], [int(x) for x in sys.argv[5].split(",")]
    g = f[grp]
    res = {"idxs": np.array(idxs)}
    for i in idxs:
        res[f"xyz_{i}"] = g["l515_pc_xyzs_l515"][i]
        res[f"rgb_{i}"] = g["l515_pc_rgbs"][i]
    for k in ["pyft_fs", "pyft_ts", "angler_widths", "gripper_xyzs_l515", "pyft_xyzs_l515", "a", "o"]:
        res[k] = g[k][()]
    res["n_demos"] = np.array(len(f["data"].keys()))
    np.savez(out, **res)
    print("saved", out, "demos in file:", int(res["n_demos"]))
elif cmd == "tree":
    lines = []
    def visit(name, obj):
        if len(lines) > 400:
            return
        if isinstance(obj, h5py.Dataset):
            lines.append(f"D {name} shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks} compression={obj.compression}")
        else:
            lines.append(f"G {name} attrs={dict(list(obj.attrs.items())[:8])}")
    print("root attrs:", {k: (v if not isinstance(v, bytes) else v[:200]) for k, v in list(f.attrs.items())[:10]})
    f.visititems(visit)
    print("\n".join(lines))
elif cmd == "frames":
    from PIL import Image
    out, path, idxs = sys.argv[3], sys.argv[4], [int(x) for x in sys.argv[5].split(",")]
    os.makedirs(out, exist_ok=True)
    d = f[path]
    for i in idxs:
        a = d[i]
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        fn = os.path.join(out, path.replace("/", "__") + f"_{i:05d}.png")
        Image.fromarray(a).save(fn); print("saved", fn, a.shape)
elif cmd == "stats":
    path = sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    a = f[path][:n]
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(path, a.shape, "min", a.min(0), "max", a.max(0))
print("bytes fetched:", rf.fetched)
