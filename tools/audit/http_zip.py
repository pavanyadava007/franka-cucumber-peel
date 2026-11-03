"""Read a remote ZIP via HTTP range requests (no full download).
Usage:
  python -I http_zip.py <url> list [max_lines]
  python -I http_zip.py <url> get <out_dir> <member1> [member2 ...]
Members are written under out_dir with path separators flattened. Refuses members > 20 MB and total transfer > 80 MB."""
import io, os, sys, zipfile, urllib.request

MAX_TOTAL = 80 * 1024 * 1024


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n - 1
        self.fetched += n
        if self.fetched > MAX_TOTAL:
            raise RuntimeError("transfer cap reached")
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


url, cmd = sys.argv[1], sys.argv[2]
rf = RangeFile(url)
zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
infos = zf.infolist()
if cmd == "list":
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    print(f"zip size {rf.size} bytes, members {len(infos)}")
    for i in infos[:mx]:
        print(f"{i.file_size:>12} {i.compress_size:>12} {i.filename}")
else:
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    names = {i.filename: i for i in infos}
    for m in sys.argv[4:]:
        i = names[m]
        if i.file_size > 20 * 1024 * 1024:
            print("skip (too big)", m, i.file_size)
            continue
        data = zf.read(m)
        fn = os.path.join(out, m.replace("/", "__"))
        open(fn, "wb").write(data)
        print("got", m, len(data))
print("bytes fetched:", rf.fetched)
