"""Download selected files from a HF dataset repo into a local dir.
Usage: python -I hf_get.py <repo_id> <local_dir> <file1> [file2 ...]
       python -I hf_get.py <repo_id> <local_dir> --meta [prefix]   (README + meta/ files under prefix, max 30 MB each)
"""
import sys
from huggingface_hub import HfApi, hf_hub_download

repo, out = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
api = HfApi()
if args and args[0] == "--meta":
    prefix = args[1] if len(args) > 1 else ""
    files = []
    for f in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=prefix or None, recursive=True):
        size = getattr(f, "size", None)
        if size is None:
            continue
        p = f.path
        rel = p[len(prefix):].lstrip("/") if prefix else p
        if (rel.startswith("meta/") or rel == "README.md") and size < 30_000_000:
            files.append(p)
    if not prefix:
        files.append("README.md") if "README.md" not in files else None
else:
    files = args
for p in files:
    try:
        path = hf_hub_download(repo, p, repo_type="dataset", local_dir=out)
        print("ok", p)
    except Exception as e:
        print("FAIL", p, repr(e)[:200])
