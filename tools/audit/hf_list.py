"""List files (with sizes) of a Hugging Face dataset repo. Usage: python -I hf_list.py <repo_id> [path_prefix]"""
import sys
from huggingface_hub import HfApi

repo = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else ""
api = HfApi()
try:
    info = api.dataset_info(repo, files_metadata=False)
    print("gated:", getattr(info, "gated", None), "private:", info.private)
    print("tags:", [t for t in (info.tags or []) if "license" in t or "task" in t][:10])
    cd = info.card_data
    if cd is not None:
        print("card license:", cd.get("license"))
except Exception as e:
    print("dataset_info error:", repr(e)[:300])
try:
    files = list(api.list_repo_tree(repo, repo_type="dataset", path_in_repo=prefix or None, recursive=True))
except Exception as e:
    print("list error:", repr(e)[:300]); sys.exit(0)
tot = 0
nfile = 0
for f in files:
    size = getattr(f, "size", None)
    if size is not None:
        tot += size; nfile += 1
print(f"files: {nfile}, total bytes: {tot} ({tot/1e9:.3f} GB)")
for f in files:
    size = getattr(f, "size", None)
    if size is None:
        continue
    print(f"{size:>14}  {f.path}")
