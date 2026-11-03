"""Build RDP replay_buffer.zarr datasets from raw demo episodes.

  python scripts/make_rdp_dataset.py --kind stroke   -> data/rdp/peel_stroke_zarr  (stroke skill, 10-D action)
  python scripts/make_rdp_dataset.py --kind e2e      -> data/rdp/peel_e2e_zarr     (whole peel, both arms, 20-D)
Only successful demos are used (coverage >= 90 %, gouge <= 3 %, no handle collision) unless --all is given.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peel.recorder import make_rdp_zarr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=[str(ROOT / "data" / "raw")], help="one or more raw episode folders")
    ap.add_argument("--kind", choices=["stroke", "e2e"], default="stroke")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    a = ap.parse_args()
    # only episodes the collector has finished writing (listed in its log), so a running collection is safe
    files = []
    for raw in a.raw:
        log = Path(raw) / "collect_log.jsonl"
        done = {json.loads(x)["seed"] for x in log.read_text().splitlines() if x.strip()} if log.exists() else None
        files += [f for f in sorted(Path(raw).glob("ep_*.npz")) if done is None or int(f.stem.split("_")[1]) in done]
    files = files[: a.max_episodes]
    out = Path(a.out or ROOT / "data" / "rdp" / f"peel_{a.kind}_zarr")
    summary = make_rdp_zarr(files, out, kind=a.kind, success_only=not a.all)
    summary["raw_files"] = len(files)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
