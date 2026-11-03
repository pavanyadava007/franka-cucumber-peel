"""Print how many epochs the latest.ckpt in a checkpoints folder has completed (0 if there is none).

RDP's trainer re-runs `training.num_epochs` epochs on every resume, so scripts/train_rdp.sh uses this to train only
the epochs that are left. A reboot while a checkpoint is being written leaves a cut-off latest.ckpt; in that case
it is replaced by the newest readable top-k checkpoint of the same folder, or removed.
  python scripts/ckpt_epochs.py runs/dp_v2/checkpoints
"""

import shutil
import sys
from pathlib import Path

import dill
import torch


def epochs_done(path: Path) -> int:
    try:  # mmap reads only the header; fall back to a full read if the filesystem cannot map files
        payload = torch.load(path, pickle_module=dill, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        payload = torch.load(path, pickle_module=dill, map_location="cpu", weights_only=False)
    return dill.loads(payload["pickles"]["epoch"]) + 1  # saved before the epoch counter is increased


def main():
    d = Path(sys.argv[1])
    latest = d / "latest.ckpt"
    if not latest.exists():
        print(0)
        return
    try:
        print(epochs_done(latest))
        return
    except Exception as e:  # cut-off file
        print(f"latest.ckpt unreadable ({type(e).__name__}), looking for a top-k checkpoint", file=sys.stderr)
    latest.rename(d / "latest.ckpt.broken")
    for cand in sorted(d.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            n = epochs_done(cand)
        except Exception:
            continue
        shutil.copyfile(cand, latest)
        print(f"restored latest.ckpt from {cand.name}", file=sys.stderr)
        print(n)
        return
    print(0)


if __name__ == "__main__":
    main()
