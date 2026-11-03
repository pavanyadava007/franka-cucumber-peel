#!/bin/bash
# After the v2 (real cucumber types) demos are collected: build the combined dataset, fine-tune RDP and train the
# vision-only DP baseline on exactly the same data. LDP fine-tune and DP run side by side on the 4 GB GPU.
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P"
.venv/bin/python scripts/make_rdp_dataset.py --kind stroke --raw data/raw data/raw_v2 --out data/rdp/peel_stroke_v2_zarr
export DATA="$P/data/rdp/peel_stroke_v2_zarr" NW=3
EPOCHS=24 bash scripts/train_rdp.sh ldp_ft &
sleep 600  # stagger the two normalizer passes
DP_RUN="$P/runs/dp_v2" bash scripts/train_rdp.sh dp &
wait
echo PIPELINE_V2_DONE
