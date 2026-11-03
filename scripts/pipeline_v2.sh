#!/bin/bash
# v2 (real cucumber types) pipeline after the demos are collected. Safe to re-run after a reboot: every step skips
# or resumes what is already done.
#   1. combined stroke dataset (v1 + v2 demos)
#   2. RDP stage-2 fine-tune and the vision-only DP baseline, side by side on the 4 GB GPU (both resume per epoch)
#   3. evaluation of both policies on the v2 held-out set (seeds 10000+), 5 per class, one video per class
#   4. docs/RESULTS.md
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P"
Z="$P/data/rdp/peel_stroke_v2_zarr"
if [ ! -f "$Z/summary.json" ]; then
  rm -rf "$Z"  # a build cut off by a reboot leaves a partial zarr
  .venv/bin/python scripts/make_rdp_dataset.py --kind stroke --raw data/raw data/raw_v2 --out "$Z"
fi
export DATA="$Z" NW=3 MUJOCO_GL=glfw

eval_policy() {  # method ckpt: two processes split the 7 classes
  .venv/bin/python scripts/eval_policies.py --method "$1" --ckpt "$2" --class-set v2 --per-class 5 --video-first \
    --classes pickling,persian,slicing,english > "results/eval_${1}_v2_a.log" 2>&1 &
  .venv/bin/python scripts/eval_policies.py --method "$1" --ckpt "$2" --class-set v2 --per-class 5 --video-first \
    --classes crooked,necked,realistic > "results/eval_${1}_v2_b.log" 2>&1 &
  wait
}

LDP_CKPT="$P/runs/rdp_ldp_v2/checkpoints/latest.ckpt"
DP_CKPT="$P/runs/dp_v2/checkpoints/latest.ckpt"
(
  if [ ! -f runs/rdp_ldp_v2.done ]; then
    EPOCHS=24 bash scripts/train_rdp.sh ldp_ft || { echo "RDP fine-tune FAILED, not evaluating"; exit 1; }
    touch runs/rdp_ldp_v2.done
  fi
  eval_policy rdp "$LDP_CKPT"
) &
(
  [ -f runs/dp_v2/checkpoints/latest.ckpt ] || [ -f runs/rdp_ldp_v2.done ] || sleep 600  # stagger the normalizer passes
  if [ ! -f runs/dp_v2.done ]; then
    DP_RUN="$P/runs/dp_v2" bash scripts/train_rdp.sh dp || { echo "DP training FAILED, not evaluating"; exit 1; }
    touch runs/dp_v2.done
  fi
  eval_policy dp "$DP_CKPT"
) &
wait  # make_results still runs if one branch failed, so the other one's numbers are kept
.venv/bin/python scripts/make_results.py > /dev/null
echo PIPELINE_V2_DONE
