#!/bin/bash
# v2 (real cucumber types) pipeline after the demos are collected. Safe to re-run after a reboot: every step skips
# or resumes what is already done.
#   1. combined stroke dataset (v1 + v2 demos)
#   2. RDP stage-2 fine-tune, then the vision-only DP baseline (one at a time: two copies of the ~5 GB dataset in
#      RAM would push the 12 GB VM into swap)
#   3. RDP is evaluated while DP trains; then DP is evaluated (v2 held-out seeds 10000+, 5 per class, one video each)
#   4. docs/RESULTS.md
# Large outputs (dataset, checkpoints) go to $BIG, linked into data/ and runs/. On this PC the WSL disk lives on a
# nearly full C: drive, and growing it by several GB crashed the VM twice, so $BIG is on D:.
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P"
BIG="${BIG:-/mnt/d/franka-cucumber-peel-big}"
mkdir -p "$BIG/runs"

check_space() {  # stop cleanly instead of crashing the VM when the Windows drive under the WSL disk is full
  if [ -d /mnt/c ] && [ "$(df --output=avail -BG /mnt/c | tail -1 | tr -dc 0-9)" -lt 2 ]; then
    echo "C: has less than 2 GB free, stopping before $1"; exit 1
  fi
}
link() {  # make $P/$1 a link to $BIG/$2 (left alone if something is already there)
  if [ ! -e "$P/$1" ] && [ ! -L "$P/$1" ]; then ln -s "$BIG/$2" "$P/$1"; fi
}

Z="$BIG/peel_stroke_v2_zarr"
link data/rdp/peel_stroke_v2_zarr peel_stroke_v2_zarr
if ! .venv/bin/python -c "import json,sys; json.load(open(sys.argv[1]))" "$Z/summary.json" 2>/dev/null; then
  check_space "the dataset build"
  rm -rf "$Z"  # a build cut off by a reboot leaves a partial zarr
  .venv/bin/python scripts/make_rdp_dataset.py --kind stroke --raw data/raw data/raw_v2 --out "$Z"
  sync
fi
export DATA="$Z" NW=3 MUJOCO_GL=glfw
for r in rdp_ldp_v2 dp_v2; do mkdir -p "$BIG/runs/$r"; link runs/$r runs/$r; link runs/$r.log runs/$r.log; done

eval_policy() {  # method ckpt classes tag
  .venv/bin/python scripts/eval_policies.py --method "$1" --ckpt "$2" --class-set v2 --per-class 5 --video-first \
    --classes "$3" >> "results/eval_${1}_v2_$4.log" 2>&1
}
ALL=pickling,persian,slicing,english,crooked,necked,realistic

check_space "the RDP fine-tune"
if [ ! -f runs/rdp_ldp_v2.done ]; then
  EPOCHS=24 DST="$BIG/runs/rdp_ldp_v2" bash scripts/train_rdp.sh ldp_ft || { echo "RDP fine-tune FAILED"; exit 1; }
  sync; touch runs/rdp_ldp_v2.done
fi
eval_policy rdp "$P/runs/rdp_ldp_v2/checkpoints/latest.ckpt" "$ALL" all &
RDP_EVAL=$!

check_space "the DP training"
if [ ! -f runs/dp_v2.done ]; then
  DP_RUN="$BIG/runs/dp_v2" bash scripts/train_rdp.sh dp || { echo "DP training FAILED"; wait $RDP_EVAL; exit 1; }
  sync; touch runs/dp_v2.done
fi
wait $RDP_EVAL || echo "RDP evaluation FAILED"
DP_CKPT="$P/runs/dp_v2/checkpoints/latest.ckpt"
eval_policy dp "$DP_CKPT" pickling,persian,slicing,english a &
eval_policy dp "$DP_CKPT" crooked,necked,realistic b &
wait
.venv/bin/python scripts/make_results.py > /dev/null
echo PIPELINE_V2_DONE
