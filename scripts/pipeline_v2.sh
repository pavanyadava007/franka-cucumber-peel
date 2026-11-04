#!/bin/bash
# v2 (real cucumber types) pipeline after the demos are collected. Safe to re-run after a reboot: every step skips
# or resumes what is already done.
#   1. combined stroke dataset (v1 + v2 demos)
#   2. RDP stage 1 (tokenizer), the vision-only DP baseline and RDP stage 2, all trained from scratch on v1 + v2 with
#      the same budget as v1, one at a time on the 4 GB GPU (the small tokenizer may run next to DP)
#   3. DP is evaluated while RDP stage 2 trains; then RDP is evaluated (v2 held-out seeds 10000+, 5 per class, one
#      video each)
#   4. docs/RESULTS.md
# A first attempt fine-tuned the v1 RDP stage 2 on v1 + v2 instead. It never touched the cucumber (0/13): the v1
# tokenizer was trained with the v1 action normalizer, and the fine-tune replaced that normalizer with the much wider
# v1 + v2 one. Kept for the record in data/big/runs/rdp_ldp_v2_finetune_failed (docs/RESULTS.md).
# Large outputs (dataset ~5 GB, checkpoints ~6 GB) go to $BIG, linked into data/ and runs/. Under WSL the Linux disk
# grows on the Windows drive that holds it; if that drive is short of space, point BIG at another drive, e.g.
# BIG=/mnt/d/franka-cucumber-peel-big (slower: the dataset is ~200k small files). check_space stops before a crash.
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P"
BIG="${BIG:-$P/data/big}"
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
for r in rdp_at_v2 rdp_ldp_v2 dp_v2; do mkdir -p "$BIG/runs/$r"; link runs/$r runs/$r; link runs/$r.log runs/$r.log; done

eval_policy() {  # method ckpt classes tag
  .venv/bin/python scripts/eval_policies.py --method "$1" --ckpt "$2" --class-set v2 --per-class 5 --video-first \
    --classes "$3" >> "results/eval_${1}_v2_$4.log" 2>&1
}
ALL=pickling,persian,slicing,english,crooked,necked,realistic

while pgrep -f "config-name=train_at_workspace" > /dev/null; do sleep 60; done  # tokenizer started by hand
check_space "the RDP tokenizer"
if [ ! -f runs/rdp_at_v2.done ]; then
  AT_RUN="$BIG/runs/rdp_at_v2" bash scripts/train_rdp.sh at || { echo "RDP tokenizer FAILED"; exit 1; }
  sync; touch runs/rdp_at_v2.done
fi
check_space "the DP training"
if [ ! -f runs/dp_v2.done ]; then
  DP_RUN="$BIG/runs/dp_v2" bash scripts/train_rdp.sh dp || { echo "DP training FAILED"; exit 1; }
  sync; touch runs/dp_v2.done
fi
eval_policy dp "$P/runs/dp_v2/checkpoints/latest.ckpt" "$ALL" all &
DP_EVAL=$!

check_space "the RDP stage 2 training"
if [ ! -f runs/rdp_ldp_v2.done ]; then
  AT_RUN="$BIG/runs/rdp_at_v2" LDP_RUN="$BIG/runs/rdp_ldp_v2" bash scripts/train_rdp.sh ldp \
    || { echo "RDP stage 2 FAILED"; wait $DP_EVAL; exit 1; }
  sync; touch runs/rdp_ldp_v2.done
fi
RDP_CKPT="$P/runs/rdp_ldp_v2/checkpoints/latest.ckpt"
eval_policy rdp "$RDP_CKPT" pickling,persian,slicing,english a &
eval_policy rdp "$RDP_CKPT" crooked,necked,realistic b &
wait
.venv/bin/python scripts/make_results.py > /dev/null
echo PIPELINE_V2_DONE
