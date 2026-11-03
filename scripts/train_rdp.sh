#!/bin/bash
# Train RDP (stage 1 Asymmetric Tokenizer, stage 2 Latent Diffusion Policy) and the vision-only Diffusion Policy
# baseline on the simulated stroke dataset, with the official RDP code in third_party/rdp (unmodified).
# Sized for a 4 GB GTX 1650: UNet widths 256/512/1024, 96x128 images, batch 32 (LDP/DP), fixed step budgets.
#   bash scripts/train_rdp.sh at|ldp|ldp_ft|dp|all
# ldp_ft and dp resume after a reboot: RDP's trainer runs training.num_epochs MORE epochs on every resume, so they
# pass only the epochs still missing (scripts/ckpt_epochs.py reads the checkpoint, repairs a cut-off one).
set -eo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$P/data/rdp/peel_stroke_zarr}"
PY="$P/.venv/bin/python"
export PYTHONPATH="$P/rdp_shims" WANDB_MODE=offline WANDB_DIR="$P/runs" HYDRA_FULL_ERROR=1
cd "$P/third_party/rdp"
COMMON=("hydra.searchpath=[file://$P/rdp_configs]" "task.dataset_path=$DATA" logging.mode=offline
        dataloader.num_workers=${NW:-5} val_dataloader.num_workers=2 training.val_every=5)
stage="${1:-all}"
epochs_left() {  # checkpoints_dir total -> epochs still to train
  local done; done=$($PY "$P/scripts/ckpt_epochs.py" "$1"); echo $(( $2 - done ))
}

if [ "$stage" = at ] || [ "$stage" = all ]; then
  $PY train.py --config-name=train_at_workspace task=sim_stroke_wrench_at at=at_peel "${COMMON[@]}" \
    training.num_epochs=40 training.max_train_steps=200 training.checkpoint_every=10 \
    hydra.run.dir="$P/runs/rdp_at" 2>&1 | tr '\r' '\n' | grep --line-buffered -v "Calculating\|it/s\]$" > "$P/runs/rdp_at.log"
fi
if [ "$stage" = ldp ] || [ "$stage" = all ]; then
  $PY train.py --config-name=train_latent_diffusion_unet_real_image_workspace task=sim_stroke_wrench_ldp at=at_peel \
    at_load_dir="$P/runs/rdp_at/checkpoints/latest.ckpt" "${COMMON[@]}" \
    "policy.down_dims=[256,512,1024]" "policy.obs_encoder.resize_shape=[96,128]" \
    dataloader.batch_size=32 val_dataloader.batch_size=32 training.sample_every=1000 \
    training.num_epochs=20 training.max_train_steps=1000 training.checkpoint_every=5 \
    hydra.run.dir="$P/runs/rdp_ldp" 2>&1 | tr '\r' '\n' | grep --line-buffered -v "Calculating\|it/s\]$" > "$P/runs/rdp_ldp.log"
fi
if [ "$stage" = ldp_ft ]; then
  # fine-tune RDP stage 2 from an existing run on new data: copy the run, then resume it with more epochs
  # EPOCHS is the total including the source run (v1 run: 16 epochs, so EPOCHS=24 adds 8 on the new data)
  SRC="${SRC:-$P/runs/rdp_ldp}"; DST="${DST:-$P/runs/rdp_ldp_v2}"; EPOCHS="${EPOCHS:-24}"
  mkdir -p "$DST/checkpoints"; $PY "$P/scripts/ckpt_epochs.py" "$DST/checkpoints" > /dev/null
  [ -f "$DST/checkpoints/latest.ckpt" ] || cp "$SRC/checkpoints/latest.ckpt" "$DST/checkpoints/"
  LEFT=$(epochs_left "$DST/checkpoints" "$EPOCHS"); echo "ldp_ft: $LEFT epochs left of $EPOCHS"
  [ "$LEFT" -gt 0 ] || exit 0
  $PY train.py --config-name=train_latent_diffusion_unet_real_image_workspace task=sim_stroke_wrench_ldp at=at_peel \
    at_load_dir="$P/runs/rdp_at/checkpoints/latest.ckpt" "${COMMON[@]}" \
    "policy.down_dims=[256,512,1024]" "policy.obs_encoder.resize_shape=[96,128]" \
    dataloader.batch_size=32 val_dataloader.batch_size=32 training.sample_every=1000 training.resume=True \
    training.num_epochs=$LEFT training.max_train_steps=1000 training.checkpoint_every=1 \
    hydra.run.dir="$DST" 2>&1 | tr '\r' '\n' | grep --line-buffered -v "Calculating\|it/s\]$" >> "$DST.log"
fi
if [ "$stage" = dp ] || [ "$stage" = all ]; then
  DPR="${DP_RUN:-$P/runs/dp}"; mkdir -p "$DPR/checkpoints"
  LEFT=$(epochs_left "$DPR/checkpoints" 20); echo "dp: $LEFT epochs left of 20"
  [ "$LEFT" -gt 0 ] || exit 0
  $PY train.py --config-name=train_diffusion_unet_real_image_workspace task=sim_stroke_image_dp "${COMMON[@]}" \
    horizon=32 "policy.down_dims=[256,512,1024]" "policy.obs_encoder.resize_shape=[96,128]" \
    dataloader.batch_size=32 val_dataloader.batch_size=32 training.sample_every=1000 \
    training.num_epochs=$LEFT training.max_train_steps=1000 training.checkpoint_every=1 \
    hydra.run.dir="$DPR" 2>&1 | tr '\r' '\n' | grep --line-buffered -v "Calculating\|it/s\]$" >> "$DPR.log"
fi
