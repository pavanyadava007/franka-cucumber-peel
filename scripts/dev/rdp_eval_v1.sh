#!/bin/bash
cd "$(dirname "$0")/../.."
MUJOCO_GL=glfw .venv/bin/python scripts/eval_policies.py --method rdp --ckpt runs/rdp_ldp/checkpoints/latest.ckpt --per-class 3 --video-first
echo RDP_EVAL_DONE
