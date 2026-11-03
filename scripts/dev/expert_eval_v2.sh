#!/bin/bash
# Scripted-expert baselines on the v2 held-out set (seeds 10000+), low priority so training keeps the CPU.
cd "$(dirname "$0")/../.."
for m in expert_force expert_noforce expert_force_nominal expert_noforce_nominal; do
  MUJOCO_GL=glfw nice -n 19 .venv/bin/python scripts/eval_policies.py --method $m --class-set v2 --per-class 5 --workers 2
done
echo EXPERT_EVAL_V2_DONE
