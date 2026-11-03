#!/bin/bash
cd "$(dirname "$0")/../.."
MUJOCO_GL=glfw .venv/bin/python scripts/collect_demos.py --n 63 --start-seed 3003 --workers 5 --class-set v2 --out data/raw_v2 --skip-done
echo COLLECT_V2_DONE
