#!/bin/bash
# After a reboot: restart the v2 pipeline in the background (it skips finished work) and keep Windows awake while it
# runs. The expert baselines are re-run only for seeds that are missing.
cd "$(dirname "$0")/../.."
setsid nohup bash scripts/pipeline_v2.sh >> runs/pipeline_v2.log 2>&1 < /dev/null &
PIPE=$!
setsid nohup bash scripts/dev/keep_awake.sh "$PIPE" > runs/keep_awake.log 2>&1 < /dev/null &
setsid nohup bash scripts/dev/expert_eval_v2.sh >> results/expert_eval_v2.log 2>&1 < /dev/null &
echo "started pipeline (pid $PIPE); logs: runs/pipeline_v2.log runs/rdp_ldp_v2.log runs/dp_v2.log"
