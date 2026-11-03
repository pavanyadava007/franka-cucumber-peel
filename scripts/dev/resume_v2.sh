#!/bin/bash
# After a reboot: restart the v2 pipeline and the expert baselines in the background. Both skip finished work.
cd "$(dirname "$0")/../.."
setsid nohup bash scripts/pipeline_v2.sh >> runs/pipeline_v2.log 2>&1 < /dev/null &
setsid nohup bash scripts/dev/expert_eval_v2.sh >> results/expert_eval_v2.log 2>&1 < /dev/null &
echo "started; logs: runs/pipeline_v2.log runs/rdp_ldp_v2.log runs/dp_v2.log results/expert_eval_v2.log"
