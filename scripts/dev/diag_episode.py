"""Diagnostic: run one policy episode with a given checkpoint and append the metrics to a jsonl file (not an eval
results file). Used to show that the failed v2 RDP fine-tune is a model problem, not a v2 test-harness problem: the
v1 model on a v2 cucumber, and the fine-tuned model on a v1 cucumber.
  MUJOCO_GL=glfw python scripts/dev/diag_episode.py CKPT SEED CLASS results/diag_rdp_v2_finetune.jsonl
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from peel.policy_runner import SimPolicy, run_policy_episode  # noqa: E402


def main():
    ckpt, seed, cls, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    t0 = time.time()
    m = run_policy_episode(seed, SimPolicy(ckpt), shape_class=cls)
    keys = ("coverage", "gouge", "contact_steps", "cut_steps", "strokes", "success")
    r = {"ckpt": ckpt, "seed": seed, "shape_class": cls}
    r.update({k: m[k] if isinstance(m[k], bool) else float(m[k]) for k in keys})
    r["wall_s"] = round(time.time() - t0, 1)
    with open(out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(r, flush=True)


if __name__ == "__main__":
    main()
