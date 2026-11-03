"""Evaluate methods on held-out cucumbers (seeds 9000+, never used for training), stratified by shape class.

  MUJOCO_GL=glfw python scripts/eval_policies.py --method expert_force   --per-class 5
  MUJOCO_GL=glfw python scripts/eval_policies.py --method expert_noforce --per-class 5
  MUJOCO_GL=glfw python scripts/eval_policies.py --method rdp --ckpt runs/rdp_ldp/checkpoints/latest.ckpt
  MUJOCO_GL=glfw python scripts/eval_policies.py --method dp  --ckpt runs/dp/checkpoints/latest.ckpt
Results append to results/eval_<method>.jsonl (one line per episode); scripts/make_results.py builds the tables.
Policies (rdp, dp) run the learned stroke skill under the same orchestrator; the experts run the scripted stroke.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peel.shape import CLASS_SETS  # noqa: E402

CLASSES = CLASS_SETS["v1"]
METHODS = ["expert_force", "expert_noforce", "expert_force_nominal", "expert_noforce_nominal", "rdp", "dp"]


def expert_job(args):
    seed, cls, use_force, geometry = args
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from peel.rollout import run_expert_episode

    t0 = time.time()
    m, _, ep, cfg = run_expert_episode(
        seed, use_force=use_force, render_obs=False, record=False, shape_class=cls, geometry=geometry
    )
    return {"seed": seed, "shape_class": cls, **_clean(m), "wall_s": round(time.time() - t0, 1)}


def _clean(m):
    return {k: (v if isinstance(v, (bool, int, str)) else float(v)) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        choices=METHODS,
    )
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--start-seed", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--class-set", default="v1", choices=["v1", "v2"], help="v1 first shapes, v2 real types")
    ap.add_argument("--classes", default=None, help="comma list (default: all classes of the set)")
    ap.add_argument("--video-first", action="store_true", help="also save a video of the first seed per class")
    a = ap.parse_args()
    set_classes = CLASS_SETS[a.class_set]
    classes = a.classes.split(",") if a.classes else set_classes
    start = a.start_seed if a.class_set == "v1" else a.start_seed + 1000  # v2 held-out seeds 10000+
    jobs = [(start + 100 * ci + k, c) for ci, c in enumerate(set_classes) if c in classes for k in range(a.per_class)]
    suffix = "" if a.class_set == "v1" else "_v2"
    out = ROOT / "results" / f"eval_{a.method}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["seed"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    jobs = [j for j in jobs if j[0] not in done]
    with open(out, "a") as f:
        if a.method.startswith("expert"):
            use_force = a.method.startswith("expert_force")
            geometry = "nominal" if a.method.endswith("nominal") else "true"
            with get_context("spawn").Pool(a.workers) as pool:
                for r in pool.imap_unordered(expert_job, [(s, c, use_force, geometry) for s, c in jobs]):
                    f.write(json.dumps({"method": a.method, **r}) + "\n")
                    f.flush()
                    print(r, flush=True)
        else:
            from peel.policy_runner import SimPolicy, run_policy_episode

            sp = SimPolicy(a.ckpt)
            first = set()
            for seed, cls in jobs:
                t0 = time.time()
                video = None
                if a.video_first and cls not in first:
                    first.add(cls)
                    video = ROOT / "media" / "videos" / f"{a.method}{suffix}_{cls}_seed{seed}.mp4"
                m = run_policy_episode(seed, sp, shape_class=cls, video=video)
                r = {
                    "method": a.method,
                    "seed": seed,
                    "shape_class": cls,
                    **_clean(m),
                    "wall_s": round(time.time() - t0, 1),
                    "ckpt": a.ckpt,
                }
                f.write(json.dumps(r) + "\n")
                f.flush()
                print({k: r[k] for k in ("seed", "shape_class", "coverage", "gouge", "success", "wall_s")}, flush=True)


if __name__ == "__main__":
    main()
