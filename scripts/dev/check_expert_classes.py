"""Run the scripted expert (no rendering) over shape classes x seeds and print a table."""
import json
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def one(args):
    cls, seed, use_force = args
    import os; os.environ.setdefault("MUJOCO_GL", "glfw")
    import numpy as np

    from peel.env import EpisodeConfig, PeelEnv
    from peel.expert import ExpertParams, PeelExpert
    from peel.rollout import is_success
    cfg = EpisodeConfig.sample(seed, render_obs=False, shape_class=cls); env = PeelEnv(cfg)
    rng = np.random.default_rng(seed); ex = PeelExpert(env, ExpertParams.sample(rng, use_force=use_force), rng)
    for phase, sid, js, a in ex.command_stream():
        env.command_joint_state(js)
        if env.t > 150: break
    m = env.metrics(); m["success"] = is_success(m)
    return {"class": cls, "seed": seed, "force": use_force, **{k: round(float(m[k]), 3) for k in ("coverage", "gouge", "peak_fn", "time")}, "success": bool(m["success"])}

if __name__ == "__main__":
    classes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["mini", "standard", "long", "thick", "tapered", "curved", "mixed"]
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [11, 12, 13]
    use_force = not (len(sys.argv) > 2 and sys.argv[2] == "noforce")
    jobs = [(c, s, use_force) for c in classes for s in seeds]
    with get_context("spawn").Pool(6) as pool:
        rows = pool.map(one, jobs)
    for r in rows: print(json.dumps(r))
    for c in classes:
        rs = [r for r in rows if r["class"] == c]
        print(f"{c:9s} success {sum(r['success'] for r in rs)}/{len(rs)}  coverage {min(r['coverage'] for r in rs):.2f}-{max(r['coverage'] for r in rs):.2f}  gouge max {max(r['gouge'] for r in rs):.3f}  peak F max {max(r['peak_fn'] for r in rs):.1f}")
