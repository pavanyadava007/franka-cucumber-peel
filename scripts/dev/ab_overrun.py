"""A/B: blade overrun past the peel-zone end (3 mm vs 8 mm) on given seeds and class."""
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def one(args):
    seed, ov, cls = args
    import os

    os.environ.setdefault("MUJOCO_GL", "glfw")
    import numpy as np

    from peel.env import EpisodeConfig, PeelEnv
    from peel.expert import ExpertParams, PeelExpert
    from peel.rollout import is_success

    cfg = EpisodeConfig.sample(seed, render_obs=False, shape_class=cls)
    env = PeelEnv(cfg)
    rng = np.random.default_rng(seed)
    ep = ExpertParams.sample(rng)
    ep.end_overrun = ov
    ex = PeelExpert(env, ep, rng)
    for _phase, _sid, js, _a in ex.command_stream():
        env.command_joint_state(js)
        if env.t > 150:
            break
    m = env.metrics()
    sh = cfg.cucumber.shape
    return (seed, ov, m["coverage"], m["gouge"], m["peak_fn"], is_success(m), sh.arc_mm_per_10cm, sh.length * 100)


OVERRUNS = tuple(float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0.003", "0.008"]))

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    cls = sys.argv[2]
    with get_context("spawn").Pool(4) as pool:
        for r in pool.map(one, [(s, ov, cls) for s in seeds for ov in OVERRUNS]):
            print("seed %d overrun %.3f: coverage %.3f gouge %.3f peak %.1f N success %s (arc %.1f mm/10cm, L %.0f cm)" % r)
