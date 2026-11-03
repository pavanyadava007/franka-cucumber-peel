"""Run one scripted-expert episode through the GELLO path and print what happened."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peel.env import EpisodeConfig, PeelEnv  # noqa: E402
from peel.expert import ExpertParams, PeelExpert  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
use_force = (sys.argv[2] != "noforce") if len(sys.argv) > 2 else True
cls = sys.argv[3] if len(sys.argv) > 3 else None
cfg = EpisodeConfig.sample(seed, render_obs=False, shape_class=cls)
env = PeelEnv(cfg)
rng = np.random.default_rng(seed)
ep = ExpertParams.sample(rng, use_force=use_force)
ex = PeelExpert(env, ep, rng, gello=True)
t0 = time.time(); last = None; fns = []; phases = {}
for phase, sid, js, a in ex.command_stream():
    o = env.command_joint_state(js)
    phases[phase] = phases.get(phase, 0) + 1
    if phase == "stroke":
        fns.append(float(o["blade_fn"][0]))
    if sid != last:
        last = sid
        print(f"  stroke {sid}: coverage {env.skin.coverage():.3f} gouge {env.skin.gouge_fraction():.3f} roll {o['cucumber_roll'][0]:.2f}")
    if env.t > 120: break
m = env.metrics()
sh = cfg.cucumber.shape
print(f"class {cfg.cucumber.shape_class} L {sh.length*100:.0f} cm r {sh.r_mid*1000:.0f} mm taper {sh.taper:+.2f} bend {sh.bend_tip*1000:.0f} mm | seed {seed} R {cfg.cucumber.radius*1000:.1f} mm f_cut {cfg.skin.f_cut:.1f} f_target {ep.f_target:.1f} force_fb {use_force}")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
print("phase steps", phases, "stroke Fn mean %.1f p95 %.1f" % (np.mean(fns), np.percentile(fns, 95)) if fns else "")
print("wall %.1f s for %.1f s sim" % (time.time() - t0, env.t))
