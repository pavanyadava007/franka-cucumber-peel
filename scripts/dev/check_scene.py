"""Smoke check: build, reset, hold still for 2 s, print IK errors, drift and F/T at rest, save camera images."""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peel.env import EpisodeConfig, PeelEnv  # noqa: E402

out = Path(__file__).resolve().parents[2] / "media" / "debug"
out.mkdir(parents=True, exist_ok=True)
env = PeelEnv(EpisodeConfig.sample(0))
print("IK err left", np.round(env.ik_err_left, 5), "right", np.round(env.ik_err_right, 5))
print("q7 start", round(env.q7_start, 3), "roll range rad", round(env.roll_max, 3))
p0 = {s: a.pose(env.data)[0] for s, a in env.arms.items()}
for _ in range(48):
    o = env.step_tcp()
for s, a in env.arms.items():
    print(s, "drift mm", np.round((a.pose(env.data)[0] - p0[s]) * 1000, 2), "wrench", np.round(o[f"{s}_robot_tcp_wrench"], 2))
print("cucumber axis world:", np.round(env.cucumber_frame()[1][:, 2], 3), "centre", np.round(env.cucumber_frame()[0], 3))
print("blade pos", np.round(env.data.site_xpos[env.blade_site], 3), "top of cucumber far end", np.round(env.cucumber_point(env.cfg.cucumber.peel_zone[1]), 3))
for cam in ("external", "right_wrist", "overview"):
    imageio.imwrite(out / f"{cam}.png", env.render(cam, hw=(480, 640), pretty=True))
imageio.imwrite(out / "external_policy_96x128.png", env.render("external"))
print("saved images to", out)
