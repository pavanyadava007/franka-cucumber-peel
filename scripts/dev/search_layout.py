"""Search base spacing + IK seeds so the two arms never touch along the whole stroke (and keep joint margin)."""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import peel.scene as scene  # noqa: E402
from peel.env import EpisodeConfig, PeelEnv  # noqa: E402
from peel.expert import blade_frame  # noqa: E402


def arm_contacts(m, d):
    bad = 0
    for c in d.contact[:d.ncon]:
        b1 = m.body(m.geom_bodyid[c.geom1]).name; b2 = m.body(m.geom_bodyid[c.geom2]).name
        if (b1.startswith("left") and b2.startswith("right")) or (b1.startswith("right") and b2.startswith("left")):
            bad += 1
    return bad

args = sys.argv[1:]
BASES = [float(args[0])] if args else [0.25]
LX = float(args[1]) if len(args) > 1 else 0.48
TRIALS = int(args[2]) if len(args) > 2 else 120
rng = np.random.default_rng(int(1000 * LX + 100 * BASES[0]))
nat_l = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.0])
results = []
for base_y in BASES:
    scene.BASE_Y = base_y
    PeelEnv.LEFT_P0 = np.array([LX, 0.0, 0.33])
    env = PeelEnv(EpisodeConfig(render_obs=False))
    m = env.model; d = mujoco.MjData(m); sc = env.scratch
    L, R = env.left, env.right
    best = None
    for trial in range(TRIALS):
        sl = np.clip(nat_l + rng.normal(0, 0.7, 7), L.jlo, L.jhi)
        ql, el = L.ik(m, sc, env.LEFT_P0, env.LEFT_R0, sl, iters=300, free_roll=True)
        if np.linalg.norm(el[:3]) > 1e-4: continue
        ml = np.min(np.minimum(ql[:6] - L.jlo[:6], L.jhi[:6] - ql[:6]))
        if ml < 0.3: continue
        ql[6] = L.jlo[6] + 0.08
        d.qpos[:] = 0; d.qpos[L.qadr] = ql; mujoco.mj_forward(m, d)
        pc = d.xpos[env.cuc_body]; Rc = d.xmat[env.cuc_body].reshape(3, 3)
        c = env.cfg.cucumber
        def top(z, up): return pc + Rc @ np.array([0, 0, z]) + np.array([0, 0, c.radius + up])
        Rb = blade_frame(Rc[:, 2], np.deg2rad(15))
        path = [top(z, up) for up in (0.05, -0.012) for z in np.linspace(c.peel_zone[0], c.peel_zone[1] + 0.01, 8)]
        sr = np.clip(np.array([0.0, -0.2, 0.0, -2.2, 0.0, 2.4, 0.0]) + rng.normal(0, 0.7, 7), R.jlo, R.jhi)
        Rb = env.RIGHT_R0 if False else Rb
        qr = sr; ok = True; mr = 9.0
        for i, p in enumerate(path):
            qr, er = R.ik(m, sc, p, Rb, qr, iters=200 if i == 0 else 60)
            if np.linalg.norm(er[:3]) > 5e-4: ok = False; break
            mr = min(mr, np.min(np.minimum(qr - R.jlo, R.jhi - qr)))
            d.qpos[R.qadr] = qr; mujoco.mj_forward(m, d)
            if arm_contacts(m, d) or mr < 0.25: ok = False; break
        if not ok: continue
        # clearance: distance between left and right link5/link6/link7 bodies at the near end
        dist = min(np.linalg.norm(d.xpos[m.body(f"left_link{k}").id] - d.xpos[m.body(f"right_link{j}").id]) for k in (4, 5, 6) for j in (4, 5, 6))
        score = min(ml, mr) + 0.5 * dist
        if best is None or score > best[0]:
            q_r0, _ = R.ik(m, sc, path[0], Rb, sr, iters=300)
            best = (score, ql, q_r0, ml, mr, dist)
    if best:
        print(f"LX {LX} BASE_Y {base_y}: score {best[0]:.2f} margins L {best[3]:.2f} R {best[4]:.2f} link clearance {best[5]:.3f} m")
        print("   LEFT", np.round(best[1], 4).tolist()); print("   RIGHT", np.round(best[2], 4).tolist())
    else:
        print(f"LX {LX} BASE_Y {base_y}: no collision-free layout found")
