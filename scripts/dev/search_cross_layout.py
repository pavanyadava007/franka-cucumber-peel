"""Cross layout search: left arm holds the cucumber across the front (tip toward the right arm), right arm peels on top.
Accept only postures with no left-vs-right(+peeler) contact along the whole stroke and joint margin."""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peel.env import EpisodeConfig, PeelEnv  # noqa: E402
from peel.expert import blade_frame  # noqa: E402

X, Y, Z = (float(a) for a in sys.argv[1:4])
PeelEnv.LEFT_P0 = np.array([X, Y, Z])
env = PeelEnv(EpisodeConfig(render_obs=False)); m = env.model; d = mujoco.MjData(m); sc = env.scratch
L, R = env.left, env.right
RIGHT_SIDE = {"peeler", "blade_body"}

def side(b):
    n = m.body(b).name
    return "L" if n.startswith("left") else "R" if (n.startswith("right") or n in RIGHT_SIDE) else "o"

def arm_arm_contact():
    return any({side(m.geom_bodyid[c.geom1]), side(m.geom_bodyid[c.geom2])} == {"L", "R"} for c in d.contact[:d.ncon])

rng = np.random.default_rng(int(X * 100 + Y * 1000 + Z * 10)); reasons = {}; best = None
nat = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.0])
for trial in range(150):
    ql, el = L.ik(m, sc, env.LEFT_P0, env.LEFT_R0, np.clip(nat + rng.normal(0, 0.8, 7), L.jlo, L.jhi), iters=300,
                  free_roll=True)
    if np.linalg.norm(el[:3]) > 1e-4 or np.linalg.norm(el[3:]) > 1e-3:
        reasons["left_ik"] = reasons.get("left_ik", 0) + 1; continue
    ml = np.min(np.minimum(ql[:6] - L.jlo[:6], L.jhi[:6] - ql[:6]))
    if ml < 0.3:
        reasons["left_margin"] = reasons.get("left_margin", 0) + 1; continue
    ql[6] = L.jlo[6] + 0.08
    d.qpos[:] = 0; d.qpos[L.qadr] = ql; mujoco.mj_forward(m, d)
    pc = d.xpos[env.cuc_body]; Rc = d.xmat[env.cuc_body].reshape(3, 3); c = env.cfg.cucumber
    Rb = blade_frame(Rc[:, 2], np.deg2rad(15))
    path = [pc + Rc @ np.array([0, 0, z]) + np.array([0, 0, c.radius + up]) for up in (0.05, -0.012)
            for z in np.linspace(c.peel_zone[0], c.peel_zone[1] + 0.01, 8)]
    qr = np.clip(np.array([0.0, -0.2, 0.0, -2.2, 0.0, 2.4, 0.0]) + rng.normal(0, 0.8, 7), R.jlo, R.jhi)
    status, mr, qr0 = "ok", 9.0, None
    for i, p in enumerate(path):
        qr, er = R.ik(m, sc, p, Rb, qr, iters=300 if i == 0 else 80)
        if qr0 is None: qr0 = qr.copy()
        if np.linalg.norm(er[:3]) > 5e-4 or np.linalg.norm(er[3:]) > 5e-3: status = "right_ik"; break
        d.qpos[R.qadr] = qr; mujoco.mj_forward(m, d)
        if arm_arm_contact(): status = "arm_contact"; break
        mr = min(mr, np.min(np.minimum(qr - R.jlo, R.jhi - qr)))
        if mr < 0.25: status = "right_margin"; break
    reasons[status] = reasons.get(status, 0) + 1
    if status == "ok":
        cost = np.linalg.norm((ql - nat)[:6]) - 2 * min(ml, mr)
        if best is None or cost < best[0]:
            best = (cost, ql, qr0, ml, mr)
print(f"LEFT_P0 ({X}, {Y}, {Z}):", reasons)
if best:
    print("   margins L %.2f R %.2f" % (best[3], best[4]))
    print("   LEFT", np.round(best[1], 4).tolist()); print("   RIGHT", np.round(best[2], 4).tolist())
