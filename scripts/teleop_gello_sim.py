"""Drive the bimanual simulation with a real pair of GELLO leader arms (Franka/Panda version) and record demos.

Hardware: two GELLO Panda leader arms (7 Dynamixel XL330 joints + gripper trigger each), U2D2 USB adapters.
On WSL2, attach the adapters first (Windows PowerShell as admin):  usbipd list ; usbipd attach --wsl --busid <id>

Calibrate each arm once with GELLO's tool (offsets are multiples of pi/2):
  python third_party/gello/scripts/gello_get_offset.py --start-joints 0 0 0 -1.57 0 1.57 0 \
      --joint-signs 1 -1 1 1 1 -1 1 --port /dev/serial/by-id/<left adapter>
(GELLO's README lists Panda signs 1 -1 1 1 1 -1 1 in its reference table but 1 1 1 1 1 -1 1 in the example;
 if joint 2 moves the wrong way in the viewer, flip the second sign.)

Run:
  MUJOCO_GL=glfw python scripts/teleop_gello_sim.py --left-port /dev/serial/by-id/... --right-port ... \
      --left-offsets 3.14 1.57 ... --right-offsets ... --record --seed 5000 --shape-class curved
A MuJoCo viewer opens; a bar next to the cucumber shows the blade force (green inside 8-13 N, red above 20 N).
Press Ctrl+C to stop; the episode is saved to data/teleop/ep_<seed>.npz in the same format as scripted demos.

No hardware?  --replay data/raw/ep_002002.npz  feeds a recorded GELLO command stream through the same loop,
which tests everything except the Dynamixel reading.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "gello"))

from peel.env import EpisodeConfig, PeelEnv  # noqa: E402
from peel.recorder import EpisodeRecorder, action20_from_env  # noqa: E402
from peel.rollout import is_success  # noqa: E402

PANDA_SIGNS = (1, -1, 1, 1, 1, -1, 1)


def make_leader(port, offsets, signs, gripper):
    from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent

    cfg = DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=tuple(offsets),
        joint_signs=tuple(signs),
        gripper_config=tuple(gripper),
    )
    return GelloAgent(port=port, dynamixel_config=cfg)


def force_bar(viewer, env):
    """Draw the blade normal force as a vertical bar next to the cucumber (operator force display)."""
    import mujoco

    fn = float(getattr(env, "mean_fn_step", 0.0))
    scn = viewer.user_scn
    scn.ngeom = 0
    base = env.cucumber_point(env.cfg.cucumber.peel_zone[0], 0.0) + np.array([0.0, 0.08, -0.05])
    h = min(fn, 30.0) / 30.0 * 0.15
    rgba = [0.2, 0.8, 0.2, 1] if 8.0 <= fn <= 13.0 else ([0.9, 0.2, 0.2, 1] if fn > 20.0 else [0.95, 0.7, 0.1, 1])
    for k, (height, color) in enumerate(((0.15, [0.8, 0.8, 0.8, 0.3]), (max(h, 1e-3), rgba))):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([0.008, 0.008 + 0.001 * k, height / 2]),
            base + np.array([0, 0, height / 2]),
            np.eye(3).flatten(),
            np.array(color, np.float32),
        )
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-port")
    ap.add_argument("--right-port")
    ap.add_argument("--left-offsets", type=float, nargs=7)
    ap.add_argument("--right-offsets", type=float, nargs=7)
    ap.add_argument("--signs", type=int, nargs=7, default=PANDA_SIGNS)
    ap.add_argument("--gripper", type=float, nargs=3, default=(8, 195, 152), help="id, open deg, closed deg")
    ap.add_argument("--replay", default=None, help="npz with a recorded gello_cmd stream (no hardware needed)")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--shape-class", default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--max-time", type=float, default=120.0)
    a = ap.parse_args()

    if a.replay:
        ep = np.load(a.replay)
        stream = iter(ep["gello_cmd"])
        seed = int(ep["meta_seed"]) if "meta_seed" in ep else a.seed
        shape_class = str(ep["meta_shape_class"]) if "meta_shape_class" in ep else a.shape_class
        act = lambda obs: next(stream)  # noqa: E731
    else:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            make_leader(a.left_port, a.left_offsets, a.signs, a.gripper),
            make_leader(a.right_port, a.right_offsets, a.signs, a.gripper),
        )
        seed, shape_class = a.seed, a.shape_class
        act = agent.act

    env = PeelEnv(EpisodeConfig.sample(seed, render_obs=a.record, shape_class=shape_class))
    if not a.replay:
        # like GELLO's run_env.py: do not start until the leader roughly matches the follower (max 0.8 rad)
        target = env.get_joint_state()
        while True:
            q = np.asarray(act({}))
            delta = np.abs(q - target)
            delta[[7, 15]] = 0
            if delta.max() < 0.8:
                break
            j = int(delta.argmax())
            print(
                f"move leader joint {j % 8 + 1} of the {'left' if j < 8 else 'right'} arm by "
                f"{np.rad2deg(target[j] - q[j]):+.0f} deg to start",
                end="\r",
            )
            time.sleep(0.1)
        print("\nleader aligned, starting")

    rec = EpisodeRecorder() if a.record else None
    viewer = None
    if not a.no_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    obs = env.obs()
    dt = 1.0 / 24.0
    try:
        while env.t < a.max_time:
            t0 = time.time()
            try:
                js = np.asarray(act({}), dtype=float)
            except StopIteration:
                break
            new_obs = env.command_joint_state(js)
            if rec is not None:
                rec.add(obs, action20_from_env(env), "policy", 0, js)
            obs = new_obs
            if viewer is not None:
                force_bar(viewer, env)
                viewer.sync()
            print(
                f"t {env.t:5.1f} s  blade force {getattr(env, 'mean_fn_step', 0):5.1f} N  "
                f"skin removed {100 * env.skin.coverage():5.1f} %  gouged {100 * env.skin.gouge_fraction():4.1f} %",
                end="\r",
            )
            if not a.replay:
                time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    m = env.metrics()
    m["success"] = is_success(m)
    print("\n", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    if rec is not None and len(rec):
        out = ROOT / "data" / "teleop"
        out.mkdir(parents=True, exist_ok=True)
        rec.save(
            out / f"ep_{seed:06d}.npz",
            dict(
                seed=seed,
                success=m["success"],
                coverage=m["coverage"],
                gouge=m["gouge"],
                source="gello_human" if not a.replay else "gello_replay",
                shape_class=env.cfg.cucumber.shape_class,
            ),
        )
        print("saved", out / f"ep_{seed:06d}.npz")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
