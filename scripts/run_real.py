"""Run the bimanual peeling stack on two real Franka Emika Pandas, or end to end on the mock backend.

UNTESTED ON HARDWARE. Only `--backend mock` is tested (tests/test_real_mock.py). Hardware runs are DRY RUNS by
default: the arms are connected and read, the cameras run, the safety checks run, targets are computed and logged,
but no motion is sent. Add --execute to move the robots (you will be asked to type YES). See docs/REAL_ROBOT.md.

Modes
  status        connect, print joint states, TCP poses, wrenches, numpy-FK consistency and camera frames
  teleop-gello  GELLO leader pair -> real robots (or mock), recorded in the same .npz format as the sim demos
  policy        orchestrator (peel/expert.py) + learned stroke skill (peel/policy_runner.py) on the robots
                (--policy expert runs the scripted, force-regulated stroke instead: the baseline)
  scan          roll the peeled cucumber in front of the external camera and save frames for coverage scoring

Examples
  # no hardware at all: synthetic leader / stub policy on the mock robots
  python scripts/run_real.py teleop-gello --backend mock --gello mock --steps 240 --record
  python scripts/run_real.py policy --backend mock --policy stub --max-strokes 2
  # replay a recorded sim GELLO stream through the mock robots
  python scripts/run_real.py teleop-gello --backend mock --gello-replay data/raw/ep_002002.npz --record
  # hardware dry run (no motion), then the real thing
  python scripts/run_real.py status --backend franky --config configs/real_cell.json
  python scripts/run_real.py teleop-gello --backend franky --config configs/real_cell.json \\
      --left-port /dev/serial/by-id/... --right-port /dev/serial/by-id/... \\
      --left-offsets ... --right-offsets ... --cucumber 0.22 0.038 0.041 --record
  python scripts/run_real.py policy --backend franky --config configs/real_cell.json --ckpt <ckpt> --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peel.env import GRIP_MAX, rot6d
from peel.real_robot import (
    CucumberFixture,
    RealBimanualFranka,
    RealConfig,
    SafetyAbort,
    StubStrokePolicy,
    panda_fk,
)
from peel.recorder import PHASE_ID, EpisodeRecorder, action20_from_env
from peel.rollout import is_success

PANDA_SIGNS = (1, -1, 1, 1, 1, -1, 1)  # GELLO README "Joint Signs Reference" for Panda (see docs: inconsistency)


# ---------------------------------------------------------------------------------------------- leaders
class SyntheticLeader:
    """No-hardware GELLO stand-in: the follower's start joints plus a slow sine on right-arm joints 1, 2 and 4."""

    def __init__(self, q16, amp: float = 0.04, hz: float = 0.25):
        self.q0 = np.asarray(q16, float).copy()
        self.amp, self.hz, self.k = amp, hz, 0

    def act(self, obs):
        t = self.k / 24.0
        self.k += 1
        q = self.q0.copy()
        for j, ph in ((8, 0.0), (9, 1.0), (11, 2.0)):
            q[j] += self.amp * np.sin(2 * np.pi * self.hz * t + ph)
        return q


class ReplayLeader:
    def __init__(self, npz):
        self.stream = iter(np.load(npz)["gello_cmd"])

    def act(self, obs):
        return next(self.stream)


def make_gello_pair(a):
    sys.path.insert(0, str(ROOT / "third_party" / "gello"))
    from gello.agents.agent import BimanualAgent
    from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent

    def one(port, offsets):
        cfg = DynamixelRobotConfig(
            joint_ids=(1, 2, 3, 4, 5, 6, 7),
            joint_offsets=tuple(offsets),
            joint_signs=tuple(a.signs),
            gripper_config=tuple(int(x) for x in a.gripper),
        )
        return GelloAgent(port=port, dynamixel_config=cfg)

    if not (a.left_port and a.right_port and a.left_offsets and a.right_offsets):
        raise SystemExit(
            "real GELLO needs --left-port --right-port --left-offsets --right-offsets "
            "(offsets from third_party/gello/scripts/gello_get_offset.py)"
        )
    return BimanualAgent(one(a.left_port, a.left_offsets), one(a.right_port, a.right_offsets))


# ---------------------------------------------------------------------------------------------- recording helpers
def label_strokes_from_force(
    rec: EpisodeRecorder, f_on: float = 1.0, min_len: int = 12, merge_gap: int = 6, pre: int = 24, post: int = 12
) -> int:
    """Human demos have no phase labels. Mark contact segments (estimated blade normal force > f_on N for at least
    min_len steps) as "stroke", up to `pre` steps before as "approach" and `post` after as "lift", one stroke_id each,
    so make_rdp_zarr(kind="stroke") can cut clips. Everything else keeps phase "policy" and stroke_id -1."""
    fn = np.array([float(x[0]) for x in rec.buf.get("blade_fn", [])])
    n = len(fn)
    idx = np.flatnonzero(fn > f_on)
    segs = []
    if idx.size:
        start = prev = int(idx[0])
        for j in idx[1:]:
            if j - prev > merge_gap:
                segs.append((start, prev))
                start = int(j)
            prev = int(j)
        segs.append((start, prev))
    segs = [(s, e) for s, e in segs if e - s + 1 >= min_len]
    phase = np.full(n, PHASE_ID["policy"], np.int8)
    sid = np.full(n, -1, np.int16)
    for k, (s, e) in enumerate(segs):
        lo = max(s - pre, segs[k - 1][1] + 1 if k else 0)
        hi = min(e + post, segs[k + 1][0] - 1 if k + 1 < len(segs) else n - 1)
        phase[lo:s], phase[s : e + 1], phase[e + 1 : hi + 1] = (
            PHASE_ID["approach"],
            PHASE_ID["stroke"],
            PHASE_ID["lift"],
        )
        sid[lo : hi + 1] = k
    rec.buf["phase"] = [np.array([p], np.int8) for p in phase]
    rec.buf["stroke_id"] = [np.array([s], np.int16) for s in sid]
    return len(segs)


def ask_success(mode: str, default: bool | None = None) -> bool:
    if mode == "yes":
        return True
    if mode == "no":
        return False
    ans = input("\nWas this a good demo/trial (no gouge, no safety stop)? [y/N] ").strip().lower()
    return ans in ("y", "yes")


def out_path(out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"ep_{seed:06d}.npz"
    k = 1
    while p.exists():
        p = out_dir / f"ep_{seed:06d}_{k}.npz"
        k += 1
    return p


# ---------------------------------------------------------------------------------------------- setup
def build_env(a) -> RealBimanualFranka:
    cfg = RealConfig.from_json(a.config) if a.config else RealConfig()
    fixture = CucumberFixture(
        length=a.cucumber[0], d_grip=a.cucumber[1], d_tip=a.cucumber[2], grasp_depth=a.grasp_depth
    )
    dry_run = not a.execute
    if a.backend != "mock" and not dry_run and not a.yes:
        print("EXECUTE MODE: the robots WILL MOVE. Enable device in hand, workspace clear, blade guarded.")
        if input("Type YES to continue: ").strip() != "YES":
            raise SystemExit("aborted by operator")
    cams = "mock" if a.mock_cameras else "auto"
    env = RealBimanualFranka(
        cfg,
        backend=a.backend,
        fixture=fixture,
        dry_run=dry_run,
        render_obs=not a.no_images,
        seed=a.seed,
        cameras=cams,
        realtime=True if a.realtime else None,
    )
    print(
        f"backend={env.backend.name} dry_run={env.dry_run} realtime={env.realtime} "
        f"cucumber L={fixture.length:.3f} d={fixture.d_grip:.3f}/{fixture.d_tip:.3f} m"
    )
    return env


def operator_insert_cucumber(env, a):
    if env.backend.name != "mock" and not env.dry_run and not a.yes:
        input("Left gripper is open. Insert the cucumber (end against the finger stop) and press Enter to grasp.")


def print_status(env: RealBimanualFranka):
    o = env.obs()
    for side in ("left", "right"):
        st = env._states[side]
        cfg = env.config.arm(side)
        T_fk = panda_fk(st.q, cfg.F_T_TCP)
        err = float(np.linalg.norm(T_fk[:3, 3] - st.T[:3, 3]))
        print(f"[{side}] q = {np.round(st.q, 3)}")
        print(
            f"        TCP world = {np.round(o[f'{side}_robot_tcp_pose'][:3], 4)}  gripper = "
            f"{o[f'{side}_robot_gripper_width'][0]:.4f} m  numpy-FK vs robot TCP = {1000 * err:.2f} mm"
        )
        print(f"        wrench (env on tool, world) = {np.round(o[f'{side}_robot_tcp_wrench'], 2)}")
    for k in ("external_img", "right_wrist_img"):
        if k in o:
            print(f"[camera] {k}: {o[k].shape} {o[k].dtype} mean {o[k].mean():.1f}")
    print("safety:", env.config.safety)


# ---------------------------------------------------------------------------------------------- modes
def run_teleop(env: RealBimanualFranka, a) -> dict:
    if a.gello_replay:
        leader = ReplayLeader(a.gello_replay)
        source = "gello_replay"
    elif a.gello == "mock":
        leader = None  # built after reset (needs the start joints)
        source = "gello_synthetic"
    else:
        leader = make_gello_pair(a)
        source = "gello_human"
    env.reset(before_grasp=lambda: operator_insert_cucumber(env, a))
    if leader is None:
        leader = SyntheticLeader(env.get_joint_state())
    if source == "gello_human":
        # like GELLO's run_env.py: do not start until the leader matches the follower (max 0.8 rad)
        while True:
            target = env.get_joint_state()
            delta = np.asarray(leader.act({})) - target
            delta[[7, 15]] = 0
            if np.abs(delta).max() < 0.8:
                break
            print(
                "leader - follower [deg]  L:",
                np.round(np.rad2deg(delta[:7])).astype(int),
                " R:",
                np.round(np.rad2deg(delta[8:15])).astype(int),
                end="\r",
            )
            time.sleep(0.1)
            env.obs(render=False)  # keep reading the robots (safety checks)
        print("\nleader aligned, starting")
    rec = EpisodeRecorder() if a.record else None
    obs = env.last_obs
    aborted = ""
    try:
        while env.t < a.max_time and (a.steps is None or env.step_count < a.steps):
            try:
                js = np.asarray(leader.act({}), dtype=float)
            except StopIteration:
                break
            new_obs = env.command_joint_state(js)
            if rec is not None:
                rec.add(obs, action20_from_env(env), "policy", 0, js)
                rec.buf.setdefault("t_wall", []).append(np.array([time.monotonic()]))  # true loop timing
            obs = new_obs
            if not a.quiet:
                print(
                    f"t {env.t:5.1f} s  blade force {env.mean_fn_step:5.1f} N  |F_right| "
                    f"{np.linalg.norm(obs['right_robot_tcp_wrench'][:3]):5.1f} N  skin removed (estimate) "
                    f"{100 * env.skin.coverage():5.1f} %",
                    end="\r",
                )
    except SafetyAbort as e:
        aborted = str(e)
        print(f"\nSAFETY ABORT: {e}")
    except KeyboardInterrupt:
        print("\nstopped by operator")
    return finish(env, a, rec, source, aborted)


def warm_up(runner, policy, env: RealBimanualFranka) -> float:
    """One slow-policy inference on the current observation, timed, with the arms not yet streaming."""
    torch = policy.torch
    o = env.obs()
    obs_t, _ = runner._obs_dict([o] * policy.To_full)
    t0 = time.monotonic()
    with torch.no_grad():
        if policy.latent:
            policy.policy.predict_action(
                obs_t, dataset_obs_temporal_downsample_ratio=policy.dataset_obs_ds, return_latent_action=True
            )
        else:
            policy.policy.predict_action(obs_t)
    dt = time.monotonic() - t0
    if hasattr(policy, "calls"):
        policy.calls = 0
    print(f"policy warm-up inference: {dt:.2f} s")
    return dt


def run_policy(env: RealBimanualFranka, a) -> dict:
    from peel.expert import ExpertParams, PeelExpert
    from peel.policy_runner import StrokeSkillRunner

    if a.policy == "expert":  # scripted stroke with force regulation from the measured wrench (baseline)
        policy, runner = None, None
    elif a.policy == "stub":
        policy = StubStrokePolicy(drift=a.stub_drift)
    else:
        from peel.policy_runner import SimPolicy

        if not a.ckpt:
            raise SystemExit("--ckpt is required unless --policy stub or expert")
        policy = SimPolicy(a.ckpt, device=a.device)
    if policy is not None:
        runner = StrokeSkillRunner(policy)
        warm_up(runner, policy, env)  # before streaming: the first CUDA call can outlast the heartbeat limits
    env.reset(before_grasp=lambda: operator_insert_cucumber(env, a))
    rng = np.random.default_rng(a.seed)
    ex = PeelExpert(env, ExpertParams.sample(rng), rng, gello=False)
    ex.orch.max_strokes = a.max_strokes
    rec = EpisodeRecorder() if a.record else None

    def on_step(e, phase, k):
        if rec is not None and e.prev_obs is not None:
            rec.add(e.prev_obs, action20_from_env(e), phase, ex.stroke_id, None)
            rec.buf.setdefault("t_wall", []).append(np.array([time.monotonic()]))

    aborted = ""
    gen = ex.plan()
    try:
        phase, sid, left, right = next(gen)
    except StopIteration:
        phase = None
    try:
        while phase is not None and env.t < a.max_time:
            if runner is not None and phase in ("approach", "descend", "stroke", "lift"):
                runner.run(env, left, max_steps=a.stroke_max_steps, on_step=on_step)
                while phase in ("approach", "descend", "stroke", "lift"):  # skip the scripted stroke
                    try:
                        phase, sid, left, right = gen.send(None)
                    except StopIteration:
                        phase = None
                        break
                continue
            act = np.concatenate(
                [
                    left[0],
                    rot6d(left[1]),
                    right[0],
                    rot6d(right[1]),
                    [2 * env.cfg.cucumber.shape.r_grip, env.right.grip_w],
                ]
            )
            env.step_action(act)
            on_step(env, phase, env.step_count)
            if not a.quiet:
                print(
                    f"t {env.t:5.1f} s  phase {phase:8s} stroke {sid:2d}  coverage (estimate) "
                    f"{100 * env.skin.coverage():5.1f} %",
                    end="\r",
                )
            try:
                phase, sid, left, right = next(gen)
            except StopIteration:
                phase = None
    except SafetyAbort as e:
        aborted = str(e)
        print(f"\nSAFETY ABORT: {e}")
    except KeyboardInterrupt:
        print("\nstopped by operator")
    name = "scripted_expert" if policy is None else policy.name.replace(" ", "_")
    out = finish(env, a, rec, "policy_" + name, aborted)
    out["strokes"] = ex.orch.strokes
    out["policy_calls"] = getattr(policy, "calls", None)
    return out


def run_scan(env: RealBimanualFranka, a) -> dict:
    """Coverage scan for the evaluation: park the peeler arm, roll the cucumber with left joint 7 through
    --scan-views evenly spaced angles over its available range and save one full-resolution external frame per view
    (plus the 96x128 policy image). Segment skin vs flesh offline (see docs/REAL_ROBOT.md)."""
    env.reset(grasp=False, move=not a.scan_in_place)
    rolls = np.linspace(0.0, env.roll_max, a.scan_views)
    frames, small, done = [], [], []
    cam = env.cameras.get("external_img")
    try:
        for r in rolls:
            for _ in range(240):  # at most 10 s per view
                env.step_tcp(left=(env.left.tgt_p, env.left_R(r)))
                if abs(env.last_obs["cucumber_roll"][0] - r) < 0.02:
                    break
            for _ in range(12):  # settle 0.5 s
                env.step_tcp(left=(env.left.tgt_p, env.left_R(r)))
            o = env.obs()
            if cam is not None:
                raw = getattr(cam, "last_raw", None)
                frames.append(raw if raw is not None else o["external_img"])
                small.append(o["external_img"])
            done.append(float(o["cucumber_roll"][0]))
    except SafetyAbort as e:
        print(f"\nSAFETY ABORT: {e}")
    env.stop("scan end")
    p = out_path(Path(a.out) / "scans", a.seed)
    np.savez_compressed(p, roll=np.array(done), external_raw=np.array(frames), external_img=np.array(small))
    print("saved", p, f"({len(done)} views)")
    return {"saved": str(p), "views": len(done)}


def finish(env: RealBimanualFranka, a, rec, source: str, aborted: str) -> dict:
    env.stop("episode end")
    m = env.metrics()
    m["aborted"] = aborted
    m["sim_rule_success"] = bool(is_success(m)) and not aborted  # from the ESTIMATED skin map only
    print("\n", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    if rec is not None and len(rec):
        n_strokes = label_strokes_from_force(rec) if a.label_strokes else None
        success = False if aborted else ask_success(a.success)
        f = env.fixture
        meta = {
            "seed": a.seed,
            "success": success,
            "coverage": m["coverage"],
            "gouge": m["gouge"],
            "coverage_is_estimate": True,
            "source": source + ("_mock" if env.backend.name == "mock" else "_real"),
            "shape_class": f.shape_class,
            "backend": env.backend.name,
            "dry_run": env.dry_run,
            "aborted": aborted,
            "cucumber_length": f.length,
            "cucumber_d_grip": f.d_grip,
            "cucumber_d_tip": f.d_tip,
            "labelled_strokes": -1 if n_strokes is None else n_strokes,
            "grip_max": GRIP_MAX,
        }
        p = out_path(Path(a.out), a.seed)
        rec.save(p, meta)
        m["saved"] = str(p)
        print("saved", p, f"({len(rec)} steps)")
    return m


# ---------------------------------------------------------------------------------------------- CLI
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["status", "teleop-gello", "policy", "scan"])
    ap.add_argument("--backend", default="mock", choices=["mock", "franky"])
    ap.add_argument("--config", default=None, help="RealConfig JSON (IPs, calibrated frames, serials, limits)")
    ap.add_argument("--write-config", default=None, help="write the default RealConfig JSON here and exit")
    ap.add_argument("--execute", action="store_true", help="hardware: actually move (default is a dry run)")
    ap.add_argument("--yes", action="store_true", help="skip interactive confirmations (scripted runs)")
    ap.add_argument("--realtime", action="store_true", help="pace the mock at 24 Hz wall clock")
    ap.add_argument("--mock-cameras", action="store_true", help="synthetic images instead of RealSense")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument(
        "--cucumber",
        type=float,
        nargs=3,
        default=(0.23, 0.040, 0.040),
        metavar=("L", "D_GRIP", "D_TIP"),
        help="caliper measurements in metres",
    )
    ap.add_argument("--grasp-depth", type=float, default=0.080, help="hand-z of the gripped cucumber end (m)")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--max-time", type=float, default=120.0)
    ap.add_argument("--steps", type=int, default=None, help="stop after this many control steps")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "real"))
    ap.add_argument("--label-strokes", action="store_true", help="auto-label stroke phases from the blade force")
    ap.add_argument("--success", choices=["ask", "yes", "no"], default="ask")
    ap.add_argument("--quiet", action="store_true")
    # GELLO
    ap.add_argument("--gello", choices=["real", "mock"], default="real")
    ap.add_argument("--gello-replay", default=None, help="npz with a gello_cmd stream (no leader hardware)")
    ap.add_argument("--left-port")
    ap.add_argument("--right-port")
    ap.add_argument("--left-offsets", type=float, nargs=7)
    ap.add_argument("--right-offsets", type=float, nargs=7)
    ap.add_argument("--signs", type=int, nargs=7, default=PANDA_SIGNS)
    ap.add_argument("--gripper", type=float, nargs=3, default=(8, 195, 152), help="id, open deg, closed deg")
    # policy
    ap.add_argument(
        "--policy",
        choices=["ckpt", "stub", "expert"],
        default="ckpt",
        help="ckpt = trained RDP/DP stroke skill, expert = scripted force-regulated stroke, stub = test",
    )
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stub-drift", type=float, default=0.0)
    ap.add_argument("--max-strokes", type=int, default=14)
    ap.add_argument("--stroke-max-steps", type=int, default=160)
    # scan
    ap.add_argument("--scan-views", type=int, default=12)
    ap.add_argument("--scan-in-place", action="store_true", help="do not move to the start pose first")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse(argv)
    if a.write_config:
        RealConfig().to_json(a.write_config)
        print("wrote", a.write_config)
        return {}
    env = build_env(a)
    try:
        if a.mode == "status":
            print_status(env)
            return {}
        if a.mode == "teleop-gello":
            return run_teleop(env, a)
        if a.mode == "scan":
            return run_scan(env, a)
        return run_policy(env, a)
    finally:
        env.close()


if __name__ == "__main__":
    main()
