"""Mock-backend tests for the real-robot layer (peel/real_robot.py, scripts/run_real.py).

No hardware is involved. These tests check that the real env exposes the same interface and observation layout as
the simulation, that GELLO and policy code paths run on it, and that the safety layer trips. They say nothing about
how franky, the RealSense cameras or the Bota sensor behave on real hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from peel import real_robot as rr
from peel.env import EpisodeConfig, PeelEnv
from peel.real_robot import (
    BotaFTSensor,
    CucumberFixture,
    RealBimanualFranka,
    RealConfig,
    SafetyAbort,
    StubStrokePolicy,
)
from peel.recorder import EpisodeRecorder, action20_from_env

DT = 1.0 / 24


def make_env(**kw) -> RealBimanualFranka:
    env = RealBimanualFranka(backend="mock", **kw)
    env.reset()
    return env


@pytest.fixture(scope="module")
def sim_env():
    env = PeelEnv(EpisodeConfig(cucumber=CucumberFixture().spec(), render_obs=False))
    yield env
    env.close()


# ---------------------------------------------------------------------------------------------- import hygiene
def test_hardware_imports_are_lazy():
    for mod in ("franky", "pyrealsense2", "bota_driver", "polymetis"):
        assert mod not in sys.modules, f"{mod} was imported by peel.real_robot"


def test_dry_run_is_default():
    env = RealBimanualFranka(backend="mock")
    assert env.dry_run and env.config.safety.dry_run
    import run_real

    assert not run_real.parse(["status"]).execute  # the CLI needs --execute to move hardware
    be = rr.FrankyBimanualBackend(RealConfig())  # builds worker handles only, nothing is started or imported
    assert be.dry_run
    assert "franky" not in sys.modules


# ---------------------------------------------------------------------------------------------- kinematics / frames
def test_numpy_fk_matches_mujoco_and_tool_transforms():
    import mujoco

    from peel.scene import SceneSpec, build_model

    m = build_model(SceneSpec())
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for side in ("left", "right"):
        jid = [m.joint(f"{side}_joint{i}").id for i in range(1, 8)]
        l0, l7 = m.body(f"{side}_link0").id, m.body(f"{side}_link7").id
        for _ in range(20):
            q = rng.uniform(rr.Q_MIN, rr.Q_MAX)
            d.qpos[[m.jnt_qposadr[j] for j in jid]] = q
            mujoco.mj_kinematics(m, d)
            W_l0 = rr.make_T(d.xpos[l0], d.xmat[l0].reshape(3, 3))
            W_F = rr.make_T(d.xpos[l7], d.xmat[l7].reshape(3, 3)) @ rr.make_T([0, 0, rr.FLANGE_D])
            assert np.allclose(W_l0 @ rr.panda_fk(q), W_F, atol=1e-9)
        assert np.allclose(W_l0, rr.default_world_T_base(side), atol=1e-12)
    st = rr.sim_tool_transforms()
    for side in ("left", "right"):
        assert np.allclose(st[side], rr.default_F_T_tcp(side), atol=1e-5)


def test_jacobian_matches_finite_differences():
    q = PeelEnv.Q_RIGHT_SEED.copy()
    F = rr.default_F_T_tcp("right")
    J = rr.panda_jacobian(q, F)
    eps = 1e-6
    T0 = rr.panda_fk(q, F)
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        T1 = rr.panda_fk(q + dq, F)
        assert np.allclose((T1[:3, 3] - T0[:3, 3]) / eps, J[:3, i], atol=1e-4)
        assert np.allclose(rr.rotvec(T1[:3, :3] @ T0[:3, :3].T) / eps, J[3:, i], atol=1e-4)


def test_start_pose_matches_sim(sim_env):
    env = make_env(render_obs=False)
    so, ro = sim_env.obs(render=False), env.obs(render=False)
    for side in ("left", "right"):
        assert np.allclose(so[f"{side}_robot_tcp_pose"], ro[f"{side}_robot_tcp_pose"], atol=2e-3)
    assert np.allclose(sim_env.cucumber_frame()[0], env.cucumber_frame()[0], atol=1e-3)
    assert np.allclose(sim_env.cucumber_frame()[1], env.cucumber_frame()[1], atol=1e-3)
    assert np.allclose(sim_env.park_pose(), env.park_pose(), atol=2e-3)
    assert abs(sim_env.roll_max - env.roll_max) < 1e-3


# ---------------------------------------------------------------------------------------------- observations
def test_obs_keys_and_shapes_match_peelenv(sim_env):
    env = make_env()
    so = sim_env.obs(render=False)
    ro = env.obs()
    for k, v in so.items():
        assert k in ro, k
        assert ro[k].shape == v.shape and ro[k].dtype == v.dtype, (k, ro[k].shape, v.shape, ro[k].dtype, v.dtype)
    hw = tuple(sim_env.image_hw)
    for k in ("external_img", "right_wrist_img"):  # PeelEnv renders (H, W, 3) uint8 at image_hw
        assert ro[k].shape == (*hw, 3) and ro[k].dtype == np.uint8
    assert env.num_dofs() == sim_env.num_dofs() == 16
    assert env.get_joint_state().shape == sim_env.get_joint_state().shape
    assert set(env.get_observations()) == set(sim_env.get_observations())
    obs = env.step_action(np.concatenate([ro["left_robot_tcp_pose"], ro["right_robot_tcp_pose"], [0.04, 0.018]]))
    assert set(obs) == set(ro)


def test_wrench_conversion_from_libfranka():
    # environment pushes the tool UP with 10 N at the TCP; TCP is 0.1 m in x and 0.2 m in z from the flange
    F_T_EE = rr.make_T([0.1, 0.0, 0.2])
    O_T_EE = rr.make_T([0.4, 0.1, 0.3])
    O_F_ext_hat_K = np.array([0, 0, -10.0, 0, 0, 0])  # libfranka: robot on environment is positive
    w = rr.franka_wrench_to_sim(O_F_ext_hat_K, O_T_EE, F_T_EE)
    assert np.allclose(w[:3], [0, 0, 10.0])
    assert np.allclose(w[3:], np.cross([0.1, 0.0, 0.2], [0, 0, 10.0]))  # torque about the flange


def test_mock_contact_force_reads_like_sim():
    env = make_env(render_obs=False)
    top, n, _ = env.surface_top(0.0)
    o = None
    for _ in range(40):  # press the blade 1 cm below the estimated top surface
        o = env.step_tcp(right=(top - 0.01 * n, env.right.tgt_R))
    f = o["right_robot_tcp_wrench"][:3]
    assert f @ n > 3.0  # environment pushes the blade outward, like PeelEnv.wrench()
    assert o["blade_fn"][0] > 3.0


# ---------------------------------------------------------------------------------------------- teleop path
def test_gello_teleop_steps_and_recording(tmp_path):
    env = make_env()
    q0 = env.get_joint_state()
    rec = EpisodeRecorder()
    obs = env.last_obs
    for k in range(36):
        js = q0.copy()
        js[8 + 1] += 0.15 * min(1.0, k / 24)  # right shoulder moves the blade
        js[6] += 0.4 * min(1.0, k / 24)  # left joint 7 rolls the cucumber
        new = env.command_joint_state(js)
        rec.add(obs, action20_from_env(env), "policy", 0, js)
        obs = new
    T_goal = env.config.right.W_T_B @ rr.panda_fk(js[8:15], env.config.right.F_T_TCP)
    assert np.linalg.norm(obs["right_robot_tcp_pose"][:3] - T_goal[:3, 3]) < 0.01
    assert abs(obs["cucumber_roll"][0] - 0.4) < 0.05
    path = tmp_path / "ep.npz"
    rec.save(path, {"seed": 1, "success": True})
    ep = np.load(path)
    assert ep["action"].shape == (36, 20) and ep["gello_cmd"].shape == (36, 16)
    assert ep["external_img"].shape == (36, 96, 128, 3)


def test_cli_teleop_mock_end_to_end(tmp_path):
    import run_real

    m = run_real.main(
        [
            "teleop-gello",
            "--backend",
            "mock",
            "--gello",
            "mock",
            "--steps",
            "48",
            "--record",
            "--label-strokes",
            "--success",
            "yes",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert not m["aborted"]
    ep = np.load(m["saved"])
    assert len(ep["action"]) == 48 and str(ep["meta_source"]) == "gello_synthetic_mock"


# ---------------------------------------------------------------------------------------------- policy path
def test_stroke_runner_with_stub_policy():
    from peel.policy_runner import StrokeSkillRunner

    env = make_env()
    pol = StubStrokePolicy(drift=0.002)
    runner = StrokeSkillRunner(pol)
    left_hold = (env.left.tgt_p.copy(), env.left.tgt_R.copy())
    x0 = env.tcp_world("right")[:3, 0].copy()
    p0 = env.tcp_world("right")[:3, 3].copy()
    steps = runner.run(env, left_hold, max_steps=16)
    assert steps == 16 and pol.calls == 4  # 24 Hz control, 6 Hz inference
    moved = env.tcp_world("right")[:3, 3] - p0
    assert moved @ x0 > 0.01  # the blade went along its edge direction (relative actions were applied)


def test_cli_policy_mock_end_to_end(tmp_path):
    import run_real

    m = run_real.main(
        [
            "policy",
            "--backend",
            "mock",
            "--policy",
            "stub",
            "--device",
            "cpu",
            "--max-strokes",
            "1",
            "--stroke-max-steps",
            "12",
            "--record",
            "--success",
            "no",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert not m["aborted"] and m["strokes"] == 1 and m["policy_calls"] == 3
    ep = np.load(m["saved"])
    assert ep["action"].shape[1] == 20


def test_cli_expert_baseline_and_scan_mock(tmp_path):
    import run_real

    m = run_real.main(
        [
            "policy",
            "--backend",
            "mock",
            "--policy",
            "expert",
            "--max-strokes",
            "1",
            "--no-images",
            "--quiet",
            "--out",
            str(tmp_path),
        ]
    )
    assert not m["aborted"] and m["strokes"] == 1 and m["policy_calls"] is None
    s = run_real.main(["scan", "--backend", "mock", "--scan-views", "3", "--quiet", "--out", str(tmp_path)])
    sc = np.load(s["saved"])
    assert sc["external_img"].shape == (3, 96, 128, 3) and sc["roll"][-1] > 5.0


# ---------------------------------------------------------------------------------------------- safety
def test_force_limit_aborts_and_latches():
    env = make_env(render_obs=False)
    env.backend.extra_wrench["right"][:3] = [0.0, 0.0, 40.0]  # a collision the arm cannot see coming
    with pytest.raises(SafetyAbort, match="N >"):
        env.step_tcp()
    assert env.backend.stopped
    with pytest.raises(SafetyAbort, match="aborted earlier"):
        env.step_tcp()
    env.backend.extra_wrench["right"][:] = 0
    env.reset()  # a deliberate reset re-arms the backend and clears the latch
    p0 = env.right.tgt_p.copy()
    o = env.step_tcp(right=(p0 + np.array([0.0, 0.0, 0.005]), env.right.tgt_R))
    assert not env.backend.stopped and o["right_robot_tcp_pose"][2] > p0[2]  # it moves again


def test_targets_clipped_to_workspace_and_speed():
    env = make_env(render_obs=False)
    s = env.config.safety
    p_prev = env.right.tgt_p.copy()
    env.step_tcp(right=(p_prev + np.array([0.0, 0.0, 0.10]), env.right.tgt_R))  # 10 cm jump in one step
    assert np.linalg.norm(env.right.tgt_p - p_prev) <= s.max_lin_speed * DT + 1e-9
    lo, hi = env.safety.box("right")
    for _ in range(3):
        env.step_tcp(right=(np.array([2.0, 2.0, 2.0]), env.right.tgt_R))  # far outside the box
    assert np.all(env.right.tgt_p <= hi + 1e-9) and np.all(env.right.tgt_p >= lo - 1e-9)
    kinds = {e[1] for e in env.safety.events}
    assert {"target_speed_limited", "target_clipped_to_workspace"} <= kinds
    R_prev = env.right.tgt_R.copy()
    env.step_tcp(right=(env.right.tgt_p, rr.rotvec_to_R([0, 0, 1.0]) @ R_prev))  # 1 rad turn in one step
    assert np.linalg.norm(rr.rotvec(env.right.tgt_R @ R_prev.T)) <= s.max_ang_speed * DT + 1e-6


def test_measured_tcp_outside_workspace_aborts():
    env = make_env(render_obs=False)
    q = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.8])  # TCP about 0.6 m high, above the right box (0.49 m)
    p = (env.config.right.W_T_B @ rr.panda_fk(q, env.config.right.F_T_TCP))[:3, 3]
    lo, hi = env.safety.box("right")
    assert np.any(p > hi + 0.05) or np.any(p < lo - 0.05)
    env.backend.move_to_joints("right", q)
    with pytest.raises(SafetyAbort, match="outside the workspace"):
        env.step_tcp()


def test_dry_run_logs_workspace_but_still_aborts_on_force():
    mon = rr.SafetyMonitor(rr.SafetyConfig(), DT)
    far, now = np.array([2.0, 2.0, 2.0]), time.monotonic()
    assert mon.check_state("left", far, np.zeros(3), np.zeros(6), now, now, "", enforce_motion=False) is None
    assert mon.events[-1][1] == "measured_outside_workspace"
    assert "outside" in mon.check_state("left", far, np.zeros(3), np.zeros(6), now, now, "", enforce_motion=True)
    w = np.array([0, 31.0, 0, 0, 0, 0])
    assert "N >" in mon.check_state("left", np.array([0.48, 0.06, 0.3]), np.zeros(3), w, now, now, "", False)
    assert "old" in mon.check_state(
        "left", np.array([0.48, 0.06, 0.3]), np.zeros(3), np.zeros(6), now - 1, now, "", False
    )


def test_heartbeat_abort_between_steps():
    cfg = RealConfig()
    cfg.safety.heartbeat_abort = 0.15
    env = RealBimanualFranka(cfg, backend="mock", render_obs=False)
    env.reset()
    env.step_tcp()
    time.sleep(0.25)
    with pytest.raises(SafetyAbort, match="heartbeat"):
        env.step_tcp()


def test_backend_watchdog_holds_pose():
    cfg = RealConfig()
    cfg.safety.heartbeat_hold = 0.05
    env = RealBimanualFranka(cfg, backend="mock", render_obs=False)
    env.reset()
    env.step_tcp(right=(env.right.tgt_p + np.array([0.0, 0.0, 0.01]), env.right.tgt_R))
    time.sleep(0.1)  # the "main loop" stalls
    env.idle(0.5)
    assert any(e[1] == "heartbeat_hold" for e in env.safety.events)
    arm = env.backend.arms["right"]
    assert np.allclose(arm.tgt[:3, 3], arm.T()[:3, 3], atol=1e-3)  # holding where it was, not chasing the target


# ---------------------------------------------------------------------------------------------- helpers
def test_ft_gravity_fit_and_compensation():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(3)
    bias = np.array([1.0, -2.0, 0.5, 0.05, -0.02, 0.01])
    mass, com, sign = 0.45, np.array([0.01, -0.005, 0.07]), -1.0
    Rs = Rotation.random(12, random_state=4).as_matrix()

    def raw_of(R, f_env_s=None):
        f_env_s = np.zeros(3) if f_env_s is None else f_env_s
        g = R.T @ rr.GRAVITY
        f = mass * g + f_env_s  # force of the tool on the sensor
        return bias + sign * np.concatenate([f, np.cross(com, mass * g)])

    raws = np.array([raw_of(R) + rng.normal(0, 1e-3, 6) for R in Rs])
    c = rr.fit_ft_gravity(raws, Rs)
    assert c["sign"] == sign and abs(abs(c["mass"]) - mass) < 1e-3 and np.allclose(c["com"], com, atol=1e-3)

    class FakeDriver:
        def __init__(self, raw):
            self.raw = raw

        def read_frame(self):
            return type("F", (), {"force": self.raw[:3], "torque": self.raw[3:]})()

    ft = BotaFTSensor("unused.json", calib=c)
    O_T_flange = rr.make_T([0.4, 0.0, 0.3], Rs[0])
    f_env_base = np.array([0.0, 0.0, 8.0])  # environment pushes the tool up
    ft.driver = FakeDriver(raw_of(Rs[0], Rs[0].T @ f_env_base))
    w = ft.wrench_sim(O_T_flange)
    assert np.allclose(w[:3], f_env_base, atol=0.02)


def test_pivot_calibration_and_kabsch():
    from scipy.spatial.transform import Rotation

    tip, c = np.array([0.01, -0.02, 0.205]), np.array([0.5, -0.1, 0.3])
    poses = []
    for R in Rotation.random(6, random_state=5).as_matrix():
        poses.append(rr.make_T(c - R @ tip, R))
    out = rr.pivot_calibration(poses)
    assert np.allclose(out["tip_in_flange"], tip, atol=1e-9) and out["rms_m"] < 1e-9
    T_true = rr.make_T([0.02, -0.51, 0.003], Rotation.from_euler("z", 2.0, degrees=True).as_matrix())
    P = np.random.default_rng(6).uniform(-0.3, 0.3, (5, 3))
    Q = (T_true[:3, :3] @ P.T).T + T_true[:3, 3]
    assert np.allclose(rr.kabsch(P, Q), T_true, atol=1e-9)


def test_config_json_roundtrip(tmp_path):
    cfg = RealConfig()
    cfg.left.ip, cfg.right.ip = "172.16.0.2", "172.16.1.2"
    cfg.safety.max_force = 25.0
    cfg.cameras["right_wrist_img"].serial = "123"
    cfg.to_json(tmp_path / "c.json")
    back = RealConfig.from_json(tmp_path / "c.json")
    assert back.left.ip == "172.16.0.2" and back.safety.max_force == 25.0
    assert back.cameras["right_wrist_img"].serial == "123"
    assert np.allclose(back.right.F_T_TCP, cfg.right.F_T_TCP) and not back.right.has_gripper


def test_resize_rgb_block_mean():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:240] = 200
    out = rr.resize_rgb(img)
    assert out.shape == (96, 128, 3) and out.dtype == np.uint8
    assert out[:48].min() == 200 and out[48:].max() == 0


def test_label_strokes_from_force():
    import run_real

    rec = EpisodeRecorder()
    fn = np.zeros(200)
    fn[50:90] = 8.0
    fn[130:170] = 9.0
    for f in fn:
        rec.add({"blade_fn": np.array([f], np.float32)}, np.zeros(20), "policy", 0, None)
    assert run_real.label_strokes_from_force(rec) == 2
    sid = np.array([s[0] for s in rec.buf["stroke_id"]])
    assert set(sid[50:90]) == {0} and set(sid[130:170]) == {1} and sid[0] == -1


# ---------------------------------------------------------------------------------------------- franky contract
def test_franky_worker_contract_with_fake_robot():
    """Runs the real _franky_worker_main against the real franky package, with only Robot and Gripper faked.
    Checks that every franky call the worker makes exists and accepts our arguments. Skipped if franky is not
    installed. It does NOT test real-time behaviour, libfranka, or the robot."""
    franky = pytest.importorskip("franky")
    import multiprocessing as mp
    import threading

    class FakeState:
        def __init__(self, robot):
            self.robot = robot

        def __getattr__(self, k):
            r = self.robot
            vals = {
                "F_T_NE": franky.Affine(rr.make_T([0, 0, 0.1034], rr.rot_z(-np.pi / 4))),
                "EE_T_K": franky.Affine(np.eye(4)),
                "q": r.q.copy(),
                "dq": np.zeros(7),
                "O_F_ext_hat_K": np.array([0, 0, -5.0, 0, 0, 0]),
            }
            if k == "F_T_EE":
                return franky.Affine(np.asarray(vals["F_T_NE"].matrix) @ r.NE_T_EE)
            if k == "O_T_EE":
                return franky.Affine(r.O_T_EE)
            return vals[k]

    robots: list = []

    class FakeRobot:
        def __init__(self, ip, **kw):
            assert isinstance(kw["realtime_config"], franky.RealtimeConfig)
            self.q = PeelEnv.Q_RIGHT_SEED.copy()
            self.NE_T_EE = np.eye(4)
            self.O_T_EE = rr.make_T([0.45, 0.25, 0.38])
            self.motion = None
            self.moves = []
            robots.append(self)

        state = property(lambda self: FakeState(self))
        is_in_control = property(lambda self: self.motion is not None)
        current_cartesian_velocity = property(
            lambda self: franky.RobotVelocity(franky.Twist([0.0, 0.0, 0.01], [0.0, 0.0, 0.0]))
        )

        has_errors = False

        def recover_from_errors(self):
            return True

        def set_ee(self, v):
            assert len(v) == 16
            self.NE_T_EE = np.asarray(v, float).reshape(4, 4, order="F")

        def set_load(self, m, com, inertia):
            assert len(com) == 3 and len(inertia) == 9

        def move(self, motion, asynchronous=False):
            self.moves.append(type(motion).__name__)
            if isinstance(motion, franky.CartesianImpedanceTrackingMotion):
                self.motion = motion
            elif isinstance(motion, franky.TorqueStopMotion):
                self.motion = None

        def join_motion(self, timeout=None):
            return True

        def poll_motion(self):
            return False

    class FakeGripper:
        def __init__(self, ip):
            self.width = 0.08

        def grasp(self, w, speed, force, epsilon_inner=0.005, epsilon_outer=0.005):
            self.width = max(w, 0.04)
            return True

        def move(self, w, speed):
            self.width = w
            return True

    real_robot_cls, real_gripper_cls = franky.Robot, franky.Gripper
    franky.Robot, franky.Gripper = FakeRobot, FakeGripper
    try:
        cfg = RealConfig()
        params = {
            "worker_hz": 250.0,
            "control_hz": 24,
            "max_force": 30.0,
            "heartbeat_hold": 0.2,
            "heartbeat_stop": 1.0,
            "reset_dynamics_factor": 0.1,
        }
        parent, child = mp.Pipe()
        th = threading.Thread(
            target=rr._franky_worker_main, args=(rr.asdict(cfg.left), False, params, child), daemon=True
        )
        th.start()
        kind, info = parent.recv()
        assert kind == "ready", info
        assert np.allclose(info["F_T_EE"], cfg.left.F_T_TCP, atol=1e-9)  # set_ee puts the TCP where we asked
        parent.send(("rearm",))
        assert parent.recv() == ("ok", "recovered")
        parent.send(("move_joints", PeelEnv.Q_LEFT_SEED.tolist()))
        assert parent.recv()[0] == "ok"
        parent.send(("start",))
        assert parent.recv() == ("ok", "streaming")
        rob = robots[-1]
        goal = rr.make_T([0.46, 0.25, 0.37])
        parent.send(("target", goal))
        parent.send(("gripper", "grasp", 0.04, 20.0))
        time.sleep(0.15)  # longer than one 24 Hz period: interpolation reaches the goal
        ref = np.asarray(rob.motion.get_reference().target.matrix)
        assert np.allclose(ref, goal, atol=1e-6)
        parent.send(("state",))
        kind, st = parent.recv()
        assert kind == "state" and np.allclose(st["wrench"][:3], [0, 0, 5.0])  # sign flipped to env-on-tool
        time.sleep(0.3)  # no new target for > heartbeat_hold: the worker holds the measured pose
        parent.send(("state",))
        assert "heartbeat_hold" in parent.recv()[1]["events"]
        assert np.allclose(np.asarray(rob.motion.get_reference().target.matrix), rob.O_T_EE, atol=1e-6)
        parent.send(("stop",))
        assert parent.recv() == ("ok", "stopped")
        parent.send(("close",))
        th.join(timeout=5)
        assert rob.moves == ["JointMotion", "CartesianImpedanceTrackingMotion", "TorqueStopMotion"]
    finally:
        franky.Robot, franky.Gripper = real_robot_cls, real_gripper_cls
