"""Real-robot layer: two Franka Emika Panda (FER) arms behind the same interface as peel.env.PeelEnv.

STATUS: written against current API docs and offline object construction only. Nothing in the "franky" backend,
the RealSense cameras or the Bota sensor hook has run on hardware. Only the "mock" backend is tested
(tests/test_real_mock.py). Treat every real-hardware path as untested until a supervised dry run passes.

Interface (same as PeelEnv, so peel/policy_runner.py, peel/expert.py and peel/recorder.py work unchanged):
  obs(), step_tcp(left=(p, R), right=(p, R), grip=(wl, wr)), step_action(a20), command_joint_state(q16),
  get_joint_state(), num_dofs(), get_observations(), reset(), metrics(), close()
  plus the geometry helpers the orchestrator and the stroke runner read (cucumber_frame, surface_top,
  cucumber_point, left_R, park_pose, cfg.cucumber, shape, skin, roll_max, data.site_xpos[blade_site]).
  On the robot the cucumber geometry comes from caliper measurements and the grasp (CucumberFixture), not from
  ground truth, and the skin map is an ESTIMATE from the sim's own skin rule fed with measured force and pose.

Frames and conventions (match the sim exactly):
  * world frame = sim world: left base at (0, +0.25, 0), right base at (0, -0.25, 0), both facing +x
    (peel/scene.py BASE_Y). Each real base pose is a calibrated 4x4 world_T_base in the config.
  * TCP pose obs: xyz + 6D rotation (first two columns of R), world frame.
  * tcp_vel: [linear, angular], world axes.
  * tcp_wrench: force and torque applied BY the environment ON the tool, torque about the flange, world axes,
    tool gravity removed. libfranka's O_F_ext_hat_K has the OPPOSITE sign ("Forces applied by the robot to the
    environment are positive", RobotState docstring in franky 2.0.0 / libfranka) and its torque is about the
    stiffness frame K, so franka_wrench_to_sim() negates it and moves the torque to the flange.

Real backend choice: franky (pip franky-control), https://github.com/TimSchneider42/franky
  * franky 2.0.0 (PyPI, 2026-08-24) ships wheels built against libfranka 0.9.2, the last libfranka for the Panda
    (https://pypi.org/project/panda-python/), for CPython 3.9 to 3.14: install with
    pip install franky-control --extra-index-url https://timschneider42.github.io/franky/whl/libfranka-0.9.2/
    (https://timschneider42.github.io/franky/getting-started/installation.html; "franky supports Robot Server
    versions 2 through 10"). Our project venv is Python 3.10, so the policy and the robot client share one env.
  * CartesianImpedanceTrackingMotion keeps one torque-level Cartesian impedance controller alive and "reads the
    latest reference ... every control cycle" (set_reference), with per-axis stiffness/damping
    (CartesianImpedanceGains.diagonal), translational/rotational error clipping (same idea as the sim's
    5 cm / 0.4 rad clip), a nullspace PostureTask, max_delta_tau and force Reactions
    (https://timschneider42.github.io/franky/guide/impedance-control.html). Any target rate works; we stream
    24 Hz targets, linearly interpolated at ~250 Hz in the arm worker.
  * External wrench: RobotState.O_F_ext_hat_K, read through robot.state.
  * The API above was checked against the franky 2.0.0+libfranka.0.9.2 wheel's stubs, and the motion, gains,
    reference and reaction objects used here were constructed in a scratch venv. Robot() itself needs a robot.
  Alternative (documented in docs/REAL_ROBOT.md, not implemented): Polymetis (facebookresearch/fairo), used by
  GELLO (third_party/gello/gello/robots/panda.py) and RDP (third_party/rdp/.../franka_server.py:
  start_cartesian_impedance + update_desired_ee_pose, so it does take a runtime target). Not chosen because its
  Python client needs Python 3.8 via conda (third_party/rdp/docs/franka_setup_instructions.md) while this project
  runs on 3.10, and RDP issue #31 (https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/31) reports a
  Python 3.8 vs ROS 2 3.10 conflict.

Two arms = two FCI connections. Each arm runs in its own process (FrankyArmProxy -> _franky_worker_main), so each
libfranka control thread gets its own real-time process and a crash or a Python stall in one cannot block the other.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .env import CONTROL_HZ, GRIP_MAX, PeelEnv, rot6d, rot6d_to_mat
from .expert import blade_frame
from .scene import BASE_Y, BLADE_Z, HAND_TCP_Z, CucumberSpec
from .shape import CucumberShape
from .skin import SkinGrid, SkinParams


class SafetyAbort(RuntimeError):
    """Raised after both arms were told to stop. The env is unusable until reset()."""


# ============================================================================================== Panda kinematics
# Modified (Craig) DH parameters (a, d, alpha) per joint and the 0.107 m flange, from the Franka docs
# (https://frankarobotics.github.io/docs/robot_specifications.html, listed there for FR3). tests/test_real_mock.py
# checks this FK against the MuJoCo Menagerie Panda used by the sim (agreement ~1e-15 m).
PANDA_DH = np.array(
    [
        [0.0, 0.333, 0.0],
        [0.0, 0.0, -np.pi / 2],
        [0.0, 0.316, np.pi / 2],
        [0.0825, 0.0, np.pi / 2],
        [-0.0825, 0.384, -np.pi / 2],
        [0.0, 0.0, np.pi / 2],
        [0.088, 0.0, np.pi / 2],
    ]
)
FLANGE_D = 0.107
# FER (Panda) joint limits from the same Franka page (identical to the Menagerie model's ranges)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def make_T(p=None, R=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform."""
    R, p = T[:3, :3], T[:3, 3]
    return make_T(-R.T @ p, R.T)


def cross3(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector (log map). Closed form away from pi, scipy near pi."""
    c = np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * w
    if ang < np.pi - 1e-3:
        return w * (ang / (2.0 * np.sin(ang)))
    return Rotation.from_matrix(R).as_rotvec()


def rotvec_to_R(v: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(v).as_matrix()


def _dh(a, d, alpha, theta) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1.0]])


def panda_frames(q) -> list[np.ndarray]:
    """Joint frames 1..7 (z = joint axis) and the flange frame, all in the robot base frame."""
    T = np.eye(4)
    out = []
    for (a, d, al), th in zip(PANDA_DH, q):
        T = T @ _dh(a, d, al, th)
        out.append(T)
    out.append(T @ make_T([0.0, 0.0, FLANGE_D]))
    return out


def panda_fk(q, F_T_tcp: np.ndarray | None = None) -> np.ndarray:
    """Base_T_tcp (or base_T_flange without a tool transform)."""
    T = panda_frames(q)[-1]
    return T if F_T_tcp is None else T @ F_T_tcp


def panda_jacobian(q, F_T_tcp: np.ndarray) -> np.ndarray:
    """Geometric 6x7 Jacobian [linear; angular] of the TCP, base frame."""
    fr = panda_frames(q)
    p = (fr[-1] @ F_T_tcp)[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z, o = fr[i][:3, 2], fr[i][:3, 3]
        J[:3, i] = cross3(z, p - o)
        J[3:, i] = z
    return J


def panda_ik(
    T_tgt: np.ndarray, q_seed, F_T_tcp: np.ndarray, iters: int = 300, tol: float = 2e-4, free_roll: bool = False
):
    """Damped least-squares IK in the base frame; the same scheme as peel.env.Arm.ik (pull toward the seed)."""
    q_seed = np.asarray(q_seed, float)
    q = q_seed.copy()
    p_t, R_t = T_tgt[:3, 3], T_tgt[:3, :3]
    e = np.zeros(6)
    for _ in range(iters):
        T = panda_fk(q, F_T_tcp)
        p, R = T[:3, 3], T[:3, :3]
        rot_e = np.cross(R[:, 2], R_t[:, 2]) if free_roll else rotvec(R_t @ R.T)
        e = np.concatenate([p_t - p, rot_e])
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 10 * tol:
            break
        J = panda_jacobian(q, F_T_tcp)
        if free_roll:
            z = R[:, 2]
            J[3:] = J[3:] - np.outer(z, z) @ J[3:]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
        dq += 0.02 * (np.eye(7) - np.linalg.pinv(J) @ J) @ (q_seed - q)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), Q_MIN + 1e-3, Q_MAX - 1e-3)
    return q, e


# ============================================================================================== tool transforms
# Franka Hand mounting on the flange: rotated -45 deg about the flange z (Menagerie hand quat 0.92388 0 0 -0.38268).
F_T_HAND = make_T(R=rot_z(-np.pi / 4))


def default_F_T_tcp(side: str) -> np.ndarray:
    """Flange-to-TCP transform of the SIM design (checked against MuJoCo by sim_tool_transforms() in the tests).

    left : Franka Hand fingertip centre, 0.1034 m along the hand z axis.
    right: blade edge of the peeler, 0.205 m out along the hand z axis, tilted as in peel/scene.py _peeler_xml().
    On the robot, MEASURE the real offset of your adapter and put it in the config (docs/REAL_ROBOT.md)."""
    if side == "left":
        return F_T_HAND @ make_T([0.0, 0.0, HAND_TCP_Z])
    g, a = np.deg2rad(50.0), np.deg2rad(15.0)  # same constants as peel/scene.py _peeler_xml()
    Rh = np.column_stack([[-np.sin(g), 0, -np.cos(g)], [0, 1, 0], [np.cos(g), 0, -np.sin(g)]])
    Rb = np.column_stack([[-np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, -np.cos(a)]])
    return F_T_HAND @ make_T([0.0, 0.0, BLADE_Z], Rh.T @ Rb)


def default_world_T_base(side: str) -> np.ndarray:
    return make_T([0.0, BASE_Y if side == "left" else -BASE_Y, 0.0])


def sim_tool_transforms() -> dict:
    """Flange-to-TCP of both arms read from the compiled MuJoCo scene (needs mujoco; used by tests and docs)."""
    import mujoco

    from .scene import SceneSpec, build_model

    m = build_model(SceneSpec())
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    out = {}
    for side, site in (("left", "left_tcp"), ("right", "blade")):
        b = m.body(f"{side}_link7").id
        W_F = make_T(d.xpos[b], d.xmat[b].reshape(3, 3)) @ make_T([0.0, 0.0, FLANGE_D])
        s = m.site(site).id
        out[side] = inv_T(W_F) @ make_T(d.site_xpos[s], d.site_xmat[s].reshape(3, 3))
    return out


# ============================================================================================== wrench conventions
def franka_wrench_to_sim(O_F_ext_hat_K, O_T_EE, F_T_EE, EE_T_K=None) -> np.ndarray:
    """libfranka external wrench -> sim convention (base axes).

    libfranka O_F_ext_hat_K: wrench of the ROBOT on the environment, at the stiffness frame K, base axes.
    sim: wrench of the ENVIRONMENT on the tool, torque about the flange, base axes."""
    w = np.asarray(O_F_ext_hat_K, float)
    O_T_EE = np.asarray(O_T_EE, float)
    O_T_K = O_T_EE if EE_T_K is None else O_T_EE @ np.asarray(EE_T_K, float)
    O_T_F = O_T_EE @ inv_T(np.asarray(F_T_EE, float))
    f_rob, tau_rob_K = w[:3], w[3:]
    tau_rob_F = tau_rob_K + np.cross(O_T_K[:3, 3] - O_T_F[:3, 3], f_rob)
    return -np.concatenate([f_rob, tau_rob_F])


GRAVITY = np.array([0.0, 0.0, -9.81])


def fit_ft_gravity(raw: np.ndarray, R_base_sensor: np.ndarray) -> dict:
    """Least-squares bias + tool mass/CoM from N static poses (raw (N, 6) readings, R_base_sensor (N, 3, 3)).

    Model: raw_f = b_f + m_s * g_s ;  raw_t = b_t + (m_s c) x g_s   with g_s = R^T g  (sensor axes).
    m_s comes out NEGATIVE if the sensor reports the opposite sign convention; BotaFTSensor uses sign(m_s).
    Needs >= 3 clearly different orientations (pure numpy, tested with synthetic data)."""
    raw = np.asarray(raw, float)
    Af, bf, At, bt = [], [], [], []
    for w, R in zip(raw, np.asarray(R_base_sensor, float)):
        g = R.T @ GRAVITY
        Af.append(np.hstack([np.eye(3), g[:, None]]))
        bf.append(w[:3])
        gx = np.array([[0, -g[2], g[1]], [g[2], 0, -g[0]], [-g[1], g[0], 0]])
        At.append(np.hstack([np.eye(3), -gx]))  # (m c) x g = -g x (m c)
        bt.append(w[3:])
    xf = np.linalg.lstsq(np.vstack(Af), np.concatenate(bf), rcond=None)[0]
    xt = np.linalg.lstsq(np.vstack(At), np.concatenate(bt), rcond=None)[0]
    m_s = float(xf[3])
    return {
        "bias": np.concatenate([xf[:3], xt[:3]]),
        "mass": m_s,
        "com": xt[3:] / m_s if abs(m_s) > 1e-6 else np.zeros(3),
        "sign": 1.0 if m_s >= 0 else -1.0,
    }


# ============================================================================================== calibration helpers
def pivot_calibration(base_T_flange_list) -> dict:
    """Tool-tip offset from >= 4 flange poses that all touch one fixed point with the blade edge centre.

    Solves R_i t + p_i = c for t (tip in the flange frame) and c (the point in the base frame), least squares.
    The rotation of the blade frame cannot be found this way; take it from the adapter CAD."""
    A, b = [], []
    for T in base_T_flange_list:
        T = np.asarray(T, float)
        A.append(np.hstack([T[:3, :3], -np.eye(3)]))
        b.append(-T[:3, 3])
    x, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    res = np.vstack(A) @ x - np.concatenate(b)
    return {"tip_in_flange": x[:3], "point_in_base": x[3:], "rms_m": float(np.sqrt(np.mean(res**2)))}


def kabsch(P, Q) -> np.ndarray:
    """Rigid 4x4 T with Q ~ T @ P for matched 3-D points P, Q (N >= 3, not collinear).

    Use: touch the same N points with both TCPs; P in the right base frame, Q in the left base frame gives
    left_T_right; then world_T_right = world_T_left @ left_T_right."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(cq - R @ cp, R)


# ============================================================================================== configuration
def _arr(x, shape=None):
    a = np.asarray(x, float)
    return a.reshape(shape) if shape else a


@dataclass
class ArmConfig:
    side: str = "left"
    ip: str = ""
    world_T_base: list | None = None  # 4x4, calibrated (docs/REAL_ROBOT.md "Calibration")
    F_T_tcp: list | None = None  # 4x4 flange -> TCP, measured
    stiffness: list | None = None  # [x, y, z, rx, ry, rz] N/m, Nm/rad (defaults = sim gains in peel/env.py)
    effective_mass: tuple = (2.5, 2.5, 2.5, 0.06, 0.06, 0.06)  # damping = 2 sqrt(K m), as in peel/env.py Arm
    error_clip: tuple = (0.05, 0.4)  # m, rad (the sim clips the pose error the same way)
    posture_stiffness: float = 8.0  # nullspace, Nm/rad (sim k_null)
    max_delta_tau: float = 1.0  # Nm per 1 kHz cycle (franky default)
    has_gripper: bool = True
    fixed_gripper_width: float = 0.018  # reported when has_gripper is False (sim right-arm value)
    grasp_force: float = 20.0  # N, our default for holding the cucumber; tune on the real fruit
    gripper_speed: float = 0.05  # m/s
    load_mass: float = 0.0  # kg extra load for robot.set_load (e.g. the cucumber on the left arm)
    load_com: tuple = (0.0, 0.0, 0.0)  # m, flange frame
    torque_threshold: float = 20.0  # Nm, franky Robot(default_torque_threshold) default
    force_threshold: float = 30.0  # N, franky Robot(default_force_threshold) default
    realtime: bool = True  # RealtimeConfig.Enforce (needs a PREEMPT_RT kernel)
    ft_sensor: dict | None = None  # {"type": "bota", "config_path": ..., "F_T_sensor": 4x4, "calib": {...}}

    def __post_init__(self):
        if self.world_T_base is None:
            self.world_T_base = default_world_T_base(self.side).tolist()
        if self.F_T_tcp is None:
            self.F_T_tcp = default_F_T_tcp(self.side).tolist()
        if self.stiffness is None:
            self.stiffness = [2200, 2200, 2200, 150, 150, 150] if self.side == "left" else [1100, 1100, 900, 60, 60, 60]

    @property
    def W_T_B(self) -> np.ndarray:
        return _arr(self.world_T_base, (4, 4))

    @property
    def F_T_TCP(self) -> np.ndarray:
        return _arr(self.F_T_tcp, (4, 4))

    @property
    def damping(self) -> np.ndarray:
        return 2.0 * np.sqrt(_arr(self.stiffness) * _arr(self.effective_mass))


@dataclass
class SafetyConfig:
    dry_run: bool = True  # hardware backends only: connect and read, never command motion
    # world-frame TCP boxes: min/max of 16 recorded sim episodes (data/raw) plus a 10 cm margin
    workspace: dict = field(
        default_factory=lambda: {
            "left": [[0.38, -0.05, 0.19], [0.58, 0.16, 0.40]],
            "right": [[0.36, -0.28, 0.20], [0.60, 0.16, 0.49]],
        }
    )
    workspace_abort_margin: float = 0.03  # m outside the box (measured) -> abort
    max_lin_speed: float = 0.25  # m/s, limit on the target step (sim demos reach 0.52 m/s in scripted moves)
    max_ang_speed: float = 2.0  # rad/s (sim roll runs at about 1.6 rad/s)
    abort_lin_speed: float = 0.8  # m/s measured TCP speed -> abort (sim demos peak at 0.49 m/s)
    max_force: float = 30.0  # N, |F| on either arm -> abort (sim demos peak at 26.7 N, see docs)
    heartbeat_hold: float = 0.2  # s without a new target -> the arm holds its current pose
    heartbeat_stop: float = 1.0  # s without a new target -> the arm worker stops the controller
    heartbeat_abort: float = 1.0  # s between two step calls while streaming -> abort
    state_timeout: float = 0.25  # s, stale robot state -> abort


@dataclass
class CameraConfig:
    serial: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    crop: list | None = None  # [x0, y0, x1, y1] pixels before resizing (match the sim field of view)


@dataclass
class RealConfig:
    left: ArmConfig = field(default_factory=lambda: ArmConfig(side="left"))
    right: ArmConfig = field(default_factory=lambda: ArmConfig(side="right", has_gripper=False))
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    cameras: dict = field(default_factory=lambda: {"external_img": CameraConfig(), "right_wrist_img": CameraConfig()})
    image_hw: tuple = (96, 128)
    control_hz: int = CONTROL_HZ
    worker_hz: float = 250.0  # arm-worker interpolation rate (franky backend)
    reset_dynamics_factor: float = 0.1  # franky JointMotion speed scale for reset moves

    def arm(self, side: str) -> ArmConfig:
        return self.left if side == "left" else self.right

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path) -> RealConfig:
        d = json.loads(Path(path).read_text())
        cfg = RealConfig()
        for side in ("left", "right"):
            if side in d:
                setattr(cfg, side, ArmConfig(**{"side": side, **d[side]}))
        if "safety" in d:
            cfg.safety = SafetyConfig(**d["safety"])
        if "cameras" in d:
            cfg.cameras = {k: CameraConfig(**v) for k, v in d["cameras"].items()}
        for k in ("image_hw", "control_hz", "worker_hz", "reset_dynamics_factor"):
            if k in d:
                setattr(cfg, k, tuple(d[k]) if k == "image_hw" else d[k])
        return cfg


# ============================================================================================== cucumber fixture stub
@dataclass
class CucumberFixture:
    """Perception/fixture STUB: the measured cucumber and how it sits in the left gripper.

    Measure length and the two end diameters with calipers. grasp_depth is the hand-z coordinate of the gripped
    end of the cucumber (the sim uses 0.080 m, i.e. 23 mm behind the fingertip centre at 0.1034 m). Offsets and
    tilt of the axis in the hand stay 0 unless measured. TODO(perception): estimate these from the external
    camera instead of assuming the nominal grasp."""

    length: float = 0.23
    d_grip: float = 0.040
    d_tip: float = 0.040
    bend_tip: float = 0.0
    bend_dir: float = 0.0
    grasp_depth: float = 0.080
    grasp_offset: tuple = (0.0, 0.0)
    grasp_tilt: tuple = (0.0, 0.0)
    shape_class: str = "real"

    def shape(self) -> CucumberShape:
        r_g, r_t = self.d_grip / 2, self.d_tip / 2
        return CucumberShape(
            length=self.length,
            r_mid=0.5 * (r_g + r_t),
            taper=(r_t - r_g) / (r_t + r_g),
            bend_tip=self.bend_tip,
            bend_dir=self.bend_dir,
        )

    def spec(self) -> CucumberSpec:
        return CucumberSpec(
            shape=self.shape(),
            grasp_depth=self.grasp_depth,
            grasp_offset=tuple(self.grasp_offset),
            grasp_tilt=tuple(self.grasp_tilt),
            shape_class=self.shape_class,
        )

    def hand_T_cucumber(self) -> np.ndarray:
        """Same math as peel/scene.py _cucumber_xml (cucumber body frame in the hand frame)."""
        tx, ty = self.grasp_tilt
        Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        R = Rx @ Ry
        c = np.array([self.grasp_offset[0], self.grasp_offset[1], 0.0]) + R @ np.array(
            [0, 0, self.grasp_depth + self.length / 2]
        )
        return make_T(c, R)


@dataclass
class RealEpisodeConfig:
    """Stand-in for peel.env.EpisodeConfig (the attributes the orchestrator and the runner read)."""

    cucumber: CucumberSpec
    skin: SkinParams = field(default_factory=SkinParams)
    seed: int = 0
    render_obs: bool = True
    ft_noise: float = 0.0


# ============================================================================================== backends
@dataclass
class ArmState:
    T: np.ndarray  # 4x4 TCP pose, robot base frame
    twist: np.ndarray  # (6,) [v, w], base axes
    wrench: np.ndarray  # (6,) sim convention (env on tool, about the flange), base axes
    q: np.ndarray
    dq: np.ndarray
    gripper_width: float
    stamp: float  # time.monotonic() of the reading
    fault: str = ""
    events: tuple = ()


class MockArm:
    """Pure-numpy fake arm: velocity-resolved tracking of the TCP target (time constant ~0.05 s, pose error
    clipped like the sim), joint limits, nullspace pull and a first-order gripper. Not a dynamics model."""

    def __init__(self, cfg: ArmConfig, q0):
        self.cfg = cfg
        self.F_T_tcp = cfg.F_T_TCP
        self.q = np.array(q0, float)
        self.dq = np.zeros(7)
        self.q_null = self.q.copy()
        self.tgt = panda_fk(self.q, self.F_T_tcp)
        self.grip_cmd = cfg.fixed_gripper_width if not cfg.has_gripper else GRIP_MAX
        self.grip_w = self.grip_cmd
        self.min_grip = 0.0  # object width when something is grasped
        self.last_send = time.monotonic()
        self.wrench = np.zeros(6)
        self.events: list[str] = []

    def T(self) -> np.ndarray:
        return panda_fk(self.q, self.F_T_tcp)

    def integrate(self, dt: float, n_sub: int = 5):
        h = dt / n_sub
        for _ in range(n_sub):
            T = self.T()
            ep = np.clip(self.tgt[:3, 3] - T[:3, 3], -self.cfg.error_clip[0], self.cfg.error_clip[0])
            er = np.clip(rotvec(self.tgt[:3, :3] @ T[:3, :3].T), -self.cfg.error_clip[1], self.cfg.error_clip[1])
            v = np.concatenate([ep, er]) / 0.05
            v[:3] *= min(1.0, 1.0 / (np.linalg.norm(v[:3]) + 1e-12))
            v[3:] *= min(1.0, 3.0 / (np.linalg.norm(v[3:]) + 1e-12))
            J = panda_jacobian(self.q, self.F_T_tcp)
            A = J @ J.T + 1e-4 * np.eye(6)
            z = 1.0 * (self.q_null - self.q)  # nullspace pull, projected with the damped pseudo-inverse
            dq = J.T @ np.linalg.solve(A, v) + z - J.T @ np.linalg.solve(A, J @ z)  # vector solves only (fast)
            dq = np.clip(dq, -2.0, 2.0)
            q_new = np.clip(self.q + dq * h, Q_MIN, Q_MAX)
            self.dq = (q_new - self.q) / h
            self.q = q_new
        if self.cfg.has_gripper:
            goal = max(self.grip_cmd, self.min_grip)
            self.grip_w += float(np.clip(goal - self.grip_w, -0.1 * dt, 0.1 * dt))


class MockBimanualBackend:
    """Fake bimanual backend for tests and dry runs. Includes a fake contact: the blade TCP pressing into the
    cucumber (geometry from CucumberFixture) feels a linear spring force along the surface normal, and the left
    arm feels the reaction. The spring constant is a made-up test value, not a cucumber property."""

    name = "mock"
    realtime = False

    def __init__(
        self,
        config: RealConfig,
        fixture: CucumberFixture,
        seed: int = 0,
        contact_stiffness: float = 700.0,
        force_noise: float = 0.05,
    ):
        self.config = config
        self.fixture = fixture
        self.rng = np.random.default_rng(seed)
        self.shape = fixture.shape()
        self.k_contact = contact_stiffness
        self.force_noise = force_noise
        self.arms: dict[str, MockArm] = {}
        self.streaming = False
        self.stopped = ""
        self.extra_wrench = {"left": np.zeros(6), "right": np.zeros(6)}  # tests inject disturbances here

    def connect(self):
        self.arms = {
            "left": MockArm(self.config.left, PeelEnv.Q_LEFT_SEED),
            "right": MockArm(self.config.right, PeelEnv.Q_RIGHT_SEED),
        }

    def rearm(self):
        self.stopped = ""

    def move_to_joints(self, side: str, q):
        a = self.arms[side]
        a.q = np.clip(np.array(q, float), Q_MIN, Q_MAX)
        a.q_null = a.q.copy()
        a.tgt = a.T()

    def start_streaming(self):
        for a in self.arms.values():
            a.tgt = a.T()
            a.last_send = time.monotonic()
        self.streaming = True

    def send(self, side: str, T_base: np.ndarray, grip_width: float):
        if self.stopped:
            return
        a = self.arms[side]
        a.tgt = np.array(T_base, float)
        a.last_send = time.monotonic()

    def set_posture(self, side: str, q):
        self.arms[side].q_null = np.clip(np.array(q, float), Q_MIN, Q_MAX)

    def gripper_command(self, side: str, mode: str, width: float, force: float = 0.0):
        a = self.arms[side]
        if not a.cfg.has_gripper:
            return
        a.grip_cmd = float(width)
        a.min_grip = 2 * self.shape.r_grip if (mode == "grasp" and side == "left") else 0.0

    def advance(self, dt: float):
        if self.stopped:
            return
        now = time.monotonic()
        for a in self.arms.values():
            if self.streaming and now - a.last_send > self.config.safety.heartbeat_hold:
                a.tgt = a.T()  # watchdog: hold the current pose, like the franky worker
                a.events.append("heartbeat_hold")
                a.last_send = now
            a.integrate(dt)
        self._contact()

    def _contact(self):
        cl, cr = self.config.left, self.config.right
        W_tl = cl.W_T_B @ self.arms["left"].T()
        W_tr = cr.W_T_B @ self.arms["right"].T()
        W_c = cucumber_world_T(W_tl, cl.F_T_TCP, self.fixture)
        shape = self.shape
        p_b = W_tr[:3, 3]
        local = inv_T(W_c) @ np.append(p_b, 1.0)
        z, _, r, n_loc = shape.project(local[:3])
        pen = float(shape.radius(z)) - r
        f = np.zeros(3)
        if pen > 0 and shape.z_cyl[0] <= local[2] <= shape.z_cyl[1]:
            f = self.k_contact * pen * (W_c[:3, :3] @ n_loc)
        for side, cfg, W_t, sign in (("right", cr, W_tr, 1.0), ("left", cl, W_tl, -1.0)):
            p_flange = (W_t @ inv_T(cfg.F_T_TCP))[:3, 3]
            fw = sign * f
            w_world = np.concatenate([fw, np.cross(p_b - p_flange, fw)]) + self.extra_wrench[side]
            w_world[:3] += self.rng.normal(0, self.force_noise, 3)
            Rb = cfg.W_T_B[:3, :3]
            self.arms[side].wrench = np.concatenate([Rb.T @ w_world[:3], Rb.T @ w_world[3:]])

    def read(self, side: str) -> ArmState:
        a = self.arms[side]
        J = panda_jacobian(a.q, a.F_T_tcp)
        ev = tuple(a.events)
        a.events.clear()
        return ArmState(
            T=a.T(),
            twist=J @ a.dq,
            wrench=a.wrench.copy(),
            q=a.q.copy(),
            dq=a.dq.copy(),
            gripper_width=a.grip_w if a.cfg.has_gripper else a.cfg.fixed_gripper_width,
            stamp=time.monotonic(),
            fault="",
            events=ev,
        )

    def stop(self, reason: str = ""):
        self.stopped = reason or "stopped"
        for a in self.arms.values():
            a.tgt = a.T()
            a.dq[:] = 0

    def close(self):
        self.streaming = False


# ---------------------------------------------------------------------------------------------- franky backend
def _interp_T(T0: np.ndarray, T1: np.ndarray, s: float) -> np.ndarray:
    R0 = T0[:3, :3]
    return make_T(T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3]), R0 @ rotvec_to_R(s * rotvec(R0.T @ T1[:3, :3])))


def _franky_worker_main(cfg_dict: dict, dry_run: bool, params: dict, conn) -> None:
    """One process per arm. UNTESTED ON HARDWARE. API names checked against franky 2.0.0 (libfranka 0.9.2 wheel).

    Messages in : ("state",) ("target", T_base, grip_w) ("posture", q) ("move_joints", q) ("start",) ("stop",)
                  ("gripper", mode, width, force) ("close",)
    Messages out: ("ready", info) | ("error", text) once, then replies to "state", "move_joints", "start", "stop".
    """
    import franky

    cfg = ArmConfig(**cfg_dict)
    try:
        robot = franky.Robot(
            cfg.ip,
            relative_dynamics_factor=params["reset_dynamics_factor"],
            default_torque_threshold=cfg.torque_threshold,
            default_force_threshold=cfg.force_threshold,
            realtime_config=franky.RealtimeConfig.Enforce if cfg.realtime else franky.RealtimeConfig.Ignore,
        )
        robot.recover_from_errors()
        F_T_NE = np.asarray(robot.state.F_T_NE.matrix)
        NE_T_EE = inv_T(F_T_NE) @ cfg.F_T_TCP
        robot.set_ee(NE_T_EE.flatten(order="F").tolist())  # column-major 4x4 (franky/libfranka set_ee docstring)
        if cfg.load_mass > 0:
            robot.set_load(cfg.load_mass, list(cfg.load_com), [0.0] * 9)  # point mass
        gripper = franky.Gripper(cfg.ip) if cfg.has_gripper else None
    except Exception as e:  # noqa: BLE001 - report any connection error to the parent
        conn.send(("error", repr(e)))
        return

    grip = {"width": cfg.fixed_gripper_width, "queue": [], "lock": threading.Lock(), "run": True}

    def gripper_loop():  # blocking gripper calls stay out of the target loop
        while grip["run"]:
            cmd = None
            with grip["lock"]:
                if grip["queue"]:
                    cmd = grip["queue"].pop(0)
            try:
                if cmd is not None and not dry_run:
                    mode, w, force = cmd
                    if mode == "grasp":
                        gripper.grasp(w, cfg.gripper_speed, force, epsilon_inner=0.005, epsilon_outer=0.08)
                    else:
                        gripper.move(w, cfg.gripper_speed)
                grip["width"] = float(gripper.width)
            except Exception as e:  # noqa: BLE001
                grip["error"] = repr(e)
            time.sleep(0.05)

    if gripper is not None:
        threading.Thread(target=gripper_loop, daemon=True).start()

    def current_T() -> np.ndarray:
        return np.asarray(robot.state.O_T_EE.matrix)

    def stop_motion():
        if robot.is_in_control:
            robot.move(franky.TorqueStopMotion(), asynchronous=True)
        try:
            robot.join_motion(timeout=3.0)
        except franky.ControlException as e:
            if "preempted" not in str(e):
                raise

    conn.send(
        (
            "ready",
            {"F_T_NE": F_T_NE.tolist(), "F_T_EE": np.asarray(robot.state.F_T_EE.matrix).tolist(), "dry_run": dry_run},
        )
    )
    motion = None
    streaming = False
    T_prev = T_goal = current_T()
    t_goal = last_target = time.monotonic()
    held = False
    fault = ""
    events: list[str] = []
    period = 1.0 / params["worker_hz"]
    interp = 1.0 / params["control_hz"]
    try:
        while True:
            if conn.poll(period if streaming else 0.05):
                msg = conn.recv()
                op = msg[0]
                if op == "state":
                    s = robot.state
                    v = robot.current_cartesian_velocity.end_effector_twist
                    conn.send(
                        (
                            "state",
                            {
                                "T": np.asarray(s.O_T_EE.matrix),
                                "q": np.asarray(s.q),
                                "dq": np.asarray(s.dq),
                                "twist": np.concatenate([np.asarray(v.linear), np.asarray(v.angular)]),
                                "wrench": franka_wrench_to_sim(
                                    np.asarray(s.O_F_ext_hat_K),
                                    np.asarray(s.O_T_EE.matrix),
                                    np.asarray(s.F_T_EE.matrix),
                                    np.asarray(s.EE_T_K.matrix),
                                ),
                                "gripper_width": grip["width"],
                                "fault": fault or grip.get("error", ""),
                                "events": list(events),
                                "stamp": time.monotonic(),
                            },
                        )
                    )
                    events.clear()
                elif op == "target":
                    T_prev = _interp_T(T_prev, T_goal, min(1.0, (time.monotonic() - t_goal) / interp))
                    T_goal, t_goal = np.asarray(msg[1], float), time.monotonic()
                    last_target, held = t_goal, False
                elif op == "rearm":  # deliberate restart after a fault (env.reset): libfranka error recovery
                    ok = bool(robot.recover_from_errors()) if robot.has_errors else True
                    fault = "" if ok else fault
                    conn.send(("ok", "recovered" if ok else "recovery failed"))
                elif op == "move_joints":
                    if not dry_run:
                        robot.move(
                            franky.JointMotion(
                                np.asarray(msg[1], float), relative_dynamics_factor=params["reset_dynamics_factor"]
                            )
                        )
                    T_prev = T_goal = current_T()
                    conn.send(("ok", "dry_run" if dry_run else "moved"))
                elif op == "start":
                    T_prev = T_goal = current_T()
                    t_goal = last_target = time.monotonic()
                    if not dry_run:
                        motion = franky.CartesianImpedanceTrackingMotion(
                            posture_task=franky.PostureTask(np.asarray(robot.state.q), stiffness=cfg.posture_stiffness),
                            max_delta_tau=cfg.max_delta_tau,
                            translational_error_clip=np.full(3, cfg.error_clip[0]),
                            rotational_error_clip=np.full(3, cfg.error_clip[1]),
                        )
                        motion.set_gains(
                            franky.CartesianImpedanceGains.diagonal(np.asarray(cfg.stiffness, float), cfg.damping)
                        )
                        fx, fy, fz = franky.Measure.FORCE_X, franky.Measure.FORCE_Y, franky.Measure.FORCE_Z
                        motion.add_reaction(
                            franky.Reaction(
                                fx * fx + fy * fy + fz * fz > params["max_force"] ** 2, franky.TorqueStopMotion()
                            )
                        )
                        motion.set_reference(franky.CartesianReference(target=franky.Affine(T_goal)))
                        robot.move(motion, asynchronous=True)
                        streaming = True
                    conn.send(("ok", "dry_run" if dry_run else "streaming"))
                elif op == "posture":
                    pass  # franky 2.0.0 fixes the PostureTask target when the motion is built (see "start")
                elif op == "gripper":
                    if gripper is not None:
                        with grip["lock"]:
                            grip["queue"] = [(msg[1], float(msg[2]), float(msg[3]))]
                elif op == "stop":
                    if streaming:
                        try:
                            stop_motion()
                        except Exception as e:  # noqa: BLE001 - a stored reflex error must not kill the worker
                            fault = repr(e)
                    streaming = False
                    conn.send(("ok", "stopped"))
                elif op == "close":
                    break
            if not streaming:
                continue
            if not robot.is_in_control:  # reflex, reaction (force limit) or libfranka error
                streaming = False
                try:
                    robot.poll_motion()
                    fault = "controller stopped (force reaction or reflex)"
                except Exception as e:  # noqa: BLE001
                    fault = repr(e)
                continue
            now = time.monotonic()
            if now - last_target > params["heartbeat_stop"]:
                with contextlib.suppress(Exception):
                    stop_motion()
                streaming, fault = False, "heartbeat: no target for too long, controller stopped"
                continue
            if now - last_target > params["heartbeat_hold"] and not held:
                T_prev = T_goal = current_T()
                held = True
                events.append("heartbeat_hold")
            T_cmd = _interp_T(T_prev, T_goal, min(1.0, (now - t_goal) / interp))
            motion.set_reference(franky.CartesianReference(target=franky.Affine(T_cmd)))
    finally:
        grip["run"] = False
        if streaming:
            with contextlib.suppress(Exception):
                stop_motion()


class FrankyArmProxy:
    """Parent-side handle of one arm worker process."""

    def __init__(self, cfg: ArmConfig, dry_run: bool, params: dict):
        ctx = mp.get_context("spawn")  # never fork a process that may hold CUDA/torch state
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_franky_worker_main,
            args=(asdict(cfg), dry_run, params, child),
            daemon=True,
            name=f"franky-{cfg.side}",
        )
        self.side = cfg.side
        self.info: dict = {}

    def start(self, timeout: float = 30.0):
        self.proc.start()
        if not self.conn.poll(timeout):
            raise RuntimeError(f"{self.side} arm worker did not answer within {timeout} s")
        kind, payload = self.conn.recv()
        if kind != "ready":
            raise RuntimeError(f"{self.side} arm worker failed: {payload}")
        self.info = payload

    def call(self, *msg, timeout: float = 120.0):
        self.conn.send(msg)
        if not self.conn.poll(timeout):
            raise RuntimeError(f"{self.side} arm worker timed out on {msg[0]}")
        return self.conn.recv()

    def send(self, *msg):
        self.conn.send(msg)


class FrankyBimanualBackend:
    name = "franky"
    realtime = True

    def __init__(self, config: RealConfig, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        s = config.safety
        params = {
            "worker_hz": config.worker_hz,
            "control_hz": config.control_hz,
            "max_force": s.max_force,
            "heartbeat_hold": s.heartbeat_hold,
            "heartbeat_stop": s.heartbeat_stop,
            "reset_dynamics_factor": config.reset_dynamics_factor,
        }
        self.arms = {side: FrankyArmProxy(config.arm(side), dry_run, params) for side in ("left", "right")}

    def connect(self):
        for a in self.arms.values():
            if not a.proc.is_alive():
                a.start()

    def rearm(self):
        for side, a in self.arms.items():
            _kind, msg = a.call("rearm", timeout=10.0)
            if msg != "recovered":
                raise RuntimeError(f"{side} arm: {msg}; clear the error in Desk")

    def move_to_joints(self, side: str, q):
        self.arms[side].call("move_joints", np.asarray(q, float).tolist())

    def start_streaming(self):
        for a in self.arms.values():
            a.call("start")

    def send(self, side: str, T_base: np.ndarray, grip_width: float):
        self.arms[side].send("target", np.asarray(T_base, float))

    def set_posture(self, side: str, q):
        pass  # franky 2.0.0: the PostureTask target is fixed when the tracking motion is built ("start")

    def gripper_command(self, side: str, mode: str, width: float, force: float = 0.0):
        if self.config.arm(side).has_gripper:
            self.arms[side].send("gripper", mode, float(width), float(force))

    def advance(self, dt: float):
        pass  # the arms run in real time in their workers; the env keeps the 24 Hz rate

    def read(self, side: str) -> ArmState:
        _kind, s = self.arms[side].call("state", timeout=1.0)
        cfg = self.config.arm(side)
        width = s["gripper_width"] if cfg.has_gripper else cfg.fixed_gripper_width
        return ArmState(
            T=s["T"],
            twist=s["twist"],
            wrench=s["wrench"],
            q=s["q"],
            dq=s["dq"],
            gripper_width=width,
            stamp=s["stamp"],
            fault=s["fault"],
            events=tuple(s["events"]),
        )

    def stop(self, reason: str = ""):
        for a in self.arms.values():  # stopping must try both arms even if one worker is dead
            with contextlib.suppress(Exception):
                a.call("stop", timeout=5.0)

    def close(self):
        for a in self.arms.values():
            with contextlib.suppress(Exception):
                a.send("close")
                a.proc.join(timeout=5.0)


# ============================================================================================== sensors
def resize_rgb(img: np.ndarray, out_hw=(96, 128), crop=None) -> np.ndarray:
    """Crop (x0, y0, x1, y1) then area-resize to out_hw, uint8 RGB. 640x480 -> 128x96 is an exact 5x5 block mean."""
    if crop is not None:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    H, W = out_hw
    if h % H == 0 and w % W == 0:
        return img.reshape(H, h // H, W, w // W, 3).mean(axis=(1, 3)).round().astype(np.uint8)
    import cv2

    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


class RealSenseCamera:
    """Intel RealSense colour stream via pyrealsense2 (D435/D455 external, D405 wrist). UNTESTED ON HARDWARE.
    API checked against pyrealsense2 2.59 (pipeline/config.enable_device/enable_stream/poll_for_frames).
    Check the supported colour profiles of your unit with `rs-enumerate-devices`."""

    def __init__(self, cfg: CameraConfig, out_hw=(96, 128)):
        self.cfg, self.out_hw = cfg, out_hw
        self.pipeline = None
        self.last = None
        self.last_raw = None  # full-resolution RGB frame (for the coverage scan)

    def start(self):
        import pyrealsense2 as rs

        if not self.cfg.serial:
            raise ValueError("camera serial missing in the config (see `rs-enumerate-devices`)")
        self.pipeline = rs.pipeline()
        c = rs.config()
        c.enable_device(self.cfg.serial)
        c.enable_stream(rs.stream.color, self.cfg.width, self.cfg.height, rs.format.rgb8, self.cfg.fps)
        self.pipeline.start(c)
        self._grab(self.pipeline.wait_for_frames(5000))

    def _grab(self, frames):
        if frames:
            color = frames.get_color_frame()
            if color:
                self.last_raw = np.asanyarray(color.get_data()).copy()
                self.last = resize_rgb(self.last_raw, self.out_hw, self.cfg.crop)

    def read(self) -> np.ndarray:
        self._grab(self.pipeline.poll_for_frames())  # non-blocking: newest frame or nothing
        return self.last

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


class MockCamera:
    """Synthetic image with the RealSense output format (uint8 RGB at the sim resolution)."""

    def __init__(self, name: str, out_hw=(96, 128)):
        self.name, self.out_hw, self.k = name, out_hw, 0

    def start(self):
        pass

    def read(self) -> np.ndarray:
        self.k += 1
        img = np.full((*self.out_hw, 3), 90, np.uint8)
        img[:, (self.k * 3) % self.out_hw[1]] = (30, 120, 30)
        return img

    def close(self):
        pass


class BotaFTSensor:
    """Bota Systems Gen A F/T sensor (SensONE) through the official Python driver (pip install bota-driver).

    API from https://code.botasys.com/en/gen_a/layer1/driver/driver_py.html and the bota_driver 1.1.17 package
    docstring: BotaDriver(config_path) -> configure() -> tare() (INACTIVE state) -> activate() -> read_frame()
    (.force [N], .torque [Nm], .timestamp [us], .status). UNTESTED: no sensor was available.
    TODO(hardware): confirm the sensor frame and sign with fit_ft_gravity() on >= 3 static poses, and confirm
    the read rate and latency of read_frame() over EtherCAT.

    calib: output of fit_ft_gravity() (bias, mass, com, sign). F_T_sensor: sensor frame in the flange frame."""

    def __init__(self, config_path: str, F_T_sensor=None, calib: dict | None = None, tare_on_start: bool = False):
        self.config_path = config_path
        self.F_T_sensor = np.eye(4) if F_T_sensor is None else _arr(F_T_sensor, (4, 4))
        c = calib or {"bias": np.zeros(6), "mass": 0.0, "com": np.zeros(3), "sign": 1.0}
        self.bias, self.mass = np.asarray(c["bias"], float), float(c["mass"])
        self.com, self.sign = np.asarray(c["com"], float), float(c["sign"])
        self.tare_on_start = tare_on_start
        self.driver = None

    def start(self):
        import bota_driver

        self.driver = bota_driver.BotaDriver(self.config_path)
        if not self.driver.configure():
            raise RuntimeError("Bota driver configure() failed")
        if self.tare_on_start and not self.driver.tare():  # tare only in the INACTIVE state
            raise RuntimeError("Bota driver tare() failed")
        if not self.driver.activate():
            raise RuntimeError("Bota driver activate() failed")

    def read_raw(self) -> np.ndarray:
        fr = self.driver.read_frame()
        return np.concatenate([np.asarray(fr.force, float), np.asarray(fr.torque, float)])

    def wrench_sim(self, O_T_flange: np.ndarray) -> np.ndarray:
        """External wrench in the sim convention (env on tool, about the flange, base axes)."""
        raw = self.read_raw()
        O_T_s = O_T_flange @ self.F_T_sensor
        R = O_T_s[:3, :3]
        g = R.T @ GRAVITY
        f_s = self.sign * (raw[:3] - self.bias[:3] - self.mass * g)
        t_s = self.sign * (raw[3:] - self.bias[3:] - np.cross(self.mass * self.com, g))
        f, t = R @ f_s, R @ t_s
        t = t + np.cross(O_T_s[:3, 3] - O_T_flange[:3, 3], f)
        return np.concatenate([f, t])

    def close(self):
        if self.driver is not None:
            self.driver.deactivate()
            self.driver.shutdown()
            self.driver = None


# ============================================================================================== safety
class SafetyMonitor:
    """Target filtering (workspace box, speed) and state checks (force, workspace, speed, stale state)."""

    def __init__(self, cfg: SafetyConfig, dt: float):
        self.cfg, self.dt = cfg, dt
        self.events: list[tuple] = []

    def box(self, side: str):
        lo, hi = self.cfg.workspace[side]
        return np.asarray(lo, float), np.asarray(hi, float)

    def filter_target(self, side: str, p, R, p_prev, R_prev, t: float = 0.0):
        p = np.asarray(p, float)
        lo, hi = self.box(side)
        pc = np.clip(p, lo, hi)
        if np.any(pc != p):
            self.events.append((t, "target_clipped_to_workspace", side, float(np.abs(pc - p).max())))
        step = pc - p_prev
        n, lim = np.linalg.norm(step), self.cfg.max_lin_speed * self.dt
        if n > lim:
            pc = p_prev + step * (lim / n)
            self.events.append((t, "target_speed_limited", side, float(n / self.dt)))
        rv = rotvec(np.asarray(R, float) @ R_prev.T)
        a, alim = np.linalg.norm(rv), self.cfg.max_ang_speed * self.dt
        Rc = np.asarray(R, float)
        if a > alim:
            Rc = rotvec_to_R(rv * (alim / a)) @ R_prev
            self.events.append((t, "target_rot_speed_limited", side, float(a / self.dt)))
        return pc, Rc

    def check_state(
        self,
        side: str,
        p_world,
        v_world,
        wrench,
        stamp: float,
        now: float,
        fault: str,
        enforce_motion: bool,
        t: float = 0.0,
    ) -> str | None:
        """Returns an abort reason or None. Force, fault and stale-state checks always abort. Workspace and speed
        checks abort only when we are moving the arm (streaming, and not a hardware dry run); otherwise logged."""
        if fault:
            return f"{side}: backend fault: {fault}"
        if now - stamp > self.cfg.state_timeout:
            return f"{side}: robot state is {now - stamp:.2f} s old"
        F = float(np.linalg.norm(wrench[:3]))
        if F > self.cfg.max_force:
            return f"{side}: |F| = {F:.1f} N > {self.cfg.max_force} N"
        lo, hi = self.box(side)
        m = self.cfg.workspace_abort_margin
        out = bool(np.any(p_world < lo - m) or np.any(p_world > hi + m))
        speed = float(np.linalg.norm(v_world[:3]))
        enforce = enforce_motion
        if out:
            if enforce:
                return f"{side}: TCP {np.round(p_world, 3)} outside the workspace box"
            self.events.append((t, "measured_outside_workspace", side, p_world.tolist()))
        if speed > self.cfg.abort_lin_speed:
            if enforce:
                return f"{side}: TCP speed {speed:.2f} m/s > {self.cfg.abort_lin_speed} m/s"
            self.events.append((t, "measured_speed_high", side, speed))
        return None


class RateKeeper:
    def __init__(self, hz: float):
        self.dt = 1.0 / hz
        self.next = None
        self.overruns = 0

    def sleep(self):
        now = time.monotonic()
        if self.next is None:
            self.next = now + self.dt
            return
        if now < self.next:
            time.sleep(self.next - now)
            self.next += self.dt
        else:
            self.overruns += 1
            self.next = now + self.dt


# ============================================================================================== geometry helpers
def cucumber_world_T(W_T_left_tcp: np.ndarray, F_T_tcp_left: np.ndarray, fixture: CucumberFixture) -> np.ndarray:
    """World pose of the cucumber body frame: left TCP -> flange -> Franka Hand -> grasp (CucumberFixture)."""
    return W_T_left_tcp @ inv_T(F_T_tcp_left) @ F_T_HAND @ fixture.hand_T_cucumber()


@dataclass
class ArmTarget:
    """What PeelExpert, StrokeSkillRunner and recorder.action20_from_env read from env.left / env.right."""

    side: str
    tgt_p: np.ndarray
    tgt_R: np.ndarray
    grip_w: float
    jlo: np.ndarray = field(default_factory=lambda: Q_MIN.copy())
    jhi: np.ndarray = field(default_factory=lambda: Q_MAX.copy())


class _DataShim:
    """env.data.site_xpos[env.blade_site] is read by policy_runner.StrokeSkillRunner for its termination test."""

    def __init__(self, env: RealBimanualFranka):
        self._env = env

    @property
    def site_xpos(self) -> np.ndarray:
        return self._env.tcp_world("right")[None, :3, 3]


# ============================================================================================== the environment
class RealBimanualFranka:
    """PeelEnv interface on two Pandas. backend: "mock" (default, no hardware) or "franky" (two FCI connections).

    dry_run (default True) only matters for hardware: the arms are connected and read, cameras run, safety checks
    run, targets are computed and logged, but no motion command is sent. The mock has no hardware and always
    integrates its fake arms."""

    LEFT_P0 = PeelEnv.LEFT_P0
    LEFT_R0 = PeelEnv.LEFT_R0
    RAKE0 = PeelEnv.RAKE0
    Q_LEFT_SEED = PeelEnv.Q_LEFT_SEED
    Q_RIGHT_SEED = PeelEnv.Q_RIGHT_SEED

    def __init__(
        self,
        config: RealConfig | None = None,
        backend="mock",
        fixture: CucumberFixture | None = None,
        dry_run: bool = True,
        render_obs: bool = True,
        seed: int = 0,
        cameras: str = "auto",
        ft_sensor: BotaFTSensor | None = None,
        realtime: bool | None = None,
    ):
        self.config = config or RealConfig()
        self.config.safety.dry_run = bool(dry_run)
        self.dry_run = bool(dry_run)
        self.fixture = fixture or CucumberFixture()
        self.dt = 1.0 / self.config.control_hz
        if isinstance(backend, str):
            if backend == "mock":
                backend = MockBimanualBackend(self.config, self.fixture, seed=seed)
            elif backend == "franky":
                backend = FrankyBimanualBackend(self.config, dry_run=self.dry_run)
            else:
                raise ValueError(f"unknown backend {backend!r} (mock | franky)")
        self.backend = backend
        self.realtime = self.backend.realtime if realtime is None else bool(realtime)  # pace steps at 24 Hz
        self.safety = SafetyMonitor(self.config.safety, self.dt)
        self.rate = RateKeeper(self.config.control_hz)
        self.cfg = RealEpisodeConfig(cucumber=self.fixture.spec(), seed=seed, render_obs=render_obs)
        self.shape = self.cfg.cucumber.shape
        ft_cfg = self.config.right.ft_sensor
        if ft_sensor is None and ft_cfg and ft_cfg.get("type") == "bota":
            ft_sensor = BotaFTSensor(ft_cfg["config_path"], ft_cfg.get("F_T_sensor"), ft_cfg.get("calib"))
        self.ft_sensor = ft_sensor
        if cameras == "auto":
            cameras = "mock" if self.backend.name == "mock" else "realsense"
        self.cameras = {}
        if render_obs:
            for name, cc in self.config.cameras.items():
                self.cameras[name] = (
                    MockCamera(name, self.config.image_hw)
                    if cameras == "mock"
                    else RealSenseCamera(cc, self.config.image_hw)
                )
        self.data = _DataShim(self)
        self.blade_site = 0
        self.t = 0.0
        self.last_obs = self.prev_obs = None  # obs after the latest step and the one before (for recording)
        self.aborted = ""
        self._streaming = False
        self._states: dict[str, ArmState] = {}
        self._last_step_wall = None
        self.backend.connect()
        for c in self.cameras.values():
            c.start()
        if self.ft_sensor is not None:
            self.ft_sensor.start()
        self._refresh(check=False)
        self.left = ArmTarget("left", *self._tcp_pR("left"), self._states["left"].gripper_width)
        self.right = ArmTarget("right", *self._tcp_pR("right"), self._states["right"].gripper_width)
        self.arms = {"left": self.left, "right": self.right}
        self.q_cmd = {s: self._states[s].q.copy() for s in ("left", "right")}
        self._last_grip_cmd = {"left": None, "right": None}
        self._last_grip_time = {"left": -1e9, "right": -1e9}
        self._init_episode_state()

    # ------------------------------------------------------------------ bookkeeping

    def blade_pos(self) -> np.ndarray:
        """World position of the blade edge (the peeler TCP); same method as PeelEnv.blade_pos."""
        return self.tcp_world("right")[:3, 3].copy()

    def _init_episode_state(self):
        c = self.cfg.cucumber
        lo, hi = c.peel_zone
        self.skin = SkinGrid(c.shape.r_mid, lo, hi, c.n_axial, c.n_ang, self.cfg.skin)
        self.cut_widths: list[float] = []
        self.t = 0.0
        self.step_count = 0
        self.peak_fn = 0.0
        self.mean_fn_step = 0.0
        self.handle_hits = 0
        self._blade_local_prev = None
        self.q7_start = float(self._states["left"].q[6])
        self.left_R_start = self.tcp_world("left")[:3, :3].copy()
        self.roll_max = float((Q_MAX[6] - 0.08) - self.q7_start)

    def _tcp_pR(self, side: str):
        T = self.tcp_world(side)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def tcp_world(self, side: str) -> np.ndarray:
        return self.config.arm(side).W_T_B @ self._states[side].T

    def _refresh(self, check: bool = True):
        now = time.monotonic()
        for side in ("left", "right"):
            st = self.backend.read(side)
            if side == "right" and self.ft_sensor is not None:
                O_T_F = st.T @ inv_T(self.config.right.F_T_TCP)
                st.wrench = self.ft_sensor.wrench_sim(O_T_F)
            self._states[side] = st
            for ev in st.events:
                self.safety.events.append((self.t, ev, side, None))
        if not check:
            return
        for side in ("left", "right"):
            st = self._states[side]
            Rb = self.config.arm(side).W_T_B[:3, :3]
            reason = self.safety.check_state(
                side,
                self.tcp_world(side)[:3, 3],
                Rb @ st.twist[:3],
                st.wrench,
                st.stamp,
                now,
                st.fault,
                self._moves_arm(),
                self.t,
            )
            if reason:
                self._abort(reason)

    def _moves_arm(self) -> bool:
        """True when our targets actually move an arm: streaming, and either the mock or a non-dry-run."""
        return self._streaming and (self.backend.name == "mock" or not self.dry_run)

    def _abort(self, reason: str):
        self.aborted = reason
        self.safety.events.append((self.t, "abort", "", reason))
        try:
            self.backend.stop(reason)
        finally:
            self._streaming = False
        raise SafetyAbort(reason)

    def _check_alive(self):
        if self.aborted:
            raise SafetyAbort(f"env aborted earlier ({self.aborted}); call reset()")

    # ------------------------------------------------------------------ reset / start pose
    def plan_start_joints(self) -> dict:
        """The sim's start pose for this cucumber, solved with the numpy IK in each real base frame:
        left TCP at LEFT_P0 with the hand z along -y (roll free), joint 7 near its lower limit (sim reset);
        right blade parked 6 cm above the gripper end of the peel zone, in the nominal peeling orientation."""
        cl, cr = self.config.left, self.config.right
        T_L = inv_T(cl.W_T_B) @ make_T(self.LEFT_P0, self.LEFT_R0)
        qL, eL = panda_ik(T_L, self.Q_LEFT_SEED, cl.F_T_TCP, free_roll=True)
        qL[6] = Q_MIN[6] + 0.08
        W_tl = cl.W_T_B @ panda_fk(qL, cl.F_T_TCP)
        W_c = cucumber_world_T(W_tl, cl.F_T_TCP, self.fixture)
        z0 = self.cfg.cucumber.peel_zone[0] + 0.01
        park = self._surface_top(z0, W_c)[0] + np.array([0.0, 0.0, 0.06])
        Rb = blade_frame(W_c[:3, :3][:, 2], self.RAKE0)
        qR, eR = panda_ik(inv_T(cr.W_T_B) @ make_T(park, Rb), self.Q_RIGHT_SEED, cr.F_T_TCP)
        return {
            "left": qL,
            "right": qR,
            "ik_err_left": float(np.linalg.norm(eL[:3])),
            "ik_err_right": float(np.linalg.norm(eR[:3])),
            "park": park,
        }

    def reset(self, cfg=None, grasp: bool = True, move: bool = True, before_grasp=None):
        """Move to the sim start pose (slow joint moves; skipped in a dry run), open the left gripper, call
        before_grasp() (e.g. ask the operator to insert the cucumber), grasp it, start the impedance controllers
        and zero the episode counters. cfg is ignored (kept for the PeelEnv signature)."""
        if self._streaming:
            self.backend.stop("reset")
            self._streaming = False
        self.backend.rearm()
        self.aborted = ""
        plan = self.plan_start_joints()
        if max(plan["ik_err_left"], plan["ik_err_right"]) > 0.002:
            raise RuntimeError(f"start pose IK did not converge: {plan}")
        if move:
            # right arm first: it parks above the cucumber; the left arm then brings the cucumber in from below
            for side in ("right", "left"):
                self.backend.move_to_joints(side, plan[side])
        if grasp:
            self.backend.gripper_command("left", "move", GRIP_MAX, 0.0)
            if self.realtime and not self.dry_run:
                self._wait_gripper("left")
            if before_grasp is not None:
                before_grasp()
            self.backend.gripper_command("left", "grasp", 2 * self.shape.r_grip, self.config.left.grasp_force)
            self._last_grip_cmd["left"] = 2 * self.shape.r_grip
        self.backend.advance(0.5)
        if grasp and self.realtime and not self.dry_run:
            self._wait_gripper("left")
        self._refresh(check=False)
        self.backend.start_streaming()
        self._streaming = True
        for side, arm in self.arms.items():
            arm.tgt_p, arm.tgt_R = self._tcp_pR(side)
            self.q_cmd[side] = self._states[side].q.copy()
        self.left.grip_w = 2 * self.shape.r_grip if grasp else self._states["left"].gripper_width
        self.right.grip_w = self._states["right"].gripper_width
        self._init_episode_state()
        self._last_step_wall = None
        self.rate = RateKeeper(self.config.control_hz)
        self.last_obs = self.obs()
        self.prev_obs = None
        return self.last_obs

    def _wait_gripper(self, side: str, timeout: float = 5.0, still: float = 0.5):
        """Block until the gripper width has not changed by more than 1 mm for `still` seconds."""
        t0 = t_still = time.monotonic()
        w_ref = self.backend.read(side).gripper_width
        while time.monotonic() - t0 < timeout:
            time.sleep(0.05)
            w = self.backend.read(side).gripper_width
            if abs(w - w_ref) > 0.001:
                w_ref, t_still = w, time.monotonic()
            elif time.monotonic() - t_still > still:
                return
        raise RuntimeError(f"{side} gripper did not settle within {timeout} s")

    # ------------------------------------------------------------------ stepping
    def step_tcp(self, left=None, right=None, grip=None):
        """Absolute TCP targets (pos, R) in the world frame for either arm; one 24 Hz control step."""
        self._check_alive()
        now = time.monotonic()
        if (
            self._streaming
            and self._last_step_wall is not None
            and now - self._last_step_wall > self.config.safety.heartbeat_abort
        ):
            self._abort(f"heartbeat: {now - self._last_step_wall:.2f} s between control steps")
        for arm, tgt in ((self.left, left), (self.right, right)):
            if tgt is not None:
                arm.tgt_p, arm.tgt_R = self.safety.filter_target(arm.side, tgt[0], tgt[1], arm.tgt_p, arm.tgt_R, self.t)
        if grip is not None:
            self.left.grip_w = float(np.clip(grip[0], 0.0, GRIP_MAX))
            self.right.grip_w = float(np.clip(grip[1], 0.0, GRIP_MAX))
        for side, arm in self.arms.items():
            T_base = inv_T(self.config.arm(side).W_T_B) @ make_T(arm.tgt_p, arm.tgt_R)
            self.backend.send(side, T_base, arm.grip_w)
            self._gripper_update(side, arm.grip_w)
        self.backend.advance(self.dt)
        if self.realtime:
            self.rate.sleep()
        self._last_step_wall = time.monotonic()
        self._refresh(check=True)
        self._update_contact_estimate()
        self.t += self.dt
        self.step_count += 1
        self.prev_obs, self.last_obs = self.last_obs, self.obs(fresh=False)
        return self.last_obs

    def step_action(self, a):
        """RDP bimanual action: [l_xyz, l_rot6d, r_xyz, r_rot6d, l_grip, r_grip] (20,)."""
        a = np.asarray(a, dtype=float)
        return self.step_tcp(
            left=(a[0:3], rot6d_to_mat(a[3:9])), right=(a[9:12], rot6d_to_mat(a[12:18])), grip=(a[18], a[19])
        )

    def idle(self, duration: float):
        """Wait without sending targets (lets the heartbeat watchdog act; used by the tests)."""
        self.backend.advance(duration)
        if self.backend.realtime:  # hardware runs in real time; the mock integrates `duration` at once
            time.sleep(duration)
        self._refresh(check=False)

    def _gripper_update(self, side: str, width: float, precision: float = 0.01, min_interval: float = 0.5):
        """Franka Hand commands are slow and blocking, so only send on a real change (RDP's real_franka_env.yaml
        uses a 0.01 m width precision for the same reason). Closing = grasp with force, opening = move."""
        cfg = self.config.arm(side)
        if not cfg.has_gripper:
            return
        last = self._last_grip_cmd[side]
        now = time.monotonic()
        if last is not None and (abs(width - last) < precision or now - self._last_grip_time[side] < min_interval):
            return
        mode = "grasp" if width < self._states[side].gripper_width - 0.005 else "move"
        self.backend.gripper_command(side, mode, width, cfg.grasp_force)
        self._last_grip_cmd[side], self._last_grip_time[side] = width, now

    # GELLO Robot protocol -------------------------------------------------------
    def num_dofs(self) -> int:
        return 16

    def get_joint_state(self) -> np.ndarray:
        out = []
        for side in ("left", "right"):
            st = self._states[side]
            out += list(st.q) + [1.0 - st.gripper_width / GRIP_MAX]
        return np.array(out)

    def command_joint_state(self, joint_state):
        """GELLO bimanual command [left q(7), left grip (0 open..1 closed), right q(7), right grip]. Leader joints ->
        numpy FK -> TCP target, so GELLO and policies share one controller (as in PeelEnv). The mock also uses the
        leader joints as nullspace posture; franky 2.0.0 fixes the posture target when streaming starts."""
        js = np.asarray(joint_state, dtype=float)
        targets, grips = {}, []
        for k, side in enumerate(("left", "right")):
            q = np.clip(js[8 * k : 8 * k + 7], Q_MIN, Q_MAX)
            self.q_cmd[side] = q
            self.backend.set_posture(side, q)
            T = self.config.arm(side).W_T_B @ panda_fk(q, self.config.arm(side).F_T_TCP)
            targets[side] = (T[:3, 3], T[:3, :3])
            grips.append((1.0 - np.clip(js[8 * k + 7], 0, 1)) * GRIP_MAX)
        return self.step_tcp(left=targets["left"], right=targets["right"], grip=grips)

    def get_observations(self):
        o = self.obs(render=False)
        return {
            "joint_positions": self.get_joint_state(),
            "joint_velocities": np.concatenate([self._states[s].dq for s in ("left", "right")]),
            "ee_pos_quat": np.zeros(14),
            "gripper_position": np.array([o["left_robot_gripper_width"][0], o["right_robot_gripper_width"][0]]),
        }

    # ------------------------------------------------------------------ geometry (fixture stub)
    def cucumber_frame(self):
        W_c = cucumber_world_T(self.tcp_world("left"), self.config.left.F_T_TCP, self.fixture)
        return W_c[:3, 3].copy(), W_c[:3, :3].copy()

    def _surface_top(self, z_local: float, W_c: np.ndarray):
        pc, Rc = W_c[:3, 3], W_c[:3, :3]
        t = Rc @ self.shape.tangent(z_local)
        up = np.array([0, 0, 1.0])
        n = up - (up @ t) * t
        n /= np.linalg.norm(n)
        p = pc + Rc @ self.shape.centre(z_local) + float(self.shape.radius(z_local)) * n
        return p, n, t

    def surface_top(self, z_local: float):
        """Top-surface point (world), outward normal and centre-line tangent at local z (same as PeelEnv)."""
        pc, Rc = self.cucumber_frame()
        return self._surface_top(z_local, make_T(pc, Rc))

    def cucumber_point(self, z_local: float, up: float = 0.0) -> np.ndarray:
        p, n, _ = self.surface_top(z_local)
        return p + up * n

    def left_R(self, roll: float) -> np.ndarray:
        c, s = np.cos(roll), np.sin(roll)
        return self.left_R_start @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def park_pose(self) -> np.ndarray:
        return self.cucumber_point(self.cfg.cucumber.peel_zone[0] + 0.01, 0.0) + np.array([0.0, 0.0, 0.06])

    # ------------------------------------------------------------------ contact and skin ESTIMATE
    def _update_contact_estimate(self, contact_band: float = 0.006):
        """Blade normal force from the measured wrench along the estimated surface normal, and the sim's skin rule
        (peel/skin.py) applied to it. An ESTIMATE for the orchestrator, not a coverage measurement."""
        pc, Rc = self.cucumber_frame()
        Tb = self.tcp_world("right")
        local = Rc.T @ (Tb[:3, 3] - pc)
        z_c, th_c, r_c, n_local = self.shape.project(local)
        height = r_c - float(self.shape.radius(z_c))
        f_w = self.config.right.W_T_B[:3, :3] @ self._states["right"].wrench[:3]
        fn = max(float(f_w @ (Rc @ n_local)), 0.0) if height < contact_band else 0.0
        v_local = np.zeros(3) if self._blade_local_prev is None else (local - self._blade_local_prev) / self.dt
        self._blade_local_prev = local
        self.mean_fn_step = fn
        if fn < 0.3:
            self.skin.lift()
            return
        self.peak_fn = max(self.peak_fn, fn)
        out = self.skin.apply_contact(
            z_c, th_c, float(self.shape.radius(z_c)), n_local, self.shape.tangent(z_c), fn, v_local, Rc.T @ Tb[:3, 0]
        )
        if out["cut"]:
            self.cut_widths.append(out["width"])

    # ------------------------------------------------------------------ observations
    def obs(self, render: bool | None = None, fresh: bool = True) -> dict:
        if fresh:
            self._refresh(check=self._streaming)
        render = self.cfg.render_obs if render is None else render
        o = {}
        for side in ("left", "right"):
            st = self._states[side]
            Rb = self.config.arm(side).W_T_B[:3, :3]
            T = self.tcp_world(side)
            o[f"{side}_robot_tcp_pose"] = np.concatenate([T[:3, 3], rot6d(T[:3, :3])]).astype(np.float32)
            o[f"{side}_robot_tcp_vel"] = np.concatenate([Rb @ st.twist[:3], Rb @ st.twist[3:]]).astype(np.float32)
            o[f"{side}_robot_tcp_wrench"] = np.concatenate([Rb @ st.wrench[:3], Rb @ st.wrench[3:]]).astype(np.float32)
            o[f"{side}_robot_gripper_width"] = np.array([st.gripper_width], np.float32)
            o[f"{side}_robot_gripper_force"] = np.zeros(1, np.float32)  # libfranka reports no gripper force
            o[f"{side}_robot_joint_pos"] = np.asarray(st.q, np.float32)
        o["cucumber_roll"] = np.array([self._states["left"].q[6] - self.q7_start], np.float32)
        o["coverage"] = np.array([self.skin.coverage()], np.float32)  # ESTIMATE (see _update_contact_estimate)
        o["gouge"] = np.array([self.skin.gouge_fraction()], np.float32)  # ESTIMATE
        o["blade_fn"] = np.array([self.mean_fn_step], np.float32)
        if render:
            for name, cam in self.cameras.items():
                o[name] = cam.read()
        return o

    def metrics(self) -> dict:
        s = self.skin
        return {
            "coverage": s.coverage(),
            "gouge": s.gouge_fraction(),
            "coverage_is_estimate": True,
            "peak_fn": self.peak_fn,
            "handle_hits": self.handle_hits,
            "time": self.t,
            "safety_events": len(self.safety.events),
            "aborted": self.aborted,
            **s.events,
        }

    def stop(self, reason: str = "user stop"):
        self.backend.stop(reason)
        self._streaming = False

    def close(self):
        try:
            if self._streaming:
                self.backend.stop("close")
        finally:
            self._streaming = False
            self.backend.close()
            for c in self.cameras.values():
                c.close()
            if self.ft_sensor is not None:
                self.ft_sensor.close()


# ============================================================================================== stub policy
class StubStrokePolicy:
    """Stand-in for policy_runner.SimPolicy (no checkpoint): exercises StrokeSkillRunner on any env.

    Relative-action DP-style policy over the stroke task's shape_meta (rdp_configs/task/sim_stroke_wrench_ldp.yaml
    keys). Each chunk moves the blade `drift` metres per step along its own x axis (the stroke direction in the
    sim) and keeps the gripper at 0.018 m. Only for interface tests and dry runs."""

    def __init__(self, drift: float = 0.0, n_action_steps: int = 8, image_shape=(3, 96, 128)):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        self.latent = False
        self.relative = True
        self.n_obs_steps, self.obs_ds, self.dataset_obs_ds = 2, 1, 1
        self.n_action_steps = n_action_steps
        self.To_full = self.n_obs_steps * self.dataset_obs_ds
        self.name = "stub policy"
        self.shape_meta = {
            "obs": {
                "external_img": {"shape": list(image_shape), "type": "rgb"},
                "right_wrist_img": {"shape": list(image_shape), "type": "rgb"},
                "right_robot_tcp_pose": {"shape": [9], "type": "low_dim"},
                "right_robot_tcp_wrench": {"shape": [6], "type": "low_dim"},
            },
            "action": {"shape": [10]},
        }
        self.drift = drift
        self.calls = 0
        outer = self

        class _P:
            def reset(self):
                pass

            def predict_action(self, obs_t):
                outer.calls += 1
                for k in outer.shape_meta["obs"]:
                    assert k in obs_t, f"missing obs key {k}"
                n = outer.n_action_steps
                a = np.zeros((1, n, 10), np.float32)
                a[0, :, 0] = outer.drift * np.arange(1, n + 1)  # relative x, in the TCP frame at inference time
                a[0, :, 3:9] = [1, 0, 0, 0, 1, 0]  # identity rotation in RDP's 6D layout
                a[0, :, 9] = 0.018
                return {"action": outer.torch.from_numpy(a)}

        self.policy = _P()
