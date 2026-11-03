"""Bimanual Franka peeling environment (MuJoCo).

Control stack, same shape as the real robot plan:
  policy / GELLO leader (24 Hz)  ->  TCP targets for both arms  ->  Cartesian impedance at 480 Hz  ->  joint torques

Interfaces:
  * step_tcp(...)            : absolute TCP targets (what RDP outputs and what we record as "action")
  * step_action(a20)         : the RDP 20-D bimanual action (x,y,z, 6D rot) left, right, gripper widths
  * command_joint_state(q16) : GELLO's Robot protocol (7 joints + gripper per arm). Joint targets are turned into
                               TCP targets by forward kinematics, so GELLO and policies share one controller.
Observations use RDP's zarr key names (left_robot_tcp_pose, right_robot_tcp_wrench, external_img, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import CucumberSpec, SceneSpec, build_model
from .shape import CLASS_SETS, sample_shape
from .skin import RGBA, SkinGrid, SkinParams

CONTROL_HZ = 24
GRIP_MAX = 0.08  # m, Franka Hand opening


# ---------------------------------------------------------------------------------------------- rotations
def rot6d(R: np.ndarray) -> np.ndarray:
    """RDP convention: first two columns of R, i.e. R[:3, :2].T.flatten()."""
    return np.asarray(R)[:3, :2].T.flatten()


def rot6d_to_mat(v: np.ndarray) -> np.ndarray:
    x = v[0:3] / (np.linalg.norm(v[0:3]) + 1e-12)
    z = np.cross(x, v[3:6])
    z /= np.linalg.norm(z) + 1e-12
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_err(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """Orientation error as a rotation vector (world frame) taking R_cur to R_tgt."""
    Re = R_tgt @ R_cur.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Re.reshape(9))
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


# ---------------------------------------------------------------------------------------------- episode config
@dataclass
class EpisodeConfig:
    seed: int = 0
    cucumber: CucumberSpec = field(default_factory=CucumberSpec)
    skin: SkinParams = field(default_factory=SkinParams)
    ft_noise: float = 0.15  # N, white noise on each force axis (torque noise scaled by 0.01 m)
    render_obs: bool = True

    @staticmethod
    def sample(
        seed: int, render_obs: bool = True, shape_class: str | None = None, class_set: str = "v2"
    ) -> EpisodeConfig:
        """Domain randomisation: cucumber shape class (mini, standard, long, thick, tapered, curved, mixed), size,
        taper, bend, mass, grasp error, friction, firmness (contact stiffness) and skin toughness."""
        rng = np.random.default_rng(seed)
        classes = CLASS_SETS[class_set]
        cls = shape_class or classes[int(rng.integers(len(classes)))]
        shape = sample_shape(rng, cls)
        mass = 960.0 * shape.volume() * rng.uniform(0.9, 1.1)  # cucumber density ~0.96 g/cm^3
        cuc = CucumberSpec(
            shape=shape,
            mass=mass,
            grasp_offset=tuple(rng.uniform(-0.003, 0.003, 2)),
            grasp_tilt=tuple(rng.uniform(-np.deg2rad(2.5), np.deg2rad(2.5), 2)),
            friction=rng.uniform(0.15, 0.35),  # wet skin on a steel blade; MuJoCo uses the max of both geoms
            solref=(rng.uniform(0.008, 0.014), 1.0),  # firm to soft fruit: a few mm of contact compliance
            shape_class=cls,
        )
        skin = SkinParams(
            f_cut=rng.uniform(2.5, 4.5),
            f_gouge=rng.uniform(20.0, 28.0),
            resist=rng.uniform(160.0, 280.0),
        )
        return EpisodeConfig(
            seed=seed, cucumber=cuc, skin=skin, ft_noise=rng.uniform(0.05, 0.25), render_obs=render_obs
        )


# ---------------------------------------------------------------------------------------------- one arm
class Arm:
    """Index bookkeeping, FK/IK and the Cartesian impedance law for one Panda."""

    TAU_LIM = np.array([87, 87, 87, 87, 12, 12, 12.0])

    def __init__(self, m: mujoco.MjModel, side: str, tcp_site: str, K: np.ndarray, k_null: float = 8.0):
        self.side = side
        jid = [m.joint(f"{side}_joint{i}").id for i in range(1, 8)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.jlo, self.jhi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self.act = np.array([m.actuator(f"{side}_actuator{i}").id for i in range(1, 8)])
        self.grip_act = m.actuator(f"{side}_actuator8").id
        self.f1 = m.jnt_qposadr[m.joint(f"{side}_finger_joint1").id]
        self.f2 = m.jnt_qposadr[m.joint(f"{side}_finger_joint2").id]
        self.tcp = m.site(tcp_site).id
        self.ft_site = m.site(f"{side}_ft").id
        self.hand = m.body(f"{side}_hand").id
        self.ft_f = m.sensor_adr[m.sensor(f"{side}_ft_force").id]
        self.ft_t = m.sensor_adr[m.sensor(f"{side}_ft_torque").id]
        self.K = np.asarray(K, dtype=float)
        # damping for an effective arm mass of ~2.5 kg (trans) and 0.06 kg m^2 (rot); unit-mass damping rang in contact
        self.D = 2.0 * np.sqrt(self.K * np.array([2.5, 2.5, 2.5, 0.06, 0.06, 0.06]))
        self.k_null = k_null
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        # targets
        self.tgt_p = np.zeros(3)
        self.tgt_R = np.eye(3)
        self.q_null = np.zeros(7)
        self.grip_w = GRIP_MAX
        # MuJoCo's force sensor reports the force of the parent (link 7) on the hand subtree (checked: +m g at rest)
        self.ft_sign = 1.0

    def pose(self, d):
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def jac(self, m, d):
        mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.tcp)
        return np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])

    def impedance_torque(self, m, d):
        J = self.jac(m, d)
        p, R = self.pose(d)
        e = np.concatenate([self.tgt_p - p, rot_err(R, self.tgt_R)])
        # clip the pose error (as serl_franka_controllers / CRISP do) so a far target cannot slam the arm
        e[:3] = np.clip(e[:3], -0.05, 0.05)
        e[3:] = np.clip(e[3:], -0.4, 0.4)
        qd = d.qvel[self.dadr]
        F = self.K * e - self.D * (J @ qd)
        tau = J.T @ F + d.qfrc_bias[self.dadr]
        Jpinv = np.linalg.pinv(J.T)
        N = np.eye(7) - J.T @ Jpinv
        q = d.qpos[self.qadr]
        tau += N @ (self.k_null * (self.q_null - q) - 2.0 * np.sqrt(self.k_null) * qd)
        return np.clip(tau, -self.TAU_LIM, self.TAU_LIM)

    def grip_ctrl(self, width):
        # actuator8 drives the finger tendon: ctrl 0..255 -> each finger 0..0.04 m
        return float(np.clip(width / 2 / 0.04 * 255, 0, 255))

    def gripper_width(self, d):
        return float(d.qpos[self.f1] + d.qpos[self.f2])

    def ik(self, m, d_scratch, p, R, q_seed, iters=60, tol=2e-4, free_roll=False):
        """Damped least-squares IK on a scratch MjData (does not touch the simulation).

        free_roll: only the TCP position and the direction of the TCP z axis are constrained (roll about z is free).
        Used for the left arm, whose roll about the cucumber axis is set afterwards by joint 7 alone."""
        q = q_seed.copy()
        for _ in range(iters):
            d_scratch.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d_scratch)
            mujoco.mj_comPos(m, d_scratch)
            cur_p = d_scratch.site_xpos[self.tcp]
            cur_R = d_scratch.site_xmat[self.tcp].reshape(3, 3)
            if free_roll:
                e = np.concatenate([p - cur_p, np.cross(cur_R[:, 2], R[:, 2])])
            else:
                e = np.concatenate([p - cur_p, rot_err(cur_R, R)])
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 10 * tol:
                break
            mujoco.mj_jacSite(m, d_scratch, self.jacp, self.jacr, self.tcp)
            Jr = self.jacr[:, self.dadr]
            if free_roll:
                z = cur_R[:, 2]
                Jr = Jr - np.outer(z, z) @ Jr  # angular velocity about the TCP z axis is unconstrained
            J = np.vstack([self.jacp[:, self.dadr], Jr])
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            # small pull toward the seed keeps the elbow where it was
            dq += 0.02 * (np.eye(7) - np.linalg.pinv(J) @ J) @ (q_seed - q)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.jlo + 1e-3, self.jhi - 1e-3)
        return q, e

    def fk(self, m, d_scratch, q):
        d_scratch.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d_scratch)
        return d_scratch.site_xpos[self.tcp].copy(), d_scratch.site_xmat[self.tcp].reshape(3, 3).copy()


# ---------------------------------------------------------------------------------------------- environment
class PeelEnv:
    # nominal frames (world). Left hand points +x with fingers left/right; cucumber axis = left hand z axis.
    # cross layout: the cucumber lies across the front of both robots (along -y, tip toward the right arm)
    LEFT_P0 = np.array([0.48, 0.06, 0.30])
    LEFT_R0 = np.column_stack([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).astype(float)  # hand z = -y, fingers along x
    # right blade frame: edge leads toward +x (the right arm pulls the peeler toward itself, from the gripper end
    # out to the tip), dipping 15 deg into the skin
    RAKE0 = np.deg2rad(15.0)
    RIGHT_R0 = np.column_stack(
        [[0, -np.cos(RAKE0), -np.sin(RAKE0)], [-1, 0, 0], [0, np.sin(RAKE0), -np.cos(RAKE0)]]
    ).astype(float)
    # IK seeds for the cross layout from scripts/search_cross_layout.py: no arm-arm contact along the whole stroke,
    # joint margins 0.73 rad (left) and 0.79 rad (right)
    Q_LEFT_SEED = np.array([0.0355, 0.1672, 0.0125, -2.3388, -1.5335, 1.5407, -2.8173])
    Q_RIGHT_SEED = np.array([-1.005, -0.4736, 1.0144, -2.0773, 1.1011, 1.653, 2.1068])

    def __init__(self, cfg: EpisodeConfig | None = None, image_hw=(96, 128)):
        self.image_hw = image_hw
        self.renderers = {}
        self.reset(cfg or EpisodeConfig())

    # ------------------------------------------------------------------ build / reset
    def _build(self, cfg: EpisodeConfig):
        self.cfg = cfg
        self.model = build_model(SceneSpec(cucumber=cfg.cucumber, image_hw=self.image_hw))
        m = self.model
        self.data = mujoco.MjData(m)
        self.scratch = mujoco.MjData(m)
        self.left = Arm(m, "left", "left_tcp", K=[2200, 2200, 2200, 150, 150, 150])
        self.right = Arm(m, "right", "blade", K=[1100, 1100, 900, 60, 60, 60])
        self.arms = {"left": self.left, "right": self.right}
        self.cuc_body = m.body("cucumber").id
        self.peeler_body = m.body("blade_body").id
        self.blade_geom = m.geom("blade").id
        self.cuc_geoms = {m.geom(f"cucumber_seg{k}").id for k in range(cfg.cucumber.shape.n_seg)}
        self.shape = cfg.cucumber.shape
        self.handle_geom = m.geom("peeler_handle").id
        self.blade_site = m.site("blade").id
        c = cfg.cucumber
        lo, hi = c.peel_zone
        self.skin = SkinGrid(c.shape.r_mid, lo, hi, c.n_axial, c.n_ang, cfg.skin)
        self.skin_ids = np.array([[m.geom(f"skin_{i}_{j}").id for j in range(c.n_ang)] for i in range(c.n_axial)])
        self.rng = np.random.default_rng(cfg.seed + 7)
        self.substeps = int(round(1.0 / CONTROL_HZ / m.opt.timestep))
        for r in self.renderers.values():
            r.close()
        self.renderers = {}

    def reset(self, cfg: EpisodeConfig | None = None):
        self._build(cfg or self.cfg)
        m, d = self.model, self.data
        # left arm: IK to the nominal holding pose, then put joint 7 near its lower limit. Rolling the cucumber
        # about its own axis is then joint 7 alone (the TCP sits on the joint-7 axis), giving ~320 deg of roll.
        qL, self.ik_err_left = self.left.ik(
            m, self.scratch, self.LEFT_P0, self.LEFT_R0, self.Q_LEFT_SEED, iters=300, free_roll=True
        )
        qL[6] = self.left.jlo[6] + 0.08
        self.q7_start = qL[6]
        self.roll_max = (self.left.jhi[6] - 0.08) - self.q7_start
        _, self.left_R_start = self.left.fk(m, self.scratch, qL)
        d.qpos[:] = 0
        d.qpos[self.left.qadr] = qL
        c = self.cfg.cucumber
        for arm, w in ((self.left, 2 * c.shape.r_grip), (self.right, 0.018)):
            d.qpos[arm.f1] = d.qpos[arm.f2] = w / 2
            arm.grip_w = w
        mujoco.mj_forward(m, d)
        # right arm: park above the far end of the cucumber, blade in the nominal peeling orientation
        from .expert import blade_frame

        qR, self.ik_err_right = self.right.ik(
            m,
            self.scratch,
            self.park_pose(),
            blade_frame(self.cucumber_frame()[1][:, 2], self.RAKE0),
            self.Q_RIGHT_SEED,
            iters=300,
        )
        self.q_right_seed = qR.copy()
        d.qpos[self.right.qadr] = qR
        mujoco.mj_forward(m, d)
        for arm in self.arms.values():
            arm.tgt_p, arm.tgt_R = arm.pose(d)
            arm.q_null = d.qpos[arm.qadr].copy()
        self.q_cmd = {"left": qL.copy(), "right": qR.copy()}
        self.t = 0.0
        self.step_count = 0
        self.xfrc = np.zeros(6)
        self.last_contact = {"fn": 0.0, "cut": False, "gouge": False, "rake": 0.0, "width": 0.0}
        self.contact_log = []
        self.peak_fn = 0.0
        self.handle_hits = 0
        self.cut_widths = []
        self._blade_local_prev = None
        # settle the controller
        for _ in range(int(0.4 / m.opt.timestep)):
            self._physics_step(update_skin=False)
        self.t = 0.0
        self._update_skin_rgba(force=True)
        return self.obs()

    def left_R(self, roll: float) -> np.ndarray:
        """Left TCP orientation after rolling the cucumber by `roll` (rad, from the start) about the hand z axis."""
        c, s = np.cos(roll), np.sin(roll)
        return self.left_R_start @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def park_pose(self) -> np.ndarray:
        c = self.cfg.cucumber
        return self.cucumber_point(c.peel_zone[0] + 0.01, 0.0) + np.array([0.0, 0.0, 0.06])

    # ------------------------------------------------------------------ geometry helpers
    def blade_pos(self) -> np.ndarray:
        """World position of the blade edge (the peeler TCP)."""
        return self.data.site_xpos[self.blade_site].copy()

    def cucumber_frame(self):
        d = self.data
        return d.xpos[self.cuc_body].copy(), d.xmat[self.cuc_body].reshape(3, 3).copy()

    def surface_top(self, z_local: float):
        """Top-surface point (world), outward normal and centre-line tangent at local z of the real shape."""
        pc, Rc = self.cucumber_frame()
        t = Rc @ self.shape.tangent(z_local)
        up = np.array([0, 0, 1.0])
        n = up - (up @ t) * t
        n /= np.linalg.norm(n)
        p = pc + Rc @ self.shape.centre(z_local) + float(self.shape.radius(z_local)) * n
        return p, n, t

    def cucumber_point(self, z_local: float, up: float = 0.0) -> np.ndarray:
        """World point on the top surface at local z, plus `up` metres above it."""
        p, n, _ = self.surface_top(z_local)
        return p + up * n

    # ------------------------------------------------------------------ force/torque
    def _subtree_gravity_wrench(self, arm: Arm):
        """Wrench of gravity on everything distal to the flange site (hand, fingers, tool or cucumber)."""
        m, d = self.model, self.data
        mass = m.body_subtreemass[arm.hand]
        com = d.subtree_com[arm.hand]
        f = mass * m.opt.gravity
        tau = np.cross(com - d.site_xpos[arm.ft_site], f)
        return f, tau

    def wrench(self, arm: Arm, noisy: bool = True) -> np.ndarray:
        """External wrench applied BY the environment ON the tool/cucumber, about the flange, world axes
        (like Franka's O_F_ext_hat_K). Gravity of the distal bodies is removed (tool load compensation)."""
        d = self.data
        Rs = d.site_xmat[arm.ft_site].reshape(3, 3)
        f_w = Rs @ d.sensordata[arm.ft_f : arm.ft_f + 3]
        t_w = Rs @ d.sensordata[arm.ft_t : arm.ft_t + 3]
        fg, tg = self._subtree_gravity_wrench(arm)
        if arm.ft_sign > 0:  # reading = force of parent on child = -(gravity + external)
            f_ext, t_ext = -f_w - fg, -t_w - tg
        else:
            f_ext, t_ext = f_w - fg, t_w - tg
        w = np.concatenate([f_ext, t_ext])
        if noisy and self.cfg.ft_noise > 0:
            n = self.rng.normal(0, self.cfg.ft_noise, 6)
            n[3:] *= 0.01
            w = w + n
        return w

    # ------------------------------------------------------------------ blade contact and the skin rule
    def _blade_contact(self):
        """Sum the blade-cucumber normal force and its mean contact point (world)."""
        m, d = self.model, self.data
        fn, pt, hit_handle = 0.0, np.zeros(3), False
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            pair = {con.geom1, con.geom2}
            if not (pair & self.cuc_geoms):
                continue
            if self.handle_geom in pair:
                hit_handle = True
                continue
            if self.blade_geom not in pair:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn += f6[0]
            pt += f6[0] * con.pos
        if fn > 1e-6:
            pt /= fn
        return fn, pt, hit_handle

    def _skin_update(self, dt: float):
        d = self.data
        fn, pt, hit_handle = self._blade_contact()
        self.handle_hits += int(hit_handle)
        pc, Rc = self.cucumber_frame()
        blade_p = d.site_xpos[self.blade_site]
        blade_R = d.site_xmat[self.blade_site].reshape(3, 3)
        blade_local = Rc.T @ (blade_p - pc)
        v_local = np.zeros(3) if self._blade_local_prev is None else (blade_local - self._blade_local_prev) / dt
        self._blade_local_prev = blade_local
        self.xfrc[:] = 0
        d.xfrc_applied[self.peeler_body] = 0
        d.xfrc_applied[self.cuc_body] = 0
        # gouging is decided on force averaged over ~30 ms (a 4 ms contact impulse does not dig into the flesh);
        # cutting and strip width use the instantaneous force. Model assumption, see docs/SIMULATION.md.
        a = dt / (0.03 + dt)
        self._fn_filt = (1 - a) * getattr(self, "_fn_filt", 0.0) + a * fn
        if fn < 0.3:
            self.skin.lift()
            self.last_contact = {"fn": fn, "cut": False, "gouge": False, "rake": 0.0, "width": 0.0}
            return
        self.peak_fn = max(self.peak_fn, fn)
        self.peak_fn_filt = max(getattr(self, "peak_fn_filt", 0.0), self._fn_filt)
        p_local = Rc.T @ (pt - pc)
        z_c, th_c, r_c, n_local = self.shape.project(p_local)
        t_local = self.shape.tangent(z_c)
        edge_local = Rc.T @ blade_R[:, 0]
        out = self.skin.apply_contact(
            z_c, th_c, float(self.shape.radius(z_c)), n_local, t_local, fn, v_local, edge_local, fn_gouge=self._fn_filt
        )
        self.last_contact = out
        if out["cut"]:
            self.cut_widths.append(out["width"])
        if out["cut"]:
            # cutting resistance: opposes the blade's motion along the surface, equal and opposite on the cucumber
            v_w = Rc @ v_local
            n_w = Rc @ n_local
            v_t = v_w - (v_w @ n_w) * n_w
            sp = np.linalg.norm(v_t)
            if sp > 1e-6:
                F = -self.cfg.skin.resist * out["width"] * v_t / sp
                for body, sign in ((self.peeler_body, 1.0), (self.cuc_body, -1.0)):
                    torque = np.cross(pt - d.xipos[body], sign * F)
                    d.xfrc_applied[body, :3] = sign * F
                    d.xfrc_applied[body, 3:] = torque

    def _update_skin_rgba(self, force=False):
        if not (self.skin.changed or force):
            return
        cols = np.stack([RGBA[s] for s in range(3)])
        self.model.geom_rgba[self.skin_ids] = cols[self.skin.state]
        self.skin.changed = False

    # ------------------------------------------------------------------ stepping
    def _physics_step(self, update_skin=True, k=0):
        m, d = self.model, self.data
        for arm in self.arms.values():
            d.ctrl[arm.act] = arm.impedance_torque(m, d)
            d.ctrl[arm.grip_act] = arm.grip_ctrl(arm.grip_w)
        mujoco.mj_step(m, d)
        if update_skin and k % 2 == 1:
            self._skin_update(2 * m.opt.timestep)

    def step_tcp(self, left=None, right=None, grip=None):
        """Set absolute TCP targets (pos, R) for either arm and advance one 24 Hz control step."""
        for arm, tgt in ((self.left, left), (self.right, right)):
            if tgt is not None:
                arm.tgt_p = np.asarray(tgt[0], dtype=float).copy()
                arm.tgt_R = np.asarray(tgt[1], dtype=float).copy()
        if grip is not None:
            self.left.grip_w, self.right.grip_w = float(grip[0]), float(grip[1])
        fns = []
        for k in range(self.substeps):
            self._physics_step(update_skin=True, k=k)
            fns.append(self.last_contact["fn"])
        self.mean_fn_step = float(np.mean(fns))
        self.contact_log.append(
            (self.t, self.mean_fn_step, bool(self.last_contact.get("cut")), bool(self.last_contact.get("gouge")))
        )
        self.t += self.substeps * self.model.opt.timestep
        self.step_count += 1
        self._update_skin_rgba()
        return self.obs()

    def step_action(self, a: np.ndarray):
        """RDP bimanual action: [l_xyz, l_rot6d, r_xyz, r_rot6d, l_grip, r_grip] (20,)."""
        a = np.asarray(a, dtype=float)
        return self.step_tcp(
            left=(a[0:3], rot6d_to_mat(a[3:9])), right=(a[9:12], rot6d_to_mat(a[12:18])), grip=(a[18], a[19])
        )

    # GELLO Robot protocol -------------------------------------------------------
    def num_dofs(self) -> int:
        return 16

    def get_joint_state(self) -> np.ndarray:
        d = self.data
        out = []
        for arm in (self.left, self.right):
            out += list(d.qpos[arm.qadr]) + [1.0 - self.gripper_width_of(arm) / GRIP_MAX]
        return np.array(out)

    def command_joint_state(self, joint_state: np.ndarray):
        """GELLO bimanual command: [left q(7), left grip(0 open..1 closed), right q(7), right grip].
        Leader joints -> FK -> TCP target; leader joints also become the null-space posture target."""
        js = np.asarray(joint_state, dtype=float)
        targets = {}
        grips = []
        for k, arm in enumerate((self.left, self.right)):
            q = np.clip(js[8 * k : 8 * k + 7], arm.jlo, arm.jhi)
            self.q_cmd[arm.side] = q
            arm.q_null = q.copy()
            targets[arm.side] = arm.fk(self.model, self.scratch, q)
            grips.append((1.0 - np.clip(js[8 * k + 7], 0, 1)) * GRIP_MAX)
        return self.step_tcp(left=targets["left"], right=targets["right"], grip=grips)

    def get_observations(self):
        o = self.obs(render=False)
        return {
            "joint_positions": self.get_joint_state(),
            "joint_velocities": np.concatenate([self.data.qvel[a.dadr] for a in (self.left, self.right)]),
            "ee_pos_quat": np.zeros(14),
            "gripper_position": np.array([o["left_robot_gripper_width"][0], o["right_robot_gripper_width"][0]]),
        }

    # ------------------------------------------------------------------ observations
    def gripper_width_of(self, arm: Arm) -> float:
        return arm.gripper_width(self.data)

    def render(self, camera: str, hw=None, pretty: bool = False) -> np.ndarray:
        hw = tuple(hw or self.image_hw)
        key = (hw, pretty)
        if key not in self.renderers:
            r = mujoco.Renderer(self.model, *hw)
            for f in (
                mujoco.mjtRndFlag.mjRND_SHADOW,
                mujoco.mjtRndFlag.mjRND_REFLECTION,
                mujoco.mjtRndFlag.mjRND_SKYBOX,
            ):
                r.scene.flags[f] = 0
            self.renderers[key] = r
        r = self.renderers[key]
        opt = mujoco.MjvOption()
        # policy cameras use the low-poly collision meshes (12 ms/frame on CPU rendering); videos use full meshes
        opt.geomgroup[:] = [1, 1, 1, 0, 1, 0] if pretty else [1, 1, 0, 1, 1, 0]
        opt.sitegroup[:] = 0
        r.update_scene(self.data, camera, scene_option=opt)
        return r.render()

    def obs(self, render: bool | None = None) -> dict:
        d = self.data
        render = self.cfg.render_obs if render is None else render
        o = {}
        for side, arm in self.arms.items():
            p, R = arm.pose(d)
            v = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_SITE, arm.tcp, v, 0)
            o[f"{side}_robot_tcp_pose"] = np.concatenate([p, rot6d(R)]).astype(np.float32)
            o[f"{side}_robot_tcp_vel"] = np.concatenate([v[3:], v[:3]]).astype(np.float32)
            o[f"{side}_robot_tcp_wrench"] = self.wrench(arm).astype(np.float32)
            o[f"{side}_robot_gripper_width"] = np.array([self.gripper_width_of(arm)], np.float32)
            o[f"{side}_robot_gripper_force"] = np.zeros(1, np.float32)
            o[f"{side}_robot_joint_pos"] = d.qpos[arm.qadr].astype(np.float32)
        o["cucumber_roll"] = np.array([d.qpos[self.left.qadr[6]] - self.q7_start], np.float32)
        o["coverage"] = np.array([self.skin.coverage()], np.float32)
        o["gouge"] = np.array([self.skin.gouge_fraction()], np.float32)
        o["blade_fn"] = np.array([self.mean_fn_step if hasattr(self, "mean_fn_step") else 0.0], np.float32)
        if render:
            o["external_img"] = self.render("external")
            o["right_wrist_img"] = self.render("right_wrist")
        return o

    def metrics(self) -> dict:
        s = self.skin
        return {
            "coverage": s.coverage(),
            "gouge": s.gouge_fraction(),
            "peak_fn": self.peak_fn,
            "handle_hits": self.handle_hits,
            "time": self.t,
            **s.events,
        }

    def close(self):
        for r in self.renderers.values():
            r.close()
        self.renderers = {}
