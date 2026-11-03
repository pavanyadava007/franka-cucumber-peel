"""Scripted demonstrator, the orchestrator, and the virtual GELLO operator.

IMPORTANT (honesty): no human or real GELLO device recorded these demos. A scripted expert plans blade poses
and regulates force from the wrist sensor, the way a person watching a force display would. Its pose targets are
turned into JOINT angles by IK, quantised to GELLO's Dynamixel resolution, given a little hand tremor, and sent
through env.command_joint_state(), i.e. exactly the path a real GELLO leader arm uses. Plug a real GELLO pair in
(scripts/teleop_gello_sim.py) and the same recorder stores human demos instead.

Layers (see docs/ARCHITECTURE.md):
  Orchestrator   decides which side of the cucumber to peel next (from the skin map) and how far to roll it
  Stroke skill   approach -> touch -> press to the target force -> slide along the axis -> lift
                 (scripted here; RDP learns to replace it)
  Controller     480 Hz Cartesian impedance in env.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PeelEnv, rot6d

GELLO_RES = 2 * np.pi / 4096  # rad, Dynamixel XL330 encoder resolution


# ---------------------------------------------------------------------------------------------- GELLO operator
class VirtualGelloOperator:
    """Turns desired TCP poses into the joint commands a GELLO leader would send."""

    def __init__(self, env: PeelEnv, rng: np.random.Generator, tremor_m: float = 0.0004):
        self.env = env
        self.rng = rng
        self.tremor_m = tremor_m
        self.q_seed = {"left": env.q_cmd["left"].copy(), "right": env.q_cmd["right"].copy()}
        self._tremor = {"left": np.zeros(3), "right": np.zeros(3)}

    def joints_for(self, side: str, p: np.ndarray, R: np.ndarray) -> np.ndarray:
        env = self.env
        arm = env.arms[side]
        # low-pass filtered hand tremor (a real operator's hand is never perfectly still)
        self._tremor[side] = 0.85 * self._tremor[side] + 0.15 * self.rng.normal(0, self.tremor_m, 3) * 3
        q, _ = arm.ik(env.model, env.scratch, p + self._tremor[side], R, self.q_seed[side], iters=30)
        self.q_seed[side] = q
        return np.round(q / GELLO_RES) * GELLO_RES  # encoder quantisation

    def command(self, left, right, grip_closed=(1.0, 1.0)):
        js = np.concatenate(
            [self.joints_for("left", *left), [grip_closed[0]], self.joints_for("right", *right), [grip_closed[1]]]
        )
        return js


# ---------------------------------------------------------------------------------------------- blade geometry
def blade_frame(axis: np.ndarray, rake: float) -> np.ndarray:
    """Blade orientation for a stroke from the gripper end out to the tip, along the cucumber axis."""
    s = axis / np.linalg.norm(axis)  # stroke direction (cucumber +z = toward the tip)
    up = np.array([0.0, 0.0, 1.0])
    n = up - (up @ s) * s
    n /= np.linalg.norm(n)
    x_b = np.cos(rake) * s - np.sin(rake) * n
    y_b = np.cross(s, n)
    z_b = np.cross(x_b, y_b)
    return np.column_stack([x_b, y_b, z_b])


@dataclass
class ExpertParams:
    f_target: float = 10.0  # N, pressing force the operator aims for
    speed: float = 0.08  # m/s along the axis
    rake: float = np.deg2rad(15.0)
    use_force: bool = True  # False = open-loop depth (an operator with no force display)
    k_i: float = 0.00025  # m per N per control step, depth integrator
    f_filter: float = 0.35  # weight of the newest force sample (a smoothed force display)
    approach_h: float = 0.03  # m above the surface
    max_strokes: int = 18
    coverage_goal: float = 0.93
    # "true": the expert knows the exact surface (privileged, used for the demos).
    # "nominal": it only knows a straight cylinder along the gripper axis with a radius guessed with ~1.5 mm error
    # (what a camera estimate or a person's eye gives); grasp tilt/offset, taper and bend are unknown.
    geometry: str = "true"
    end_overrun: float = 0.008  # m the blade edge runs past the end of the peel zone
    r_est_err: float = 0.0

    @staticmethod
    def sample(rng: np.random.Generator, use_force: bool = True, geometry: str = "true") -> ExpertParams:
        return ExpertParams(
            f_target=rng.uniform(8.0, 13.0),
            speed=rng.uniform(0.06, 0.11),
            rake=np.deg2rad(rng.uniform(10.0, 20.0)),
            use_force=use_force,
            geometry=geometry,
            r_est_err=float(rng.normal(0.0, 0.0015)),
        )


# ---------------------------------------------------------------------------------------------- orchestrator
class Orchestrator:
    """Chooses the next side to peel from the skin map and the roll needed to bring it to the top.

    In simulation the skin map is the ground truth. On the real robot peel/real_robot.py keeps an estimate (the
    same skin rule fed with measured force and pose) and `scripts/run_real.py scan` scores coverage from the camera.
    """

    def __init__(self, env: PeelEnv, goal: float = 0.93, max_strokes: int = 14):
        self.env = env
        self.goal = goal
        self.max_strokes = max_strokes
        self.strokes = 0

    def top_theta(self, roll: float | None = None) -> float:
        """Cucumber-frame angle currently facing up (where the blade works)."""
        env = self.env
        _, Rc = env.cucumber_frame()
        up_local = Rc.T @ np.array([0, 0, 1.0])
        th_now = float(np.arctan2(up_local[1], up_local[0]))
        if roll is None:
            return th_now % (2 * np.pi)
        cur_roll = float(env.obs(render=False)["cucumber_roll"][0])
        return (th_now - (roll - cur_roll)) % (2 * np.pi)

    def next_roll(self) -> float | None:
        """Roll target (rad from the start) for the next stroke; None when the cucumber is done.

        Sweep: after each stroke, measure the angular width of the strip just cut and roll forward by 80 % of it
        (strips overlap a little, as when peeling by hand). When joint 7 runs out of range, fill the remaining gaps:
        roll to the centre of the largest unpeeled sector, forward or back.
        """
        env = self.env
        cov = env.skin.angular_coverage()
        cur_roll = float(env.obs(render=False)["cucumber_roll"][0])
        n = cov.size
        dth = 2 * np.pi / n
        if self.strokes == 0:
            self._cov_before = cov.copy()
            self.sweep = True
            return 0.0
        # strip width actually cut by the blade during the last stroke (what an operator sees on the cucumber)
        if env.cut_widths:
            self.last_strip = float(np.median(env.cut_widths)) / env.cfg.cucumber.radius
        env.cut_widths.clear()
        self.last_strip = getattr(self, "last_strip", 0.7)
        self._cov_before = cov.copy()
        if env.skin.coverage() >= self.goal or self.strokes >= self.max_strokes:
            return None
        # stop gap filling when strokes no longer help (a sector the blade cannot reach): 2 strokes in a row < 0.5 %
        cov_now = env.skin.coverage()
        gains = getattr(self, "_gains", [])
        gains.append(cov_now - getattr(self, "_cov_last", 0.0))
        self._gains, self._cov_last = gains, cov_now
        if not getattr(self, "sweep", True) and len(gains) >= 3 and max(gains[-2:]) < 0.005:
            return None  # only while filling gaps; during the first sweep a single stroke may miss on a bent body
        if self.sweep:
            step = float(np.clip(0.8 * self.last_strip, 0.2, 0.9))
            if cur_roll + step <= env.roll_max:
                return cur_roll + step
            self.sweep = False
        # gap filling: centre of the longest run of sectors that are still mostly skin
        todo = cov < 0.6
        if not todo.any():
            return None
        best, run = None, []
        for j in list(range(n)) + list(range(n)):  # wrap around
            if todo[j % n]:
                run.append(j)
            else:
                if run and (best is None or len(run) > len(best)):
                    best = run
                run = []
        if run and (best is None or len(run) > len(best)):
            best = run
        best = best[:n]
        th_gap = ((np.mean(best) + 0.5) * dth) % (2 * np.pi)
        d = (self.top_theta() - th_gap) % (2 * np.pi)  # rolling by +d brings angle (top - d) to the top
        for cand in sorted((cur_roll + d, cur_roll + d - 2 * np.pi), key=lambda r: abs(r - cur_roll)):
            if 0.0 <= cand <= env.roll_max:
                return cand
        return None


# ---------------------------------------------------------------------------------------------- the expert
class PeelExpert:
    """Generates one GELLO joint command per control step for the whole peel (orchestrator + stroke skill)."""

    def __init__(self, env: PeelEnv, params: ExpertParams, rng: np.random.Generator, gello: bool = True):
        self.env = env
        self.p = params
        self.rng = rng
        self.gello = gello
        self.op = VirtualGelloOperator(env, rng)
        self.orch = Orchestrator(env, params.coverage_goal, params.max_strokes)
        self.roll = 0.0
        self.phase = "start"
        self.stroke_id = -1
        self.depth = 0.0
        self.done = False

    # targets -----------------------------------------------------------------------------------------------
    def _left_target(self):
        return self.env.LEFT_P0.copy(), self.env.left_R(self.roll)

    def _surface(self, z_local: float):
        """Top-surface point, outward normal and centre-line tangent at cucumber-frame z, from the true shape or from
        the nominal estimate (see ExpertParams.geometry)."""
        env = self.env
        if self.p.geometry == "true":
            return env.surface_top(z_local)
        # nominal: straight cylinder on the gripper axis, nominal grasp depth, estimated radius
        d = env.data
        hand = env.left.hand
        Rh = d.xmat[hand].reshape(3, 3)
        c = env.cfg.cucumber
        axis = Rh[:, 2]
        centre = d.xpos[hand] + Rh @ np.array([0.0, 0.0, c.grasp_depth + c.shape.length / 2 + z_local])
        up = np.array([0, 0, 1.0])
        n = up - (up @ axis) * axis
        n /= np.linalg.norm(n)
        r = c.shape.r_mid + self.p.r_est_err
        return centre + r * n, n, axis

    def plan(self):
        """Yields (phase, stroke_id, left_target, right_target) every control step."""
        env, p = self.env, self.p
        dt = 1.0 / 24.0
        c = env.cfg.cucumber
        # strokes run from the gripper end (z_start) out to the tip (z_end), edge leading toward the tip
        # start a little before the peel zone and finish a little past it: on short cucumbers (pickling, Persian) a few
        # millimetres of margin are 10-20 % of the skin
        # (the blade removes skin ~4 mm behind its edge, so the edge must run past the zone end onto the tip cap)
        z_start, z_end = c.peel_zone[0] - 0.003, c.peel_zone[1] + p.end_overrun
        while True:
            roll = self.orch.next_roll()
            if roll is None:
                break
            # ---------------- roll the cucumber (right arm parked above)
            self.stroke_id += 1
            self.orch.strokes += 1
            start_roll = self.roll
            n_roll = max(6, int(abs(roll - start_roll) / 1.6 / dt))  # ~1.6 rad/s
            surf, n, axis = self._surface(z_start - 0.006)
            Rb = blade_frame(axis, p.rake)
            park = surf + n * (p.approach_h + 0.02)
            for k in range(n_roll):
                self.roll = start_roll + (roll - start_roll) * (k + 1) / n_roll
                yield "roll", self.stroke_id, self._left_target(), (park, Rb)
            for _ in range(4):
                yield "roll", self.stroke_id, self._left_target(), (park, Rb)
            # ---------------- approach above the far end
            surf, n, axis = self._surface(z_start)
            above = surf + n * p.approach_h
            cur = env.right.tgt_p.copy()
            for k in range(10):
                yield "approach", self.stroke_id, self._left_target(), (cur + (above - cur) * (k + 1) / 10, Rb)
            # ---------------- descend until the blade bites (or the depth limit)
            self.depth = -p.approach_h
            for _ in range(40):
                o = env.obs(render=False)
                fn = float(o["right_robot_tcp_wrench"][:3] @ n)
                if fn > 0.6 * p.f_target and p.use_force:
                    break
                if not p.use_force and self.depth >= p.f_target / 700.0:
                    break
                # fast until ~6 mm above the skin, then 1 mm per step so the first touch does not spike the force
                self.depth += 0.0025 if self.depth < -0.006 else 0.001
                surf, n, axis = self._surface(z_start)
                yield "descend", self.stroke_id, self._left_target(), (surf - n * self.depth, Rb)
            # ---------------- stroke toward the gripper, regulating force
            length = z_end - z_start
            n_steps = max(8, int(length / p.speed / dt))
            fn_f = p.f_target
            depth0 = self.depth
            for k in range(n_steps):
                z = z_start + length * (k + 1) / n_steps
                o = env.obs(render=False)
                fn = float(o["right_robot_tcp_wrench"][:3] @ n)
                fn_f = (1 - p.f_filter) * fn_f + p.f_filter * fn
                if p.use_force:
                    self.depth = float(
                        np.clip(self.depth + p.k_i * (p.f_target - fn_f), depth0 - 0.008, depth0 + 0.008)
                    )
                surf, n, axis = self._surface(z)
                Rb = blade_frame(axis, p.rake)
                yield "stroke", self.stroke_id, self._left_target(), (surf - n * self.depth, Rb)
            # ---------------- lift and go back to the far end
            surf, n, axis = self._surface(z_end)
            top = surf + n * (p.approach_h + 0.02)
            cur = env.right.tgt_p.copy()
            for k in range(8):
                yield "lift", self.stroke_id, self._left_target(), (cur + (top - cur) * (k + 1) / 8, Rb)
            surf, n, axis = self._surface(z_start - 0.006)
            park = surf + n * (p.approach_h + 0.02)
            for k in range(10):
                yield "return", self.stroke_id, self._left_target(), (top + (park - top) * (k + 1) / 10, Rb)
        self.done = True

    def command_stream(self):
        """Yields (phase, stroke_id, gello_joint_command or None, tcp_action20)."""
        for phase, sid, left, right in self.plan():
            if self.gello:
                js = self.op.command(
                    left, right, grip_closed=(1 - 2 * self.env.cfg.cucumber.shape.r_grip / 0.08, 1 - 0.018 / 0.08)
                )
                yield phase, sid, js, None
            else:
                a = np.concatenate(
                    [
                        left[0],
                        rot6d(left[1]),
                        right[0],
                        rot6d(right[1]),
                        [2 * self.env.cfg.cucumber.shape.r_grip, 0.018],
                    ]
                )
                yield phase, sid, None, a
