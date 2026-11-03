"""One rollout loop used for demo collection, videos and policy evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import EpisodeConfig, PeelEnv
from .expert import ExpertParams, PeelExpert
from .recorder import EpisodeRecorder, action20_from_env

SUCCESS_COVERAGE = 0.90
SUCCESS_MAX_GOUGE = 0.03


def is_success(m: dict) -> bool:
    return m["coverage"] >= SUCCESS_COVERAGE and m["gouge"] <= SUCCESS_MAX_GOUGE and m["handle_hits"] == 0


class VideoSink:
    """Renders the pretty views every step and writes a composed MP4."""

    def __init__(self, path: Path, title: str, policy: str, every: int = 1):
        from .viz import VideoWriter

        self.w = VideoWriter(path)
        self.title, self.policy, self.every = title, policy, every
        self.t, self.fn = [], []

    def add(self, env: PeelEnv, phase: str, stroke: int, note: str = ""):
        from .viz import compose

        self.t.append(env.t)
        self.fn.append(float(env.mean_fn_step) if hasattr(env, "mean_fn_step") else 0.0)
        if env.step_count % self.every:
            return
        ext = env.render("external", hw=(480, 640), pretty=True)
        wrist = env.render("right_wrist", hw=(240, 320), pretty=True)
        lc = env.last_contact
        info = dict(
            title=self.title,
            policy=self.policy,
            phase=phase,
            stroke=stroke,
            coverage=env.skin.coverage(),
            gouge=env.skin.gouge_fraction(),
            fn=self.fn[-1],
            cut=lc.get("cut", False),
            roll=float(env.data.qpos[env.left.qadr[6]] - env.q7_start),
            time=env.t,
            f_gouge=env.cfg.skin.f_gouge,
            note=note,
        )
        self.w.add(compose(ext, wrist, env.skin.state, self.t, self.fn, info))

    def close(self):
        self.w.close()


def run_expert_episode(
    seed: int,
    use_force: bool = True,
    render_obs: bool = True,
    record: bool = True,
    video: Path | None = None,
    max_time: float = 150.0,
    gello: bool = True,
    shape_class: str | None = None,
    geometry: str = "true",
):
    """Scripted expert through the GELLO joint path. Returns (metrics, recorder or None, expert params)."""
    cfg = EpisodeConfig.sample(seed, render_obs=render_obs, shape_class=shape_class)
    env = PeelEnv(cfg)
    rng = np.random.default_rng(seed)
    ep = ExpertParams.sample(rng, use_force=use_force, geometry=geometry)
    ex = PeelExpert(env, ep, rng, gello=gello)
    rec = EpisodeRecorder() if record else None
    vid = (
        VideoSink(
            video,
            f"Scripted expert via GELLO path, {cfg.cucumber.shape_class} cucumber, seed {seed}",
            "expert" + ("" if use_force else " (no force feedback)"),
        )
        if video
        else None
    )
    obs = env.obs()
    for phase, sid, js, a in ex.command_stream():
        if gello:
            new_obs = env.command_joint_state(js)
        else:
            new_obs = env.step_action(a)
        if rec is not None:
            rec.add(obs, action20_from_env(env), phase, sid, js)
        if vid is not None:
            vid.add(env, phase, sid)
        obs = new_obs
        if env.t > max_time:
            break
    m = env.metrics()
    m["success"] = is_success(m)
    m["strokes"] = ex.orch.strokes
    if vid is not None:
        vid.close()
    env.close()
    return m, rec, ep, cfg
