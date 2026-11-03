"""Fast checks of the simulation core (run: MUJOCO_GL=glfw .venv/bin/python -m pytest tests/test_sim.py -q)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peel.shape import SHAPE_CLASSES, CucumberShape, sample_shape  # noqa: E402
from peel.skin import GOUGED, PEELED, SKIN, SkinGrid, SkinParams  # noqa: E402


def _grid():
    return SkinGrid(0.02, -0.05, 0.05, 20, 36, SkinParams(f_cut=3.0, f_gouge=20.0))


def _cut(g, fn, rake_deg=15.0, v=0.08, z=0.0, theta=0.1, edge_sign=1.0):
    n = np.array([np.cos(theta), np.sin(theta), 0.0])
    t = np.array([0.0, 0.0, 1.0])
    a = np.deg2rad(rake_deg)
    edge = edge_sign * np.cos(a) * t - np.sin(a) * n
    return g.apply_contact(z, theta, 0.02, n, t, fn, np.array([0, 0, v]), edge)


def test_skin_cuts_only_with_force_rake_and_leading_edge():
    g = _grid()
    assert not _cut(g, 2.0)["cut"]  # below the cut threshold: skid
    assert not _cut(g, 10.0, rake_deg=-5.0)["cut"]  # edge lifted: skid
    assert not _cut(g, 10.0, edge_sign=-1.0)["cut"]  # edge trailing: skid
    g.lift()
    _cut(g, 10.0, z=-0.02)
    out = _cut(g, 10.0, z=0.02)
    assert out["cut"] and not out["gouge"]
    assert (g.state == PEELED).sum() > 0 and (g.state == GOUGED).sum() == 0


def test_skin_gouges_above_threshold_and_width_grows_with_force():
    g = _grid()
    w_lo = _cut(g, 5.0)["width"]
    w_hi = _cut(g, 12.0)["width"]
    assert w_hi > w_lo
    g.lift()
    _cut(g, 25.0, z=-0.01)
    _cut(g, 25.0, z=0.01)
    assert (g.state == GOUGED).sum() > 0


def test_shape_projection_roundtrip():
    sh = CucumberShape(length=0.25, r_mid=0.02, taper=0.2, bend_tip=0.02, bend_dir=0.7)
    for z in np.linspace(sh.z_cyl[0] + 0.01, sh.z_cyl[1] - 0.01, 7):
        for th in (0.3, 2.0, 4.5):
            p = sh.surface_point(z, th)
            z2, th2, r2, n = sh.project(p)
            assert abs(z2 - z) < 1e-4
            assert abs((th2 - th + np.pi) % (2 * np.pi) - np.pi) < 1e-4
            assert abs(r2 - sh.radius(z)) < 1e-5
            assert abs(np.linalg.norm(n) - 1) < 1e-9


@pytest.mark.parametrize("cls", list(SHAPE_CLASSES))
def test_scene_builds_and_rests(cls):
    from peel.env import EpisodeConfig, PeelEnv

    env = PeelEnv(EpisodeConfig.sample(3, render_obs=False, shape_class=cls))
    assert np.linalg.norm(env.ik_err_left[:3]) < 1e-3 and np.linalg.norm(env.ik_err_right[:3]) < 1e-3
    for _ in range(12):
        o = env.step_tcp()
    for side in ("left", "right"):
        assert np.linalg.norm(o[f"{side}_robot_tcp_wrench"][:3]) < 2.0  # tool gravity compensated at rest
    assert env.skin.coverage() == 0.0 and (env.skin.state == SKIN).all()


def test_gello_joint_path_tracks_forward_kinematics():
    from peel.env import EpisodeConfig, PeelEnv

    env = PeelEnv(EpisodeConfig.sample(4, render_obs=False, shape_class="standard"))
    js = env.get_joint_state()
    js[8 + 1] += 0.05  # nudge right joint 2
    for _ in range(36):
        env.command_joint_state(js)
    p_cmd, _ = env.right.fk(env.model, env.scratch, js[8:15])
    p_now, _ = env.right.pose(env.data)
    assert np.linalg.norm(p_cmd - p_now) < 0.005


def test_sample_shape_ranges():
    rng = np.random.default_rng(0)
    for cls, r in SHAPE_CLASSES.items():
        sh = sample_shape(rng, cls)
        assert r["length"][0] <= sh.length <= r["length"][1]
        assert r["r_mid"][0] <= sh.r_mid <= r["r_mid"][1]
