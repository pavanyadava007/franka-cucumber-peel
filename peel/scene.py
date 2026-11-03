"""Build the bimanual peeling scene as MJCF: two Franka Emika Panda arms (7 joints each) from MuJoCo Menagerie.

Left arm: holds the cucumber (rigid grasp, randomised offset and tilt) and rolls it about its own axis with joint 7.
Right arm: a Y-style peeler is mounted rigidly in the hand (like a printed adapter on the flange), with a 6-axis
force/torque site at the flange. Its tool centre point (TCP) is the blade edge.

The model is rebuilt per episode because cucumber size and grasp pose are randomised (compiling takes well under
a second). Everything is plain MJCF text so it does not depend on a specific MjSpec API version.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .shape import CucumberShape

ROOT = Path(__file__).resolve().parents[1]
PANDA_DIR = ROOT / "models" / "menagerie" / "franka_emika_panda"

TIMESTEP = 1.0 / 480.0  # 480 Hz physics, 20 substeps per 24 Hz control step
HAND_TCP_Z = 0.1034  # Menagerie hand: fingertip centre along the hand z axis
BASE_Y = 0.25  # side-by-side layout: arm bases at y = +-BASE_Y, both facing +x
BLADE_Z = 0.205  # blade edge along the right hand z axis (peeler sticks ~10 cm out of the fingers)


@dataclass
class CucumberSpec:
    shape: CucumberShape = field(default_factory=CucumberShape)
    mass: float = 0.30  # kg
    grasp_depth: float = 0.080  # m, hand z of the gripped tip (fingers grip the start of the body)
    grasp_offset: tuple = (0.0, 0.0)  # m, lateral offset of the axis in the hand (x, y)
    grasp_tilt: tuple = (0.0, 0.0)  # rad, tilt of the axis about hand x and y
    friction: float = 0.25
    solref: tuple = (0.01, 1.0)
    n_ang: int = 36
    cell_len: float = 0.005
    grip_clear: float = 0.025  # m, unpeelable band next to the fingers
    shape_class: str = "standard"

    @property
    def radius(self) -> float:
        """Nominal (middle) radius."""
        return self.shape.r_mid

    @property
    def length(self) -> float:
        return self.shape.length

    @property
    def peel_zone(self) -> tuple[float, float]:
        """(z_lo, z_hi) of the peelable body in the cucumber frame (rounded ends and the gripped band excluded)."""
        z0, z1 = self.shape.z_cyl
        return (z0 + self.grip_clear, z1)

    @property
    def n_axial(self) -> int:
        lo, hi = self.peel_zone
        return max(1, int(round((hi - lo) / self.cell_len)))


@dataclass
class SceneSpec:
    cucumber: CucumberSpec = field(default_factory=CucumberSpec)
    image_hw: tuple = (96, 128)


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    """w, x, y, z quaternion from a rotation matrix."""
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """MJCF xyaxes for a camera at `pos` looking at `target` (MuJoCo cameras look along their -z axis)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return _fmt(np.concatenate([x, y]))


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.ravel(v))


def _prefix_tree(elem: ET.Element, pre: str):
    """Prefix every name and every reference to a body/joint/tendon/site inside one arm copy."""
    for e in elem.iter():
        for key in ("name", "joint", "joint1", "joint2", "tendon", "body1", "body2", "site"):
            if key in e.attrib:
                e.attrib[key] = pre + e.attrib[key]


def _yoke_xml(R_hb: np.ndarray) -> str:
    """Visual yoke in the hand frame joining the handle end to the two blade ends."""
    tip = np.array([0, 0, BLADE_Z])
    y_b, x_b = R_hb[:, 1], R_hb[:, 0]
    handle_end = np.array([0, 0, 0.175])
    geoms = []
    for s in (-1, 1):
        end = tip - 0.004 * x_b + s * 0.022 * y_b
        mid = handle_end + s * 0.020 * y_b
        for a, b in ((handle_end, mid), (mid, end)):
            geoms.append(
                f'<geom type="capsule" fromto="{_fmt(a)} {_fmt(b)}" size="0.0022" rgba="0.6 0.6 0.62 1" '
                'contype="0" conaffinity="0" group="1"/>'
            )
    return "".join(geoms)


def _peeler_xml() -> str:
    """Peeler bodies inside the right hand frame (z = approach axis, fingers open along y)."""
    # blade frame relative to the hand: hand z points down-forward to the blade; the edge leads towards -x world.
    # Nominal world frames (see expert.py): hand z = (cos g, 0, -sin g), blade x = (-cos a, 0, -sin a).
    g, a = np.deg2rad(50.0), np.deg2rad(15.0)
    Rh = np.column_stack([[-np.sin(g), 0, -np.cos(g)], [0, 1, 0], [np.cos(g), 0, -np.sin(g)]])
    Rb = np.column_stack([[-np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, -np.cos(a)]])
    q_hb = _quat_from_mat(Rh.T @ Rb)
    return f"""
      <body name="peeler" pos="0 0 0">
        <inertial mass="0.08" pos="0 0 0.14" diaginertia="2e-4 2e-4 1e-5"/>
        <geom name="peeler_handle" type="capsule" fromto="0 0 0.075 0 0 0.175" size="0.009" rgba="0.85 0.2 0.15 1"
              contype="4" conaffinity="0" group="1"/>
        {_yoke_xml(Rh.T @ Rb)}
        <site name="blade" pos="0 0 {BLADE_Z}" quat="{_fmt(q_hb)}" size="0.003" rgba="1 0 1 0.6"/>
        <body name="blade_body" pos="0 0 {BLADE_Z}" quat="{_fmt(q_hb)}">
          <geom name="blade" type="box" pos="-0.004 0 0.0006" size="0.004 0.022 0.0006" rgba="0.75 0.78 0.8 1"
                contype="2" conaffinity="2" condim="3" friction="0.12 0.005 0.0001" solref="0.01 1"
                solimp="0.9 0.95 0.002"/>
        </body>
      </body>"""


def _cucumber_xml(c: CucumberSpec) -> str:
    """Cucumber as a child of the left hand (rigid grasp): a chain of capsules along a curved, tapered centre line,
    with visual skin cells lying on the same surface."""
    sh = c.shape
    tx, ty = c.grasp_tilt
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    R = Rx @ Ry
    centre = np.array([c.grasp_offset[0], c.grasp_offset[1], 0.0]) + R @ np.array([0, 0, c.grasp_depth + sh.length / 2])
    q = _quat_from_mat(R)
    lo, hi = c.peel_zone
    na, nt = c.n_axial, c.n_ang
    dz = (hi - lo) / na
    dth = 2 * np.pi / nt
    cells = []
    for i in range(na):
        z = lo + (i + 0.5) * dz
        t, e1, e2 = sh.frame(z)
        r = float(sh.radius(z)) + 0.0003
        for j in range(nt):
            th = (j + 0.5) * dth
            radial = np.cos(th) * e1 + np.sin(th) * e2
            tang = np.cross(t, radial)
            qc = _quat_from_mat(np.column_stack([radial, tang, t]))
            pos = sh.centre(z) + r * radial
            cells.append(
                f'<geom name="skin_{i}_{j}" type="box" pos="{_fmt(pos)}" quat="{_fmt(qc)}" '
                f'size="0.0003 {0.51 * r * dth:.6g} {0.51 * dz:.6g}" '
                f'rgba="0.10 0.32 0.10 1" contype="0" conaffinity="0" group="4"/>'
            )
    segs = []
    for k, (p0, p1, rad) in enumerate(sh.segments()):
        segs.append(
            f'<geom name="cucumber_seg{k}" type="capsule" fromto="{_fmt(p0)} {_fmt(p1)}" size="{rad:.6g}" '
            f'rgba="0.12 0.34 0.11 1" contype="2" conaffinity="6" condim="3" friction="{c.friction} 0.005 0.0001" '
            f'solref="{_fmt(c.solref)}" solimp="0.9 0.95 0.002" mass="0"/>'
        )
    L, rm = sh.length, sh.r_mid
    return f"""
      <body name="cucumber" pos="{_fmt(centre)}" quat="{_fmt(q)}">
        <inertial mass="{c.mass}" pos="{_fmt(sh.centre(0.0))}"
          diaginertia="{c.mass * L**2 / 12:.6g} {c.mass * L**2 / 12:.6g} {0.5 * c.mass * rm**2:.6g}"/>
        {"".join(segs)}
        <site name="cucumber_axis" pos="0 0 0" size="0.003" rgba="0 1 1 0.5"/>
        {"".join(cells)}
      </body>"""


def build_xml(spec: SceneSpec) -> str:
    src = ET.parse(PANDA_DIR / "panda.xml").getroot()
    root = ET.Element("mujoco", model="bimanual_franka_peel")
    comp = copy.deepcopy(src.find("compiler"))
    comp.attrib["meshdir"] = str(PANDA_DIR / "assets")
    root.append(comp)
    ET.SubElement(root, "option", timestep=f"{TIMESTEP:.8f}", integrator="implicitfast", cone="elliptic")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="640", offheight="480")
    ET.SubElement(vis, "quality", shadowsize="2048")
    root.append(copy.deepcopy(src.find("default")))
    asset = copy.deepcopy(src.find("asset"))
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=".82 .80 .76",
        rgb2=".74 .72 .68",
        width="256",
        height="256",
    )
    ET.SubElement(asset, "material", name="table", texture="grid", texrepeat="6 6", reflectance="0.05")
    root.append(asset)

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="key", pos="0.4 -0.6 1.6", dir="0 0.4 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(wb, "light", name="fill", pos="0.6 0.6 1.4", dir="-0.2 -0.4 -1", diffuse="0.4 0.4 0.4")
    ET.SubElement(
        wb,
        "geom",
        name="table",
        type="box",
        pos="0.35 0 -0.02",
        size="0.6 0.6 0.02",
        material="table",
        contype="1",
        conaffinity="1",
    )
    # external camera: front-right, looking at the cucumber
    # external camera: front, looking at the cucumber (lying across the front, x 0.45, y +0.05..-0.2, z 0.3)
    ext_pos, ext_tgt = (1.0, -0.32, 0.62), (0.45, -0.06, 0.30)
    ET.SubElement(wb, "camera", name="external", pos=_fmt(ext_pos), xyaxes=lookat_xyaxes(ext_pos, ext_tgt), fovy="40")
    ov_pos, ov_tgt = (1.6, -0.2, 1.0), (0.35, 0.0, 0.28)
    ET.SubElement(wb, "camera", name="overview", pos=_fmt(ov_pos), xyaxes=lookat_xyaxes(ov_pos, ov_tgt), fovy="45")

    link0 = src.find("worldbody").find("body")
    sides = {"left": (f"0 {BASE_Y} 0", "1 0 0 0"), "right": (f"0 {-BASE_Y} 0", "1 0 0 0")}
    for side, (bpos, bquat) in sides.items():
        base = ET.SubElement(wb, "body", name=f"{side}_base", pos=bpos, quat=bquat)
        arm = copy.deepcopy(link0)
        _prefix_tree(arm, f"{side}_")
        base.append(arm)
        hand = next(b for b in arm.iter("body") if b.attrib.get("name") == f"{side}_hand")
        ET.SubElement(hand, "site", name=f"{side}_ft", pos="0 0 0", size="0.006", rgba="1 1 0 0.3")
        ET.SubElement(hand, "site", name=f"{side}_tcp", pos=f"0 0 {HAND_TCP_Z}", size="0.004", rgba="1 0 1 0.4")
        if side == "left":
            hand.append(ET.fromstring(_cucumber_xml(spec.cucumber)))
        else:
            hand.append(ET.fromstring(_peeler_xml()))
            # wrist camera on the right hand: above and behind the hand, looking at the skin just ahead of the
            # blade. Designed in the nominal world frame, then expressed in the hand frame so it moves with the hand.
            g = np.deg2rad(50.0)
            Rh = np.column_stack([[-np.sin(g), 0, -np.cos(g)], [0, 1, 0], [np.cos(g), 0, -np.sin(g)]])
            cam_w = np.array([0.10, -0.12, -0.05])  # side bracket near blade height, clear of the hand housing
            blade_w = Rh @ np.array([0, 0, BLADE_Z])
            tgt_w = blade_w + np.array([-0.03, 0.0, -0.008])
            cam_h, tgt_h, up_h = Rh.T @ cam_w, Rh.T @ tgt_w, Rh.T @ np.array([0, 0, 1.0])
            ET.SubElement(
                hand, "camera", name="right_wrist", pos=_fmt(cam_h), xyaxes=lookat_xyaxes(cam_h, tgt_h, up_h), fovy="62"
            )

    tendon = ET.SubElement(root, "tendon")
    equality = ET.SubElement(root, "equality")
    actuator = ET.SubElement(root, "actuator")
    sensor = ET.SubElement(root, "sensor")
    contact = ET.SubElement(root, "contact")
    for side in sides:
        for t in src.find("tendon"):
            t2 = copy.deepcopy(t)
            _prefix_tree(t2, f"{side}_")
            tendon.append(t2)
        for e in src.find("equality"):
            e2 = copy.deepcopy(e)
            _prefix_tree(e2, f"{side}_")
            equality.append(e2)
        for a in src.find("actuator"):
            a2 = copy.deepcopy(a)
            _prefix_tree(a2, f"{side}_")
            if "joint" in a2.attrib:
                # torque control for the Cartesian impedance controller (ctrl = joint torque, N m)
                n = int(a2.attrib["joint"][-1])
                lim = 87 if n <= 4 else 12
                for k in ("gainprm", "biasprm", "forcerange"):
                    a2.attrib.pop(k, None)
                a2.attrib.update(gainprm="1", biasprm="0 0 0", ctrlrange=f"-{lim} {lim}")
            else:
                # Menagerie's gripper servo squeezes with ~3 N; kp 1000 gives ~30 N (real Franka Hand: 70 N)
                a2.attrib.update(gainprm="0.1568627451 0 0", biasprm="0 -1000 -30")
            actuator.append(a2)
        ET.SubElement(sensor, "force", name=f"{side}_ft_force", site=f"{side}_ft")
        ET.SubElement(sensor, "torque", name=f"{side}_ft_torque", site=f"{side}_ft")
        for ex in src.find("contact"):
            ex2 = copy.deepcopy(ex)
            _prefix_tree(ex2, f"{side}_")
            contact.append(ex2)
    return ET.tostring(root, encoding="unicode")


def build_model(spec: SceneSpec | None = None):
    import mujoco

    return mujoco.MjModel.from_xml_string(build_xml(spec or SceneSpec()))
