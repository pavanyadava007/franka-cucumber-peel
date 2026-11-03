"""Cucumber shape: tapered, curved body in the cucumber's local frame.

Local frame: the gripped end is at z = -L/2, the tip at z = +L/2. The centre line leaves the gripper along +z and
bends progressively (parabola) in direction `bend_dir`, so the tip is offset sideways by `bend_tip` metres. The
radius changes linearly along z (taper), and both ends are rounded. Collision geometry is a chain of overlapping
capsules along the centre line (MuJoCo capsules cannot taper or bend), the skin grid sits on the same surface.

Real cucumbers vary a lot: mini/snack types are ~12-16 cm and ~3 cm thick, slicing types ~20-24 cm, long English
types up to ~30-35 cm, many are thicker at one end and many are slightly bent. Those ranges drive SHAPE_CLASSES.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CucumberShape:
    length: float = 0.23  # m, tip to tip along the local z axis
    r_mid: float = 0.020  # m, radius at the middle
    taper: float = 0.0  # radius = r_mid * (1 + taper * z / (L/2)); > 0 means thicker at the tip
    bend_tip: float = 0.0  # m, sideways offset of the tip end of the centre line
    bend_dir: float = 0.0  # rad, direction of the bend in the local xy plane
    neck: float = 0.0  # stem-end "neck": radius reduced by this fraction at the stem end, blending out over neck_len
    neck_len: float = 0.25  # fraction of the length over which the neck blends into the body
    neck_at_tip: bool = False  # stem end at the free tip (True) or at the gripped end (False)
    n_seg: int = 16

    # ---------------------------------------------------------------- profile
    def radius(self, z):
        z = np.asarray(z, dtype=float)
        r = self.r_mid * (1.0 + self.taper * z / (self.length / 2))
        if self.neck > 0.0:
            d = (self.length / 2 - z) if self.neck_at_tip else (z + self.length / 2)
            u = np.clip(d / max(self.neck_len * self.length, 1e-6), 0.0, 1.0)
            r = r * (1.0 - self.neck * (1.0 - u * u * (3.0 - 2.0 * u)))  # smoothstep blend
        return r

    @property
    def arc_mm_per_10cm(self) -> float:
        """Bend in the grading-standard unit: height of the inner arc over the chord, per 10 cm of length.
        For a constant-curvature body held at one end the tip offset from the grip axis is 4 x the arc height."""
        span = self.z_cyl[1] - self.z_cyl[0]
        return 1000.0 * (self.bend_tip / 4.0) / (span / 0.1)

    @property
    def r_grip(self) -> float:
        return float(self.radius(-self.length / 2))

    @property
    def z_cyl(self) -> tuple[float, float]:
        """Axial range of the body without the rounded ends."""
        return (-self.length / 2 + self.r_grip, self.length / 2 - float(self.radius(self.length / 2)))

    @property
    def kappa(self) -> float:
        span = self.z_cyl[1] - self.z_cyl[0]
        return 2.0 * self.bend_tip / max(span, 1e-6) ** 2

    @property
    def d(self) -> np.ndarray:
        return np.array([np.cos(self.bend_dir), np.sin(self.bend_dir), 0.0])

    # ---------------------------------------------------------------- centre line and frames
    def centre(self, z):
        z = np.asarray(z, dtype=float)
        u = np.clip(z - self.z_cyl[0], 0.0, None)
        off = 0.5 * self.kappa * u**2
        return np.stack([self.d[0] * off, self.d[1] * off, z], axis=-1)

    def tangent(self, z):
        z = np.asarray(z, dtype=float)
        u = np.clip(z - self.z_cyl[0], 0.0, None)
        t = np.stack([self.d[0] * self.kappa * u, self.d[1] * self.kappa * u, np.ones_like(z)], axis=-1)
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def frame(self, z):
        """(t, e1, e2) at axial coordinate z. e1 stays in the bend plane, so angles are consistent along z."""
        t = self.tangent(z)
        ref = np.array([1.0, 0.0, 0.0]) if self.bend_tip == 0.0 else self.d
        e1 = ref - (t @ ref) * t
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        return t, e1, e2

    def surface_point(self, z, theta):
        t, e1, e2 = self.frame(z)
        return self.centre(z) + self.radius(z) * (np.cos(theta) * e1 + np.sin(theta) * e2)

    # ---------------------------------------------------------------- projection of a point onto the body
    def project(self, p):
        """Closest centre-line parameter z, angle theta, distance r and outward normal n for a local point p."""
        p = np.asarray(p, dtype=float)
        z = float(np.clip(p[2], *self.z_cyl))
        for _ in range(4):  # Newton on (c(z) - p) . c'(z) = 0
            c = self.centre(z)
            u = max(z - self.z_cyl[0], 0.0)
            c1 = np.array([self.d[0] * self.kappa * u, self.d[1] * self.kappa * u, 1.0])
            c2 = np.array([self.d[0] * self.kappa, self.d[1] * self.kappa, 0.0])
            f = (c - p) @ c1
            df = c1 @ c1 + (c - p) @ c2
            z = float(np.clip(z - f / df, *self.z_cyl))
        t, e1, e2 = self.frame(z)
        v = p - self.centre(z)
        v -= (v @ t) * t
        r = float(np.linalg.norm(v))
        n = v / (r + 1e-12)
        theta = float(np.arctan2(v @ e2, v @ e1)) % (2 * np.pi)
        return z, theta, r, n

    # ---------------------------------------------------------------- collision segments
    def segments(self):
        """[(p0, p1, radius)] capsules along the centre line covering the whole body."""
        z0, z1 = self.z_cyl
        zs = np.linspace(z0, z1, self.n_seg + 1)
        return [
            (self.centre(zs[k]), self.centre(zs[k + 1]), float(self.radius(0.5 * (zs[k] + zs[k + 1]))))
            for k in range(self.n_seg)
        ]

    def volume(self) -> float:
        z0, z1 = self.z_cyl
        zs = np.linspace(z0, z1, 50)
        r = self.radius(zs)
        return float(np.trapz(np.pi * r**2, zs) + 2.0 / 3.0 * np.pi * (self.r_grip**3 + self.radius(z1) ** 3))


def tip_offset_from_arc(arc_mm_per_10cm: float, length: float, r_ends: float) -> float:
    """Tip offset (m) of a constant-curvature body whose inner arc is `arc_mm_per_10cm` (UNECE FFV-15 unit)."""
    span = length - 2 * r_ends
    return 4.0 * (arc_mm_per_10cm / 1000.0) * (span / 0.1)


# v1 classes (first experiments; kept so earlier results stay reproducible)
SHAPE_CLASSES = {
    "mini": dict(length=(0.12, 0.16), r_mid=(0.014, 0.018), taper=(-0.08, 0.08), bend_tip=(0.0, 0.006)),
    "standard": dict(length=(0.19, 0.24), r_mid=(0.017, 0.022), taper=(-0.08, 0.08), bend_tip=(0.0, 0.006)),
    "long": dict(length=(0.26, 0.30), r_mid=(0.018, 0.023), taper=(-0.08, 0.08), bend_tip=(0.0, 0.008)),
    "thick": dict(length=(0.18, 0.25), r_mid=(0.024, 0.028), taper=(-0.08, 0.08), bend_tip=(0.0, 0.006)),
    "tapered": dict(length=(0.18, 0.26), r_mid=(0.017, 0.023), taper=(0.15, 0.25), bend_tip=(0.0, 0.006)),
    "curved": dict(length=(0.19, 0.27), r_mid=(0.017, 0.022), taper=(-0.08, 0.08), bend_tip=(0.012, 0.025)),
    "mixed": dict(length=(0.12, 0.30), r_mid=(0.014, 0.028), taper=(-0.22, 0.22), bend_tip=(0.0, 0.025)),
}


# v2 classes: real cucumber types. Lengths from variety guides (Persian 4-6 in, Kirby 3-6 in, English about 14 in,
# https://www.webstaurantstore.com/article/1063/types-of-cucumbers.html); bend from UNECE FFV-15 (Class I: inner arc
# <= 10 mm per 10 cm, Class II: <= 20 mm per 10 cm, crooked: more, https://unece.org/fileadmin/DAM/15Cucumbers_2010.pdf).
# Diameters are typical market values (not from a standard): pickling 30-40 mm, Persian 24-32, slicing 42-56,
# English 38-48. "arc" is in mm per 10 cm; "neck" is the stem-end radius reduction.
SHAPE_CLASSES_V2 = {
    "pickling": dict(length=(0.09, 0.14), r_mid=(0.015, 0.020), taper=(-0.10, 0.10), arc=(0.0, 10.0), neck=(0.0, 0.15)),
    "persian": dict(length=(0.11, 0.16), r_mid=(0.012, 0.016), taper=(-0.08, 0.08), arc=(0.0, 8.0), neck=(0.0, 0.15)),
    "slicing": dict(length=(0.18, 0.25), r_mid=(0.021, 0.028), taper=(-0.10, 0.10), arc=(0.0, 10.0), neck=(0.0, 0.20)),
    "english": dict(length=(0.28, 0.36), r_mid=(0.019, 0.024), taper=(-0.08, 0.08), arc=(0.0, 6.0), neck=(0.10, 0.30)),
    "crooked": dict(length=(0.18, 0.30), r_mid=(0.019, 0.026), taper=(-0.10, 0.10), arc=(10.0, 20.0), neck=(0.0, 0.20)),
    "necked": dict(length=(0.18, 0.30), r_mid=(0.019, 0.026), taper=(-0.05, 0.10), arc=(0.0, 10.0), neck=(0.25, 0.40)),
    "realistic": None,  # a mix: one of the four market types with a realistic spread of bend and neck
}
# stress test only (out of grade, not used for training): very crooked
SHAPE_CLASSES_V2_STRESS = {
    "very_crooked": dict(
        length=(0.18, 0.28), r_mid=(0.019, 0.025), taper=(-0.10, 0.10), arc=(20.0, 28.0), neck=(0.0, 0.20)
    ),
}
CLASS_SETS = {
    "v1": list(SHAPE_CLASSES),
    "v2": list(SHAPE_CLASSES_V2),
}


def _sample_v2(rng: np.random.Generator, r: dict) -> CucumberShape:
    length = rng.uniform(*r["length"])
    r_mid = rng.uniform(*r["r_mid"])
    taper = rng.uniform(*r["taper"])
    neck = rng.uniform(*r["neck"])
    shape = CucumberShape(
        length=length,
        r_mid=r_mid,
        taper=taper,
        bend_dir=rng.uniform(0, 2 * np.pi),
        neck=neck,
        neck_len=rng.uniform(0.18, 0.30),
        neck_at_tip=bool(rng.random() < 0.5),
    )
    r_ends = 0.5 * (shape.r_grip + float(shape.radius(length / 2)))
    shape.bend_tip = tip_offset_from_arc(rng.uniform(*r["arc"]), length, r_ends)
    return shape


def sample_shape(rng: np.random.Generator, cls: str) -> CucumberShape:
    if cls in SHAPE_CLASSES_V2 or cls in SHAPE_CLASSES_V2_STRESS:
        r = SHAPE_CLASSES_V2.get(cls) or SHAPE_CLASSES_V2_STRESS.get(cls)
        if r is None:  # "realistic": market type, then a realistic spread of bend (mostly Class I) and neck
            base = dict(
                SHAPE_CLASSES_V2[
                    str(rng.choice(["pickling", "persian", "slicing", "english"], p=[0.15, 0.2, 0.4, 0.25]))
                ]
            )
            u = rng.random()
            base["arc"] = (0.0, 10.0) if u < 0.75 else (10.0, 20.0)
            base["neck"] = (0.0, 0.05) if rng.random() < 0.5 else (0.10, 0.35)
            r = base
        return _sample_v2(rng, r)
    r = SHAPE_CLASSES[cls]
    taper = rng.uniform(*r["taper"])
    if cls == "tapered" and rng.random() < 0.5:
        taper = -taper  # thick end at the gripper or at the tip
    return CucumberShape(
        length=rng.uniform(*r["length"]),
        r_mid=rng.uniform(*r["r_mid"]),
        taper=taper,
        bend_tip=rng.uniform(*r["bend_tip"]),
        bend_dir=rng.uniform(0, 2 * np.pi),
        n_seg=12,
    )
