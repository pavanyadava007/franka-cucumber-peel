"""Cucumber skin model: a removable grid of skin cells on a rigid cylinder.

No mainstream simulator can cut a thin skin off a vegetable with a blade, so the skin is not simulated as a
material. The cucumber is a rigid capsule with soft contact. Its cylindrical surface is divided into cells
(axial x angular). A cell comes off when the peeler blade sweeps over it while three conditions hold at the
contact, which are the conditions human and robot peelers need in the papers we read:

1. the normal force is above a cut threshold (too light and the blade skids over the wax),
2. the blade edge dips into the surface by a small rake angle (edge lifted: skid; edge too steep: dig),
3. the blade moves along the cucumber with its edge leading.

If the normal force is above a gouge threshold, or the blade is far too steep, the cell is marked "gouged":
the skin is gone but flesh was wasted. Strip width grows with force, like a real peeler biting deeper.

All thresholds are model assumptions, randomised per episode. They are not measured cucumber properties
(the research found no published cucumber skin mechanics). See docs/SIMULATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SKIN, PEELED, GOUGED = 0, 1, 2

RGBA = {
    SKIN: np.array([0.10, 0.32, 0.10, 1.0]),
    PEELED: np.array([0.78, 0.88, 0.58, 1.0]),
    GOUGED: np.array([0.97, 0.96, 0.86, 1.0]),
}


@dataclass
class SkinParams:
    f_cut: float = 3.5  # N, minimum normal force to bite into the skin
    f_gouge: float = 24.0  # N, above this the blade digs into the flesh
    rake_min: float = np.deg2rad(2.0)  # rad, edge must dip at least this much into the surface
    rake_max: float = np.deg2rad(35.0)  # rad, steeper than this the edge digs instead of slicing
    rake_gouge: float = np.deg2rad(50.0)  # rad, steeper than this always gouges when cutting
    v_min: float = 0.004  # m/s, minimum axial blade speed for a slicing stroke
    w_min: float = 0.008  # m, strip width at the cut threshold
    w_max: float = 0.020  # m, strip width cap (blade length minus the yoke)
    k_w: float = 0.0012  # m per N, extra strip width per newton above the cut threshold
    resist: float = 220.0  # N per m of strip width, cutting resistance along the stroke


class SkinGrid:
    """State of every skin cell, in the cucumber's local frame (z = cucumber axis, origin at capsule centre)."""

    def __init__(self, radius: float, z_lo: float, z_hi: float, n_axial: int, n_ang: int, params: SkinParams):
        self.radius = radius
        self.z_lo, self.z_hi = z_lo, z_hi
        self.n_axial, self.n_ang = n_axial, n_ang
        self.dz = (z_hi - z_lo) / n_axial
        self.dth = 2 * np.pi / n_ang
        self.p = params
        self.state = np.zeros((n_axial, n_ang), dtype=np.int8)
        self.events = {"cut_steps": 0, "skid_steps": 0, "gouge_steps": 0, "contact_steps": 0}
        self._last_z: float | None = None
        self.changed = False

    # ------------------------------------------------------------------ geometry helpers
    def cell_centres(self):
        """(n_axial, n_ang, 2) array of (z, theta) cell centres."""
        z = self.z_lo + (np.arange(self.n_axial) + 0.5) * self.dz
        th = (np.arange(self.n_ang) + 0.5) * self.dth
        return np.stack(np.meshgrid(z, th, indexing="ij"), axis=-1)

    # ------------------------------------------------------------------ metrics
    def coverage(self) -> float:
        return float((self.state != SKIN).mean())

    def gouge_fraction(self) -> float:
        return float((self.state == GOUGED).mean())

    def angular_coverage(self) -> np.ndarray:
        """Fraction peeled per angular sector (n_ang,), useful to see which side still has skin."""
        return (self.state != SKIN).mean(axis=0)

    # ------------------------------------------------------------------ the removal rule
    def apply_contact(
        self,
        z: float,
        theta: float,
        r_here: float,
        n_local: np.ndarray,
        t_local: np.ndarray,
        fn: float,
        v_local: np.ndarray,
        edge_dir_local: np.ndarray,
        fn_gouge: float | None = None,
    ) -> dict:
        """Apply one skin update of blade contact, given in surface coordinates of the (curved, tapered) body.

        z, theta: contact position along the centre line and around it; r_here: local radius (m)
        n_local, t_local: outward surface normal and centre-line tangent at the contact, cucumber frame (unit)
        fn: total blade normal force (N); v_local: blade velocity in the cucumber frame (m/s)
        fn_gouge: force used for the gouge decision (env passes a ~30 ms average; defaults to fn)
        edge_dir_local: unit vector the cutting edge points along (blade x axis), cucumber frame
        """
        p = self.p
        self.events["contact_steps"] += 1
        rake = float(np.arcsin(np.clip(-edge_dir_local @ n_local, -1.0, 1.0)))  # > 0: edge dips into the skin
        v_ax = float(v_local @ t_local)
        speed_ok = abs(v_ax) > p.v_min
        v_norm = np.linalg.norm(v_local) + 1e-9
        leading = float(edge_dir_local @ v_local) / v_norm > 0.5  # the edge must lead the motion
        out = {"cut": False, "gouge": False, "rake": rake, "fn": fn, "width": 0.0, "theta": theta, "z": z}
        z_prev = self._last_z if self._last_z is not None else z
        self._last_z = z
        cutting = fn >= p.f_cut and speed_ok and leading and rake >= p.rake_min
        if not cutting:
            self.events["skid_steps"] += 1
            return out
        gouge = (fn if fn_gouge is None else fn_gouge) >= p.f_gouge or rake >= p.rake_gouge
        if rake > p.rake_max and not gouge:
            self.events["skid_steps"] += 1  # too steep: the edge chatters, no clean strip this step
            return out
        width = float(np.clip(p.w_min + p.k_w * (fn - p.f_cut), p.w_min, p.w_max))
        out.update(cut=True, gouge=gouge, width=width)
        za, zb = sorted((z_prev, z))
        i0 = int(np.floor((za - self.z_lo) / self.dz))
        i1 = int(np.floor((zb - self.z_lo) / self.dz))
        i0, i1 = max(i0, 0), min(i1, self.n_axial - 1)
        if i0 > i1:
            return out  # contact outside the peelable zone (the gripped end or the tip)
        half = 0.5 * width / max(r_here, 1e-3)
        th_c = (np.arange(self.n_ang) + 0.5) * self.dth
        dth = np.abs((th_c - theta + np.pi) % (2 * np.pi) - np.pi)
        js = np.nonzero(dth <= half)[0]
        if js.size == 0:
            js = np.array([int(theta // self.dth) % self.n_ang])
        block = self.state[i0 : i1 + 1][:, js]
        if gouge:
            block[:] = GOUGED
            self.events["gouge_steps"] += 1
        else:
            block[block == SKIN] = PEELED
            self.events["cut_steps"] += 1
        self.state[i0 : i1 + 1, js] = block
        self.changed = True
        return out

    def lift(self):
        """Blade left the surface: the next contact starts a new sweep."""
        self._last_z = None
