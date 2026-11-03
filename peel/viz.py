"""Video frames: external view + wrist view + unrolled skin map + live force trace with text overlay."""

from __future__ import annotations

import cv2
import numpy as np

from .skin import RGBA

W, H = 1280, 720


def skin_map_image(state: np.ndarray, hw=(240, 320)) -> np.ndarray:
    """Unrolled skin: rows = along the cucumber (gripper end at the bottom), columns = angle around it."""
    cols = (np.stack([RGBA[s][:3] for s in range(3)]) * 255).astype(np.uint8)
    img = cols[state[::-1]]  # flip so the gripper end is at the bottom
    img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return img


def force_trace_image(t: list, fn: list, band=(8.0, 13.0), gouge=24.0, hw=(170, 640), window=8.0) -> np.ndarray:
    h, w = hw
    img = np.full((h, w, 3), 245, np.uint8)
    fmax = 30.0

    def y(f):
        return int(h - 18 - (h - 30) * min(f, fmax) / fmax)

    cv2.rectangle(img, (0, y(band[1])), (w, y(band[0])), (205, 235, 205), -1)
    cv2.line(img, (0, y(gouge)), (w, y(gouge)), (60, 60, 220), 1)
    cv2.putText(img, "gouge risk", (w - 110, y(gouge) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 220), 1)
    for f in (0, 10, 20, 30):
        cv2.putText(img, f"{f} N", (4, y(f) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
    if len(t) > 1:
        t = np.asarray(t)
        fn = np.asarray(fn)
        sel = t >= t[-1] - window
        xs = (50 + (w - 60) * (t[sel] - (t[-1] - window)) / window).astype(int)
        pts = np.stack([xs, [y(f) for f in fn[sel]]], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, (40, 40, 40), 2)
    cv2.putText(
        img,
        "blade normal force (last 8 s), green = target band",
        (50, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (40, 40, 40),
        1,
    )
    return img


def compose(ext: np.ndarray, wrist: np.ndarray, skin_state: np.ndarray, t: list, fn: list, info: dict) -> np.ndarray:
    """1280x720 frame. ext 480x640, wrist 240x320 resized, skin map 240x320, force trace across the bottom."""
    frame = np.full((H, W, 3), 255, np.uint8)
    frame[0:480, 0:640] = cv2.resize(ext, (640, 480))
    frame[0:240, 640:960] = cv2.resize(wrist, (320, 240))
    frame[0:240, 960:1280] = skin_map_image(skin_state)
    frame[480:650, 0:640] = force_trace_image(
        t, fn, band=info.get("band", (8.0, 13.0)), gouge=info.get("f_gouge", 24.0)
    )
    # text panel
    x0, y0 = 660, 280
    lines = [
        info.get("title", ""),
        f"policy: {info.get('policy', '')}",
        f"phase: {info.get('phase', '')}   stroke: {info.get('stroke', '')}",
        f"skin removed: {100 * info.get('coverage', 0):5.1f} %   gouged: {100 * info.get('gouge', 0):4.1f} %",
        f"blade force: {info.get('fn', 0):5.1f} N   cutting: {'yes' if info.get('cut') else 'no'}",
        f"cucumber roll: {np.rad2deg(info.get('roll', 0)):6.1f} deg   time: {info.get('time', 0):5.1f} s",
        info.get("note", ""),
    ]
    for i, s in enumerate(lines):
        cv2.putText(
            frame, s, (x0, y0 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if i else 0.7, (30, 30, 30), 2 if i == 0 else 1
        )
    cv2.putText(frame, "wrist camera", (650, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, "skin map (unrolled)", (968, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(
        frame,
        "green = skin, pale = peeled, white = gouged",
        (660, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 90, 90),
        1,
    )
    cv2.putText(
        frame,
        "Simulation (MuJoCo). Skin removal is a rule-based model, see docs/SIMULATION.md",
        (10, 705),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (110, 110, 110),
        1,
    )
    return frame


class VideoWriter:
    def __init__(self, path, fps=24):
        import imageio.v2 as imageio

        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=8)

    def add(self, frame):
        self.w.append_data(frame)

    def close(self):
        self.w.close()
