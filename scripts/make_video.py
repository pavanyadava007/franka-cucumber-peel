"""Render a full peeling episode to MP4 (external view, wrist view, skin map, force trace).

MUJOCO_GL=glfw python scripts/make_video.py --seed 0                 # scripted expert via the GELLO path
MUJOCO_GL=glfw python scripts/make_video.py --seed 0 --noforce       # expert without force feedback
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peel.rollout import run_expert_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--noforce", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--shape-class", default=None)
ap.add_argument("--geometry", default="true", choices=["true", "nominal"])
a = ap.parse_args()
name = f"expert_{'noforce' if a.noforce else 'force'}_{a.geometry}geom_{a.shape_class or 'any'}_seed{a.seed}.mp4"
out = Path(a.out or ROOT / "media" / "videos" / name)
m, _, ep, cfg = run_expert_episode(
    a.seed,
    use_force=not a.noforce,
    render_obs=False,
    record=False,
    video=out,
    shape_class=a.shape_class,
    geometry=a.geometry,
)
print(json.dumps({"video": str(out), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}}))
