"""Collect scripted-expert demos through the GELLO joint path, in parallel, as raw .npz episodes.

Example:  MUJOCO_GL=glfw python scripts/collect_demos.py --n 60 --start-seed 1000 --workers 4
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def work(args):
    seed, out, use_force, class_set = args
    from peel.shape import CLASS_SETS

    classes = CLASS_SETS[class_set]
    shape_class = classes[seed % len(classes)]  # balanced classes
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("LP_NUM_THREADS", "1")  # CPU rendering: one thread per worker
    from peel.rollout import run_expert_episode

    t0 = time.time()
    m, rec, ep, cfg = run_expert_episode(
        seed, use_force=use_force, render_obs=True, record=True, shape_class=shape_class
    )
    sh = cfg.cucumber.shape
    meta = dict(
        seed=seed,
        shape_class=shape_class,
        success=m["success"],
        coverage=m["coverage"],
        gouge=m["gouge"],
        peak_fn=m["peak_fn"],
        strokes=m["strokes"],
        f_target=ep.f_target,
        speed=ep.speed,
        rake=ep.rake,
        use_force=use_force,
        radius=sh.r_mid,
        length=sh.length,
        taper=sh.taper,
        bend_tip=sh.bend_tip,
        f_cut=cfg.skin.f_cut,
        f_gouge=cfg.skin.f_gouge,
        friction=cfg.cucumber.friction,
    )
    rec.save(Path(out) / f"ep_{seed:06d}.npz", meta)
    clean = {k: (v if isinstance(v, (bool, int, str)) else float(v)) for k, v in meta.items()}
    return {**clean, "frames": len(rec), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--noforce", action="store_true", help="expert without force feedback (ablation)")
    ap.add_argument("--class-set", default="v2", choices=["v1", "v2"], help="v1 first shapes, v2 real cucumber types")
    ap.add_argument("--skip-done", action="store_true", help="skip seeds already recorded successfully in the log")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "collect_log.jsonl"
    done = set()
    if a.skip_done and log.exists():
        last = {}
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                last[r["seed"]] = r["success"]
        done = {s for s, ok in last.items() if ok}
    seeds = [s for s in range(a.start_seed, a.start_seed + a.n) if s not in done]
    print(f"{len(seeds)} seeds to run ({len(done)} already done)", flush=True)
    jobs = [(s, str(out), not a.noforce, a.class_set) for s in seeds]
    with get_context("spawn").Pool(a.workers) as pool, open(log, "a") as f:
        for r in pool.imap_unordered(work, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(
                f"seed {r['seed']} {r['shape_class']}: success {r['success']} coverage {r['coverage']:.3f} gouge {r['gouge']:.3f} "  # noqa: E501
                f"frames {r['frames']} wall {r['wall_s']} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
