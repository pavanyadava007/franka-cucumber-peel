"""Build docs/RESULTS.md from results/eval_*.jsonl and the data logs. Every number in RESULTS.md comes from here."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peel.shape import CLASS_SETS  # noqa: E402

METHODS = [
    ("expert_force", "Scripted expert, force feedback"),
    ("expert_noforce", "Scripted expert, no force feedback"),
    ("expert_force_nominal", "Scripted expert, force feedback, estimated geometry"),
    ("expert_noforce_nominal", "Scripted expert, no force feedback, estimated geometry"),
    ("dp", "Diffusion Policy (vision + pose, no force)"),
    ("rdp", "RDP (vision + pose + wrench)"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant pair counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def load(method, suffix=""):
    f = ROOT / "results" / f"eval_{method}{suffix}.jsonl"
    if not f.exists():
        return []
    rows = {}
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["seed"]] = r  # last result per seed wins
    return list(rows.values())


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def data_summary(out, raw_dir, label, classes, zarr_name):
    log = ROOT / "data" / raw_dir / "collect_log.jsonl"
    if not log.exists():
        return
    # a seed collected again (after a fix) overwrites its episode file, so the last log row per seed is the one used
    rows = list({r["seed"]: r for r in (json.loads(x) for x in log.read_text().splitlines() if x.strip())}.values())
    ok = [r for r in rows if r["success"]]
    out.append(
        f"- {label}: {len(rows)} scripted-expert episodes through the GELLO joint path, {len(ok)} successful and "
        f"used ({sum(r['frames'] for r in ok)} frames at 24 Hz). Per class (successful / recorded): "
        + ", ".join(
            f"{c} {sum(1 for r in ok if r.get('shape_class') == c)}/{sum(1 for r in rows if r.get('shape_class') == c)}"
            for c in classes
        )
        + "."
    )
    summ = ROOT / "data" / "rdp" / zarr_name / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        out.append(f"  RDP stroke dataset `{zarr_name}`: {s['clips']} single-stroke clips, {s['frames']} frames.")


def section(out, title, classes, suffix, note):
    res = {m: load(m, suffix) for m, _ in METHODS}
    if not any(res.values()):
        return
    out += [
        f"## {title}",
        "",
        note,
        "",
        "| Method | Episodes | Success | 95 % CI | Mean coverage | Mean gouge | Median peak force |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, name in METHODS:
        rs = res[m]
        if not rs:
            continue
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        cov = 100 * sum(r["coverage"] for r in rs) / len(rs)
        gou = 100 * sum(r["gouge"] for r in rs) / len(rs)
        out.append(
            f"| {name} | {len(rs)} | {k}/{len(rs)} ({100 * k / len(rs):.0f} %) | {100 * lo:.0f} to "
            f"{100 * hi:.0f} % | {cov:.1f} % | {gou:.2f} % | {med([r['peak_fn'] for r in rs]):.1f} N |"
        )
    for label, key, fmt in (("successes / episodes", "success", None), ("mean coverage", "coverage", "pct")):
        out += [
            "",
            f"By cucumber class ({label}):",
            "",
            "| Method | " + " | ".join(classes) + " |",
            "|---|" + "---|" * len(classes),
        ]
        for m, name in METHODS:
            rs = res[m]
            if not rs:
                continue
            cells = []
            for c in classes:
                rc = [r for r in rs if r["shape_class"] == c]
                if not rc:
                    cells.append("-")
                elif fmt:
                    cells.append(f"{100 * sum(r[key] for r in rc) / len(rc):.0f} %")
                else:
                    cells.append(f"{sum(r[key] for r in rc)}/{len(rc)}")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", "Paired comparisons on the same cucumbers (exact McNemar test on success):", ""]
    for a, b in (
        ("rdp", "dp"),
        ("expert_force", "expert_noforce"),
        ("expert_force_nominal", "expert_noforce_nominal"),
        ("rdp", "expert_force_nominal"),
    ):
        ra = {r["seed"]: r for r in res[a]}
        rb = {r["seed"]: r for r in res[b]}
        common = sorted(set(ra) & set(rb))
        if not common:
            continue
        only_a = sum(ra[s]["success"] and not rb[s]["success"] for s in common)
        only_b = sum(rb[s]["success"] and not ra[s]["success"] for s in common)
        out.append(
            f"- {a} vs {b}, {len(common)} cucumbers: {a} wins {only_a}, {b} wins {only_b}, "
            f"p = {mcnemar_exact(only_a, only_b):.3g}."
        )
    out.append("")


def main():
    out = [
        "# Results",
        "",
        "All numbers below are produced by `scripts/make_results.py` from `results/eval_*.jsonl`. They are "
        "**simulation** results on held-out cucumbers (never used for training), with the rule-based skin model "
        "described in SIMULATION.md. They are not real-robot results.",
        "",
        "Success = at least 90 % of the peelable skin removed, at most 3 % gouged, no handle collision. 95 % "
        "intervals are Wilson score intervals. 'Estimated geometry' means the scripted expert only knows a straight "
        "cylinder on the gripper axis with a radius guessed to about 1.5 mm, as a camera estimate would give.",
        "",
        "## Training data",
        "",
    ]
    data_summary(out, "raw", "v1 shapes", CLASS_SETS["v1"], "peel_stroke_zarr")
    data_summary(out, "raw_v2", "v2 real cucumber types", CLASS_SETS["v2"], "peel_stroke_v2_zarr")
    out.append("")
    section(
        out,
        "Test set v1 (first shapes, seeds 9000+)",
        CLASS_SETS["v1"],
        "",
        "Policies here were trained on v1 demos only.",
    )
    section(
        out,
        "Test set v2 (real cucumber types, seeds 10000+)",
        CLASS_SETS["v2"],
        "_v2",
        "Policies here were trained on v1 + v2 demos unless the method name says otherwise.",
    )
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
