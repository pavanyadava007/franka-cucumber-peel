"""Summarise LeRobot meta folder. Usage: python -I lerobot_meta.py <dataset_root_containing_meta>"""
import json, sys, os, glob
import pandas as pd

root = sys.argv[1]
meta = os.path.join(root, "meta")
info = json.load(open(os.path.join(meta, "info.json")))
print("codebase_version:", info.get("codebase_version"), "robot_type:", info.get("robot_type"), "fps:", info.get("fps"))
for k in ["total_episodes", "total_frames", "total_tasks", "total_videos", "total_chunks", "chunks_size", "data_path", "video_path", "splits", "data_files_size_in_mb", "video_files_size_in_mb"]:
    if k in info:
        print(f"  {k}: {info[k]}")
print("features:")
for name, f in info["features"].items():
    names = f.get("names")
    vinfo = f.get("info") or f.get("video_info")
    extra = ""
    if vinfo:
        extra = " video:" + json.dumps({kk: vinfo.get(kk) for kk in ["video.height", "video.width", "video.codec", "video.fps", "video.is_depth_map"] if kk in vinfo})
    nm = json.dumps(names) if names is not None else ""
    if len(nm) > 600:
        nm = nm[:600] + "..."
    print(f"  {name}: dtype={f.get('dtype')} shape={f.get('shape')} names={nm}{extra}")
# tasks
tj = os.path.join(meta, "tasks.jsonl")
tp = os.path.join(meta, "tasks.parquet")
if os.path.exists(tj):
    print("tasks.jsonl:")
    for line in open(tj):
        print("  ", line.strip()[:500])
if os.path.exists(tp):
    df = pd.read_parquet(tp)
    print("tasks.parquet:")
    print(df.reset_index().to_string()[:3000])
# episodes
ej = os.path.join(meta, "episodes.jsonl")
if os.path.exists(ej):
    rows = [json.loads(l) for l in open(ej)]
    print(f"episodes.jsonl: {len(rows)} rows")
    lens = [r.get("length") for r in rows]
    print("  lengths min/max/sum:", min(lens), max(lens), sum(lens))
    tasks = {}
    for r in rows:
        for t in r.get("tasks", []):
            tasks[t] = tasks.get(t, 0) + 1
    print("  task strings per episode:", tasks)
    print("  first row:", json.dumps(rows[0])[:400])
for p in sorted(glob.glob(os.path.join(meta, "episodes", "*", "*.parquet"))):
    df = pd.read_parquet(p)
    print(f"episodes parquet {os.path.relpath(p, meta)}: {len(df)} rows; columns (first 40): {list(df.columns)[:40]}")
    if "length" in df:
        print("  lengths min/max/sum:", df["length"].min(), df["length"].max(), df["length"].sum())
    if "tasks" in df:
        from collections import Counter
        c = Counter()
        for t in df["tasks"]:
            for x in list(t):
                c[x] += 1
        print("  task strings per episode:", dict(c))
    vidcols = [c for c in df.columns if c.startswith("videos/") and (c.endswith("file_index") or c.endswith("from_timestamp") or c.endswith("to_timestamp") or c.endswith("chunk_index"))]
    cols = ["episode_index", "length"] + [c for c in df.columns if c.startswith("data/")] + vidcols[:8]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(6).to_string())
    print(df[cols].tail(3).to_string())
