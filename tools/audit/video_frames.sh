#!/usr/bin/env bash
# Extract single frames at given timestamps from a video with ffmpeg.
# Usage: video_frames.sh <video.mp4> <out_prefix> t1 [t2 ...]   (seconds)
set -euo pipefail
v="$1"; pre="$2"; shift 2
for t in "$@"; do
  ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$t" -i "$v" -frames:v 1 "${pre}_t$(printf '%07.2f' "$t").png"
done
