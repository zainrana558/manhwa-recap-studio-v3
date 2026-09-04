#!/usr/bin/env python3
"""Extract distinct panel frames from a YouTube manhwa-recap video.

A recap channel (Asura Chronicles etc.) holds one human-cropped panel per beat.
This grabs one representative frame per hold, then trims the blurred backdrop /
letterbox so what's left is the panel crop the editor chose.

    python extract_recap_frames.py --url "<yt url>" --out data/recap/<slug> \
        [--fps 1] [--skip-intro 80] [--min-gap 2.0]

Needs: yt-dlp, ffmpeg, opencv (`pip install yt-dlp opencv-python`).

Output: <out>/frame_00001.jpg …  — each an editor-cropped panel (sharp region
only, backdrop removed). Feed to align_frames_to_source.py to recover boxes.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _download(url: str, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    mp4 = dst / "video.mp4"
    if not mp4.exists():
        subprocess.run(
            ["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]",
             "--merge-output-format", "mp4", "-o", str(mp4), url],
            check=True,
        )
    return mp4


def _sharp_region(frame: np.ndarray) -> "tuple[int,int,int,int] | None":
    """Bounding box of the in-focus foreground panel (the recap composites a
    sharp crop over a heavy-Gaussian-blur copy of itself). Local variance of
    Laplacian separates the two."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    local = cv2.blur(lap * lap, (25, 25))
    sharp = local > (local.max() * 0.06)
    sharp = cv2.morphologyEx(sharp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    ys, xs = np.where(sharp)
    if xs.size < g.size * 0.02:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 - x1 < 60 or y2 - y1 < 60:
        return None
    return x1, y1, x2, y2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0, help="sample rate for scene scan")
    ap.add_argument("--skip-intro", type=float, default=80.0, help="seconds to skip at start")
    ap.add_argument("--skip-outro", type=float, default=15.0)
    ap.add_argument("--scene-thresh", type=float, default=0.12, help="mean abs frame diff to call a new beat")
    args = ap.parse_args()

    out = Path(args.out)
    mp4 = _download(args.url, out)
    cap = cv2.VideoCapture(str(mp4))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / vfps
    step = max(1, int(round(vfps / args.fps)))

    prev = None
    saved = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / vfps
        fi += 1
        if fi % step:
            continue
        if t < args.skip_intro or t > total - args.skip_outro:
            continue
        small = cv2.resize(frame, (160, 90))
        if prev is not None:
            d = float(np.abs(small.astype(np.int16) - prev).mean()) / 255.0
            if d < args.scene_thresh:
                continue
        prev = small.astype(np.int16)
        box = _sharp_region(frame)
        crop = frame if box is None else frame[box[1]:box[3], box[0]:box[2]]
        if crop.shape[0] < 80 or crop.shape[1] < 80:
            continue
        saved += 1
        cv2.imwrite(str(out / f"frame_{saved:05d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
    cap.release()
    print(f"{saved} distinct panel frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
