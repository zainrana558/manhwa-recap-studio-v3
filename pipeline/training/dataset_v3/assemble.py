#!/usr/bin/env python3
"""Merge the per-source YOLO label sets into one training dataset.

Sources (each: images/ + labels/, class 0 = panel):
  - yolo_kumiko/        Kumiko on the 4 bordered-manga series  (independent)
  - yolo/               Gemini flash-lite on webtoon pages     (independent)
  - yolo_webtoon_slicer/ pipeline slicer on webtoon pages      (optional, opt-in)

Geometric QC + per-series cap + series-held-out val split. Output: a flat
YOLO dir ready to zip for Kaggle.

    python assemble.py --out train_ready [--include-slicer]
"""
import argparse
import glob
import hashlib
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
VAL_SERIES = {"tbate"}          # one series held out entirely (style-generalisation check)
VAL_FRAC = 0.10                 # + a slice of every other series
CAP = 400


def read_boxes(lp):
    out = []
    if Path(lp).exists():
        for ln in Path(lp).read_text().split("\n"):
            f = ln.split()
            if len(f) == 5:
                out.append([float(x) for x in f[1:]])  # cx cy w h
    return out


def ok_box(b):
    cx, cy, w, h = b
    if w < 0.12 or h < 0.010 or w > 1.02 or h > 1.02:
        return False
    if w * h > 0.99:
        return False
    return True


def phash(im):
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (16, 16))
    return hashlib.md5((g > g.mean()).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "train_ready"))
    ap.add_argument("--include-slicer", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    srcs = [HERE / "yolo_kumiko", HERE / "yolo"]
    if args.include_slicer:
        srcs.append(HERE / "yolo_webtoon_slicer")

    rng = random.Random(0)
    by_series = {}
    for sd in srcs:
        for ip in glob.glob(f"{sd}/images/*"):
            stem = Path(ip).stem
            series = stem.split("__")[0]
            by_series.setdefault(series, []).append((sd, ip, stem))

    seen = set()
    kept = dropped = 0
    per_series = {}
    for series, items in sorted(by_series.items()):
        rng.shuffle(items)
        items = items[:CAP]
        for sd, ip, stem in items:
            im = cv2.imread(ip)
            if im is None:
                dropped += 1
                continue
            H, W = im.shape[:2]
            hh = phash(im)
            if hh in seen:
                dropped += 1
                continue
            boxes = [b for b in read_boxes(f"{sd}/labels/{stem}.txt") if ok_box(b)]
            # coverage sanity: a page with real ink but no boxes is a miss -> skip
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            inky = float((g < 235).mean()) > 0.05
            if not boxes and inky:
                dropped += 1
                continue
            seen.add(hh)
            split = "val" if (series in VAL_SERIES or rng.random() < VAL_FRAC) else "train"
            name = f"{stem}"[:120]
            shutil.copy(ip, out / "images" / split / f"{name}.jpg")
            with open(out / "labels" / split / f"{name}.txt", "w") as f:
                for cx, cy, w, h in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            kept += 1
            per_series[series] = per_series.get(series, 0) + 1

    ntr = len(glob.glob(f"{out}/images/train/*"))
    nva = len(glob.glob(f"{out}/images/val/*"))
    import yaml
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                    "nc": 1, "names": ["panel"]}, open(out / "data.yaml", "w"))
    print(f"kept {kept}  dropped {dropped}  | train {ntr}  val {nva}")
    for s, n in sorted(per_series.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
