#!/usr/bin/env python3
"""Free webtoon panel labels via the pipeline's own detector+splitter.

Runs _detect_page_panels (trained YOLO + borderless-column CV split + XY-cut)
on every page of the 11 borderless webtoon series and writes YOLO labels.
Self-distillation flavour, but on 11 series the v2 model never trained on, and
heavily QC'd downstream. Manga stays with Kumiko.

    .venv/bin/python3 pipeline/training/dataset_v3/slice_webtoons.py
"""
import glob
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/
os.environ.setdefault("RECAP_OCR_FRAMES", "0")
import master_pipeline as mp  # noqa: E402


def _panels(rgb_arr):
    return mp._detect_page_panels(Image.fromarray(rgb_arr))

HERE = Path(__file__).parent
SRC = HERE / "sources"
OUT = HERE / "yolo_webtoon_slicer"
MANGA = {"chainsaw-man", "one-piece", "berserk", "jujutsu-kaisen"}
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TILE = 2200          # split very tall strips so the 1024 detector isn't starved
OVL = 400


def detect_tall(img_rgb):
    H, W = img_rgb.shape[:2]
    if H <= int(TILE * 1.3):
        return _panels(img_rgb)
    boxes = []
    y = 0
    while y < H:
        yb = min(H, y + TILE)
        for (x1, t1, x2, t2) in _panels(img_rgb[y:yb]):
            boxes.append((x1, y + t1, x2, y + t2))
        if yb >= H:
            break
        y += TILE - OVL
    # merge boxes duplicated across the tile seam
    boxes.sort(key=lambda b: b[1])
    merged = []
    for b in boxes:
        if merged:
            m = merged[-1]
            xov = (min(m[2], b[2]) - max(m[0], b[0])) / max(1, min(m[2]-m[0], b[2]-b[0]))
            if xov > 0.6 and b[1] - m[3] < 40:
                merged[-1] = (min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3]))
                continue
        merged.append(tuple(b))
    return merged


def main():
    for d in ("images", "labels", "overlays"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    pages = []
    for ser_dir in sorted(SRC.iterdir()):
        if not ser_dir.is_dir() or ser_dir.name in MANGA:
            continue
        for p in sorted(ser_dir.rglob("*")):
            if p.suffix.lower() in EXTS:
                pages.append((ser_dir.name, p))
    print(f"{len(pages)} webtoon pages / {len(set(s for s, _ in pages))} series", flush=True)

    t0 = time.time()
    done = fail = tot = nov = 0
    for i, (ser, p) in enumerate(pages):
        stem = f"{ser}__{p.parent.name}__{p.stem}"
        lp = OUT / "labels" / f"{stem}.txt"
        if lp.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            fail += 1
            continue
        H, W = im.shape[:2]
        try:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            boxes = detect_tall(rgb)
        except Exception as e:
            fail += 1
            if fail <= 8:
                print(f"  ERR {stem}: {repr(e)[:140]}", flush=True)
            continue
        clean = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if (x2 - x1) < 0.12 * W or (y2 - y1) < 0.012 * H:
                continue
            if (x2 - x1) * (y2 - y1) > 0.99 * W * H and len(boxes) > 1:
                continue
            clean.append((x1, y1, x2, y2))
        cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(lp, "w") as f:
            for x1, y1, x2, y2 in clean:
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        tot += len(clean)
        done += 1
        if nov < 80 and clean:
            ov = im.copy()
            for x1, y1, x2, y2 in clean:
                cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 4)
            sc = 1100 / max(ov.shape[0], 1)
            if sc < 1:
                ov = cv2.resize(ov, (int(ov.shape[1] * sc), int(ov.shape[0] * sc)))
            cv2.imwrite(str(OUT / "overlays" / f"{stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 78])
            nov += 1
        if (i + 1) % 50 == 0:
            el = (time.time() - t0) / 60
            print(f"  {i+1}/{len(pages)} done={done} fail={fail} boxes={tot} "
                  f"{el:.0f}min ({done/max(el,.1):.0f}/min)", flush=True)

    print(f"DONE done={done} fail={fail} boxes={tot} ({tot/max(done,1):.1f}/pg)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
