#!/usr/bin/env python3
"""Panel labels for the BORDERLESS webtoon series via a pure-CV gutter split.

Webtoons are structured around full-width blank horizontal bands between shots,
so cut there — the right tool for the medium, the way Kumiko is right for
framed manga. Bubble-cluster over-segmentation is suppressed by requiring a
THICK gutter and merging thin/short neighbours.

    .venv/bin/python3 pipeline/training/dataset_v3/webtoon_cv.py [--overlays 80]
"""
import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "sources"
OUT = HERE / "yolo_webtoon_cv"
MANGA = {"chainsaw-man", "one-piece", "berserk", "jujutsu-kaisen"}
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def split_webtoon(im):
    H, W = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cx0, cx1 = int(W * 0.08), int(W * 0.92)
    band = g[:, cx0:cx1]
    ink = (band < 245).mean(axis=1)
    dark = (band < 40).mean(axis=1)
    blank = (ink < 0.010) & (dark < 0.5)

    # a gutter must be a THICK blank band (>= 2.2% of H, min 26px) so the white
    # gaps *between speech bubbles* don't count
    min_gut = max(26, int(H * 0.022))
    cuts = [0]
    i = 0
    while i < H:
        if blank[i]:
            j = i
            while j < H and blank[j]:
                j += 1
            if j - i >= min_gut and i > 0 and j < H:
                cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    cuts.append(H)

    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        s = g[a:b, cx0:cx1]
        rows = np.where((s < 245).mean(axis=1) > 0.010)[0]
        if rows.size < 3:
            continue
        y1, y2 = a + int(rows[0]), a + int(rows[-1]) + 1
        col = g[y1:y2]
        cc = np.where((col < 245).mean(axis=0) > 0.02)[0]
        x1, x2 = (int(cc[0]), int(cc[-1]) + 1) if cc.size > 3 else (0, W)
        segs.append([x1, y1, x2, y2])

    # merge a run of short segments (bubble shards) with a neighbour, and drop
    # anything still tiny
    min_h = int(H * 0.028)
    merged = []
    for s in segs:
        if merged and (s[1] - merged[-1][3] < min_gut) and \
           (s[3] - s[1] < min_h or merged[-1][3] - merged[-1][1] < min_h):
            m = merged[-1]
            merged[-1] = [min(m[0], s[0]), m[1], max(m[2], s[2]), s[3]]
        else:
            merged.append(s)
    out = [s for s in merged
           if (s[3] - s[1]) >= max(24, int(H * 0.018)) and (s[2] - s[0]) >= 0.30 * W]
    # a page that is basically one image -> one box
    if not out and segs:
        y = [min(s[1] for s in segs), max(s[3] for s in segs)]
        x = [min(s[0] for s in segs), max(s[2] for s in segs)]
        if y[1] - y[0] > 0.1 * H:
            out = [[x[0], y[0], x[1], y[1]]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", type=int, default=80)
    args = ap.parse_args()
    for d in ("images", "labels", "overlays"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    pages = []
    for sd in sorted(SRC.iterdir()):
        if not sd.is_dir() or sd.name in MANGA:
            continue
        for p in sorted(sd.rglob("*")):
            if p.suffix.lower() in EXTS:
                pages.append((sd.name, p))
    print(f"{len(pages)} webtoon pages / {len(set(s for s, _ in pages))} series", flush=True)

    t0 = time.time()
    done = fail = tot = empty = nov = 0
    per_series = {}
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
            boxes = split_webtoon(im)
        except Exception:
            fail += 1
            continue
        cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(lp, "w") as f:
            for x1, y1, x2, y2 in boxes:
                f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                        f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        tot += len(boxes)
        done += 1
        per_series[ser] = per_series.get(ser, 0) + 1
        if not boxes:
            empty += 1
        if nov < args.overlays and boxes and i % 7 == 0:
            ov = im.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 4)
            s = 1100 / max(ov.shape[0], 1)
            if s < 1:
                ov = cv2.resize(ov, (int(ov.shape[1] * s), int(ov.shape[0] * s)))
            cv2.imwrite(str(OUT / "overlays" / f"{stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 78])
            nov += 1
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(pages)} done={done} boxes={tot} empty={empty} "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)

    print(f"DONE {done} pages, {tot} boxes ({tot/max(done,1):.1f}/pg), "
          f"{empty} empty, {fail} failed", flush=True)
    for s, n in sorted(per_series.items()):
        print(f"  {s}: {n}", flush=True)


if __name__ == "__main__":
    main()
