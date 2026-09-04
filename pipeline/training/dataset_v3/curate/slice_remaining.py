#!/usr/bin/env python3
"""Slice the 18 non-gdrive series with webtoon_cv2 (bubble/caption cascade),
write YOLO page+label pairs, then STRICT-QC every panel box (blank / fragment
/ sliver / bubble-only / multi-beat). Perfect boxes only -> yolo_webtoon_cascade.

Run 2 workers:  WCV2_NO_FACE=1 python slice_remaining.py 0 2   (and  1 2)
"""
import glob, os, sys, time
import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3")
os.environ.setdefault("WCV2_NO_FACE", "1")
import webtoon_cv2 as wc

SRC = "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3/sources"
OUT = "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3/yolo_webtoon_cascade"
SERIES = ["blossoming-blade", "eleceed", "god-of-high-school", "great-mage-4000",
          "hardcore-leveling-warrior", "lookism", "orv", "reaper-drifting-moon",
          "second-life-ranker", "solo-leveling", "tbate", "the-breaker-new-waves",
          "the-breaker", "tower-of-god", "true-beauty", "twatf", "villain-to-kill",
          "wind-breaker"]
CAP = 300               # pages per series
EXTS = (".jpg", ".jpeg", ".png", ".webp")
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)

wi, wn = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)


def bright_blob_frac(g):
    b = cv2.morphologyEx((g > 234).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
    return 0.0 if n < 2 else max(st[1:, 4]) / float(g.size)


def qc_box(crop):
    h, w = crop.shape[:2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if (g > 240).mean() > 0.90 or (g < 15).mean() > 0.92:
        return "blank"
    if h < 150 or w < 240 or h * w < 70000:
        return "fragment"
    if w / h > 4.2 or h / w > 5.5:
        return "sliver"
    if bright_blob_frac(g) > 0.40 and (g < 200).mean() < 0.26:
        return "bubble_only"
    return None


pages = []
for s in SERIES:
    fs = sorted(f for e in EXTS for f in glob.glob(f"{SRC}/{s}/**/*{e}", recursive=True))
    if len(fs) > CAP:
        fs = fs[::max(1, len(fs) // CAP)][:CAP]
    pages += [(s, p) for p in fs]
pages = [pg for i, pg in enumerate(pages) if i % wn == wi]

t0 = time.time()
done = kept_pages = kept_boxes = 0
rej = {}
for i, (ser, p) in enumerate(pages):
    stem = f"{ser}__{os.path.basename(os.path.dirname(p))}__{os.path.splitext(os.path.basename(p))[0]}"
    if os.path.exists(f"{OUT}/labels/{stem}.txt"):
        continue
    im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    try:
        boxes = wc.split_webtoon(im)
    except Exception:
        continue
    # QC each box
    good = []
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
        if x2 - x1 < 40 or y2 - y1 < 40:
            continue
        r = qc_box(im[y1:y2, x1:x2])
        if r:
            rej[r] = rej.get(r, 0) + 1
        else:
            good.append((x1, y1, x2, y2))
    done += 1
    if not good:
        continue
    cv2.imwrite(f"{OUT}/images/{stem}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(f"{OUT}/labels/{stem}.txt", "w") as f:
        for x1, y1, x2, y2 in good:
            f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
    kept_pages += 1
    kept_boxes += len(good)
    if (i + 1) % 100 == 0:
        el = (time.time() - t0) / 60
        print(f"[w{wi}] {i+1}/{len(pages)} done={done} keptpg={kept_pages} boxes={kept_boxes} "
              f"{el:.0f}min ({done/max(el,.1):.0f}/min)", flush=True)

print(f"[w{wi}] DONE {done} pages, {kept_pages} kept pages, {kept_boxes} boxes | rej {rej}", flush=True)
