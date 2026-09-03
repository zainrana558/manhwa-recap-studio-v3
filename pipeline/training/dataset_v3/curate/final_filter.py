#!/usr/bin/env python3
"""Final pass over the two curated tiers -> pipeline/training/dataset_v3/yolo_curated.
Drops the residuals the manual review caught: low-content 'black panel + one
bubble' crops, logo / title cards, barcode slivers. Drops reaper-drifting-moon
(dark, too many marginal crops)."""
import glob, os, shutil
import cv2
import numpy as np

TIERS = ["pipeline/training/dataset_v3/yolo_gdrive",
         "pipeline/training/dataset_v3/yolo_webtoon_cascade_qc"]
OUT = "pipeline/training/dataset_v3/yolo_curated"
DROP_SERIES = {"reaper-drifting-moon"}
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
REJ = "/tmp/review/final_rej"
for r in ("low_content", "sliver2", "logo_card", "dropped_series"):
    os.makedirs(f"{REJ}/{r}", exist_ok=True)


def verdict(crop):
    h, w = crop.shape[:2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mid = ((g > 20) & (g < 235)).mean()          # real tone, not pure black/white
    if mid < 0.18:
        return "low_content"
    if w / h > 3.6 or h / w > 5.0:
        return "sliver2"
    # logo / title card: one flat colour dominates + a compact centred bright block
    hist = cv2.calcHist([g], [0], None, [32], [0, 256]).flatten()
    if hist.max() / g.size > 0.72:
        return "logo_card"
    return None


kept_p = kept_b = 0
rej = {"low_content": 0, "sliver2": 0, "logo_card": 0, "dropped_series": 0}
merged = {}   # stem -> (imgpath, [boxes])
for D in TIERS:
    for lp in sorted(glob.glob(f"{D}/labels/*.txt")):
        stem = os.path.basename(lp)[:-4]
        ser = stem.split("__")[0]
        ip = f"{D}/images/{stem}.jpg"
        if not os.path.exists(ip):
            for e in (".png", ".webp", ".jpeg"):
                if os.path.exists(f"{D}/images/{stem}{e}"):
                    ip = f"{D}/images/{stem}{e}"
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        good = []
        for ln in open(lp):
            f = ln.split()
            if len(f) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in f[1:])
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            if ser in DROP_SERIES:
                rej["dropped_series"] += 1
                continue
            v = verdict(im[max(0, y1):y2, max(0, x1):x2])
            if v:
                rej[v] += 1
                cv2.imwrite(f"{REJ}/{v}/{stem}__{y1}.jpg", im[max(0, y1):y2, max(0, x1):x2],
                            [cv2.IMWRITE_JPEG_QUALITY, 60])
            else:
                good.append((x1, y1, x2, y2, W, H))
        if good:
            merged.setdefault(stem, [ip, []])[1].extend(good)

for stem, (ip, boxes) in merged.items():
    W, H = boxes[0][4], boxes[0][5]
    shutil.copy(ip, f"{OUT}/images/{stem}{os.path.splitext(ip)[1]}")
    with open(f"{OUT}/labels/{stem}.txt", "w") as f:
        for x1, y1, x2, y2, *_ in sorted(set(boxes)):
            f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        kept_b += len(set(boxes))
    kept_p += 1

import collections
bs = collections.Counter()
bp = collections.Counter()
for stem in merged:
    s = stem.split("__")[0]
    bp[s] += 1
    bs[s] += len(set(merged[stem][1]))
print(f"FINAL: {kept_p} pages, {kept_b} panels")
print(f"dropped: {rej}")
for s in sorted(bp):
    print(f"  {s:24s} {bp[s]:4d} pages  {bs[s]:4d} panels")
