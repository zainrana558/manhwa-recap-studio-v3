#!/usr/bin/env python3
"""Tighter final pass. Adds: higher mid-tone floor, flat/gradient reject,
caption-dominated reject, looser multi-beat.  yolo_curated2."""
import glob, os, shutil, sys
import cv2
import numpy as np
sys.path.insert(0, "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3")
os.environ.setdefault("WCV2_NO_FACE", "1")
import webtoon_cv2 as wc

TIERS = ["pipeline/training/dataset_v3/yolo_gdrive",
         "pipeline/training/dataset_v3/yolo_webtoon_cascade_qc"]
OUT = "pipeline/training/dataset_v3/yolo_curated2"
DROP_SERIES = {"reaper-drifting-moon"}
REJ = "/tmp/review/rej2"
for d in ("images", "labels"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
for r in ("low_content", "flat", "caption_dom", "sliver", "multi_beat"):
    os.makedirs(f"{REJ}/{r}", exist_ok=True)
wi, wn = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)


def verdict(crop, prot_rows):
    h, w = crop.shape[:2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mid = ((g > 22) & (g < 233)).mean()
    if mid < 0.28:
        return "low_content"
    if w / h > 3.4 or h / w > 4.6:
        return "sliver"
    # flat / gradient: very low local texture everywhere
    lap = cv2.Laplacian(cv2.resize(g, (128, 128)), cv2.CV_32F).var()
    if lap < 55:
        return "flat"
    # caption/bubble dominates the crop
    if prot_rows is not None and prot_rows.mean() > 0.42:
        return "caption_dom"
    # multi-beat: >=2 protected bands separated inside a tallish crop
    if prot_rows is not None and h / w > 2.1:
        runs, inr = 0, False
        for v in prot_rows:
            if v and not inr:
                runs += 1; inr = True
            elif not v:
                inr = False
        if runs >= 2:
            return "multi_beat"
    return None


kept_p = kept_b = 0
rej = {k: 0 for k in ("low_content", "flat", "caption_dom", "sliver", "multi_beat")}
allpages = []
for D in TIERS:
    for lp in sorted(glob.glob(f"{D}/labels/*.txt")):
        allpages.append((D, lp))
allpages = [x for i, x in enumerate(allpages) if i % wn == wi]

for D, lp in allpages:
    stem = os.path.basename(lp)[:-4]
    if stem.split("__")[0] in DROP_SERIES:
        continue
    ip = next((f"{D}/images/{stem}{e}" for e in (".jpg", ".png", ".webp")
               if os.path.exists(f"{D}/images/{stem}{e}")), None)
    im = cv2.imread(ip) if ip else None
    if im is None:
        continue
    H, W = im.shape[:2]
    bub, cap, face = wc._protected_rows(im, H, W)
    prot = bub | cap
    good = []
    for ln in open(lp):
        f = ln.split()
        if len(f) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in f[1:])
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        pr = prot[max(0, y1):min(H, y2)]
        v = verdict(im[max(0, y1):y2, max(0, x1):x2], pr)
        if v:
            rej[v] += 1
            cv2.imwrite(f"{REJ}/{v}/{stem}__{y1}.jpg", im[max(0, y1):y2, max(0, x1):x2],
                        [cv2.IMWRITE_JPEG_QUALITY, 55])
        else:
            good.append((x1, y1, x2, y2))
    if not good:
        continue
    shutil.copy(ip, f"{OUT}/images/{stem}{os.path.splitext(ip)[1]}")
    with open(f"{OUT}/labels/{stem}.txt", "w") as fo:
        for x1, y1, x2, y2 in sorted(set(good)):
            fo.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        kept_b += len(set(good))
    kept_p += 1

print(f"[w{wi}] kept_pages={kept_p} kept_boxes={kept_b} rej={rej}", flush=True)
