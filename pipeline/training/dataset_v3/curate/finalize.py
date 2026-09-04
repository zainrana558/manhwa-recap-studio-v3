#!/usr/bin/env python3
"""Final merge+clean pass (fast, CV-only, no ONNX).
 - merge small / SFX-only / low-content boxes into an adjacent panel when the
   union stays sane (<=3.2 screen-heights, boxes are actually adjacent)
 - drop what can't be merged
   -> pipeline/training/dataset_v3/yolo_final
"""
import glob, os, shutil, sys
import cv2
import numpy as np

TIERS = ["pipeline/training/dataset_v3/yolo_gdrive",
         "pipeline/training/dataset_v3/yolo_webtoon_cascade_qc"]
OUT = "pipeline/training/dataset_v3/yolo_final"
DROP_SERIES = {"reaper-drifting-moon"}
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
wi, wn = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)


def bubble_rows(g, W, H):
    """fast bubble/caption row proxy: bright blobs sized like a bubble."""
    b = cv2.morphologyEx((g > 235).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
    rows = np.zeros(H, bool)
    scr = W * W
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if 0.004 * scr < ar < 0.35 * scr and 0.08 * W < w < 0.92 * W and 18 < h < 0.5 * W:
            rows[y:y + h] = True
    return rows


def classify(crop):
    h, w = crop.shape[:2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mid = ((g > 22) & (g < 233)).mean()
    lap = cv2.Laplacian(cv2.resize(g, (128, 128)), cv2.CV_32F).var()
    small = h < 0.62 * w or h * w < 62000
    dark = (g < 45).mean()
    lite = (g > 240).mean()
    sfx = (mid < 0.34) and (lap < 130) and (dark + lite > 0.55)
    weak = mid < 0.24 or lap < 45
    # divider / episode card: one flat colour dominates OR a tiny centred
    # text+icon block on a smooth pastel field
    hist = cv2.calcHist([g], [0], None, [16], [0, 256]).flatten()
    if hist.max() / g.size > 0.80:
        return "card"
    # colour-abstract 'energy + caption' beat: little edge detail, and the frame
    # is dominated by one hue with a bright text band
    hsv = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    if lap < 95 and hue_hist.max() / (96 * 96) > 0.55 and (g > 235).mean() > 0.06:
        return "abstract"
    if sfx:
        return "sfx"
    if weak:
        return "weak"
    if small:
        return "small"
    return "ok"


kept_p = kept_b = merged_n = dropped_n = 0
allp = [(D, lp) for D in TIERS for lp in sorted(glob.glob(f"{D}/labels/*.txt"))]
allp = [x for i, x in enumerate(allp) if i % wn == wi]

for D, lp in allp:
    stem = os.path.basename(lp)[:-4]
    if stem.split("__")[0] in DROP_SERIES:
        continue
    ip = next((f"{D}/images/{stem}{e}" for e in (".jpg", ".png", ".webp")
               if os.path.exists(f"{D}/images/{stem}{e}")), None)
    im = cv2.imread(ip) if ip else None
    if im is None:
        continue
    H, W = im.shape[:2]
    gfull = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    brow = bubble_rows(gfull, W, H)

    B = []
    for ln in open(lp):
        f = ln.split()
        if len(f) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in f[1:])
        B.append([int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                  int((cx + bw / 2) * W), int((cy + bh / 2) * H)])
    B.sort(key=lambda b: b[1])
    if not B:
        continue

    MAXH = 3.2 * W
    GAP = 0.55 * W
    changed = True
    while changed and len(B) > 1:
        changed = False
        cls = [classify(im[max(0, b[1]):b[3], max(0, b[0]):b[2]]) for b in B]
        for i, c in enumerate(cls):
            if c == "ok":
                continue
            cand = []
            if i > 0:
                cand.append(i - 1)
            if i + 1 < len(B):
                cand.append(i + 1)
            # SFX/small prefer merging UP (into preceding beat)
            cand.sort(key=lambda j: (j > i, abs(B[j][1] - B[i][1])))
            for j in cand:
                lo, hi = min(i, j), max(i, j)
                gap = B[hi][1] - B[lo][3]
                nh = B[hi][3] - B[lo][1]
                if gap < GAP and nh < MAXH:
                    B[lo] = [min(B[lo][0], B[hi][0]), B[lo][1],
                             max(B[lo][2], B[hi][2]), B[hi][3]]
                    del B[hi]
                    merged_n += 1
                    changed = True
                    break
            if changed:
                break

    # anything still weak/sfx/small and unmergeable -> drop
    final = []
    for b in B:
        if classify(im[max(0, b[1]):b[3], max(0, b[0]):b[2]]) == "ok":
            final.append(b)
        else:
            dropped_n += 1
    if not final:
        continue
    shutil.copy(ip, f"{OUT}/images/{stem}{os.path.splitext(ip)[1]}")
    with open(f"{OUT}/labels/{stem}.txt", "w") as fo:
        for x1, y1, x2, y2 in sorted(set(map(tuple, final))):
            fo.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        kept_b += len(set(map(tuple, final)))
    kept_p += 1

print(f"[w{wi}] kept_pages={kept_p} kept_boxes={kept_b} merges={merged_n} drops={dropped_n}", flush=True)
