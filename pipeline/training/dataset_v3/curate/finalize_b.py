#!/usr/bin/env python3
"""Option B: one tuned merge+clean pass over yolo_final3, codifying the manual
sheet review. A box is 'ok' only if it has real scene content; otherwise it is
merged UP into the preceding beat (union stays <=3.2 screens, boxes adjacent),
or dropped if unmergeable.  -> yolo_final4
"""
import glob, os, shutil, sys
import cv2
import numpy as np

IN = "pipeline/training/dataset_v3/yolo_final3"
OUT = "pipeline/training/dataset_v3/yolo_final4"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
REJ = "/tmp/review/rejB"
os.makedirs(REJ, exist_ok=True)
wi, wn = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
DUMP = "--dump" in sys.argv


def _sfx_frac(g):
    H, W = g.shape
    best = 0.0
    for m in ((g < 55).astype(np.uint8), (g > 236).astype(np.uint8)):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, w, hh, ar = st[i]
            if w > 0.93 * W or hh > 0.93 * H:
                continue
            best = max(best, ar / float(H * W))
    return best


def _regions(g):
    e = cv2.Canny(g, 60, 160)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(e, 8)
    return sum(1 for i in range(1, n) if st[i, 4] > 0.002 * g.size)


def classify(crop):
    """HIGH-PRECISION only: flag a box solely when it is *unambiguously* not a
    panel. Anything that could be a real (if dark / simple) beat -> 'ok'."""
    h, w = crop.shape[:2]
    if h < 0.42 * w or h * w < 46000:
        return "small"                       # genuine thin fragment
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gs = cv2.resize(g, (180, max(1, int(180 * h / w))))
    mid = ((g > 20) & (g < 235)).mean()
    hist = cv2.calcHist([g], [0], None, [24], [0, 256]).flatten()

    if hist.max() / g.size > 0.86:            # one flat tone almost everywhere
        return "card"
    if mid < 0.14:                            # essentially pure black/white field
        return "blank"
    if _sfx_frac(gs) > 0.28 and mid < 0.45:   # a huge SFX/word mass, sparse scene
        return "sfx"
    return "ok"


if DUMP:
    import random
    random.seed(1)
    labs = random.sample(glob.glob(f"{IN}/labels/*.txt"), 150)
    for lp in labs:
        stem = os.path.basename(lp)[:-4]
        ip = f"{IN}/images/{stem}.jpg"
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        for ln in open(lp):
            f = ln.split()
            cx, cy, bw, bh = (float(v) for v in f[1:])
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            c = im[max(0, y1):y2, max(0, x1):x2]
            v = classify(c)
            cv2.imwrite(f"{REJ}/{v}__{stem}__{y1}.jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 55])
    print("dumped to", REJ)
    sys.exit()

allp = [(D, lp) for D in [IN] for lp in sorted(glob.glob(f"{IN}/labels/*.txt"))]
allp = [x for i, x in enumerate(allp) if i % wn == wi]
kp = kb = mg = dr = 0
for D, lp in allp:
    stem = os.path.basename(lp)[:-4]
    ip = next((f"{IN}/images/{stem}{e}" for e in (".jpg", ".png", ".webp")
               if os.path.exists(f"{IN}/images/{stem}{e}")), None)
    im = cv2.imread(ip) if ip else None
    if im is None:
        continue
    H, W = im.shape[:2]
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
    MAXH, GAP = 3.2 * W, 0.55 * W
    changed = True
    while changed and len(B) > 1:
        changed = False
        cls = [classify(im[max(0, b[1]):b[3], max(0, b[0]):b[2]]) for b in B]
        for i, c in enumerate(cls):
            if c == "ok":
                continue
            cand = [j for j in (i - 1, i + 1) if 0 <= j < len(B)]
            cand.sort(key=lambda j: (j > i, abs(B[j][1] - B[i][1])))
            for j in cand:
                lo, hi = min(i, j), max(i, j)
                if B[hi][1] - B[lo][3] < GAP and B[hi][3] - B[lo][1] < MAXH:
                    B[lo] = [min(B[lo][0], B[hi][0]), B[lo][1],
                             max(B[lo][2], B[hi][2]), B[hi][3]]
                    del B[hi]
                    mg += 1
                    changed = True
                    break
            if changed:
                break
    final = [b for b in B if classify(im[max(0, b[1]):b[3], max(0, b[0]):b[2]]) == "ok"]
    dr += len(B) - len(final)
    if not final:
        continue
    shutil.copy(ip, f"{OUT}/images/{stem}{os.path.splitext(ip)[1]}")
    with open(f"{OUT}/labels/{stem}.txt", "w") as fo:
        for x1, y1, x2, y2 in sorted(set(map(tuple, final))):
            fo.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        kb += len(set(map(tuple, final)))
    kp += 1
print(f"[w{wi}] kept_pg={kp} kept_bx={kb} merges={mg} drops={dr}", flush=True)
