#!/usr/bin/env python3
"""Second QC pass on yolo_webtoon_cascade: drop any panel box whose top or
bottom edge cuts THROUGH a speech bubble / caption (ogkalu detector on the
page). Drop manga-style series entirely. Perfect boxes only.

  WCV2_NO_FACE=1 python qc_cascade.py <wi> <wn>
"""
import glob, os, sys, time
import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3")
os.environ.setdefault("WCV2_NO_FACE", "1")
import webtoon_cv2 as wc

D = "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3/yolo_webtoon_cascade"
OUT = "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3/yolo_webtoon_cascade_qc"
DROP_SERIES = {"the-breaker", "the-breaker-new-waves"}   # manga-style, wrong tool
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)

wi, wn = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
labs = sorted(glob.glob(f"{D}/labels/*.txt"))
labs = [lp for i, lp in enumerate(labs) if i % wn == wi]

t0 = time.time()
seen = kept_pg = kept_bx = drop_bx = drop_series = 0
for k, lp in enumerate(labs):
    stem = os.path.basename(lp)[:-4]
    if stem.split("__")[0] in DROP_SERIES:
        drop_series += 1
        continue
    ip = f"{D}/images/{stem}.jpg"
    im = cv2.imread(ip)
    if im is None:
        continue
    H, W = im.shape[:2]
    rows = [ln.split() for ln in open(lp) if len(ln.split()) == 5]
    if not rows:
        continue
    seen += 1
    bub, cap, face = wc._protected_rows(im, H, W)
    prot = bub | cap
    good = []
    for r in rows:
        cx, cy, bw, bh = (float(v) for v in r[1:])
        y1 = int((cy - bh / 2) * H)
        y2 = int((cy + bh / 2) * H)
        x1 = int((cx - bw / 2) * W)
        x2 = int((cx + bw / 2) * W)
        bad = False
        for yy in (y1, y2):
            a, b = max(0, yy - 3), min(H, yy + 3)
            up = prot[max(0, yy - 22):yy].any()
            dn = prot[yy:min(H, yy + 22)].any()
            if up and dn and prot[a:b].any():
                bad = True
                break
        if bad:
            drop_bx += 1
        else:
            good.append((x1, y1, x2, y2))
    if not good:
        continue
    cv2.imwrite(f"{OUT}/images/{stem}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(f"{OUT}/labels/{stem}.txt", "w") as f:
        for x1, y1, x2, y2 in good:
            f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
    kept_pg += 1
    kept_bx += len(good)
    if (k + 1) % 150 == 0:
        el = (time.time() - t0) / 60
        print(f"[w{wi}] {k+1}/{len(labs)} seen={seen} keptpg={kept_pg} keptbx={kept_bx} "
              f"dropbx={drop_bx} {el:.0f}min", flush=True)

print(f"[w{wi}] DONE seen={seen} kept_pages={kept_pg} kept_boxes={kept_bx} "
      f"dropped_boxes={drop_bx} dropped_series_pages={drop_series}", flush=True)
