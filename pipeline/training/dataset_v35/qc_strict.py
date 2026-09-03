#!/usr/bin/env python3
"""Strict cleanliness QC for the CV-cascade webtoon boxes.

"Less but perfect > more but flawed."  Keep a box only if every edge is a clean
boundary — content just inside, blank/gutter just outside (or the box sits on
the image border, i.e. full-bleed).  Keep a page only if all its boxes pass and
the box union covers most of the non-blank page (no whole panels missed).

    python qc_strict.py <in_dir> <out_dir>
      in_dir : a yolo_*/  (images/ + labels/)  single-class panel labels
"""
import glob, os, shutil, sys
import cv2
import numpy as np

BAND = 6            # px band to sample at each edge
EDGE_MARGIN = 4     # box within this many px of the image edge == at the border
IN_MIN = 0.06       # inside band must have at least this much content
OUT_MAX = 0.045     # outside band must have at most this much content (gutter)
BOX_CONTENT = (0.14, 0.995)
COVER_MIN = 0.70    # box union must cover this fraction of page content


def content_map(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # "content" = not near-white and not near-black flat gutter
    m = ((g > 18) & (g < 244)).astype(np.uint8)
    # a solid black gutter is also blank; treat long flat runs as blank via local std
    return m, g


def edge_clean(m, x1, y1, x2, y2, W, H):
    def frac(a):
        a = a[(a >= 0)]
        return float(m.flat[a].mean()) if a.size else 0.0
    for side in ("t", "b", "l", "r"):
        if side == "t":
            at_border = y1 <= EDGE_MARGIN
            ins = m[max(0, y1):y1 + BAND, x1:x2]
            out = m[max(0, y1 - BAND):y1, x1:x2]
        elif side == "b":
            at_border = y2 >= H - EDGE_MARGIN
            ins = m[y2 - BAND:y2, x1:x2]
            out = m[y2:min(H, y2 + BAND), x1:x2]
        elif side == "l":
            at_border = x1 <= EDGE_MARGIN
            ins = m[y1:y2, max(0, x1):x1 + BAND]
            out = m[y1:y2, max(0, x1 - BAND):x1]
        else:
            at_border = x2 >= W - EDGE_MARGIN
            ins = m[y1:y2, x2 - BAND:x2]
            out = m[y1:y2, x2:min(W, x2 + BAND)]
        if at_border:
            continue
        iv = ins.mean() if ins.size else 0
        ov = out.mean() if out.size else 0
        if iv < IN_MIN:            # nothing right inside the edge -> loose box
            return False
        if ov > OUT_MAX:           # content spills past the edge -> cuts through
            return False
    return True


def main():
    ind, outd = sys.argv[1], sys.argv[2]
    os.makedirs(f"{outd}/images", exist_ok=True)
    os.makedirs(f"{outd}/labels", exist_ok=True)
    kp = kb = pg = drp_box = drp_pg = 0
    for lp in sorted(glob.glob(f"{ind}/labels/*.txt")):
        stem = os.path.basename(lp)[:-4]
        ip = next((f"{ind}/images/{stem}{e}" for e in (".jpg", ".png", ".webp")
                   if os.path.exists(f"{ind}/images/{stem}{e}")), None)
        im = cv2.imread(ip) if ip else None
        if im is None:
            continue
        H, W = im.shape[:2]
        m, g = content_map(im)
        total_content = max(1.0, m.sum())
        good, bad = [], 0
        for ln in open(lp):
            q = ln.split()
            if len(q) < 5:
                continue
            cx, cy, bw, bh = map(float, q[1:5])
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 24 or y2 - y1 < 24:
                bad += 1
                continue
            c = m[y1:y2, x1:x2].mean()
            if not (BOX_CONTENT[0] <= c <= BOX_CONTENT[1]):
                bad += 1
                continue
            if not edge_clean(m, x1, y1, x2, y2, W, H):
                bad += 1
                continue
            good.append((x1, y1, x2, y2))
        if not good or bad > 0:            # strict: any bad box fails the page
            drp_pg += 1
            drp_box += bad
            continue
        cov = np.zeros((H, W), np.uint8)
        for x1, y1, x2, y2 in good:
            cov[y1:y2, x1:x2] = 1
        if (cov * m).sum() / total_content < COVER_MIN:   # missed a whole panel
            drp_pg += 1
            continue
        shutil.copy(ip, f"{outd}/images/{stem}{os.path.splitext(ip)[1]}")
        with open(f"{outd}/labels/{stem}.txt", "w") as fo:
            for x1, y1, x2, y2 in good:
                fo.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                         f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
                kb += 1
        kp += 1
    print(f"{ind}: kept {kp} pages / {kb} boxes   dropped {drp_pg} pages ({drp_box} bad boxes)")


if __name__ == "__main__":
    main()
