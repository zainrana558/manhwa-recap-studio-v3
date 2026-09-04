#!/usr/bin/env python3
"""QC the raw Gemini YOLO labels: geometric sanity, dedup, drop misfires.

  python qc.py --in yolo --out yolo_clean [--overlays 60]

Keeps a page when its boxes are geometrically plausible; keeps genuinely blank
pages (real whitespace, no ink) as empty-label hard-negatives; drops pages
where Gemini clearly misfired (coverage way off, huge overlaps, full-page box).
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_lbl(p):
    out = []
    if p.exists():
        for ln in p.read_text().split("\n"):
            f = ln.split()
            if len(f) == 5:
                out.append([float(x) for x in f[1:]])  # cx,cy,w,h
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = a[2]*a[3] + b[2]*b[3] - inter
    return inter/ua if ua > 0 else 0.0


def page_has_ink(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float((g < 235).mean()) > 0.04


def phash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16))
    return hashlib.md5((g > g.mean()).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(Path(__file__).parent / "yolo"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "yolo_clean"))
    ap.add_argument("--overlays", type=int, default=60)
    args = ap.parse_args()

    src, out = Path(args.inp), Path(args.out)
    for d in ("images", "labels", "overlays"):
        (out / d).mkdir(parents=True, exist_ok=True)

    imgs = sorted((src / "images").glob("*.jpg"))
    seen_hash = {}
    kept = dropped = blank_neg = 0
    reasons = {}
    n_ov = 0
    for ip in imgs:
        lp = src / "labels" / f"{ip.stem}.txt"
        im = cv2.imread(str(ip))
        if im is None:
            dropped += 1; reasons["unreadable"] = reasons.get("unreadable", 0)+1; continue
        H, W = im.shape[:2]
        boxes = read_lbl(lp)

        # perceptual-dedup (webtoon reposts / recap of same beat)
        h = phash(im)
        if h in seen_hash:
            dropped += 1; reasons["dup_page"] = reasons.get("dup_page", 0)+1; continue

        if not boxes:
            if page_has_ink(im):
                dropped += 1; reasons["empty_label_inky_page"] = reasons.get("empty_label_inky_page", 0)+1
                continue
            # true blank -> keep as hard negative
            seen_hash[h] = ip.stem
            shutil.copy(ip, out / "images" / ip.name)
            (out / "labels" / f"{ip.stem}.txt").write_text("")
            blank_neg += 1; kept += 1
            continue

        # geometric filters
        good = []
        for b in boxes:
            cx, cy, w, hh = b
            if w < 0.14 or hh < 0.012 or w*hh > 0.985 or w > 1.02 or hh > 1.02:
                continue
            good.append([min(max(cx, w/2), 1-w/2 if w < 1 else cx),
                         min(max(cy, hh/2), 1-hh/2 if hh < 1 else cy), w, hh])
        # drop pair with huge overlap (keep the larger)
        good.sort(key=lambda b: -b[2]*b[3])
        final = []
        for b in good:
            if all(iou(b, f) < 0.45 for f in final):
                final.append(b)
        if not final:
            dropped += 1; reasons["no_valid_box"] = reasons.get("no_valid_box", 0)+1; continue

        # coverage sanity
        grid = np.zeros((100, 100), bool)
        for cx, cy, w, hh in final:
            x1, y1 = int((cx-w/2)*100), int((cy-hh/2)*100)
            x2, y2 = int((cx+w/2)*100), int((cy+hh/2)*100)
            grid[max(0, y1):y2, max(0, x1):x2] = True
        cov = grid.mean()
        if cov < 0.18 or cov > 0.996:
            dropped += 1; reasons[f"coverage_{'lo' if cov<0.18 else 'hi'}"] = reasons.get(f"coverage_{'lo' if cov<0.18 else 'hi'}", 0)+1
            continue

        seen_hash[h] = ip.stem
        shutil.copy(ip, out / "images" / ip.name)
        (out / "labels" / f"{ip.stem}.txt").write_text(
            "\n".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in final) + "\n")
        kept += 1
        if n_ov < args.overlays:
            ov = im.copy()
            for cx, cy, w, hh in final:
                cv2.rectangle(ov, (int((cx-w/2)*W), int((cy-hh/2)*H)),
                              (int((cx+w/2)*W), int((cy+hh/2)*H)), (0, 0, 255), 4)
            s = 900/max(ov.shape[0], 1)
            if s < 1:
                ov = cv2.resize(ov, (int(ov.shape[1]*s), int(ov.shape[0]*s)))
            cv2.imwrite(str(out / "overlays" / ip.name), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
            n_ov += 1

    print(f"kept {kept} ({blank_neg} blank negatives), dropped {dropped}")
    print(f"keep-rate {100*kept/max(len(imgs),1):.0f}%")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  drop:{r} = {c}")


if __name__ == "__main__":
    main()
