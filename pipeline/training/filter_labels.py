#!/usr/bin/env python3
"""Self-distillation filter: keep only the auto-labelled pages the current
detectors are clearly RIGHT about, so bootstrap noise doesn't poison training.

A page passes when:
  - 1..9 panel boxes
  - panels cover 55-99% of the page, pairwise IoU < 0.30 (no bad overlaps)
  - no panel wildly extreme aspect (h/w > 4.5 or w/h > 4.5 -> a failed split)
  - >= 65% of total bubble area lands inside some panel box
  - the page is actually a comic page (not near-blank)

    python filter_labels.py --in data/.../bootstrap --out data/.../clean [--copy]

Writes a filtered YOLO folder + prints keep rate. `--copy` hard-copies images;
default symlinks.
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _boxes(txt: Path):
    panels, bubbles = [], []
    if not txt.exists():
        return panels, bubbles
    for ln in txt.read_text().split("\n"):
        p = ln.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        b = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        (panels if c == 0 else bubbles).append(b)
    return panels, bubbles


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _inside_frac(bb, panels):
    ba = (bb[2] - bb[0]) * (bb[3] - bb[1])
    if ba <= 0:
        return 1.0
    best = 0.0
    for p in panels:
        ix1, iy1 = max(bb[0], p[0]), max(bb[1], p[1])
        ix2, iy2 = min(bb[2], p[2]), min(bb[3], p[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        best = max(best, inter / ba)
    return best


def passes(img: Path, txt: Path) -> bool:
    panels, bubbles = _boxes(txt)
    if not (1 <= len(panels) <= 9):
        return False
    # aspect sanity (need pixel dims for true aspect)
    try:
        W, H = Image.open(img).size
    except Exception:
        return False
    if W < 200 or H < 200:
        return False
    for (x1, y1, x2, y2) in panels:
        pw, ph = (x2 - x1) * W, (y2 - y1) * H
        if pw < 8 or ph < 8:
            return False
        r = max(pw, ph) / max(1.0, min(pw, ph))
        if r > 4.5:
            return False
    # coverage + overlap
    grid = np.zeros((100, 100), bool)
    for (x1, y1, x2, y2) in panels:
        grid[int(y1 * 100):int(y2 * 100), int(x1 * 100):int(x2 * 100)] = True
    cover = grid.mean()
    if not (0.55 <= cover <= 0.999):
        return False
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            if _iou(panels[i], panels[j]) > 0.30:
                return False
    # bubbles should sit inside panels
    if bubbles:
        tot = sum((b[2] - b[0]) * (b[3] - b[1]) for b in bubbles)
        ins = sum(_inside_frac(b, panels) * (b[2] - b[0]) * (b[3] - b[1]) for b in bubbles)
        if tot > 0 and ins / tot < 0.65:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    src = Path(args.inp)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    imgs = sorted((src / "images").glob("*"))
    kept = 0
    for ip in imgs:
        tp = src / "labels" / f"{ip.stem}.txt"
        if not passes(ip, tp):
            continue
        dst_i = out / "images" / ip.name
        if args.copy:
            shutil.copy(ip, dst_i)
        else:
            if dst_i.exists() or dst_i.is_symlink():
                dst_i.unlink()
            os.symlink(ip.resolve(), dst_i)
        shutil.copy(tp, out / "labels" / f"{ip.stem}.txt")
        kept += 1

    print(f"{kept}/{len(imgs)} pages kept ({100 * kept / max(1, len(imgs)):.0f}%) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
