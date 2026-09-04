#!/usr/bin/env python3
"""Recover ground-truth panel boxes by matching recap frames to source pages.

Given the editor-cropped panel frames from extract_recap_frames.py and the raw
chapter images they were made from, template-match each frame into the vertically
concatenated chapter strip. A confident match = that frame's rectangle IS a
human-chosen panel box -> emit it as a YOLO label on the source page.

    python align_frames_to_source.py --frames data/recap/<slug> \
        --source ~/chapters/<slug> --out data/webtoon-yolo/recap-aligned \
        [--min-score 0.55]

Needs opencv. Multi-scale, so the recap's resize doesn't matter.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_pages(src: Path):
    exts = {".webp", ".jpg", ".jpeg", ".png"}
    pages = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in exts:
            im = cv2.imread(str(p))
            if im is not None:
                pages.append((p, im))
    return pages


def _match(frame_gray, page_gray, scales=(0.35, 0.45, 0.55, 0.7, 0.85, 1.0, 1.2)):
    """Best (score, x1, y1, x2, y2) of frame within page over scales."""
    fh, fw = frame_gray.shape
    best = (0.0, 0, 0, 0, 0)
    ph, pw = page_gray.shape
    for s in scales:
        tw, th = int(fw * s), int(fh * s)
        if tw < 24 or th < 24 or tw > pw or th > ph:
            continue
        tmpl = cv2.resize(frame_gray, (tw, th))
        res = cv2.matchTemplate(page_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, mloc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, mloc[0], mloc[1], mloc[0] + tw, mloc[1] + th)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.55)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    pages = _load_pages(Path(args.source))
    if not pages:
        print("no source pages found")
        return 1
    pages_gray = [(p, im, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for p, im in pages]

    # accumulate boxes per page, then write once
    per_page: "dict[Path, list]" = {}
    frames = sorted(Path(args.frames).glob("frame_*.jpg"))
    hits = 0
    for fi, fp in enumerate(frames):
        fr = cv2.imread(str(fp))
        if fr is None:
            continue
        fg = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        best = (0.0, None, None)
        for (pp, pim, pg) in pages_gray:
            sc, x1, y1, x2, y2 = _match(fg, pg)
            if sc > best[0]:
                best = (sc, pp, (x1, y1, x2, y2))
        if best[0] >= args.min_score:
            per_page.setdefault(best[1], []).append(best[2])
            hits += 1
        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(frames)} frames, {hits} confident matches")

    for pp, boxes in per_page.items():
        im = cv2.imread(str(pp))
        H, W = im.shape[:2]
        stem = f"{pp.parent.name}__{pp.stem}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    print(f"\n{hits}/{len(frames)} frames aligned -> {len(per_page)} labelled source pages in {out}")
    print("  spot-check a few overlays before training (matchTemplate can lock onto a repeated background)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
