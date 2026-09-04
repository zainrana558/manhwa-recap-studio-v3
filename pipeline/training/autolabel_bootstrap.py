#!/usr/bin/env python3
"""Bootstrap YOLO panel labels for our own raw chapters.

Runs the pipeline's CURRENT panel + text/bubble detectors over every source
image and writes YOLO-format labels (`class cx cy w h`, normalised). The output
is meant to be opened in a labelling tool (Label Studio / CVAT / Roboflow) and
CORRECTED — roughly 20% of webtoon boxes will be wrong — then fed into training.

class 0 = panel   (from _detect_page_panels, which already includes the
                   borderless-column split + text-cluster split)
class 1 = bubble  (from the RT-DETR comic text/bubble detector)

Usage:
    python autolabel_bootstrap.py --chapters DIR [DIR ...] --out OUTDIR
      DIR = a folder of source page images (…/chapter_001/*.webp) or a parent of
            such folders.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import master_pipeline as mp  # noqa: E402


def _iter_images(root: Path):
    exts = {".webp", ".jpg", ".jpeg", ".png"}
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap images per chapter dir (0 = all)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    n_img = n_panel = n_bubble = 0
    for c in args.chapters:
        imgs = list(_iter_images(Path(c)))
        if args.limit:
            imgs = imgs[: args.limit]
        for ip in imgs:
            try:
                im = Image.open(ip).convert("RGB")
            except Exception as e:
                print(f"skip {ip}: {e}")
                continue
            W, H = im.size
            panels = mp._detect_page_panels(im)
            gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
            try:
                bubbles = mp._detect_text_boxes(gray)
            except Exception:
                bubbles = []

            stem = f"{ip.parent.name}__{ip.stem}"
            im.save(out / "images" / f"{stem}.jpg", quality=92)
            lines = []

            def _yolo(cls, x1, y1, x2, y2):
                # clamp to the image — detectors here grow boxes past the edge
                # for bubble overflow, which YOLO rejects as "out of bounds"
                x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
                y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
                if x2 - x1 < 3 or y2 - y1 < 3:
                    return None
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

            for b in panels:
                ln = _yolo(0, *b)
                if ln:
                    lines.append(ln)
                    n_panel += 1
            for b in bubbles:
                ln = _yolo(1, *b)
                if ln:
                    lines.append(ln)
                    n_bubble += 1
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_img += 1
            if n_img % 25 == 0:
                print(f"  {n_img} images…")

    print(f"\n{n_img} images -> {out}")
    print(f"  {n_panel} panel boxes, {n_bubble} bubble boxes  (CORRECT THESE before training)")
    print("  classes: 0=panel 1=bubble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
