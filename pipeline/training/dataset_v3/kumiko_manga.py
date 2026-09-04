#!/usr/bin/env python3
"""Panel-detect the BORDERED-manga series with Kumiko (CV contour walk).

Kumiko nails framed manga panels — tight on the frame lines, no gutter bleed,
no bubble boxes. Used only for the manga series; borderless webtoons go to the
VLM. Output: YOLO labels (class 0 = panel) + overlays, same layout as the
Qwen kernel so the two merge cleanly.

    python kumiko_manga.py --src sources --out yolo_kumiko \
        [--series chainsaw-man,one-piece,berserk,jujutsu-kaisen]
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2

KUMIKO = "/tmp/claude-1001/-home-ubuntu-manhwa-recap-studio-v3/78655276-e975-4673-a5ae-477a6153750c/scratchpad/kumiko"
sys.path.insert(0, KUMIKO)
from kumikolib import Kumiko  # noqa: E402

MANGA = ["chainsaw-man", "one-piece", "berserk", "jujutsu-kaisen"]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).parent / "sources"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "yolo_kumiko"))
    ap.add_argument("--series", default=",".join(MANGA))
    ap.add_argument("--overlays", type=int, default=60)
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("images", "labels", "overlays"):
        (out / d).mkdir(parents=True, exist_ok=True)
    series = args.series.split(",")

    pages = []
    for s in series:
        for p in sorted((Path(args.src) / s).rglob("*")):
            if p.suffix.lower() in EXTS:
                pages.append((s, p))
    print(f"{len(pages)} pages across {len(series)} manga series", flush=True)

    done = empty = fail = tot = n_ov = 0
    for i, (s, p) in enumerate(pages):
        stem = f"{s}__{p.parent.name}__{p.stem}"
        lp = out / "labels" / f"{stem}.txt"
        if lp.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            fail += 1
            continue
        H, W = im.shape[:2]
        try:
            k = Kumiko({"progress": False})
            k.parse_image(str(p))
            panels = k.get_infos()[0]["panels"]
        except Exception as e:
            fail += 1
            if fail < 10:
                print(f"  ERR {stem}: {repr(e)[:120]}", flush=True)
            continue

        boxes = []
        for (x, y, w, h) in panels:
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) < 0.05 * W or (y2 - y1) < 0.03 * H:
                continue
            if (x2 - x1) * (y2 - y1) > 0.99 * W * H and len(panels) > 1:
                continue
            boxes.append((x1, y1, x2, y2))

        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(lp, "w") as f:
            for x1, y1, x2, y2 in boxes:
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        tot += len(boxes)
        done += 1
        if not boxes:
            empty += 1
        if n_ov < args.overlays and boxes:
            ov = im.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 4)
            sc = 1100 / max(ov.shape[0], 1)
            if sc < 1:
                ov = cv2.resize(ov, (int(ov.shape[1] * sc), int(ov.shape[0] * sc)))
            cv2.imwrite(str(out / "overlays" / f"{stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
            n_ov += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pages)} done={done} empty={empty} fail={fail} "
                  f"boxes={tot}", flush=True)

    print(f"DONE {done} pages, {tot} boxes ({tot/max(done,1):.1f}/pg), "
          f"{empty} empty, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
