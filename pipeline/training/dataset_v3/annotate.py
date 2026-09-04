#!/usr/bin/env python3
"""Panel-annotate scraped source pages with Gemini -> YOLO labels.

Tall webtoon pages are tiled (Gemini downsamples a 5000 px strip to a few
hundred px and only sees 3-4 blobs), boxes lifted back to page space and
merged across tile seams. Output: a YOLO dataset (class 0 = panel).

  python annotate.py --src sources --out yolo [--model gemini-flash-lite-latest]
                     [--tile 1500 --overlap 350] [--limit N] [--keys k1,k2]

Resumable: skips pages that already have a label file. Rotates through keys
on 429/RESOURCE_EXHAUSTED.
"""
import argparse
import base64
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

PROMPT = (
    "This is a vertical slice of a full-colour Korean webtoon / manhwa page. "
    "Panels are frequently BORDERLESS — separated only by blank gutter space, a "
    "thin line, or a change of shot, with no drawn frame.\n"
    "Detect EVERY distinct STORY PANEL. A panel = one camera shot / one visual "
    "beat. Be GRANULAR:\n"
    "- If the art stacks several different shots vertically (different "
    "framing, angle, subject, or a time cut) box EACH one separately, even "
    "when the background colour runs continuously between them.\n"
    "- Only merge into one box when it is truly a single unbroken illustration.\n"
    "- EXCLUDE blank gutter space above/below/between panels — box only drawn art.\n"
    "- A speech bubble / caption belongs to the panel it overlaps; never box "
    "text on its own.\n"
    "- Ignore chapter-title cards, credit lines, author notes, ads, page "
    "numbers, site watermarks.\n"
    "- If a panel is cut off at the top or bottom edge of this slice, still box "
    "the visible part.\n"
    "Typical dense webtoon slice has 3-7 panels.\n"
    'Return ONLY a JSON array: [{"box_2d":[ymin,xmin,ymax,xmax]}], coords '
    "normalised 0-1000, ordered top-to-bottom. No prose, no code fence."
)


def load_keys(arg):
    if arg:
        return [k.strip() for k in arg.split(",") if k.strip()]
    kf = HERE / ".gemini_key"
    if kf.exists():
        return [l.strip() for l in kf.read_text().splitlines() if l.strip()]
    return [os.environ["GEMINI_API_KEY"]]


class Gemini:
    def __init__(self, keys, model):
        self.keys = keys
        self.model = model
        self.i = 0
        self.calls = 0

    def _url(self):
        k = self.keys[self.i % len(self.keys)]
        return (f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent?key={k}")

    def detect(self, img_bgr):
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf.tobytes()).decode()
        body = json.dumps({
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": PROMPT},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096,
                                 "responseMimeType": "application/json"},
        }).encode()
        last = None
        for att in range(6):
            try:
                req = urllib.request.Request(
                    self._url(), data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=100) as r:
                    j = json.load(r)
                self.calls += 1
                cand = (j.get("candidates") or [{}])[0]
                txt = "".join(p.get("text", "") for p in
                              cand.get("content", {}).get("parts", []))
                m = re.search(r"\[.*\]", txt, re.S)
                arr = json.loads(m.group(0)) if m else []
                out = []
                for it in arr:
                    b = it.get("box_2d") or it.get("box") or it.get("bbox")
                    if b and len(b) == 4:
                        out.append([float(x) for x in b])
                return out
            except urllib.error.HTTPError as e:
                msg = e.read()[:180].decode("utf-8", "replace")
                last = f"HTTP {e.code} {msg}"
                if e.code in (429, 403) and len(self.keys) > 1:
                    self.i += 1
                    time.sleep(2)
                elif e.code == 429:
                    time.sleep(20 * (att + 1))
                elif e.code >= 500:
                    time.sleep(6 * (att + 1))
                else:
                    break
            except Exception as e:
                last = repr(e)
                time.sleep(4 * (att + 1))
        raise RuntimeError(last)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_boxes(boxes, H, W):
    """Merge tile-seam duplicates + vertically-adjacent fragments of one panel."""
    boxes = sorted(boxes, key=lambda b: b[1])  # by y
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            a = list(boxes[i])
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b = boxes[j]
                # near-duplicate
                dup = iou(a, b) > 0.55
                # x-overlap high AND y-gap small -> same panel split across a seam
                xov = (min(a[2], b[2]) - max(a[0], b[0])) / max(1, min(a[2] - a[0], b[2] - b[0]))
                ygap = b[1] - a[3]
                frag = xov > 0.6 and -20 < ygap < max(18, 0.02 * H)
                if dup or frag:
                    a[0], a[1] = min(a[0], b[0]), min(a[1], b[1])
                    a[2], a[3] = max(a[2], b[2]), max(a[3], b[3])
                    used[j] = True
                    changed = True
            out.append(a)
            used[i] = True
        boxes = out
    # drop degenerate / full-page / tiny
    clean = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        if w < 0.15 * W or h < 0.012 * H:
            continue
        if w * h > 0.985 * W * H:
            continue
        clean.append([max(0, x1), max(0, y1), min(W, x2), min(H, y2)])
    return clean


def annotate_page(g, path, tile, overlap):
    im = cv2.imread(str(path))
    if im is None:
        return None, "unreadable"
    H, W = im.shape[:2]
    all_boxes = []
    if H <= int(tile * 1.35):
        ys = [(0, H)]
    else:
        ys, y = [], 0
        step = tile - overlap
        while y < H:
            ys.append((y, min(H, y + tile)))
            if y + tile >= H:
                break
            y += step
    for (ya, yb) in ys:
        crop = im[ya:yb]
        ch, cw = crop.shape[:2]
        for nb in g.detect(crop):
            ymin, xmin, ymax, xmax = nb
            x1 = xmin / 1000 * cw
            x2 = xmax / 1000 * cw
            y1 = ya + ymin / 1000 * ch
            y2 = ya + ymax / 1000 * ch
            if x2 - x1 > 3 and y2 - y1 > 3:
                all_boxes.append([x1, y1, x2, y2])
    boxes = merge_boxes(all_boxes, H, W)
    return (im, boxes), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "sources"))
    ap.add_argument("--out", default=str(HERE / "yolo"))
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--tile", type=int, default=1500)
    ap.add_argument("--overlap", type=int, default=350)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys", default="")
    ap.add_argument("--overlays", type=int, default=40)
    ap.add_argument("--shard", default="0/1", help="i/n — process only pages where idx%n==i")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    g = Gemini(load_keys(args.keys), args.model)

    pages = []
    for series in sorted(Path(args.src).iterdir()):
        if not series.is_dir():
            continue
        for p in sorted(series.rglob("*")):
            if p.suffix.lower() in EXTS:
                pages.append((series.name, p))
    random.Random(0).shuffle(pages)
    if args.limit:
        pages = pages[:args.limit]
    if sn > 1:
        pages = [p for k, p in enumerate(pages) if k % sn == si]
    print(f"{len(pages)} pages to annotate with {args.model} "
          f"({len(g.keys)} key(s))", flush=True)

    done = skip = fail = total_boxes = 0
    for idx, (series, p) in enumerate(pages):
        stem = f"{series}__{p.parent.name}__{p.stem}"
        lp = out / "labels" / f"{stem}.txt"
        if lp.exists():
            skip += 1
            continue
        try:
            res, err = annotate_page(g, p, args.tile, args.overlap)
            if err:
                fail += 1
                continue
            im, boxes = res
            H, W = im.shape[:2]
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), im,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            lines = []
            for x1, y1, x2, y2 in boxes:
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lp.write_text("\n".join(lines) + ("\n" if lines else ""))
            total_boxes += len(boxes)
            done += 1
            if done <= args.overlays:
                ov = im.copy()
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(ov, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), 4)
                sc = 900 / max(ov.shape[0], 1)
                if sc < 1:
                    ov = cv2.resize(ov, (int(ov.shape[1] * sc), int(ov.shape[0] * sc)))
                cv2.imwrite(str(out / "overlays" / f"{stem}.jpg"), ov,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:
            fail += 1
            print(f"  FAIL {stem}: {e}", flush=True)
            time.sleep(3)
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(pages)}  done={done} skip={skip} fail={fail} "
                  f"boxes={total_boxes} calls={g.calls}", flush=True)

    print(f"\nDONE: {done} annotated ({total_boxes} boxes, "
          f"{total_boxes/max(done,1):.1f}/page), {skip} skipped, {fail} failed, "
          f"{g.calls} API calls", flush=True)


if __name__ == "__main__":
    sys.exit(main())
