#!/usr/bin/env python3
"""Standalone text-region cropper for the OCR-labelling pipeline.

Walks data/ocr_train_raw/<title>/chapter_*/*.{webp,jpg,png} , runs the ogkalu
comic-text-and-bubble detector (detector-v4-s_int8.onnx, RT-DETR-v2, INT8) with
the exact preprocessing master_pipeline._detect_text_boxes_raw uses, including
the tall-webtoon vertical-band split, and writes one tight crop per detected
text_bubble / text_free box.

Output: crops/<title>_c<ch>_p<page>_b<box>.jpg  (+ crops/index.tsv)

  .venv/bin/python crop_text_regions.py [--titles a,b] [--per-title 900] [--min-side 26]
"""
import os, sys, csv, glob, argparse
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

ROOT = "/home/azureuser/manhwa-recap-studio"
RAW = f"{ROOT}/data/ocr_train_raw"
DET = f"{ROOT}/pipeline/models/comic-text-and-bubble-detector/detector-v4-s_int8.onnx"
OUT = "/tmp/claude-1000/-home-azureuser/88ec5773-0abc-4c6b-afe1-a01b0a9ee495/scratchpad/crops"
S = 640
SPLIT_AR = 2.0            # tall-image band split, mirrors master_pipeline
SCORE = 0.35
PAD = 8

ap = argparse.ArgumentParser()
ap.add_argument("--titles", default="")
ap.add_argument("--per-title", type=int, default=1200)
ap.add_argument("--min-side", type=int, default=26)
ap.add_argument("--max-boxes-page", type=int, default=40)
args = ap.parse_args()

os.makedirs(OUT, exist_ok=True)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
sess = ort.InferenceSession(DET, sess_options=so, providers=["CPUExecutionProvider"])


def det_raw(rgb):
    """text_bubble(1) + text_free(2) boxes in rgb's own pixel coords."""
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (S, S)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    labels, boxes, scores = sess.run(
        None, {"images": x, "orig_target_sizes": np.array([[h, w]], dtype=np.int64)})
    out = []
    for lab, box, sc in zip(labels[0], boxes[0], scores[0]):
        if float(sc) < SCORE or int(lab) not in (1, 2):
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2), float(sc)))
    return out


def det_page(rgb):
    h, w = rgb.shape[:2]
    if w == 0 or h / max(1, w) <= SPLIT_AR:
        return det_raw(rgb)
    band_h = w * SPLIT_AR
    step = band_h - band_h * 0.2
    allb, y = [], 0.0
    while y < h:
        t = int(y); b = int(min(h, y + band_h))
        for (x1, y1, x2, y2, sc) in det_raw(rgb[t:b, :]):
            allb.append((x1, y1 + t, x2, y2 + t, sc))
        if b >= h:
            break
        y += step
    # de-dup near-identical boxes from overlapping bands
    kept = []
    for bx in sorted(allb, key=lambda z: -z[4]):
        if any(abs(bx[0]-k[0]) < 12 and abs(bx[1]-k[1]) < 12 and
               abs(bx[2]-k[2]) < 12 and abs(bx[3]-k[3]) < 12 for k in kept):
            continue
        kept.append(bx)
    return kept


titles = [t for t in (args.titles.split(",") if args.titles else
                      sorted(os.path.basename(p) for p in glob.glob(f"{RAW}/*") if os.path.isdir(p)))
          if t]
print("titles:", titles)

idx_path = f"{OUT}/index.tsv"
seen = set()
if os.path.exists(idx_path):
    for r in csv.reader(open(idx_path), delimiter="\t"):
        if r:
            seen.add(r[0])
idxf = open(idx_path, "a", newline="")
idx = csv.writer(idxf, delimiter="\t")

total = 0
for title in titles:
    pages = sorted(glob.glob(f"{RAW}/{title}/chapter_*/*.webp") +
                   glob.glob(f"{RAW}/{title}/chapter_*/*.jpg") +
                   glob.glob(f"{RAW}/{title}/chapter_*/*.png") +
                   glob.glob(f"{RAW}/{title}/chapter_*/*.jpeg"))
    tcount = 0
    for pg in pages:
        if tcount >= args.per_title:
            break
        ch = os.path.basename(os.path.dirname(pg)).replace("chapter_", "")
        pn = os.path.splitext(os.path.basename(pg))[0]
        try:
            im = Image.open(pg).convert("RGB")
        except Exception as e:
            print("  skip", pg, e); continue
        rgb = np.asarray(im)                       # true RGB, same as master_pipeline
        boxes = det_page(rgb)
        boxes = sorted(boxes, key=lambda z: -((z[2]-z[0])*(z[3]-z[1])))[:args.max_boxes_page]
        H, W = rgb.shape[:2]
        for bi, (x1, y1, x2, y2, sc) in enumerate(boxes):
            if min(x2-x1, y2-y1) < args.min_side:
                continue
            cx1, cy1 = max(0, x1-PAD), max(0, y1-PAD)
            cx2, cy2 = min(W, x2+PAD), min(H, y2+PAD)
            crop = im.crop((cx1, cy1, cx2, cy2))
            name = f"{title}_c{ch}_p{pn}_b{bi:02d}.jpg"
            if name in seen:
                continue
            crop.save(f"{OUT}/{name}", quality=92)
            idx.writerow([name, title, ch, pn, f"{sc:.2f}",
                          f"{cx1},{cy1},{cx2},{cy2}", f"{cx2-cx1}x{cy2-cy1}"])
            seen.add(name); tcount += 1; total += 1
    idxf.flush()
    print(f"{title}: {tcount} crops  ({len(pages)} pages)")
print(f"TOTAL: {total} crops -> {OUT}/")
