#!/usr/bin/env python3
"""Kaggle kernel: panel-annotate scraped webtoon/manga pages with Qwen3-VL-8B.

Reads pages from an attached Kaggle dataset, tiles tall pages, asks Qwen3-VL for
comic-panel boxes, merges tile-seam boxes, then a CV pass snaps each box to its
content bounding box and splits obvious internal gutters ("perfectly sliced").
Writes a YOLO dataset (class 0 = panel) + overlays to /kaggle/working.

Resumable: skips pages whose label file already exists (checkpoint the output
dir as a Kaggle dataset between runs if you hit the 12h wall).

GPU: T4 x2 or P100. Model in 4-bit (~7 GB).
"""
import gc
import glob
import json
import os
import re
import subprocess
import sys
import time


def _pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=False)


# Kaggle GPU kernels always get a P100 (Pascal / sm_60). The base image's torch
# dropped Pascal kernels -> every CUDA op fails "no kernel image available".
# Pin the last cu121 build that still ships sm_60, exactly like the train
# kernel. Qwen2.5-VL (not 3-VL) so we don't need bleeding-edge transformers.
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle"):
    _pip("torch==2.4.1", "torchvision==0.19.1",
         "--index-url", "https://download.pytorch.org/whl/cu121")
    _pip("-U", "transformers==4.49.0", "accelerate", "qwen-vl-utils", "pillow",
         "bitsandbytes==0.43.3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

IN_ROOTS = [
    "/kaggle/input/panel-source-pages",
    "/kaggle/input/webtoon-yolo",
    "/kaggle/input",
]
WORK_SRC = "/kaggle/tmp/src"
OUT = "/kaggle/working/yolo"
MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
LOAD_8BIT = os.environ.get("LOAD_8BIT", "1") == "1"
TILE = int(os.environ.get("TILE", "1400"))
OVERLAP = int(os.environ.get("OVERLAP", "340"))
MAX_MIN = int(os.environ.get("MAX_MIN", "660"))          # stop before the 12h wall
PER_SERIES_CAP = int(os.environ.get("PER_SERIES_CAP", "260"))  # balance the set
SHARD = os.environ.get("SHARD", "0/1")                   # "i/n" — process pages where idx%n==i
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

PROMPT = (
    "Outline the position of every individual comic panel in this manga / "
    "webtoon image and output the coordinates in JSON.\n"
    "A panel = one drawn picture / camera shot. In manga they usually have "
    "black frame lines; in Korean webtoons they are often borderless, separated "
    "only by blank space or a scene change. Box the ARTWORK of each panel "
    "tightly, one box per panel:\n"
    "- do NOT include the blank gutter between or around panels\n"
    "- a speech bubble counts as part of the panel it sits on — do not box "
    "bubbles or text by themselves\n"
    "- adjacent panels that touch still get separate boxes\n"
    "- skip title logos, credits, author notes, ads, page numbers, watermarks\n"
    'Output ONLY: [{"bbox_2d": [x1, y1, x2, y2], "label": "panel"}, ...] '
    "using pixel coordinates of this image, ordered top-to-bottom."
)


# ----------------------------------------------------------------------------
def load_model():
    import transformers
    from transformers import AutoProcessor
    MC = None
    for name in ("Qwen2_5_VLForConditionalGeneration",
                 "Qwen2VLForConditionalGeneration",
                 "AutoModelForImageTextToText"):
        try:
            MC = getattr(__import__("transformers", fromlist=[name]), name)
            break
        except (ImportError, AttributeError):
            continue
    print("transformers", transformers.__version__, "class", MC.__name__, flush=True)
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    dt = torch.bfloat16 if cap[0] >= 8 else torch.float16
    print(f"GPU cc {cap}, dtype {dt}", flush=True)
    # cap image tokens so a tall tile can't blow VRAM
    proc = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=1280 * 28 * 28)
    kw = dict(device_map="auto", attn_implementation="eager")
    if LOAD_8BIT and "7B" in MODEL_ID:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0)
        print("loading 8-bit", flush=True)
    else:
        kw["torch_dtype"] = dt
    model = MC.from_pretrained(MODEL_ID, **kw)
    model.eval()
    return model, proc


@torch.inference_mode()
def detect(model, proc, img_bgr):
    from PIL import Image
    rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": rgb}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[rgb], padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=1536, do_sample=False)
    gen = out[:, inp.input_ids.shape[1]:]
    txt = proc.batch_decode(gen, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)[0]
    if os.environ.get("DUMP_RAW"):
        print("RAW>>>", txt[:600].replace("\n", " "), flush=True)
    H, W = img_bgr.shape[:2]
    raw = []
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            for it in json.loads(m.group(0)):
                b = it.get("bbox_2d") or it.get("bbox") or it.get("box_2d") or it.get("box")
                if b and len(b) == 4:
                    raw.append([float(v) for v in b])
        except Exception:
            pass
    if not raw:  # Qwen native box tokens: (x1,y1),(x2,y2)
        for mm in re.finditer(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", txt):
            raw.append([float(x) for x in mm.groups()])
    boxes = []
    for b in raw:
        x1, y1, x2, y2 = b
        # Qwen3-VL usually emits absolute px; if it emitted 0-1000, rescale
        if max(x1, x2) <= 1000 and max(y1, y2) <= 1000 and (W > 1000 or H > 1000):
            x1, x2 = x1 / 1000 * W, x2 / 1000 * W
            y1, y2 = y1 / 1000 * H, y2 / 1000 * H
        boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    return boxes


# ----------------------------------------------------------------------------
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge(boxes, H, W):
    boxes = sorted(boxes, key=lambda b: b[1])
    changed = True
    while changed:
        changed = False
        res, used = [], [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            a = list(boxes[i])
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b = boxes[j]
                dup = _iou(a, b) > 0.55
                xov = (min(a[2], b[2]) - max(a[0], b[0])) / max(1, min(a[2]-a[0], b[2]-b[0]))
                ygap = b[1] - a[3]
                frag = xov > 0.6 and -25 < ygap < max(22, 0.02 * H)
                if dup or frag:
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    used[j] = changed = True
            res.append(a); used[i] = True
        boxes = res
    out = []
    for x1, y1, x2, y2 in boxes:
        if (x2 - x1) < 0.14 * W or (y2 - y1) < 0.012 * H or (x2-x1)*(y2-y1) > 0.985*W*H:
            continue
        out.append([max(0, x1), max(0, y1), min(W, x2), min(H, y2)])
    return out


def snap_and_split(im, box):
    """Trim white margins to content bbox; split on a big internal blank band.
    Returns a list of tightened sub-boxes."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return []
    g = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    ink_rows = (g < 232).mean(axis=1)
    ink_cols = (g < 232).mean(axis=0)
    rr = np.where(ink_rows > 0.015)[0]
    cc = np.where(ink_cols > 0.015)[0]
    if rr.size < 4 or cc.size < 4:
        return []
    ty1, ty2 = y1 + int(rr[0]), y1 + int(rr[-1]) + 1
    tx1, tx2 = x1 + int(cc[0]), x1 + int(cc[-1]) + 1
    # internal gutter split
    sub = cv2.cvtColor(im[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY)
    Hs = sub.shape[0]
    row_ink = (sub[:, int(sub.shape[1]*0.1):int(sub.shape[1]*0.9)] < 224).mean(axis=1)
    blank = row_ink < 0.012
    runs, i, m = [], 0, max(4, int(Hs * 0.03))
    while i < Hs:
        if blank[i]:
            j = i
            while j < Hs and blank[j]:
                j += 1
            if m < i and j < Hs - m and (j - i) >= max(46, int(Hs * 0.09)):
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return [[tx1, ty1, tx2, ty2]]
    parts, prev = [], 0
    for (a, b) in runs:
        if a - prev >= max(38, int(Hs * 0.10)):
            parts.append([tx1, ty1 + prev, tx2, ty1 + a])
        prev = b
    if Hs - prev >= max(38, int(Hs * 0.10)):
        parts.append([tx1, ty1 + prev, tx2, ty1 + Hs])
    return parts or [[tx1, ty1, tx2, ty2]]


# ----------------------------------------------------------------------------
def list_pages():
    import subprocess
    os.makedirs(WORK_SRC, exist_ok=True)
    # extract any source tarballs found in the attached datasets
    for root in IN_ROOTS:
        for tb in glob.glob(f"{root}/**/*sources*.tar.gz", recursive=True) + \
                  glob.glob(f"{root}/**/all_sources*.tar*", recursive=True):
            print(f"extracting {tb} ...", flush=True)
            subprocess.run(["tar", "xf", tb, "-C", WORK_SRC], check=False)
    scan_roots = [WORK_SRC] + [r for r in IN_ROOTS if os.path.isdir(r)]
    pages, seen = [], set()
    for root in scan_roots:
        for p in glob.glob(f"{root}/**/*", recursive=True):
            if os.path.splitext(p)[1].lower() not in EXTS:
                continue
            parts = p.split("/")
            rel = "__".join(parts[-3:])          # series__chapter__file
            if rel in seen or "/gemini_partial/" in p or "/images/" in p:
                continue
            seen.add(rel)
            pages.append((os.path.splitext(rel)[0], p))
    pages.sort()
    # manga (framed) is handled by Kumiko, not the VLM — skip those series here
    MANGA = {"chainsaw-man", "one-piece", "berserk", "jujutsu-kaisen"}
    if os.environ.get("WEBTOON_ONLY", "1") == "1":
        pages = [(r, p) for (r, p) in pages if r.split("__")[0] not in MANGA]
    # per-series cap for class/style balance (deterministic stride sample)
    import random as _r
    byser = {}
    for rel, p in pages:
        byser.setdefault(rel.split("__")[0], []).append((rel, p))
    capped = []
    for ser, lst in byser.items():
        if len(lst) > PER_SERIES_CAP:
            _r.Random(1).shuffle(lst)
            lst = sorted(lst[:PER_SERIES_CAP])
        capped.extend(lst)
    capped.sort()
    si, sn = (int(x) for x in SHARD.split("/"))
    if sn > 1:
        capped = [pp for k, pp in enumerate(capped) if k % sn == si]
    return capped


def main():
    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/labels", exist_ok=True)
    os.makedirs(f"{OUT}/overlays", exist_ok=True)
    pages = list_pages()
    print(f"{len(pages)} source pages", flush=True)
    model, proc = load_model()
    print("model loaded", flush=True)

    t0 = time.time()
    done = skipped = failed = tot_boxes = 0
    for k, (rel, path) in enumerate(pages):
        stem = os.path.splitext(rel)[0]
        lp = f"{OUT}/labels/{stem}.txt"
        if os.path.exists(lp):
            skipped += 1
            continue
        if (time.time() - t0) / 60 > MAX_MIN:
            print(f"time budget hit at {k}/{len(pages)}", flush=True)
            break
        im = cv2.imread(path)
        if im is None:
            failed += 1
            continue
        H, W = im.shape[:2]
        try:
            raw = []
            if H <= int(TILE * 1.35):
                tiles = [(0, H)]
            else:
                tiles, y = [], 0
                while y < H:
                    tiles.append((y, min(H, y + TILE)))
                    if y + TILE >= H:
                        break
                    y += TILE - OVERLAP
            for (ya, yb) in tiles:
                for b in detect(model, proc, im[ya:yb]):
                    raw.append([b[0], ya + b[1], b[2], ya + b[3]])
            merged = merge(raw, H, W)
            final = []
            for b in merged:
                final.extend(snap_and_split(im, b))
            # final geo filter
            final = [b for b in final
                     if (b[2]-b[0]) >= 0.12*W and (b[3]-b[1]) >= 0.010*H]
            cv2.imwrite(f"{OUT}/images/{stem}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            with open(lp, "w") as f:
                for x1, y1, x2, y2 in final:
                    cx, cy = (x1+x2)/2/W, (y1+y2)/2/H
                    bw, bh = (x2-x1)/W, (y2-y1)/H
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            tot_boxes += len(final)
            done += 1
            if done <= 80:
                ov = im.copy()
                for x1, y1, x2, y2 in final:
                    cv2.rectangle(ov, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                s = 1000 / max(ov.shape[0], 1)
                if s < 1:
                    ov = cv2.resize(ov, (int(ov.shape[1]*s), int(ov.shape[0]*s)))
                cv2.imwrite(f"{OUT}/overlays/{stem}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 78])
        except Exception as e:
            failed += 1
            print(f"  FAIL {stem}: {repr(e)[:160]}", flush=True)
        if (k + 1) % 50 == 0:
            el = (time.time() - t0) / 60
            print(f"  {k+1}/{len(pages)} done={done} skip={skipped} fail={failed} "
                  f"boxes={tot_boxes} {el:.0f}min ({done/max(el,0.1):.0f}/min)", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    with open(f"{OUT}/_summary.json", "w") as f:
        json.dump({"pages": len(pages), "done": done, "skipped": skipped,
                   "failed": failed, "boxes": tot_boxes,
                   "minutes": round((time.time()-t0)/60, 1)}, f, indent=1)
    print(f"DONE done={done} skip={skipped} fail={failed} boxes={tot_boxes}", flush=True)


if __name__ == "__main__":
    main()
