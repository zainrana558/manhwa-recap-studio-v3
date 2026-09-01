#!/usr/bin/env python3
"""Merge the per-source YOLO label sets into one balanced training dataset.

  yolo_kumiko/       Kumiko on 4 bordered-manga series      (CV, excellent)
  yolo_webtoon_cv2/  CV gutter + edge/palette/luma split    (CV, good; v1 fallback)
  yolo_roboflow/     Roboflow comic/manga panel datasets    (human bbox, capped)
  yolo/              Gemini flash-lite on webtoon pages     (VLM, ~decent)

Priorities: keep every manga page; keep every MULTI-box webtoon page (real
segmentation signal); add a capped sample of single-box webtoon pages
(content-region signal). Per-series cap so ORV can't dominate. `tbate` held
out entirely for val (style generalisation).

    python assemble.py --out train_ready
"""
import argparse
import glob
import hashlib
import random
from pathlib import Path
import shutil

import cv2

HERE = Path(__file__).parent
VAL_SERIES = {"tbate"}
VAL_FRAC = 0.08
CAP = 340                       # per series
SINGLE_BOX_KEEP = 900          # cap on single-box webtoon pages


def nboxes(lp):
    return sum(1 for ln in Path(lp).read_text().split("\n") if len(ln.split()) == 5) \
        if Path(lp).exists() else 0


def read_boxes(lp):
    out = []
    for ln in Path(lp).read_text().split("\n"):
        f = ln.split()
        if len(f) == 5:
            cx, cy, w, h = (float(x) for x in f[1:])
            if 0.10 < w <= 1.001 and 0.008 < h <= 1.001 and w * h < 0.995:
                out.append((cx, cy, w, h))
    return out


def phash(im):
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (16, 16))
    return hashlib.md5((g > g.mean()).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "train_ready"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    # gather candidates: (priority, series, img_path, label_path)
    cands = []
    KUM = HERE / "yolo_kumiko"
    for ip in glob.glob(f"{KUM}/images/*"):
        st = Path(ip).stem
        cands.append((0, st.split("__")[0], ip, f"{KUM}/labels/{st}.txt"))
    WC = HERE / "yolo_webtoon_cv2"
    if not (WC / "images").is_dir():
        WC = HERE / "yolo_webtoon_cv"
    sb = []
    for ip in glob.glob(f"{WC}/images/*"):
        st = Path(ip).stem
        lp = f"{WC}/labels/{st}.txt"
        n = nboxes(lp)
        if n >= 2:
            cands.append((1, st.split("__")[0], ip, lp))
        elif n == 1:
            sb.append((2, st.split("__")[0], ip, lp))
    rng.shuffle(sb)
    cands += sb[:SINGLE_BOX_KEEP]
    RF = HERE / "yolo_roboflow"          # human-annotated comic/manga panels
    for ip in glob.glob(f"{RF}/images/*"):
        st = Path(ip).stem
        cands.append((1, "rf_" + st.split("_")[0], ip, f"{RF}/labels/{st}.txt"))
    GEM = HERE / "yolo"
    for ip in glob.glob(f"{GEM}/images/*"):
        st = Path(ip).stem
        if st.split("__")[0] in ("chainsaw-man",):     # covered by Kumiko
            continue
        cands.append((1, st.split("__")[0], ip, f"{GEM}/labels/{st}.txt"))

    rng.shuffle(cands)
    cands.sort(key=lambda c: c[0])          # priority 0 (manga) first
    seen_hash = set()
    per_series = {}
    kept = dropped = 0
    stats = {}
    for prio, series, ip, lp in cands:
        if per_series.get(series, 0) >= CAP:
            dropped += 1
            continue
        im = cv2.imread(ip)
        if im is None:
            dropped += 1
            continue
        hh = phash(im)
        if hh in seen_hash:
            dropped += 1
            continue
        boxes = read_boxes(lp) if Path(lp).exists() else []
        seen_hash.add(hh)
        per_series[series] = per_series.get(series, 0) + 1
        split = "val" if (series in VAL_SERIES or rng.random() < VAL_FRAC) else "train"
        name = Path(ip).stem[:120]
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(out / "labels" / split / f"{name}.txt", "w") as f:
            for cx, cy, w, h in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        kept += 1
        stats[series] = stats.get(series, 0) + 1

    import yaml
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                    "nc": 1, "names": ["panel"]}, open(out / "data.yaml", "w"))
    ntr = len(glob.glob(f"{out}/images/train/*"))
    nva = len(glob.glob(f"{out}/images/val/*"))
    print(f"kept {kept}  dropped {dropped}  | train {ntr}  val {nva}")
    manga = sum(v for k, v in stats.items() if k.startswith("rf_") or k in
                ("chainsaw-man", "one-piece", "berserk", "jujutsu-kaisen"))
    print(f"  manga {manga}  webtoon {kept - manga}")
    for s, n in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
