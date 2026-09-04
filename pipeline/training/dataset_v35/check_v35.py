#!/usr/bin/env python3
"""Full integrity check of the assembled dataset_v35/ (local stage).

Runs every invariant the train kernels assume.  Exit 0 = clean.
    python check_v35.py [dataset_v35]
"""
import csv, glob, hashlib, json, os, sys, collections
import cv2
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "dataset_v35")
PANEL = ["rectangle", "square", "noborder", "diagonal", "irregular", "split", "outbound"]
P5 = ["rectangle", "square", "noborder", "irregular", "outbound"]
MAP75 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4}
errs, warns = [], []


def E(m):
    errs.append(m)


def W(m):
    warns.append(m)


# ---- 1. file pairing ----
for sp in ("train", "val"):
    imgs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/images/{sp}/*")}
    for sub in ("labels", "labels_seg", "labels5"):
        labs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/{sub}/{sp}/*.txt")}
        if imgs - labs:
            E(f"{sp}: {len(imgs-labs)} images with no {sub} (e.g. {sorted(imgs-labs)[:2]})")
        if labs - imgs:
            E(f"{sp}: {len(labs-imgs)} {sub} with no image (e.g. {sorted(labs-imgs)[:2]})")

# ---- 2. train / val disjoint ----
tr = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/images/train/*")}
va = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/images/val/*")}
if tr & va:
    E(f"{len(tr&va)} stems in BOTH train and val: {sorted(tr&va)[:3]}")

# ---- 3. label content: ranges, class ids, polygon parity ----
cls_bbox = collections.Counter()
cls_seg = collections.Counter()
cls5 = collections.Counter()
n_lines_mismatch = 0
for sp in ("train", "val"):
    for lp in glob.glob(f"{ROOT}/labels/{sp}/*.txt"):
        st = os.path.splitext(os.path.basename(lp))[0]
        b_lines = [l.split() for l in open(lp) if l.strip()]
        s_lines = [l.split() for l in open(f"{ROOT}/labels_seg/{sp}/{st}.txt") if l.strip()]
        f5_lines = [l.split() for l in open(f"{ROOT}/labels5/{sp}/{st}.txt") if l.strip()]
        if not (len(b_lines) == len(s_lines) == len(f5_lines)):
            n_lines_mismatch += 1
            if n_lines_mismatch <= 3:
                E(f"{sp}/{st}: line count bbox={len(b_lines)} seg={len(s_lines)} l5={len(f5_lines)}")
        for q in b_lines:
            if len(q) != 5:
                E(f"{sp}/{st}: bbox line has {len(q)} fields")
                continue
            c = int(q[0])
            cls_bbox[c] += 1
            cx, cy, bw, bh = map(float, q[1:])
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                E(f"{sp}/{st}: bbox out of range {q}")
            if not (0 <= c <= 6):
                E(f"{sp}/{st}: bad class {c}")
        for q in s_lines:
            c = int(q[0])
            cls_seg[c] += 1
            coords = list(map(float, q[1:]))
            if len(coords) < 6 or len(coords) % 2:
                E(f"{sp}/{st}: seg polygon has {len(coords)} coords")
            if any(v < -0.001 or v > 1.001 for v in coords):
                E(f"{sp}/{st}: seg coord out of range")
        for q in f5_lines:
            c = int(q[0])
            cls5[c] += 1
            if not (0 <= c <= 4):
                E(f"{sp}/{st}: labels5 bad class {c}")

# ---- 4. bbox vs seg class agreement + 7->5 map ----
if cls_bbox != cls_seg:
    W(f"bbox vs seg class counts differ: {dict(cls_bbox)} vs {dict(cls_seg)}")
exp5 = collections.Counter()
for c, n in cls_bbox.items():
    exp5[MAP75[c]] += n
if exp5 != cls5:
    E(f"labels5 != map(labels): got {dict(cls5)} expected {dict(exp5)}")

# ---- 5. images readable + size sane ----
bad_img = 0
small = 0
for sp in ("train", "val"):
    for ip in glob.glob(f"{ROOT}/images/{sp}/*"):
        im = cv2.imread(ip)
        if im is None:
            bad_img += 1
            if bad_img <= 3:
                E(f"unreadable image {ip}")
            continue
        h, w = im.shape[:2]
        if min(h, w) < 120:
            small += 1
if small:
    W(f"{small} images with a side < 120px")

# ---- 6. dedup: phash collisions within dedup tiers ----
man = list(csv.reader(open(f"{ROOT}/manifest.csv")))[1:]
tier_of = {r[0]: r[1] for r in man}
ph = {}
dups = 0
for sp in ("train", "val"):
    for ip in glob.glob(f"{ROOT}/images/{sp}/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        if tier_of.get(st) == "T2_webtoon_human":
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (16, 16))
        h = hashlib.md5((g > g.mean()).tobytes()).hexdigest()
        if h in ph:
            dups += 1
            if dups <= 3:
                W(f"phash dup: {st} ~ {ph[h]}")
        else:
            ph[h] = st

# ---- 7. val series policy ----
val_series = collections.Counter(r[2] for r in man if r[3] == "val")
tbate_tr = [r[0] for r in man if r[2] == "tbate" and r[3] == "train"]
if tbate_tr:
    E(f"{len(tbate_tr)} tbate pages leaked into train")

# ---- 8. COCO sanity ----
for sp in ("train", "val"):
    d = json.load(open(f"{ROOT}/coco/{sp}/_annotations.coco.json"))
    iid = {im["id"] for im in d["images"]}
    fns = {im["file_name"] for im in d["images"]}
    disk = {os.path.basename(p) for p in glob.glob(f"{ROOT}/images/{sp}/*")}
    if fns - disk:
        E(f"coco/{sp}: {len(fns-disk)} referenced images missing on disk")
    if len(iid) != len(d["images"]):
        E(f"coco/{sp}: duplicate image ids")
    for a in d["annotations"]:
        if a["image_id"] not in iid:
            E(f"coco/{sp}: ann {a['id']} -> missing image {a['image_id']}")
            break
        if not (0 <= a["category_id"] <= 6):
            E(f"coco/{sp}: bad category {a['category_id']}")
            break
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            E(f"coco/{sp}: non-positive bbox {a['bbox']}")
            break
    n_ann_disk = sum(1 for _ in open(f"{ROOT}/labels/{sp}/{os.path.splitext(list(fns)[0])[0]}.txt")) if fns else 0
    if len(d["annotations"]) != sum(cls_bbox.values()) if sp == "train" else True:
        pass

# ---- 9. yaml + ext ----
import yaml
for y, nc in (("data.yaml", 7), ("data_seg.yaml", 7), ("data5.yaml", 5), ("data_aux.yaml", 4)):
    try:
        cfg = yaml.safe_load(open(f"{ROOT}/{y}"))
        if cfg["nc"] != nc or len(cfg["names"]) != nc:
            E(f"{y}: nc/names mismatch ({cfg['nc']}, {len(cfg['names'])}) expected {nc}")
    except Exception as e:
        E(f"{y}: {e}")
n_coo = len(glob.glob(f"{ROOT}/ext/coo_yolo/*.txt"))
if n_coo < 5000:
    W(f"only {n_coo} COO pages converted")

# ---- report ----
print(f"\n=== dataset_v35 check ===  ({ROOT})")
print(f"images: train {len(tr)}  val {len(va)}")
print(f"panels (7-cls): {dict(sorted(cls_bbox.items()))}")
print(f"  -> {[f'{PANEL[k]}={v}' for k, v in sorted(cls_bbox.items())]}")
print(f"panels (5-cls): {[f'{P5[k]}={v}' for k, v in sorted(cls5.items())]}")
print(f"val series: {dict(val_series)}")
print(f"COO pages converted: {n_coo}")
print(f"phash dups (non-T2): {dups}")
print(f"\n{len(errs)} ERRORS, {len(warns)} warnings")
for m in errs:
    print("  ERR ", m)
for m in warns[:15]:
    print("  warn", m)
sys.exit(1 if errs else 0)
