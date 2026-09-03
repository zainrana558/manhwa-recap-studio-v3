#!/usr/bin/env python3
"""v3.5 dataset — LOCAL stage.  REAL data only (no synthetic pages).

Panel taxonomy (Roboflow "webtoon-manhwa-panels" 7-class, kept un-collapsed):
  0 rectangle  1 square  2 noborder  3 diagonal  4 irregular  5 split  6 outbound

Local tiers:
  T1 webtoon_cascade  yolo_final3          our curated CV+cascade slices   -> noborder
  T2 webtoon_human    roboflow_webtoon     238 hand-labelled webtoon pages -> 7-class  (kept in full)
  T3 comic_human      yolo_roboflow        Roboflow comic panels           -> rectangle (capped)
  T4 manga_cv         yolo_kumiko          Kumiko CV manga panels          -> rectangle (capped)

Kaggle stage (build_v35 / train kernels) then folds in, from images that only
live there:
  T5 manga_human      Manga109 <frame>     -> rectangle / square / outbound   (classify_shape, geometry)
  T6 manga_teacher    koharu RF-DETR-seg   -> diagonal / irregular / ...       (real polygons on manga)
  T7 teacher_magi     Magi v2              -> panels                           (webtoon + manga)

The diagonal / split / square classes are DELIBERATELY sourced only from real
labels (T2) and real model teachers on manga (T6/T7) + geometry on Manga109
(T5).  They will be lower-support; the train kernels apply a class-balanced loss
weight and rotation augmentation rather than baking fake tilted pages.

Outputs under dataset_v35/:
  images/{train,val}/  labels/{train,val}/       YOLO bbox, 7-class
  labels_seg/{train,val}/                        YOLO polygon (rect-poly for bbox tiers)
  labels_aux/{train,val}/                        filled by relabel_aux.py (webtoon) + kernels (manga)
  coco/{train,val}/_annotations.coco.json
  data.yaml  data_seg.yaml  data_aux.yaml  manifest.csv
  ext/coo/*.xml  ext/dialog/*.xml  ext/SOURCES.md

    python assemble_v35.py
"""
import csv, glob, hashlib, json, os, random, shutil, collections
from pathlib import Path
import cv2

HERE = Path(__file__).resolve().parent
D3 = HERE.parent / "dataset_v3"
EXT = HERE / "ext_annotations" / "m109_public"
OUT = HERE / "dataset_v35"

PANEL = ["rectangle", "square", "noborder", "diagonal", "irregular", "split", "outbound"]
# label files stay 7-class (canonical); training collapses to 5 because the recap
# pipeline crops axis-aligned and never rotates content, so diagonal/split are
# functionally "irregular" (use the polygon, mask the bleed).
PANEL5 = ["rectangle", "square", "noborder", "irregular", "outbound"]
MAP_7_TO_5 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4}
AUX = ["bubble", "text", "onomatopoeia", "face"]
RBW_TO_OURS = {0: 3, 1: 4, 2: 2, 3: 6, 4: 0, 5: 5, 6: 1}   # Roboflow order -> ours
VAL_SERIES = {"tbate"}
# LEAN build: only labels we can vouch for.
#   T2 human webtoon (95, x6)      -- perfect, multi-panel
#   T1s strict-QC'd cascade (635)  -- verified clean edges (mostly single-panel)
#   T3s strict-QC'd comic (80)     -- verified clean edges
#   Manga109 <frame>               -- human, added in the train kernels
# dropped: raw cascade, kumiko CV, raw roboflow, restitch  (gutter-bleed / noise)
HUMAN_OVERSAMPLE = 3
CAPS = {}


def phash(im):
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (16, 16))
    return hashlib.md5((g > g.mean()).tobytes()).hexdigest()


def series_of(stem):
    if "__" in stem:
        return stem.split("__")[0]
    return stem.split("_jpg")[0].rstrip("0123456789_-") or stem


def collect():
    out = []
    for sub, tier, remap, dedup in [
        ("roboflow_webtoon/_merged", "T2_webtoon_human", lambda k: RBW_TO_OURS.get(k, 0), False),
        ("yolo_final3_strict", "T1s_webtoon_cascade", lambda k: 2, True),
        ("yolo_roboflow_strict", "T3s_comic_human", lambda k: 0, True),
    ]:
        for ip in sorted(glob.glob(str(D3 / sub / "images" / "*"))):
            lp = D3 / sub / "labels" / (Path(ip).stem + ".txt")
            if lp.exists():
                out.append((tier, series_of(Path(ip).stem), ip, str(lp), remap, dedup, ""))
    # oversample the only truly-accurate webtoon labels (95 human pages)
    for extra in range(1, HUMAN_OVERSAMPLE):
        for r in [c for c in out if c[0] == "T2_webtoon_human"]:
            out.append((r[0], r[1], r[2], r[3], r[4], False, f"_ov{extra}"))
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        for d in ("images", "labels", "labels_seg", "labels_aux"):
            (OUT / d / s).mkdir(parents=True, exist_ok=True)
        (OUT / "coco" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "ext" / "coo").mkdir(parents=True, exist_ok=True)
    (OUT / "ext" / "dialog").mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    cands = collect()
    rng.shuffle(cands)

    seen = set()
    per_tier = collections.Counter()
    coco = {sp: {"images": [], "annotations": [],
                 "categories": [{"id": i, "name": n} for i, n in enumerate(PANEL)]}
            for sp in ("train", "val")}
    aid = {"train": 1, "val": 1}
    iid = 0
    manifest = [("image", "tier", "series", "split", "n_panels")]

    for tier, ser, ip, lp, remap, dedup, *rest in cands:
        suf = rest[0] if rest else ""
        if tier in CAPS and per_tier[tier] >= CAPS[tier]:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        if min(H, W) < 120:
            continue
        if dedup:
            h = phash(im)
            if h in seen:
                continue
            seen.add(h)

        rows = []
        for ln in open(lp):
            f = ln.split()
            if len(f) < 5:
                continue
            k = int(float(f[0]))
            cx, cy, bw, bh = (float(x) for x in f[1:5])
            # clamp to the page — some source labels (roboflow rfm_*) have boxes
            # that run off the image, which breaks the rect-polygon export
            x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
            x2, y2 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
            if x2 - x1 < 0.012 or y2 - y1 < 0.012:
                continue
            rows.append((remap(k), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        if not rows:
            continue

        per_tier[tier] += 1
        iid += 1
        is_val = (ser in VAL_SERIES
                  or (tier == "T2_webtoon_human" and not suf and rng.random() < 0.15)
                  or rng.random() < 0.05)
        if suf:
            is_val = False                      # oversample copies stay in train
        sp = "val" if is_val else "train"
        stem = (Path(ip).stem[:112] + suf)
        fn = f"{stem}.jpg"
        cv2.imwrite(str(OUT / "images" / sp / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 92])

        with open(OUT / "labels" / sp / f"{stem}.txt", "w") as fb, \
             open(OUT / "labels_seg" / sp / f"{stem}.txt", "w") as fs:
            for c, cx, cy, bw, bh in rows:
                fb.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                x1 = min(max(cx - bw / 2, 0.0), 1.0)
                y1 = min(max(cy - bh / 2, 0.0), 1.0)
                x2 = min(max(cx + bw / 2, 0.0), 1.0)
                y2 = min(max(cy + bh / 2, 0.0), 1.0)
                fs.write(f"{c} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} "
                         f"{x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n")
                coco[sp]["annotations"].append({
                    "id": aid[sp], "image_id": iid, "category_id": c,
                    "bbox": [round((cx - bw / 2) * W, 2), round((cy - bh / 2) * H, 2),
                             round(bw * W, 2), round(bh * H, 2)],
                    "area": round(bw * W * bh * H, 2), "iscrowd": 0})
                aid[sp] += 1
        coco[sp]["images"].append({"id": iid, "file_name": fn, "width": W, "height": H})
        manifest.append((stem, tier, ser, sp, len(rows)))

    for sp in ("train", "val"):
        json.dump(coco[sp], open(OUT / "coco" / sp / "_annotations.coco.json", "w"))

    # 5-class training view: labels5/ = labels/ with {diagonal,split}->irregular
    for sp in ("train", "val"):
        (OUT / "labels5" / sp).mkdir(parents=True, exist_ok=True)
        for lp in glob.glob(str(OUT / "labels" / sp / "*.txt")):
            with open(lp) as fi, open(OUT / "labels5" / sp / Path(lp).name, "w") as fo:
                for ln in fi:
                    p = ln.split()
                    if len(p) >= 5:
                        fo.write(" ".join([str(MAP_7_TO_5[int(p[0])])] + p[1:]) + "\n")

    import yaml
    # data.yaml -> 7-class canonical (labels/).  data5.yaml -> 5-class (labels5/).
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                    "nc": 7, "names": PANEL}, open(OUT / "data.yaml", "w"), sort_keys=False)
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                    "nc": 7, "names": PANEL, "task": "segment"},
                   open(OUT / "data_seg.yaml", "w"), sort_keys=False)
    (OUT / "data5.yaml").write_text(
        "# 5-class training view. Point YOLO here after: rm -r labels && mv labels5 labels\n"
        "# (or use the train kernel, which remaps 7->5 on load).\n"
        + yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                          "nc": 5, "names": PANEL5}, sort_keys=False))
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                    "nc": 4, "names": AUX}, open(OUT / "data_aux.yaml", "w"), sort_keys=False)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows(manifest)

    for x in glob.glob(str(EXT / "COO-Comic-Onomatopoeia" / "**" / "*.xml"), recursive=True):
        shutil.copy(x, OUT / "ext" / "coo" / Path(x).name)
    for x in glob.glob(str(EXT / "Manga109Dialog" / "*.xml")):
        shutil.copy(x, OUT / "ext" / "dialog" / Path(x).name)
    # pre-convert COO polygons to YOLO-seg (aux class 2 = onomatopoeia), keyed
    # <Title>__<idx:03d>.txt so the build kernel can map them onto Manga109 images
    from coo_to_yolo import convert as _coo
    _coo(str(EXT / "COO-Comic-Onomatopoeia"), str(OUT / "ext" / "coo_yolo"), cls=2)
    for lic in glob.glob(str(EXT / "LICENSE*")):
        shutil.copy(lic, OUT / "ext" / Path(lic).name)
    (OUT / "ext" / "SOURCES.md").write_text(
        "coo/    Comic Onomatopoeia (Manga109 SFX polygons)  CC-BY-4.0  github.com/manga109/public-annotations\n"
        "dialog/ Manga109Dialog speaker<->text links                    github.com/manga109/public-annotations\n"
        "Manga109 images: Kaggle btlam0507/manga109 (research licence). Folded in by the train kernels.\n")

    tc = collections.Counter(m[1] for m in manifest[1:])
    cls = collections.Counter()
    for sp in ("train", "val"):
        for lp in glob.glob(str(OUT / "labels" / sp / "*.txt")):
            for ln in open(lp):
                cls[PANEL[int(ln.split()[0])]] += 1
    print(f"images {iid}   train {len(coco['train']['images'])}   val {len(coco['val']['images'])}")
    for t, n in tc.most_common():
        print(f"  {t:20s} {n}")
    print("panel classes:", {k: cls[k] for k in PANEL})
    print(f"COO xml {len(list((OUT/'ext'/'coo').glob('*.xml')))}   -> {OUT}")


if __name__ == "__main__":
    main()
