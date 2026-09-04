# v3.4-rfdetr — RF-DETR (DINOv2 ViT backbone) 3-class panel detector.
# DETR queries + no NMS handle the tall/crowded webtoon case better than YOLO
# on the CoMix benchmark. Trains on the same webtoon-yolo v3.4 dataset + Manga109,
# converted YOLO -> COCO on the fly.  Parallel experiment to panel-train-v34.
import glob
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image


def _pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


_pip("torch==2.4.1", "torchvision==0.19.1", "--index-url",
     "https://download.pytorch.org/whl/cu121")
_pip("rfdetr", "pycocotools", "supervision")

import torch  # noqa: E402
print("cuda", torch.cuda.is_available(), flush=True)

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/coco"
CATS = [{"id": 1, "name": "rect"}, {"id": 2, "name": "noborder"}, {"id": 3, "name": "irregular"}]


def _find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


def coco_split(pairs, split):
    d = os.path.join(ROOT, split)
    os.makedirs(d, exist_ok=True)
    images, anns = [], []
    aid = 1
    for iid, (ip, boxes, W, H) in enumerate(pairs, 1):
        fn = f"{iid:07d}.jpg"
        try:
            Image.open(ip).convert("RGB").save(os.path.join(d, fn), quality=88)
        except Exception:
            continue
        images.append({"id": iid, "file_name": fn, "width": W, "height": H})
        for c, x1, y1, bw, bh in boxes:
            anns.append({"id": aid, "image_id": iid, "category_id": c + 1,
                         "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
    json.dump({"images": images, "annotations": anns, "categories": CATS},
              open(os.path.join(d, "_annotations.coco.json"), "w"))
    print(f"{split}: {len(images)} imgs / {len(anns)} anns", flush=True)


tr, va = [], []
wy = _find("webtoon-yolo")
for sp, bucket in (("train", tr), ("val", va)):
    idir, ldir = f"{wy}/images/{sp}", f"{wy}/labels/{sp}"
    for ip in glob.glob(idir + "/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        try:
            W, H = Image.open(ip).size
        except Exception:
            continue
        boxes = []
        lp = os.path.join(ldir, st + ".txt")
        if os.path.exists(lp):
            for ln in open(lp):
                q = ln.split()
                if len(q) != 5:
                    continue
                c, cx, cy, w, h = int(q[0]), *map(float, q[1:])
                boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H))
        bucket.append((ip, boxes, W, H))

m109 = _find("Manga109") or _find("manga109")
if m109:
    import random
    random.seed(2)
    andir = next((p for p in glob.glob(m109 + "/**/annotations", recursive=True) if os.path.isdir(p)), None)
    imroot = next((p for p in glob.glob(m109 + "/**/images", recursive=True) if os.path.isdir(p)), None)
    for xf in glob.glob(f"{andir}/*.xml") if andir else []:
        title = os.path.splitext(os.path.basename(xf))[0]
        try:
            root = ET.parse(xf).getroot()
        except Exception:
            continue
        for pg in root.iter("page"):
            W, H = float(pg.get("width", 0)), float(pg.get("height", 0))
            fr = list(pg.iter("frame"))
            ip = f"{imroot}/{title}/{int(pg.get('index')):03d}.jpg"
            if not (W and H and fr and os.path.exists(ip)):
                continue
            boxes = [(0, float(f.get("xmin")), float(f.get("ymin")),
                      float(f.get("xmax")) - float(f.get("xmin")),
                      float(f.get("ymax")) - float(f.get("ymin"))) for f in fr]
            (va if random.random() < 0.05 else tr).append((ip, boxes, W, H))

coco_split(tr, "train")
coco_split(va, "valid")

from rfdetr import RFDETRBase  # noqa: E402

model = RFDETRBase()
model.train(dataset_dir=ROOT, epochs=60, batch_size=8, grad_accum_steps=2,
            lr=1e-4, output_dir="/kaggle/working/rfdetr_v34", resolution=1120,
            num_classes=3)
try:
    model.export(output_dir="/kaggle/working/rfdetr_v34")
except Exception as e:
    print("export:", e)
print("DONE", flush=True)
