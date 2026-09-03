# v3.5 alt arch: RF-DETR (DINOv2 ViT backbone), 5-class panels.
# DETR queries + no-NMS beat YOLO on tall/crowded pages (CoMix benchmark).
# Same webtoon-panels-v35 dataset; YOLO 7-class -> 5-class -> COCO on the fly.
import glob, json, os, subprocess, sys
from PIL import Image


def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


pip("torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121")
pip("rfdetr", "pycocotools", "supervision")

import torch  # noqa: E402
print("cuda", torch.cuda.is_available(), flush=True)

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/coco_v35"
MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4}
CATS = [{"id": i + 1, "name": n} for i, n in enumerate(
    ["rectangle", "square", "noborder", "irregular", "outbound"])]


def find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


DS = find("webtoon-panels-v35") or find("webtoon-panels-v35-src")
print("dataset:", DS, flush=True)


def coco_split(src_sp, out_sp):
    d = f"{ROOT}/{out_sp}"
    os.makedirs(d, exist_ok=True)
    images, anns = [], []
    aid = 1
    for iid, ip in enumerate(sorted(glob.glob(f"{DS}/images/{src_sp}/*")), 1):
        st = os.path.splitext(os.path.basename(ip))[0]
        lp = f"{DS}/labels/{src_sp}/{st}.txt"
        if not os.path.exists(lp):
            continue
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        fn = f"{iid:07d}.jpg"
        im.save(f"{d}/{fn}", quality=88)
        images.append({"id": iid, "file_name": fn, "width": W, "height": H})
        for ln in open(lp):
            q = ln.split()
            if len(q) < 5:
                continue
            c = MAP.get(int(q[0]), 0)
            cx, cy, bw, bh = map(float, q[1:5])
            anns.append({"id": aid, "image_id": iid, "category_id": c + 1,
                         "bbox": [(cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H],
                         "area": bw * W * bh * H, "iscrowd": 0})
            aid += 1
    json.dump({"images": images, "annotations": anns, "categories": CATS},
              open(f"{d}/_annotations.coco.json", "w"))
    print(f"{out_sp}: {len(images)} imgs / {len(anns)} anns", flush=True)


coco_split("train", "train")
coco_split("val", "valid")

from rfdetr import RFDETRBase  # noqa: E402

model = RFDETRBase()
model.train(dataset_dir=ROOT, epochs=70, batch_size=8, grad_accum_steps=2, lr=1e-4,
            output_dir="/kaggle/working/rfdetr_v35", resolution=1120, num_classes=5)
try:
    model.export(output_dir="/kaggle/working/rfdetr_v35")
except Exception as e:
    print("export:", e)
print("DONE", flush=True)
