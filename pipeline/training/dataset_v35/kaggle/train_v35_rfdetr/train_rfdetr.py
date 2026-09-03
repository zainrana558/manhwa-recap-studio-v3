# v3.5 alt arch: RF-DETR (DINOv2 ViT backbone), 5-class panels.
# DETR queries + no-NMS beat YOLO on tall/crowded pages (CoMix benchmark).
# Same webtoon-panels-v35 dataset; YOLO 7-class -> 5-class -> COCO on the fly.
import glob, json, os, random, subprocess, sys, xml.etree.ElementTree as ET
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


def _get_ds():
    """webtoon-yolo / webtoon-panels-v35 as a Kaggle dataset if present, else
    wget the tar straight from the box (Kaggle dataset versioning kept hanging)."""
    import glob, os, subprocess, tarfile
    ex = "/kaggle/tmp/ds35"
    if os.path.isdir(ex + "/images"):
        return ex
    d = find("webtoon-yolo") or find("webtoon-panels-v35")
    if d and os.path.isdir(d + "/images"):
        return d
    tar = glob.glob(f"{d}/**/webtoon_v35.tar", recursive=True) if d else []
    src = tar[0] if tar else None
    if not src:
        src = "/kaggle/tmp/webtoon_v35.tar"
        if not os.path.exists(src):
            subprocess.run(["wget", "-q", "--tries=4", "--timeout=60", "-O", src,
                            "http://80.225.248.230/slicer/dl/webtoon_v35.tar"], check=True)
    os.makedirs(ex, exist_ok=True)
    with tarfile.open(src) as t:
        t.extractall(ex)
    return ex


DS = _get_ds()
print("dataset:", DS, flush=True)


def coco_split(src_sp, out_sp):
    d = f"{ROOT}/{out_sp}"
    os.makedirs(d, exist_ok=True)
    images, anns = [], []
    aid = 1
    iid = 0
    for ip in sorted(glob.glob(f"{DS}/images/{src_sp}/*")):
        st = os.path.splitext(os.path.basename(ip))[0]
        lp = f"{DS}/labels/{src_sp}/{st}.txt"
        if not os.path.exists(lp):
            continue
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            continue
        iid += 1
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
    # Manga109 <frame> -> rectangle / square, appended to the same COCO split
    if out_sp == "train":
        m109 = next((p for p in glob.glob(f"{IN}/**", recursive=True)
                     if os.path.isdir(p) and "manga109" in p.lower()), None)
        andir = next((p for p in glob.glob(f"{m109}/**/annotations", recursive=True)
                      if os.path.isdir(p)), None) if m109 else None
        imroot = next((p for p in glob.glob(f"{m109}/**/images", recursive=True)
                       if os.path.isdir(p)), None) if m109 else None
        pages = []
        for xf in sorted(glob.glob(f"{andir}/*.xml")) if andir else []:
            base = os.path.splitext(os.path.basename(xf))[0]
            try:
                root = ET.parse(xf).getroot()
            except Exception:
                continue
            for pg in root.iter("page"):
                W2, H2 = float(pg.get("width", 0)), float(pg.get("height", 0))
                ip2 = f"{imroot}/{base}/{int(pg.get('index')):03d}.jpg"
                fr = list(pg.iter("frame"))
                if W2 and H2 and fr and os.path.exists(ip2):
                    pages.append((ip2, W2, H2, fr))
        random.seed(5)
        random.shuffle(pages)
        for ip2, W2, H2, fr in pages[:3000]:
            iid += 1
            try:
                Image.open(ip2).convert("RGB").save(f"{d}/{iid:07d}.jpg", quality=85)
            except Exception:
                continue
            images.append({"id": iid, "file_name": f"{iid:07d}.jpg", "width": int(W2), "height": int(H2)})
            for f in fr:
                x1, y1 = float(f.get("xmin")), float(f.get("ymin"))
                x2, y2 = float(f.get("xmax")), float(f.get("ymax"))
                fw, fh = (x2 - x1) / W2, (y2 - y1) / H2
                if fw < 0.02 or fh < 0.02:
                    continue
                ar = fw / max(1e-6, fh)
                c = 1 if (0.80 <= ar <= 1.25 and fw * fh < 0.22) else 0
                anns.append({"id": aid, "image_id": iid, "category_id": c + 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
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
