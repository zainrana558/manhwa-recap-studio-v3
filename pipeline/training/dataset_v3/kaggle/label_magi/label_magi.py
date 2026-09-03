# Magi v2 label-teacher: run ragavsachdeva/magiv2 over every webtoon/manhwa page
# in the webtoon-manhwa-raw dataset, tile tall strips, map panel boxes back,
# 1-D NMS the seam duplicates, write YOLO labels (class 0 = panel).
# Output: /kaggle/working/yolo_magi/{images,labels}.zip   (download, unzip into
# pipeline/training/dataset_v3/yolo_magi/, then assemble_v34.py --with-magi)
import glob
import os
import subprocess
import sys
import traceback
import zipfile


def _pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


_pip("torch==2.4.1", "torchvision==0.19.1", "--index-url",
     "https://download.pytorch.org/whl/cu121")
_pip("transformers==4.44.2", "einops", "shapely", "timm", "pillow")

import numpy as np           # noqa: E402
import torch                 # noqa: E402
from PIL import Image        # noqa: E402
from transformers import AutoModel  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)

model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
model = model.to(DEV).eval()

SRC = None
for p in glob.glob("/kaggle/input/**/webtoon_manhwa*", recursive=True):
    if os.path.isdir(p):
        SRC = p
        break
if SRC is None:
    SRC = "/kaggle/input/webtoon-manhwa-raw"
pages = [p for p in glob.glob(f"{SRC}/**/*", recursive=True)
         if p.rsplit(".", 1)[-1].lower() in ("webp", "jpg", "jpeg", "png")]
print(f"{len(pages)} pages from {SRC}", flush=True)

OUT = "/kaggle/working/yolo_magi"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
TILE, OV = 2000, 350


def read_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


done = 0
for i, p in enumerate(pages):
    try:
        arr = read_rgb(p)
        H, W = arr.shape[:2]
        tiles, offs = [], []
        y = 0
        while y < H:
            yb = min(H, y + TILE)
            tiles.append(arr[y:yb])
            offs.append(y)
            if yb >= H:
                break
            y += TILE - OV
        with torch.no_grad():
            res = model.predict_detections_and_associations(tiles)
        boxes = []
        for r, oy in zip(res, offs):
            for b in r["panels"]:
                x1, y1, x2, y2 = b
                boxes.append((x1, y1 + oy, x2, y2 + oy))
        # 1-D interval NMS on the vertical axis for seam dupes
        boxes.sort(key=lambda b: b[1])
        merged = []
        for b in boxes:
            if merged:
                m = merged[-1]
                iy = min(m[3], b[3]) - max(m[1], b[1])
                if iy > 0 and iy / min(m[3] - m[1], b[3] - b[1]) > 0.5:
                    merged[-1] = (min(m[0], b[0]), min(m[1], b[1]),
                                  max(m[2], b[2]), max(m[3], b[3]))
                    continue
            merged.append(tuple(b))
        if not merged:
            continue
        rel = os.path.relpath(p, SRC)
        stem = rel.replace("/", "__").rsplit(".", 1)[0]
        Image.fromarray(arr).save(f"{OUT}/images/{stem}.jpg", quality=88)
        with open(f"{OUT}/labels/{stem}.txt", "w") as f:
            for x1, y1, x2, y2 in merged:
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                        f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        done += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pages)} done={done}", flush=True)
    except Exception:
        if done < 5:
            traceback.print_exc()

print(f"DONE {done} pages labelled", flush=True)
for d in ("images", "labels"):
    with zipfile.ZipFile(f"/kaggle/working/yolo_magi_{d}.zip", "w", zipfile.ZIP_STORED) as z:
        for f in glob.glob(f"{OUT}/{d}/*"):
            z.write(f, f"{d}/{os.path.basename(f)}")
print("zipped", flush=True)
