# v3.5 primary detector: YOLO11m-seg, 5-class panel INSTANCE SEGMENTATION.
#
# Masks (not just boxes) are what let the recap pipeline crop a diagonal /
# irregular panel to its real polygon and blur-fill the neighbour bleed WITHOUT
# ever rotating the content (see dataset_v35/clean_panel.py).
#
# labels_seg/ in webtoon-panels-v35 is 7-class polygons; we remap 7->5:
#   rectangle square noborder [diagonal|split->]irregular outbound
# Rotation aug (degrees=8) gives rotated-panel robustness the honest way -
# no synthetic tilted pages in the data.
import glob, os, shutil, subprocess, sys, yaml


def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


pip("torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121")
pip("ultralytics==8.4.137")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/v35seg"
MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4}
NAMES = ["rectangle", "square", "noborder", "irregular", "outbound"]


def find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


DS = find("webtoon-panels-v35") or find("webtoon-panels-v35-src")
print("dataset:", DS, flush=True)

for sp in ("train", "val"):
    os.makedirs(f"{ROOT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{sp}", exist_ok=True)

seg_dir = "labels_seg" if os.path.isdir(f"{DS}/labels_seg") else "labels"
kept = {sp: 0 for sp in ("train", "val")}
for sp in ("train", "val"):
    for ip in glob.glob(f"{DS}/images/{sp}/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        lp = f"{DS}/{seg_dir}/{sp}/{st}.txt"
        if not os.path.exists(lp):
            continue
        rows = []
        for ln in open(lp):
            q = ln.split()
            if len(q) < 7:                      # need a polygon (>=3 pts)
                if len(q) == 5:                 # bbox fallback -> rect poly
                    c, cx, cy, bw, bh = q
                    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                    x1, y1, x2, y2 = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
                    q = [c, x1, y1, x2, y1, x2, y2, x1, y2]
                else:
                    continue
            c = MAP.get(int(q[0]), 0)
            pts = " ".join(f"{max(0.0,min(1.0,float(v))):.5f}" for v in q[1:])
            rows.append(f"{c} {pts}")
        if not rows:
            continue
        shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
        open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(rows) + "\n")
        kept[sp] += 1
print("kept", kept, flush=True)

yaml.safe_dump({"path": ROOT, "train": "images/train", "val": "images/val",
                "nc": 5, "names": NAMES}, open(f"{ROOT}/data.yaml", "w"), sort_keys=False)

m = YOLO("yolo11m-seg.pt")
m.train(
    data=f"{ROOT}/data.yaml", task="segment",
    epochs=200, imgsz=1024, batch=10, optimizer="AdamW", lr0=1e-3,
    cos_lr=True, warmup_epochs=4, patience=40,
    degrees=8.0, translate=0.06, scale=0.45, shear=2.0, perspective=0.0002,
    fliplr=0.5, flipud=0.0, mosaic=0.6, close_mosaic=20, mixup=0.05,
    cls=1.4, box=7.5, overlap_mask=True,
    project="/kaggle/working", name="v35_seg", exist_ok=True,
)
best = "/kaggle/working/v35_seg/weights/best.pt"
try:
    YOLO(best).export(format="onnx", imgsz=1024, opset=13, simplify=True)
except Exception as e:
    print("export onnx:", e)
print("DONE", best, flush=True)
