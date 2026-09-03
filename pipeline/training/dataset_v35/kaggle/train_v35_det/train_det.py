# v3.5 detect baseline: YOLO11m, 5-class panel bounding boxes.
# Same data as panel-train-v35-seg but boxes only - faster, and the head-to-head
# vs the seg model tells us whether masks are worth the runtime cost.
import glob, os, shutil, subprocess, sys, yaml


def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


pip("torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121")
pip("ultralytics==8.4.137")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
print("cuda", torch.cuda.is_available(), flush=True)

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/v35det"
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
    for ip in glob.glob(f"{DS}/images/{sp}/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        lp = f"{DS}/labels/{sp}/{st}.txt"
        if not os.path.exists(lp):
            continue
        rows = []
        for ln in open(lp):
            q = ln.split()
            if len(q) < 5:
                continue
            c = MAP.get(int(q[0]), 0)
            rows.append(f"{c} {q[1]} {q[2]} {q[3]} {q[4]}")
        if not rows:
            continue
        shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
        open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(rows) + "\n")

yaml.safe_dump({"path": ROOT, "train": "images/train", "val": "images/val",
                "nc": 5, "names": NAMES}, open(f"{ROOT}/data.yaml", "w"), sort_keys=False)

m = YOLO("yolo11m.pt")
m.train(data=f"{ROOT}/data.yaml", epochs=200, imgsz=1024, batch=14, optimizer="AdamW",
        lr0=1e-3, cos_lr=True, warmup_epochs=4, patience=40,
        degrees=8.0, translate=0.06, scale=0.45, shear=2.0,
        fliplr=0.5, mosaic=0.6, close_mosaic=20, mixup=0.05, cls=1.4,
        project="/kaggle/working", name="v35_det", exist_ok=True)
best = "/kaggle/working/v35_det/weights/best.pt"
try:
    YOLO(best).export(format="onnx", imgsz=1024, opset=13, simplify=True)
except Exception as e:
    print("export:", e)
print("DONE", best, flush=True)
