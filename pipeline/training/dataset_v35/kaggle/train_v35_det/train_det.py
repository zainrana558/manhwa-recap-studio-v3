# v3.5 detect baseline: YOLO11m, 5-class panel bounding boxes.
# Same data as panel-train-v35-seg (webtoon-yolo v3.5 + Manga109 <frame>),
# boxes only - the head-to-head vs the seg model says whether masks earn their
# runtime cost.
import glob, os, random, shutil, subprocess, sys, xml.etree.ElementTree as ET
import yaml


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
MANGA_CAP = 3000


def find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


DS = find("webtoon-yolo") or find("webtoon-panels-v35")
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

m109 = find("manga109") or find("Manga109")
andir = next((p for p in glob.glob(f"{m109}/**/annotations", recursive=True) if os.path.isdir(p)), None) if m109 else None
imroot = next((p for p in glob.glob(f"{m109}/**/images", recursive=True) if os.path.isdir(p)), None) if m109 else None
pages = []
if andir and imroot:
    for xf in sorted(glob.glob(f"{andir}/*.xml")):
        base = os.path.splitext(os.path.basename(xf))[0]
        try:
            root = ET.parse(xf).getroot()
        except Exception:
            continue
        for pg in root.iter("page"):
            idx = int(pg.get("index"))
            W, H = float(pg.get("width", 0)), float(pg.get("height", 0))
            ip = f"{imroot}/{base}/{idx:03d}.jpg"
            fr = list(pg.iter("frame"))
            if W and H and fr and os.path.exists(ip):
                pages.append((ip, W, H, [(float(f.get("xmin")), float(f.get("ymin")),
                                          float(f.get("xmax")), float(f.get("ymax"))) for f in fr]))
random.seed(5)
random.shuffle(pages)
n_m = 0
for ip, W, H, frames in pages[:MANGA_CAP]:
    st = "m109__" + os.path.basename(os.path.dirname(ip)) + "__" + os.path.basename(ip)[:-4]
    sp = "val" if random.random() < 0.04 else "train"
    rows = []
    for x1, y1, x2, y2 in frames:
        fw, fh = (x2 - x1) / W, (y2 - y1) / H
        if fw < 0.02 or fh < 0.02:
            continue
        ar = fw / max(1e-6, fh)
        c = 1 if (0.80 <= ar <= 1.25 and fw * fh < 0.22) else 0
        cx, cy = min(1, max(0, (x1 + x2) / 2 / W)), min(1, max(0, (y1 + y2) / 2 / H))
        rows.append(f"{c} {cx:.6f} {cy:.6f} {min(1,fw):.6f} {min(1,fh):.6f}")
    if not rows:
        continue
    shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
    open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(rows) + "\n")
    n_m += 1
print(f"manga109 pages added: {n_m}", flush=True)

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
