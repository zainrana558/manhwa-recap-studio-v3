# v3.4 — 3-class webtoon panel detector (rect / noborder / irregular).
# Reads the webtoon-yolo dataset (versioned from _v34_trainset by
# `kaggle datasets version`). yolo11m base, imgsz 1024, class-balanced.
import glob
import os
import shutil
import subprocess
import sys
import traceback


def _pipq(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


_pipq("torch==2.4.1", "torchvision==0.19.1",
      "--index-url", "https://download.pytorch.org/whl/cu121")
# pin: newer ultralytics ships the Muon optimiser which crashes on P100/AMP
_pipq("--no-deps", "ultralytics==8.4.137", "ultralytics-thop", "py-cpuinfo")
for _m in ("onnx", "onnxslim"):
    try:
        __import__(_m)
    except ImportError:
        _pipq("--no-deps", _m)

import yaml   # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
DEV = 0 if torch.cuda.is_available() else "cpu"
if DEV == 0:
    try:
        _ = (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).cpu()
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    except Exception as e:
        print("GPU unusable:", e)
        DEV = "cpu"

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/ds"
for s in ("train", "val"):
    os.makedirs(f"{ROOT}/images/{s}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{s}", exist_ok=True)


def _find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


N = 0
wy = _find("webtoon-yolo")
for sp in ("train", "val"):
    idir = f"{wy}/images/{sp}"
    ldir = f"{wy}/labels/{sp}"
    if not os.path.isdir(idir):
        continue
    for ip in glob.glob(idir + "/*"):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(ldir, stem + ".txt")
        lines = []
        if os.path.exists(lp):
            for ln in open(lp):
                q = ln.split()
                if len(q) == 5 and q[0] in ("0", "1", "2"):
                    lines.append(" ".join(q))
        shutil.copy(ip, f"{ROOT}/images/{sp}/{stem}{os.path.splitext(ip)[1].lower()}")
        open(f"{ROOT}/labels/{sp}/{stem}.txt", "w").write("\n".join(lines))
        N += 1

# --- Manga109: 109 volumes, ~10k human-labelled pages, <frame> boxes -> rect(0)
import xml.etree.ElementTree as ET  # noqa: E402

m109 = _find("Manga109") or _find("manga109")
mg_added = 0
if m109:
    andir = next((p for p in glob.glob(m109 + "/**/annotations", recursive=True)
                  if os.path.isdir(p)), None)
    imroot = next((p for p in glob.glob(m109 + "/**/images", recursive=True)
                   if os.path.isdir(p)), None)
    if andir and imroot:
        import random as _r
        _r.seed(2)
        for xf in glob.glob(andir + "/*.xml"):
            title = os.path.splitext(os.path.basename(xf))[0]
            try:
                root = ET.parse(xf).getroot()
            except Exception:
                continue
            for pg in root.iter("page"):
                idx = pg.get("index")
                W = float(pg.get("width", 0))
                H = float(pg.get("height", 0))
                if not (W and H):
                    continue
                fr = [f for f in pg.iter("frame")]
                if not fr:
                    continue
                ip = f"{imroot}/{title}/{int(idx):03d}.jpg"
                if not os.path.exists(ip):
                    continue
                sp = "val" if _r.random() < 0.05 else "train"
                nm = f"m109_{title}_{int(idx):03d}"
                shutil.copy(ip, f"{ROOT}/images/{sp}/{nm}.jpg")
                with open(f"{ROOT}/labels/{sp}/{nm}.txt", "w") as fo:
                    for f in fr:
                        x1, y1 = float(f.get("xmin")), float(f.get("ymin"))
                        x2, y2 = float(f.get("xmax")), float(f.get("ymax"))
                        if x2 - x1 < 8 or y2 - y1 < 8:
                            continue
                        fo.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                                 f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
                mg_added += 1
print(f"Manga109 pages added: {mg_added}", flush=True)

nt = len(glob.glob(f"{ROOT}/images/train/*"))
nv = len(glob.glob(f"{ROOT}/images/val/*"))
print(f"MERGED {N}+m109 | train {nt} | val {nv}", flush=True)
assert nt > 800 and nv > 80

yaml.safe_dump({"path": ROOT, "train": "images/train", "val": "images/val",
                "nc": 3, "names": ["rect", "noborder", "irregular"]},
               open("/kaggle/working/data.yaml", "w"))

try:
    base = "yolo11m.pt"
    try:
        YOLO(base)
    except Exception:
        base = "yolo11s.pt"
    ep = 200 if DEV == 0 else 40
    m = YOLO(base)
    m.train(data="/kaggle/working/data.yaml", epochs=ep, imgsz=1024,
            batch=(12 if DEV == 0 else 4), device=DEV, workers=4,
            optimizer="AdamW", patience=45, close_mosaic=15,
            cls=1.2,                              # push the minority classes
            fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
            mosaic=0.4, hsv_h=0.0, hsv_s=0.3, hsv_v=0.35, translate=0.05, scale=0.3,
            project="/kaggle/working/runs", name="v34", exist_ok=True)
    best = "/kaggle/working/runs/v34/weights/best.pt"
    mm = YOLO(best)
    r = mm.val(data="/kaggle/working/data.yaml", imgsz=1024)
    print(f"mAP50 {float(r.box.map50):.4f} | mAP50-95 {float(r.box.map):.4f}", flush=True)
    try:
        for i, nm in enumerate(["rect", "noborder", "irregular"]):
            print(f"  {nm}: mAP50 {float(r.box.ap50[i]):.3f}", flush=True)
    except Exception:
        pass
    onnx = mm.export(format="onnx", imgsz=1024, opset=13, simplify=True, nms=True)
    shutil.copy(onnx, "/kaggle/working/manga_panel_detector_v34_1024.onnx")
    shutil.copy(best, "/kaggle/working/best_v34.pt")
    print("DONE", flush=True)
except Exception:
    traceback.print_exc()
    raise
