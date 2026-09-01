# Fresh YOLO11n panel+bubble detector.
#  class 0 = panel : Kumiko(manga) + Gemini(webtoon) labels, pre-split, in
#                    /kaggle/input/webtoon-yolo/{images,labels}/{train,val}
#  class 1 = bubble: lie-eater-speech-bubbles + comic-segmentation (public)
import glob
import os
import random
import shutil
import subprocess
import sys
import traceback


def _pipq(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


_pipq("torch==2.4.1", "torchvision==0.19.1",
      "--index-url", "https://download.pytorch.org/whl/cu121")
_pipq("--no-deps", "ultralytics", "ultralytics-thop", "py-cpuinfo")
for _m in ("onnx", "onnxslim"):
    try:
        __import__(_m)
    except ImportError:
        _pipq("--no-deps", _m)

import yaml  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
DEV = 0
if torch.cuda.is_available():
    try:
        _ = (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).cpu()
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    except Exception as e:
        print("GPU unusable:", e); DEV = "cpu"
else:
    DEV = "cpu"

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/ds"
for s in ("train", "val"):
    os.makedirs(f"{ROOT}/images/{s}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{s}", exist_ok=True)
random.seed(1)
N = 0


def _find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


def add(img_dir, remap, tag, split=None, val_frac=0.10):
    global N
    if not img_dir or not os.path.isdir(img_dir):
        print(f"  [{tag}] MISSING {img_dir}", flush=True)
        return
    lbl_dir = (img_dir.replace("/images", "/labels") if "/images" in img_dir
               else img_dir.rsplit("/", 1)[0] + "/labels")
    imgs = [p for p in glob.glob(img_dir + "/*")
            if p.rsplit(".", 1)[-1].lower() in ("jpg", "jpeg", "png", "webp")]
    random.shuffle(imgs)
    n_val = int(len(imgs) * val_frac)
    kept = 0
    for i, ip in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")
        lines = []
        if os.path.exists(lp):
            for ln in open(lp):
                q = ln.split()
                if len(q) != 5:
                    continue
                c = int(float(q[0]))
                if c not in remap:
                    continue
                cx, cy, w, h = (float(v) for v in q[1:])
                x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                if x2 - x1 < 0.003 or y2 - y1 < 0.003:
                    continue
                lines.append(f"{remap[c]} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} "
                             f"{x2-x1:.6f} {y2-y1:.6f}")
        sp = split or ("val" if i < n_val else "train")
        name = f"{tag}_{i}_{stem}"[:120]
        shutil.copy(ip, f"{ROOT}/images/{sp}/{name}{os.path.splitext(ip)[1].lower()}")
        open(f"{ROOT}/labels/{sp}/{name}.txt", "w").write("\n".join(lines))
        kept += 1
        N += 1
    print(f"  [{tag}] +{kept} from {img_dir}", flush=True)


# --- panels: the pre-split v3 trainset ---
wy = _find("webtoon-yolo")
add(f"{wy}/images/train", {0: 0}, "pan_tr", split="train")
add(f"{wy}/images/val", {0: 0}, "pan_va", split="val")

# --- bubbles ---
le = _find("lie-eater")
add(f"{le}/train/images", {0: 1}, "le_tr", split="train")
add(f"{le}/valid/images", {0: 1}, "le_va", split="val")
cs = _find("comic-segmentation")
add(f"{cs}/images/train", {26: 1, 27: 1}, "bd_tr")
add(f"{cs}/images/val", {26: 1, 27: 1}, "bd_va", split="val")

nt = len(glob.glob(f"{ROOT}/images/train/*"))
nv = len(glob.glob(f"{ROOT}/images/val/*"))
print(f"MERGED {N} | train {nt} | val {nv}", flush=True)
assert nt > 500 and nv > 80

yaml.safe_dump({"path": ROOT, "train": "images/train", "val": "images/val",
                "nc": 2, "names": ["panel", "bubble"]},
               open("/kaggle/working/data.yaml", "w"))

try:
    base = "yolo11n.pt"
    try:
        YOLO(base)
    except Exception:
        base = "yolov8n.pt"
    ep = 160 if DEV == 0 else 40
    m = YOLO(base)
    m.train(data="/kaggle/working/data.yaml", epochs=ep, imgsz=1024,
            batch=(16 if DEV == 0 else 6), device=DEV, workers=4,
            patience=35, close_mosaic=15,
            fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
            mosaic=0.5, hsv_h=0.0, hsv_s=0.3, hsv_v=0.35, translate=0.06, scale=0.35,
            project="/kaggle/working/runs", name="v3", exist_ok=True)
    best = "/kaggle/working/runs/v3/weights/best.pt"
    mm = YOLO(best)
    r = mm.val(data="/kaggle/working/data.yaml", imgsz=1024)
    print(f"mAP50-95 {float(r.box.map):.4f} | mAP50 {float(r.box.map50):.4f}", flush=True)
    onnx = mm.export(format="onnx", imgsz=1024, opset=13, simplify=True, nms=True)
    shutil.copy(onnx, "/kaggle/working/manga_panel_detector_v3_1024.onnx")
    shutil.copy(best, "/kaggle/working/best_v3.pt")
    print("DONE", flush=True)
except Exception:
    traceback.print_exc()
    raise
