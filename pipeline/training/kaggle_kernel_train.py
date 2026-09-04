# Webtoon panel+bubble detector — YOLOv8n fine-tune on Kaggle P100.
# Merges our webtoon set + 2 public Kaggle sets, remapped to 0=panel 1=bubble.
import glob, os, shutil, subprocess, sys, random, traceback
def _pipq(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)

# Kaggle's preinstalled torch 2.10+cu128 DROPPED Pascal (sm_60 / P100) support ->
# "no kernel image is available for execution on the device". Pin to a build
# that still covers sm_60..sm_90.
_pipq("torch==2.4.1", "torchvision==0.19.1",
      "--index-url", "https://download.pytorch.org/whl/cu121")
_pipq("--no-deps", "ultralytics", "ultralytics-thop", "py-cpuinfo")
for _m in ("onnx", "onnxslim"):
    try:
        __import__(_m)
    except ImportError:
        _pipq("--no-deps", _m)

import yaml, torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
DEV = 0
if torch.cuda.is_available():
    try:
        _ = (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).cpu()
        print("GPU OK:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("GPU unusable (", e, ") -> CPU"); DEV = "cpu"
else:
    DEV = "cpu"
from ultralytics import YOLO

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/ds"                       # NOT /kaggle/working -> keeps output small
for s in ("train", "val"):
    os.makedirs(f"{ROOT}/images/{s}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{s}", exist_ok=True)
random.seed(0)
N = 0


def _find(*frags):
    """first dir under /kaggle/input whose path contains all frags"""
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


def add(img_dir, remap, tag, val_frac=0.12, force_split=None):
    global N
    if not img_dir or not os.path.isdir(img_dir):
        print(f"  [{tag}] MISSING {img_dir}")
        return
    lbl_dir = img_dir.replace("/images", "/labels") if "/images" in img_dir \
        else img_dir.rsplit("/", 1)[0] + "/labels"
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
                # clamp any out-of-bounds / negative box (some source sets have them)
                x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                if x2 - x1 < 0.003 or y2 - y1 < 0.003:
                    continue
                lines.append(f"{remap[c]} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} "
                             f"{x2-x1:.6f} {y2-y1:.6f}")
        split = force_split or ("val" if i < n_val else "train")
        name = f"{tag}_{i}_{stem}"[:120]
        shutil.copy(ip, f"{ROOT}/images/{split}/{name}{os.path.splitext(ip)[1].lower()}")
        open(f"{ROOT}/labels/{split}/{name}.txt", "w").write("\n".join(lines))
        kept += 1
        N += 1
    print(f"  [{tag}] +{kept} from {img_dir}")


wy = _find("webtoon-yolo")
add(f"{wy}/images/train", {0: 0, 1: 1}, "wtn_tr", force_split="train")
add(f"{wy}/images/val", {0: 0, 1: 1}, "wtn_va", force_split="val")

le = _find("lie-eater")
add(f"{le}/train/images", {0: 1}, "le_tr", force_split="train")
add(f"{le}/valid/images", {0: 1}, "le_va", force_split="val")

cs = _find("comic-segmentation")
add(f"{cs}/images/train", {25: 0, 26: 1, 27: 1}, "bd_tr", val_frac=0.12)
add(f"{cs}/images/val", {25: 0, 26: 1, 27: 1}, "bd_va", force_split="val")

nt = len(glob.glob(f"{ROOT}/images/train/*"))
nv = len(glob.glob(f"{ROOT}/images/val/*"))
print(f"MERGED: {N} images | train {nt} | val {nv}")
assert nt > 200 and nv > 20, "merge produced too few images"

yaml.safe_dump({"path": ROOT, "train": "images/train", "val": "images/val",
                "nc": 2, "names": ["panel", "bubble"]},
               open("/kaggle/working/data.yaml", "w"))

try:
    model = YOLO("yolov8n.pt")
    _ep = 120 if DEV == 0 else 60
    model.train(data="/kaggle/working/data.yaml", epochs=_ep, imgsz=1024,
                batch=(16 if DEV == 0 else 8),
                device=DEV, workers=4, patience=25, cache=False,
                fliplr=0.0, degrees=0.0, shear=0.0, perspective=0.0, mosaic=0.4,
                hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,
                project="/kaggle/working/runs", name="webtoon", exist_ok=True)
    best = "/kaggle/working/runs/webtoon/weights/best.pt"
    m = YOLO(best)
    r = m.val(data="/kaggle/working/data.yaml", imgsz=1024)
    print("mAP50-95:", float(r.box.map), "| mAP50:", float(r.box.map50))
    onnx = m.export(format="onnx", imgsz=1024, opset=13, simplify=True, nms=True)
    shutil.copy(onnx, "/kaggle/working/webtoon_panel_yolo.onnx")
    shutil.copy(best, "/kaggle/working/best.pt")
    print("DONE — /kaggle/working/webtoon_panel_yolo.onnx")
except Exception:
    traceback.print_exc()
    raise
