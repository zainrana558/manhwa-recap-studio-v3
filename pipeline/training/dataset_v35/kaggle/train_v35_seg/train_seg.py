# v3.5 primary detector: YOLO11m-seg, 5-class panel INSTANCE SEGMENTATION.
#
# Masks (not just boxes) let the recap pipeline crop a diagonal / irregular panel
# to its real polygon and blur-fill the neighbour bleed WITHOUT ever rotating the
# content (dataset_v35/clean_panel.py).
#
# Data:
#   webtoon-yolo   local v3.5 build: images/ labels_seg/ (7-class polygons)
#                  7 -> 5 : rectangle square noborder [diagonal|split ->]irregular outbound
#   Manga109       <frame> -> rectangle / square (aspect), folded in here
#                  (btlam0507/manga109), capped so it can't swamp the webtoons
# Rotation aug (degrees=8) gives rotated-panel robustness the honest way.
import glob, os, random, shutil, subprocess, sys, xml.etree.ElementTree as ET
import yaml


def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


pip("torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121")
pip("ultralytics==8.4.137", "opencv-python-headless")

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/v35seg"
MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4}
NAMES = ["rectangle", "square", "noborder", "irregular", "outbound"]
MANGA_CAP = 3000


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
for sp in ("train", "val"):
    os.makedirs(f"{ROOT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{sp}", exist_ok=True)

seg_dir = "labels_seg" if os.path.isdir(f"{DS}/labels_seg") else "labels"
kept = {"train": 0, "val": 0}
for sp in ("train", "val"):
    for ip in glob.glob(f"{DS}/images/{sp}/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        lp = f"{DS}/{seg_dir}/{sp}/{st}.txt"
        if not os.path.exists(lp):
            continue
        rows = []
        for ln in open(lp):
            q = ln.split()
            if len(q) == 5:                       # bbox -> rect poly
                c, cx, cy, bw, bh = q
                cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                q = [c, x1, y1, x2, y1, x2, y2, x1, y2]
            elif len(q) < 7:
                continue
            c = MAP.get(int(q[0]), 0)
            pts = " ".join(f"{max(0.0, min(1.0, float(v))):.5f}" for v in q[1:])
            rows.append(f"{c} {pts}")
        if not rows:
            continue
        shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
        open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(rows) + "\n")
        kept[sp] += 1
print("webtoon/comic kept:", kept, flush=True)

# ---- Manga109 <frame> -> rectangle / square polygons ----
m109 = find("manga109") or find("Manga109")
andir = next((p for p in glob.glob(f"{m109}/**/annotations", recursive=True) if os.path.isdir(p)), None) if m109 else None
imroot = next((p for p in glob.glob(f"{m109}/**/images", recursive=True) if os.path.isdir(p)), None) if m109 else None
print("manga109:", andir, imroot, flush=True)
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
pages = pages[:MANGA_CAP]
n_m = 0
for ip, W, H, frames in pages:
    st = "m109__" + os.path.basename(os.path.dirname(ip)) + "__" + os.path.basename(ip)[:-4]
    sp = "val" if random.random() < 0.04 else "train"
    rows = []
    for x1, y1, x2, y2 in frames:
        fw, fh = (x2 - x1) / W, (y2 - y1) / H
        if fw < 0.02 or fh < 0.02:
            continue
        ar = fw / max(1e-6, fh)
        c = 1 if (0.80 <= ar <= 1.25 and fw * fh < 0.22) else 0
        a, b, cc, d = x1 / W, y1 / H, x2 / W, y2 / H
        a, b = max(0, a), max(0, b)
        cc, d = min(1, cc), min(1, d)
        rows.append(f"{c} {a:.5f} {b:.5f} {cc:.5f} {b:.5f} {cc:.5f} {d:.5f} {a:.5f} {d:.5f}")
    if not rows:
        continue
    shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
    open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(rows) + "\n")
    n_m += 1
print(f"manga109 pages added: {n_m}", flush=True)

yaml.safe_dump({"path": ROOT, "train": "images/train", "val": "images/val",
                "nc": 5, "names": NAMES}, open(f"{ROOT}/data.yaml", "w"), sort_keys=False)

m = YOLO("yolo11m-seg.pt")
m.train(
    data=f"{ROOT}/data.yaml", task="segment",
    epochs=200, imgsz=1024, batch=8, optimizer="AdamW", lr0=1e-3,
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
