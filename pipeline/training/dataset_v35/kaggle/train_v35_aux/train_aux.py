# v3.5 aux detector: YOLO11s-seg, 4-class  bubble / text / onomatopoeia / face.
#
# Replaces the slow 3-ONNX cascade in the slicer's _protected_rows guard with
# one fast model, so panel cuts never bisect a speech bubble, caption or face.
#
# Aux labels:
#   manga pages  -> already in webtoon-panels-v35/labels_aux/ (Manga109 <face>/
#                   <text> + COO onomatopoeia + koharu bubble)
#   webtoon pages -> generated here with ogkalu (bubble/text) + manga109-yolo
#                    (face) from panel-detector-onnx
import glob, os, shutil, subprocess, sys, yaml
import numpy as np


def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


pip("torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121")
pip("ultralytics==8.4.137", "opencv-python-headless")
try:
    pip("onnxruntime-gpu")
    import onnxruntime as ort
    PROV = (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"])
except Exception:
    ort = None
    PROV = None

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import torch  # noqa: E402
print("cuda", torch.cuda.is_available(), "ort", None if ort is None else PROV, flush=True)

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/v35aux"
NAMES = ["bubble", "text", "onomatopoeia", "face"]


def find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


DS = find("webtoon-panels-v35") or find("webtoon-panels-v35-src")
ONNX = find("panel-detector-onnx")
print("dataset:", DS, "onnx:", ONNX, flush=True)
for sp in ("train", "val"):
    os.makedirs(f"{ROOT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{sp}", exist_ok=True)


def seg_line(c, poly, W, H):
    return f"{c} " + " ".join(f"{max(0,min(1,x/W)):.5f} {max(0,min(1,y/H)):.5f}" for x, y in poly)


BUB = FACE = None
if ort is not None and ONNX:
    bp = f"{ONNX}/comic-text-and-bubble-detector/detector-v4-s_int8.onnx"
    fp = f"{ONNX}/manga109-yolo/model.onnx"
    if os.path.exists(bp):
        BUB = ort.InferenceSession(bp, providers=PROV)
    if os.path.exists(fp):
        FACE = ort.InferenceSession(fp, providers=PROV)
print("aux onnx:", BUB is not None, FACE is not None, flush=True)


def bub_tiles(im):
    H, W = im.shape[:2]
    out, y = [], 0
    TH = max(2200, W)
    while y < H:
        yb = min(H, y + TH)
        c = im[y:yb]
        ch, cw = c.shape[:2]
        x = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (640, 640)).transpose(2, 0, 1)[None].astype(np.float32) / 255
        lb, bx, sc = BUB.run(None, {"images": x, "orig_target_sizes": np.array([[ch, cw]], np.int64)})
        for l, b, s in zip(lb[0], bx[0], sc[0]):
            if float(s) < 0.4 or int(l) not in (0, 1, 2):
                continue
            cid = 0 if int(l) in (0, 1) else 1
            out.append((cid, np.array([[b[0], b[1] + y], [b[2], b[1] + y],
                                       [b[2], b[3] + y], [b[0], b[3] + y]], np.float32)))
        if yb >= H:
            break
        y += TH - 200
    return out


def face_tiles(im):
    H, W = im.shape[:2]
    out, y = [], 0
    TH = max(2200, W)
    while y < H:
        yb = min(H, y + TH)
        c = im[y:yb]
        ch, cw = c.shape[:2]
        s = min(640 / cw, 640 / ch)
        nw, nh = int(cw * s), int(ch * s)
        cv_ = np.full((640, 640, 3), 114, np.uint8)
        cv_[:nh, :nw] = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (nw, nh))
        o = FACE.run(None, {"images": cv_.transpose(2, 0, 1)[None].astype(np.float32) / 255})[0][0].T
        sm = o[:, 4:]
        cls, cf = sm.argmax(1), sm.max(1)
        for i in np.where((cf > 0.45) & (cls == 1))[0]:
            cx, cy, bw, bh = o[i, :4] / s
            out.append((3, np.array([[cx - bw / 2, cy - bh / 2 + y], [cx + bw / 2, cy - bh / 2 + y],
                                     [cx + bw / 2, cy + bh / 2 + y], [cx - bw / 2, cy + bh / 2 + y]], np.float32)))
        if yb >= H:
            break
        y += TH - 200
    return out


kept = {"train": 0, "val": 0}
for sp in ("train", "val"):
    for ip in glob.glob(f"{DS}/images/{sp}/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        lp = f"{DS}/labels_aux/{sp}/{st}.txt"
        rows = []
        if os.path.exists(lp):
            rows = [ln.strip() for ln in open(lp) if len(ln.split()) >= 7]
        is_manga = st.startswith("m109__")
        if not rows and not is_manga and (BUB is not None or FACE is not None):
            im = cv2.imread(ip)
            if im is not None:
                H, W = im.shape[:2]
                aux = []
                try:
                    if BUB is not None:
                        aux += bub_tiles(im)
                    if FACE is not None:
                        aux += face_tiles(im)
                except Exception:
                    pass
                rows = [seg_line(c, p, W, H) for c, p in aux]
        if not rows:
            continue
        shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
        open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(rows) + "\n")
        kept[sp] += 1
    print("aux", sp, kept[sp], flush=True)

yaml.safe_dump({"path": ROOT, "train": "images/train", "val": "images/val",
                "nc": 4, "names": NAMES}, open(f"{ROOT}/data.yaml", "w"), sort_keys=False)

m = YOLO("yolo11s-seg.pt")
m.train(data=f"{ROOT}/data.yaml", task="segment", epochs=120, imgsz=1024, batch=16,
        optimizer="AdamW", lr0=1e-3, cos_lr=True, patience=30,
        degrees=6.0, translate=0.06, scale=0.4, fliplr=0.5, mosaic=0.5, close_mosaic=15,
        cls=1.2, project="/kaggle/working", name="v35_aux", exist_ok=True)
best = "/kaggle/working/v35_aux/weights/best.pt"
try:
    YOLO(best).export(format="onnx", imgsz=1024, opset=13, simplify=True)
except Exception as e:
    print("export:", e)
print("DONE", best, flush=True)
