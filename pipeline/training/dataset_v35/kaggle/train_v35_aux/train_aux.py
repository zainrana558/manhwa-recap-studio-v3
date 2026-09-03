# v3.5 aux detector: YOLO11s-seg, 4-class  bubble / text / onomatopoeia / face.
#
# Replaces the slow 3-ONNX cascade in the slicer's _protected_rows guard with
# one fast model, so panel cuts never bisect a speech bubble, caption or face.
#
# Aux labels are built here:
#   webtoon pages (webtoon-yolo) -> ogkalu (bubble/text) + manga109-yolo (face)
#                                   from panel-detector-onnx
#   Manga109 pages               -> <face>/<text> boxes + COO onomatopoeia
#                                   (ext/coo_yolo/ ships in webtoon-yolo)
import glob, os, random, shutil, subprocess, sys, xml.etree.ElementTree as ET
import numpy as np
import yaml


def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


pip("torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121")
pip("ultralytics==8.4.137", "opencv-python-headless")
ort = None
try:
    pip("onnxruntime-gpu==1.19.2")
    import onnxruntime as ort
except Exception:
    try:
        pip("onnxruntime")
        import onnxruntime as ort
    except Exception:
        ort = None
PROV = (["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort is not None and "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"])

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
print("cuda", torch.cuda.is_available(), "ort", None if ort is None else PROV, flush=True)

IN = "/kaggle/input"
ROOT = "/kaggle/tmp/v35aux"
NAMES = ["bubble", "text", "onomatopoeia", "face"]
WEB_CAP = 2200
MANGA_CAP = 2500


def find(*frags):
    for p in glob.glob(IN + "/**", recursive=True):
        if os.path.isdir(p) and all(f in p for f in frags):
            return p
    return None


def _untar_ds(d):
    import glob, os, tarfile
    if not d:
        return d
    t = glob.glob(f"{d}/**/webtoon_v35.tar", recursive=True)
    if not t:
        return d
    ex = "/kaggle/tmp/ds35"
    if not os.path.isdir(ex + "/images"):
        os.makedirs(ex, exist_ok=True)
        with tarfile.open(t[0]) as tf:
            tf.extractall(ex)
    return ex


DS = _untar_ds(find("webtoon-yolo") or find("webtoon-panels-v35"))
ONNX = find("panel-detector-onnx")
M109 = find("manga109") or find("Manga109")
print("dataset:", DS, "onnx:", ONNX, "m109:", M109, flush=True)
for sp in ("train", "val"):
    os.makedirs(f"{ROOT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{sp}", exist_ok=True)


def seg_line(c, poly, W, H):
    return f"{c} " + " ".join(f"{max(0, min(1, x / W)):.5f} {max(0, min(1, y / H)):.5f}" for x, y in poly)


def rect(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


BUB = FACE = None
if ort is not None and ONNX:
    bp = f"{ONNX}/comic-text-and-bubble-detector/detector-v4-s_int8.onnx"
    fp = f"{ONNX}/manga109-yolo/model.onnx"
    for path, name in ((bp, "BUB"), (fp, "FACE")):
        if os.path.exists(path):
            try:
                s = ort.InferenceSession(path, providers=PROV)
                globals()[name] = s
            except Exception as e:
                print(name, "session failed", e, flush=True)
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
            out.append((0 if int(l) in (0, 1) else 1, rect(b[0], b[1] + y, b[2], b[3] + y)))
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
            out.append((3, rect(cx - bw / 2, cy - bh / 2 + y, cx + bw / 2, cy + bh / 2 + y)))
        if yb >= H:
            break
        y += TH - 200
    return out


# ---- webtoon aux ----
web = []
for sp in ("train", "val"):
    for ip in glob.glob(f"{DS}/images/{sp}/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        if not st.startswith("m109__"):
            web.append((sp, ip, st))
random.seed(7)
random.shuffle(web)
n_w = 0
for sp, ip, st in web[:WEB_CAP]:
    if BUB is None and FACE is None:
        break
    im = cv2.imread(ip)
    if im is None:
        continue
    H, W = im.shape[:2]
    aux = []
    try:
        if BUB is not None:
            aux += bub_tiles(im)
        if FACE is not None:
            aux += face_tiles(im)
    except Exception:
        pass
    if not aux:
        continue
    shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
    open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(seg_line(c, p, W, H) for c, p in aux) + "\n")
    n_w += 1
    if n_w % 300 == 0:
        print(f"  web aux {n_w}", flush=True)
print("webtoon aux pages:", n_w, flush=True)

# ---- manga109 aux: <face>/<text> + COO onomatopoeia ----
coo_dir = next((p for p in glob.glob(f"{DS}/**/coo_yolo", recursive=True) if os.path.isdir(p)), None)
andir = next((p for p in glob.glob(f"{M109}/**/annotations", recursive=True) if os.path.isdir(p)), None) if M109 else None
imroot = next((p for p in glob.glob(f"{M109}/**/images", recursive=True) if os.path.isdir(p)), None) if M109 else None
print("coo_dir:", coo_dir, "m109 ann:", andir, flush=True)
mpages = []
if andir and imroot:
    for xf in sorted(glob.glob(f"{andir}/*.xml")):
        base = os.path.splitext(os.path.basename(xf))[0]
        try:
            root = ET.parse(xf).getroot()
        except Exception:
            continue
        btitle = root.get("title") or base
        for pg in root.iter("page"):
            idx = int(pg.get("index"))
            W, H = float(pg.get("width", 0)), float(pg.get("height", 0))
            ip = f"{imroot}/{base}/{idx:03d}.jpg"
            if not (W and H and os.path.exists(ip)):
                continue
            fa = [(float(f.get("xmin")), float(f.get("ymin")), float(f.get("xmax")), float(f.get("ymax")))
                  for f in pg.iter("face")]
            tx = [(float(f.get("xmin")), float(f.get("ymin")), float(f.get("xmax")), float(f.get("ymax")))
                  for f in pg.iter("text")]
            mpages.append((ip, btitle, idx, W, H, fa, tx))
random.shuffle(mpages)
n_m = 0
for ip, btitle, idx, W, H, fa, tx in mpages[:MANGA_CAP]:
    st = "m109__" + os.path.basename(os.path.dirname(ip)) + "__" + os.path.basename(ip)[:-4]
    rows = []
    for x1, y1, x2, y2 in fa:
        rows.append(seg_line(3, rect(x1, y1, x2, y2), W, H))
    for x1, y1, x2, y2 in tx:
        rows.append(seg_line(1, rect(x1, y1, x2, y2), W, H))
    if coo_dir:
        cf = f"{coo_dir}/{btitle}__{idx:03d}.txt"
        if os.path.exists(cf):
            for ln in open(cf):
                q = ln.split()
                if len(q) >= 7:
                    rows.append("2 " + " ".join(q[1:]))
    if not rows:
        continue
    sp = "val" if random.random() < 0.04 else "train"
    shutil.copy(ip, f"{ROOT}/images/{sp}/{st}.jpg")
    open(f"{ROOT}/labels/{sp}/{st}.txt", "w").write("\n".join(rows) + "\n")
    n_m += 1
print("manga109 aux pages:", n_m, flush=True)

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
