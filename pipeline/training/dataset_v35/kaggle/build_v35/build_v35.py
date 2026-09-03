# v3.5 dataset builder (Kaggle, GPU).
#
# Merges the local webtoon/comic tiers (already in webtoon-panels-v35-src) with
# the manga tiers + model teachers that can only run here:
#
#   src           local build: images/ labels/(7-cls) labels_seg/ ext/coo_yolo/
#   Manga109      <frame> -> rectangle/square/outbound (geometry);
#                 <face>/<text> -> aux;  images copied in
#   COO           onomatopoeia polygons -> aux  (mapped onto Manga109 images)
#   koharu-seg    RF-DETR-seg panel masks on a Manga109 sample -> diagonal /
#                 irregular / ... via shape geometry  (the real non-rect signal)
#   ogkalu+m109   bubble / text / face on the webtoon images -> aux
#   Magi          (optional) label-magi output -> extra panel pages
#
# Output: /kaggle/working/v35/  + v35_partNN.zip  ->  pull, push webtoon-panels-v35
import glob, json, math, os, shutil, subprocess, sys, time, traceback
import xml.etree.ElementTree as ET
import numpy as np

t0 = time.time()


def sh(*a):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=True)


sh("opencv-python-headless")
import cv2                                   # noqa: E402
# Kaggle's stock onnxruntime-gpu wants CUDA 13 / cuDNN 9 (P100 image is CUDA 12);
# 1.19.2 is the last build that runs on CUDA 12. Falls back to CPU otherwise.
ort = None
ORT_GPU = False
for attempt in (("onnxruntime-gpu==1.19.2",), ("onnxruntime",)):
    try:
        sh(*attempt)
        import onnxruntime as ort             # noqa: F811
        ORT_GPU = "CUDAExecutionProvider" in ort.get_available_providers()
        break
    except Exception as e:
        print("onnxruntime install/import failed for", attempt, e, flush=True)
        ort = None
print("ort:", None if ort is None else ort.__version__, "gpu:", ORT_GPU, flush=True)

IN = "/kaggle/input"
OUT = "/kaggle/working/v35"
PANEL = ["rectangle", "square", "noborder", "diagonal", "irregular", "split", "outbound"]
AUX = ["bubble", "text", "onomatopoeia", "face"]
# keep manga from swamping the webtoon signal; val stays webtoon-weighted so
# best.pt is still selected on webtoon performance
MANGA_PAGES_FRAME = 3200      # Manga109 frame pages (square / outbound / rect)
MANGA_PAGES_KOHARU = 2400     # koharu sample -> diagonal / irregular (GPU budget)
JPEGQ = 90


def find(*frags, isdir=True):
    for p in glob.glob(IN + "/**", recursive=True):
        if (os.path.isdir(p) if isdir else os.path.isfile(p)) and all(f in p for f in frags):
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


# ---------- shape geometry (compact classify_shape) ----------
def _hull(p):
    pts = sorted(range(len(p)), key=lambda i: (float(p[i][0]), float(p[i][1])))

    def cr(o, a, b):
        return (p[a][0]-p[o][0])*(p[b][1]-p[o][1]) - (p[a][1]-p[o][1])*(p[b][0]-p[o][0])
    lo = []
    for i in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], i) <= 0:
            lo.pop()
        lo.append(i)
    up = []
    for i in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], i) <= 0:
            up.pop()
        up.append(i)
    return p[lo[:-1] + up[:-1]]


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def classify(poly, W, H, has_border=True):
    p = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(p) < 3:
        return 0
    x1, y1, x2, y2 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    h = _hull(p)
    ha = max(1.0, _area(h))
    pa = _area(p)
    # min-area rect angle
    best = None
    n = len(h)
    for i in range(n):
        e = h[(i + 1) % n] - h[i]
        L = math.hypot(*e)
        if L < 1e-6:
            continue
        u = e / L
        R = np.array([[u[0], u[1]], [-u[1], u[0]]], np.float32)
        q = h @ R.T
        w = q[:, 0].max() - q[:, 0].min()
        hh = q[:, 1].max() - q[:, 1].min()
        if best is None or w * hh < best[0]:
            best = (w * hh, math.degrees(math.atan2(u[1], u[0])))
    marea, ang = best if best else (bw * bh, 0.0)
    off = min(abs(ang) % 90, 90 - (abs(ang) % 90))
    convex = pa / ha
    extent = pa / max(1.0, marea)
    ex, ey = 0.01 * W, 0.01 * H
    edges = int(x1 <= ex) + int(y1 <= ey) + int(x2 >= W - ex) + int(y2 >= H - ey)
    big = float(bw * bh) > 0.14 * W * H
    if off >= 6.0 and convex > 0.80:
        return 3
    if convex < 0.86 or extent < 0.72:
        return 4
    if edges >= 2 and big and has_border:
        return 6
    if not has_border:
        return 2
    ar = bw / bh
    if 0.78 <= ar <= 1.28 and bw * bh < 0.30 * W * H:
        return 1
    return 0


# ---------- IO helpers ----------
def wl(path, rows):
    with open(path, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))


def seg_line(cid, poly, W, H):
    return f"{cid} " + " ".join(f"{max(0,min(1,x/W)):.5f} {max(0,min(1,y/H)):.5f}" for x, y in poly)


def bbox_line(cid, poly, W, H):
    p = np.asarray(poly, float)
    x1 = min(max(p[:, 0].min(), 0), W)
    y1 = min(max(p[:, 1].min(), 0), H)
    x2 = min(max(p[:, 0].max(), 0), W)
    y2 = min(max(p[:, 1].max(), 0), H)
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    return (f"{cid} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} "
            f"{max(0.002, (x2-x1)/W):.6f} {max(0.002, (y2-y1)/H):.6f}")


for d in ("images/train", "images/val", "labels/train", "labels/val",
          "labels_seg/train", "labels_seg/val", "labels_aux/train", "labels_aux/val"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

man = [("image", "tier", "src", "split", "n_panels", "n_aux")]
cls_hist = {k: 0 for k in PANEL}
aux_hist = {k: 0 for k in AUX}

# ============ 1. copy the local src tiers ============
SRC = _get_ds()
print("src:", SRC, flush=True)
for sp in ("train", "val"):
    for ip in glob.glob(f"{SRC}/images/{sp}/*"):
        st = os.path.splitext(os.path.basename(ip))[0]
        shutil.copy(ip, f"{OUT}/images/{sp}/{st}.jpg")
        for sub in ("labels", "labels_seg"):
            s = f"{SRC}/{sub}/{sp}/{st}.txt"
            if os.path.exists(s):
                shutil.copy(s, f"{OUT}/{sub}/{sp}/{st}.txt")
        open(f"{OUT}/labels_aux/{sp}/{st}.txt", "a").close()
        for ln in open(f"{OUT}/labels/{sp}/{st}.txt"):
            if ln.strip():
                cls_hist[PANEL[int(ln.split()[0])]] += 1
        man.append((st, "src", "local", sp, 0, 0))
print(f"src copied: {len(man)-1}  ({time.time()-t0:.0f}s)", flush=True)

# ============ 2. Manga109 <frame>/<face>/<text> ============
m109 = find("manga109") or find("Manga109")
andir = next((p for p in glob.glob(m109 + "/**/annotations*", recursive=True)
              if os.path.isdir(p)), None) if m109 else None
imroot = next((p for p in glob.glob(m109 + "/**/images", recursive=True)
               if os.path.isdir(p)), None) if m109 else None
print("manga109:", andir, imroot, flush=True)

# COO onomatopoeia (pre-converted in src/ext/coo_yolo/)
coo_dir = f"{SRC}/ext/coo_yolo"
coo_have = set(os.path.splitext(os.path.basename(x))[0] for x in glob.glob(f"{coo_dir}/*.txt")) \
    if os.path.isdir(coo_dir) else set()
print("coo pages:", len(coo_have), flush=True)

import random
random.seed(5)
m109_pages = []          # (title, idx, W, H, imgpath)
if andir and imroot:
    for xf in sorted(glob.glob(f"{andir}/*.xml")):
        fbase = os.path.splitext(os.path.basename(xf))[0]
        try:
            root = ET.parse(xf).getroot()
        except Exception:
            continue
        btitle = root.get("title") or fbase
        for pg in root.iter("page"):
            idx = int(pg.get("index"))
            W, H = float(pg.get("width", 0)), float(pg.get("height", 0))
            ip = f"{imroot}/{fbase}/{idx:03d}.jpg"
            if not (W and H and os.path.exists(ip)):
                ip2 = f"{imroot}/{btitle}/{idx:03d}.jpg"
                if os.path.exists(ip2):
                    ip = ip2
                else:
                    continue
            frames = [(float(f.get("xmin")), float(f.get("ymin")),
                       float(f.get("xmax")), float(f.get("ymax"))) for f in pg.iter("frame")]
            faces = [(float(f.get("xmin")), float(f.get("ymin")),
                      float(f.get("xmax")), float(f.get("ymax"))) for f in pg.iter("face")]
            texts = [(float(f.get("xmin")), float(f.get("ymin")),
                      float(f.get("xmax")), float(f.get("ymax"))) for f in pg.iter("text")]
            if not frames:
                continue
            m109_pages.append((btitle, idx, W, H, ip, frames, faces, texts))

random.shuffle(m109_pages)
m109_pages = m109_pages[:MANGA_PAGES_FRAME]
koharu_set = set(range(min(len(m109_pages), MANGA_PAGES_KOHARU)))
print(f"manga109 pages: {len(m109_pages)}  koharu on {len(koharu_set)}", flush=True)

# ---- koharu session ----
KO = None
onnx_dir = find("panel-detector-onnx")
if ort is not None and onnx_dir:
    kp = find("rfdetr-seg", isdir=False) or f"{onnx_dir}/koharu-layout/rfdetr-seg-2xlarge.onnx"
    if os.path.exists(kp):
        for prov in (["CUDAExecutionProvider", "CPUExecutionProvider"] if ORT_GPU else [],
                     ["CPUExecutionProvider"]):
            if not prov:
                continue
            try:
                KO = ort.InferenceSession(kp, providers=prov)
                print("koharu providers:", KO.get_providers(), flush=True)
                break
            except Exception as e:
                print("koharu session failed", prov, e, flush=True)
print("koharu:", KO is not None, flush=True)


def koharu_page(im):
    """-> list of (cls, poly)  cls in koharu space 0text 1onom 2bubble 3panel"""
    H, W = im.shape[:2]
    x = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (1152, 1152)).transpose(2, 0, 1)[None].astype(np.float32) / 255
    d, lab, m = KO.run(None, {"input": x})
    lp = 1 / (1 + np.exp(-lab[0]))
    sc, cl = lp.max(1), lp.argmax(1)
    res = []
    for i in np.where(sc > 0.35)[0]:
        mk = (m[0][i] > 0).astype(np.uint8)
        if mk.sum() < 8:
            continue
        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 20:
            continue
        eps = 0.012 * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        ap[:, 0] *= W / 288.0
        ap[:, 1] *= H / 288.0
        res.append((int(cl[i]), ap))
    return res


n_m = 0
for k, (title, idx, W, H, ip, frames, faces, texts) in enumerate(m109_pages):
    sp = "val" if random.random() < 0.04 else "train"
    st = f"m109__{title}__{idx:03d}"
    im = cv2.imread(ip)
    if im is None:
        continue
    ih, iw = im.shape[:2]
    sx, sy = iw / W, ih / H

    pane = []           # (cid, poly[px in image space])
    for (x1, y1, x2, y2) in frames:
        q = np.array([[x1*sx, y1*sy], [x2*sx, y1*sy], [x2*sx, y2*sy], [x1*sx, y2*sy]], np.float32)
        fw, fh = (x2 - x1) * sx, (y2 - y1) * sy
        # Manga109 frames are axis-aligned drawn frames -> only rectangle/square
        # (bleed/outbound can't be read from a frame box; leave that to koharu +
        # the Roboflow webtoon labels)
        ar = fw / max(1.0, fh)
        cid = 1 if (0.80 <= ar <= 1.25 and fw * fh < 0.22 * iw * ih) else 0
        pane.append((cid, q))

    aux = []
    for (x1, y1, x2, y2) in faces:
        aux.append((3, np.array([[x1*sx, y1*sy], [x2*sx, y1*sy], [x2*sx, y2*sy], [x1*sx, y2*sy]], np.float32)))
    for (x1, y1, x2, y2) in texts:
        aux.append((1, np.array([[x1*sx, y1*sy], [x2*sx, y1*sy], [x2*sx, y2*sy], [x1*sx, y2*sy]], np.float32)))

    def _iou_bb(a, b):
        ax1, ay1, ax2, ay2 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
        bx1, by1, bx2, by2 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
        iw_ = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih_ = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw_ * ih_
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0.0

    if KO is not None and k in koharu_set:
        try:
            for kc, poly in koharu_page(im):
                if kc == 3:                     # panel
                    cid = classify(poly, iw, ih, has_border=True)
                    # add only genuine non-rect shapes koharu found that the
                    # axis-aligned frame boxes could not represent
                    if cid in (3, 4) and not any(_iou_bb(poly, fp) > 0.55 for _, fp in pane):
                        pane.append((cid, poly))
                elif kc == 2:
                    aux.append((0, poly))       # bubble
                elif kc == 1:
                    aux.append((2, poly))       # onomatopoeia
                elif kc == 0:
                    aux.append((1, poly))       # text
        except Exception:
            pass

    # COO onomatopoeia for this page
    cf = f"{coo_dir}/{title}__{idx:03d}.txt"
    if os.path.exists(cf):
        for ln in open(cf):
            q = ln.split()
            pts = np.array(q[1:], float).reshape(-1, 2) * [iw, ih]
            aux.append((2, pts))

    if not pane:
        continue
    cv2.imwrite(f"{OUT}/images/{sp}/{st}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEGQ])
    wl(f"{OUT}/labels/{sp}/{st}.txt", [bbox_line(c, p, iw, ih) for c, p in pane])
    wl(f"{OUT}/labels_seg/{sp}/{st}.txt", [seg_line(c, p, iw, ih) for c, p in pane])
    wl(f"{OUT}/labels_aux/{sp}/{st}.txt", [seg_line(c, p, iw, ih) for c, p in aux])
    for c, _ in pane:
        cls_hist[PANEL[c]] += 1
    for c, _ in aux:
        aux_hist[AUX[c]] += 1
    man.append((st, "T5_manga_human" + ("+koharu" if k in koharu_set else ""), title, sp, len(pane), len(aux)))
    n_m += 1
    if (k + 1) % 400 == 0:
        print(f"  m109 {k+1}/{len(m109_pages)} written={n_m}  ({time.time()-t0:.0f}s)", flush=True)
print(f"manga109 done: {n_m}  ({time.time()-t0:.0f}s)", flush=True)

# webtoon aux (ogkalu bubble/text + face) is done inside panel-train-v35-aux,
# which already has GPU + the onnx models and only it needs those labels.

# ============ 4. Magi teacher (optional) ============
magi = find("label-magi") or find("yolo_magi")
n_g = 0
if magi:
    mi = next((p for p in glob.glob(magi + "/**/images", recursive=True) if os.path.isdir(p)), None)
    ml = next((p for p in glob.glob(magi + "/**/labels", recursive=True) if os.path.isdir(p)), None)
    if mi and ml:
        for ip in glob.glob(f"{mi}/*"):
            st = "magi__" + os.path.splitext(os.path.basename(ip))[0][:90]
            lp = f"{ml}/{os.path.splitext(os.path.basename(ip))[0]}.txt"
            if not os.path.exists(lp):
                continue
            if os.path.exists(f"{OUT}/images/train/{st}.jpg") or os.path.exists(f"{OUT}/images/val/{st}.jpg"):
                continue
            im = cv2.imread(ip)
            if im is None:
                continue
            H, W = im.shape[:2]
            web = "__" in os.path.basename(ip) and not os.path.basename(ip).startswith("m109")
            rows_b, rows_s = [], []
            for ln in open(lp):
                q = ln.split()
                if len(q) < 5:
                    continue
                cx, cy, bw, bh = map(float, q[1:5])
                poly = np.array([[cx-bw/2, cy-bh/2], [cx+bw/2, cy-bh/2],
                                 [cx+bw/2, cy+bh/2], [cx-bw/2, cy+bh/2]], np.float32) * [W, H]
                cid = 2 if web else classify(poly, W, H, has_border=True)
                rows_b.append(bbox_line(cid, poly, W, H))
                rows_s.append(seg_line(cid, poly, W, H))
                cls_hist[PANEL[cid]] += 1
            if not rows_b:
                continue
            sp = "val" if np.random.random() < 0.04 else "train"
            cv2.imwrite(f"{OUT}/images/{sp}/{st}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEGQ])
            wl(f"{OUT}/labels/{sp}/{st}.txt", rows_b)
            wl(f"{OUT}/labels_seg/{sp}/{st}.txt", rows_s)
            open(f"{OUT}/labels_aux/{sp}/{st}.txt", "a").close()
            man.append((st, "T7_teacher_magi", "magi", sp, len(rows_b), 0))
            n_g += 1
print(f"magi added: {n_g}", flush=True)

# ============ 5. yamls + manifest + zip ============
import yaml
P5 = ["rectangle", "square", "noborder", "irregular", "outbound"]
yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "nc": 7, "names": PANEL},
               open(f"{OUT}/data.yaml", "w"), sort_keys=False)
yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "nc": 7,
                "names": PANEL, "task": "segment"}, open(f"{OUT}/data_seg.yaml", "w"), sort_keys=False)
yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "nc": 5, "names": P5},
               open(f"{OUT}/data5.yaml", "w"), sort_keys=False)
yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "nc": 4, "names": AUX},
               open(f"{OUT}/data_aux.yaml", "w"), sort_keys=False)
with open(f"{OUT}/manifest.csv", "w") as f:
    f.write("\n".join(",".join(str(x) for x in r) for r in man))

n_tr = len(glob.glob(f"{OUT}/images/train/*"))
n_va = len(glob.glob(f"{OUT}/images/val/*"))
summary = {"train": n_tr, "val": n_va, "panels": cls_hist, "aux": aux_hist,
          "secs": round(time.time() - t0)}
json.dump(summary, open(f"{OUT}/BUILD_SUMMARY.json", "w"), indent=2)
print("SUMMARY", json.dumps(summary, indent=2), flush=True)

# zip in <=1.4GB parts
import zipfile
allf = [os.path.relpath(p, OUT) for p in glob.glob(f"{OUT}/**", recursive=True) if os.path.isfile(p)]
allf.sort()
part, sz, idx = [], 0, 1
LIM = 1_400_000_000


def flush_part(files, idx):
    with zipfile.ZipFile(f"/kaggle/working/v35_part{idx:02d}.zip", "w", zipfile.ZIP_STORED) as z:
        for rel in files:
            z.write(f"{OUT}/{rel}", rel)
    print(f"  wrote v35_part{idx:02d}.zip ({len(files)} files)", flush=True)


for rel in allf:
    s = os.path.getsize(f"{OUT}/{rel}")
    if sz + s > LIM and part:
        flush_part(part, idx)
        idx += 1
        part, sz = [], 0
    part.append(rel)
    sz += s
if part:
    flush_part(part, idx)
print(f"DONE  {n_tr} train / {n_va} val  ({time.time()-t0:.0f}s)", flush=True)
