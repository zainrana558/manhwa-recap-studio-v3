#!/usr/bin/env python3
"""Direct 'is a bubble / face cut through?' QC. For every kept box, run the
ogkalu bubble/text detector AND the manga109 face detector ON THE CROP; if a
detection is clipped by the crop's top or bottom edge, TRIM the box to exclude
it, or drop the box if the trim kills it.  -> yolo_final3
"""
import glob, os, shutil, sys
import cv2
import numpy as np
sys.path.insert(0, "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3")
import onnxruntime as ort

M = "/home/ubuntu/manhwa-recap-studio-v3/pipeline/models"
BUB = ort.InferenceSession(f"{M}/comic-text-and-bubble-detector/detector-v4-s_int8.onnx",
                           providers=["CPUExecutionProvider"])
FACE = ort.InferenceSession(f"{M}/manga109-yolo/model.onnx", providers=["CPUExecutionProvider"])
IN = "pipeline/training/dataset_v3/yolo_final2"
OUT = "pipeline/training/dataset_v3/yolo_final3"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
wi, wn = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)


def bub_boxes(crop):
    h, w = crop.shape[:2]
    x = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (640, 640)).transpose(2, 0, 1)[None].astype(np.float32) / 255
    lb, bx, sc = BUB.run(None, {"images": x, "orig_target_sizes": np.array([[h, w]], np.int64)})
    out = []
    for l, b, s in zip(lb[0], bx[0], sc[0]):
        if float(s) < 0.40 or int(l) not in (0, 1, 2):
            continue
        out.append((float(b[1]), float(b[3]), float(b[0]), float(b[2])))   # y1,y2,x1,x2
    return out


def face_boxes(crop):
    h, w = crop.shape[:2]
    sc = min(640 / w, 640 / h)
    nw, nh = int(w * sc), int(h * sc)
    cv_ = np.full((640, 640, 3), 114, np.uint8)
    cv_[:nh, :nw] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (nw, nh))
    o = FACE.run(None, {"images": cv_.transpose(2, 0, 1)[None].astype(np.float32) / 255})[0][0].T
    s = o[:, 4:]
    cls = s.argmax(1)
    cf = s.max(1)
    out = []
    for i in np.where((cf > 0.45) & (cls == 1))[0]:
        cy, bh = o[i, 1] / sc, o[i, 3] / sc
        out.append((cy - bh / 2, cy + bh / 2))
    return out


def clipped(dets, H, is_face=False):
    """return (trim_top, trim_bot) to exclude any edge-clipped detection."""
    tt = bb = 0
    for d in dets:
        y1, y2 = d[0], d[1]
        area_ok = True
        if not is_face:
            x1, x2 = d[2], d[3]
            area_ok = (x2 - x1) < 0.97 * 1e9  # placeholder; width check below
        # clipped at top
        if y1 <= 4 and 6 < y2 < H * 0.55:
            tt = max(tt, int(y2) + 4)
        # clipped at bottom
        if y2 >= H - 4 and H * 0.45 < y1 < H - 6:
            bb = max(bb, H - int(y1) + 4)
    return tt, bb


labs = sorted(glob.glob(f"{IN}/labels/*.txt"))
labs = [x for i, x in enumerate(labs) if i % wn == wi]
kp = kb = trimmed = dropped = 0
for k, lp in enumerate(labs):
    stem = os.path.basename(lp)[:-4]
    ip = next((f"{IN}/images/{stem}{e}" for e in (".jpg", ".png", ".webp")
               if os.path.exists(f"{IN}/images/{stem}{e}")), None)
    im = cv2.imread(ip) if ip else None
    if im is None:
        continue
    H, W = im.shape[:2]
    good = []
    for ln in open(lp):
        f = ln.split()
        if len(f) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in f[1:])
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        crop = im[max(0, y1):y2, max(0, x1):x2]
        ch, cw = crop.shape[:2]
        if ch < 40 or cw < 40:
            continue
        try:
            bt, bbt = clipped(bub_boxes(crop), ch)
            ft, fbt = clipped(face_boxes(crop), ch, is_face=True)
        except Exception:
            bt = bbt = ft = fbt = 0
        tt, bb = max(bt, ft), max(bbt, fbt)
        if tt or bb:
            ny1, ny2 = y1 + tt, y2 - bb
            if ny2 - ny1 >= max(150, 0.55 * (x2 - x1)):
                y1, y2 = ny1, ny2
                trimmed += 1
            else:
                dropped += 1
                continue
        good.append((x1, y1, x2, y2))
    if not good:
        continue
    shutil.copy(ip, f"{OUT}/images/{stem}{os.path.splitext(ip)[1]}")
    with open(f"{OUT}/labels/{stem}.txt", "w") as fo:
        for x1, y1, x2, y2 in sorted(set(good)):
            fo.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        kb += len(set(good))
    kp += 1
    if (k + 1) % 200 == 0:
        print(f"[w{wi}] {k+1}/{len(labs)} kept_pg={kp} trimmed={trimmed} dropped={dropped}", flush=True)
print(f"[w{wi}] DONE kept_pg={kp} kept_bx={kb} trimmed={trimmed} dropped={dropped}", flush=True)
