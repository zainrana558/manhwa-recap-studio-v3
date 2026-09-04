#!/usr/bin/env python3
"""SFX-only panel cleanup, using koharu-layout RF-DETR's onomatopoeia class.

The one residual the CV heuristics could not catch: a box that is dominated by
stylised sound-effect lettering with little scene content. koharu detects
onomatopoeia directly (class 1). For every panel box: if an onomatopoeia region
covers >= SFX_COV of the box AND koharu finds no strong 'panel' inside it, the
box is merged into the preceding panel on the page (union stays <=3.2 screens)
or dropped if unmergeable.

    WCV2_NO_FACE=1 python sfx_filter.py <in_dir> <out_dir> [wi wn]
      in_dir : a yolo_*/  (images/ + labels/)
"""
import glob, os, shutil, sys
import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "..", "..", "models", "koharu-layout", "rfdetr-seg-2xlarge.onnx")
SFX_COV = 0.45
SESS = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]) if os.path.exists(ONNX) else None


def koharu(im):
    """-> (sfx_boxes, panel_boxes) in image px, from tiled RF-DETR-seg."""
    H, W = im.shape[:2]
    sfx, pan = [], []
    tile = max(1400, W * 2)
    y = 0
    while y < H:
        yb = min(H, y + tile)
        c = im[y:yb]
        ch, cw = c.shape[:2]
        x = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (1152, 1152)).transpose(2, 0, 1)[None].astype(np.float32) / 255
        d, lab, m = SESS.run(None, {"input": x})
        lp = 1 / (1 + np.exp(-lab[0]))
        sc, cl = lp.max(1), lp.argmax(1)
        for i in np.where(sc > 0.30)[0]:
            mk = m[0][i] > 0
            ys, xs = np.where(mk)
            if len(ys) < 6:
                continue
            bx = (xs.min() / 288 * cw, y + ys.min() / 288 * ch,
                  xs.max() / 288 * cw, y + ys.max() / 288 * ch)
            if cl[i] == 1 and sc[i] > 0.35:
                sfx.append(bx)
            elif cl[i] == 3 and sc[i] > 0.4:
                pan.append(bx)
        if yb >= H:
            break
        y += tile - 300
    return sfx, pan


def _ov(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def main():
    ind, outd = sys.argv[1], sys.argv[2]
    wi, wn = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    os.makedirs(f"{outd}/images", exist_ok=True)
    os.makedirs(f"{outd}/labels", exist_ok=True)
    labs = sorted(glob.glob(f"{ind}/labels/*.txt"))
    labs = [x for i, x in enumerate(labs) if i % wn == wi]
    merged = dropped = 0
    for lp in labs:
        stem = os.path.basename(lp)[:-4]
        ip = next((f"{ind}/images/{stem}{e}" for e in (".jpg", ".png", ".webp")
                   if os.path.exists(f"{ind}/images/{stem}{e}")), None)
        im = cv2.imread(ip) if ip else None
        if im is None:
            continue
        H, W = im.shape[:2]
        B = []
        for ln in open(lp):
            f = ln.split()
            if len(f) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in f[1:5])
            B.append([f[0], int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                      int((cx + bw / 2) * W), int((cy + bh / 2) * H)])
        B.sort(key=lambda b: b[2])
        if not B:
            continue
        sfx, pan = ([], [])
        if SESS is not None:
            try:
                sfx, pan = koharu(im)
            except Exception:
                pass
        out = []
        for k, b in enumerate(B):
            _, x1, y1, x2, y2 = b
            area = max(1, (x2 - x1) * (y2 - y1))
            cov = sum(_ov((x1, y1, x2, y2), s) for s in sfx) / area
            has_panel = any(_ov((x1, y1, x2, y2), p) / area > 0.35 for p in pan)
            if cov >= SFX_COV and not has_panel:
                if out and (y1 - out[-1][4]) < 0.55 * W and (y2 - out[-1][2]) < 3.2 * W:
                    o = out[-1]
                    out[-1] = [o[0], min(o[1], x1), o[2], max(o[3], x2), y2]
                    merged += 1
                else:
                    dropped += 1
            else:
                out.append(b)
        if not out:
            continue
        shutil.copy(ip, f"{outd}/images/{stem}{os.path.splitext(ip)[1]}")
        with open(f"{outd}/labels/{stem}.txt", "w") as f:
            for c, x1, y1, x2, y2 in out:
                f.write(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
    print(f"[w{wi}] merged={merged} dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
