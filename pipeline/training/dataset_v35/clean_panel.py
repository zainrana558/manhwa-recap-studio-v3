#!/usr/bin/env python3
"""Crop a detected panel for the recap video WITHOUT ever rotating its content.

The recap must always show panels upright — a character must never end up
sideways or upside-down.  So we never deskew.  For a non-rectangular panel
(diagonal / irregular / split) we take the panel polygon's AXIS-ALIGNED
bounding box and blur-fill only the wedges that fall outside the polygon
(they belong to the neighbouring panels), leaving every real pixel in its
original orientation.

    from clean_panel import crop_panel
    img = crop_panel(page_bgr, poly_or_box, cls_name)   # -> BGR crop, upright

poly_or_box : Nx2 pixel polygon from the seg model (preferred) or (x1,y1,x2,y2).
cls_name    : one of rectangle/square/noborder/diagonal/irregular/split/outbound
"""
import cv2
import numpy as np

OUTBOUND_PAD = 0.05      # keep a little of the art that bleeds past the frame
BLEED_CLASSES = {"diagonal", "irregular", "split"}
FEATHER_PX = 7


def _poly(p, W, H):
    a = np.asarray(p, np.float32).reshape(-1, 2)
    if len(a) == 2:
        (x1, y1), (x2, y2) = a
        a = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    a[:, 0] = np.clip(a[:, 0], 0, W - 1)
    a[:, 1] = np.clip(a[:, 1], 0, H - 1)
    return a


def crop_panel(page, poly_or_box, cls_name="rectangle"):
    H, W = page.shape[:2]
    poly = _poly(poly_or_box, W, H)
    x1, y1 = np.floor(poly.min(0)).astype(int)
    x2, y2 = np.ceil(poly.max(0)).astype(int)

    if cls_name == "outbound":
        px, py = int((x2 - x1) * OUTBOUND_PAD), int((y2 - y1) * OUTBOUND_PAD)
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return page[max(0, y1):y2, max(0, x1):x2].copy()

    sub = page[y1:y2, x1:x2].copy()

    # only mask bleed when the panel is genuinely non-rectangular and the polygon
    # actually deviates from its bbox (>4% of the bbox area lies outside it)
    if cls_name in BLEED_CLASSES and len(poly) >= 4:
        loc = poly - [x1, y1]
        m = np.zeros(sub.shape[:2], np.uint8)
        cv2.fillPoly(m, [loc.astype(np.int32)], 255)
        outside = 1.0 - m.mean() / 255.0
        if outside > 0.04:
            mb = cv2.GaussianBlur(m, (0, 0), FEATHER_PX)
            small = cv2.resize(sub, (max(1, sub.shape[1] // 8), max(1, sub.shape[0] // 8)))
            bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (sub.shape[1], sub.shape[0]))
            a = (mb / 255.0)[..., None]
            sub = (sub * a + bg * (1 - a)).astype(np.uint8)
    return sub


if __name__ == "__main__":
    import sys, glob
    # visual check on the assembled val set (all rect-polys -> no masking) +
    # any labels_seg polygon that is non-rectangular
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset_v35"
    n = 0
    for lp in glob.glob(f"{root}/labels_seg/val/*.txt")[:50]:
        ip = lp.replace("/labels_seg/", "/images/").replace(".txt", ".jpg")
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        for ln in open(lp):
            q = ln.split()
            c = int(q[0])
            poly = np.array(q[1:], float).reshape(-1, 2) * [W, H]
            out = crop_panel(im, poly, ["rectangle", "square", "noborder", "diagonal",
                                        "irregular", "split", "outbound"][c])
            n += 1
    print(f"crop_panel ran on {n} panels, no errors")
