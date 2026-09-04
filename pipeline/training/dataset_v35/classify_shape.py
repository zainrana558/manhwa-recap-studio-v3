#!/usr/bin/env python3
"""Geometry -> 7-class panel taxonomy.

One shared classifier so every polygon-producing source (Roboflow, synthetic,
Magi teacher, Manga109 frames, koharu masks) lands on the same labels:

  0 rectangle  1 square  2 noborder  3 diagonal  4 irregular  5 split  6 outbound

`classify(poly, W, H, has_border=True, split_hint=False)` -> (class_id, name)

poly : list/ndarray of (x, y) image-pixel vertices (>=3). A bbox is just its
       4 corners.
has_border : False when the source says the panel has no gutter/frame (our
       webtoon cascade, koharu "noborder"-ish) -> forces class 2 unless the
       shape is clearly diagonal/irregular.
split_hint : True when the source knows the panel is bisected by a thin seam.
"""
import numpy as np

NAMES = ["rectangle", "square", "noborder", "diagonal", "irregular", "split", "outbound"]

# tunables
ANG_DIAG = 6.0        # deg off horizontal/vertical -> diagonal
CONVEX_IRREG = 0.86   # contour_area / hull_area below this -> irregular
EXTENT_IRREG = 0.72   # contour_area / minAreaRect_area below this -> irregular
SQUARE_LO, SQUARE_HI = 0.78, 1.28
EDGE_FRAC = 0.010     # within 1.0% of page edge counts as touching
OUTBOUND_EDGES = 2    # touch >=2 page edges (and be large) -> outbound


def _poly(poly):
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return p


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _min_area_rect(p):
    """(w, h, angle_deg) of the minimum-area enclosing rectangle (no cv2)."""
    hull = p[_convex_hull_idx(p)]
    best = None
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        L = np.hypot(*edge)
        if L < 1e-6:
            continue
        ux, uy = edge / L
        R = np.array([[ux, uy], [-uy, ux]])
        q = hull @ R.T
        w = q[:, 0].max() - q[:, 0].min()
        h = q[:, 1].max() - q[:, 1].min()
        if best is None or w * h < best[0]:
            ang = np.degrees(np.arctan2(uy, ux))
            best = (w * h, w, h, ang)
    _, w, h, ang = best
    return w, h, ang


def _convex_hull_idx(p):
    pts = sorted(range(len(p)), key=lambda i: (float(p[i][0]), float(p[i][1])))

    def cross(o, a, b):
        return (p[a][0] - p[o][0]) * (p[b][1] - p[o][1]) - (p[a][1] - p[o][1]) * (p[b][0] - p[o][0])

    lower = []
    for i in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def classify(poly, W, H, has_border=True, split_hint=False):
    p = _poly(poly)
    if len(p) < 3:
        return 0, NAMES[0]
    x1, y1 = p[:, 0].min(), p[:, 1].min()
    x2, y2 = p[:, 0].max(), p[:, 1].max()
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pa = _area(p)
    mw, mh, ang = _min_area_rect(p)
    marea = max(1.0, mw * mh)
    hull = p[_convex_hull_idx(p)]
    ha = max(1.0, _area(hull))

    convex = pa / ha
    extent = pa / marea
    off = min(abs(ang) % 90, 90 - (abs(ang) % 90))    # 0..45 deg from axis

    ex, ey = EDGE_FRAC * W, EDGE_FRAC * H
    edges = int(x1 <= ex) + int(y1 <= ey) + int(x2 >= W - ex) + int(y2 >= H - ey)
    big = float(bw * bh) > 0.14 * float(W) * float(H)

    if split_hint:
        return 5, NAMES[5]
    if off >= ANG_DIAG and convex > 0.80:
        return 3, NAMES[3]
    if convex < CONVEX_IRREG or extent < EXTENT_IRREG:
        return 4, NAMES[4]
    if edges >= OUTBOUND_EDGES and big and has_border:
        return 6, NAMES[6]
    if not has_border:
        return 2, NAMES[2]
    ar = bw / bh
    if SQUARE_LO <= ar <= SQUARE_HI and (bw * bh) < 0.30 * W * H:
        return 1, NAMES[1]
    return 0, NAMES[0]


if __name__ == "__main__":
    # sanity
    W = H = 1000
    tests = [
        ([(10, 10), (400, 10), (400, 300), (10, 300)], True, False, "rectangle"),
        ([(100, 100), (300, 100), (300, 300), (100, 300)], True, False, "square"),
        ([(10, 10), (400, 60), (380, 360), (0, 300)], True, False, "diagonal"),
        ([(0, 0), (300, 0), (300, 200), (150, 200), (150, 400), (0, 400)], True, False, "irregular"),
        ([(0, 0), (1000, 0), (1000, 600), (0, 600)], True, False, "outbound"),
        ([(10, 10), (400, 10), (400, 300), (10, 300)], False, False, "noborder"),
    ]
    for poly, hb, sh, want in tests:
        cid, name = classify(poly, W, H, hb, sh)
        print(f"{'OK ' if name == want else 'BAD'} got={name:10s} want={want}")
