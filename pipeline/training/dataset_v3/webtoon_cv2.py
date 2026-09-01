#!/usr/bin/env python3
"""webtoon_cv v2 — the PDF methodology (Section 8) turned into code.

Adds to webtoon_cv.py's pure white-gutter split:
  8.1  horizontal edge-density map  -> gutters read as low-edge-energy bands
       even when the paper is grey / tinted, not pure white
  8.8  Lab palette-change detector  -> panel transitions with NO gutter
       (the dense full-bleed failure: 2 boxes on an 800x8225 strip)
  8.6  hierarchical: coarse gutter blocks first, then subdivide any block
       that is still much taller than its siblings
  8.9  every cut carries a confidence; low-confidence internal cuts are
       dropped and the block is left whole rather than mis-split

Drop-in: same split_webtoon(im) -> [[x1,y1,x2,y2], ...] contract as
webtoon_cv.py, so main()/assemble.py are unchanged.

    .venv/bin/python3 pipeline/training/dataset_v3/webtoon_cv2.py --compare 120
"""
import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "sources"
OUT = HERE / "yolo_webtoon_cv2"
MANGA = {"chainsaw-man", "one-piece", "berserk", "jujutsu-kaisen"}
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _row_signals(im):
    """Per-row signals on the centre band: ink %, dark %, edge, Lab, luma."""
    H, W = im.shape[:2]
    cx0, cx1 = int(W * 0.08), int(W * 0.92)
    band = im[:, cx0:cx1]
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink = (g < 245).mean(axis=1)
    dark = (g < 40).mean(axis=1)
    luma = g.mean(axis=1)                       # 4.2 brightness-jump source
    # 8.1 horizontal gradient magnitude, averaged per row, normalised 0..1
    hx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    vx = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    edge = (hx + vx).mean(axis=1)
    edge = edge / (np.percentile(edge, 95) + 1e-6)
    # 8.8 Lab colour, coarse rows (downsample width for speed)
    small = cv2.resize(band, (32, H), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32).mean(axis=1)
    return ink, dark, edge, lab, luma, (cx0, cx1)


def _runs(mask, min_len):
    """Contiguous True runs of length >= min_len -> list of (start, end)."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _tighten(g_gray, a, b, W):
    """Trim blank margins of a [a,b) row span -> [x1,y1,x2,y2] or None."""
    s = g_gray[a:b]
    rows = np.where((s < 245).mean(axis=1) > 0.010)[0]
    if rows.size < 3:
        return None
    y1, y2 = a + int(rows[0]), a + int(rows[-1]) + 1
    col = g_gray[y1:y2]
    cc = np.where((col < 245).mean(axis=0) > 0.02)[0]
    x1, x2 = (int(cc[0]), int(cc[-1]) + 1) if cc.size > 3 else (0, W)
    return [x1, y1, x2, y2]


def split_webtoon(im, debug=False):
    H, W = im.shape[:2]
    g_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ink, dark, edge, lab, luma, _ = _row_signals(im)

    min_gut = max(24, int(H * 0.020))

    # ---- 8.1: a gutter row is low-ink AND low-edge-energy -----------------
    #   pure-white gutters: ink < .010
    #   tinted/grey gutters: ink can be higher but edge energy stays tiny
    quiet = ((ink < 0.010) & (dark < 0.5)) | ((ink < 0.055) & (edge < 0.045) & (dark < 0.35))
    # close 1-2px speckle so a bubble tail crossing the gutter doesn't break it
    quiet = np.convolve(quiet.astype(np.float32), np.ones(3) / 3, "same") > 0.5

    gut_runs = _runs(quiet, min_gut)
    cuts = [0]
    for a, b in gut_runs:
        if a > 0 and b < H:
            cuts.append((a + b) // 2)
    cuts.append(H)
    cuts = sorted(set(cuts))

    blocks = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        box = _tighten(g_gray, a, b, W)
        if box:
            blocks.append(box)

    if not blocks:
        return []

    # ---- 8.6 + 8.8: subdivide any block that is far taller than one screen
    heights = sorted(bx[3] - bx[1] for bx in blocks)
    med_h = heights[len(heights) // 2]
    # absolute trigger too: a lone full-strip block has no taller sibling to
    # compare against, so key off screen-heights (W ~= one screen wide)
    big_cut = max(1400, min(med_h * 1.7, W * 1.9))

    refined = []
    for x1, y1, x2, y2 in blocks:
        bh = y2 - y1
        if bh < big_cut:
            refined.append([x1, y1, x2, y2])
            continue
        # candidate cuts = local maxima of a combined "boundary-ness" score:
        #   8.1 low edge energy  +  8.8 Lab palette shift  +  4.2 luma jump
        e = edge[y1:y2]
        L = lab[y1:y2]
        lum = luma[y1:y2]
        win = max(9, int(bh * 0.012))
        pal = np.zeros(len(L), np.float32)
        jump = np.zeros(len(L), np.float32)
        for i in range(win, len(L) - win):
            pal[i] = np.linalg.norm(L[i:i + win].mean(0) - L[i - win:i].mean(0))
            jump[i] = abs(lum[i:i + win].mean() - lum[i - win:i].mean())
        pal = pal / (np.percentile(pal, 97) + 1e-6)
        jump = jump / (np.percentile(jump, 97) + 1e-6)
        e_s = np.convolve(e, np.ones(win) / win, "same")
        score = (np.clip(1.0 - e_s, 0, 1) * 0.42
                 + np.clip(pal, 0, 1) * 0.33
                 + np.clip(jump, 0, 1) * 0.25)
        # a cut may land on a quiet row OR on a sharp scene transition, but
        # never mid-way through busy line-art with no palette/brightness event
        local_ink = ink[y1:y2]
        quiet_row = (local_ink < np.percentile(local_ink, 40)) & (e_s < 0.55)
        event_row = ((pal > 0.6) | ((jump > 0.65) & (e_s < 0.45)))
        score = np.where(quiet_row | event_row, score, 0.0)

        # 8.5 caption veto: never cut *through* a tall bright box (narration)
        # floating on darker art. Small margin only, so gutters that merely
        # sit next to a speech bubble stay usable.
        base = np.percentile(lum, 35)
        for ba, bb in _runs(lum > base + 55, max(45, int(bh * 0.02))):
            lo, hi = max(0, ba - 12), min(len(score), bb + 12)
            score[lo:hi] = 0.0

        gap = max(int(bh * 0.10), 480)          # min sub-panel height
        picks = []
        order = np.argsort(score)[::-1]
        for idx in order:
            if score[idx] < 0.55:               # 8.9 confidence floor
                break
            if idx < gap or idx > bh - gap:
                continue
            if all(abs(idx - q) >= gap for q in picks):
                picks.append(int(idx))
            if len(picks) >= 10:
                break
        picks.sort()
        segs = list(zip([0] + picks, picks + [bh]))
        for sa, sb in segs:
            sub = _tighten(g_gray, y1 + sa, y1 + sb, W)
            if sub:
                refined.append(sub)
        if debug:
            print(f"    block {y1}-{y2} h={bh} -> {len(segs)} sub @ {picks}")

    # ---- merge bubble-shard runs, drop tiny (same as v1) -----------------
    min_h = int(H * 0.028)
    merged = []
    for s in sorted(refined, key=lambda b: b[1]):
        if merged and (s[1] - merged[-1][3] < min_gut) and \
           (s[3] - s[1] < min_h or merged[-1][3] - merged[-1][1] < min_h):
            m = merged[-1]
            merged[-1] = [min(m[0], s[0]), m[1], max(m[2], s[2]), s[3]]
        else:
            merged.append(s)
    out = [s for s in merged
           if (s[3] - s[1]) >= max(24, int(H * 0.018)) and (s[2] - s[0]) >= 0.30 * W]
    if not out and blocks:
        y = [min(s[1] for s in blocks), max(s[3] for s in blocks)]
        x = [min(s[0] for s in blocks), max(s[2] for s in blocks)]
        if y[1] - y[0] > 0.1 * H:
            out = [[x[0], y[0], x[1], y[1]]]
    return out


# --------------------------------------------------------------------------
def _iter_pages():
    for sd in sorted(SRC.iterdir()):
        if not sd.is_dir() or sd.name in MANGA:
            continue
        for p in sorted(sd.rglob("*")):
            if p.suffix.lower() in EXTS:
                yield sd.name, p


def _compare(n):
    """A/B box-count: webtoon_cv v1 vs v2 on a stratified sample."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wc1", HERE / "webtoon_cv.py")
    wc1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wc1)

    by_ser = {}
    for ser, p in _iter_pages():
        by_ser.setdefault(ser, []).append(p)
    per = max(4, n // max(len(by_ser), 1))
    rows = []
    (OUT / "cmp").mkdir(parents=True, exist_ok=True)
    for ser, ps in sorted(by_ser.items()):
        step = max(1, len(ps) // per)
        for p in ps[::step][:per]:
            im = cv2.imread(str(p))
            if im is None:
                continue
            b1 = wc1.split_webtoon(im)
            b2 = split_webtoon(im)
            rows.append((ser, p.name, im.shape[0], len(b1), len(b2)))
            if len(rows) <= 40:
                ov = im.copy()
                for x1, y1, x2, y2 in b1:
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (255, 120, 0), 6)
                for x1, y1, x2, y2 in b2:
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 3)
                sc = 1400 / max(ov.shape[0], 1)
                if sc < 1:
                    ov = cv2.resize(ov, (int(ov.shape[1] * sc), int(ov.shape[0] * sc)))
                cv2.imwrite(str(OUT / "cmp" / f"{ser}__{p.stem}.jpg"), ov,
                            [cv2.IMWRITE_JPEG_QUALITY, 72])
    print(f"{'series':20s} {'page':14s} {'H':>6s} {'v1':>4s} {'v2':>4s}  d")
    agg = {}
    for ser, pg, h, n1, n2 in rows:
        print(f"{ser:20s} {pg:14s} {h:6d} {n1:4d} {n2:4d}  {n2-n1:+d}")
        a = agg.setdefault(ser, [0, 0, 0])
        a[0] += n1
        a[1] += n2
        a[2] += 1
    print("\n--- per-series mean boxes/page ---")
    t1 = t2 = 0
    for ser, (s1, s2, c) in sorted(agg.items()):
        print(f"  {ser:20s} v1 {s1/c:4.1f}   v2 {s2/c:4.1f}   ({c} pages)")
        t1 += s1
        t2 += s2
    tc = sum(a[2] for a in agg.values())
    print(f"\n  ALL  v1 {t1/tc:.2f}   v2 {t2/tc:.2f}   over {tc} pages")
    print(f"  overlays: {OUT/'cmp'}  (orange=v1  red=v2)")


def _run_full(overlays):
    for d in ("images", "labels", "overlays"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    pages = list(_iter_pages())
    print(f"{len(pages)} webtoon pages", flush=True)
    t0 = time.time()
    done = fail = tot = empty = nov = 0
    for i, (ser, p) in enumerate(pages):
        stem = f"{ser}__{p.parent.name}__{p.stem}"
        lp = OUT / "labels" / f"{stem}.txt"
        if lp.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            fail += 1
            continue
        H, W = im.shape[:2]
        try:
            boxes = split_webtoon(im)
        except Exception:
            fail += 1
            continue
        cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(lp, "w") as f:
            for x1, y1, x2, y2 in boxes:
                f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                        f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        tot += len(boxes)
        done += 1
        if not boxes:
            empty += 1
        if nov < overlays and boxes and i % 7 == 0:
            ov = im.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 4)
            s = 1100 / max(ov.shape[0], 1)
            if s < 1:
                ov = cv2.resize(ov, (int(ov.shape[1] * s), int(ov.shape[0] * s)))
            cv2.imwrite(str(OUT / "overlays" / f"{stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 78])
            nov += 1
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(pages)} done={done} boxes={tot} empty={empty} "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)
    print(f"DONE {done} pages, {tot} boxes ({tot/max(done,1):.1f}/pg), "
          f"{empty} empty, {fail} failed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=int, metavar="N",
                    help="A/B box-count sample vs webtoon_cv.py, no writes")
    ap.add_argument("--overlays", type=int, default=80)
    args = ap.parse_args()
    if args.compare:
        _compare(args.compare)
    else:
        _run_full(args.overlays)


if __name__ == "__main__":
    main()
