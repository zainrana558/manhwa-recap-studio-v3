#!/usr/bin/env python3
"""Rebuild MULTI-panel webtoon training pages from the cascade slices.

yolo_final3 is 86 % single-box images — a detector trained on it learns "find
THE panel", not "divide a layout", which is exactly why v3.2/v3.3 merge and
split on real strips.  The slices are consecutive vertical chunks of the
original strip, so stacking 2-4 back together gives a real multi-panel page
with real (offset) panel boxes.

    python restitch_pages.py            ->  yolo_restitch/{images,labels}
"""
import glob, os, re, random, collections
import cv2
import numpy as np

D3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_v3")
SRC = os.path.join(D3, "yolo_final3")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_v3", "yolo_restitch")
MAX_SLICES = 5          # keep aspect <= ~1:4 so it trains at imgsz 1152 ok
MAX_H_OVER_W = 5.5
PER_CHAPTER = 16
SEED = 7


def boxes(stem, W, H):
    out = []
    lp = f"{SRC}/labels/{stem}.txt"
    if not os.path.exists(lp):
        return out
    for ln in open(lp):
        q = ln.split()
        if len(q) < 5:
            continue
        cx, cy, bw, bh = map(float, q[1:5])
        out.append([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H])
    return out


def main():
    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/labels", exist_ok=True)
    rng = random.Random(SEED)

    by_ch = collections.defaultdict(list)
    for f in glob.glob(f"{SRC}/images/*.jpg"):
        b = os.path.basename(f)[:-4]
        m = re.match(r"(.+__chapter_\d+)__(\d+)$", b)
        if m:
            by_ch[m.group(1)].append((int(m.group(2)), b))

    n_pages = n_boxes = 0
    hist = collections.Counter()
    for ch, items in by_ch.items():
        items.sort()
        nums = [n for n, _ in items]
        stem_of = {n: s for n, s in items}
        # consecutive runs
        runs, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if j > i:
                runs.append(nums[i:j + 1])
            i = j + 1

        made = 0
        windows = []
        for run in runs:
            k = 2
            while k <= min(MAX_SLICES, len(run)):
                for a in range(0, len(run) - k + 1, max(1, k - 1)):
                    windows.append(run[a:a + k])
                k += 1
        rng.shuffle(windows)
        for win in windows:
            if made >= PER_CHAPTER:
                break
            imgs, W = [], None
            ok = True
            for num in win:
                ip = f"{SRC}/images/{stem_of[num]}.jpg"
                im = cv2.imread(ip)
                if im is None:
                    ok = False
                    break
                W = im.shape[1] if W is None else min(W, im.shape[1])
                imgs.append(im)
            if not ok or W is None:
                continue
            # normalise width, stack, offset boxes
            parts, all_b, y = [], [], 0
            for num, im in zip(win, imgs):
                h0, w0 = im.shape[:2]
                if w0 != W:
                    im = cv2.resize(im, (W, round(h0 * W / w0)), interpolation=cv2.INTER_AREA)
                h = im.shape[0]
                for x1, y1, x2, y2 in boxes(stem_of[num], w0, h0):
                    s = W / w0
                    all_b.append([x1 * s, y1 * s + y, x2 * s, y2 * s + y])
                parts.append(im)
                y += h
            page = np.vstack(parts)
            H = page.shape[0]
            if H / W > MAX_H_OVER_W or not all_b:
                continue
            name = f"rs_{ch}__{win[0]:03d}x{len(win)}"
            cv2.imwrite(f"{OUT}/images/{name}.jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(f"{OUT}/labels/{name}.txt", "w") as fo:
                for x1, y1, x2, y2 in all_b:
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                    fo.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
                    n_boxes += 1
            hist[len(win)] += 1
            n_pages += 1
            made += 1
    print(f"restitched pages: {n_pages}  boxes: {n_boxes}  by slice-count: {dict(hist)}")
    print(f"-> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
