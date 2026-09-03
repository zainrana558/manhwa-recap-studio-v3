#!/usr/bin/env python3
"""STRICT QC of the gdrive panel crops -> only perfect crops become YOLO
labels. Key check (rule C): match each crop to its source page, then use
the ogkalu bubble/text detector on the source to reject any crop whose
top or bottom edge cuts THROUGH a speech bubble / caption."""
import csv, glob, os, shutil, sys
import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3")
import webtoon_cv2 as wc  # for _protected_rows (bubble + caption + face row mask)

EX = "/tmp/gdrive_panels/ex"
SRC = "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3/sources"
OUT = "/home/ubuntu/manhwa-recap-studio-v3/pipeline/training/dataset_v3/yolo_gdrive"
QC = "/tmp/gdrive_panels/qc2"
FOLDERS = {"mount-hua": "mount-hua", "nano-machine__1_": "nano-machine",
           "northern-blade_panels": "northern-blade", "sss-suicide-hunter__1_": "sss-suicide-hunter"}
REASONS = ["blank", "fragment", "sliver", "bubble_only", "multi_beat",
           "cut_thru_bubble_top", "cut_thru_bubble_bot", "no_match"]

for d in ("images", "labels"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
for r in REASONS:
    os.makedirs(f"{QC}/rej_{r}", exist_ok=True)
os.makedirs(f"{QC}/kept", exist_ok=True)


def src_path(ser, ch, page):
    for ext in (".webp", ".jpg", ".png", ".jpeg"):
        c = f"{SRC}/{ser}/{ch}/{page}{ext}"
        if os.path.exists(c):
            return c
    return None


def bright_blob_frac(g):
    b = cv2.morphologyEx((g > 234).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
    return 0.0 if n < 2 else max(st[1:, 4]) / float(g.size)


def n_bubble_bands(mask):
    runs, inr = 0, False
    for v in mask:
        if v and not inr:
            runs += 1; inr = True
        elif not v:
            inr = False
    return runs


# ---- group crops by source page ----
by_src = {}
for fld, ser in FOLDERS.items():
    for cf in glob.glob(f"{EX}/{fld}/{ser}/**/*.webp", recursive=True):
        ch = os.path.basename(os.path.dirname(cf))
        page = os.path.basename(cf).split("_")[0]
        sp = src_path(ser, ch, page)
        by_src.setdefault((ser, ch, page, sp), []).append(cf)

log, kept_pages, rej = [], {}, {r: 0 for r in REASONS}
nkept = 0
for i, ((ser, ch, page, sp), crops) in enumerate(sorted(by_src.items())):
    if sp is None:
        for cf in crops:
            log.append([ser, ch, page, os.path.basename(cf), "REJECT", "no_src_page", "", ""])
            rej["no_match"] += 1
        continue
    S = cv2.imread(sp)
    SH, SW = S.shape[:2]
    Sg = cv2.cvtColor(S, cv2.COLOR_BGR2GRAY)
    bub, cap, face = wc._protected_rows(S, SH, SW)
    prot = bub | cap                          # bubbles + captions (faces handled by 'multi_beat'/eyeball)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(by_src)} pages  kept={nkept}", flush=True)

    for cf in sorted(crops):
        pk = os.path.basename(cf).rsplit(".", 1)[0]
        im = cv2.imread(cf)
        if im is None:
            continue
        h, w = im.shape[:2]
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        reason = None
        if (g > 240).mean() > 0.90 or (g < 15).mean() > 0.92:
            reason = "blank"
        elif h < 150 or w < 240 or h * w < 70000:
            reason = "fragment"
        elif w / h > 4.2 or h / w > 5.5:
            reason = "sliver"
        elif bright_blob_frac(g) > 0.40 and (g < 200).mean() < 0.26:
            reason = "bubble_only"

        y1 = y2 = None
        if reason is None:
            cc = im if w == SW else cv2.resize(im, (SW, int(h * SW / w)))
            if cc.shape[0] > SH:
                reason = "no_match"
            else:
                res = cv2.matchTemplate(Sg, cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if mx < 0.93:
                    reason = "no_match"
                else:
                    y1, y2 = loc[1], loc[1] + cc.shape[0]

        if reason is None:
            # rule C: does either edge cut THROUGH a bubble/caption in the source?
            for yy, tag in ((y1, "cut_thru_bubble_top"), (y2, "cut_thru_bubble_bot")):
                a, b = max(0, yy - 3), min(SH, yy + 3)
                # a straddle = protected rows on BOTH sides of the cut within +-22px
                up = prot[max(0, yy - 22):yy].any()
                dn = prot[yy:min(SH, yy + 22)].any()
                if up and dn and prot[a:b].any():
                    reason = tag
                    break

        if reason is None:
            # multi-beat: >=2 separated bubble bands inside a tall crop
            inside = prot[y1:y2]
            if h / w > 2.4 and n_bubble_bands(inside) >= 2:
                reason = "multi_beat"

        if reason:
            rej[reason] = rej.get(reason, 0) + 1
            shutil.copy(cf, f"{QC}/rej_{reason}/{ser}__{ch}__{pk}.webp")
            log.append([ser, ch, page, pk, "REJECT", reason, y1 or "", y2 or ""])
            continue
        kept_pages.setdefault((ser, ch, page, sp, SW, SH), []).append((0, y1, SW, y2))
        shutil.copy(cf, f"{QC}/kept/{ser}__{ch}__{pk}.webp")
        log.append([ser, ch, page, pk, "KEEP", "", y1, y2])
        nkept += 1

np_ = 0
for (ser, ch, page, sp, SW, SH), boxes in kept_pages.items():
    boxes = sorted(set(boxes))
    stem = f"{ser}__{ch}__{page}"
    shutil.copy(sp, f"{OUT}/images/{stem}{os.path.splitext(sp)[1]}")
    with open(f"{OUT}/labels/{stem}.txt", "w") as f:
        for x1, yy1, x2, yy2 in boxes:
            f.write(f"0 {(x1+x2)/2/SW:.6f} {(yy1+yy2)/2/SH:.6f} {(x2-x1)/SW:.6f} {(yy2-yy1)/SH:.6f}\n")
    np_ += 1

with open(f"{QC}/qc_log.csv", "w", newline="") as f:
    csv.writer(f).writerows([["series", "chapter", "page", "panel", "verdict", "reason", "y1", "y2"]] + log)

tot = len(log)
print(f"\ntotal {tot} | KEPT {nkept} ({100*nkept/tot:.0f}%) | rejected {tot-nkept}")
for r, n in sorted(rej.items(), key=lambda x: -x[1]):
    if n:
        print(f"  {r:22s} {n}")
print(f"\nYOLO pages: {np_}  -> {OUT}")
bs = {}
for (ser, *_), b in kept_pages.items():
    bs.setdefault(ser, [0, 0]); bs[ser][0] += 1; bs[ser][1] += len(b)
for s, (p, b) in sorted(bs.items()):
    print(f"  {s:20s} {p} pages, {b} panels")
