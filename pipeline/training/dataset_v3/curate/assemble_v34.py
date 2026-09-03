#!/usr/bin/env python3
"""Assemble the v3.4 training set — multi-class webtoon panel taxonomy.

Classes (3, collapsed from the Roboflow Webtoon-Manhwa 7-class taxonomy so the
rare types have enough support):
  0 rect      bordered rectangular/square panel  (manga pages, Western comics)
  1 noborder  full-width borderless webtoon band (the webtoon default)
  2 irregular diagonal / split / outbound / non-rectangular panel

Sources
  yolo_final3      curated webtoon panels (our cascade)  -> 1 noborder
  yolo_kumiko      Kumiko manga frames                   -> 0 rect
  yolo_roboflow    Roboflow comic/manga panel sets       -> 0 rect
  roboflow_webtoon 238 human-labelled webtoon pages      -> mapped per their 7 classes
  yolo_magi        (optional) Magi teacher boxes          -> 0/1 by border heuristic

    python assemble_v34.py --out _v34_trainset
"""
import argparse, glob, hashlib, os, random, shutil
from pathlib import Path
import cv2

HERE = Path(__file__).resolve().parent.parent
RB7 = ["Diagonal Panel", "Irregular Panel", "Noborder Rectangle Panel",
       "Outbound Rectangle Panel", "Rectangle Panel", "Split Panel", "Square Panel"]
RB7_TO_3 = {0: 2, 1: 2, 2: 1, 3: 2, 4: 0, 5: 2, 6: 0}
VAL_SERIES = {"tbate"}
CAP = 500


def phash(p):
    im = cv2.imread(p)
    if im is None:
        return None
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (16, 16))
    return hashlib.md5((g > g.mean()).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "_v34_trainset"))
    ap.add_argument("--with-magi", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    # (priority, series, imgpath, labelpath, class_map)   class_map: fn(old_cls)->new_cls
    cands = []
    for D, cm, tag in [
        (HERE / "yolo_final3", lambda c: 1, "wt"),
        (HERE / "yolo_kumiko", lambda c: 0, "km"),
        (HERE / "yolo_roboflow", lambda c: 0, "rf"),
    ]:
        for ip in glob.glob(f"{D}/images/*"):
            st = Path(ip).stem
            lp = D / "labels" / f"{st}.txt"
            if lp.exists():
                cands.append((st.split("__")[0], ip, str(lp), cm))
    # Roboflow webtoon taxonomy (the only Diagonal/Irregular/Split/Outbound data)
    RBW = HERE / "roboflow_webtoon" / "_merged"
    for ip in glob.glob(f"{RBW}/images/*"):
        st = Path(ip).stem
        lp = RBW / "labels" / f"{st}.txt"
        if lp.exists():
            cands.append(("rbw", ip, str(lp), lambda c: RB7_TO_3.get(int(c), 0)))
    if a.with_magi and (HERE / "yolo_magi").exists():
        for ip in glob.glob(f"{HERE}/yolo_magi/images/*"):
            st = Path(ip).stem
            lp = HERE / "yolo_magi" / "labels" / f"{st}.txt"
            if lp.exists():
                cands.append(("magi_" + st.split("__")[0], ip, str(lp), lambda c: 1))

    rng.shuffle(cands)
    seen, per = set(), {}
    ntr = nva = btr = bva = 0
    cls_tot = {}
    for ser, ip, lp, cm in cands:
        if per.get(ser, 0) >= CAP:
            continue
        rows = []
        for ln in open(lp):
            f = ln.split()
            if len(f) < 5:
                continue
            nc = cm(f[0])
            rows.append(f"{nc} {' '.join(f[1:5])}")
            cls_tot[nc] = cls_tot.get(nc, 0) + 1
        if not rows:
            continue
        h = phash(ip)
        if h is None or h in seen:
            continue
        seen.add(h)
        per[ser] = per.get(ser, 0) + 1
        sp = "val" if (ser in VAL_SERIES or ser == "rbw" and rng.random() < 0.2
                       or rng.random() < 0.07) else "train"
        name = Path(ip).stem[:120]
        shutil.copy(ip, out / "images" / sp / f"{name}{Path(ip).suffix}")
        open(out / "labels" / sp / f"{name}.txt", "w").write("\n".join(rows) + "\n")
        if sp == "train":
            ntr += 1
            btr += len(rows)
        else:
            nva += 1
            bva += len(rows)

    import yaml
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                    "nc": 3, "names": ["rect", "noborder", "irregular"]},
                   open(out / "data.yaml", "w"))
    print(f"train {ntr} imgs / {btr} boxes | val {nva} / {bva}")
    print(f"class totals (0 rect / 1 noborder / 2 irregular): {cls_tot}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
