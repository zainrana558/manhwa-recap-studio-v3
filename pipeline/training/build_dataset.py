#!/usr/bin/env python3
"""Assemble a YOLO dataset for the webtoon panel/bubble detector.

Merges:
  - Roboflow "Webtoon Panel" + "webtoon-manhwa-panels" (needs ROBOFLOW_API_KEY)
  - any --extra folders in YOLO layout (images/ + labels/) — e.g. the output of
    autolabel_bootstrap.py (after correction) and align_frames_to_source.py

    python build_dataset.py --out data/webtoon-yolo \
        [--extra data/webtoon-yolo/bootstrap data/webtoon-yolo/recap-aligned] \
        [--val-frac 0.12]

Writes <out>/{images,labels}/{train,val}/ and <out>/data.yaml.

All source class ids are remapped to: 0 = panel, 1 = bubble.  Roboflow sets that
only label panels contribute class 0 only; that's fine.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

CLASSES = ["panel", "bubble"]

# how each known source's class ids map onto ours; anything unlisted -> panel(0)
ROBOFLOW_SETS = [
    # (workspace, project, version)  — public, YOLOv8 format
    ("teste-lk8f9", "webtoon-panel", None),
]


def _pull_roboflow(out_raw: Path):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("  ROBOFLOW_API_KEY not set — skipping Roboflow pull "
              "(add --extra folders instead, or set the key)")
        return []
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    dirs = []
    for ws, proj, ver in ROBOFLOW_SETS:
        p = rf.workspace(ws).project(proj)
        v = p.version(ver) if ver else p.versions()[0]
        ds = v.download("yolov8", location=str(out_raw / proj))
        dirs.append(Path(ds.location))
    return dirs


def _iter_pairs(root: Path):
    """yield (image_path, label_path) for a YOLO-layout folder (recursive)."""
    for img in root.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lbl.exists():
            lbl = img.with_suffix(".txt")
        yield img, (lbl if lbl.exists() else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    roots = _pull_roboflow(out / "_raw") + [Path(e) for e in args.extra]
    pairs = []
    for r in roots:
        if not r.exists():
            print(f"  missing: {r}")
            continue
        got = list(_iter_pairs(r))
        pairs += got
        print(f"  {r}: {len(got)} images")

    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_frac)
    n = 0
    for i, (img, lbl) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        stem = f"{img.parent.parent.name}_{img.stem}_{i}"
        shutil.copy(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
        (out / "labels" / split / f"{stem}.txt").write_text(
            lbl.read_text() if lbl else "")
        n += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )
    print(f"\n{n} images ({n_val} val) -> {out}\n  data.yaml written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
