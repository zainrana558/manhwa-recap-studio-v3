#!/usr/bin/env python3
"""COO (Comic Onomatopoeia, Manga109 SFX polygons) -> YOLO polygon labels.

COO XML (CC-BY-4.0, github.com/manga109/public-annotations):
  <book title="X"><pages>
    <page height="1170" index="1" width="1654">
      <onomatopoeia id=".." x0=".." y0=".." x1=".." y1=".." ... > text </onomatopoeia>

Each onomatopoeia is a 4..17-point polygon.  Page width/height are in the XML,
so no images are needed for the geometry.  Emits one YOLO-seg line per SFX with
class id `cls` (default 2 = onomatopoeia in the aux taxonomy
bubble/text/onomatopoeia/face).

    python coo_to_yolo.py ext_annotations/m109_public/COO-Comic-Onomatopoeia OUTDIR [cls]

OUTDIR gets <Title>__<index:03d>.txt  (matches Manga109 image naming Title/000.jpg
-> the build kernel maps these onto the real image paths).
"""
import glob, os, sys
import xml.etree.ElementTree as ET


def convert(coo_root, out_dir, cls=2):
    os.makedirs(out_dir, exist_ok=True)
    n_pages = n_sfx = 0
    for xf in glob.glob(os.path.join(coo_root, "**", "*.xml"), recursive=True):
        try:
            root = ET.parse(xf).getroot()
        except ET.ParseError:
            continue
        title = root.get("title") or os.path.splitext(os.path.basename(xf))[0]
        for pg in root.iter("page"):
            W = float(pg.get("width", 0))
            H = float(pg.get("height", 0))
            idx = pg.get("index")
            sfx = list(pg.iter("onomatopoeia"))
            if not (W and H and idx is not None and sfx):
                continue
            lines = []
            for o in sfx:
                pts = []
                i = 0
                while o.get(f"x{i}") is not None:
                    x = float(o.get(f"x{i}")); y = float(o.get(f"y{i}"))
                    pts.append((min(max(x / W, 0), 1), min(max(y / H, 0), 1)))
                    i += 1
                if len(pts) < 3:
                    continue
                lines.append(f"{cls} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in pts))
            if not lines:
                continue
            with open(os.path.join(out_dir, f"{title}__{int(idx):03d}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            n_pages += 1
            n_sfx += len(lines)
    print(f"COO -> {out_dir}: {n_pages} pages, {n_sfx} onomatopoeia polygons")
    return n_pages, n_sfx


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
