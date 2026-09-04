# v3.5 — webtoon / manhwa panel dataset + detector

Everything from the v3.3/v3.4 work + the worldwide research, combined into one
clean, provenanced dataset with a **7-class panel taxonomy**, **instance
polygons**, and an **aux (bubble / text / SFX / face) layer** — plus the
pipeline change that actually fixes diagonal/irregular slices.

## What changed since v3.4

| area | v3.4 | v3.5 |
|---|---|---|
| taxonomy | 3-class (rect/noborder/irregular) | **7-class in the label files** (rectangle, square, noborder, diagonal, irregular, split, outbound); **trained as 5** (diagonal+split → irregular) because the pipeline never rotates content |
| labels | boxes | **boxes + instance polygons** (`labels_seg/`) → YOLO-seg / RF-DETR |
| non-rect panels | axis-aligned box only | seg mask → `clean_panel.crop_panel()`: crop the polygon's upright bbox, blur-fill the neighbour wedges, **never deskew / rotate** |
| SFX | koharu onon manga only, ad-hoc | **COO — 61,465 onomatopoeia polygons** (Manga109, CC-BY-4.0) folded into the aux layer |
| faces / bubbles | 3-ONNX cascade at slice time (slow) | **one `panel-train-v35-aux` model** (bubble/text/onomatopoeia/face) replaces the cascade |
| manga | inline `<frame>` parse | `<frame>` → rect/square/outbound by geometry + **koharu RF-DETR-seg** teacher → real diagonal/irregular + `<face>`/`<text>` → aux |
| teacher | Magi v2 (optional) | Magi v2 (optional) + koharu-seg (manga) |
| synthetic tilted pages | — | **tried and rejected** — rotated fake content teaches the detector artifacts and risks sideways panels in the video |

## Taxonomy

`labels/` and `labels_seg/` are **7-class**:

`0 rectangle · 1 square · 2 noborder · 3 diagonal · 4 irregular · 5 split · 6 outbound`

Training collapses to **5** (`labels5/`, `data5.yaml`, and every train kernel):

`0 rectangle · 1 square · 2 noborder · 3 irregular · 4 outbound`   (diagonal, split → irregular)

Why: the recap pipeline crops a panel to its polygon's **axis-aligned** box and
blur-fills the bleed — it never rotates content — so `diagonal` vs `irregular`
makes no functional difference at slice time. The 7-class labels are kept so a
future oriented-box / deskew model can use them.

## Local build — `assemble_v35.py` → `dataset_v35/`

| tier | source | → class | count |
|---|---|---|---|
| T1 webtoon_cascade | `dataset_v3/yolo_final3` (our CV+cascade curation) | noborder | 2536 pg |
| T2 webtoon_human | `dataset_v3/roboflow_webtoon/_merged` (95 hand-labelled webtoon pages ×2.5 aug) | 7-class | 238 pg |
| T3 comic_human | `dataset_v3/yolo_roboflow` (Roboflow comic panels, capped) | rectangle | 1000 pg |
| T4 manga_cv | `dataset_v3/yolo_kumiko` (Kumiko CV, capped) | rectangle | 600 pg |

Boxes are clamped to the page; degenerate/off-page boxes dropped. phash dedup on
T1/T3/T4 (T2 kept whole). Held-out val = **`tbate`** (whole series) + 15 % of T2
+ 5 % random — val stays webtoon-weighted so `best.pt` is selected on webtoon
performance.

Outputs: `images/ labels/(7) labels_seg/(7) labels5/(5) labels_aux/(empty)
coco/ data.yaml data5.yaml data_seg.yaml data_aux.yaml manifest.csv
ext/coo/(xml) ext/coo_yolo/(8718 pre-converted) ext/dialog/`

`python check_v35.py` — 30 invariants (pairing, ranges, class map, dedup, val
policy, COCO, yaml). Must print `0 ERRORS`.

## Kaggle stage

```
# 0. one-time uploads
kaggle datasets create -p dataset_v35            # -> zainrana1122/webtoon-panels-v35-src
kaggle datasets create -p <onnx models dir>      # -> zainrana1122/panel-detector-onnx
#    (koharu-layout/ comic-text-and-bubble-detector/ manga109-yolo/ anime-face/)

# 1. (optional) Magi teacher
cd ../dataset_v3/kaggle/label_magi && kaggle kernels push -p .      # -> label-magi

# 2. build the full dataset  (adds Manga109 + COO + koharu-seg teacher)
cd kaggle/build_v35 && kaggle kernels push -p .                    # -> build-v35 (GPU ~1.5h)
kaggle kernels output zainrana1122/build-v35 -p /tmp/v35 && cd /tmp/v35
for z in v35_part*.zip; do unzip -o "$z" -d v35_merged; done
kaggle datasets create -p v35_merged                              # -> zainrana1122/webtoon-panels-v35

# 3. train
cd kaggle/train_v35_seg    && kaggle kernels push -p .   # YOLO11m-seg  5-class  (PRIMARY)
cd kaggle/train_v35_det    && kaggle kernels push -p .   # YOLO11m      5-class  (baseline)
cd kaggle/train_v35_aux    && kaggle kernels push -p .   # YOLO11s-seg  bubble/text/sfx/face
cd kaggle/train_v35_rfdetr && kaggle kernels push -p .   # RF-DETR      5-class  (alt arch)
```

`build_v35.py` adds, from images that only live on Kaggle:

* **Manga109 `<frame>`** (≤3200 pg) → `classify_shape` geometry → rectangle /
  square / outbound. This is the real source of the `square` class.
* **koharu RF-DETR-seg** on ≤2400 of those pages → panel masks → contour →
  `classify_shape` → **diagonal / irregular** (koharu scores 0.97-0.99 on manga;
  it is unusable on webtoons, so it is not run on them).
* **Manga109 `<face>` / `<text>`** → aux.
* **COO** onomatopoeia polygons → aux (mapped onto the Manga109 images).
* **Magi v2** output (if present) → extra panel pages.

Webtoon aux (ogkalu bubble/text + manga109-yolo face) is generated inside
`panel-train-v35-aux` itself.

## Pipeline integration

`clean_panel.crop_panel(page, polygon, cls_name)` — the slicer calls this with
the seg model's polygon. Rectangle/square/noborder/outbound → plain axis-aligned
crop (outbound padded outward 5 %). diagonal/irregular/split → axis-aligned crop
of the polygon bbox with the out-of-polygon wedges feathered into a blurred
background. **Content orientation is never changed.**

`classify_shape.classify(poly, W, H, has_border)` — shared 7-class geometry
classifier (angle / convexity / extent / edge-touch / aspect). Used by the build
kernel for every polygon-producing teacher.

## Known limitations / access-request items

* `diagonal` (16) and `split` (16) locally — padded only by koharu on manga +
  the 95 Roboflow pages. Trained as `irregular`; if a dedicated oriented-box
  model is ever wanted, get more of these first.
* `square` comes entirely from Manga109 geometry — verify after `build_v35`.
* **MangaSeg** (`MS92/MangaSegmentation`, 700k masks) — HF-gated, request access;
  would give real manga panel polygons at scale for an RF-DETR-seg upgrade.
* **PopManga** (`ragavsachdeva/popmanga_test`) — HF-gated.
* **DCM772** + **eBDtheque** — registration only (`iapr-tc10.univ-lr.fr`,
  `ebdtheque.univ-lr.fr/registration`); western comics, rich in diagonal /
  irregular panels, but bbox-only.
* Onomatopoeia aux is Manga109 (Japanese) only — no labelled Korean/English
  webtoon SFX exists.
