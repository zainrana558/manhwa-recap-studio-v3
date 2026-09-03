# v3.4 — multi-class webtoon panel detector (ready to train)

Incorporates the 5 items from the model research:

| # | from | what | where |
|---|------|------|-------|
| 1 | manhwa/webtoon search | **7→3 class taxonomy** `rect / noborder / irregular` | `curate/assemble_v34.py` |
| 2 | manhwa/webtoon search | **238 human-labelled Roboflow webtoon pages** (the only Diagonal/Irregular/Split/Outbound data) | `roboflow_webtoon/_merged/` |
| 3 | manhwa/webtoon search | RF-DETR-seg noted as the right arch for irregular panels — **not adopted yet** (no polygon labels available; the Roboflow export is bbox-only). Kept as a v3.5 option. | — |
| 4 | manga search | **koharu-layout RF-DETR onomatopoeia class** → SFX-only panel cleanup | `models/koharu-layout/` + `curate/sfx_filter.py` |
| 5 | manga search | **Magi v2 as a label teacher** | `kaggle/label_magi/` |

## Current state — ready to train NOW

`_v34_trainset/` is built: **5,018 train / 628 val imgs, ~17k boxes**
class balance: `rect 17,785 · noborder 3,293 · irregular 428` (irregular is scarce — only 238 source pages have it; the `cls=1.2` loss weight in the kernel compensates)

## Run sequence

### A. Train v3.4 immediately (no Magi)
```bash
cd pipeline/training/dataset_v3
# push the current _v34_trainset to the webtoon-yolo dataset
rsync -a --delete _v34_trainset/images/ _train_ds/images/
rsync -a --delete _v34_trainset/labels/ _train_ds/labels/
cp _v34_trainset/data.yaml _train_ds/data.yaml
cd _train_ds && kaggle datasets version -p . -m "v3.4 3-class taxonomy" --dir-mode zip --delete-old-versions
cd ../kaggle/train_v34 && kaggle kernels push -p .        # -> panel-train-v34
```

### B. (optional, better) Magi teacher first → v3.4 with more labels
```bash
cd pipeline/training/dataset_v3/kaggle/label_magi
kaggle kernels push -p .                                   # -> label-magi  (GPU, ~3-5h)
# on COMPLETE:
kaggle kernels output zainrana1122/label-magi -p /tmp/magi
mkdir -p ../../yolo_magi && cd ../../yolo_magi
unzip -o /tmp/magi/yolo_magi_images.zip && unzip -o /tmp/magi/yolo_magi_labels.zip
cd ../curate && python assemble_v34.py --with-magi --out ../_v34_trainset
# then step A from the rsync line
```

### C. (optional) SFX cleanup before assembling
```bash
cd pipeline/training/dataset_v3/curate
for w in 0 1 2; do WCV2_NO_FACE=1 python sfx_filter.py ../yolo_final3 ../yolo_final3_sfx $w 3 & done; wait
# then point assemble_v34.py's yolo_final3 -> yolo_final3_sfx
```

## Notes
- `panel-train-v34` kernel: yolo11m (up from 11s), 200ep, imgsz 1024, AdamW, `cls=1.2`.
- koharu classes: `0 text · 1 onomatopoeia · 2 bubble · 3 panel`. Decode = sigmoid(labels), bbox from `mask > 0`.
- Held-out val series: `tbate` + 20% of the Roboflow webtoon pages.
- v3.3 (`panel-train-v33`, yolo11s, 2-class panel+bubble) is the current baseline — compare v3.4 against it and prod before installing.
