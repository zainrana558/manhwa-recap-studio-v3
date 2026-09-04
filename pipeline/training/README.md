# Webtoon panel detector — training

Goal: replace `pipeline/models/manga-panel-yolo/manga_panel_detector_fp32_1024.onnx`
(YOLO26n trained on **bordered print manga**, Manga109-s) with one trained on
**webtoons / manhwa** so the reference-style slicer stops leaving tall strips and
under-splitting dialogue-dense pages.

## Why not train here

This box is **2 CPU, 11 GB RAM, no GPU**. Inference is fine (we already run YOLO
ONNX at ~30 ms/page). **Training is not** — fine-tuning YOLOv8n on ~3–5k images is
~1–2 h on any GPU but **days** on 2 CPUs. Run the notebook on a **free Colab or
Kaggle GPU** (T4 / P100), download the ~12 MB `best.onnx`, drop it in
`pipeline/models/manga-panel-yolo/` (keep the same filename or set
`PANEL_YOLO_PATH`).

## Data sources (in priority order)

1. **Roboflow "Webtoon Panel"** — 2,836 labelled webtoon images, YOLO-export,
   free. `build_dataset.py` pulls it (needs a free Roboflow API key in
   `ROBOFLOW_API_KEY`). Plus the smaller `webtoon-manhwa-panels` sets.
2. **Our own output, auto-labelled then corrected** — `autolabel_bootstrap.py`
   runs the *current* detector + the RT-DETR bubble detector over raw AsuraScans /
   Gatekeeper chapters and writes YOLO `.txt` labels. Open the folder in
   [Label Studio](https://labelstud.io) / CVAT / Roboflow, fix the ~20 % that are
   wrong, keep ~300–500. This is the **highest-value** data — it is exactly our
   distribution.
3. **YouTube manhwa-recap frames** — `extract_recap_frames.py` +
   `align_frames_to_source.py`. A recap channel shows one human-cropped panel per
   beat; if we also have the raw chapter, template-matching the frame back to its
   page recovers a **ground-truth panel box for free, at scale**. Send the video
   URLs + which manhwa each covers.

## Steps

```bash
# 0. (once) free Roboflow API key -> export ROBOFLOW_API_KEY=...
pip install ultralytics roboflow yt-dlp

# 1. pull the labelled webtoon sets
python pipeline/training/build_dataset.py --out data/webtoon-yolo

# 2. bootstrap-label our own chapters (then correct in a labelling tool)
python pipeline/training/autolabel_bootstrap.py \
    --chapters ~/chapters/solo-leveling ~/chapters/gatekeeper \
    --out data/webtoon-yolo/bootstrap

# 3. (optional) recap frames -> labels
python pipeline/training/extract_recap_frames.py --url "<yt url>" --out data/recap/solo-leveling
python pipeline/training/align_frames_to_source.py \
    --frames data/recap/solo-leveling --source ~/chapters/solo-leveling \
    --out data/webtoon-yolo/recap-aligned

# 4. train (on Colab/Kaggle GPU — open train_webtoon_yolo.ipynb)
#    -> produces best.onnx

# 5. deploy
cp best.onnx pipeline/models/manga-panel-yolo/manga_panel_detector_fp32_1024.onnx
```

## Classes

`0 = panel`, `1 = bubble` (speech balloon / caption box).  Training a bubble class
into the same model means one 30 ms pass replaces both the panel YOLO *and* the
separate RT-DETR bubble detector wired in as an interim fix.

## Label format

Standard YOLO: one `.txt` per image, `class cx cy w h` normalised 0–1.
`data.yaml` written by `build_dataset.py`.
