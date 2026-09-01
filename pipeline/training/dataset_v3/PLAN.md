# YOLO v3 panel-detector — dataset build + training plan

Goal: a webtoon/manhwa panel detector that beats v2 (which is near its
self-distillation ceiling). Key move: **independent labels from Gemini**, not
the pipeline's own slicer output.

## Status legend:  [ ] todo   [~] running   [x] done

## 1. Source material  [x]
`pipeline/training/dataset_v3/sources/<series>/chapter_NNN/*.png`
- chainsaw-man (B&W bordered manga)   ~12 ch
- nano-machine (borderless colour manhwa) 19 ch
- solo-leveling (borderless colour action webtoon) 12 ch
- orv (whitespace-heavy colour webtoon) 12 ch
- tbate (colour fantasy webtoon) 12 ch
- tower-of-god — FAILED to scrape (bad WeebCentral id); optional to retry
~2,625 pages total.

## 2. Gemini annotation  [~]
`annotate.py` — tiled (tile 1400 / overlap 320), `gemini-flash-lite-latest`,
merge tile-seam dup/fragments, YOLO out to `dataset_v3/yolo/{images,labels,overlays}`.
- key: `dataset_v3/.gemini_key` (key 2 — key 1 is project-blocked)
- ~7 s/page, ~2 boxes/page, no rate-limit issues seen
- resumable (skips existing label files)

## 3. QC / clean  [ ]  -> `qc.py`
- drop pages with 0 boxes only if the page has real ink (keep true blanks as
  hard negatives with an empty label file — teaches "no panel here")
- geometric filters: box w >= 0.15*W, h >= 1.2%*H, area < 98.5%; pairwise
  IoU < 0.35 (merge/drop bad overlaps); clamp to image
- drop a page if boxes cover < 25% or > 99.5% of page area (Gemini misfire)
- dedup near-identical pages (webtoon reposts) via perceptual hash
- sample 60 overlays -> eyeball; record keep-rate

## 4. Assemble training set  [ ]  -> `assemble.py`
Merge, with per-source weighting:
- (A) Gemini labels from step 3      — PRIMARY, weight 1.0
- (B) existing v2 clean set (pipeline/training data if still on disk / in the
      last Kaggle dataset `webtoon-panel-yolo`) — weight 0.5
- (C) recap `source_box` self-distillation (`dataset_v3/recap_selfdistill/` +
      any saved from the cancelled job) — weight 0.3, only pages passing the
      step-3 geometric filters
Single class: `0 = panel`. (Bubble detection stays with the existing RT-DETR;
not retraining that here unless time permits.)
80/10/10 train/val/test split, no series leakage across splits.

## 5. Train on Kaggle  [ ]
Reuse `pipeline/training/kaggle_kernel_train.py` pattern:
- upload the assembled dataset as a Kaggle dataset (or version the existing one)
- YOLOv8n or YOLO11n, imgsz 1024, ~120 epochs, P100
- torch==2.4.1 pin (P100 sm_60), `--no-deps ultralytics`
- export ONNX with NMS baked (nms=True) -> (1,300,6), input [1,3,1024,1024]
- kernel-metadata: enable_gpu, enable_internet, dataset_sources

## 6. Validate + install  [ ]
- compare v2 vs v3 on the held-out test split (mAP50, mAP50-95) AND on a
  fresh un-annotated chapter by eyeball (gutter bleed? over/under-split?)
- if better: install to `pipeline/models/manga-panel-yolo/manga_panel_detector_fp32_1024.onnx`
  (git add -f), commit, redeploy
- if not clearly better: keep v2, document why

## Notes / decisions
- Gemini flash-lite under-segments stacked same-background panels and
  sometimes pads whitespace on borderless webtoon beats. Accepted as noise —
  it's independent of v2's biases and never bleeds gutters (v2's main flaw).
- If key 2 also gets blocked: fall back to a local VLM (Florence-2 / Qwen2.5-VL
  7B) or the `source_box` self-distillation alone.

---
## UPDATE 2026-09-01 (Kaggle P100 constraints)

- Kaggle API always assigns **P100 (Pascal/sm_60)**; base torch dropped Pascal
  kernels. Must pin `torch==2.4.1+cu121` in every GPU kernel.
- **Qwen2.5-VL-3B fp16**: fast (~15s/pg) but quality NOT up to par — loose
  boxes, under-segments manga badly, misplaced on webtoons.
- **Qwen2.5-VL-7B 8-bit (bnb int8)**: ~6 min/page on P100 (int8 emulated on
  Pascal) — unusable; also returned 0 parsed boxes (parser now handles Qwen
  native `(x1,y1),(x2,y2)` box tokens; retesting 3B with better prompt).
- **Kumiko (CV) on the 4 bordered-manga series: EXCELLENT.** 761 pages,
  4.1 boxes/pg, 0 empty, tight on frame lines, no gutter bleed. DONE, free,
  local. Output: pipeline/training/dataset_v3/yolo_kumiko/.

### Revised annotation plan
- manga (chainsaw-man, one-piece, berserk, jujutsu-kaisen) -> Kumiko  ✅ done
- ~11 borderless webtoon series -> VLM:
  - if Qwen-3B + improved prompt is trainable-quality -> use it (free)
  - else -> Gemini flash-lite ($2 pay-as-you-go, or the daily-quota grind);
    quality was decent on the earlier 215-page sample
- merge -> train_v3.py (yolo11n, panel + bubble)
