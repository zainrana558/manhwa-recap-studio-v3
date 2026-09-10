# OCR-accuracy training toolkit

Scripts + Kaggle notebooks for the post-OCR **garble-elimination** work: a ByT5
text corrector, a PP-OCRv5 recognition fine-tune, and the data pipeline that
feeds both. Built during the Nano Machine 328-chapter run and generalised to a
cross-series (multi-manhwa) corpus.

These scripts were run from a session scratch dir; paths to the repo
(`/home/azureuser/manhwa-recap-studio`), a live job id, and the OCR service
(`localhost:3002`) are hard-coded near the top of each file — adjust before
re-running. The large generated corpora (synthetic crops, scraped pages, sliced
frames, Qwen batches) are **not** in git — they live in private Kaggle datasets
(see below).

## Data pipeline

| step | script | output |
|---|---|---|
| 1. scrape hard manhwa/webtoons (no manga) | `scrape_ocr_train.sh` | `data/ocr_train_raw/<title>/chapter_*/` |
| 2. slice to canonical 1920×1080 panel frames + RapidOCR baseline | `slice_all.sh` (wraps `master_pipeline.py --slice-only`) | `slice_work/<title>/temp_slices/chap_*/` + `manifest.json` |
| 3. curate title-balanced Qwen batches (15% OCR-fail / 50% garbled / 35% clean; fail bucket detector-filtered) | `curate_qwen.py` | `qwen_out/batch_XX.zip` + `INDEX.json` |
| 4. **(manual)** transcribe batches in Qwen chat | `QWEN_PROMPT.md` | `qwen_returns/*.txt` |
| 5. cross-check Qwen vs RapidOCR → build both corpora | `crosscheck_build.py` | `xseries_corrector_pairs.jsonl`, `rec_real/` verified line crops |
| synthetic bubble crops (perfect labels) | `gen_synth.py` | `synth/` |
| synthetic single-line crops for PP-OCR rec | `gen_lines.py` | `rec_synth/` |
| ByT5 corpus assembly (real + synthetic + identity) | `build_corpus.py` | `byt5_data/{train,val,eval_real}.jsonl` |

`garble_helpers.py` — standalone garble detector (wordfreq zipf thresholds,
SFX/scream heuristics, credit-line regex). Shared by the curate / audit / apply
scripts. `audit_garble.py`, `apply_final.py` — the Nano-Machine-run garble audit
and merge-into-narration.json step. `ab_ocr.py` — RapidOCR vs Baberu A/B harness.

## Training

- `kbyt5/train_byt5.py` — ByT5-small post-OCR corrector (Kaggle P100). Handles
  the Kaggle torch/`sm_60` breakage (uninstall torchvision, pin torch 2.4.1,
  `os.execv` restart). `predict_with_generate=False` mid-training, one honest
  generate-eval at the end (`cer_raw_vs_gold` vs `cer_model_vs_gold`).
- `krec/train_rec.py` — PP-OCRv5 English recognition fine-tune (PaddleOCR 3.x →
  `paddle2onnx`). **Draft — needs a Kaggle test pass** (PaddleOCR config keys
  shift between versions).
- `kaggle_api.py` — REST client for the `KGAT_` bearer token (the `kaggle` CLI
  1.7.x can't use it): `kernel-push`, `kernel-status`, `kernel-output`.

## Kaggle datasets (private)

| dataset | contents |
|---|---|
| `nm-ocr-byt5-data` | ByT5 corrector `train/val/eval_real.jsonl` |
| `manhwa-ocr-synth` | `synth/` bubble crops + `rec_synth/` line crops |
| `manhwa-ocr-raw-scrapes` | scraped chapter pages (11 titles × 10 ch) |
| `manhwa-ocr-slice-frames` | sliced 1920×1080 frames + manifests |
| `manhwa-ocr-qwen-batches` | curated Qwen batches + `INDEX.json` |
| `manhwa-ocr-rec-data` | merged synthetic + verified-real line crops for `krec` |

## Corrector pairs

`corrected_pairs.jsonl` — 1,586 raw→clean pairs from the Nano Machine run
(4 Grok rounds + panel-vision + mechanical + credit-strip). `panel_vision.json`
— Grok vision transcripts of fully-destroyed panels. `manual_overrides.json` —
hand-fixed lines. `crosscheck_build.py` appends `xseries_corrector_pairs.jsonl`
(cross-manhwa) from the Qwen pass.
