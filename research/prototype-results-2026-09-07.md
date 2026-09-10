# Prototype + bench results — slicer / OCR upgrades (2026-09-07)

Follows `webtoon-slicer-ocr-landscape-2026.md`. Prototyped and tested the 4
non-training items on the one test series available (`data/jobs/cmtq7fu67…`,
"Nano Machine", 2 chapters, 34 source strips → 182 sliced panels, OCR truth
from the job's own narration.json). Bench scripts in `scratchpad/bench/`.

| # | Prototype | Verdict | Integrated |
|---|---|---|---|
| A | `comic-text-segmenter-yolov8m` as a 2nd text-region detector | ❌ **net negative on this content** — NOT better than current RT-DETR | ❌ no (model + bench kept for other series) |
| B | Cloud-VLM OCR fallback tier (Gemini/OpenRouter) | ➖ built, dormant without a key; not proven better here (2/182 eligible) | ✅ yes, opt-in — `main.py`, no-op until a key is added |
| C | Gutter-carve: split stacked beats a floating caption used to glue | ✅ **better** — 4 clean fixes / 0 regressions / 34 pages + full render | ✅ yes, default ON (`master_pipeline.py`) |
| D | Multi-column reading order | ✅ already correct (recursive XY-cut) | — no change needed |
| — | `start.sh` sources `../../.env` | ✅ needed regardless (OCR service never saw `.env`) | ✅ yes (`start.sh`) |

All live now: `master_pipeline.py` is spawned fresh per job so **C** applies to the next
render; the OCR service was restarted so **B**'s tier + the `.env` sourcing are active
(dormant — no key). No pipeline-service / frontend / schema changes were needed.

---

## A — comic-text-segmenter-yolov8m  ❌ not adopted

`bench_text_regions.py` + `diag_text.py`: ran the current RT-DETR INT8 text
detector and `ogkalu/comic-text-segmenter-yolov8m` (Apache-2.0, imgsz 1024,
conf 0.25) on all 182 real OCR crops.

| | RT-DETR (current) | segmenter |
|---|---|---|
| recall on text-bearing panels | 93.7 % (74/79) | 93.7 % (74/79) |
| avg boxes / frame | 1.53 | 1.53 |
| panels segmenter caught that RT-DETR missed | — | **0** |
| speed (CPU) | **114 ms** | **598 ms** (5.2×) |

The handful of per-frame disagreements were inspected in `overlays/`. Every
extra region the segmenter found was **untranslated Korean SFX lettering**
(차앙, 허억, 후욱 …) left as art in the scanlation — feeding those to OCR
produces junk, not narration. On this content the segmenter costs 5× the time
to add detections we specifically do **not** want.

**Keep for**: a series whose sound effects are rendered as stylised *English*
(BOOM / KRRSH / …). Re-run `scratchpad/bench/bench_text_regions.py` against it;
if `seg_only > 0` and those are real English SFX, wire it into
`_detect_text_boxes` behind `RECAP_TEXT_SEG=1`. Model is at
`pipeline/models/comic-text-segmenter/` (gitignored).

---

## B — VLM OCR fallback tier  ✅ built (opt-in, no-op without a key)

`mini-services/paddleocr-service/main.py`:

- New tier at the end of `_ocr_with_cascade`, **after** RapidOCR → PaddleOCR →
  Tesseract. Fires only when a text region *was* localised but nothing read it
  to `SUCCESS` (stylised SFX, warped/curved text, text on busy art). A blank /
  zero-region panel never reaches it (`OCR_VLM_ON_ZERO_REGIONS=1` opts that in,
  with a cheap pixel pre-filter).
- `_get_vlm_client()` mirrors the pipeline's `_resolve_visual_client`: OpenAI
  client pointed at Gemini's OpenAI-compatible endpoint (or OpenRouter). **No
  key → `(None, "")` → the tier is a hard no-op, zero behaviour change.** No new
  dependency (`openai` already installed).
- Prompt transcribes verbatim, separates bubbles with ` / `, and returns
  `NO_TEXT` for non-Latin / untranslatable lettering so garbage never reaches
  narration.
- Sliding-window rate limiter (`OCR_VLM_MAX_CALLS_PER_MIN`, default 12) keeps
  it inside the free tier; `_vlm_stats` counters exposed on `/health`.
- `start.sh` now sources `../../.env` (PATH preserved) so `GEMINI_API_KEY`
  reaches this pure-Python service — it never did before (documented quirk).

**Test status**: `scratchpad/bench/test_vlm_ocr.py` confirms the no-key path —
cascade runs unchanged, `_vlm_stats.calls == 0`, `/health` shows
`vlm_fallback: null`. On this series only **2 / 182** panels are VLM-eligible
(both untranslatable Korean SFX → the prompt would return `NO_TEXT`), so it is
near-inert here — its value is a safety net for messier series.

**To finish testing**: put `GEMINI_API_KEY=…` (free — aistudio.google.com/apikey)
in `.env`, restart the OCR service, re-run `test_vlm_ocr.py` — it will transcribe
the hard crops live and print before/after.

---

## C — gutter-carve  ✅ shipped, default ON

`pipeline/master_pipeline.py` `_split_borderless_column`.

**Bug**: a tall panel box is split at *clean full-width background bands*
(`quiet` rows). When a lone narration caption floats in an otherwise-clean
400 px gutter, the band's **midpoint** lands inside that caption →
`_text_straddles()` is true → the **entire band is discarded** → the beats on
both sides stay glued into one 16-second static frame. (The smarter
text-cluster and low-ink passes further down never run — the function returns
before them.)

**Fix**: `_clear_subrun(a, b)` finds the largest sub-range of the gutter band
that no *wide* text box overlaps, and cuts there instead of throwing the band
away. Only a bubble taller than the whole gutter (genuinely uncuttable) still
skips. `RECAP_GUTTER_CARVE=0` restores the old behaviour.

**A/B on all 34 source strips** (`bench_slicing_ab.py`, overlays in
`scratchpad/bench/slices_ab/`):

| | baseline | fix |
|---|---|---|
| total panels | 184 | 188 |
| pages changed | — | **4** (+1 each) |
| regressions (cut through art / bubble) | — | **0** |

The 4 changes were inspected frame-by-frame:
- **006**: one 2400 px box holding 4 beats → 2 boxes, cut in the white gutter,
  the "HOWEVER…" caption correctly rides with the beat above it.
- **012**: 5-beat block → 2, cut in the gutter between the hand-on-ground panel
  and the screaming panel; "IT'S ALRIGHT…" caption stays with the first.
- **c2/005**: two stacked face panels separated; SFX in the gutter carved out.
- **024**: the "STARTING ACTIVATION" caption separated from the end-of-chapter
  title logo — which then trips `_looks_like_credits_panel` and is **dropped**
  (net: cleaner, one fewer junk frame).

End-to-end reslice of chapter 1: 104 frames vs 102, same ~25 s, no crash.

**Full pipeline run** (`master_pipeline.py` on the 2-chapter test set, gutter-carve
ON): PIPELINE COMPLETE in 3.3 min — chap_001 104 frames / chap_002 81 frames, both
rendered + QA-validated, merge QA-validated, final 721 s / 1920×1080 h264 video
valid. OCR: 81 SUCCESS on chap_001 (79 baseline + the 2 new text-bearing frames).
No `duration_out_of_tolerance`, no regressions.

---

## D — reading order  ✅ no change

`_reading_order_boxes` already does a recursive XY-cut (split on the widest gap
no box crosses, horizontal then vertical, left-first) — handles staggered /
diagonal / side-by-side layouts. No ordering errors surfaced in any of the 34
pages' overlays. Left as-is.

---

## Bottom line

On real content the current pipeline's slicing + OCR are **already strong** —
which matches your experience that the off-the-shelf "better" models (Manga109
YOLOs, Magi, the comic-text segmenter) don't actually beat it. The one concrete
defect found was the gutter-carve under-segmentation (**C, fixed**). The VLM
fallback (**B**) is worthwhile insurance for messier series but needs a free key
to exercise. Everything is opt-in / reversible via env flags.
