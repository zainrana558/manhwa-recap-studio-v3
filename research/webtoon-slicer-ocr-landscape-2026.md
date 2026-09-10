# Webtoon / Manhwa panel-slicing + OCR — free-tool landscape (Sept 2026)

Research request: find better **free** panel slicers and OCR for **webtoons / tall vertical
strips** — including diagonal, borderless, and full-bleed panels — plus OCR that reads the
English lettering accurately. Constraint from the box: **CPU-only, 4 vCPU / 15 GB RAM, no
GPU** (`nvidia-smi` absent). Source material is **English scanlations** (uppercase comic
lettering + stylised SFX like `EUAACK...!`), not Korean raws.

Key user note that shaped this: *"the manga-trained ones are only good at manga — tried
those, and Magi too."* That is a real and well-documented failure mode (see §2.1), so this
report is built around **webtoon-native** and **general-comic** tools, not Manga109 models.

---

## 0. TL;DR — what's actually worth trying

| Area | Current | Best free upgrade | Why | Effort |
|---|---|---|---|---|
| **Vertical-strip cutting** | `manga-panel-detector-yolo26n` (Manga109) + gutter-aware fallback | **Flip the priority**: classical gutter/void cutter *primary*, ML detector only for grid pages / borderless blocks | Manga109 YOLO is trained on rectangular bordered manga frames; webtoons are gutter-separated strips. Your own `webtoon_panel_slicer.py` v6.1 methodology is the right primary. | low |
| **Borderless / diagonal panels** | none really | **`ogkalu/comic-text-segmenter` sibling approach** + a YOLO fine-tuned on **Roboflow "Webtoon-Manhwa Panels"** (8 classes incl. Diagonal / Noborder / Outbound) | Only labelled data that actually contains these panel types | med (train) |
| **Text-region detection** | RT-DETR-v2 (ogkalu 11k manga+webtoon+western) + INT8 ONNX | **`deepghs/AnimeText_yolo`** (YOLO12, 735K imgs, 4.2M stylised text blocks, mAP50-95 ≈ 0.90) *or* **`ogkalu/comic-text-segmenter-yolov8m`** (3k manga+**webtoon**+manhua, 1024px, Apache-2.0, handles extreme ratios) | Both are explicitly built for stylised / irregular comic text, not document text | low–med |
| **OCR (recognition)** | RapidOCR **PP-OCRv5-mobile-EN** (bake-off winner, word-recall 0.99) | **Keep it.** It's already near-optimal for a classical engine. | You already ran a 10-panel hand-transcribed bake-off vs v4/v6/server — don't relitigate | none |
| **The hard 5-10 %** (SFX, warped text, low-contrast) | falls to Paddle → Tesseract → FAILED | **Free-tier VLM** (Gemini 2.x Flash) on *only* the `UNCERTAIN`/`FAILED` panels | VLMs beat every classical OCR on stylised text; free tier covers the volume if gated to fallbacks | med |

Nothing here needs a GPU except local VLMs (which you should not run on this box).

---

## 1. Your current stack (baseline for comparison)

From `pipeline/master_pipeline.py` + `mini-services/paddleocr-service/main.py`:

**Slicing**
- `_detect_panels_for_page()` → for tall strips tries `_detect_panels_gutter_aware()` first
  (row-projection: content rows vs near-empty rows, bubble-safe merge), else
  `_detect_panels_auto()` = **`leoxs22/manga-panel-detector-yolo26n`** (YOLO26-nano fine-tuned
  on **Manga109-s**, ONNX FP32 1024px, class 0 = panel / 1 = text) + contour fallback.
- Text/bubble detector: **RT-DETR-v2 r50vd** (Apache-2.0, 42.9 M) — this is the
  `ogkalu/comic-text-and-bubble-detector`, trained on ~11k **manga + webtoon + western**
  images — or its INT8 ONNX export (~11 MB, 640px).
- Dynamic bubble-tail padding, seam-stitch, compose-canvas.

**OCR** (`paddleocr-service`)
- Primary: **RapidOCR** with `PP-OCRv5 mobile det` + `en_PP-OCRv5_rec_mobile` (ONNXRuntime,
  pool of 3 engines). Chosen by a hand-transcribed 10-panel bake-off over PP-OCRv4, PP-OCRv6,
  and every v5/v6 det+rec mix — best word recall (0.99), best precision (0.97). Note in code:
  *"v5 SERVER detection over-segments"*.
- Fallback tier: **PaddleOCR PP-OCRv4** (serial, `_inference_lock`), then **Tesseract**.
- Cascade in `_ocr_with_cascade`: RapidOCR → if SUCCESS return / if 0 regions return / if
  UNCERTAIN fall through to Paddle → Tesseract.

**Verdict:** the OCR *recognition* head is already well-tuned; the *detection* and *slicing*
sides are where manga-bias still lives.

---

## 2. Panel slicing / segmentation

### 2.1 Why Manga109 / Magi models underperform on webtoons (confirmed)

- **Manga109** = 109 printed Japanese manga volumes, page-based, near-always **rectangular
  bordered frames** in grid/staggered layouts. Detectors trained on it (`deepghs/manga109_yolo`,
  `leoxs22/manga-panel-detector-yolo26n`, the CVPR-2025 Manga109-seg models) learn "find the
  black rectangle." Webtoons are a **single 800×10 000+ px column**, panels separated by
  **whitespace or black-void gutters**, frequently **borderless**, **full-bleed**, **diagonal**,
  or bleeding into each other. Different problem.
- **Magi / Magiv2 / Magiv3** (Oxford VGG, *The Manga Whisperer* CVPR'24, *Tails Tell Tales*
  ACCV'24, *From Panels to Prose* ICCV'25): DETR-style panel+text+character+speaker graph,
  trained/evaluated on **PopManga / Manga109** — B&W Japanese manga pages. Reported by many
  users (and consistent with your experience) to degrade on colour webtoons and on
  full-bleed/borderless layouts, and it's heavy (needs a GPU to be practical).
- Practical implication: for webtoons, a **classical vertical cutter is the right primary**,
  and ML is a *secondary* tool for the cases the cutter can't split (tall action blocks,
  borderless montages).

### 2.2 Classical vertical-strip cutters (best *primary* for webtoons — all CPU, ~free)

| Tool | Approach | Handles | License | Notes |
|---|---|---|---|---|
| **Your `webtoon_panel_slicer.py` v6.1** (in repo PDFs) | Signal hierarchy: L1 white **+ black-void** gutters → L2 rolling-std "event zones" (SFX) cut *after* the zone → L3 projection valleys → L4 Hough → L5 recursive binary (Pang et al. ACM MM'14). `merge_nearby_cuts` + fragment merge = no double cuts. | tall strips, variable gutters, black manhwa dividers, full-bleed SFX; optional YOLO sliding-window L0 | CC0 | Already the most webtoon-aware thing you have. The methodology PDF is 11 MB of worked examples. **This should be the default path**, not the Manga109 YOLO. |
| **`indivisible/webtoon_slicer`** (GitHub) | merge/split helper for long strips, whitespace-based | classic gutter webtoons | open | simple, battle-tested for the "just cut the whitespace" case |
| **`eskutcheon/ManhwaFormatter`** (GitHub) | vertical segmentation by low-variance blank rows; won't cut through panels/bubbles; re-stacks to cbz/cbr/pdf | poorly-cropped webtoon dumps | open | good reference for the "don't cut a bubble" heuristic |
| **`njean42/kumiko`** (Kumiko, the Comics Cutter) | OpenCV contour detection → panel polygons, JSON `[x,y,w,h]`, reading-order sort | **bordered** pages/panels, some non-rect | AGPL-3.0 | great for grid pages, weak on borderless; HF Space `avans06/KumikoMangaPanelExtractor` |
| **`maxhalford` blog method** / `comic-panel-detection` (SourceForge) | connected-components on white background, remainder = panels | bordered pages | open | the canonical "CCL on the gutter" writeup |
| **Max-Halford-style + watershed ensembles** | contour + watershed + projection | mixed | open | the survey notes these **over-segment** when every stage votes independently — your v6.1's "exclusive roles" design is the fix |

**Recommendation:** promote the classical cutter to primary for `frame_kind == "scroll_frame"`
/ tall inputs; keep an ML detector only as the L0 (grid pages) and for subdividing tall
borderless blocks.

### 2.3 Webtoon / manhwa-trained ML detectors

| Model / dataset | What | Webtoon coverage | License | Get it |
|---|---|---|---|---|
| **Roboflow "Webtoon-Manhwa Panels"** (`manhwa-pannel/webtoon-manhwa-panels-bwdzz`, and `asdasdada/…`, `star-m9qc2/…`) | Object-detection dataset, **8 classes**: Diagonal, Irregular, Noborder-Rect, Outbound-Irregular, Outbound-Rect, Rectangle, Split, Square | **native** — this is the only labelled set that names diagonal/borderless/outbound | CC BY 4.0 / MIT (varies by fork) | Roboflow Universe; export YOLO format, fine-tune a `yolov8n/s` or `yolo11n` in ~1 GPU-hr on Colab free |
| **Roboflow "Webtoon Panel"** (`teste-lk8f9/webtoon-panel`) | 2 836 panel images + hosted pre-trained model + API | native, single "Panel" class | open | quick baseline; hosted inference is rate-limited but the weights export |
| **`hanish3464/WORD-pytorch`** (Webtoon Object Recognition & Detection) | Korean-webtoon: **cut detection (OpenCV)** + speech-bubble (Faster-RCNN) + line-text (CRAFT) + Korean recognition + Papago translate | native (Korean webtoons) | check repo | old (≈2019, CRAFT+FRCNN) and unmaintained, but it's the closest public clone of Naver's own pipeline; mine the **cut-detection** code |
| **`Mochel-Talebis/webtoon-akhza`** (Roboflow) | webtoon **instance segmentation** (bubbles), 101 imgs | native | open | small; bubble-only |

### 2.4 General-comic detectors that *partially* generalise (better than Manga109-only)

| Model | Arch / data | License | Note |
|---|---|---|---|
| **`ogkalu/comic-text-and-bubble-detector`** | RT-DETR-v2, ~11k **manga + webtoon + western** | Apache-2.0 | **you already run this** for text/bubbles — it's a good, webtoon-inclusive choice; consider also using its `panel`-ish outputs if present, or pair with a panel model |
| **`mosesb/best-comic-panel-detection`** | YOLOv12x fine-tuned from COCO on a custom Roboflow comic set; "near-perfect" P/R on its val | check card | mostly **western/bordered** pages; test before trusting on webtoons |
| **`deepghs/manga109_yolo`** family (n→x) | body/face/frame/text, F1 0.88–0.92 | check | **Manga109 — expect the failure mode you already hit**; listed for completeness |
| **CVPR 2025 "Advancing Manga Analysis: Comprehensive Segmentation Annotations for Manga109"** (Xie et al.) | instance masks for frame / text / onomatopoeia / body / face / **balloon** | dataset: research use | genuinely better *masks* than boxes, but still **Manga109 domain**. `ShadowB/Manga109-panel-balloon-text-yolov26-segmentation` + `MS92/MangaSegmentation` are derived YOLO-seg models |
| **`arxiv 2605.21182` "Manga109-v2026"** | refreshed Manga109 annotations for modern understanding | research | same domain caveat |

### 2.5 Zero-shot (no training) — GroundingDINO + SAM

- **Grounded-SAM**: prompt GroundingDINO with `"comic panel . manga panel . frame . window"`
  → boxes → SAM → instance masks. Used this way in *From Panels to Prose* and the *Comics
  Understanding* survey for panels/characters/text/faces with **no comic-specific training**.
- Reality on your box: GroundingDINO-T + SAM-B on CPU is **~5–20 s/image** — too slow for
  bulk chapter slicing, but viable as a **one-off "hard page" fallback** or for building a
  training set to distill into a YOLO. SAM-2.1-tiny or MobileSAM/EdgeSAM cut that a lot.

### 2.6 If you want to train your own (recommended, cheap)

Datasets, all free:
- **Roboflow Webtoon-Manhwa Panels** (diagonal/borderless/outbound classes) — the key one.
- **Manga109-s `frame`** + `deepghs/manga109_yolo` labels — for grid pages only.
- **CoMix** (NeurIPS'24) / **Comics Datasets Framework** (`arxiv 2407.03540`) — unified
  detection benchmark, mostly western + some manga; reading-order + character + dialog labels.
- **DCM772** (Digital Comics Museum, western golden-age) — panels + balloons + faces.
- **`comix_books_v0`** — ~950k western pages with panel detections (weak labels).
- Distill GroundingDINO+SAM pseudo-labels on your *own* scraped chapters → best domain match.

Train a `yolo11n`/`yolov8s` at 1024px on Colab/Kaggle free GPU (~1–2 hrs), export ONNX,
drop into `PANEL_YOLO_PATH`. Expected: materially better borderless/diagonal recall than the
Manga109 nano you have now.

---

## 3. Text-region detection (finding *where* the lettering is)

This is the part most worth upgrading — classical OCR recognition is fine once the crop is
right, but manga-trained text detectors miss webtoon SFX / narration boxes / low-contrast
overlay text.

| Model | Arch | Training data | Res | License | Fit for webtoon English |
|---|---|---|---|---|---|
| **`deepghs/AnimeText_yolo`** ⭐ | YOLO12 (n 2.6M → xl 59M) | **AnimeText**: 735K images, 4.2M text blocks, hierarchical + hard negatives, explicitly "stylised / handwritten / irregular / confusable-with-symbols" text (`arxiv 2510.07951`, Oct 2025) | — | **GPL-3.0** | **best available** for stylised/irregular comic text; paper shows it beats models trained on prior datasets. GPL is the catch — fine for internal use, matters if you redistribute. `s` or `m` runs fast on CPU. |
| **`ogkalu/comic-text-segmenter-yolov8m`** ⭐ | YOLOv8m **segmentation** | ~3k **manga + webtoon + manhua + western**, 1024px, resized not cropped | 1024 | **Apache-2.0** | explicitly "handles extreme aspect ratios common in Korean webtoons"; gives masks (good for tight crops + inpaint). Best license/fit combo. |
| **`dmMaze/comic-text-detector`** | YOLOv5 backbone + **UNet mask head** + **DBNet line head** | ~13k: ⅓ Manga109-s, ⅓ **DCM (western)**, ⅓ synthetic anime | 1024 | check repo | mature, ONNX (`mayocream/comic-text-detector-onnx`), used by manga-image-translator + BallonsTranslator; more general than Manga109-only |
| **`ogkalu/comic-text-and-bubble-detector`** | RT-DETR-v2 42.9M | ~11k manga + **webtoon** + western | 640 | Apache-2.0 | **what you use now** — keep as the bubble detector; AnimeText/segmenter can run alongside for loose SFX text |
| **CRAFT** (`clovaai/CRAFT-pytorch`) | char-region affinity | SynthText + IC13/17 (scene text) | — | MIT | generic scene-text; WORD-pytorch uses it; decent on isolated stylised words, weaker on dense bubbles |
| **PP-OCRv5 det (mobile)** | DB++ | general | — | Apache-2.0 | you already use it inside RapidOCR; the note "server det over-segments" is real for tight comic text |

**Recommendation:** add **`ogkalu/comic-text-segmenter-yolov8m`** (Apache-2.0, webtoon-aware,
mask output) as the text-region proposer feeding OCR crops, or **`AnimeText_yolo-s/m`** if
GPL is acceptable. Keep RT-DETR for bubble geometry. Union the regions; dedupe by IoU.

---

## 4. Text recognition / OCR engine

### 4.1 Keep PP-OCRv5-mobile-EN

Your code comment documents a hand-transcribed 10-panel bake-off: PP-OCRv5-mobile det+rec
(EN) beat PP-OCRv4, PP-OCRv6, and every v5/v6/server mix — **word recall 0.99, precision
0.97, char-sim 0.98**. Independent 2026 comparisons agree PaddleOCR/RapidOCR is the strongest
*free classical* engine on clean printed text and the best on handwriting-ish (~73 % vs
EasyOCR 62 % vs Tesseract 45 %). **No change recommended here.**

PP-OCRv5 itself (May 2025, PaddleOCR 3.0): single model, 5 text types + cursive, **+13 pts
over PP-OCRv4**, 106 languages, ~30 % better multilingual. You're already on it.

### 4.2 Alternatives, and why they don't beat what you have

| Engine | CPU speed | English comic-text accuracy | Verdict |
|---|---|---|---|
| **Surya** (`datalab-to/surya`) | **~50× slower on CPU than GPU**; layout ≈0.14 s/img *on an A6000* | strong on documents, 90+ langs, has **reading-order + layout**; docs say "will likely not work on photos… designed for printed text" | **layout/reading-order module is interesting for panel+text ordering**, but full OCR on CPU is too slow for bulk. GPL-ish (check). |
| **docTR / OnnxTR** (`mindee/doctr`, `felixdittrich92/OnnxTR`) | **fast on CPU** (~0.12–0.17 s/page mobilenet; 8-bit quant available; no torch/TF dep) | default vocab is French; one HF model does Latin-Extended (EN/FR/DE/ES/PT/IT). Document-tuned, not comic-tuned. | good **fast CPU** option if you ever drop Paddle; would need comic fine-tune to match v5 |
| **EasyOCR** | ~3× slower than Paddle, ~500 MB | below PaddleOCR on clean text; decent mixed-script | no reason to switch |
| **manga-ocr** (`kha-white`, ViT+BERT, JA) | ~1 s/crop CPU | **Japanese only** | **skip** — you confirmed; `ogkalu/manga-ocr-onnx` / `-mobile` are ONNX repackages, still JA |
| **Pororo `brainocr`** (Kakao Brain, KO+EN) | moderate | Korean+English, trained on AI-Hub font data | **only if you add Korean-raw support**; comic-translate uses it for KO |
| **TrOCR** (MS, ViT+RoBERTa) | slow on CPU | needs fine-tune for comics | not worth it without a GPU + labelled set |
| **VLM OCR local** (GOT-OCR2, dots.ocr 3B, PaddleOCR-VL 0.9B, DeepSeek-OCR, Qwen2.5-VL) | **needs GPU** (7–20 GB VRAM) | SOTA on hard text | **not on this box** — only via cloud API |

### 4.3 The accuracy ceiling — VLM on the fallback panels only

VLMs (Gemini 2.x Flash, GPT-4o, Claude) beat every classical OCR on **stylised / warped /
low-contrast / SFX** text — the exact `UNCERTAIN`/`FAILED` bucket in your cascade. Strategy:
after RapidOCR → Paddle → Tesseract all fail, send that one panel crop to a **free-tier VLM**.

- **Gemini 2.x Flash free tier**: image understanding is included; reported 88–95 % OCR
  accuracy on hard docs; generous free RPM/RPD (check current quota — was ~15 RPM / 1500 RPD
  for 2.0 Flash). Gated to *only* fallback panels, a chapter has maybe 5–15 such crops →
  well within free limits.
- **`StoneSteel27/ComiQ`** — a comic-specific hybrid OCR library that already does exactly
  this: EasyOCR/PaddleOCR for detection + **Gemini Flash** for recognition/assembly in
  bubbles. Worth reading even if you don't adopt it wholesale.
- **`ComiCap`** (`arxiv 2409.16159`) — VLM pipeline for dense **panel captioning** (grounded
  attributes) — relevant to your `--describe-visuals` feature, not OCR per se.

Local Ollama VLMs stay impractical here (documented: llava:7b ~50–70 s/panel, moondream
garbled) — the cloud free tier is the realistic path, same conclusion as the visual-caption
feature.

---

## 5. Full open pipelines worth mining for parts

| Project | Stack | What to steal |
|---|---|---|
| **`ogkalu2/comic-translate`** | RT-DETR-v2 (manga+webtoon+western) detect → algorithmic bubble seg → OCR routed by language (JA=manga-ocr, KO=Pororo, else **PP-OCRv5**) → LLM translate → LaMa inpaint. GPT-4.1 / Claude-4.5 / Gemini-2.5 supported. | its **detector + OCR-routing** design is basically your target architecture; models are the `ogkalu/*` HF repos |
| **`zyddnys/manga-image-translator`** | CTD (comic-text-detector) or CRAFT detect → OCR `32px`/`48px`/`48px_ctc`/`mocr` (48px default; ctc faster+KO but less accurate) → translate → inpaint | CTD ONNX; the `48px` CTC recognizer is a compact comic-tuned CRNN you could try as a fallback tier |
| **`dmMaze/BallonsTranslator`** (+ `-Pro` forks) | one-click detect/OCR/inpaint/typeset; Pro fork claims **20+ detectors / 30+ OCR / 15+ inpainters** | menu of every OCR/detector wired behind one interface — good shopping list |
| **`hanish3464/WORD-pytorch`** | Korean-webtoon cut + bubble + text + KO recognition | **cut-detection** (OpenCV) tuned for webtoons |
| **`njean42/kumiko`** | OpenCV contour panels → JSON + reading order | drop-in panel JSON for bordered pages; reading-order sort logic |
| **`hummat/panelizer`** | panel reader, DL detection + manual override + PWA | the **human-override UX** — pairs with your manual review-gate |
| **`reidenong/ComicPanelSegmentation`** | classical, exploits pixel-perfect digital panel edges | fast path for clean digital webtoons |

---

## 6. Academic / studies (the "every article" ask)

**Webtoon-specific**
- **Naver — "Barrier-Free WEBTOON"** (navercorp press + about.webtoon.com/sustainability/43,
  2023). Their production pipeline for blind-accessible alt-text: *image segmentation →
  speech-bubble vs background-text discrimination → text extraction → dialogue ordering*,
  built on **"WEBTOON Object Detection"** (panels + bubbles + dialogue areas) + OCR. Not
  open-sourced, but it's the reference design and confirms classical segmentation + a
  webtoon-trained detector + reading-order is the right shape. WORD-pytorch is the public echo.
- **"A harmless webtoon for all"** (Expert Systems w/ Apps 2022) — 1 094 webtoons, ML on
  panel content; age-restriction, but useful webtoon-CV methodology.

**Comic panel extraction / segmentation**
- **Pang, Qin, Wong et al. "A Robust Panel Extraction Method for Manga"** (ACM MM 2014) —
  CCL on background + diagonal-stripe handling; the recursive-binary-split your v6.1 cites.
- **"Segmentation-Free Detection of Comic Panels"** — outline-based, structured-background
  tolerant.
- **DeepPanel** (`pedrovgs/DeepPanel`) — U-Net pixel classes {background, panel, border};
  <1 s/page. Bordered-page oriented but the seg formulation beats box IoU for touching panels.
- **"Advancing Manga Analysis: Comprehensive Segmentation Annotations for Manga109"**
  (Xie, Lin, Liu, Li, Wong — **CVPR 2025**) — instance masks incl. **balloon**; iterative
  coarse→refine annotation pipeline.
- **"Towards Accurate Panel Detection in Manga: CNN + Heuristics"** — the hybrid pattern.
- **"ComicScene154"** (`arxiv 2508.16190`) — scene dataset for comic analysis.
- **Survey: "One missing piece in Vision and Language: A Survey on Comics Understanding"**
  (`arxiv 2409.09502`, 2024) — the map of the whole field; § on panel instance segmentation
  notes bleeds/insets/staggered layouts as the open problem and that naive multi-signal
  ensembles over-segment.

**Comic text detection / recognition**
- **AnimeText** (`arxiv 2510.07951`, Oct 2025) — 735K img / 4.2M text-block dataset +
  YOLO12 models (`deepghs/AnimeText_yolo`); explicitly targets stylised/handwritten/
  symbol-confusable text; beats prior datasets. **Most relevant single paper for your OCR gap.**
- **"COMICS Text+: A Comprehensive Gold Standard and Benchmark for Comics Text Detection
  and Recognition"** (ICDAR 2024 W) — first western-comic text det+rec benchmark.
- **"Unconstrained Text Detection in Manga: a New Dataset and Baseline"** (`2009.04042`) +
  `juvian/Manga-Text-Segmentation`.
- **COO: Comic Onomatopoeia** dataset — SFX-specific (irregular, warped) — for the SFX bucket.
- **"Artistic-style text detector + Movie-Poster dataset"** (`2406.16307`), **"Advancing
  WordArt-Oriented Scene Text Recognition"** (`2606.24484`), **"De-rendering Stylized Texts"**
  (`2110.01890`) — the stylised-lettering recognition sub-field.
- **CNN-based speech-balloon / narrative-box segmentation** (IJDAR 2021).

**Comic → text / narrative (your actual product)**
- **"The Manga Whisperer" (Magi, CVPR'24)** / **"Tails Tell Tales" (Magiv2, ACCV'24)** /
  **"From Panels to Prose: Generating Literary Narratives from Comics" (Magiv3, ICCV'25,
  Sachdeva & Zisserman)** — Magiv3 detects chars/text/panels/tails, OCR, implicit panel+text
  ordering; then a VLM captions panels and an LLM writes prose. **This is your pipeline's
  research twin** — but manga-domain; the *architecture* (transcript → grounded caption →
  LLM prose) transfers, the *models* don't.
- **"Comics for Everyone: Generating Accessible Text Descriptions for Comic Strips"**
  (`2310.00698`).
- **"From Panels to Prose: …"** and **"From Comics to Prose"** — narrative generation.

**Benchmarks / competitions**
- **CoMix** (NeurIPS 2024, `2407.03550`) — multi-task: detection + reading order + character
  naming + dialog generation; adds american-comic data.
- **ICDAR 2025 "Comics Understanding"** (Robust Reading Competition ch.31) — "Pick a Panel" +
  multi-task single-page (FR + US + manga): object detection, speaker ID, character
  re-ID/naming, dialog generation.
- **CoSMo** (`2507.10053`) — multimodal transformer for page-stream segmentation.
- **"Semantic Similarity is a Spurious Measure of Comic Understanding"** (`2603.01950`) —
  cautionary, for how you *evaluate* recap quality.

**OCR engine comparisons (2025–26)**
- PaddleOCR-VL-1.6 (96.3 % OmniDocBench), dots.ocr (88.6 EN), GOT-OCR2 — all **GPU**; noted
  for the record. On clean printed text docTR (CER 0.197) ≈ Surya2 (0.191): "VLM is
  inherently more accurate" does **not** hold on clean bubble text — it only wins on the
  stylised bucket, which is why the fallback-only VLM strategy is the right cost trade.

---

## 7. Concrete change-list for this box, ranked by value/effort

1. **Reorder the slicer (low effort, high value).** For tall/scroll inputs make the classical
   gutter+void cutter (`webtoon_panel_slicer.py` v6.1 logic — port the L1–L5 hierarchy into
   `_detect_panels_gutter_aware`) the **primary**; use the Manga109 YOLO only for `full_page`
   grid inputs and as an L0 on borderless blocks. Add **black-void gutter** detection (you may
   only have white-gutter now) — manhwa uses black dividers.
2. **Swap/augment the text-region detector (low–med).** Add `ogkalu/comic-text-segmenter-yolov8m`
   (Apache-2.0, 1024px, webtoon-aware, mask output) as a region proposer; union with RT-DETR
   bubbles. If GPL is OK for internal use, `deepghs/AnimeText_yolo-s` is stronger still.
   Bench both on 20 hand-picked hard panels (SFX, overlay text) the same way you did the OCR
   bake-off.
3. **VLM fallback tier (med).** In `_ocr_with_cascade`, after Tesseract fails, call Gemini
   2.x Flash (free tier) on that crop. Gate hard: only `FAILED`/`UNCERTAIN`, cap N/chapter,
   cache by content-hash (you already have that infra). Big quality win on SFX/warped text
   for ~zero cost.
4. **Fine-tune a webtoon panel YOLO (med, one-off).** Roboflow "Webtoon-Manhwa Panels" (8
   classes) + your own GroundingDINO+SAM pseudo-labels on scraped chapters → `yolo11n` @1024
   on free Colab GPU → ONNX → replace `PANEL_YOLO_PATH`. This is the real fix for
   diagonal/borderless/outbound.
5. **Reading order (low, quality).** For multi-column webtoon moments, borrow Kumiko's or
   Surya's reading-order sort (or Magiv3's panel-DAG idea) instead of pure top-to-bottom.
6. **Don't touch the OCR recognizer.** PP-OCRv5-mobile-EN already won your bake-off.

---

## 8. Link index

Slicers / panels
- https://github.com/njean42/kumiko · https://github.com/pedrovgs/DeepPanel ·
  https://github.com/hummat/panelizer · https://github.com/eskutcheon/ManhwaFormatter ·
  https://github.com/indivisible/webtoon_slicer · https://github.com/reidenong/ComicPanelSegmentation
- https://github.com/hanish3464/WORD-pytorch
- Roboflow: https://universe.roboflow.com/manhwa-pannel/webtoon-manhwa-panels-bwdzz ·
  https://universe.roboflow.com/teste-lk8f9/webtoon-panel
- HF: https://huggingface.co/mosesb/best-comic-panel-detection ·
  https://huggingface.co/spaces/deepghs/manga109_yolo ·
  https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation

Text detection
- https://huggingface.co/deepghs/AnimeText_yolo · https://huggingface.co/datasets/deepghs/AnimeText ·
  https://arxiv.org/abs/2510.07951
- https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m ·
  https://huggingface.co/ogkalu/comic-text-and-bubble-detector
- https://github.com/dmMaze/comic-text-detector · https://huggingface.co/mayocream/comic-text-detector-onnx
- https://github.com/clovaai/CRAFT-pytorch

OCR
- https://github.com/PaddlePaddle/PaddleOCR · https://github.com/RapidAI/RapidOCR ·
  https://huggingface.co/blog/baidu/ppocrv5
- https://github.com/datalab-to/surya · https://github.com/mindee/doctr ·
  https://github.com/felixdittrich92/OnnxTR
- https://github.com/kha-white/manga-ocr · https://kakaobrain.github.io/pororo/miscs/ocr.html
- https://github.com/StoneSteel27/ComiQ

Full pipelines
- https://github.com/ogkalu2/comic-translate · https://github.com/zyddnys/manga-image-translator ·
  https://github.com/dmMaze/BallonsTranslator

Papers / datasets
- Survey: https://arxiv.org/abs/2409.09502 · CoMix: https://arxiv.org/abs/2407.03550 ·
  Comics Datasets Framework: https://arxiv.org/pdf/2407.03540
- Magi: https://arxiv.org/abs/2401.10224 · Magiv2: https://github.com/ragavsachdeva/magi ·
  From Panels to Prose: https://arxiv.org/abs/2503.23344
- Manga109 seg (CVPR'25): https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_Advancing_Manga_Analysis_Comprehensive_Segmentation_Annotations_for_the_Manga109_Dataset_CVPR_2025_paper.pdf
- COMICS Text+: https://arxiv.org/abs/2212.14674 · ComiCap: https://arxiv.org/pdf/2409.16159
- Naver Barrier-Free Webtoon: https://about.webtoon.com/sustainability/43
- Pang panel extraction (ACM MM'14): http://visal.cs.cityu.edu.hk/static/pubs/conf/mm14-panels.pdf
- ICDAR 2025 Comics Understanding: https://rrc.cvc.uab.es/?ch=31

---
*Compiled 2026-09-07. Box: Azure D4as_v7, CPU-only. Cross-refs: `worklog.md`,
`webtoon_panel_slicing_methodology.pdf`, memory `manhwa-recap-perf-baseline`.*
