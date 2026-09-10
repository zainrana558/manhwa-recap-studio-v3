# Worldwide research — free tools to make the recap pipeline better (2026-09)

Scope: everything free/open. Target box: Azure VM, 4 vCPU / 15 GB, **CPU-only, no GPU**.
Ranked by (impact ÷ effort) for *this* setup. Sources at the bottom of each section.

> ⚠️ **Licensing reality check.** "Denji Recaps" is a monetised channel → the pipeline is a
> **commercial** use. That rules out a few otherwise-perfect models — flagged inline with 🚫.
> Apache-2.0 / MIT / CC-BY are fine; "research/non-commercial only" is not.

---

## TL;DR — the shortlist (do these first)

| # | Change | Why | Effort |
|---|---|---|---|
| 1 | **Swap edge-tts → local TTS** — Kokoro-82M for quality, **Supertonic** for big-job speed | edge-tts failed 141 segments on the 328-ch run; local = one consistent voice, no network, no rate limit. On your exact box Kokoro runs ~2× real-time, Supertonic ~8× | M |
| 2 | **Retrain the face/bubble detectors on YOLO26n** | 43 % faster CPU inference, NMS-free, *higher* mAP than YOLOv8n — free win on a CPU box that runs these per strip | M |
| 3 | **Add Cerebras free tier** (1 M tokens/day) as the narration-LLM primary, Groq as fallback | 328-ch jobs blow past Groq's 200 K/day; Cerebras gives 5× the headroom, still $0 | S |
| 4 | **`curl_cffi` (lexiforest fork) for the CF-blocked sources** | toonily / comick-deep / weebcentral are TLS-fingerprint blocks, not JS challenges — `curl_cffi` impersonates Chrome's handshake, no browser needed | S |
| 5 | **`imagededup` near-duplicate pass** before render | webtoons reuse panels in "previously on…" intros and flashbacks; dedup shaves minutes off a 30 h recap and removes repetition | S |
| 6 | **`aeneas` forced alignment** for frame↔narration timing | decouples timing from the TTS engine (currently locked to edge-tts word stamps); CPU C-extensions, ~real-time | M |
| 7 | **Systematic ONNX-Runtime CPU pass** (graph-opt + INT8) on every model | documented up to 4× on CPU; you're 100 % CPU-bound in slice+OCR | M |

---

## 1. Slicer / panel detection

### What's new and free
- **YOLO26 (Ultralytics, Sept 2025).** NMS-free end-to-end head, **43 % faster CPU inference**,
  YOLO26n = 38.9 ms/img on CPU (ONNX) vs YOLOv8n 80.4 ms, and mAP 40.9 vs 37.3. Direct
  drop-in upgrade path for your `anime-face` and `comic-bubble` detectors — retrain the same
  data on `yolo26n`/`yolo26s`, export ONNX, done. License: AGPL-3.0 for the *training code*;
  the **weights you train are yours** (Ultralytics' position), but if AGPL is a worry, the
  inference is just ONNX — you never ship their code.
- **`Kiuyha/Manga-Bubble-YOLO`** (HF) — YOLO26, NMS-free, speech-bubble + text-region, trained
  on **English + Vietnamese + Japanese** manga (5,595 imgs). Closer to your mix than a manga-
  only model. Check the repo license before shipping.
- **`kitsumed/yolov8m_seg-speech-bubble`** (HF) — instance *segmentation* (mask, not box) for
  bubbles; a mask lets you carve the bubble out cleanly instead of a rectangle that clips art.
- **`ogkalu/comic-text-and-bubble-detector`** — the RT-DETR-v2 you already use. Still the best
  free "text + bubble in one model trained on webtoons/manhua/western too". Keep it.
- **Kumiko / `kumikolib`** (njean42, GPL) — pure-OpenCV contour panel finder, zero ML. Useful
  as a **cheap cross-check / tie-breaker** for gutter cuts on bordered pages, and it's tiny.
- **CVPR 2025 "Advancing Manga Analysis"** (Xie et al.) — released **pixel-level segmentation
  masks** for Manga109 across 6 classes (frame, text, onomatopoeia, body, face, balloon) via
  SAM + LoRA. Masks > boxes for content-aware cropping. Manga-only art though.
- 🚫 **Magiv3 / "From Panels to Prose"** (ICCV 2025, Ragav Sachdeva) — a *unified* model:
  panel + character + text + **speech-bubble-tail** detection, OCR, character grounding, and
  the paper pairs it with a VLM to write literary narrative from a chapter. This is *exactly*
  your problem statement. **License is "academic research only" → cannot use commercially.**
  But the *architecture* is the blueprint (see §3).
- **`ShadowB/Manga109-panel-balloon-text-yolov26-segmentation`** — MIT model, YOLO26s-seg,
  mask mAP@0.5 = 0.97… but **explicitly "not suitable for webtoons"** (Manga109 style only).
  Confirms your own finding that manga-trained models die on manhwa.

### Reading order
For your **reconstructed vertical strip** the reading order is trivially top-to-bottom, so
you don't need the manga TSP/tree algorithms. If you ever slice multi-column manga pages:
`manga109/panel-order-estimator` implements the Kovanen recursive pivot-tree method (free).

### Takeaways
1. **Retrain your two YOLO detectors on YOLO26n** — free ~40 % CPU speedup on the per-strip
   detection, and NMS-free means simpler ONNX export.
2. Try **`Kiuyha/Manga-Bubble-YOLO`** as a second bubble opinion (EN/VI/JP training set).
3. A **bubble segmentation mask** (kitsumed) would let `_tall_display_window` and the
   no-cut mask hug the actual bubble shape instead of its bounding box.
4. Steal Magiv3's **speech-bubble-tail** idea — the tail points at the speaker; that's the
   single most useful free signal for "which panel does this bubble belong to" (see §3).

**Sources:**
[YOLO26 blog](https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai) ·
[YOLO26 docs](https://docs.ultralytics.com/models/yolo26) ·
[Kiuyha/Manga-Bubble-YOLO](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO) ·
[kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) ·
[ogkalu/comic-speech-bubble-detector](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) ·
[Kumiko](https://github.com/njean42/kumiko) ·
[CVPR2025 Manga109 segmentation](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_Advancing_Manga_Analysis_Comprehensive_Segmentation_Annotations_for_the_Manga109_Dataset_CVPR_2025_paper.pdf) ·
[From Panels to Prose / Magiv3](https://arxiv.org/abs/2503.23344) ·
[ShadowB Manga109 YOLO26-seg](https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation) ·
[panel-order-estimator](https://github.com/manga109/panel-order-estimator)

---

## 2. OCR (English scanlation text)

You already run **RapidOCR PP-OCRv5-mobile-EN + Tesseract + VLM fallback** — that's a solid
2026 stack. Marginal gains only:

- **PP-OCRv5** (May 2025) is the current PaddleOCR line; you're on the mobile variant which is
  the right call for CPU. The server variant is ~2× slower for ~1-2 pp accuracy.
- **Surya OCR** (datalab-to) — detection+recognition+layout+reading-order, 90+ languages,
  strong on rotated/curved text. **License: GPLv3 code + CC-BY-NC-SA weights, BUT the
  non-commercial restriction is explicitly waived for any org under $2 M revenue AND under
  $2 M funding** — a recap channel qualifies, so it's effectively free for you. Worth a
  **head-to-head on your hard THWT/nano crops**; if it beats RapidOCR on stylised
  SFX-adjacent dialogue, swap it in as the primary.
- **`dots.ocr` (3B, RED AI Lab)** / **DeepSeek-OCR** — layout-aware VLM-OCR, match proprietary
  quality, but **3 B params = too heavy for your CPU** as a per-panel primary. Only viable as
  a *rare* fallback (you already have that tier via the OpenAI-client VLM hook).
- **Korean raw pages**: if a source ever serves un-scanlated Korean, **Pororo** (KakaoBrain,
  Apache-2.0) is what `comic-translate` uses for Korean OCR — free and local.
- **manga-ocr (kha-white)** is **Japanese-only** — not relevant to your English-scanlation input.

**Takeaway:** benchmark **Surya** vs RapidOCR on your existing hard-image test set; keep
whichever wins. Otherwise leave OCR alone — it's not the bottleneck any more.

**Sources:**
[7 best open-source OCR 2025](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025) ·
[PP-OCRv5 collection](https://huggingface.co/collections/PaddlePaddle/pp-ocrv5) ·
[comic-translate OCR stack](https://github.com/ogkalu2/comic-translate) ·
[manga-ocr-2025-onnx](https://huggingface.co/l0wgear/manga-ocr-2025-onnx)

---

## 3. Speech-bubble → speaker attribution (the unsolved bit)

Magiv2/v3 nail this but are 🚫 non-commercial. **Reimplement the pipeline from free parts:**

```
per reconstructed strip:
  bubbles      = ogkalu RT-DETR-v2            (have it)
  tails        = train a tiny YOLO26n on Manga109Dialog / PopManga tail annots   ← the key add
  faces+bodies = deepghs anime_face_detection + a body detector (imgutils person) (have face)
  identity     = deepghs CCIP embeddings → cluster → per-series "character bank"  ← free re-ID
  assign bubble→speaker:
     1. if tail present → nearest face/body along the tail direction
     2. else → nearest face/body in the same panel, tie-break by reading order
  name a cluster once (first chapter it speaks a name near a bubble), reuse chapter-wide
```

- **`deepghs/ccip` (CCIP)** — "Contrastive Character Image Pre-training". `ccip_difference`,
  `ccip_clustering` in `dghs-imgutils`. **ONNX** (`deepghs/ccip_onnx`), fits your ORT infra,
  ~free. Danbooru-trained so anime-styled — like the face detector it'll be **weaker on
  realistic manhwa art**, but it's the only free character-identity model that exists.
- **Manga109Dialog** — large speaker-detection dataset (bubble↔character links). Use it to
  train the tail detector + validate the assignment heuristic.
- **DASSDet** — free face+body detector often cited alongside Magi; alternative to the deepghs
  face model if you want body boxes too.

This is a **multi-week project**, not a config change — but it's the thing that would take the
recap from "80 %, occasionally one beat off" to "actually correct speaker attribution", which
is your #1 recurring complaint.

**Sources:**
[Magi repo](https://github.com/ragavsachdeva/magi) ·
[Tails Tell Tales (Magiv2)](https://huggingface.co/papers/2408.00298) ·
[deepghs/ccip](https://huggingface.co/deepghs/ccip) ·
[imgutils CCIP docs](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html) ·
[Manga109Dialog](https://arxiv.org/pdf/2306.17469) ·
[Occlusion-aware manga re-ID](https://dl.acm.org/doi/fullHtml/10.1145/3595916.3626401)

---

## 4. TTS — the single biggest quality + reliability win

**Current:** edge-tts (MS online, "Andrew" neural) → Piper local fallback. On the 328-ch run
edge-tts failed the 1st attempt on 134 segments and *both* attempts on 7 (→ Piper voice
switch). It's a network dependency that degrades under load.

### The speed problem — measured on *your* hardware
A public CPU-TTS benchmark ran on an **Intel Xeon Platinum 8272CL, 4 cores, 15.6 GB, no GPU
— i.e. this exact Azure VM**:

| model | params | license | RTF on your box | 30 h recap ≈ | quality |
|---|---|---|---|---|---|
| **Supertonic 3** | 99 M | MIT code / OpenRAIL-M weights ✅ | **~0.13** (≈8× real-time) | **~4 h** | good, "competitive WER/CER" |
| **Kokoro-82M** | 82 M | Apache-2.0 ✅ | **~0.47** (≈2× real-time) | **~14 h** | excellent, MOS 4.2, #1 TTS-Arena |
| edge-tts (current) | — | MS ToS ⚠️ | network-parallel, ~fast | ~1.5 h | good, but 141 failures / 30 h |
| Piper | ~20 M | GPL-3.0 (piper1-gpl) ⚠️ | very fast | <1 h | robotic |

**Recommendation:**
- **Normal jobs (≤ ~30 ch): Kokoro-82M** via `kokoro-onnx` — the quality jump is worth the
  minutes.
- **Huge batch jobs (100+ ch): Supertonic 3** — 8× real-time keeps a 328-ch recap's TTS
  under ~4 h and it's ONNX-native (there's an INT8 `sherpa-onnx` build). Or run Kokoro
  6-way-parallel like edge-tts is now (4 cores → effective ~1× RT → ~30 h… still slow; a
  job-size switch is cleaner).
- Either way: **one voice, offline, no rate-limit, no network failure mode.**

### `kokoro-onnx` integration notes
- PyPI `kokoro-onnx` (`thewh1teagle/kokoro-onnx`), **quantized weights**
  (`onnx-community/Kokoro-82M-v1.0-ONNX`, `NeuML/kokoro-fp16-onnx`) — slots into your existing
  `onnxruntime`. Needs `espeak-ng` (already in `start.sh`) via `misaki` G2P. Python 3.10–3.12.
- Voices: `af_heart`, `am_michael`, `am_puck` are the strong American-English picks.
- **Word timestamps**: Kokoro emits token/word timings; or decouple with `aeneas` (§5).

### Keep Piper1-GPL as the true-offline fallback
`OHF-Voice/piper1-gpl` (⚠️ **GPL-3.0**; original MIT Rhasspy repo archived Oct 2025). Fine to
run on a hosted service; GPL only bites if you *distribute* the pipeline.

### Not recommended for CPU-only
F5-TTS / XTTSv2 / Chatterbox / Orpheus — better prosody / cloning, need a GPU.

**Sources:**
[CPU TTS benchmark on the same Xeon 8272CL/4-core box](https://heyneo.com/blog/kokoro-supertonic-inflect-nano-pocket-tts-cpu-benchmark) ·
[Open-source TTS 2026 comparison](https://ocdevel.com/blog/20250720-tts) ·
[Best local TTS 2026](https://localaimaster.com/blog/best-local-tts-models) ·
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) ·
[Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) ·
[Kokoro word timestamps](https://ryanwelch.co.uk/blog/kokoro-word-timestamps/) ·
[Supertonic (GitHub)](https://github.com/supertone-inc/supertonic) ·
[Supertonic-3 (HF)](https://huggingface.co/Supertone/supertonic-3) ·
[sherpa-onnx Supertonic INT8](https://huggingface.co/csukuangfj2/sherpa-onnx-supertonic-tts-int8-2026-03-06) ·
[misaki G2P](https://github.com/hexgrad/misaki) ·
[piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)

---

## 5. Frame ↔ narration timing

Currently welded to edge-tts word timestamps. Free, TTS-agnostic options:

- **`aeneas`** (readbeyond, GPL) — the classic forced aligner. Python + C extensions (MFCC/DTW),
  **CPU, ~real-time**, uses espeak-ng internally, outputs SRT/JSON/VTT sync maps. Give it the
  narration text + the rendered chapter audio → get per-fragment timestamps regardless of
  which TTS made the audio. This is the clean way to switch to Kokoro without rewriting timing.
- **`stable-ts`** — Whisper-based forced alignment; heavier (Whisper model) but also gives you
  **burn-in subtitle** timings for free if you ever want captions on the video.
- **`faster-whisper` + wav2vec2 alignment** (WhisperX-style, `--compute_type int8 --device cpu`)
  — overkill unless you want ASR-verified narration.
- **Silero VAD** (1 MB, <1 ms per 30 ms chunk on CPU, MIT) — if you want to detect actual
  speech regions in a chapter's assembled audio for pacing / gap-trimming / ducking triggers.

**Takeaway:** adopt **`aeneas`** as the timing layer — it unblocks the Kokoro migration.

**Sources:**
[aeneas](https://github.com/readbeyond/aeneas) ·
[stable-ts](https://github.com/jianfch/stable-ts) ·
[WhisperX](https://github.com/m-bain/whisperX) ·
[Silero VAD guide](https://aiadoptionagency.com/silero-vad-voice-activity-detection/)

---

## 6. VLM panel description (`--describe-visuals`)

For a CPU box, smallest-that-works wins:

| Model | Size | CPU verdict | Notes |
|---|---|---|---|
| **SmolVLM2** (256M / 500M) | 0.25–0.5 B | ✅ fast | Apache-2.0, HF-native, "captioning + basic VQA". Start here. |
| **Moondream 2** | 2 B (int4/int8) | ✅ usable | Apache-2.0, does caption **+ detection + pointing** in one model — could also give you cheap face/object boxes. |
| **Florence-2** | 0.23 / 0.77 B | ✅ fast | MIT, task-prompted (`<CAPTION>`, `<OD>`, `<DENSE_REGION_CAPTION>`). You may already know it. |
| Qwen2.5-VL-3B GGUF (Q4) | ~2–3 GB | ⚠️ slow | `Mungert/Qwen2.5-VL-3B-Instruct-GGUF` via `llama.cpp` + mmproj. Best quality/size but seconds per image on CPU. |
| JoyCaption Beta One | 8 B | 🚫 CPU | Great anime coverage, GGUF exists, but 8 B is impractical without a GPU. |

**Takeaway:** default `--visual-provider` to **SmolVLM2-500M** locally (free, no key), keep the
cloud VLM as the opt-in quality tier. Moondream2 if you also want it to double as a detector.

**Sources:**
[VLMs 2025 (HF)](https://huggingface.co/blog/vlms-2025) ·
[SmolVLM](https://huggingface.co/blog/smolvlm) ·
[Moondream 2](https://blog.roboflow.com/moondream-2/) ·
[Qwen2.5-VL-3B GGUF](https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF) ·
[JoyCaption](https://github.com/fpgaminer/joycaption)

---

## 7. Recap-script LLM (narration rewrite)

You use **Groq free tier** (200 K tokens/day ceiling). A 328-ch job with ~24 K panels of
verbatim text easily exceeds that → the run silently degrades to raw-verbatim mode.

- **Cerebras free tier — 1 M tokens/day** (Llama-3.3-70B / Qwen). 5× Groq's ceiling, same
  OpenAI-compatible API, sub-second. **Add as the primary, Groq as fallback.**
- **Google AI Studio free** — Gemini 2.5 Flash-Lite: 1,000 requests/day, 1 M-token context
  (whole-chapter context in one call). Flash: 250/day.
- **OpenRouter** — ~30 free models behind one key (DeepSeek-R1, Llama-3.3, Qwen3, Gemma-3);
  50 free req/day under $10 credit, 1,000/day at ≥$10.
- **SambaNova** — $5 credit, Llama-3.1-405B.
- **Local, zero-rate-limit fallback**: `Qwen2.5-3B-Instruct` or `Gemma-3-4B` GGUF (Q4) on CPU
  via `llama.cpp` — slow (~10-30 tok/s) but free and unlimited; fine as the "Groq+Cerebras
  both exhausted" tier for huge jobs.

**Takeaway:** wire a **provider cascade: Cerebras → Groq → Gemini-Flash-Lite → local GGUF**.
Kills the "silently fell back to verbatim" failure on big jobs.

**Sources:**
[Free LLM APIs 2026 (OpenRouter)](https://openrouter.ai/blog/tutorials/free-llm-apis-compared/) ·
[Free LLM API tiers](https://ianlpaterson.com/blog/free-llm-api-2026/) ·
[awesome-freellm-apis](https://github.com/open-free-llm-api/awesome-freellm-apis)

---

## 8. Translation (non-English scanlations)

You lean on Groq for this. Free local options if you want offline / unlimited:

- **NLLB-200-distilled-600M** — CPU-feasible, 200 languages incl. Korean↔English. The 1.3B is
  better but ~3 GB. CC-BY-NC 🚫 for the *checkpoints* (Meta) — **non-commercial**, watch out.
- **MADLAD-400-3B-mt** (Google, Apache-2.0 ✅) — 419 languages, ~12 GB, heavy but commercially
  clean.
- **Gemma-3** via the free API cascade (§7) — quality rivals cloud MT, and it's already in
  your key rotation.
- **`sioaeko/NLLB_translator`** — reference multi-engine wrapper (local NLLB/MADLAD + free
  Gemini/Groq) worth reading for the fallback structure.

**Takeaway:** for commercial safety, prefer **Gemma-3 via API** or **MADLAD-400** over NLLB.

**Sources:**
[Open-source translation models 2026](https://picovoice.ai/blog/open-source-translation/) ·
[NLLB_translator](https://github.com/sioaeko/NLLB_translator) ·
[Best local LLMs for translation](https://insiderllm.com/guides/best-local-llms-translation/)

---

## 9. Scraping — unblock the CF sources + add more

Your memory says toonily / comick-deep / weebcentral are Cloudflare-blocked from the Azure IP.

- **`curl_cffi`** — **still actively maintained** by the **lexiforest fork** (v0.16.2b1, Aug
  2026 — the "deprecated Feb 2025" note is about the *old* yifeikong repo). Impersonates
  Chrome/Firefox/Safari **TLS/JA3/HTTP2 fingerprints** with no browser. For sites that block
  on *fingerprint only* (many manga aggregators) this is a one-line `requests` swap:
  `from curl_cffi import requests; requests.get(url, impersonate="chrome124")`. **Try this
  first** on the 3 blocked sources — cheapest possible fix.
- If a site throws a real **JS challenge / Turnstile**: `nodriver` (successor to
  undetected-chromedriver), `camoufox` (hardened Firefox), or `SeleniumBase UC mode`. All
  free, all need a headless browser (~300 MB RAM each) — run on demand, not resident.
- lexiforest also ships **`brimp`** (tiny browser for solving JS challenges) + **`impers`**
  (JS bindings) — lighter than full Chromium.
- **`AIO-Webtoon-Downloader` (zzyil)** — actively maintained, has working scrapers for
  **asuracomic.net, weebcentral.com, comix.to, mangafire.to, mangataro.org, manganato.gg,
  bato.to** + cross-site search + multi-source fallback + scanlation-group prioritisation.
  Either port its site adapters or shell out to `aio-dl.py` as a fallback downloader.
- **`gallery-dl`** — mature, huge supported-sites list (webtoons.com, many aggregators),
  handles auth/cookies, great filename templating. Good as a "source we don't have a scraper
  for" catch-all.

**Takeaway:** (1) swap the blocked-source HTTP client to **`curl_cffi` impersonate**; (2) add
**bato.to / comix.to / mangafire** adapters (crib from AIO-Webtoon-Downloader); (3) keep
`gallery-dl` as the universal fallback.

**Sources:**
[curl_cffi (lexiforest)](https://github.com/lexiforest/curl_cffi) ·
[Bypass Cloudflare 2026 (scrapfly)](https://scrapfly.io/blog/posts/how-to-bypass-cloudflare-anti-scraping) ·
[AIO-Webtoon-Downloader](https://github.com/zzyil/AIO-Webtoon-Downloader) ·
[gallery-dl supported sites](https://github.com/mikf/gallery-dl/blob/master/docs/supportedsites.md)

---

## 10. Near-duplicate frame removal

Webtoons **reuse panels** — "previously on…" recap strips, flashbacks, repeated reaction
shots. On a 30 h recap that's real bloat and on-screen repetition.

- **`imagededup` (idealo, Apache-2.0)** — pHash / dHash / wHash / CNN embeddings + NN search
  (NGT / hnsw for speed). Run a **dHash (fast) pass over the composed frames per chapter**,
  drop near-dups within a window; optionally a **CNN pass across the whole recap** to catch
  the recycled flashback panels.
- **More robust: a CLIP / DINOv2 / SigLIP embedding + cosine-distance pass** — survives the
  colour-grade / crop / scale changes that break pHash on re-used panels. DINOv2-small is
  ~22 M, ONNX-able, CPU-fine. This is also the same embedding you'd use for scene grouping
  (§13/§14), so one model, two uses.
- **imgutils CCIP** (§3) for character-level "same character, different panel" if you want to
  cap how many times one face closeup appears.
- Cheap alternative: `imagehash` (dHash) + Hamming-distance dedupe, ~20 lines, no new heavy dep.

**Takeaway:** add a **dHash dedupe** in `_frame_reconstructed_strip` / the render frame list
(threshold ~6-8 bits). Easy win, immediately shorter and less repetitive recaps.

**Sources:**
[imagededup](https://github.com/idealo/imagededup) ·
[duplicate image detection with pHash](https://benhoyt.com/writings/duplicate-image-detection/)

---

## 10b. Watermark / credit-banner removal (the "READ AT MANHWA-FREAK.COM" problem)

F3 in the bug log only *drops* whole credit frames; a watermark **stamped across real art**
(the mustache-guy panel) still leaks. Free fix:

- **IOPaint** (`Sanster/IOPaint`, Apache-2.0) with **LaMa** runs on CPU:
  `iopaint start --model=lama --device=cpu`. Feed it the panel + a mask of the watermark
  region (you already detect text boxes → dilate the ones near the bottom edge that OCR
  flags as noise) → clean fill. There's also a **manga-specific inpainting model** and an
  **AnimeMangaInpainting** finetune that handle screentones better than vanilla LaMa.
- Cheaper heuristic (no model): the aggregator bars are **flat poster-colour full-width
  bands** — detect a run of rows with near-constant hue + high saturation + text, and just
  `_safe_border_trim` them off like a letterbox. Covers the banner-at-edge case (most of them)
  without inpainting.

**Takeaway:** add the flat-colour-band trim now (cheap); add **IOPaint+LaMa** as an opt-in
`RECAP_INPAINT_WATERMARKS=1` pass for the stamped-on-art case.

**Sources:**
[IOPaint](https://github.com/Sanster/IOPaint) ·
[IOPaint manga model](https://www.iopaint.com/models/erase/manga) ·
[Er0mangaInpaint (LaMa manga finetune)](https://github.com/Er0manga/Er0mangaInpaint)

---

## 11. Video & audio polish

- **Ken Burns without the shudder**: ffmpeg `zoompan` jitters because it snaps to integer
  pixels per frame. Fixes: render at **2–4× the target size then downscale**, use a **high
  fps** (`fps=60`), keep `z`/`x`/`y` expressions linear, or use one of the sub-pixel
  `zoompan` replacements. For a mostly-static recap you may not need it at all — a slow 3-5 %
  zoom is enough and reads as intentional.
- **Audio ducking** — `ffmpeg` **`sidechaincompress`**: music track keyed by the narration
  track, `threshold≈0.02 ratio≈8-12 attack≈20 release≈300`. Music auto-drops under speech,
  comes back in gaps. One filter_complex, no extra deps.
- **Free BGM, commercial-OK, no attribution**: **Pixabay Music** (best — CC0-ish, commercial
  OK, no attribution), **Mixkit**, **Fesliyan Studios**, **Chosic**. Pre-download a small
  library of loopable cinematic / lo-fi beds and pick by tag.
- **Loudness**: you already run `loudnorm` per chapter — good. Do a **final `loudnorm` pass
  on the merged file** too (two-pass, `I=-14 TP=-1.5 LRA=11` for YouTube).

**Sources:**
[ffmpeg Ken Burns jitter fix](https://www.ffmpeg-micro.com/blog/ffmpeg-zoompan-filter-ken-burns-zoom-and-pan-without-the-jitter) ·
[ffmpeg audio mixing + ducking](https://www.ffmpeglab.com/articles/ffmpeg-audio-mixing-amix-guide.html) ·
[Pixabay Music](https://pixabay.com/music/) · [Mixkit](https://mixkit.co/free-stock-music/)

---

## 12. Cross-cutting: squeeze the CPU

You're CPU-bound in slice + OCR (~3.7 h for 328 ch). Free levers:

- **ONNX Runtime graph optimization + INT8 dynamic quantization** on *every* model
  (RT-DETR text/bubble, face, panel YOLO). Documented **up to 4× on CPU**; you already do
  this for some. Use **HF `optimum` + `ORTQuantizer`** and the ORT `transformers` optimizer;
  profile with a `bert_perf_test`-style script to find the slow op.
- **YOLO26n** for the detectors (§1) — 43 % on top of the above.
- Set `OMP_NUM_THREADS` / ORT `intra_op_num_threads` = **4** explicitly (don't let it
  oversubscribe); `sess_options.execution_mode = ORT_SEQUENTIAL` for small models.
- **Batch OCR by chapter** (you do) and make sure the OCR service uses `OCR_CONCURRENCY`
  matched to cores.
- The `_reclaim_memory` (malloc_trim) fix you just added is the right pattern — apply the
  same to the slice process between chapters if its RSS ever creeps.

**Sources:**
[ORT transformers optimization](https://onnxruntime.ai/docs/performance/transformers-optimization.html) ·
[ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) ·
[Optimum + ORT + quant walkthrough](https://www.marktechpost.com/2025/09/23/coding-implementation-to-end-to-end-transformer-model-optimization-with-hugging-face-optimum-onnx-runtime-and-quantization/)

---

## 13. Reference pipelines to mine (open source, same problem space)

- **`ogkalu2/comic-translate`** — same author as your bubble/text detectors. Full free-local
  stack: RT-DETR-v2 detect · manga-ocr(JP)/Pororo(KO)/PP-OCRv5(EN) OCR · LaMa/AOT inpaint ·
  (LLM translate = paid). Best reference for "which free model per language".
- **`dmMaze/BallonsTranslator`** + the **`-Pro` fork** ("90+ mix-and-match modules") —
  explicitly "adapted to extreme aspect ratios such as webtoons". Mine its module list.
- **`dmMaze/comic-text-detector`** — YOLOv5 + UNet-mask + DBNet-line, trained on
  Manga109-s + DCM + synthetic. The detector `manga-image-translator` uses. Free.
- **`mayocream/koharu`** — Rust, ML manga translator; clean separation of detect/OCR/inpaint.
- **`zyddnys/manga-image-translator`** — the OG; `det_auto_rotate` trick for vertical text.
- 🚫 **Magi / Magiv2 / Magiv3** — the SOTA for detect+order+speaker+names, non-commercial.
- **Commercial competitors as quality benchmarks** (not free): RecapSynth, MagaRecap, FluxNote
  — all do manhwa→recap-video; useful to see the bar for pacing / script style.

**Sources:**
[comic-translate](https://github.com/ogkalu2/comic-translate) ·
[BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) ·
[BallonsTranslator-Pro](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro) ·
[comic-text-detector](https://github.com/dmMaze/comic-text-detector) ·
[koharu](https://github.com/mayocream/koharu) ·
[manga-image-translator](https://github.com/zyddnys/manga-image-translator) ·
[RecapSynth](https://www.recapsynth.com/home)

---

## 14. Datasets (if you ever fine-tune)

- **Manga109 / Manga109-s** — bboxes for frame/text/face/body + the new **CVPR2025 masks** +
  **Manga109Dialog** (speaker links) + **Manga109-v2026** (refreshed annots).
- **PopManga** (Magi) — panels, text, characters, + tail boxes / essential-text flags (v2).
- **DCM772**, **eBDtheque**, **COMICS** — western comics, panel + balloon + text.
- **ComicScene154** (EMNLP 2025) — **scene-boundary** annotations (which panel starts a new
  narrative arc) — directly useful for structuring the recap into beats. Golden-Age US comics.
- **`MS92/MangaSegmentation`**, **`ShadowB/Manga109_RegionLevelTextSegmentation`** — HF seg
  datasets.
- ⚠️ almost all comic datasets are **research-license** — fine for training a model you then
  use, murky for redistribution. Manga109 requires an academic request.

**Sources:**
[Manga109-v2026](https://arxiv.org/html/2605.21182) ·
[ComicScene154](https://arxiv.org/abs/2508.16190) ·
[Comics understanding survey](https://arxiv.org/pdf/2409.09502)

---

## Priority-ordered action list

**This week (S):**
1. `curl_cffi` impersonate on the 3 CF-blocked sources.
2. Cerebras key → provider cascade for narration LLM (Cerebras → Groq → Gemini → local).
3. dHash (or DINOv2-embedding) dedupe pass on composed frames.
4. Flat-colour-band trim for aggregator watermark bars.

**This month (M):**
5. Migrate TTS off edge-tts + `aeneas` timing layer. **Kokoro-82M** for normal jobs,
   **Supertonic 3** for 100+ ch jobs; Piper1-GPL as the last-resort offline fallback.
6. Retrain face + bubble detectors on YOLO26n, re-export INT8 ONNX.
7. Systematic ORT graph-opt + INT8 quant pass on all detectors; pin thread counts.
8. Default `--describe-visuals` to local SmolVLM2-500M (or Moondream2 if you want it to
   double as a detector).
9. Opt-in IOPaint+LaMa watermark inpainting for stamped-on-art cases.

**Next quarter (L):**
10. Free re-implementation of bubble→speaker (tail detector + CCIP character bank) —
    the one change that fixes the recurring "wrong speaker / one beat off" complaint.
11. Add bato.to / comix.to / mangafire scrapers (crib from AIO-Webtoon-Downloader).
12. Scene-boundary segmentation (ComicScene154 approach + DINOv2 embeddings) to structure
    recaps into beats instead of a flat panel stream.

---

## Appendix: things checked and rejected for this box

- **Magi / Magiv2 / Magiv3** — perfect fit, 🚫 non-commercial license.
- **Manga109-trained panel/seg models** (ShadowB YOLO26-seg, Manga109 SAM+LoRA) — explicitly
  fail on webtoons; matches your own testing.
- **manga-ocr (kha-white)** — Japanese only; your input is English scanlation.
- **dots.ocr / DeepSeek-OCR / JoyCaption-8B / Qwen2.5-VL-7B+** — accuracy is there, but 3–8 B
  params on a 4-core CPU = seconds-to-minutes per image. Fine as a *rare* fallback, not a
  primary.
- **F5-TTS / XTTS-v2 / Chatterbox** — need a GPU for tolerable RTF.
- **NLLB checkpoints** — CC-BY-NC 🚫; use MADLAD-400 (Apache-2.0) or Gemma-3-via-API instead.
- **FlareSolverr** — deprecated, CAPTCHA solvers dead as of 2026; use `curl_cffi` / nodriver.
- Full browser automation (Playwright/Puppeteer stealth) — works but ~300 MB RAM resident
  per instance on a 15 GB box already running 3 services; only spin up on demand.

---

## Implementation status — 2026-09-09

All items below are **opt-in** and **degrade gracefully** (missing package /
model / venv → the pipeline behaves exactly as before).

| # | item | status | how to turn on |
|---|---|---|---|
| 1 | LLM narration/translation provider cascade | ✅ done | set any of `OPENAI/GROQ/GEMINI/OPENROUTER_API_KEY` — auto-cascades |
| 2 | OCR verbatim — local GOT-OCR2.0 fallback tier + garble routing | ✅ done | `OCR_LOCAL_VLM=1` (restart OCR svc). Garble routing is always-on. |
| 3 | Near-duplicate frame dedup (512-bit dHash on OCR crop) | ✅ done | on by default; `RECAP_DEDUP_FRAMES=0` to disable |
| 4 | F2/F3/F4 overnight bugs | ✅ fixed | see `research/overnight-bug-log.md` |
| 5 | ONNX Runtime graph-opt + BLAS thread pinning | ✅ done | automatic; `RECAP_ORT_THREADS` / `RECAP_CPU_THREADS` override |
| 6 | YOLO26n detectors — loader auto-pickup + Colab notebook | ✅ done (wiring) | drop `*_yolo26n.onnx` in `pipeline/models/*/`; train via `pipeline/training/yolo26_detectors_colab.ipynb` |
| 7 | Kokoro-82M neural TTS (isolated venv) | ✅ done | `bash pipeline/setup_kokoro.sh` then `RECAP_TTS_ENGINE=kokoro` |
| 8 | SmolVLM2-500M local panel captioner | ✅ done | `--visual-provider smolvlm` (or `RECAP_VISUAL_LOCAL=1`) |
| 9 | Bubble → speaker attribution (tail + CCIP bank) | ⬜ not started | — |

### Measured on this box (4-vCPU Xeon 8272CL, CPU only)

- **Kokoro-82M**: RTF ≈ 0.23 (≈4× real-time), init 0.4 s, 54 voices, `af_heart` default.
  Isolated venv because kokoro-onnx pins numpy≥2 (main venv is numpy<2).
- **SmolVLM2-500M**: ≈3.5 s/panel *with the processor `longest_edge` pinned to 512*
  (the 2048 default tiles each frame 4×4 → ≈30 s/panel). 256M variant hallucinates
  a narrative — not usable.
- **GOT-OCR2.0 (580M)**: ≈10 s/panel. Read a spiky "HE'S BEHIND US!!!" bubble that
  RapidOCR returned as "A HE'S BEHIND iiisn" at conf 0.84. Far too slow for primary
  OCR — wired as the last-resort tier only, per-process capped.

### The real OCR verbatim bug (found while testing on the Nano Machine job)

RapidOCR (PP-OCRv5 mobile) sometimes returns a **confidently wrong** transcription
that the cascade accepts as SUCCESS and never re-reads. Real Nano Machine ch.1
cases at conf 0.84–0.99:

| RapidOCR (prod) | should be |
|---|---|
| `A HE'S BEHIND iiisn` | HE'S BEHIND US!!! |
| `WHAT I CTNOHS DO?!?` | WHAT SHOULD I DO?!? |
| `IILLHS GUESS THERE IS NO CHOICE` | SHIT! I GUESS THERE IS NO CHOICE |

Fix, in `mini-services/paddleocr-service/main.py`:
1. **`_looks_like_ocr_garble()`** — flags a *mostly-clean* result that contains a
   structurally-mangled token (`_token_looks_mangled`: 4+ consonant run, ≥78%
   consonant density, vowelless ≥4 chars, lowercase 3-same run) or a casing-mismatched
   blob. Membership-checked against the bundled wordninja list so real words
   ("SHREWD", "DEATHBED", "EXTRAORDINARY") and known romanised names ("CHEON",
   "JANG") are never flagged. **0 false positives across 51 real Nano ch.1 lines;
   catches 3 of the 4 real garbles.**
2. A flagged panel is demoted from SUCCESS. If `OCR_LOCAL_VLM=1`, **GOT-OCR2 is
   called immediately** (before PaddleOCR can short-circuit with a clean-but-
   truncated read) and its output wins if it isn't itself garbled.

Warm A/B on all 102 Nano ch.1 crops: **3 panels changed, all 3 garble → correct
text; the other 99 byte-identical to prod at ~0.3 s each.** GOT-OCR2 cost: ~10 s ×
(garble panels + normal UNCERTAIN-tier panels), ≈1 min extra per 100-panel chapter.

### The BIGGER OCR fix (2026-09-09, second pass) — detection params were wrong

Testing RapidOCR directly vs through the service revealed the service's tuned
detection params (`det_db_box_thresh=0.4`, `det_db_unclip_ratio=1.8`) were
**actively worse** than RapidOCR's own defaults (`0.5` / `1.6`) on the current
reconstruction-path display-window crops. The old pair came from a param sweep on
the *pre-reconstruction tile crops* — a different crop distribution. `unclip 1.8`
over-dilates detection boxes so adjacent lines/glyphs bleed together and PP-OCRv5
mis-reads or truncates:

| crop | old 1.8/0.4 (prod) | new 1.6/0.5 |
|---|---|---|
| frame_00059 | `A HE'S BEHIND iiisn` | `A HE'S BEHIND US!!!` |
| frame_00047 | `SOOO MOH IT FEEL...` | `HOW DOES IT FEEL...` |
| frame_00048 | `AS I TRICK THEM TWICE` | `AS I THOUGHT, I CAN'T TRICK THEM TWICE ... DIE AN EXCRUCIATING DEATH!!` |
| frame_00073 | `EXACTLY WHAT IS THAT` | `EXACTLY WHAT IS THAT LIGHT?!` |
| frame_00074 | `...RISEN TO THE HIGH-` | `...RISEN TO THE HIGH- EST LEVEL?!` |
| frame_00077 | `A LOST CALSE` | `A LOST CAUSE` |
| frame_00085 | `HAPPY MOOD EVEN DEATHBED?` | `HAPPY MOOD EVEN THOUGH HE'S AT HIS DEATHBED?` |
| frame_00097 | `SIHI NANO MACHINE ... AR` | `THIS NANO MACHINE ...` |
| frame_00020 | `WAY THOSE BASTARDS...` | `THERE'S NO WAY THOSE BASTARDS...` |

Full Nano ch.1 A/B: **~11 real dialogue fixes, ~1 minor regression** (frame_00022
gains a mid-sentence junk shard — but its old read also had an error), rest neutral
(spacing / junk-for-junk on texture panels). **Zero model change, zero speed cost.**
Changed the `OCROptions` field defaults; env overrides `RECAP_OCR_UNCLIP` /
`RECAP_OCR_BOX_THRESH`. Production OCR service restarted with the fix.

Also added `_normalize_vlm_ocr_text()` — routes GOT-OCR2 / cloud VLM output through
the same repair chain the classical engines use (`_clean_and_normalize_ocr_text` +
`_repair_and_denoise`) plus GOT-specific fixes ("HE' S" → "HE'S", "IDON'T" →
"I DON'T", doubled-word collapse).

### Third pass — recovery without speed cost (user: "no speed compromise",
### "correct not eliminate" the garbles)

- **Tight-unclip recovery**: when `_looks_like_ocr_garble` flags a result, re-run
  RapidOCR at `unclip=1.3` and keep whichever `_ocr_text_quality` scores cleaner.
  Fires only on flagged panels (~3-5/chapter), ~0.9 s each. Fixes line-merge
  garble: `IT SEEOH SS SWIIS` → `IT SEEMS AS THOUGH`.
- **Widened garble detection**: also flags a run of 2+ consecutive non-dict tokens
  in a dict-heavy line (still 0 FP on the 51-line set).
- **Spurious-article strip**: `A HE'S BEHIND US` → `HE'S BEHIND US`,
  `...NO. A HE DISAPPEARED` → `...NO. HE DISAPPEARED` (removes an invented word,
  doesn't lose a real one).
- **Observer LLM auto-enabled**: `_observer_fix_ocr` (verbatim style) now gets
  `--observer` from pipeline-service whenever any LLM key is present. Repairs
  garbled tokens IN PLACE (word count preserved ±1, every clean word/name must
  survive) — corrects, never deletes. Guard only lets it touch STRONG-garble
  tokens (vowelless / 4+ consonant run / ≥78 % consonant density) — `CTNOHS` →
  `SHOULD` yes, `EHOH` (indistinguishable from a name) stays. Now gets the
  previous line as context. **llama3.2:3b is too weak — needs a free Groq/Gemini
  key.**

**Nano ch.1 net (vs original prod): ~13 dialogue lines fixed, 0 regressions, +1 s
total (one tight-unclip pass).** Remaining ~5: `CTNOHS`, `EHOH`, `I IN ITS`,
trailing SFX blobs `TOTN`/`EHOH` — need a Groq key (observer) or `OCR_LOCAL_VLM=1`.
Literal "zero garble on stylised SFX" isn't reachable with classical OCR at these
speeds; the `--narration-style cleanup` LLM rewrite removes them from the final
narration regardless.

### Website voice picker (2026-09-09)

Default narration voice is now **`am_michael`** (Kokoro-82M, local). `voice-selector.tsx`
has a "Local · Kokoro" group (11 voices) at the top; picking any of them routes the
job to the local neural engine, anything else stays edge-tts. Voice preview works
for both. No new schema field — the existing `voice` column carries the id, matched
by `_KOKORO_VOICE_RE` in `master_pipeline.py`.

### SmolVLM2 for OCR? No.

Tested — SmolVLM2-500M paraphrases ("A group of people gathered around a table"),
it doesn't transcribe. It stays the *visual captioner* (#8). For verbatim text the
answer is a dedicated OCR model: **GOT-OCR2.0** in the fallback tier (above), or
the standing #2 recommendation to retrain / move off PP-OCRv5 *mobile*.

### Pre-existing test regression (NOT from this session's work)

`tests/test_canonical_architecture.py` — 6/7 fail on `main` (HEAD passes 7/7). The
prior session's slicer rework makes `_frame_pages_reference_style` emit **0 panels**
on the test's synthetic 2-rectangle pages (`800×1200`, plain white). Real jobs are
unaffected (328-ch + 3-ch jobs rendered real panels), but it means the reference
path has an untested failure mode on minimal/degenerate pages. Reverting only this
session's changes does NOT fix it — worth a separate look.

### Also observed

The Nano Machine scanlation leaves Korean SFX untranslated (쉬익 / 카직 / 탁) — the
English OCR model returns junk ("of", "11", "$fra") for those crops. Low-value but
mostly harmless (short tokens, filtered downstream). Two other jobs (`cmtrkxad7…`,
`cmtrjlivt…` = "The House Without Time") are **full Korean raws** — English OCR
can't read them at all; that's a source-selection issue, not OCR.
