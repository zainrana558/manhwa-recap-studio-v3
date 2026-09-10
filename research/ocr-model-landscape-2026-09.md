# OCR model landscape for English-scanlation manhwa — 2026-09-09

Question: is there a better OCR than the current RapidOCR / PP-OCRv5-EN mobile,
specifically for manhwa / webtoon English scanlations?

## TL;DR

**No off-the-shelf model is targeted at English-scanlation comic OCR.** Every
"manga OCR" is Japanese-raw focused. The realistic options:

| option | verdict for this pipeline |
|---|---|
| **PP-OCRv5-EN mobile (current)** | Best *classical* choice for English. English-specific rec head → no CJK hallucination. ~0.2 s/panel. Keep it. |
| **PP-OCRv6** (June 2026, tiny/small/medium) | Available now in RapidOCR 3.9.2 (`OCRVersion.PPOCRV6`). BUT it's a single **multilingual** model — on this content (English lettering + untranslated Korean SFX) it **hedges toward CJK**: v6-medium hallucinated Chinese on `frame_00101` ("ACTIVATION. -2-人 喇勞魔神"), added "Goo UCK" garbage on `frame_00097`. v6-medium is also **5× slower** (955 ms vs 200 ms). **Net worse here.** The +5 % benchmark gain is on clean documents, not stylised comic text. |
| **GOT-OCR2.0** (0.58 B, wired as `OCR_LOCAL_VLM=1`) | Genuinely more accurate on garbled panels (CTNOHS→SHOULD, "I IN ITS"→"I GUESS"). ~10 s/panel on CPU. After the classical fixes only ~1 panel/chapter is flagged → ≈1 h added to a 328-ch job. Good opt-in. |
| **PaddleOCR-VL 0.9B** (SOTA on OmniDocBench, beats GPT-4o) | ~20-32 s/page *on a GPU*; ~60-120 s/panel on this CPU box. Unusable as primary. |
| **Surya-2 / dots.ocr / MonkeyOCR / DeepSeek-OCR** | Doc-OCR VLMs, 1-3 B, all GPU-oriented. Not comic-tuned. |
| **manga-ocr family** (kha-white, l0wgear, bluolightning) | Japanese only; explicitly bad at English letters/punctuation. |
| **Naver CLOVA OCR** | Commercial API (KR/JP/EN), strong, not free/local. |
| **WORD-pytorch** (Webtoon Object Recognition) | Korean webtoon detection + **Korean** recognition. Not English. |

## The real step-change: fine-tune

`jzhang533/PaddleOCR-VL-For-Manga` proved it: fine-tuning PaddleOCR-VL on
**0.1 M Manga109-s text-region crops + 1.5 M synthetic samples** took full-sentence
accuracy **27 % → 70 %**. Training code is public.

For English manhwa the same recipe applies with different data:
- **Synthetic**: render English comic-style lettering (Wild Words / Anime Ace /
  CC fonts) onto bubble backgrounds at varied warp/size/contrast — cheap, unlimited.
- **Real**: the recap-audio idea — Whisper a recap channel that reads dialogue
  **verbatim** (not all do; pick ones that do), align narration segments to panels
  with the existing `pipeline/training/align_frames_to_source.py`, use
  (panel crop → transcript) pairs. Caveat: paraphrasing channels would teach the
  model to paraphrase — bad for OCR.
- **Target model**: fine-tune **GOT-OCR2.0** (transformers-native, 0.58 B, already
  wired) or a PP-OCRv5/v6 rec head. GOT-OCR2 fine-tune = image→text pairs directly,
  much simpler than a line-level classical rec fine-tune.
- **Compute**: free Colab/Kaggle GPU (same as the YOLO26 notebook). Can't train on
  this CPU box.

## Recommendation

1. Keep PP-OCRv5-EN mobile as primary (+ the 2026-09-09 fixes: params, tight-unclip
   recovery, repair chain).
2. Turn on `OCR_LOCAL_VLM=1` for the residual garbles (~1 h / 328-ch job).
3. Longer term: a GOT-OCR2 fine-tune on synthetic English comic lettering + verbatim
   recap-audio pairs — the only path to a real accuracy jump.

## Sources

- [PP-OCRv6 (HF blog)](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6) ·
  [PP-OCRv6 paper 2606.13108](https://arxiv.org/abs/2606.13108) ·
  [RapidOCR #686 (v3.9.0 milestone, closed)](https://github.com/RapidAI/RapidOCR/issues/686)
- [PaddleOCR-VL 2510.14528](https://arxiv.org/abs/2510.14528) ·
  [PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga) ·
  [PaddleOCR-VL local inference ~20-32s/page on GPU (#18164)](https://github.com/PaddlePaddle/PaddleOCR/issues/18164)
- [ogkalu comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) ·
  [WORD-pytorch](https://github.com/hanish3464/WORD-pytorch)
- [Best open-source OCR 2026 (imagetotable)](https://imagetotable.ai/blog/best-open-source-ocr-tools-2026) ·
  [15-OCR benchmark on RTX 4070 (Medium)](https://adityamangal98.medium.com/the-ultimate-ocr-benchmark-15-ocr-systems-tested-on-my-rtx-4070-laptop-4a9f2c513349)
- [manga-ocr-mobile](https://huggingface.co/bluolightning/manga-ocr-mobile) ·
  [l0wgear/manga-ocr-2025-onnx](https://huggingface.co/l0wgear/manga-ocr-2025-onnx)

---

## Known-mistakes correction dictionary (2026-09-09) — implemented

Since no better model exists for English-scanlation comic OCR, the highest-leverage
fix is a **deterministic per-series correction dictionary**. A stylised bubble font
that PP-OCRv5 can't read produces the *same* wrong token every time — so a
`{"EHOH": "HOH", "SIHI": "THIS", "CTNOHS": "SHOULD"}` map fixes it for free,
instantly, with zero API cost or latency.

- `pipeline/ocr-corrections/_global.json` + `<series-slug>.json` (tracked)
- `data/ocr-corrections/<slug>.learned.jsonl` — auto-harvested from every GOT-OCR2
  garble recovery; a `from→to` pair seen 3× is auto-promoted.
- SFX list: OCR of an untranslated Korean sound effect (씨익 → "TOTN") is dropped
  when the line has real dialogue, kept when the panel IS just the SFX.
- Applied in `_merge_regions` (so a known token error no longer trips the garble
  detector → GOT-OCR2 is not called → free) and again on the final winning text.

**Nano Machine chapter 1 after the dictionary: dialogue ~97% clean, GOT-OCR2 calls
dropped from ~4/chapter to ~1/chapter.** Every entry verified against the panel
image ("EHOH" *looked* like "HOW" but the bubble says "HOH?"). See
`pipeline/ocr-corrections/README.md`.
