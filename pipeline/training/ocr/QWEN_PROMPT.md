# Qwen panel-transcription task — OCR ground truth

You will receive a ZIP of manhwa/webtoon panel images (`batch_XX.zip`, ~180 images,
names like `mounthua_chap_003_frame_00017.jpg`). For **every** image, transcribe the
text so we can use `(image, your transcription)` pairs to fine-tune an OCR model.

## Output format — one line per image, nothing else

```
<image filename> | <transcription>
```

- Keep the images in the same order as the zip. Do not skip any. If you can only do
  part of the batch, stop at a clean point and tell me the last filename done.
- Put the whole transcription for one image on **one line**. Join separate
  bubbles / caption boxes in natural reading order with `  ///  ` (space-slash-slash-slash-space).
- Reading order: top-to-bottom, left-to-right. Narration/caption boxes before speech
  bubbles when they sit above them.

## What to transcribe

- All **dialogue**, **narration/caption boxes**, **thought bubbles**, and readable
  **in-scene signs / titles / labels**.
- Transcribe **verbatim**: exact words, exact spelling, exact punctuation, and keep
  the case as drawn (most is ALL CAPS — write it ALL CAPS).
- Keep line-internal hyphenation only if it's a real hyphenated word. Drop a hyphen
  that's just a bubble line-break ("TOMOR-\nROW" -> "TOMORROW").

## What to mark or skip

- **SFX / onomatopoeia** (KRRK, BOOM, THUD, screams like AAAARGH): wrap as
  `[SFX: KRRK]`. If a panel is only SFX, the line is just `[SFX: ...]`.
- **Scanlation credit / watermark / URL / page number / "read on ___" / translator
  names**: do **not** transcribe. Skip them entirely.
- A bubble that is present but genuinely unreadable (too small/blurred/cut): `[UNREADABLE]`
  in place of that bubble's text.
- **No text at all** in the panel: the whole line is `[NONE]`.
- Do **not** guess, autocorrect, "fix", or translate anything. If the art shows a
  made-up world term or a name, write exactly what you see.

## Example output

```
mounthua_chap_002_frame_00004.jpg | THE TEN GREAT SECTS RULED THE MURIM.  ///  BUT THAT WAS A LONG TIME AGO.
mounthua_chap_002_frame_00005.jpg | [NONE]
mounthua_chap_002_frame_00006.jpg | CHUNG MYUNG!  ///  ARE YOU LISTENING TO ME?  ///  [SFX: GRAB]
mounthua_chap_002_frame_00007.jpg | [SFX: KWAAANG]
mounthua_chap_002_frame_00009.jpg | "PLUM BLOSSOM SWORD"  ///  I NEVER THOUGHT I'D SEE IT AGAIN.  ///  [UNREADABLE]
```

Return the result as plain text (in a code block is fine). Thank you.
