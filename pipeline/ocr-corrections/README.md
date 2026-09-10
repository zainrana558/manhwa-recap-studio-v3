# OCR correction dictionary — self-learning

Deterministic OCR-error correction that **improves itself** as more chapters run.

## Layers (applied in this order, per panel)

1. **Whole-line exemplar** — if this exact garbled line was resolved before, reuse the fix.
2. **Curated phrases** (`*.json` → `phrases`) — case-insensitive multi-word fixes.
3. **Curated + promoted tokens** — word-boundary, case-preserving single-word swaps.
4. **Fuzzy repair** — for a *novel* non-dictionary token, apply this series' **learned
   character confusions** (e.g. "this font's `W` scans as `V`"); accept **only** if
   exactly one result is a real word. No guessing.
5. **SFX drop** — OCR of an untranslated sound effect (씨익 → "TOTN") is dropped when
   the line has real dialogue; kept when the panel *is* the SFX.

## Files

| location | file | tracked? | contents |
|---|---|---|---|
| `pipeline/ocr-corrections/` | `_global.json`, `<slug>.json` | ✅ git | hand-curated, verified against panels |
| `data/ocr-corrections/` | `<slug>.learned.jsonl` | ✗ | raw learned candidates (one JSON/line) |
| `data/ocr-corrections/` | `<slug>.exemplars.jsonl` | ✗ | whole-line garble → fix |
| `data/ocr-corrections/` | `<slug>.review.md` | ✗ | auto-written: near-promotion + conflicting entries for you to eyeball |

`<slug>` = manga title, lowercased, non-alphanumerics → `-`.

## How the learning works

Every time a stronger reader beats the classical OCR on a panel, the difference is
recorded with an **evidence weight**:

| source | weight | why |
|---|---|---|
| GOT-OCR2 / cloud VLM re-read | 1.0 | read the actual pixels |
| cross-panel consistency | 0.7 | |
| tighter-params retry (same model) | 0.55 | weaker signal |
| PaddleOCR preprocessing variant | 0.45 | |

**Promotion** (candidate → auto-applied): weighted count ≥ **3.0**, that `to` owns
≥ **62 %** of the `from`'s total weight, and `to` is a real English word. A `from`
with two competing `to`s that split the vote is **not** promoted — it goes to the
review file instead.

Character confusions are mined only from pairs whose letters already mostly line up
(a real per-glyph mis-read), never from a total scramble.

## `*.json` format

```json
{
  "phrases": [ ["ID ON'T", "I DON'T"] ],
  "tokens":  { "EHOH": "HOH", "SIHI": "THIS" },   // "" = delete the token
  "sfx":     ["KEUK", "TOTN"]
}
```

## Adding an entry by hand

1. Find a recurring mis-read; **verify against the panel image** (`EHOH` looked like
   "HOW" but the bubble says "HOH?").
2. Add to `tokens` (one word) or `phrases` (multi-word / word-order).
3. No restart — the service reloads on file mtime change. Curated entries always
   win over learned ones.
