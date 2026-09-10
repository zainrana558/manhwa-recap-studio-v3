#!/bin/bash
# fetch_models.sh — download every model the pipeline + OCR service use.
#
# Single source of truth for model provisioning. Called by setup.sh and by
# start.sh's dependency bootstrap; also safe to run by hand any time
# (idempotent — each model is skipped if already present and non-empty).
#
# "Degrade, don't crash": every fetch is best-effort. A failed/skipped
# download leaves the pipeline on its documented fallback (pixel-only masks,
# flood-fill panels, eSpeak TTS, stock RapidOCR v6) and logs a warning — it
# never aborts setup or startup. Exit status is always 0 for the required
# tier; only an explicitly-requested optional model failing is surfaced.
#
# Required tier (always fetched):
#   - comic-text-and-bubble-detector  (RT-DETR-v2, gutter-safe text/caption mask)
#   - manga-panel-yolo                (YOLO26n, reference-style panel split)
#   - comic-bubble                    (speech-bubble no-cut zones)
#   - anime-face                      (face no-cut zones)
#   - piper voice                     (production TTS; binary handled by setup.sh)
#   - RapidOCR PP-OCRv5 mobile        (primary OCR recognition)
#
# Optional tier (env flag = 1):
#   FETCH_KOKORO=1     Kokoro-82M neural TTS (~340MB)  -> pipeline/models/kokoro
#   FETCH_GOT_OCR=1    GOT-OCR2.0 local VLM OCR (~1.4GB, HF cache)
#   FETCH_SMOLVLM=1    SmolVLM2-500M local panel captioner (~1GB, HF cache)

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"
MODELS="$HERE/models"
VENV_PY="$PROJECT_DIR/.venv/bin/python3"
[ -x "$VENV_PY" ] || VENV_PY="python3"
mkdir -p "$MODELS"

ok(){   echo "  ✅ $1"; }
warn(){ echo "  ⚠️  $1"; }
step(){ echo "── $1"; }

# curl one file to a path; verify non-empty; clean up on failure.
dl(){ # url  dest  [min_bytes]
  local url="$1" dest="$2" min="${3:-1000}"
  mkdir -p "$(dirname "$dest")"
  if curl -fsSL -m 900 -o "$dest.part" "$url"; then
    local sz; sz=$(stat -c%s "$dest.part" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$min" ]; then mv "$dest.part" "$dest"; return 0; fi
    warn "downloaded $(basename "$dest") too small ($sz B)"
  fi
  rm -f "$dest.part"; return 1
}

# ── 1. comic text/bubble detector (ogkalu/comic-text-and-bubble-detector) ──
# RT-DETR-v2. Needs the HF repo structure (config + preprocessor + weights)
# for the torch fallback path, plus the ~11MB INT8 ONNX for the fast path.
TB="$MODELS/comic-text-and-bubble-detector"
if [ -s "$TB/detector-v4-s_int8.onnx" ] && [ -s "$TB/config.json" ]; then
  ok "comic text/bubble detector present"
else
  step "comic text/bubble detector (RT-DETR-v2, ~170MB)"
  mkdir -p "$TB"
  set +e
  OUT=$("$VENV_PY" - <<PY 2>&1
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ogkalu/comic-text-and-bubble-detector", local_dir="$TB",
                  ignore_patterns=["detector.onnx", "detector_int8.onnx"])
PY
)
  RC=$?
  set -e 2>/dev/null || true
  if [ $RC -eq 0 ] && [ -s "$TB/detector-v4-s_int8.onnx" ]; then
    ok "comic text/bubble detector downloaded"
  else
    warn "comic text/bubble detector failed — pixel-only caption mask fallback"
    echo "$OUT" | tail -3
    rm -rf "$TB"
  fi
fi

# ── 2. manga panel detector (YOLO26n ONNX) ──
PY_DIR="$MODELS/manga-panel-yolo"
if [ -s "$PY_DIR/manga_panel_detector_fp32_1024.onnx" ]; then
  ok "manga panel detector present"
else
  step "manga panel detector (YOLO26n ONNX, ~12MB)"
  dl "https://huggingface.co/mednasserallah/manga-panel-detector-yolo26n-onnx/resolve/main/manga_panel_detector_fp32_1024.onnx" \
     "$PY_DIR/manga_panel_detector_fp32_1024.onnx" 1000000 \
     && ok "manga panel detector downloaded" \
     || warn "manga panel detector failed — flood-fill panel detection fallback"
fi

# ── 3. comic speech-bubble detector (no-cut zones) ──
CB="$MODELS/comic-bubble/comic-speech-bubble-detector.pt"
if [ -s "$CB" ]; then
  ok "comic speech-bubble detector present"
else
  step "comic speech-bubble detector (YOLOv8m, ~50MB)"
  dl "https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m/resolve/main/comic-speech-bubble-detector.pt" \
     "$CB" 10000000 \
     && ok "comic speech-bubble detector downloaded" \
     || warn "comic speech-bubble detector failed — bubble no-cut zones skipped"
fi

# ── 4. anime-face detector (no-cut zones) ──
AF="$MODELS/anime-face/face_v1.4_n.onnx"
if [ -s "$AF" ]; then
  ok "anime-face detector present"
else
  step "anime-face detector (deepghs anime_face_detection v1.4_n, ~12MB)"
  if dl "https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_n/model.onnx" "$AF" 1000000; then
    dl "https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_n/labels.json" \
       "$MODELS/anime-face/labels.json" 1 || echo '["face"]' > "$MODELS/anime-face/labels.json"
    ok "anime-face detector downloaded"
  else
    warn "anime-face detector failed — face no-cut zones skipped"
  fi
fi

# ── 5. Piper voice model (production TTS) ──
# The piper BINARY is installed by setup.sh / start.sh (GitHub release tarball);
# here we only fetch the ONNX voice + its json.
PV_DIR="$PROJECT_DIR/pipeline/voices"
PV_NAME="${PIPER_VOICE_NAME:-en_US-ryan-high}"
PV="$PV_DIR/${PV_NAME}.onnx"
if [ -s "$PV" ] && [ -s "$PV.json" ]; then
  ok "piper voice present ($PV_NAME)"
else
  step "piper voice model ($PV_NAME)"
  B="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
  if dl "$B/${PV_NAME}.onnx" "$PV" 1000000 && dl "$B/${PV_NAME}.onnx.json" "$PV.json" 1; then
    ok "piper voice downloaded"
  else
    warn "piper voice failed — eSpeak-NG TTS fallback at render time"
    rm -f "$PV" "$PV.json"
  fi
fi

# ── 6. RapidOCR PP-OCRv5 mobile (primary OCR recognition) ──
RO_DIR="$("$VENV_PY" - <<'PY' 2>/dev/null || true
import os, importlib.util
s = importlib.util.find_spec("rapidocr")
if s and s.submodule_search_locations:
    print(os.path.join(list(s.submodule_search_locations)[0], "models"))
PY
)"
if [ -n "$RO_DIR" ] && [ -d "$RO_DIR" ]; then
  if [ -s "$RO_DIR/en_PP-OCRv5_rec_mobile.onnx" ]; then
    ok "RapidOCR PP-OCRv5 models present"
  else
    step "RapidOCR PP-OCRv5 mobile models (det + EN rec, ~13MB)"
    RB="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5"
    if dl "$RB/det/ch_PP-OCRv5_det_mobile.onnx" "$RO_DIR/ch_PP-OCRv5_det_mobile.onnx" 1000000 \
       && dl "$RB/rec/en_PP-OCRv5_rec_mobile.onnx" "$RO_DIR/en_PP-OCRv5_rec_mobile.onnx" 1000000; then
      ok "RapidOCR PP-OCRv5 models pre-fetched"
    else
      warn "PP-OCRv5 pre-fetch failed — RapidOCR retries at init / falls back to stock v6"
      rm -f "$RO_DIR/ch_PP-OCRv5_det_mobile.onnx" "$RO_DIR/en_PP-OCRv5_rec_mobile.onnx"
    fi
  fi
else
  warn "rapidocr not importable in the venv yet — its models auto-download on first OCR call"
fi

# ── 7. OPTIONAL: Kokoro-82M neural TTS ──
if [ "${FETCH_KOKORO:-0}" = "1" ]; then
  K="$MODELS/kokoro"
  if [ -s "$K/kokoro-v1.0.onnx" ] && [ -s "$K/voices-v1.0.bin" ]; then
    ok "kokoro model present"
  else
    step "kokoro-82M TTS (~340MB)"
    R="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    if dl "$R/kokoro-v1.0.onnx" "$K/kokoro-v1.0.onnx" 200000000 \
       && dl "$R/voices-v1.0.bin" "$K/voices-v1.0.bin" 20000000; then
      ok "kokoro model downloaded"
    else
      warn "kokoro download failed — set RECAP_TTS_ENGINE!=kokoro or retry"
    fi
  fi
fi

# ── 8. OPTIONAL: GOT-OCR2.0 local VLM OCR tier (HF cache) ──
if [ "${FETCH_GOT_OCR:-0}" = "1" ]; then
  step "GOT-OCR2.0 local VLM (~1.4GB, into HF cache)"
  "$VENV_PY" - <<'PY' && ok "GOT-OCR2.0 cached" || warn "GOT-OCR2.0 prefetch failed — OCR_LOCAL_VLM tier fetches on first use"
try:
    from huggingface_hub import snapshot_download
    snapshot_download("stepfun-ai/GOT-OCR-2.0-hf")
except Exception as e:
    raise SystemExit(str(e))
PY
fi

# ── 9. OPTIONAL: SmolVLM2-500M local panel captioner (HF cache) ──
if [ "${FETCH_SMOLVLM:-0}" = "1" ]; then
  step "SmolVLM2-500M captioner (~1GB, into HF cache)"
  "$VENV_PY" - <<'PY' && ok "SmolVLM2 cached" || warn "SmolVLM2 prefetch failed — fetches on first use"
try:
    from huggingface_hub import snapshot_download
    snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
except Exception as e:
    raise SystemExit(str(e))
PY
fi

echo "── model provisioning done"
exit 0
