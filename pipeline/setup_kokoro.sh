#!/usr/bin/env bash
#
# setup_kokoro.sh — install the OPT-IN Kokoro-82M neural TTS engine.
#
# Kokoro-onnx hard-requires numpy>=2, which collides with the main pipeline
# venv (numpy<2). So it gets its own isolated venv here, and
# pipeline/kokoro_tts.py is shelled out to from master_pipeline.py — exactly
# how mini-services/pipeline-service/cf_fetch.py is handled.
#
# After running this, enable it with:
#     RECAP_TTS_ENGINE=kokoro
#     RECAP_KOKORO_VOICE=af_heart        # optional, 54 voices — see get_voices()
#
# Everything is graceful: if this venv or the model files are missing, the
# pipeline silently falls back to edge-tts → Piper → eSpeak as before.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-kokoro"
MODEL_DIR="$HERE/models/kokoro"
PYBIN="${KOKORO_PYTHON_BASE:-python3}"

REL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_ONNX="$MODEL_DIR/kokoro-v1.0.onnx"
VOICES_BIN="$MODEL_DIR/voices-v1.0.bin"

echo "[kokoro] venv:   $VENV"
echo "[kokoro] models: $MODEL_DIR"

if [[ ! -x "$VENV/bin/python3" ]]; then
    echo "[kokoro] creating isolated venv ..."
    "$PYBIN" -m venv "$VENV" || { echo "[kokoro] venv creation failed"; exit 1; }
fi

"$VENV/bin/pip" install -q --disable-pip-version-check --upgrade pip >/dev/null 2>&1 || true
echo "[kokoro] installing kokoro-onnx (pulls numpy>=2 into THIS venv only) ..."
"$VENV/bin/pip" install -q --disable-pip-version-check "kokoro-onnx>=0.4,<1.0" \
    || { echo "[kokoro] pip install kokoro-onnx failed"; exit 1; }

mkdir -p "$MODEL_DIR"
fetch() {  # url dest min_bytes
    local url="$1" dest="$2" min="$3"
    if [[ -f "$dest" && "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -ge "$min" ]]; then
        echo "[kokoro] have $(basename "$dest")"
        return 0
    fi
    echo "[kokoro] downloading $(basename "$dest") ..."
    curl -fSL --retry 3 -o "$dest" "$url" || { echo "[kokoro] download failed: $url"; return 1; }
}
fetch "$REL/kokoro-v1.0.onnx"  "$MODEL_ONNX" 200000000 || exit 1
fetch "$REL/voices-v1.0.bin"   "$VOICES_BIN"  20000000 || exit 1

echo "[kokoro] smoke test ..."
"$VENV/bin/python3" "$HERE/kokoro_tts.py" "$(cat <<JSON
{"text":"Kokoro text to speech is ready.","voice":"af_heart",
 "out_wav":"$MODEL_DIR/_smoke.wav","model_dir":"$MODEL_DIR"}
JSON
)" && echo "[kokoro] OK — set RECAP_TTS_ENGINE=kokoro to use it" || {
    echo "[kokoro] smoke test FAILED"; exit 1; }
rm -f "$MODEL_DIR/_smoke.wav"
