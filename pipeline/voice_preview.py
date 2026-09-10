#!/usr/bin/env python3
"""
voice_preview.py — Generate a short voice preview sample.

edge-tts voice ids (e.g. en-US-AndrewNeural) use edge-tts.
Kokoro voice ids (e.g. am_michael, af_bella) use the local Kokoro-82M engine
in its isolated venv (pipeline/.venv-kokoro, see setup_kokoro.sh) — same
engine the pipeline uses when a job picks a Kokoro voice.

Used by the Next.js /api/voice-preview route (via pipeline-service) to let
users hear a narration voice before starting the pipeline.

Usage:
    python voice_preview.py --voice en-US-AndrewNeural --output /path/preview.mp3
    python voice_preview.py --voice am_michael --output /path/preview.mp3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile

SAMPLE_TEXT = (
    "Hello! I'll be narrating your manhwa recap video. "
    "This is a quick preview of how my voice sounds. "
    "Let's dive into the story!"
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_KOKORO_RE = re.compile(r"^[abhijpef][fm]_[a-z][a-z_]*$")
_KOKORO_PY = os.path.join(_HERE, ".venv-kokoro", "bin", "python3")
_KOKORO_SCRIPT = os.path.join(_HERE, "kokoro_tts.py")
_KOKORO_MODELS = os.path.join(_HERE, "models", "kokoro")


async def _generate_edge(voice: str, text: str, output: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice).save(output)


def _generate_kokoro(voice: str, text: str, output: str) -> None:
    py = os.environ.get("RECAP_KOKORO_PYTHON") or _KOKORO_PY
    if not os.path.exists(py):
        raise RuntimeError("Kokoro venv not found (run pipeline/setup_kokoro.sh)")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav = tf.name
    try:
        req = {"text": text, "voice": voice, "out_wav": wav,
               "model_dir": _KOKORO_MODELS, "sample_rate": 24000}
        p = subprocess.run([py, _KOKORO_SCRIPT], input=json.dumps(req), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        payload = None
        for line in reversed((p.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    break
                except Exception:
                    pass
        if not payload or not payload.get("ok") or not os.path.exists(wav):
            raise RuntimeError((payload or {}).get("error") or (p.stderr or "")[-200:])
        # 24k mono wav -> mp3
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav,
                        "-b:a", "96k", output], check=True, timeout=30)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a voice preview MP3")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default=SAMPLE_TEXT)
    args = parser.parse_args()

    try:
        if _KOKORO_RE.match(args.voice):
            _generate_kokoro(args.voice, args.text, args.output)
        else:
            asyncio.run(_generate_edge(args.voice, args.text, args.output))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not os.path.exists(args.output) or os.path.getsize(args.output) < 100:
        print("ERROR: produced an empty or missing file", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
