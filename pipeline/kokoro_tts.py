#!/usr/bin/env python3
"""kokoro_tts — synthesize narration with Kokoro-82M ONNX in an ISOLATED venv.

Why a separate process: kokoro-onnx (>=0.4) hard-requires ``numpy>=2``, which
collides with the main pipeline venv (torch/opencv-era ``numpy<2`` pin). So
Kokoro lives in its own venv — ``pipeline/.venv-kokoro`` built by
``setup_kokoro.sh`` — and ``master_pipeline.py`` shells out to this file, the
same pattern the Cloudflare fetcher (``mini-services/pipeline-service/cf_fetch.py``)
already uses. It is entirely OPT-IN: nothing here runs unless
``RECAP_TTS_ENGINE=kokoro`` is set, and any failure here makes the caller fall
straight through to its normal edge-tts → Piper → eSpeak cascade.

Benchmarked on the 4-vCPU Xeon 8272CL box (CPU only): RTF ≈ 0.23 with the
sentence-split path below (~4× real-time), init ≈ 0.4 s.

Request  (argv[1] or stdin, one JSON object):
  {"text": "...", "voice": "af_heart", "speed": 1.0, "lang": "en-us",
   "out_wav": "/abs/path.wav", "model_dir": "/abs/pipeline/models/kokoro",
   "sample_rate": 24000, "sentence_pause": 0.18}

Response (stdout, one JSON line):
  {"ok": true, "duration": 9.82, "sample_rate": 24000,
   "sentences": [{"text": "...", "start": 0.0, "end": 2.52}, ...]}
  {"ok": false, "error": "..."}

The WAV is written mono / 16-bit PCM at ``sample_rate`` (Kokoro's native
24 kHz — the caller resamples to the render rate with the same ffmpeg
fade pass it runs on edge-tts output). ``sentences`` carries the REAL
measured duration of each sentence so the caller can build per-word
timing (Kokoro's ``create()`` returns no word-level timestamps).

Exit 0 on success, 1 on any failure.
"""
import json
import os
import re
import sys
import wave


def _split_sentences(text: str):
    """Split on sentence-ending punctuation followed by whitespace, keeping
    every original word so ``" ".join(parts).split() == text.split()`` — the
    caller relies on that invariant to map sentence timing back onto words."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in (s.strip() for s in parts) if p]


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    try:
        req = json.loads(raw)
        text = (req["text"] or "").strip()
        out_wav = req["out_wav"]
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"bad request: {e}"}))
        return 1

    if not text:
        print(json.dumps({"ok": False, "error": "empty text"}))
        return 1

    voice = req.get("voice") or "af_heart"
    speed = float(req.get("speed") or 1.0)
    lang = req.get("lang") or "en-us"
    sr = int(req.get("sample_rate") or 24000)
    pause_s = float(req.get("sentence_pause") or 0.18)
    model_dir = req.get("model_dir") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "kokoro")

    model_name = os.environ.get("RECAP_KOKORO_MODEL", "kokoro-v1.0.onnx").strip() or "kokoro-v1.0.onnx"
    model_path = model_name if os.path.isabs(model_name) else os.path.join(model_dir, model_name)
    voices_path = os.path.join(model_dir, "voices-v1.0.bin")
    for p in (model_path, voices_path):
        if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
            print(json.dumps({"ok": False, "error": f"model asset missing: {p}"}))
            return 1

    # Keep BLAS from oversubscribing the shared box (master_pipeline caps its
    # own workers the same way).
    _n = os.environ.get("RECAP_CPU_THREADS") or os.environ.get("RECAP_ORT_THREADS")
    if _n:
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(_v, _n)

    try:
        import numpy as np
        from kokoro_onnx import Kokoro
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"kokoro-onnx not available: {e}"}))
        return 1

    try:
        kok = Kokoro(model_path, voices_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Kokoro init failed: {e}"}))
        return 1

    try:
        if voice not in set(kok.get_voices()):
            voice = "af_heart" if "af_heart" in set(kok.get_voices()) else kok.get_voices()[0]
    except Exception:
        pass

    sentences = _split_sentences(text) or [text]
    pause = np.zeros(max(0, int(sr * pause_s)), dtype=np.float32)
    chunks = []
    spans = []
    cursor = 0.0
    try:
        for i, s in enumerate(sentences):
            samples, got_sr = kok.create(s, voice=voice, speed=speed, lang=lang)
            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            if got_sr and int(got_sr) != sr:
                # linear-resample (numpy-only; caller also runs an ffmpeg pass)
                n_out = int(round(len(samples) * sr / float(got_sr)))
                if n_out > 1 and len(samples) > 1:
                    xp = np.linspace(0.0, 1.0, num=len(samples), dtype=np.float64)
                    x = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
                    samples = np.interp(x, xp, samples).astype(np.float32)
            dur = len(samples) / float(sr)
            spans.append({"text": s, "start": round(cursor, 3), "end": round(cursor + dur, 3)})
            chunks.append(samples)
            cursor += dur
            if i < len(sentences) - 1 and len(pause):
                chunks.append(pause)
                cursor += len(pause) / float(sr)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"synthesis failed: {e}"}))
        return 1

    if not chunks:
        print(json.dumps({"ok": False, "error": "no audio produced"}))
        return 1

    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"wav write failed: {e}"}))
        return 1

    print(json.dumps({
        "ok": True,
        "duration": round(len(pcm) / float(sr), 3),
        "sample_rate": sr,
        "voice": voice,
        "sentences": spans,
    }))
    return 0




def _synth_one(kok, np, req: dict) -> dict:
    text = (req.get("text") or "").strip()
    out_wav = req.get("out_wav") or ""
    if not text or not out_wav:
        return {"ok": False, "error": "empty text or out_wav"}

    voice = req.get("voice") or "af_heart"
    speed = float(req.get("speed") or 1.0)
    lang = req.get("lang") or "en-us"
    sr = int(req.get("sample_rate") or 24000)
    pause_s = float(req.get("sentence_pause") or 0.18)

    try:
        voices_set = set(kok.get_voices())
        if voice not in voices_set:
            voice = "af_heart" if "af_heart" in voices_set else next(iter(voices_set))
    except Exception:
        pass

    sentences = _split_sentences(text) or [text]
    pause = np.zeros(max(0, int(sr * pause_s)), dtype=np.float32)
    chunks = []
    spans = []
    cursor = 0.0
    try:
        for i, s in enumerate(sentences):
            samples, got_sr = kok.create(s, voice=voice, speed=speed, lang=lang)
            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            if got_sr and int(got_sr) != sr:
                n_out = int(round(len(samples) * sr / float(got_sr)))
                if n_out > 1 and len(samples) > 1:
                    xp = np.linspace(0.0, 1.0, num=len(samples), dtype=np.float64)
                    x = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
                    samples = np.interp(x, xp, samples).astype(np.float32)
            dur = len(samples) / float(sr)
            spans.append({"text": s, "start": round(cursor, 3), "end": round(cursor + dur, 3)})
            chunks.append(samples)
            cursor += dur
            if i < len(sentences) - 1 and len(pause):
                chunks.append(pause)
                cursor += len(pause) / float(sr)
    except Exception as e:
        return {"ok": False, "error": f"synthesis failed: {e}"}

    if not chunks:
        return {"ok": False, "error": "no audio produced"}

    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    fade_in = float(req.get("fade_in") or 0.0)
    fade_out = float(req.get("fade_out") or 0.0)
    if audio.size > 1 and (fade_in > 0.0 or fade_out > 0.0):
        n_in = min(audio.size, max(0, int(round(fade_in * sr))))
        if n_in > 1:
            audio[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        n_out = min(audio.size, max(0, int(round(fade_out * sr))))
        if n_out > 1:
            audio[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
    except Exception as e:
        return {"ok": False, "error": f"wav write failed: {e}"}

    return {
        "ok": True,
        "duration": round(len(pcm) / float(sr), 3),
        "sample_rate": sr,
        "voice": voice,
        "sentences": spans,
    }


def _run_worker(model_dir: str) -> int:
    model_name = os.environ.get("RECAP_KOKORO_MODEL", "kokoro-v1.0.onnx").strip() or "kokoro-v1.0.onnx"
    model_path = model_name if os.path.isabs(model_name) else os.path.join(model_dir, model_name)
    voices_path = os.path.join(model_dir, "voices-v1.0.bin")
    for p in (model_path, voices_path):
        if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
            print(json.dumps({"ok": False, "error": f"model asset missing: {p}"}), flush=True)
            return 1

    _n = os.environ.get("RECAP_CPU_THREADS") or os.environ.get("RECAP_ORT_THREADS")
    if _n:
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(_v, _n)

    try:
        import numpy as np
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        so = ort.SessionOptions()
        ort_threads = int(os.environ.get("RECAP_KOKORO_ORT_THREADS", "1") or 1)
        so.intra_op_num_threads = max(1, ort_threads)
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        kok = Kokoro.from_session(sess, voices_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Kokoro worker init failed: {e}"}), flush=True)
        return 1

    import gc
    _reqs = 0
    _libc = None
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6")
    except Exception:
        pass
    print(json.dumps({"ok": True, "ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = _synth_one(kok, np, req)
        except Exception as e:
            res = {"ok": False, "error": f"worker error: {e}"}
        print(json.dumps(res), flush=True)
        _reqs += 1
        if _reqs % 100 == 0:
            gc.collect()
            if _libc is not None:
                try:
                    _libc.malloc_trim(0)
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        mdir = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "kokoro")
        sys.exit(_run_worker(mdir))
    sys.exit(main())
