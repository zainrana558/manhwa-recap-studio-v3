#!/usr/bin/env python3
"""Bounded production canary for local Oracle-style deployments.

This command deliberately fails with explicit dependency messages when required
local binaries/services are unavailable. It never fabricates success.
"""
from __future__ import annotations

import argparse, base64, json, shutil, subprocess, sys, tempfile
from pathlib import Path


try:
    from pipeline.production import QAResult, SQLiteStateStore, Stage, State, atomic_promote, checksum_file
    from pipeline.master_pipeline import audio_qa_result, video_qa_result, _synthesize_with_piper, _synthesize_with_espeak
except ModuleNotFoundError:
    from production import QAResult, SQLiteStateStore, Stage, State, atomic_promote, checksum_file  # type: ignore
    from master_pipeline import audio_qa_result, video_qa_result, _synthesize_with_piper, _synthesize_with_espeak  # type: ignore


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"UNVERIFIED — missing required binary: {name}")
    return path


def _make_panel(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("UNVERIFIED — missing Python dependency: Pillow") from exc
    img = Image.new('RGB', (640, 360), 'white')
    d = ImageDraw.Draw(img)
    d.rectangle((40, 40, 600, 320), outline='black', width=4)
    d.text((80, 140), 'CANARY PANEL: the hero wakes up.', fill='black')
    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--job-id', default='production-canary')
    ap.add_argument('--ocr-url', default='http://localhost:3002/ocr/base64')
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    state = SQLiteStateStore(args.work_dir / 'pipeline_state.sqlite')

    try:
        _require_binary('ffmpeg')
        _require_binary('ffprobe')
        if not (shutil.which('piper') and __import__('os').environ.get('PIPER_VOICE_MODEL')) and not (shutil.which('espeak-ng') or shutil.which('espeak')):
            raise RuntimeError('UNVERIFIED — no local TTS provider available: need Piper+PIPER_VOICE_MODEL or eSpeak/eSpeak NG')

        panel = args.work_dir / 'canary_panel.png'
        _make_panel(panel)
        state.record(args.job_id, Stage.PANEL, State.COMPLETE, artifact_path=panel, artifact_checksum=checksum_file(panel))

        # OCR through configured local service.
        state.record(args.job_id, Stage.OCR, State.RUNNING)
        b64 = base64.b64encode(panel.read_bytes()).decode('ascii')
        proc = _run(['python', '-c', "import json,sys,urllib.request; data=json.dumps({'image':sys.argv[2]}).encode(); req=urllib.request.Request(sys.argv[1], data=data, headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req, timeout=30).read().decode())", args.ocr_url, b64], timeout=40)
        if proc.returncode != 0:
            raise RuntimeError(f'UNVERIFIED — OCR request failed: {proc.stderr[-200:]}')
        ocr = json.loads(proc.stdout)
        if ocr.get('status') != 'SUCCESS':
            state.record(args.job_id, Stage.OCR, State.UNCERTAIN, error_message=json.dumps(ocr)[:500])
            raise RuntimeError(f"UNVERIFIED — OCR did not return SUCCESS: {ocr.get('status')}")
        state.record(args.job_id, Stage.OCR, State.COMPLETE, provider='paddleocr', model=ocr.get('model'), metadata=ocr)

        narration = (ocr.get('text') or 'Canary narration for production audio.').strip()
        state.record(args.job_id, Stage.NARRATION, State.COMPLETE, metadata={'text': narration})

        # TTS temporary artifact -> QA -> promotion.
        tmp_wav = args.work_dir / 'canary.tmp.wav'
        final_wav = args.work_dir / 'canary.wav'
        tmp_wav.unlink(missing_ok=True)
        final_wav.unlink(missing_ok=True)
        state.record(args.job_id, Stage.TTS, State.RUNNING)
        if shutil.which('piper') and __import__('os').environ.get('PIPER_VOICE_MODEL'):
            _synthesize_with_piper(narration, tmp_wav)
        else:
            _synthesize_with_espeak(narration, tmp_wav)
        aq = audio_qa_result(tmp_wav, allow_silence=False)
        if not aq.ok:
            raise RuntimeError(f'UNVERIFIED — TTS audio QA failed: {aq.reason}')
        digest = atomic_promote(tmp_wav, final_wav)
        state.record(args.job_id, Stage.TTS, State.COMPLETE, artifact_path=final_wav, artifact_checksum=digest, duration=aq.duration)

        # Small FFmpeg render -> video QA -> final promotion.
        tmp_mp4 = args.work_dir / 'canary.tmp.mp4'
        final_mp4 = args.work_dir / 'canary.mp4'
        tmp_mp4.unlink(missing_ok=True)
        final_mp4.unlink(missing_ok=True)
        state.record(args.job_id, Stage.VIDEO_RENDER, State.RUNNING)
        proc = _run(['ffmpeg', '-y', '-loop', '1', '-i', str(panel), '-i', str(final_wav), '-t', f'{max(aq.duration, 1.0):.3f}', '-vf', 'scale=1280:720', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(tmp_mp4)], timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f'UNVERIFIED — ffmpeg render failed: {proc.stderr[-300:]}')
        vq = video_qa_result(tmp_mp4, expect_audio=True, expected_duration=max(aq.duration, 1.0))
        if not vq.ok:
            raise RuntimeError(f'UNVERIFIED — video QA failed: {vq.reason}')
        vd = atomic_promote(tmp_mp4, final_mp4)
        state.record(args.job_id, Stage.FINAL_QA, State.COMPLETE, artifact_path=final_mp4, artifact_checksum=vd, duration=vq.duration)
        print(json.dumps({'ok': True, 'video': str(final_mp4), 'checksum': vd}, indent=2))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
