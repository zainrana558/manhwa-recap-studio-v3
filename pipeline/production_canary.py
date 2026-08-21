#!/usr/bin/env python3
"""Small real-production canary for local OCR/TTS/FFmpeg/state/artifact wiring."""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path
from production import ArtifactStore, ResourceGuard, SQLiteStateStore, Stage, State, RetryCategory, audio_qa, video_qa, checksum_file

CRASH_CODE = 77

def run(cmd, input_text=None, timeout=120):
    return subprocess.run(cmd, input=input_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)

def synthesize(text: str, out: Path):
    piper = shutil.which('piper'); model = os.environ.get('PIPER_VOICE_MODEL') or os.environ.get('PIPER_VOICE')
    failures=[]
    for attempt in range(2):
        if piper and model:
            r=run([piper,'--model',model,'--output_file',str(out)], input_text=text, timeout=120)
            if r.returncode == 0 and audio_qa(out, True).ok: return 'piper'
            failures.append(f'piper:{r.stderr[-160:]}')
    espeak = shutil.which('espeak-ng') or shutil.which('espeak')
    if espeak:
        r=run([espeak,'-w',str(out),text], timeout=120)
        if r.returncode == 0 and audio_qa(out, True).ok: return Path(espeak).name
        failures.append(f'espeak:{r.stderr[-160:]}')
    raise RuntimeError('local TTS failed; no silent substitution: '+' | '.join(failures))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--work-dir', required=True); ap.add_argument('--job-id', default='canary'); ap.add_argument('--crash-at', choices=[s.value for s in Stage], default=None)
    args=ap.parse_args(); wd=Path(args.work_dir); wd.mkdir(parents=True, exist_ok=True)
    st=SQLiteStateStore(wd/'state.sqlite'); store=ArtifactStore(wd); guard=ResourceGuard(state=st)
    def mark(stage):
        if not guard.check(wd,args.job_id,stage).ok: raise RuntimeError('resource guard failed')
        st.record(args.job_id, stage, State.RUNNING)
        if args.crash_at == stage.value: raise SystemExit(CRASH_CODE)
    try:
        mark(Stage.JOB); mark(Stage.CHAPTER); mark(Stage.PANEL)
        mark(Stage.OCR)
        text='This is a production canary narration.'
        st.record(args.job_id, Stage.OCR, State.COMPLETE, provider='canary', model='synthetic-panel', metadata={'status':'SUCCESS','text':text,'quality_score':1.0,'candidates':[{'text':text,'score':1.0}],'selection_reason':'canary_static_text'})
        mark(Stage.NARRATION); st.record(args.job_id, Stage.NARRATION, State.COMPLETE, metadata={'text': text})
        mark(Stage.TTS); tmp_audio=store.temporary_path('canary.wav'); provider=synthesize(text,tmp_audio); final_audio,digest,qa=store.promote(tmp_audio,'canary.wav',lambda p: audio_qa(p, True)); st.record(args.job_id, Stage.TTS, State.COMPLETE, provider=provider, artifact_path=final_audio, artifact_checksum=digest, duration=qa.duration)
        mark(Stage.AUDIO_ASSEMBLY); st.record(args.job_id, Stage.AUDIO_ASSEMBLY, State.COMPLETE, artifact_path=final_audio, artifact_checksum=digest, duration=qa.duration)
        mark(Stage.VIDEO_RENDER); tmp_video=store.temporary_path('canary.mp4')
        r=run(['ffmpeg','-y','-f','lavfi','-i','color=c=black:s=640x360:r=24','-i',str(final_audio),'-shortest','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(tmp_video)], timeout=120)
        if r.returncode != 0: raise RuntimeError('ffmpeg render failed: '+r.stderr[-400:])
        final_video,vdig,vqa=store.promote(tmp_video,'canary.mp4',lambda p: video_qa(p, True, qa.duration, 1.5)); st.record(args.job_id, Stage.VIDEO_RENDER, State.COMPLETE, artifact_path=final_video, artifact_checksum=vdig, duration=vqa.duration)
        mark(Stage.MERGE); st.record(args.job_id, Stage.MERGE, State.COMPLETE, artifact_path=final_video, artifact_checksum=vdig, duration=vqa.duration)
        mark(Stage.FINAL_QA); fq=video_qa(final_video, True); st.record(args.job_id, Stage.FINAL_QA, State.COMPLETE, artifact_path=final_video, artifact_checksum=checksum_file(final_video), duration=fq.duration)
        print(f'PASS {final_video} duration={fq.duration:.3f} checksum={checksum_file(final_video)}')
        return 0
    except SystemExit: raise
    except Exception as exc:
        st.record(args.job_id, Stage.JOB, State.RETRYABLE, error_code='CANARY_FAILED', error_message=str(exc), retry_category=RetryCategory.TRANSIENT)
        print(f'FAIL {exc}', file=sys.stderr); return 1
if __name__ == '__main__': raise SystemExit(main())
