#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

CRASH_CODE=77
STAGES=['OCR','TTS','AUDIO_ASSEMBLY','VIDEO_RENDER','MERGE']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--work-dir', required=True); ap.add_argument('--job-id', default='crash'); ap.add_argument('--stage', choices=STAGES, required=True)
    a=ap.parse_args(); script=Path(__file__).with_name('production_canary.py')
    first=subprocess.run([sys.executable,str(script),'--work-dir',a.work_dir,'--job-id',a.job_id,'--crash-at',a.stage])
    if first.returncode != CRASH_CODE:
        print(f'expected crash code {CRASH_CODE}, got {first.returncode}', file=sys.stderr); return 1
    second=subprocess.run([sys.executable,str(script),'--work-dir',a.work_dir,'--job-id',a.job_id])
    return second.returncode
if __name__ == '__main__': raise SystemExit(main())
