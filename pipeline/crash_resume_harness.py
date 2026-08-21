#!/usr/bin/env python3
"""Controlled crash/resume reconciliation harness.

Use --crash-at to leave a RUNNING/partial artifact, then rerun without
--crash-at using the same --job-id and --work-dir to verify reconciliation.
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path
try:
    from pipeline.production import QAResult, SQLiteStateStore, Stage, State, atomic_promote, checksum_file, reconcile_artifact
except ModuleNotFoundError:
    from production import QAResult, SQLiteStateStore, Stage, State, atomic_promote, checksum_file, reconcile_artifact  # type: ignore

STAGES = [Stage.OCR, Stage.TTS, Stage.AUDIO_ASSEMBLY, Stage.VIDEO_RENDER, Stage.MERGE]


def qa_non_partial(path: Path) -> QAResult:
    data = path.read_bytes() if path.exists() else b''
    return QAResult(ok=path.exists() and b'partial' not in data and len(data) > 0, reason='partial/corrupt artifact')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--job-id', default='crash-harness')
    ap.add_argument('--crash-at', choices=[s.value for s in STAGES])
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    st = SQLiteStateStore(args.work_dir / 'pipeline_state.sqlite')

    for stage in STAGES:
        artifact = args.work_dir / f'{stage.value.lower()}.artifact'
        if args.crash_at == stage.value:
            artifact.write_bytes(b'partial')
            st.record(args.job_id, stage, State.RUNNING, artifact_path=artifact, artifact_checksum=checksum_file(artifact))
            os._exit(99)
        if reconcile_artifact(st, args.job_id, stage, artifact, qa_non_partial):
            continue
        tmp = artifact.with_suffix('.tmp')
        tmp.write_bytes(f'complete:{stage.value}'.encode())
        digest = atomic_promote(tmp, artifact)
        st.record(args.job_id, stage, State.COMPLETE, artifact_path=artifact, artifact_checksum=digest)
    print('crash/resume harness complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
