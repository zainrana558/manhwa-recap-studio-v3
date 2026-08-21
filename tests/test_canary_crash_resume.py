from pathlib import Path
from pipeline.production import SQLiteStateStore, Stage, State, QAResult, atomic_promote, checksum_file, reconcile_artifact


def test_synthetic_canary_state_and_final_promotion(tmp_path: Path):
    """Synthetic canary: representative stages, checksums, final promotion."""
    st = SQLiteStateStore(tmp_path / 'canary.sqlite')
    for stage in (Stage.JOB, Stage.CHAPTER, Stage.PANEL, Stage.OCR, Stage.NARRATION):
        st.record('canary', stage, State.COMPLETE, chapter_id='chapter_001', panel_id='panel_001')
    tmp = tmp_path / 'final.tmp.mp4'
    final = tmp_path / 'final.mp4'
    tmp.write_bytes(b'synthetic-video-with-audio')
    digest = atomic_promote(tmp, final)
    st.record('canary', Stage.FINAL_QA, State.COMPLETE, artifact_path=final, artifact_checksum=digest, duration=1.0)
    assert reconcile_artifact(st, 'canary', Stage.FINAL_QA, final, lambda p: QAResult(True, duration=1.0))


def test_crash_resume_rejects_incomplete_and_reuses_complete(tmp_path: Path):
    st = SQLiteStateStore(tmp_path / 'resume.sqlite')
    complete = tmp_path / 'tts.wav'
    complete.write_bytes(b'valid-audio')
    st.record('job', Stage.TTS, State.RUNNING, artifact_path=complete)
    assert reconcile_artifact(st, 'job', Stage.TTS, complete, lambda p: QAResult(True, duration=1.0))
    assert st.get('job', Stage.TTS)['state'] == 'COMPLETE'

    corrupt = tmp_path / 'render.mp4'
    corrupt.write_bytes(b'partial')
    st.record('job', Stage.VIDEO_RENDER, State.RUNNING, artifact_path=corrupt, artifact_checksum=checksum_file(corrupt))
    assert not reconcile_artifact(st, 'job', Stage.VIDEO_RENDER, corrupt, lambda p: QAResult(False, 'ffprobe failed'))
    assert st.get('job', Stage.VIDEO_RENDER)['state'] == 'RETRYABLE'
