from pathlib import Path
from pipeline.production import *


def test_retry_classification():
    assert classify_retry('operation timed out') == RetryCategory.TIMEOUT
    assert classify_retry('piper binary not found') == RetryCategory.PROVIDER_UNAVAILABLE
    assert classify_retry('audio silence quality failure') == RetryCategory.QUALITY
    assert classify_retry('ffprobe corrupt artifact') == RetryCategory.CORRUPT_ARTIFACT


def test_state_granular_retry_persistence(tmp_path: Path):
    st = SQLiteStateStore(tmp_path / 'state.sqlite')
    st.record('job1', Stage.OCR, State.RETRYABLE, chapter_id='c1', panel_id='p1', provider='paddleocr', model='PP-OCRv5', error_code='QUALITY', retry_category=RetryCategory.QUALITY)
    row = st.get('job1', Stage.OCR, 'c1', 'p1')
    assert row and row['state'] == 'RETRYABLE'
    assert row['attempt_count'] == 1
    assert row['provider'] == 'paddleocr'
    assert row['retry_category'] == 'QUALITY'


def test_atomic_promote_and_reconcile(tmp_path: Path):
    st = SQLiteStateStore(tmp_path / 'state.sqlite')
    tmp = tmp_path / 'artifact.tmp.wav'
    final = tmp_path / 'artifact.wav'
    tmp.write_bytes(b'audio-bytes')
    digest = atomic_promote(tmp, final)
    assert final.exists() and not tmp.exists()
    st.record('job1', Stage.TTS, State.COMPLETE, chapter_id='c1', panel_id='p1', artifact_path=final, artifact_checksum=digest)
    assert reconcile_artifact(st, 'job1', Stage.TTS, final, lambda p: QAResult(True, duration=1.0), 'c1', 'p1')


def test_reconcile_quarantines_corrupt_artifact(tmp_path: Path):
    st = SQLiteStateStore(tmp_path / 'state.sqlite')
    final = tmp_path / 'bad.wav'
    final.write_bytes(b'bad')
    st.record('job1', Stage.TTS, State.COMPLETE, artifact_path=final, artifact_checksum=checksum_file(final))
    assert not reconcile_artifact(st, 'job1', Stage.TTS, final, lambda p: QAResult(False, 'ffprobe failed'))
    assert not final.exists()
    assert list((tmp_path / 'quarantine').iterdir())
    assert st.get('job1', Stage.TTS)['state'] == 'RETRYABLE'


def test_resource_guard_persists_resource_state(tmp_path: Path):
    st = SQLiteStateStore(tmp_path / 'state.sqlite')
    guard = ResourceGuard(min_free_disk_bytes=10**30, min_available_ram_bytes=0, state=st)
    status = guard.check(tmp_path, job_id='job1', stage=Stage.VIDEO_RENDER)
    assert not status.ok
    row = st.get('job1', Stage.VIDEO_RENDER)
    assert row and row['error_code'] == 'RESOURCE'


def test_artifact_store_tmp_promote_quarantine(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    final = tmp_path / 'x.wav'
    tmp = store.tmp_for(final)
    tmp.write_bytes(b'ok')
    digest = store.promote(tmp, final)
    assert digest == checksum_file(final)
    bad = tmp_path / 'bad.wav'
    bad.write_bytes(b'bad')
    quarantined = store.quarantine(bad, 'qa')
    assert quarantined.exists() and not bad.exists()
