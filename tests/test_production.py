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


def test_ocr_text_for_narration():
    res = OCRResult(text=" Hello ", status=State.COMPLETE)
    assert ocr_text_for_narration(res) == "Hello"
    res_failed = OCRResult(text=" Hello ", status=State.FAILED)
    assert ocr_text_for_narration(res_failed) == ""


def test_clean_ocr_text_with_sfx():
    # 1. Non-ASCII / unprintable stripping
    dirty = "Hello \x00World\x07! \u200bBOOM!"
    cleaned = clean_ocr_text_with_sfx(dirty)
    assert "BOOM!" in cleaned
    assert "\x00" not in cleaned

    # 2. SFX preservation & normalization
    sfx_text = "The hero strikes xz ll1 BOOM!! BANG swoosh"
    cleaned_sfx = clean_ocr_text_with_sfx(sfx_text)
    assert "xz" not in cleaned_sfx
    assert "ll1" not in cleaned_sfx
    assert "BOOM!!" in cleaned_sfx
    assert "BANG" in cleaned_sfx
    assert "Swoosh" in cleaned_sfx

    # 3. 1-2 char fragment filtering
    fragments = "I am a warrior xz q1 q2 to the end"
    cleaned_frag = clean_ocr_text_with_sfx(fragments)
    assert cleaned_frag == "I am a warrior to the end"

    # 4. Sentence boundary punctuation cleanup, character substitutions, and end cards
    mangled = "The monster is good-curdling dash. HO0 HO O B to be continued..."
    cleaned_mangled = clean_ocr_text_with_sfx(mangled)
    assert "blood-curdling" in cleaned_mangled
    assert "HOO HOO" in cleaned_mangled
    assert "To Be Continued..." in cleaned_mangled

    # 5. Spoken symbol conversions at sentence boundary
    boundary = "He ran away underscore"
    cleaned_boundary = clean_ocr_text_with_sfx(boundary)
    assert cleaned_boundary.endswith("...")
