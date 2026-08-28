"""Unit tests for the semantic content-quality QA module (pipeline.content_quality)."""

import pytest
from pipeline.content_quality import (
    ChapterQAInput,
    CheckCode,
    FrameMetadata,
    QAStatus,
    QualityQAResult,
    QualityThresholds,
    evaluate_content_quality,
)


def test_healthy_metadata_returns_pass():
    frames = [
        FrameMetadata(
            frame_id=f"frame_{i:05d}.jpg",
            source_ref=f"page_{i//2:03d}.jpg#{i%2}",
            ocr_status="SUCCESS",
            ocr_text="The hero fights the ancient dragon with full power.",
            narration_text="The hero fights the ancient dragon with full power.",
            audio_duration=3.5,
            frame_duration=3.5,
            max_volume_db=-14.0,
            confidence=0.92,
        )
        for i in range(10)
    ]
    inp = ChapterQAInput(
        frames=frames,
        job_id="job_healthy",
        chapter_id="chap_001",
        total_audio_duration=35.0,
        total_chapter_duration=35.0,
        expected_frame_count=10,
    )
    result = evaluate_content_quality(inp)

    assert result.overall_status == QAStatus.PASS
    assert len(result.failures) == 0
    assert len(result.warnings) == 0
    assert result.metrics["total_frames"] == 10
    assert result.metrics["narration_coverage_ratio"] == 1.0
    assert result.metrics["total_narration_words"] == 90

    # Ensure to_dict serializes properly
    d = result.to_dict()
    assert d["overall_status"] == "PASS"
    assert d["metrics"]["total_frames"] == 10


def test_non_blocking_quality_concerns_return_warning():
    # 6 frames, 2 with narration, 4 silent (consecutive silence = 4, warning threshold = 6 is not hit, but coverage = 33.3% > warning threshold 25%)
    # Let's trigger a warning via low OCR success ratio (<20% warning threshold)
    frames = [
        FrameMetadata(
            frame_id="frame_00000.jpg",
            ocr_status="SUCCESS",
            ocr_text="Intro speech.",
            narration_text="Intro speech.",
            audio_duration=3.0,
            frame_duration=3.0,
        ),
        FrameMetadata(
            frame_id="frame_00001.jpg",
            ocr_status="FAILED",
            ocr_text="",
            narration_text="Narrator transition text.",
            audio_duration=3.0,
            frame_duration=3.0,
        ),
        FrameMetadata(
            frame_id="frame_00002.jpg",
            ocr_status="FAILED",
            ocr_text="",
            narration_text="Action scene continuation.",
            audio_duration=3.0,
            frame_duration=3.0,
        ),
        FrameMetadata(
            frame_id="frame_00003.jpg",
            ocr_status="FAILED",
            ocr_text="",
            narration_text="Another panel dialogue.",
            audio_duration=3.0,
            frame_duration=3.0,
        ),
        FrameMetadata(
            frame_id="frame_00004.jpg",
            ocr_status="FAILED",
            ocr_text="",
            narration_text="Final chapter conclusion statement.",
            audio_duration=3.0,
            frame_duration=3.0,
        ),
        FrameMetadata(
            frame_id="frame_00005.jpg",
            ocr_status="FAILED",
            ocr_text="",
            narration_text="End card summary.",
            audio_duration=3.0,
            frame_duration=3.0,
        ),
    ]
    inp = ChapterQAInput(frames=frames)
    result = evaluate_content_quality(inp)

    assert result.overall_status == QAStatus.WARNING
    assert len(result.failures) == 0
    assert len(result.warnings) > 0
    assert any(c.code == CheckCode.OCR_COVERAGE_LOW for c in result.check_results)


def test_severe_coverage_and_pacing_failures_return_fail():
    # 10 frames, 0 narration, completely silent
    frames = [
        FrameMetadata(
            frame_id=f"frame_{i:05d}.jpg",
            ocr_status="FAILED",
            ocr_text="",
            narration_text="",
            audio_duration=0.0,
            frame_duration=4.0,
        )
        for i in range(10)
    ]
    inp = ChapterQAInput(frames=frames)
    result = evaluate_content_quality(inp)

    assert result.overall_status == QAStatus.FAIL
    assert len(result.failures) > 0
    codes = [c.code for c in result.check_results if c.status == QAStatus.FAIL]
    assert CheckCode.NARRATION_COVERAGE_FAIL in codes
    assert CheckCode.EXCESSIVE_TOTAL_SILENCE in codes


def test_duplicate_frame_ids_and_references():
    frames = [
        FrameMetadata(frame_id="frame_00.jpg", source_ref="page_0.jpg#0", frame_duration=2.0),
        FrameMetadata(frame_id="frame_00.jpg", source_ref="page_0.jpg#0", frame_duration=2.0),
        FrameMetadata(frame_id="frame_01.jpg", source_ref="page_1.jpg#0", frame_duration=2.0),
    ]
    inp = ChapterQAInput(frames=frames)
    result = evaluate_content_quality(inp)

    assert result.overall_status == QAStatus.FAIL
    codes = [c.code for c in result.check_results]
    assert CheckCode.DUPLICATE_FRAME_IDS in codes
    assert CheckCode.DUPLICATE_FRAME_REFS in codes


def test_frame_count_mismatch():
    frames = [
        FrameMetadata(frame_id="frame_0.jpg", frame_duration=2.0),
        FrameMetadata(frame_id="frame_1.jpg", frame_duration=2.0),
    ]
    inp = ChapterQAInput(frames=frames, expected_frame_count=5)
    result = evaluate_content_quality(inp)

    assert result.overall_status == QAStatus.FAIL
    codes = [c.code for c in result.check_results]
    assert CheckCode.FRAME_COUNT_MISMATCH in codes


def test_consecutive_silent_sequence():
    frames = []
    # 15 consecutive silent frames
    for i in range(16):
        frames.append(
            FrameMetadata(
                frame_id=f"frame_{i:02d}.jpg",
                ocr_status="NONE",
                narration_text="" if i < 15 else "Final speech",
                audio_duration=0.0 if i < 15 else 3.0,
                frame_duration=2.0,
            )
        )
    inp = ChapterQAInput(frames=frames)
    result = evaluate_content_quality(inp)

    assert result.overall_status == QAStatus.FAIL
    codes = [c.code for c in result.check_results]
    assert CheckCode.EXCESSIVE_CONSECUTIVE_SILENCE in codes


def test_missing_optional_metadata_handled_safely():
    # Only minimal frame metadata provided
    frames = [
        FrameMetadata(frame_id="frame_0.jpg"),
        FrameMetadata(frame_id="frame_1.jpg"),
    ]
    inp = ChapterQAInput(frames=frames)
    result = evaluate_content_quality(inp)

    # Missing optional fields should not crash, evaluate cleanly
    assert isinstance(result, QualityQAResult)
    assert result.overall_status in (QAStatus.PASS, QAStatus.WARNING, QAStatus.FAIL)


def test_threshold_customization():
    frames = [
        FrameMetadata(
            frame_id="frame_0.jpg",
            ocr_status="FAILED",
            narration_text="Hello world",
            frame_duration=2.0,
        ),
        FrameMetadata(
            frame_id="frame_1.jpg",
            ocr_status="FAILED",
            narration_text="Sample audio text",
            frame_duration=2.0,
        ),
    ]
    inp = ChapterQAInput(frames=frames)

    # Custom strict threshold for OCR
    custom_thresholds = QualityThresholds(min_ocr_attempted_success_ratio_warning=0.80)
    result = evaluate_content_quality(inp, thresholds=custom_thresholds)

    assert result.overall_status == QAStatus.WARNING
    codes = [c.code for c in result.check_results]
    assert CheckCode.OCR_COVERAGE_LOW in codes


def test_low_narration_volume_warning():
    frames = [
        FrameMetadata(
            frame_id="frame_0.jpg",
            narration_text="Spoken narration text",
            audio_duration=3.0,
            frame_duration=3.0,
            max_volume_db=-52.0,  # Below -45dB default warning threshold
        )
    ]
    inp = ChapterQAInput(frames=frames)
    result = evaluate_content_quality(inp)

    assert result.overall_status == QAStatus.WARNING
    codes = [c.code for c in result.check_results]
    assert CheckCode.NARRATION_VOLUME_LOW in codes
