"""Semantic and content-quality QA module for recap pipeline outputs.

This module provides isolated, reusable content quality evaluation for recap output metadata.
It assesses structural integrity, OCR text coverage, narration coverage, audio pacing/silence,
narration sanity, and duration sanity without requiring heavy computer vision or media re-rendering.

Evaluation outputs structured results containing an overall QA status (PASS, WARNING, FAIL),
detailed per-check results, computed quality metrics, warning messages, failure messages, and
machine-readable error/warning codes.

Integration Usage:
------------------
Future integration tasks (e.g. in master_pipeline.py or production QA steps) can construct a
`ChapterQAInput` object from slice manifests, OCR results, and audio timing metadata, then call:

    from pipeline.content_quality import evaluate_content_quality, ChapterQAInput, FrameMetadata

    frames = [
        FrameMetadata(
            frame_id="frame_00000.jpg",
            source_ref="page_001.jpg#0",
            ocr_status="SUCCESS",
            ocr_text="The hero enters the dark forest.",
            narration_text="The hero enters the dark forest.",
            audio_duration=3.5,
            frame_duration=3.5,
            max_volume_db=-18.2,
            confidence=0.95
        ),
        ...
    ]

    input_data = ChapterQAInput(
        frames=frames,
        job_id="job_123",
        chapter_id="chap_001",
        total_audio_duration=42.0,
        total_chapter_duration=42.0,
        expected_frame_count=12
    )

    result = evaluate_content_quality(input_data)
    if result.overall_status == QAStatus.FAIL:
        print("Quality QA Failed:", result.failures)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class QAStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class CheckCode(str, Enum):
    EMPTY_INPUT = "EMPTY_INPUT"
    DUPLICATE_FRAME_IDS = "DUPLICATE_FRAME_IDS"
    DUPLICATE_FRAME_REFS = "DUPLICATE_FRAME_REFS"
    FRAME_COUNT_MISMATCH = "FRAME_COUNT_MISMATCH"

    OCR_COVERAGE_FAIL = "OCR_COVERAGE_FAIL"
    OCR_COVERAGE_LOW = "OCR_COVERAGE_LOW"
    OCR_HIGH_FALLBACK = "OCR_HIGH_FALLBACK"

    NARRATION_COVERAGE_FAIL = "NARRATION_COVERAGE_FAIL"
    NARRATION_COVERAGE_LOW = "NARRATION_COVERAGE_LOW"
    NARRATION_WORD_COUNT_LOW = "NARRATION_WORD_COUNT_LOW"
    NARRATION_VOLUME_LOW = "NARRATION_VOLUME_LOW"

    EXCESSIVE_CONSECUTIVE_SILENCE = "EXCESSIVE_CONSECUTIVE_SILENCE"
    EXCESSIVE_TOTAL_SILENCE = "EXCESSIVE_TOTAL_SILENCE"

    INVALID_FRAME_DURATION = "INVALID_FRAME_DURATION"
    UNREASONABLE_PACING = "UNREASONABLE_PACING"
    AUDIO_DURATION_MISMATCH = "AUDIO_DURATION_MISMATCH"


@dataclass
class FrameMetadata:
    frame_id: str
    source_ref: Optional[str] = None
    ocr_status: str = "NONE"  # e.g. SUCCESS, UNCERTAIN, FAILED, FALLBACK, NONE, COMPLETE
    ocr_text: str = ""
    narration_text: str = ""
    audio_duration: float = 0.0
    frame_duration: float = 0.0
    is_silent: Optional[bool] = None
    max_volume_db: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChapterQAInput:
    frames: List[FrameMetadata] = field(default_factory=list)
    job_id: str = ""
    chapter_id: str = ""
    total_audio_duration: Optional[float] = None
    total_chapter_duration: Optional[float] = None
    expected_frame_count: Optional[int] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityThresholds:
    """Configurable conservative quality thresholds.

    Thresholds are intentionally set conservatively to avoid rejecting legitimate visual or slow-paced chapters.
    """

    # Structural thresholds
    allow_empty_frames: bool = False

    # OCR coverage thresholds
    min_ocr_attempted_success_ratio_warning: float = 0.20  # Warn if <20% OCR succeeded when OCR attempted
    min_ocr_attempted_success_ratio_fail: float = 0.05  # Fail if <5% OCR succeeded when OCR attempted across >= 5 frames
    max_ocr_fallback_ratio_warning: float = 0.70  # Warn if >=70% of OCR used fallbacks

    # Narration coverage thresholds
    min_narration_coverage_ratio_warning: float = 0.25  # Warn if <25% of frames have narration
    min_narration_coverage_ratio_fail: float = 0.05  # Fail if <5% of frames have narration (for chapters >= 5 frames)
    min_total_narration_words_warning: int = 5  # Warn if total chapter narration < 5 words (for >= 3 frames)
    min_total_narration_words_fail: int = 1  # Fail if total chapter narration < 1 word (for >= 5 frames)

    # Silent pacing thresholds
    max_consecutive_silent_frames_warning: int = 6  # Warn if >= 6 consecutive silent frames
    max_consecutive_silent_frames_fail: int = 15  # Fail if >= 15 consecutive silent frames
    max_silent_duration_ratio_warning: float = 0.45  # Warn if >45% of total time is silent
    max_silent_duration_ratio_fail: float = 0.75  # Fail if >75% of total time is silent (for >10s chapter)

    # Audio volume thresholds (when volume metadata is available)
    min_max_volume_db_warning: float = -45.0  # Warn if max volume <= -45dB when non-silent audio expected

    # Duration sanity thresholds
    min_frame_duration_sec_warning: float = 0.4  # Warn if average frame duration < 0.4s
    min_frame_duration_sec_fail: float = 0.1  # Fail if any individual frame duration <= 0.1s
    max_frame_duration_sec_warning: float = 45.0  # Warn if individual frame duration > 45s
    max_frame_duration_sec_fail: float = 120.0  # Fail if individual frame duration > 120s
    duration_mismatch_warning_ratio: float = 0.35  # Warn if total frame durations vs total chapter/audio duration mismatch > 35%
    duration_mismatch_fail_ratio: float = 0.60  # Fail if total frame durations vs total chapter/audio duration mismatch > 60%


@dataclass
class CheckResult:
    code: CheckCode
    status: QAStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityQAResult:
    overall_status: QAStatus
    check_results: List[CheckResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "check_results": [
                {
                    "code": c.code.value,
                    "status": c.status.value,
                    "message": c.message,
                    "metrics": c.metrics,
                }
                for c in self.check_results
            ],
            "metrics": self.metrics,
            "warnings": self.warnings,
            "failures": self.failures,
        }


def _count_words(text: str) -> int:
    if not text:
        return 0
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return len(words)


def _is_frame_silent(frame: FrameMetadata) -> bool:
    if frame.is_silent is not None:
        return frame.is_silent
    has_narration = bool(frame.narration_text and frame.narration_text.strip())
    has_audio = frame.audio_duration > 0.05
    return not (has_narration or has_audio)


def evaluate_content_quality(
    input_data: ChapterQAInput,
    thresholds: Optional[QualityThresholds] = None,
) -> QualityQAResult:
    """Evaluates the semantic content quality of a chapter's metadata against conservative quality thresholds."""
    if thresholds is None:
        thresholds = QualityThresholds()

    check_results: List[CheckResult] = []
    metrics: Dict[str, Any] = {}
    warnings: List[str] = []
    failures: List[str] = []

    frames = input_data.frames
    total_frames = len(frames)
    metrics["total_frames"] = total_frames

    # --- 1. Structural Consistency Checks ---
    if total_frames == 0:
        if not thresholds.allow_empty_frames:
            res = CheckResult(
                code=CheckCode.EMPTY_INPUT,
                status=QAStatus.FAIL,
                message="Chapter metadata contains no frames.",
                metrics={"total_frames": 0},
            )
            check_results.append(res)
            failures.append(res.message)
            return QualityQAResult(
                overall_status=QAStatus.FAIL,
                check_results=check_results,
                metrics=metrics,
                warnings=warnings,
                failures=failures,
            )

    # Check for duplicate frame IDs
    seen_frame_ids: Set[str] = set()
    duplicate_frame_ids: Set[str] = set()
    for frame in frames:
        if frame.frame_id in seen_frame_ids:
            duplicate_frame_ids.add(frame.frame_id)
        seen_frame_ids.add(frame.frame_id)

    metrics["duplicate_frame_ids_count"] = len(duplicate_frame_ids)
    if duplicate_frame_ids:
        msg = f"Duplicate frame IDs detected: {sorted(list(duplicate_frame_ids))}"
        res = CheckResult(
            code=CheckCode.DUPLICATE_FRAME_IDS,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"duplicates": sorted(list(duplicate_frame_ids))},
        )
        check_results.append(res)
        failures.append(msg)

    # Check for duplicate frame references (when source_ref is uniquely formatted per frame)
    seen_source_refs: Set[str] = set()
    duplicate_source_refs: Set[str] = set()
    for frame in frames:
        if frame.source_ref:
            if frame.source_ref in seen_source_refs:
                duplicate_source_refs.add(frame.source_ref)
            seen_source_refs.add(frame.source_ref)

    metrics["duplicate_source_refs_count"] = len(duplicate_source_refs)
    if duplicate_source_refs:
        msg = f"Duplicate frame source references detected: {sorted(list(duplicate_source_refs))}"
        res = CheckResult(
            code=CheckCode.DUPLICATE_FRAME_REFS,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"duplicates": sorted(list(duplicate_source_refs))},
        )
        check_results.append(res)
        failures.append(msg)

    # Check for frame count mismatch against expected_frame_count
    if input_data.expected_frame_count is not None and total_frames != input_data.expected_frame_count:
        msg = f"Frame count mismatch: expected {input_data.expected_frame_count}, got {total_frames}"
        res = CheckResult(
            code=CheckCode.FRAME_COUNT_MISMATCH,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"expected": input_data.expected_frame_count, "actual": total_frames},
        )
        check_results.append(res)
        failures.append(msg)

    # --- 2. OCR Coverage Checks ---
    ocr_attempted_frames = 0
    ocr_success_count = 0
    ocr_uncertain_count = 0
    ocr_failed_count = 0
    ocr_fallback_count = 0

    for frame in frames:
        status_norm = (frame.ocr_status or "").upper()
        if status_norm in ("NONE", "SKIP", "SKIPPED"):
            continue

        ocr_attempted_frames += 1
        if status_norm in ("SUCCESS", "COMPLETE", "OK"):
            ocr_success_count += 1
        elif status_norm in ("UNCERTAIN",):
            ocr_uncertain_count += 1
        elif status_norm in ("FAILED", "FAILURE", "ERROR"):
            ocr_failed_count += 1
        elif "FALLBACK" in status_norm:
            ocr_fallback_count += 1

    metrics["ocr_attempted_frames"] = ocr_attempted_frames
    metrics["ocr_success_count"] = ocr_success_count
    metrics["ocr_uncertain_count"] = ocr_uncertain_count
    metrics["ocr_failed_count"] = ocr_failed_count
    metrics["ocr_fallback_count"] = ocr_fallback_count

    if ocr_attempted_frames > 0:
        ocr_success_ratio = ocr_success_count / ocr_attempted_frames
        ocr_fallback_ratio = ocr_fallback_count / ocr_attempted_frames
        metrics["ocr_success_ratio"] = round(ocr_success_ratio, 4)
        metrics["ocr_fallback_ratio"] = round(ocr_fallback_ratio, 4)

        if total_frames >= 5 and ocr_success_ratio < thresholds.min_ocr_attempted_success_ratio_fail:
            msg = f"Extremely low OCR success ratio ({ocr_success_ratio:.1%}) across attempted frames."
            res = CheckResult(
                code=CheckCode.OCR_COVERAGE_FAIL,
                status=QAStatus.FAIL,
                message=msg,
                metrics={"ocr_success_ratio": ocr_success_ratio},
            )
            check_results.append(res)
            failures.append(msg)
        elif ocr_success_ratio < thresholds.min_ocr_attempted_success_ratio_warning:
            msg = f"Low OCR success ratio ({ocr_success_ratio:.1%}) across attempted frames."
            res = CheckResult(
                code=CheckCode.OCR_COVERAGE_LOW,
                status=QAStatus.WARNING,
                message=msg,
                metrics={"ocr_success_ratio": ocr_success_ratio},
            )
            check_results.append(res)
            warnings.append(msg)

        if ocr_fallback_ratio >= thresholds.max_ocr_fallback_ratio_warning:
            msg = f"High proportion of OCR relied on fallback mechanisms ({ocr_fallback_ratio:.1%})."
            res = CheckResult(
                code=CheckCode.OCR_HIGH_FALLBACK,
                status=QAStatus.WARNING,
                message=msg,
                metrics={"ocr_fallback_ratio": ocr_fallback_ratio},
            )
            check_results.append(res)
            warnings.append(msg)

    # --- 3. Narration Coverage Checks ---
    frames_with_narration = 0
    total_narration_words = 0

    for frame in frames:
        narr_text = (frame.narration_text or "").strip()
        word_cnt = _count_words(narr_text)
        if word_cnt > 0:
            frames_with_narration += 1
            total_narration_words += word_cnt

    frames_without_narration = total_frames - frames_with_narration
    narration_coverage_ratio = frames_with_narration / total_frames if total_frames > 0 else 0.0
    empty_narration_ratio = frames_without_narration / total_frames if total_frames > 0 else 0.0

    metrics["frames_with_narration"] = frames_with_narration
    metrics["frames_without_narration"] = frames_without_narration
    metrics["narration_coverage_ratio"] = round(narration_coverage_ratio, 4)
    metrics["empty_narration_ratio"] = round(empty_narration_ratio, 4)
    metrics["total_narration_words"] = total_narration_words

    if total_frames >= 5 and narration_coverage_ratio < thresholds.min_narration_coverage_ratio_fail:
        msg = f"Critically low narration coverage: only {narration_coverage_ratio:.1%} of frames have narration."
        res = CheckResult(
            code=CheckCode.NARRATION_COVERAGE_FAIL,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"narration_coverage_ratio": narration_coverage_ratio},
        )
        check_results.append(res)
        failures.append(msg)
    elif narration_coverage_ratio < thresholds.min_narration_coverage_ratio_warning:
        msg = f"Low narration coverage: {narration_coverage_ratio:.1%} of frames have narration."
        res = CheckResult(
            code=CheckCode.NARRATION_COVERAGE_LOW,
            status=QAStatus.WARNING,
            message=msg,
            metrics={"narration_coverage_ratio": narration_coverage_ratio},
        )
        check_results.append(res)
        warnings.append(msg)

    # --- 4. Silent Pacing Checks ---
    current_consecutive_silent = 0
    max_consecutive_silent = 0
    total_silent_duration = 0.0
    total_frame_duration = 0.0

    for frame in frames:
        total_frame_duration += frame.frame_duration
        if _is_frame_silent(frame):
            current_consecutive_silent += 1
            if current_consecutive_silent > max_consecutive_silent:
                max_consecutive_silent = current_consecutive_silent
            total_silent_duration += frame.frame_duration
        else:
            current_consecutive_silent = 0

    silent_duration_ratio = (
        total_silent_duration / total_frame_duration if total_frame_duration > 0.0 else 0.0
    )

    metrics["max_consecutive_silent_frames"] = max_consecutive_silent
    metrics["total_silent_duration"] = round(total_silent_duration, 2)
    metrics["total_frame_duration"] = round(total_frame_duration, 2)
    metrics["silent_duration_ratio"] = round(silent_duration_ratio, 4)

    if max_consecutive_silent >= thresholds.max_consecutive_silent_frames_fail:
        msg = f"Excessive consecutive silent frames detected ({max_consecutive_silent} frames)."
        res = CheckResult(
            code=CheckCode.EXCESSIVE_CONSECUTIVE_SILENCE,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"max_consecutive_silent_frames": max_consecutive_silent},
        )
        check_results.append(res)
        failures.append(msg)
    elif max_consecutive_silent >= thresholds.max_consecutive_silent_frames_warning:
        msg = f"High number of consecutive silent frames detected ({max_consecutive_silent} frames)."
        res = CheckResult(
            code=CheckCode.EXCESSIVE_CONSECUTIVE_SILENCE,
            status=QAStatus.WARNING,
            message=msg,
            metrics={"max_consecutive_silent_frames": max_consecutive_silent},
        )
        check_results.append(res)
        warnings.append(msg)

    if total_frame_duration >= 10.0 and silent_duration_ratio >= thresholds.max_silent_duration_ratio_fail:
        msg = f"Excessive total silent duration ratio ({silent_duration_ratio:.1%})."
        res = CheckResult(
            code=CheckCode.EXCESSIVE_TOTAL_SILENCE,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"silent_duration_ratio": silent_duration_ratio},
        )
        check_results.append(res)
        failures.append(msg)
    elif silent_duration_ratio >= thresholds.max_silent_duration_ratio_warning:
        msg = f"High total silent duration ratio ({silent_duration_ratio:.1%})."
        res = CheckResult(
            code=CheckCode.EXCESSIVE_TOTAL_SILENCE,
            status=QAStatus.WARNING,
            message=msg,
            metrics={"silent_duration_ratio": silent_duration_ratio},
        )
        check_results.append(res)
        warnings.append(msg)

    # --- 5. Narration Sanity Checks ---
    if total_frames >= 5 and total_narration_words < thresholds.min_total_narration_words_fail:
        msg = f"Implausibly low total narration word count ({total_narration_words} words)."
        res = CheckResult(
            code=CheckCode.NARRATION_WORD_COUNT_LOW,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"total_narration_words": total_narration_words},
        )
        check_results.append(res)
        failures.append(msg)
    elif total_frames >= 3 and total_narration_words < thresholds.min_total_narration_words_warning:
        msg = f"Suspiciously low total narration word count ({total_narration_words} words)."
        res = CheckResult(
            code=CheckCode.NARRATION_WORD_COUNT_LOW,
            status=QAStatus.WARNING,
            message=msg,
            metrics={"total_narration_words": total_narration_words},
        )
        check_results.append(res)
        warnings.append(msg)

    # Low audio volume check
    low_volume_frames: List[str] = []
    for frame in frames:
        if frame.max_volume_db is not None and frame.audio_duration > 0.05:
            if frame.max_volume_db <= thresholds.min_max_volume_db_warning:
                low_volume_frames.append(frame.frame_id)

    metrics["low_volume_frames_count"] = len(low_volume_frames)
    if low_volume_frames:
        msg = f"Suspiciously low audio volume (<= {thresholds.min_max_volume_db_warning}dB) on frames: {low_volume_frames}"
        res = CheckResult(
            code=CheckCode.NARRATION_VOLUME_LOW,
            status=QAStatus.WARNING,
            message=msg,
            metrics={"low_volume_frames": low_volume_frames},
        )
        check_results.append(res)
        warnings.append(msg)

    # --- 6. Duration Sanity Checks ---
    avg_frame_duration = total_frame_duration / total_frames if total_frames > 0 else 0.0
    metrics["avg_frame_duration"] = round(avg_frame_duration, 2)

    invalid_duration_frames: List[str] = []
    long_duration_frames: List[str] = []

    for frame in frames:
        if (
            frame.frame_duration <= thresholds.min_frame_duration_sec_fail
            or frame.frame_duration > thresholds.max_frame_duration_sec_fail
        ):
            invalid_duration_frames.append(f"{frame.frame_id}:{frame.frame_duration}s")
        elif frame.frame_duration > thresholds.max_frame_duration_sec_warning:
            long_duration_frames.append(f"{frame.frame_id}:{frame.frame_duration}s")

    if invalid_duration_frames:
        msg = f"Frames with invalid duration (<= {thresholds.min_frame_duration_sec_fail}s or > {thresholds.max_frame_duration_sec_fail}s): {invalid_duration_frames}"
        res = CheckResult(
            code=CheckCode.INVALID_FRAME_DURATION,
            status=QAStatus.FAIL,
            message=msg,
            metrics={"invalid_frames": invalid_duration_frames},
        )
        check_results.append(res)
        failures.append(msg)

    if long_duration_frames:
        msg = f"Frames with long duration (> {thresholds.max_frame_duration_sec_warning}s): {long_duration_frames}"
        res = CheckResult(
            code=CheckCode.UNREASONABLE_PACING,
            status=QAStatus.WARNING,
            message=msg,
            metrics={"long_frames": long_duration_frames},
        )
        check_results.append(res)
        warnings.append(msg)

    if avg_frame_duration > 0 and avg_frame_duration < thresholds.min_frame_duration_sec_warning:
        msg = f"Suspiciously fast average frame duration ({avg_frame_duration:.2f}s per frame)."
        res = CheckResult(
            code=CheckCode.UNREASONABLE_PACING,
            status=QAStatus.WARNING,
            message=msg,
            metrics={"avg_frame_duration": avg_frame_duration},
        )
        check_results.append(res)
        warnings.append(msg)

    # Duration comparison with chapter total audio/video metadata
    target_duration = input_data.total_chapter_duration or input_data.total_audio_duration
    if target_duration is not None and target_duration > 0 and total_frame_duration > 0:
        duration_diff = abs(total_frame_duration - target_duration)
        mismatch_ratio = duration_diff / max(target_duration, 1.0)
        metrics["duration_mismatch_ratio"] = round(mismatch_ratio, 4)

        if mismatch_ratio >= thresholds.duration_mismatch_fail_ratio:
            msg = f"Severe duration mismatch: sum of frame durations ({total_frame_duration:.1f}s) vs expected duration ({target_duration:.1f}s) differs by {mismatch_ratio:.1%}."
            res = CheckResult(
                code=CheckCode.AUDIO_DURATION_MISMATCH,
                status=QAStatus.FAIL,
                message=msg,
                metrics={"total_frame_duration": total_frame_duration, "target_duration": target_duration},
            )
            check_results.append(res)
            failures.append(msg)
        elif mismatch_ratio >= thresholds.duration_mismatch_warning_ratio:
            msg = f"Duration mismatch: sum of frame durations ({total_frame_duration:.1f}s) vs expected duration ({target_duration:.1f}s) differs by {mismatch_ratio:.1%}."
            res = CheckResult(
                code=CheckCode.AUDIO_DURATION_MISMATCH,
                status=QAStatus.WARNING,
                message=msg,
                metrics={"total_frame_duration": total_frame_duration, "target_duration": target_duration},
            )
            check_results.append(res)
            warnings.append(msg)

    # --- 7. Overall Status Determination ---
    if failures:
        overall_status = QAStatus.FAIL
    elif warnings:
        overall_status = QAStatus.WARNING
    else:
        overall_status = QAStatus.PASS

    return QualityQAResult(
        overall_status=overall_status,
        check_results=check_results,
        metrics=metrics,
        warnings=warnings,
        failures=failures,
    )
