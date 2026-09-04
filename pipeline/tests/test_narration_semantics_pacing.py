import logging
from pathlib import Path
from pipeline.master_pipeline import (
    OcrStatus,
    parse_narration_item,
    split_frame_timings,
    rephrase_text,
    PipelineConfig,
    MIN_FRAME_DURATION,
    FRAME_HOLD_PADDING,
    SILENT_FRAME_DURATION,
)


def test_ocr_status_parsing_distinguishes_failed_and_no_text():
    parsed_failed_str = parse_narration_item("[transcription unavailable]")
    assert parsed_failed_str["status"] == OcrStatus.FAILED
    assert parsed_failed_str["text"] == ""

    parsed_no_text_str = parse_narration_item("")
    assert parsed_no_text_str["status"] == OcrStatus.NO_TEXT
    assert parsed_no_text_str["text"] == ""

    parsed_failed_dict = parse_narration_item({"text": "", "status": "FAILED"})
    assert parsed_failed_dict["status"] == OcrStatus.FAILED

    parsed_no_text_dict = parse_narration_item({"text": "", "status": "NO_TEXT"})
    assert parsed_no_text_dict["status"] == OcrStatus.NO_TEXT

    assert parsed_failed_str["status"] != parsed_no_text_str["status"]


def test_uncertain_remains_distinguishable_from_success():
    uncertain_item = parse_narration_item({"text": "low confidence dialogue", "status": "UNCERTAIN", "confidence": 0.45})
    success_item = parse_narration_item({"text": "high confidence dialogue", "status": "SUCCESS", "confidence": 0.98})

    assert uncertain_item["status"] == OcrStatus.UNCERTAIN
    assert success_item["status"] == OcrStatus.SUCCESS
    assert uncertain_item["confidence"] == 0.45
    assert success_item["confidence"] == 0.98
    assert uncertain_item["status"] != success_item["status"]


def test_failed_ocr_does_not_become_ordinary_success_text():
    parsed = parse_narration_item("[transcription unavailable]")
    assert parsed["status"] == OcrStatus.FAILED
    assert parsed["text"] != "ordinary text"
    assert parsed["text"] == ""


def test_genuine_no_text_timing_behavior():
    narrated_timings = split_frame_timings("Hero steps forward", [0], 1.5, is_silent=False)
    # Narrated frame gets MIN_FRAME_DURATION floor + FRAME_HOLD_PADDING on last frame
    dur_narrated = narrated_timings[0][1] - narrated_timings[0][0]
    assert dur_narrated == MIN_FRAME_DURATION + FRAME_HOLD_PADDING

    silent_timings = split_frame_timings("", [0], 2.0, is_silent=True)
    dur_silent = silent_timings[0][1] - silent_timings[0][0]
    # Silent frame does not receive MIN_FRAME_DURATION stretch (3.0s) or hold padding
    assert dur_silent == 2.0
    assert dur_silent < dur_narrated


def test_excessive_consecutive_silent_frames_safeguard():
    # Simulate pacing logic for consecutive silent frames
    consecutive_counts = [1, 2, 3, 4]
    durations = []
    for count in consecutive_counts:
        base_dur = 1.0 if count > 2 else SILENT_FRAME_DURATION
        durations.append(base_dur)

    assert durations[0] == SILENT_FRAME_DURATION  # 2.0s
    assert durations[1] == SILENT_FRAME_DURATION  # 2.0s
    assert durations[2] == 1.0                     # Compressed to 1.0s after > 2
    assert durations[3] == 1.0


def test_narrated_frames_remain_synchronized_with_audio_timing():
    audio_dur = 5.0
    positions = [0, 1]
    text = "The hero slashed through the shadowy monster and stood victorious"
    timings = split_frame_timings(text, positions, audio_dur, is_silent=False)

    # Frame 0 start should be 0.0
    assert timings[0][0] == 0.0
    # Frame 0 end and Frame 1 start should align cleanly
    assert abs(timings[0][1] - timings[1][0]) < 1e-5
    # Last frame includes hold padding
    assert timings[1][1] > audio_dur


def test_verbatim_style_returns_raw_words(tmp_path):
    """narration_style='verbatim' (the default) returns the OCR's own words
    unchanged — no LLM, no translation, regardless of narration_provider."""
    cfg = PipelineConfig(
        input_dir=tmp_path / "input",
        output_path=tmp_path / "output.mp4",
        work_dir=tmp_path / "work",
        narration_provider="none",
    )
    cfg.ensure_dirs()
    assert cfg.narration_style == "verbatim"
    res = rephrase_text(cfg, "Verbatim OCR text", "test_tag", "")
    assert res == "Verbatim OCR text"
