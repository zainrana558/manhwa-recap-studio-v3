"""Tests for split_frame_timings' real-word-timing behavior (added
alongside the fix wiring edge-tts's actual per-word timestamps into
frame timing, in place of a pure proportional word-count estimate)."""
from pipeline.master_pipeline import split_frame_timings


def _wb(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_uses_real_timing_when_word_counts_match():
    # Two frames, 2 words each. Real timing is deliberately uneven (the
    # first word takes much longer than the rest) -- something a
    # proportional word-count split could never represent, since it
    # assumes every word takes an equal share of the total duration.
    # Spans are kept comfortably above MIN_FRAME_DURATION (3.0s) so that
    # floor's own, separate, already-correct enforcement doesn't mask
    # the real-vs-proportional difference this test is actually checking.
    text = "Hello world foo bar"
    positions = [0, 1]
    word_boundaries = [
        _wb("Hello", 0.0, 5.0),
        _wb("world", 5.0, 6.0),
        _wb("foo", 6.0, 6.5),
        _wb("bar", 6.5, 7.0),
    ]
    timings = split_frame_timings(text, positions, duration=7.0, word_boundaries=word_boundaries)

    # Frame 0 covers "Hello world" -> real span is 0.0 to 6.0 (6.0s),
    # NOT a proportional 3.5s (half of 7.0s for 2 of 4 words).
    start0, end0 = timings[0]
    assert start0 == 0.0
    assert end0 == 6.0

    # Frame 1 covers "foo bar" -> real span 6.0 to 7.0, then
    # FRAME_HOLD_PADDING (0.5s) is added to the last frame in the segment
    # by the existing downstream logic, unchanged by this fix.
    start1, end1 = timings[1]
    assert start1 == 6.0
    assert end1 > 7.0  # 0.5s FRAME_HOLD_PADDING on top of the real 7.0s end


def test_falls_back_to_proportional_when_word_count_mismatches():
    # 4 real words in text, but only 3 boundary entries -- edge-tts's own
    # tokenization can legitimately diverge from a naive whitespace split
    # (contractions, hyphenation). Forcing a mismatched 1:1 mapping here
    # would silently produce wrong timing, so this must fall back to the
    # pre-existing proportional estimate instead of guessing.
    text = "one two three four"
    positions = [0]
    word_boundaries = [_wb("one", 0.0, 1.0), _wb("two", 1.0, 2.0), _wb("three", 2.0, 3.0)]
    timings = split_frame_timings(text, positions, duration=4.0, word_boundaries=word_boundaries)
    # Single frame, all 4 words assigned to it -> proportional split just
    # gives it the full duration (before MIN_FRAME_DURATION/padding).
    start, end = timings[0]
    assert start == 0.0
    assert end >= 4.0


def test_falls_back_to_proportional_when_no_word_boundaries():
    text = "hello world"
    positions = [0, 1]
    timings = split_frame_timings(text, positions, duration=2.0, word_boundaries=None)
    # Proportional: 1 word each out of 2.0s duration -> 1.0s each
    # (before MIN_FRAME_DURATION stretches it further, which is fine --
    # this just confirms it didn't crash and produced ordered, non-empty
    # spans using the pre-existing estimate).
    assert timings[0][0] == 0.0
    assert timings[0][1] > timings[0][0]
    assert timings[1][0] >= timings[0][1]


def test_real_timing_never_used_for_silent_frames():
    # is_silent=True must never engage the real-timing path even if
    # boundaries happen to be provided -- silent frames use a much
    # shorter, differently-paced fallback (SILENT_FRAME_DURATION-based),
    # not narration timing at all.
    text = ""
    positions = [0, 1]
    word_boundaries = [_wb("x", 0.0, 1.0)]
    timings = split_frame_timings(text, positions, duration=2.0, is_silent=True, word_boundaries=word_boundaries)
    assert timings[0][1] > timings[0][0]
    assert timings[1][0] >= timings[0][1]
