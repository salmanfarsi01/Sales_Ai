from __future__ import annotations

import pytest

from copilot.behavioral_normalization import (
    NormalizedWord,
    NormalizedUtterance,
    normalize_generic_transcript,
)
from copilot.behavioral_timing import (
    TimingFeatureSnapshot,
    DeterministicTimingEngine,
)


def test_timing_engine_speech_rate_and_turn_length():
    engine = DeterministicTimingEngine(window_ms=60000)

    words = [
        NormalizedWord(word="Good", start_ms=1000, end_ms=1200, confidence=0.99),
        NormalizedWord(word="morning,", start_ms=1250, end_ms=1550, confidence=0.99),
        NormalizedWord(word="how", start_ms=1600, end_ms=1750, confidence=0.99),
        NormalizedWord(word="are", start_ms=1800, end_ms=1950, confidence=0.99),
        NormalizedWord(word="you?", start_ms=2000, end_ms=2500, confidence=0.99),
    ]

    utt1 = NormalizedUtterance(
        call_sid="call_time_1",
        speaker_id="salesperson",
        text="Good morning, how are you?",
        start_ms=1000,
        end_ms=2500,
        is_estimated_timing=False,
        words=words,
    )

    snapshot = engine.process_utterance(utt1)

    assert snapshot.turn_length_words == 5
    assert snapshot.turn_duration_ms == 1500
    assert snapshot.timing_confidence == 1.0
    # 5 words in 1.5 seconds -> (5 / (1.5 / 60)) = 200 WPM
    assert snapshot.speech_rate_wpm == 200.0


def test_timing_engine_intra_turn_pauses_and_synthetic_penalty():
    engine = DeterministicTimingEngine(min_pause_threshold_ms=250)

    # 1. Observed timestamps with deliberate pauses
    words_with_pauses = [
        NormalizedWord(word="I", start_ms=1000, end_ms=1100, confidence=0.99),
        NormalizedWord(word="think", start_ms=1450, end_ms=1600, confidence=0.99), # gap 350ms
        NormalizedWord(word="we", start_ms=1950, end_ms=2100, confidence=0.99),    # gap 350ms
        NormalizedWord(word="should.", start_ms=2150, end_ms=2350, confidence=0.99),# gap 50ms
    ]

    utt_observed = NormalizedUtterance(
        call_sid="call_time_2",
        speaker_id="client",
        text="I think we should.",
        start_ms=1000,
        end_ms=2350,
        is_estimated_timing=False,
        words=words_with_pauses,
    )

    snap1 = engine.process_utterance(utt_observed)
    assert snap1.intra_turn_pause_count == 2
    assert snap1.avg_pause_duration_ms == 350.0
    assert snap1.pause_measured is True
    assert snap1.timing_confidence == 1.0

    # 2. Synthetic timing fallback: must not count synthetic spacing and must penalize confidence
    utt_synthetic = normalize_generic_transcript(
        text="This is a synthetic fallback sentence.",
        speaker_id="client",
        start_ms=3000,
        end_ms=5000,
        call_sid="call_time_2",
        is_estimated_timing=True,
    )

    snap2 = engine.process_utterance(utt_synthetic)
    assert snap2.intra_turn_pause_count == 0
    assert snap2.avg_pause_duration_ms == 0.0
    assert snap2.pause_measured is False
    assert snap2.timing_confidence == 0.5


def test_timing_engine_response_latency_and_interruptions():
    engine = DeterministicTimingEngine(window_ms=60000)

    # Turn 1: Rep speaks 1000ms - 3000ms
    utt1 = normalize_generic_transcript(
        text="What is your timeline for moving?",
        speaker_id="salesperson",
        start_ms=1000,
        end_ms=3000,
        call_sid="call_time_3",
        is_estimated_timing=False,
    )
    engine.process_utterance(utt1)

    # Turn 2: Client responds after 800ms gap (3800ms - 5000ms)
    utt2 = normalize_generic_transcript(
        text="We want to move by end of year.",
        speaker_id="client",
        start_ms=3800,
        end_ms=5000,
        call_sid="call_time_3",
        is_estimated_timing=False,
    )
    snap2 = engine.process_utterance(utt2)
    assert snap2.response_latency_ms == 800
    assert snap2.interruptions_60s == 0

    # Turn 3: Rep interrupts while client was still speaking (overlaps at 4600ms)
    utt3 = normalize_generic_transcript(
        text="Understood, that gives us time.",
        speaker_id="salesperson",
        start_ms=4600,
        end_ms=6000,
        call_sid="call_time_3",
        is_estimated_timing=False,
    )
    snap3 = engine.process_utterance(utt3)
    assert snap3.response_latency_ms == -400
    assert snap3.interruptions_60s == 1


def test_timing_engine_question_rate():
    engine = DeterministicTimingEngine(window_ms=60000)

    q1 = normalize_generic_transcript(
        text="What is your fee?",
        speaker_id="client",
        start_ms=1000,
        end_ms=2000,
        call_sid="call_time_4",
    )
    snap1 = engine.process_utterance(q1)
    assert snap1.question_count_60s == 1
    assert snap1.question_rate_per_min > 0.0

    statement = normalize_generic_transcript(
        text="I understand your position.",
        speaker_id="client",
        start_ms=3000,
        end_ms=4500,
        call_sid="call_time_4",
    )
    snap2 = engine.process_utterance(statement)
    assert snap2.question_count_60s == 1

    q2 = normalize_generic_transcript(
        text="Can you send me recent sales data?",
        speaker_id="client",
        start_ms=6000,
        end_ms=7500,
        call_sid="call_time_4",
    )
    snap3 = engine.process_utterance(q2)
    assert snap3.question_count_60s == 2


def test_timing_engine_raw_event_retention_and_multi_horizon_querying():
    engine = DeterministicTimingEngine(window_ms=60000)

    utt1 = normalize_generic_transcript(
        text="How long has your house been on the market?",
        speaker_id="salesperson",
        start_ms=1000,
        end_ms=3000,
        call_sid="call_horizon_1",
    )
    engine.process_utterance(utt1)

    utt2 = normalize_generic_transcript(
        text="About three months now.",
        speaker_id="client",
        start_ms=3500,
        end_ms=5000,
        call_sid="call_horizon_1",
    )
    engine.process_utterance(utt2)

    # 1. Verify raw turn events are retained and queryable
    raw_events = engine.get_raw_events()
    assert len(raw_events) >= 4
    event_types = {e.event_type for e in raw_events}
    assert "turn_start" in event_types
    assert "turn_end" in event_types
    assert "speaker_switch" in event_types

    # 2. Verify multi-horizon computation (e.g. 5s horizon vs 60s horizon)
    snap_5s = engine.compute_snapshot_for_window(window_ms=5000, speaker_id="client")
    assert snap_5s.window_ms == 5000
    assert snap_5s.turn_length_words == 4

    snap_full = engine.compute_snapshot_for_window(window_ms=90000, speaker_id="client")
    assert snap_full.window_ms == 90000

