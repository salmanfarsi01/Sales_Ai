from __future__ import annotations

import pytest
from pathlib import Path

from copilot.behavioral_normalization import (
    normalize_generic_transcript,
    NormalizedWord,
    NormalizedUtterance,
)
from copilot.behavioral_timing import DeterministicTimingEngine
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.behavioral_baseline import (
    BaselineAndChangePointEngine,
    ProspectBaselineStore,
)
from copilot.behavioral_evidence import (
    SQLiteEvidenceLogStore,
    MultiWindowAggregator,
    BehavioralEvidenceSnapshot,
    MultiWindowEvidenceFrame,
)


def test_multi_window_aggregation_emits_5_horizons(tmp_path: Path):
    db_path = tmp_path / "test_evidence.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_horizons_test",
        timing_engine=timing_engine,
        store=store,
    )

    # Turn 1: 0 - 5000ms (Client)
    utt1 = normalize_generic_transcript(
        text="Hello, we are looking at our options for selling.",
        speaker_id="client",
        start_ms=1000,
        end_ms=5000,
        call_sid="call_horizons_test",
    )
    t1 = timing_engine.process_utterance(utt1)
    frame1 = aggregator.process_turn(utt1, t1)

    assert frame1.call_sid == "call_horizons_test"
    assert frame1.timestamp_ms == 5000
    assert set(frame1.windows.keys()) == {
        "current_utterance",
        "last_5_10s",
        "last_20_30s",
        "last_60_90s",
        "full_call",
    }

    curr = frame1.windows["current_utterance"]
    assert curr.window_duration_ms == 4000
    assert curr.horizon == "current_utterance"
    assert curr.contributing_utterance_ids == [utt1.utterance_id]
    assert curr.talk_ratio_client == 1.0

    store.close()


def test_multi_horizon_metrics_divergence(tmp_path: Path):
    db_path = tmp_path / "test_divergence.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_divergence",
        timing_engine=timing_engine,
        store=store,
    )

    # 1. Slow, measured conversation during first 40 seconds
    slow_turns = [
        ("We are just checking the market right now.", 1000, 6000, "client"),
        ("I understand completely, happy to walk you through recent comps.", 7000, 14000, "salesperson"),
        ("Yes, that would be helpful for us to see.", 16000, 22000, "client"),
        ("Here are the three most recent sales on your street.", 24000, 32000, "salesperson"),
    ]

    for text, start, end, speaker in slow_turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id=speaker,
            start_ms=start,
            end_ms=end,
            call_sid="call_divergence",
        )
        t_snap = timing_engine.process_utterance(utt)
        aggregator.process_turn(utt, t_snap)

    # 2. Sudden burst of rapid speech from client at second 50 (20 words in 4 seconds -> ~300 WPM)
    fast_utt = normalize_generic_transcript(
        text="Wait what are you saying why is that valuation so much lower than what our neighbor got last month!",
        speaker_id="client",
        start_ms=46000,
        end_ms=50000,
        call_sid="call_divergence",
    )
    t_fast = timing_engine.process_utterance(fast_utt)
    frame = aggregator.process_turn(fast_utt, t_fast)

    w_10s = frame.windows["last_5_10s"]
    w_60s = frame.windows["last_60_90s"]

    # In the last 10s (40s to 50s), the client's rate reflects the rapid burst
    assert w_10s.timing_features.speech_rate_wpm >= 200.0

    # In the last 60s (0 to 50s), the rate is smoothed by the earlier slow turns
    assert w_60s.timing_features.speech_rate_wpm < w_10s.timing_features.speech_rate_wpm

    store.close()


def test_backward_traceability_from_evidence_id(tmp_path: Path):
    db_path = tmp_path / "test_trace.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_trace_test",
        timing_engine=timing_engine,
        store=store,
    )

    utt = normalize_generic_transcript(
        text="Can we discuss the commission breakdown in detail?",
        speaker_id="client",
        start_ms=5000,
        end_ms=8500,
        call_sid="call_trace_test",
    )
    sem = SemanticFeatureSnapshot(
        utterance_id=utt.utterance_id,
        call_sid="call_trace_test",
        speaker_id="client",
        question_type="evaluation",
        recurrence_id="OBJ_COMM_01",
        recurrence_type="same_objection_repeated",
        specificity_score=0.75,
        future_language_score=0.20,
        boundary_score=0.0,
        agreement_score=0.10,
        semantic_confidence=0.95,
    )
    t_snap = timing_engine.process_utterance(utt)
    frame = aggregator.process_turn(utt, t_snap, semantic_snapshot=sem)

    curr_ev_id = frame.windows["current_utterance"].evidence_id

    # Backward-trace query via store
    traced_snaps = store.backward_trace([curr_ev_id])
    assert len(traced_snaps) == 1
    traced = traced_snaps[0]

    assert traced.evidence_id == curr_ev_id
    assert traced.call_sid == "call_trace_test"
    assert traced.contributing_utterance_ids == [utt.utterance_id]
    assert traced.semantic_features is not None
    assert traced.semantic_features.question_type == "evaluation"
    assert traced.semantic_features.recurrence_type == "same_objection_repeated"

    # Query latest frame
    latest = store.get_latest_frame("call_trace_test")
    assert latest is not None
    assert latest.frame_id == frame.frame_id

    store.close()


def test_evidence_confidence_preservation(tmp_path: Path):
    db_path = tmp_path / "test_conf.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_conf_test",
        timing_engine=timing_engine,
        store=store,
    )

    # Utterance with synthetic timing (timing_confidence=0.5)
    utt = normalize_generic_transcript(
        text="A turn with estimated word timestamps.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        call_sid="call_conf_test",
        is_estimated_timing=True,
    )
    sem = SemanticFeatureSnapshot(
        utterance_id=utt.utterance_id,
        call_sid="call_conf_test",
        speaker_id="client",
        question_type="clarifying",
        recurrence_type="none",
        semantic_confidence=0.95,
    )
    t_snap = timing_engine.process_utterance(utt)
    assert t_snap.timing_confidence == 0.5

    frame = aggregator.process_turn(utt, t_snap, semantic_snapshot=sem)

    # Evidence confidence must be min(timing_conf, sem_conf) = 0.5, never claiming 0.95
    for horizon, snap in frame.windows.items():
        assert snap.evidence_confidence <= 0.5

    store.close()


def test_sqlite_time_range_query(tmp_path: Path):
    db_path = tmp_path / "test_range.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_range_test",
        timing_engine=timing_engine,
        store=store,
    )

    for i in range(5):
        start = i * 5000
        end = start + 3000
        utt = normalize_generic_transcript(
            text=f"Turn sequence number {i}.",
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_range_test",
        )
        t = timing_engine.process_utterance(utt)
        aggregator.process_turn(utt, t)

    # Query time range covering only the middle turns (6000ms to 16000ms)
    snaps = store.query_time_range(
        call_sid="call_range_test",
        start_ms=6000,
        end_ms=16000,
        horizon="current_utterance",
    )
    # Turns at 8000ms and 13000ms fall inside this range
    assert len(snaps) == 2
    assert snaps[0].timestamp_ms == 8000
    assert snaps[1].timestamp_ms == 13000

    store.close()


def test_evidence_confidence_weakest_link_minimum_rule(tmp_path: Path):
    db_path = tmp_path / "test_min_rule.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_weakest_link",
        timing_engine=timing_engine,
        store=store,
    )

    # Utterance with synthetic timing (timing_confidence=0.50)
    utt = normalize_generic_transcript(
        text="Yes we definitely want to review this agreement tomorrow.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        call_sid="call_weakest_link",
        is_estimated_timing=True,
    )
    # High-confidence LLM semantic features (0.95)
    sem = SemanticFeatureSnapshot(
        utterance_id=utt.utterance_id,
        call_sid="call_weakest_link",
        speaker_id="client",
        question_type="transactional",
        recurrence_type="none",
        specificity_score=0.95,
        future_language_score=0.95,
        boundary_score=0.0,
        agreement_score=0.95,
        semantic_confidence=0.95,
    )
    t_snap = timing_engine.process_utterance(utt)
    assert t_snap.timing_confidence == 0.5

    frame = aggregator.process_turn(utt, t_snap, semantic_snapshot=sem)

    # Must be strictly 0.50 (the minimum / weakest link), NEVER diluted by averaging to ~0.86
    curr_snap = frame.windows["current_utterance"]
    assert curr_snap.evidence_confidence == 0.50

    w_10s = frame.windows["last_5_10s"]
    assert w_10s.evidence_confidence == 0.50

    store.close()

