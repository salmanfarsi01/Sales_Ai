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


def test_semantic_aggregation_preserves_objections_across_horizons(tmp_path: Path):
    """
    Verifies that semantic evidence aggregates across multi-window horizons.
    An objection raised in an earlier turn within last_20_30s and last_60_90s
    must NOT be erased from those windows when a new neutral 1-word turn occurs.
    """
    db_path = tmp_path / "test_sem_agg.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_sem_agg",
        timing_engine=timing_engine,
        store=store,
    )

    # Turn 1 at 5s: Client raises a strong commission objection with high specificity
    utt1 = normalize_generic_transcript(
        text="Your 6% commission rate is way too high compared to Redfin.",
        speaker_id="client",
        start_ms=2000,
        end_ms=6000,
        call_sid="call_sem_agg",
    )
    sem1 = SemanticFeatureSnapshot(
        utterance_id=utt1.utterance_id,
        call_sid="call_sem_agg",
        speaker_id="client",
        question_type="evaluation",
        recurrence_id="OBJ_COMMISSION_01",
        recurrence_type="same_objection_repeated",
        recurrence_count=1,
        specificity_score=0.85,
        future_language_score=0.10,
        boundary_score=0.0,
        agreement_score=0.0,
        semantic_confidence=0.95,
    )
    t1 = timing_engine.process_utterance(utt1)
    aggregator.process_turn(utt1, t1, semantic_snapshot=sem1)

    # Turn 2 at 12s: Salesperson responds
    utt2 = normalize_generic_transcript(
        text="I understand your concern about rates.",
        speaker_id="salesperson",
        start_ms=8000,
        end_ms=12000,
        call_sid="call_sem_agg",
    )
    t2 = timing_engine.process_utterance(utt2)
    aggregator.process_turn(utt2, t2)

    # Turn 3 at 20s: Client gives a brief breath continuation: "case."
    utt3 = normalize_generic_transcript(
        text="case.",
        speaker_id="client",
        start_ms=18000,
        end_ms=20000,
        call_sid="call_sem_agg",
    )
    sem3 = SemanticFeatureSnapshot(
        utterance_id=utt3.utterance_id,
        call_sid="call_sem_agg",
        speaker_id="client",
        question_type="none",
        recurrence_type="none",
        recurrence_count=0,
        specificity_score=0.0,
        future_language_score=0.0,
        boundary_score=0.0,
        agreement_score=0.0,
        semantic_confidence=0.90,
    )
    t3 = timing_engine.process_utterance(utt3)
    frame3 = aggregator.process_turn(utt3, t3, semantic_snapshot=sem3)

    # 1. Immediate current_utterance reflects Turn 3's neutral state
    w_curr = frame3.windows["current_utterance"]
    assert w_curr.semantic_features is not None
    assert w_curr.semantic_features.recurrence_type == "none"
    assert w_curr.semantic_features.specificity_score == 0.0

    # 2. last_20_30s (spans 0 - 20s) covers Turn 1!
    # It MUST retain the objection and peak specificity from Turn 1!
    w_30s = frame3.windows["last_20_30s"]
    assert w_30s.semantic_features is not None
    assert w_30s.semantic_features.recurrence_type == "same_objection_repeated"
    assert w_30s.semantic_features.specificity_score == 0.85

    # 3. last_60_90s also covers Turn 1
    w_60s = frame3.windows["last_60_90s"]
    assert w_60s.semantic_features is not None
    assert w_60s.semantic_features.recurrence_type == "same_objection_repeated"
    assert w_60s.semantic_features.specificity_score == 0.85

    store.close()


def test_semantic_aggregation_recurrence_count_coupled_to_active_event(tmp_path: Path):
    """
    Verifies that when a newer recurrence event (e.g. positive_echo, count=1) follows
    an older recurrence event (e.g. same_objection_repeated, count=3) in the same window,
    the aggregated snapshot couples recurrence_count strictly to the active event (count=1, NOT 3),
    and extraction_mode honestly reports 'mixed'.
    """
    db_path = tmp_path / "test_rec_coupling.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_rec_coupling",
        timing_engine=timing_engine,
        store=store,
    )

    # Turn 1 at 4s: Older 3rd-occurrence objection via LLM
    u1 = normalize_generic_transcript("Your fee is too high.", "client", 1000, 4000, "call_rec_coupling")
    s1 = SemanticFeatureSnapshot(
        utterance_id=u1.utterance_id,
        call_sid="call_rec_coupling",
        speaker_id="client",
        extraction_mode="llm",
        recurrence_type="same_objection_repeated",
        recurrence_count=3,
        specificity_score=0.80,
        specificity_confidence=0.95,
    )
    t1 = timing_engine.process_utterance(u1)
    aggregator.process_turn(u1, t1, semantic_snapshot=s1)

    # Turn 2 at 10s: Salesperson speaks
    u2 = normalize_generic_transcript("Let's look at net proceeds.", "salesperson", 6000, 10000, "call_rec_coupling")
    t2 = timing_engine.process_utterance(u2)
    aggregator.process_turn(u2, t2)

    # Turn 3 at 16s: Newer 1st-occurrence positive_echo via heuristic_offline
    u3 = normalize_generic_transcript("Net proceeds make sense.", "client", 13000, 16000, "call_rec_coupling")
    s3 = SemanticFeatureSnapshot(
        utterance_id=u3.utterance_id,
        call_sid="call_rec_coupling",
        speaker_id="client",
        extraction_mode="heuristic_offline",
        recurrence_type="positive_echo",
        recurrence_count=1,
        specificity_score=0.0,
        specificity_confidence=0.50,
    )
    t3 = timing_engine.process_utterance(u3)
    frame = aggregator.process_turn(u3, t3, semantic_snapshot=s3)

    w_30s = frame.windows["last_20_30s"]
    assert w_30s.semantic_features is not None
    # Recurrence type is the latest active event
    assert w_30s.semantic_features.recurrence_type == "positive_echo"
    # Recurrence count MUST be 1 (from s3), NEVER 3 (from s1)!
    assert w_30s.semantic_features.recurrence_count == 1
    # Mode honestly reports 'mixed' because s1 was 'llm' and s3 was 'heuristic_offline'
    assert w_30s.semantic_features.extraction_mode == "mixed"
    # Peak specificity is 0.80 and its confidence is 0.95 (inherited from s1)
    assert w_30s.semantic_features.specificity_score == 0.80
    assert w_30s.semantic_features.specificity_confidence == 0.95

    store.close()


def test_semantic_aggregation_heuristic_mode_peak_severity_precedence(tmp_path):
    """
    Verifies that when 0% of contributing snapshots in a window used LLM:
    1. 'heuristic_error' takes precedence over 'heuristic_timeout' and 'heuristic_offline'.
    2. 'heuristic_timeout' takes precedence over 'heuristic_offline'.
    3. An earlier extraction failure or timeout is NEVER masked by a subsequent clean offline bypass.
    """
    store = SQLiteEvidenceLogStore(tmp_path / "test_modes.db")
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_sev_test",
        timing_engine=timing_engine,
        store=store,
    )

    # Turn 1 at 2s: Extraction threw an exception (heuristic_error)
    u1 = normalize_generic_transcript("What is your fee?", "client", 1000, 2000, "call_sev_test")
    s1 = SemanticFeatureSnapshot(
        utterance_id=u1.utterance_id,
        call_sid="call_sev_test",
        speaker_id="client",
        extraction_mode="heuristic_error",
        question_type="transactional",
        semantic_confidence=0.65,
    )
    t1 = timing_engine.process_utterance(u1)
    aggregator.process_turn(u1, t1, semantic_snapshot=s1)

    # Turn 2 at 5s: Clean offline bypass for 1-word nod (heuristic_offline)
    u2 = normalize_generic_transcript("Right.", "client", 4000, 5000, "call_sev_test")
    s2 = SemanticFeatureSnapshot(
        utterance_id=u2.utterance_id,
        call_sid="call_sev_test",
        speaker_id="client",
        extraction_mode="heuristic_offline",
        agreement_score=0.30,
        specificity_confidence=0.50,
        future_language_confidence=0.50,
        agreement_confidence=0.65,
        semantic_confidence=0.50,
    )
    t2 = timing_engine.process_utterance(u2)
    frame = aggregator.process_turn(u2, t2, semantic_snapshot=s2)

    w_10s = frame.windows["last_5_10s"]
    assert w_10s.semantic_features is not None
    # Peak severity rule: heuristic_error must NOT be masked by subsequent heuristic_offline
    assert w_10s.semantic_features.extraction_mode == "heuristic_error"
    # min confidence across the window is 0.50
    assert w_10s.semantic_features.semantic_confidence == 0.50

    # Turn 3 at 8s: Timeout fallback
    u3 = normalize_generic_transcript("Can we meet next week?", "client", 6000, 8000, "call_sev_test")
    s3 = SemanticFeatureSnapshot(
        utterance_id=u3.utterance_id,
        call_sid="call_sev_test",
        speaker_id="client",
        extraction_mode="heuristic_timeout",
        semantic_confidence=0.65,
    )
    # New aggregator instance to test timeout > offline without error
    timing_engine2 = DeterministicTimingEngine()
    aggregator2 = MultiWindowAggregator(
        call_sid="call_sev_test2",
        timing_engine=timing_engine2,
        store=store,
    )
    t3 = timing_engine2.process_utterance(u3)
    aggregator2.process_turn(u3, t3, semantic_snapshot=s3)

    # Turn 4: Offline bypass
    u4 = normalize_generic_transcript("Yep.", "client", 9000, 10000, "call_sev_test2")
    s4 = SemanticFeatureSnapshot(
        utterance_id=u4.utterance_id,
        call_sid="call_sev_test2",
        speaker_id="client",
        extraction_mode="heuristic_offline",
        semantic_confidence=0.50,
    )
    t4 = timing_engine2.process_utterance(u4)
    frame2 = aggregator2.process_turn(u4, t4, semantic_snapshot=s4)

    w2_10s = frame2.windows["last_5_10s"]
    # Peak severity rule: heuristic_timeout takes precedence over heuristic_offline
    assert w2_10s.semantic_features.extraction_mode == "heuristic_timeout"

    store.close()


def test_window_semantic_confidence_matches_weakest_link_of_constituent_features(tmp_path):
    """
    Verifies that a window's aggregated semantic_confidence is mathematically consistent with
    the weakest link among the actual 6 feature confidences in that window snapshot:
    - If a feature has a positively measured peak disclosure (e.g. specificity 0.85 from LLM @ 0.95 conf),
      its confidence is 0.95.
    - A superseded 0.0 unmeasured value from an earlier bypass turn does NOT drag down
      window semantic_confidence below the confidence of the features actually present in the snapshot.
    """
    store = SQLiteEvidenceLogStore(tmp_path / "test_conf_cons.db")
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_conf_test",
        timing_engine=timing_engine,
        store=store,
    )

    # Turn 1: 1-word bypass turn with 0.50 unmeasured specificity/future
    u1 = normalize_generic_transcript("But", "client", 1000, 1500, "call_conf_test")
    s1 = SemanticFeatureSnapshot(
        utterance_id=u1.utterance_id,
        call_sid="call_conf_test",
        speaker_id="client",
        extraction_mode="heuristic_offline",
        specificity_score=0.0,
        specificity_confidence=0.50,
        future_language_score=0.0,
        future_language_confidence=0.50,
        agreement_score=0.0,
        agreement_confidence=0.65,
        question_type_confidence=0.70,
        recurrence_confidence=0.80,
        boundary_confidence=0.95,
        semantic_confidence=0.50,
    )
    t1 = timing_engine.process_utterance(u1)
    aggregator.process_turn(u1, t1, semantic_snapshot=s1)

    # Turn 2: Rich LLM disclosure with 0.95 conf across features
    u2 = normalize_generic_transcript("Our lease renews in five months.", "client", 2000, 4500, "call_conf_test")
    s2 = SemanticFeatureSnapshot(
        utterance_id=u2.utterance_id,
        call_sid="call_conf_test",
        speaker_id="client",
        extraction_mode="llm",
        specificity_score=0.85,
        specificity_confidence=0.95,
        future_language_score=0.90,
        future_language_confidence=0.95,
        agreement_score=0.10,
        agreement_confidence=0.95,
        question_type_confidence=0.95,
        recurrence_confidence=0.95,
        boundary_confidence=0.95,
        semantic_confidence=0.95,
    )
    t2 = timing_engine.process_utterance(u2)
    frame = aggregator.process_turn(u2, t2, semantic_snapshot=s2)

    w = frame.windows["last_5_10s"].semantic_features
    assert w is not None
    # Peak disclosure scores were adopted from Turn 2
    assert w.specificity_score == 0.85
    assert w.specificity_confidence == 0.95
    assert w.future_language_score == 0.90
    assert w.future_language_confidence == 0.95

    # Overall semantic_confidence is exactly min() of the window's 6 constituent feature confidences
    expected_min = min([
        w.boundary_confidence,
        w.recurrence_confidence,
        w.question_type_confidence,
        w.specificity_confidence,
        w.future_language_confidence,
        w.agreement_confidence,
    ])
    assert w.semantic_confidence == expected_min
    # And specifically, semantic_confidence cannot be 0.50 when all features are >= 0.70
    assert w.semantic_confidence >= 0.70
    assert w.semantic_confidence > 0.50

    store.close()


