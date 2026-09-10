from __future__ import annotations

import pytest
from pathlib import Path

from copilot.behavioral_normalization import (
    normalize_generic_transcript,
    NormalizedUtterance,
)
from copilot.behavioral_timing import (
    DeterministicTimingEngine,
    TimingFeatureSnapshot,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.behavioral_baseline import (
    BaselineAndChangePointEngine,
    ProspectBaselineStore,
    ChangePointEvent,
    FeatureDeviation,
)
from copilot.behavioral_evidence import (
    SQLiteEvidenceLogStore,
    MultiWindowAggregator,
    BehavioralEvidenceSnapshot,
    MultiWindowEvidenceFrame,
)
from copilot.behavioral_inference import (
    DownstreamInferenceEngine,
    DownstreamInferenceState,
    NullAcousticProvider,
)


def test_spec07_acceptance_text_only_produces_no_tone_claims(tmp_path: Path):
    """Spec 07 Acceptance Test #1:
    Text-only input produces downstream scores with acoustic_evidence = 'unavailable'
    and no tone claims made.
    """
    db_path = tmp_path / "test_inference_spec07.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_ac1",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine(acoustic_provider=NullAcousticProvider())

    utt = normalize_generic_transcript(
        text="Yes, that pricing seems reasonable and within our budget.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4500,
        call_sid="call_spec07_ac1",
    )
    t_snap = timing_engine.process_utterance(utt)
    frame = aggregator.process_turn(utt, t_snap)

    inference = inference_engine.compute_inference(
        call_sid="call_spec07_ac1",
        current_frame=frame,
    )

    # Spec 07 explicit requirements
    assert inference.acoustic_evidence == "unavailable"
    assert inference.emotion.tone_claim_made is False

    # Verify observable signals do not make acoustic pitch/tone claims
    for sig in inference.emotion.observable_signals:
        sig_lower = sig.lower()
        assert "pitch" not in sig_lower
        assert "prosody" not in sig_lower
        assert "acoustic" not in sig_lower
        assert "decibel" not in sig_lower

    store.close()


def test_weakest_link_confidence_inheritance(tmp_path: Path):
    """Verifies that downstream dimension confidence strictly inherits the minimum
    contributing evidence confidence without any dilution or averaging.
    """
    db_path = tmp_path / "test_weakest_link.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_weakest_link",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    # Synthetic timing utterance: estimated timestamps -> timing_confidence = 0.50
    utt = normalize_generic_transcript(
        text="We are currently evaluating three vendors.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        call_sid="call_weakest_link",
        is_estimated_timing=True,
    )
    assert utt.is_estimated_timing is True

    t_snap = timing_engine.process_utterance(utt)
    assert t_snap.timing_confidence == 0.50

    # High-confidence LLM semantic features
    sem = SemanticFeatureSnapshot(
        utterance_id=utt.utterance_id,
        call_sid="call_weakest_link",
        speaker_id="client",
        question_type="evaluation",
        specificity_score=0.85,
        future_language_score=0.40,
        boundary_score=0.0,
        agreement_score=0.70,
        semantic_confidence=0.95,
    )

    frame = aggregator.process_turn(utt, t_snap, semantic_snapshot=sem)

    inference = inference_engine.compute_inference(
        call_sid="call_weakest_link",
        current_frame=frame,
    )

    # Pacing consumes last_5_10s and last_20_30s which have 0.50 confidence
    assert inference.pacing.confidence == 0.50

    # Overall confidence must be 0.50 (strict minimum, NEVER an average like 0.725)
    assert inference.overall_confidence == 0.50
    assert inference.overall_confidence < 0.60

    store.close()


def test_hard_boundary_override_forces_zero_readiness(tmp_path: Path):
    """Verifies that boundary_score > 0.0 (DNC / stop-contact) overrides Readiness to 0.0
    and heavily penalizes Trust.
    """
    db_path = tmp_path / "test_boundary.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_boundary",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    utt = normalize_generic_transcript(
        text="Please stop calling me and take me off your list immediately.",
        speaker_id="client",
        start_ms=1000,
        end_ms=5000,
        call_sid="call_boundary",
    )
    t_snap = timing_engine.process_utterance(utt)
    sem = SemanticFeatureSnapshot(
        utterance_id=utt.utterance_id,
        call_sid="call_boundary",
        speaker_id="client",
        recurrence_type="boundary_repeated",
        boundary_score=1.0,
        semantic_confidence=0.95,
    )

    frame = aggregator.process_turn(utt, t_snap, semantic_snapshot=sem)
    inference = inference_engine.compute_inference(
        call_sid="call_boundary",
        current_frame=frame,
    )

    # Hard boundary zero override on readiness
    assert inference.readiness.score == 0.0
    assert any("hard boundary" in d.lower() for d in inference.readiness.drivers)

    # Trust is penalized
    assert inference.trust.score <= 0.20
    assert any("boundary" in d.lower() for d in inference.trust.drivers)

    # Emotion reflects negative valence and heightened tension
    assert inference.emotion.expressed_valence <= -0.50
    assert inference.emotion.tension_level >= 0.50

    store.close()


def test_scheduling_detail_repeated_boosts_readiness(tmp_path: Path):
    """Verifies that scheduling_detail_repeated pattern significantly boosts closing readiness."""
    db_path = tmp_path / "test_scheduling.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_sched",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    utt = normalize_generic_transcript(
        text="Let's lock in next Tuesday at 2 PM for the team demonstration.",
        speaker_id="client",
        start_ms=10000,
        end_ms=15000,
        call_sid="call_sched",
    )
    t_snap = timing_engine.process_utterance(utt)
    sem = SemanticFeatureSnapshot(
        utterance_id=utt.utterance_id,
        call_sid="call_sched",
        speaker_id="client",
        question_type="transactional",
        recurrence_type="scheduling_detail_repeated",
        future_language_score=0.85,
        agreement_score=0.80,
        semantic_confidence=0.95,
    )

    frame = aggregator.process_turn(utt, t_snap, semantic_snapshot=sem)
    inference = inference_engine.compute_inference(
        call_sid="call_sched",
        current_frame=frame,
    )

    # Readiness score should be elevated (base 0.30 + 0.35 + 0.25 + 0.20 -> 1.0 clamped)
    assert inference.readiness.score >= 0.85
    assert any("scheduling" in d.lower() for d in inference.readiness.drivers)
    assert any("future commitment" in d.lower() for d in inference.readiness.drivers)

    store.close()


def test_slow_speaker_invariance_preserves_high_pacing(tmp_path: Path):
    """Verifies Spec 07 slow-speaker invariance: a naturally slow speaker whose rate matches
    their established baseline (z ≈ 0.0) maintains a healthy pacing score >= 0.75.
    """
    db_path = tmp_path / "test_slow_speaker.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    inference_engine = DownstreamInferenceEngine()

    # Pre-create a baseline where 100 WPM is normal for this prospect
    base_store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    base_engine = BaselineAndChangePointEngine(
        call_sid="call_slow",
        prospect_id="prospect_slow_1",
        store=base_store,
    )

    aggregator = MultiWindowAggregator(
        call_sid="call_slow",
        timing_engine=timing_engine,
        baseline_engine=base_engine,
        store=store,
    )

    # Ingest clean turns at ~100 WPM across 70 seconds to satisfy the 60-90s baseline window
    turns = [
        ("We are carefully reviewing all our internal processes this quarter.", 1000, 7000),    # 10 words, 6s -> 100 WPM
        ("Every department has specific security criteria we must meet.", 10000, 16000),       # 9 words, 6s -> 90 WPM
        ("Let us proceed methodically through each requirement one by one.", 25000, 31000),    # 10 words, 6s -> 100 WPM
        ("The compliance team requires full audit documentation as well.", 65000, 71000),      # 9 words, 6s -> 90 WPM
    ]

    last_frame = None
    for text, start, end in turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_slow",
        )
        t_snap = timing_engine.process_utterance(utt)
        cps = base_engine.update_with_utterance(utt, t_snap)
        last_frame = aggregator.process_turn(utt, t_snap, new_change_points=cps)

    assert last_frame is not None
    inference = inference_engine.compute_inference(
        call_sid="call_slow",
        current_frame=last_frame,
    )

    # Even though absolute rate is only ~95 WPM, pacing score remains high because z is near 0
    assert inference.pacing.score >= 0.75
    assert any("aligned with baseline" in d.lower() for d in inference.pacing.drivers)

    store.close()


def test_downstream_backward_traceability_to_sqlite(tmp_path: Path):
    """Verifies that all contributing_evidence_ids reported by DownstreamInferenceState
    can be queried and recovered from SQLiteEvidenceLogStore.
    """
    db_path = tmp_path / "test_traceability.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_trace_integration",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    utt = normalize_generic_transcript(
        text="What is the setup time required for initial deployment?",
        speaker_id="client",
        start_ms=2000,
        end_ms=5000,
        call_sid="call_trace_integration",
    )
    t_snap = timing_engine.process_utterance(utt)
    frame = aggregator.process_turn(utt, t_snap)

    inference = inference_engine.compute_inference(
        call_sid="call_trace_integration",
        current_frame=frame,
    )

    assert len(inference.contributing_evidence_ids) > 0

    # Trace all evidence IDs via SQLite store
    traced_snaps = store.backward_trace(inference.contributing_evidence_ids)
    traced_ids = {s.evidence_id for s in traced_snaps}

    for eid in inference.contributing_evidence_ids:
        assert eid in traced_ids

    store.close()


def test_baseline_source_tracking_in_inference(tmp_path: Path):
    """Verifies that change-point baseline sources are tracked into DownstreamInferenceState."""
    db_path = tmp_path / "test_sources.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    inference_engine = DownstreamInferenceEngine()

    base_store = ProspectBaselineStore(store_path=tmp_path / "sources_baseline.json")
    # Seed historical profile in store before initializing base_engine
    from copilot.behavioral_baseline import BaselineProfile, SpeakerBaselineStats
    hist_profile = BaselineProfile(
        prospect_id="prospect_historical",
        speech_rate_wpm=SpeakerBaselineStats(mean=140.0, stddev=15.0, sample_count=20),
        avg_pause_duration_ms=SpeakerBaselineStats(mean=300.0, stddev=50.0, sample_count=20),
        response_latency_ms=SpeakerBaselineStats(mean=600.0, stddev=100.0, sample_count=20),
        turn_length_words=SpeakerBaselineStats(mean=15.0, stddev=4.0, sample_count=20),
        call_count=3,
        updated_at_ms=1000,
    )
    base_store.save_baseline("prospect_historical", hist_profile)

    base_engine = BaselineAndChangePointEngine(
        call_sid="call_sources",
        prospect_id="prospect_historical",
        store=base_store,
    )

    aggregator = MultiWindowAggregator(
        call_sid="call_sources",
        timing_engine=timing_engine,
        baseline_engine=base_engine,
        store=store,
    )

    # Ingest turn with significant surge during early call (evaluated against historical baseline)
    utt = normalize_generic_transcript(
        text="Wait wait wait hold on why is this different from what we discussed last week on Tuesday?!",
        speaker_id="client",
        start_ms=5000,
        end_ms=8000,
        call_sid="call_sources",
    )
    t_snap = timing_engine.process_utterance(utt)
    cps = base_engine.update_with_utterance(utt, t_snap)

    assert len(cps) > 0
    assert any(cp.baseline_source == "historical_only" for cp in cps)

    frame = aggregator.process_turn(utt, t_snap, new_change_points=cps)
    inference = inference_engine.compute_inference(
        call_sid="call_sources",
        current_frame=frame,
    )

    assert "historical_only" in inference.baseline_sources

    store.close()
