from __future__ import annotations

import time
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
from copilot.behavioral_semantic import (
    SemanticFeatureEngine,
    SemanticFeatureSnapshot,
)
from copilot.behavioral_baseline import (
    BaselineAndChangePointEngine,
    ProspectBaselineStore,
    ChangePointEvent,
)
from copilot.behavioral_evidence import (
    SQLiteEvidenceLogStore,
    MultiWindowAggregator,
    MultiWindowEvidenceFrame,
)
from copilot.behavioral_inference import (
    DownstreamInferenceEngine,
    DownstreamInferenceState,
    NullAcousticProvider,
    DEFAULT_INFERENCE_CONFIG,
)


def test_spec07_test1_text_only_no_tone_claims(tmp_path: Path):
    """Spec 07 Acceptance Test #1:
    Text-only input produces downstream scores with acoustic_evidence = 'unavailable'
    and no tone claims made.
    """
    db_path = tmp_path / "test1.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_t1",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine(acoustic_provider=NullAcousticProvider())

    turns = [
        ("Good morning, thanks for getting back to me.", 1000, 4000, "client"),
        ("Happy to connect! How can we help today?", 5000, 8000, "salesperson"),
        ("We are reviewing software vendors to optimize our call center pipeline.", 9000, 14000, "client"),
    ]

    last_frame = None
    for text, start, end, speaker in turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id=speaker,
            start_ms=start,
            end_ms=end,
            call_sid="call_spec07_t1",
        )
        t_snap = timing_engine.process_utterance(utt)
        last_frame = aggregator.process_turn(utt, t_snap)

    assert last_frame is not None
    inference = inference_engine.compute_inference(
        call_sid="call_spec07_t1",
        current_frame=last_frame,
    )

    # 1. Acoustic sentinel is strictly "unavailable"
    assert inference.acoustic_evidence == "unavailable"

    # 2. Emotion tone claim is False
    assert inference.emotion.tone_claim_made is False

    # 3. No acoustic or prosodic terms appear anywhere in drivers or observable signals
    forbidden_terms = ("pitch", "prosody", "decibel", "acoustic", "loudness", "harmonic")
    all_text = " ".join(
        inference.emotion.observable_signals
        + inference.trust.drivers
        + inference.pacing.drivers
        + inference.engagement.drivers
        + inference.momentum.drivers
        + inference.readiness.drivers
    ).lower()

    for term in forbidden_terms:
        assert term not in all_text, f"Acoustic term '{term}' found in text-only inference output"

    store.close()


def test_spec07_test2_pricing_slowdown_stored_as_objective_evidence(tmp_path: Path):
    """Spec 07 Acceptance Test #2:
    Slowdown after pricing objection is stored as change-point evidence with semantic
    context, and NOT auto-labeled as 'fear' or 'anxiety'.
    """
    db_path = tmp_path / "test2.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    base_store = ProspectBaselineStore(store_path=tmp_path / "baselines_t2.json")
    base_engine = BaselineAndChangePointEngine(
        call_sid="call_spec07_t2",
        prospect_id="prospect_t2",
        store=base_store,
    )
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_t2",
        timing_engine=timing_engine,
        baseline_engine=base_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    # Step 1: Establish clean speech baseline at normal conversational pace (~150 WPM) across 70s
    setup_turns = [
        ("We are looking into upgrading our existing customer support platform this quarter.", 1000, 6000),      # 12 words, 5s -> 144 WPM
        ("Our team has about twenty agents handling inquiries every day from multiple channels.", 10000, 15000), # 13 words, 5s -> 156 WPM
        ("We want to make sure the software connects with our CRM seamlessly without issues.", 30000, 35000),     # 13 words, 5s -> 156 WPM
        ("The main priority is reliability and ease of use for all our representatives.", 65000, 70000),         # 12 words, 5s -> 144 WPM
    ]

    for text, start, end in setup_turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_spec07_t2",
        )
        t_snap = timing_engine.process_utterance(utt)
        cps = base_engine.update_with_utterance(utt, t_snap)
        aggregator.process_turn(utt, t_snap, new_change_points=cps)

    assert base_engine.is_intra_call_locked is True

    # Step 2: Pricing inquiry followed by sudden deceleration
    pricing_utt = normalize_generic_transcript(
        text="What is the total annual licensing fee and implementation cost for this deployment?",
        speaker_id="client",
        start_ms=75000,
        end_ms=80000,
        call_sid="call_spec07_t2",
    )
    t_pricing = timing_engine.process_utterance(pricing_utt)
    sem_pricing = SemanticFeatureSnapshot(
        utterance_id=pricing_utt.utterance_id,
        call_sid="call_spec07_t2",
        speaker_id="client",
        question_type="evaluation",
        recurrence_type="none",
        specificity_score=0.85,
        semantic_confidence=0.95,
    )
    cps_pricing = base_engine.update_with_utterance(pricing_utt, t_pricing, semantic_snapshot=sem_pricing)
    aggregator.process_turn(pricing_utt, t_pricing, semantic_snapshot=sem_pricing, new_change_points=cps_pricing)

    # Step 3: Dramatic slowdown (client speaks only 4 words in 6 seconds -> 40 WPM, z <= -2.0)
    slow_utt = normalize_generic_transcript(
        text="That is quite high.",
        speaker_id="client",
        start_ms=83000,
        end_ms=89000,
        call_sid="call_spec07_t2",
    )
    timing_engine.process_utterance(slow_utt)
    t_slow = timing_engine.compute_snapshot_for_window(window_ms=10000, speaker_id="client")
    cps_slow = base_engine.update_with_utterance(slow_utt, t_slow)

    # Change-point event must be detected for speech deceleration
    assert len(cps_slow) > 0
    rate_cp = next((cp for cp in cps_slow if cp.feature_name == "speech_rate_wpm"), None)
    assert rate_cp is not None
    assert rate_cp.direction == "drop"
    assert rate_cp.z_score <= -2.0

    frame_slow = aggregator.process_turn(slow_utt, t_slow, new_change_points=cps_slow)
    inference = inference_engine.compute_inference(
        call_sid="call_spec07_t2",
        current_frame=frame_slow,
    )

    # Verify objective recording: stored as speech deceleration evidence
    assert any("deceleration" in sig.lower() for sig in inference.emotion.observable_signals)

    # Crucial Spec 07 check: verify emotional state is NOT auto-labeled as "fear", "anxiety", or "scared"
    all_signals_str = " ".join(inference.emotion.observable_signals).lower()
    assert "fear" not in all_signals_str
    assert "anxiety" not in all_signals_str
    assert "scared" not in all_signals_str

    store.close()


def test_spec07_test3_repeated_objection_after_failed_reframe(tmp_path: Path):
    """Spec 07 Acceptance Test #3:
    Repeated objection after failed reframe raises recurrence confidence,
    penalizes readiness and trust, elevates tension, and records failed reframe telemetry.
    """
    db_path = tmp_path / "test3.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_t3",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    # Turn 1: Client raises implementation objection
    utt1 = normalize_generic_transcript(
        text="We simply do not have the internal engineering bandwidth to integrate another tool right now.",
        speaker_id="client",
        start_ms=1000,
        end_ms=6000,
        call_sid="call_spec07_t3",
    )
    t1 = timing_engine.process_utterance(utt1)
    sem1 = SemanticFeatureSnapshot(
        utterance_id=utt1.utterance_id,
        call_sid="call_spec07_t3",
        speaker_id="client",
        recurrence_id="OBJ_BANDWIDTH",
        recurrence_type="none",
        recurrence_count=0,
        agreement_score=0.10,
        semantic_confidence=0.95,
    )
    frame1 = aggregator.process_turn(utt1, t1, semantic_snapshot=sem1)
    inf1 = inference_engine.compute_inference(call_sid="call_spec07_t3", current_frame=frame1)

    # Turn 2: Rep attempts reframe
    utt2 = normalize_generic_transcript(
        text="I completely understand. Our dedicated white-glove onboarding team manages 90% of the integration for you.",
        speaker_id="salesperson",
        start_ms=7000,
        end_ms=13000,
        call_sid="call_spec07_t3",
    )
    t2 = timing_engine.process_utterance(utt2)
    aggregator.process_turn(utt2, t2)

    # Turn 3: Client repeats the concern despite the reframe attempt
    utt3 = normalize_generic_transcript(
        text="Even if your team helps, our internal IT security team still has to review every single API endpoint.",
        speaker_id="client",
        start_ms=14000,
        end_ms=20000,
        call_sid="call_spec07_t3",
    )
    t3 = timing_engine.process_utterance(utt3)
    sem3 = SemanticFeatureSnapshot(
        utterance_id=utt3.utterance_id,
        call_sid="call_spec07_t3",
        speaker_id="client",
        recurrence_id="OBJ_BANDWIDTH",
        recurrence_type="concern_after_failed_reframe",
        recurrence_count=1,
        agreement_score=0.10,
        semantic_confidence=0.95,
    )
    frame3 = aggregator.process_turn(utt3, t3, semantic_snapshot=sem3)
    inf3 = inference_engine.compute_inference(call_sid="call_spec07_t3", current_frame=frame3)

    # Assertions for failed reframe adaptation:
    # 1. Tension level elevated on failed reframe
    assert inf3.emotion.tension_level > inf1.emotion.tension_level
    assert any("concern_after_failed_reframe" in sig for sig in inf3.emotion.observable_signals)

    # 2. Trust and Readiness suppressed by unresolved objection
    assert any("unresolved objection" in d.lower() for d in inf3.trust.drivers)
    assert any("unresolved core objection" in d.lower() for d in inf3.readiness.drivers)
    assert inf3.readiness.score <= 0.20

    store.close()


def test_spec07_test4_slow_speaker_baseline_invariance(tmp_path: Path):
    """Spec 07 Acceptance Test #4:
    A naturally slow speaker whose rate matches their established baseline
    maintains a healthy pacing score >= 0.75 without being mislabeled as disengaged.
    """
    db_path = tmp_path / "test4.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    base_store = ProspectBaselineStore(store_path=tmp_path / "baselines_t4.json")
    base_engine = BaselineAndChangePointEngine(
        call_sid="call_spec07_t4",
        prospect_id="prospect_t4",
        store=base_store,
    )
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_t4",
        timing_engine=timing_engine,
        baseline_engine=base_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    # Client consistently speaks at a measured 90-100 WPM throughout the entire call
    slow_turns = [
        ("We are carefully considering all the proposals we received from various vendors.", 1000, 7000),      # 11 words, 6s -> 110 WPM
        ("Our committee meets once every two weeks to discuss organizational priorities.", 10000, 16000),      # 10 words, 6s -> 100 WPM
        ("Each department head has specific requirements that must be thoroughly validated.", 35000, 41000),   # 10 words, 6s -> 100 WPM
        ("We will review your technical specifications at our upcoming executive meeting.", 65000, 71000),      # 10 words, 6s -> 100 WPM
    ]

    last_frame = None
    for text, start, end in slow_turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_spec07_t4",
        )
        t_snap = timing_engine.process_utterance(utt)
        cps = base_engine.update_with_utterance(utt, t_snap)
        last_frame = aggregator.process_turn(utt, t_snap, new_change_points=cps)

    assert last_frame is not None
    assert base_engine.is_intra_call_locked is True

    inference = inference_engine.compute_inference(
        call_sid="call_spec07_t4",
        current_frame=last_frame,
    )

    # Key invariant: pacing score is >= 0.75 despite raw WPM being only ~90-100
    assert inference.pacing.score >= 0.75
    assert any("aligned with baseline" in d.lower() for d in inference.pacing.drivers)

    # Engagement is not zeroed or penalized as disengaged
    assert inference.engagement.score >= 0.50

    store.close()


def test_spec07_test5_every_high_level_score_backward_traceable(tmp_path: Path):
    """Spec 07 Acceptance Test #5:
    Every high-level score can be queried backward to its exact contributing
    timestamped features, window snapshots, and utterances in SQLite.
    """
    db_path = tmp_path / "test5.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_t5",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    conversation = [
        ("Hello, we are interested in scheduling an initial walk-through.", 1000, 5000, "client"),
        ("Wonderful! Let me show you how our solution integrates with your workflow.", 6000, 11000, "salesperson"),
        ("Can you confirm whether you support automated Salesforce sync?", 12000, 16000, "client"),
        ("Yes, full bi-directional synchronization is supported out of the box.", 17000, 22000, "salesperson"),
        ("That sounds promising. Let us tentatively book next Tuesday at 3 PM.", 23000, 28000, "client"),
    ]

    last_frame = None
    client_utterance_ids = []
    for text, start, end, speaker in conversation:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id=speaker,
            start_ms=start,
            end_ms=end,
            call_sid="call_spec07_t5",
        )
        if speaker == "client":
            client_utterance_ids.append(utt.utterance_id)
        t_snap = timing_engine.process_utterance(utt)
        sem = SemanticFeatureSnapshot(
            utterance_id=utt.utterance_id,
            call_sid="call_spec07_t5",
            speaker_id=speaker,
            question_type="transactional" if "?" in text else "none",
            recurrence_type="scheduling_detail_repeated" if "Tuesday" in text else "none",
            specificity_score=0.80,
            future_language_score=0.85 if "Tuesday" in text else 0.20,
            semantic_confidence=0.95,
        )
        last_frame = aggregator.process_turn(utt, t_snap, semantic_snapshot=sem)

    assert last_frame is not None
    inference = inference_engine.compute_inference(
        call_sid="call_spec07_t5",
        current_frame=last_frame,
    )

    # 1. Verify every dimension reports contributing evidence IDs
    for dim_name, dim_obj in [
        ("pacing", inference.pacing),
        ("engagement", inference.engagement),
        ("trust", inference.trust),
        ("momentum", inference.momentum),
        ("readiness", inference.readiness),
    ]:
        assert len(dim_obj.contributing_evidence_ids) > 0, f"{dim_name} has no contributing evidence IDs"

    # 2. Backward-trace query against SQLite evidence store
    all_evidence_ids = inference.contributing_evidence_ids
    assert len(all_evidence_ids) > 0

    traced_snapshots = store.backward_trace(all_evidence_ids)
    assert len(traced_snapshots) == len(all_evidence_ids)

    traced_ids = {s.evidence_id for s in traced_snapshots}
    for eid in all_evidence_ids:
        assert eid in traced_ids

    # 3. Verify traced snapshots resolve back to physical utterances
    recovered_utterance_ids = set()
    for s in traced_snapshots:
        assert s.call_sid == "call_spec07_t5"
        assert s.window_duration_ms >= 0 or s.horizon == "full_call"
        for uid in s.contributing_utterance_ids:
            recovered_utterance_ids.add(uid)

    # Must contain contributing client utterances
    for uid in client_utterance_ids:
        assert uid in recovered_utterance_ids

    store.close()


def test_spec07_test6_latency_budget_under_10ms(tmp_path: Path):
    """Spec 07 Cross-Cutting Rule:
    The complete turn processing pipeline must execute well within the real-time budget
    (< 10ms per turn in deterministic/heuristic mode).
    """
    db_path = tmp_path / "test6.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_t6",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    latencies_ms = []
    for i in range(50):
        t_start = time.monotonic()

        utt = normalize_generic_transcript(
            text=f"Turn {i} discussing software capabilities and deployment options.",
            speaker_id="client" if i % 2 == 0 else "salesperson",
            start_ms=i * 3000,
            end_ms=(i + 1) * 3000,
            call_sid="call_spec07_t6",
        )
        t_snap = timing_engine.process_utterance(utt)
        frame = aggregator.process_turn(utt, t_snap)
        inf = inference_engine.compute_inference(
            call_sid="call_spec07_t6",
            current_frame=frame,
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        latencies_ms.append(elapsed_ms)

    # Discard cold-start turn 0 for steady-state SLA evaluation
    steady_latencies = latencies_ms[1:]
    mean_latency = sum(steady_latencies) / len(steady_latencies)
    sorted_latencies = sorted(steady_latencies)
    p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]

    # Steady-state mean latency must be under 3ms, p95 under 10ms
    assert mean_latency < 3.0, f"Mean latency too high: {mean_latency:.2f}ms"
    assert p95_latency < 10.0, f"P95 latency too high: {p95_latency:.2f}ms"

    store.close()


def test_spec07_test7_state_versioning_and_monotonicity(tmp_path: Path):
    """Spec 07 Cross-Cutting Rule:
    Successive DownstreamInferenceState objects increment state_version monotonically
    to prevent out-of-order state overwrites in asynchronous environments.
    """
    db_path = tmp_path / "test7.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator(
        call_sid="call_spec07_t7",
        timing_engine=timing_engine,
        store=store,
    )
    inference_engine = DownstreamInferenceEngine()

    prev_version = 0
    seen_ids = set()

    for i in range(5):
        utt = normalize_generic_transcript(
            text=f"Turn message sequence number {i}.",
            speaker_id="client",
            start_ms=i * 2000,
            end_ms=(i + 1) * 2000,
            call_sid="call_spec07_t7",
        )
        t_snap = timing_engine.process_utterance(utt)
        frame = aggregator.process_turn(utt, t_snap)
        inference = inference_engine.compute_inference(
            call_sid="call_spec07_t7",
            current_frame=frame,
        )

        # Monotonic state versioning: 1, 2, 3, 4, 5
        assert inference.state_version == prev_version + 1
        prev_version = inference.state_version

        # Unique inference IDs
        assert inference.inference_id not in seen_ids
        seen_ids.add(inference.inference_id)

        # Latency metric populated
        assert inference.inference_latency_ms >= 0.0

    store.close()
