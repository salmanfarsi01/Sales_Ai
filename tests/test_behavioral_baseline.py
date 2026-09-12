from __future__ import annotations

import pytest
from pathlib import Path

from copilot.behavioral_normalization import normalize_generic_transcript
from copilot.behavioral_timing import (
    DeterministicTimingEngine,
    TimingFeatureSnapshot,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.behavioral_baseline import (
    ProspectBaselineStore,
    BaselineAndChangePointEngine,
    SpeakerBaselineStats,
    BaselineProfile,
)


def test_spec07_slow_speaker_baseline_invariance(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    # Uses the production default 60000ms intra_call_window_ms
    engine = BaselineAndChangePointEngine(
        call_sid="call_slow_speaker",
        prospect_id=None,
        store=store,
        intra_call_window_ms=60000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )

    timing_engine = DeterministicTimingEngine()

    # Realistic 60-90s initial conversation sequence with a naturally slow speaker (~75-90 WPM)
    slow_calibration_turns = [
        ("Hello, yes I received your voicemail regarding our property in Austin.", 1000, 7000),
        ("Well my wife and I have been thinking about selling sometime next spring.", 15000, 25000),
        ("We are really not in any hurry and want to make sure the timing is right.", 35000, 48000),
        ("Our neighbor told us they had a good experience, so we wanted to hear what you had to say.", 55000, 68000),
    ]

    for text, start, end in slow_calibration_turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_slow_speaker",
        )
        t_snap = timing_engine.process_utterance(utt)
        engine.update_with_utterance(utt, t_snap)

    assert engine.is_intra_call_locked is True
    active_profile = engine.get_active_profile()
    assert active_profile is not None
    assert 70.0 <= active_profile.speech_rate_wpm.mean <= 100.0

    # Normal slow follow-up turn at 75s: Should NOT trigger a change-point (z approx 0.0)
    normal_slow_turn = normalize_generic_transcript(
        text="We still need to consult with our accountant before making any commitment.",
        speaker_id="client",
        start_ms=75000,
        end_ms=85000,
        call_sid="call_slow_speaker",
    )
    t_snap_slow = timing_engine.process_utterance(normal_slow_turn)
    cps = engine.update_with_utterance(normal_slow_turn, t_snap_slow)
    assert len(cps) == 0

    deviations = engine.compute_deviations(t_snap_slow)
    rate_dev = next(d for d in deviations if d.feature_name == "speech_rate_wpm")
    assert abs(rate_dev.z_score) < 2.0
    assert rate_dev.is_significant is False

    # Sudden rapid surge turn at 95s (22 words in 5 seconds -> ~260 WPM)
    fast_turn = normalize_generic_transcript(
        text="Wait what are you talking about that fee makes no sense at all why would you say that to me right now!",
        speaker_id="client",
        start_ms=95000,
        end_ms=100000,
        call_sid="call_slow_speaker",
    )
    timing_engine.process_utterance(fast_turn)
    t_snap_fast = timing_engine.compute_snapshot_for_window(window_ms=10000, speaker_id="client")
    fast_cps = engine.update_with_utterance(fast_turn, t_snap_fast)
    assert len(fast_cps) >= 1
    rate_cp = next(cp for cp in fast_cps if cp.feature_name == "speech_rate_wpm")
    assert rate_cp.z_score >= 2.0
    assert rate_cp.direction == "surge"


def test_baseline_sufficiency_gate_rejects_premature_lock(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_short_insufficient",
        prospect_id="prospect_short_1",
        store=store,
        intra_call_window_ms=60000,
        min_calibration_turns=3,
        min_cumulative_words=15,
    )

    # 1. First turn at 5 seconds: 1 word ("Hello?")
    utt1 = normalize_generic_transcript(
        text="Hello?",
        speaker_id="client",
        start_ms=1000,
        end_ms=2000,
        call_sid="call_short_insufficient",
    )
    t1 = TimingFeatureSnapshot(
        timestamp_ms=2000,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=60.0,
        avg_pause_duration_ms=0.0,
        intra_turn_pause_count=0,
        response_latency_ms=300,
        turn_length_words=1,
        turn_duration_ms=1000,
        timing_confidence=1.0,
    )
    cps1 = engine.update_with_utterance(utt1, t1)
    assert engine.is_intra_call_locked is False
    assert engine.get_active_profile() is None
    assert len(cps1) == 0

    # 2. Second turn at 15 seconds: 2 words ("Who's calling?")
    utt2 = normalize_generic_transcript(
        text="Who is calling?",
        speaker_id="client",
        start_ms=14000,
        end_ms=15500,
        call_sid="call_short_insufficient",
    )
    t2 = TimingFeatureSnapshot(
        timestamp_ms=15500,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=120.0,
        avg_pause_duration_ms=0.0,
        intra_turn_pause_count=0,
        response_latency_ms=400,
        turn_length_words=3,
        turn_duration_ms=1500,
        timing_confidence=1.0,
    )
    cps2 = engine.update_with_utterance(utt2, t2)
    assert engine.is_intra_call_locked is False
    assert len(cps2) == 0

    # 3. Call ends at 20s (hung up). Teardown should NOT persist an uncalibrated 2-turn baseline
    final_profile = engine.finalize_and_persist()
    assert final_profile is None
    assert store.get_baseline("prospect_short_1") is None


def test_baseline_sufficiency_gate_rejects_90s_fallback_with_insufficient_words(tmp_path: Path):
    """
    Verifies that exceeding the 90s maximum calibration window with >= 3 turns
    STILL refuses to lock if cumulative words < 15.
    Prevents sub-4-word utterances from locking a dummy 0.0 WPM baseline.
    """
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_90s_sparse",
        prospect_id="prospect_90s_sparse",
        store=store,
        intra_call_window_ms=60000,
        max_calibration_window_ms=90000,
        min_calibration_turns=3,
        min_cumulative_words=15,
    )

    # Turn 1 at 5s: "No" (1 word)
    u1 = normalize_generic_transcript("No", speaker_id="client", start_ms=5000, end_ms=6000, call_sid="call_90s_sparse")
    t1 = TimingFeatureSnapshot(timestamp_ms=6000, window_ms=60000, speaker_id="client", speech_rate_wpm=60.0, avg_pause_duration_ms=0.0, turn_length_words=1)
    engine.update_with_utterance(u1, t1)

    # Turn 2 at 45s: "Not sure" (2 words)
    u2 = normalize_generic_transcript("Not sure", speaker_id="client", start_ms=45000, end_ms=46500, call_sid="call_90s_sparse")
    t2 = TimingFeatureSnapshot(timestamp_ms=46500, window_ms=60000, speaker_id="client", speech_rate_wpm=80.0, avg_pause_duration_ms=0.0, turn_length_words=2)
    engine.update_with_utterance(u2, t2)

    # Turn 3 at 95s: "Maybe later" (2 words)
    # Elapsed time = 97000 - 5000 = 92000ms (> 90000ms max window)
    # Turns = 3 (>= 3 min calibration turns)
    # Words = 1 + 2 + 2 = 5 (< 15 min cumulative words)
    u3 = normalize_generic_transcript("Maybe later", speaker_id="client", start_ms=95000, end_ms=97000, call_sid="call_90s_sparse")
    t3 = TimingFeatureSnapshot(timestamp_ms=97000, window_ms=60000, speaker_id="client", speech_rate_wpm=60.0, avg_pause_duration_ms=0.0, turn_length_words=2)
    cps3 = engine.update_with_utterance(u3, t3)

    # Baseline MUST NOT lock despite 92s elapsed and 3 turns
    assert engine.is_intra_call_locked is False
    assert engine.get_active_profile() is None
    assert len(cps3) == 0
    assert engine.compute_deviations(t3) == []

    # Teardown at call close must also refuse to persist an uncalibrated 5-word baseline
    final_profile = engine.finalize_and_persist()
    assert final_profile is None
    assert store.get_baseline("prospect_90s_sparse") is None


def test_adaptive_clean_baseline_refinement_quarantines_outliers(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_adaptive_test",
        prospect_id="prospect_adapt_1",
        store=store,
        intra_call_window_ms=10000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )

    # 3 calibration turns totaling 24 words across 12 seconds
    for i in range(3):
        start = i * 4000
        end = start + 3000
        utt = normalize_generic_transcript(
            text=f"This is a standard calibration sentence number {i} spoken cleanly.",
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_adaptive_test",
        )
        t = TimingFeatureSnapshot(
            timestamp_ms=end,
            window_ms=60000,
            speaker_id="client",
            speech_rate_wpm=140.0,
            avg_pause_duration_ms=250.0,
            intra_turn_pause_count=1,
            response_latency_ms=400,
            turn_length_words=9,
            turn_duration_ms=3000,
            timing_confidence=1.0,
        )
        engine.update_with_utterance(utt, t)

    assert engine.is_intra_call_locked is True
    initial_profile = engine.get_active_profile()
    assert initial_profile is not None
    assert initial_profile.speech_rate_wpm.sample_count == 3
    initial_mean = initial_profile.speech_rate_wpm.mean

    # Clean normal turn at 16s: 145 WPM (|z| < 2.0)
    normal_utt = normalize_generic_transcript(
        text="Another normal clean utterance that fits the speaker distribution well.",
        speaker_id="client",
        start_ms=16000,
        end_ms=19000,
        call_sid="call_adaptive_test",
    )
    normal_t = TimingFeatureSnapshot(
        timestamp_ms=19000,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=145.0,
        avg_pause_duration_ms=250.0,
        intra_turn_pause_count=1,
        response_latency_ms=400,
        turn_length_words=9,
        turn_duration_ms=3000,
        timing_confidence=1.0,
    )
    engine.update_with_utterance(normal_utt, normal_t)

    # Adaptive refinement: sample_count should have grown to 4, mean updated slightly
    refined_profile = engine.get_active_profile()
    assert refined_profile.speech_rate_wpm.sample_count == 4
    assert refined_profile.speech_rate_wpm.mean == round((3 * 140.0 + 145.0) / 4, 2)

    # Outlier anomaly turn at 22s: Extreme slowdown (40 WPM, 2500ms latency)
    outlier_utt = normalize_generic_transcript(
        text="Wait... no... that... cannot... work.",
        speaker_id="client",
        start_ms=22000,
        end_ms=27000,
        call_sid="call_adaptive_test",
    )
    outlier_t = TimingFeatureSnapshot(
        timestamp_ms=27000,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=40.0,
        avg_pause_duration_ms=900.0,
        intra_turn_pause_count=3,
        response_latency_ms=2500,
        turn_length_words=5,
        turn_duration_ms=5000,
        timing_confidence=1.0,
    )
    cps = engine.update_with_utterance(outlier_utt, outlier_t)
    assert len(cps) >= 1

    # Outlier quarantine check: sample_count must REMAIN 4; 40 WPM must NOT be added to baseline!
    profile_after_outlier = engine.get_active_profile()
    assert profile_after_outlier.speech_rate_wpm.sample_count == 4
    assert profile_after_outlier.speech_rate_wpm.mean == refined_profile.speech_rate_wpm.mean


def test_first_time_caller_graceful_intra_call_fallback(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_new_prospect",
        prospect_id="prospect_unknown_123",
        store=store,
        intra_call_window_ms=5000,
        min_calibration_turns=3,
        min_cumulative_words=15,
    )

    assert engine.historical_profile is None
    assert engine.get_active_profile() is None

    turns = [
        ("Hello, we got your letter about our property.", 1000, 3000, 140.0, 280.0, 400, 8),
        ("We might be open to selling if the price makes sense.", 4000, 7000, 145.0, 300.0, 500, 10),
        ("Could you give us an idea of recent comps?", 8000, 10000, 142.0, 290.0, 450, 8),
    ]

    for text, start, end, wpm, pause, lat, words in turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_new_prospect",
        )
        t_snap = TimingFeatureSnapshot(
            timestamp_ms=end,
            window_ms=60000,
            speaker_id="client",
            speech_rate_wpm=wpm,
            avg_pause_duration_ms=pause,
            intra_turn_pause_count=1,
            response_latency_ms=lat,
            interruptions_60s=0,
            turn_length_words=words,
            turn_duration_ms=end - start,
            timing_confidence=1.0,
        )
        engine.update_with_utterance(utt, t_snap)

    assert engine.is_intra_call_locked is True
    profile = engine.get_active_profile()
    assert profile is not None
    expected_mean = round((140.0 + 145.0 + 142.0) / 3, 2)
    assert profile.speech_rate_wpm.mean == expected_mean
    assert profile.call_count == 1


def test_cross_call_weighted_blending(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    historical = BaselineProfile(
        prospect_id="prospect_repeat_999",
        speech_rate_wpm=SpeakerBaselineStats(mean=150.0, stddev=15.0, sample_count=30),
        avg_pause_duration_ms=SpeakerBaselineStats(mean=300.0, stddev=50.0, sample_count=30),
        response_latency_ms=SpeakerBaselineStats(mean=400.0, stddev=80.0, sample_count=30),
        turn_length_words=SpeakerBaselineStats(mean=10.0, stddev=3.0, sample_count=30),
        call_count=3,
        updated_at_ms=100000,
    )
    store.save_baseline("prospect_repeat_999", historical)

    engine = BaselineAndChangePointEngine(
        call_sid="call_repeat_prospect",
        prospect_id="prospect_repeat_999",
        store=store,
        intra_call_window_ms=4000,
        min_calibration_turns=3,
        min_cumulative_words=15,
    )

    assert engine.historical_profile is not None
    assert engine.historical_profile.speech_rate_wpm.mean == 150.0

    turns = [
        ("Hey again, we decided to review the numbers.", 1000, 3000, 8),
        ("Let's talk through the details of the contract.", 4000, 6000, 8),
        ("We are ready to look at the timeline.", 7000, 9000, 8),
    ]

    for text, start, end, words in turns:
        utt = normalize_generic_transcript(
            text=text,
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_repeat_prospect",
        )
        t = TimingFeatureSnapshot(
            timestamp_ms=end,
            window_ms=60000,
            speaker_id="client",
            speech_rate_wpm=130.0,
            avg_pause_duration_ms=320.0,
            intra_turn_pause_count=1,
            response_latency_ms=450,
            interruptions_60s=0,
            turn_length_words=words,
            turn_duration_ms=2000,
            timing_confidence=1.0,
        )
        engine.update_with_utterance(utt, t)

    assert engine.is_intra_call_locked is True
    blended = engine.get_active_profile()
    assert blended is not None

    # w_cross = min(0.60, 0.15 * 3) = 0.45
    # w_intra = 0.55
    # blended_mean = 0.55 * 130.0 + 0.45 * 150.0 = 71.5 + 67.5 = 139.0
    assert blended.speech_rate_wpm.mean == 139.0


def test_change_point_detection_with_semantic_context_and_confidence(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_cp_test",
        prospect_id=None,
        store=store,
        intra_call_window_ms=5000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )

    for i in range(3):
        start = i * 3000
        end = start + 2000
        utt = normalize_generic_transcript(
            text=f"Normal sentence number {i} with typical length and words.",
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_cp_test",
        )
        t = TimingFeatureSnapshot(
            timestamp_ms=end,
            window_ms=60000,
            speaker_id="client",
            speech_rate_wpm=140.0,
            avg_pause_duration_ms=250.0,
            intra_turn_pause_count=1,
            response_latency_ms=400,
            interruptions_60s=0,
            turn_length_words=8,
            turn_duration_ms=2000,
            timing_confidence=1.0,
        )
        engine.update_with_utterance(utt, t)

    assert engine.is_intra_call_locked is True

    # Dramatic hesitation and slowdown: response latency spikes to 2500ms
    spike_utt = normalize_generic_transcript(
        text="Wait... that commission fee is just way too high.",
        speaker_id="client",
        start_ms=10000,
        end_ms=13000,
        call_sid="call_cp_test",
    )
    spike_timing = TimingFeatureSnapshot(
        timestamp_ms=13000,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=60.0,
        avg_pause_duration_ms=900.0,
        intra_turn_pause_count=3,
        response_latency_ms=2500,
        interruptions_60s=0,
        turn_length_words=9,
        turn_duration_ms=3000,
        timing_confidence=0.9,
    )
    sem_snap = SemanticFeatureSnapshot(
        utterance_id="utt_cp_1",
        call_sid="call_cp_test",
        speaker_id="client",
        question_type="evaluation",
        recurrence_id="OBJ_COMM_01",
        recurrence_type="same_objection_repeated",
        specificity_score=0.4,
        future_language_score=0.0,
        boundary_score=0.0,
        agreement_score=0.0,
    )

    cps = engine.update_with_utterance(spike_utt, spike_timing, sem_snap)
    assert len(cps) >= 1

    latency_cp = next((c for c in cps if c.feature_name == "response_latency_ms"), None)
    assert latency_cp is not None
    assert latency_cp.z_score >= 2.0
    assert latency_cp.direction == "surge"
    assert latency_cp.confidence == 0.9
    assert latency_cp.semantic_context is not None
    assert latency_cp.semantic_context["recurrence_type"] == "same_objection_repeated"


def test_change_point_inherits_synthetic_timing_confidence_penalty(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_synthetic_penalty",
        prospect_id=None,
        store=store,
        intra_call_window_ms=5000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )

    for i in range(3):
        start = i * 3000
        end = start + 2000
        utt = normalize_generic_transcript(
            text=f"Sample calibration sentence number {i} spoken cleanly.",
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_synthetic_penalty",
        )
        t = TimingFeatureSnapshot(
            timestamp_ms=end,
            window_ms=60000,
            speaker_id="client",
            speech_rate_wpm=130.0,
            avg_pause_duration_ms=250.0,
            intra_turn_pause_count=1,
            response_latency_ms=350,
            turn_length_words=7,
            turn_duration_ms=2000,
            timing_confidence=1.0,
        )
        engine.update_with_utterance(utt, t)

    assert engine.is_intra_call_locked is True

    # Utterance with synthetic timing penalty (timing_confidence=0.5)
    slow_utt = normalize_generic_transcript(
        text="A very slow turn with estimated timing words.",
        speaker_id="client",
        start_ms=10000,
        end_ms=15000,
        call_sid="call_synthetic_penalty",
        is_estimated_timing=True,
    )
    slow_timing = TimingFeatureSnapshot(
        timestamp_ms=15000,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=50.0,
        avg_pause_duration_ms=800.0,
        intra_turn_pause_count=2,
        response_latency_ms=2000,
        turn_length_words=7,
        turn_duration_ms=5000,
        timing_confidence=0.5,
    )

    cps = engine.update_with_utterance(slow_utt, slow_timing)
    assert len(cps) >= 1
    for cp in cps:
        assert cp.confidence == 0.5


def test_change_point_baseline_source_tagging(tmp_path: Path):
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    # Save a historical baseline for a repeat caller
    historical = BaselineProfile(
        prospect_id="prospect_tagged_1",
        speech_rate_wpm=SpeakerBaselineStats(mean=140.0, stddev=10.0, sample_count=20),
        avg_pause_duration_ms=SpeakerBaselineStats(mean=250.0, stddev=30.0, sample_count=20),
        response_latency_ms=SpeakerBaselineStats(mean=350.0, stddev=40.0, sample_count=20),
        turn_length_words=SpeakerBaselineStats(mean=8.0, stddev=2.0, sample_count=20),
        call_count=2,
        updated_at_ms=100000,
    )
    store.save_baseline("prospect_tagged_1", historical)

    engine = BaselineAndChangePointEngine(
        call_sid="call_tagged",
        prospect_id="prospect_tagged_1",
        store=store,
        intra_call_window_ms=10000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )

    # 1. Immediate turn at second 5 with extreme response latency (2000ms vs 350ms historical mean)
    # The intra-call baseline is NOT yet locked. This deviation MUST be tagged as "historical_only"
    utt1 = normalize_generic_transcript(
        text="Wait... who is this again?",
        speaker_id="client",
        start_ms=2000,
        end_ms=5000,
        call_sid="call_tagged",
    )
    t1 = TimingFeatureSnapshot(
        timestamp_ms=5000,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=140.0,
        avg_pause_duration_ms=250.0,
        intra_turn_pause_count=1,
        response_latency_ms=2000,
        turn_length_words=5,
        turn_duration_ms=3000,
        timing_confidence=1.0,
    )
    cps1 = engine.update_with_utterance(utt1, t1)
    assert len(cps1) >= 1
    cp_hist = next(c for c in cps1 if c.feature_name == "response_latency_ms")
    assert cp_hist.baseline_source == "historical_only"
    assert engine.is_intra_call_locked is False

    # 2. Feed turns 2 and 3 so the intra-call baseline locks and transitions to "blended"
    for i in range(2, 4):
        start = i * 4000
        end = start + 3000
        utt = normalize_generic_transcript(
            text=f"This is normal follow up calibration turn {i} with clean wording.",
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_tagged",
        )
        t = TimingFeatureSnapshot(
            timestamp_ms=end,
            window_ms=60000,
            speaker_id="client",
            speech_rate_wpm=140.0,
            avg_pause_duration_ms=250.0,
            intra_turn_pause_count=1,
            response_latency_ms=350,
            turn_length_words=9,
            turn_duration_ms=3000,
            timing_confidence=1.0,
        )
        engine.update_with_utterance(utt, t)

    assert engine.is_intra_call_locked is True

    # 3. Subsequent turn with extreme speech rate drop (40 WPM vs 140 WPM blended mean)
    # The intra-call baseline IS locked with history. This deviation MUST be tagged as "blended"
    utt_slow = normalize_generic_transcript(
        text="No... I... am... not... sure.",
        speaker_id="client",
        start_ms=17000,
        end_ms=22000,
        call_sid="call_tagged",
    )
    t_slow = TimingFeatureSnapshot(
        timestamp_ms=22000,
        window_ms=60000,
        speaker_id="client",
        speech_rate_wpm=40.0,
        avg_pause_duration_ms=800.0,
        intra_turn_pause_count=2,
        response_latency_ms=350,
        turn_length_words=6,
        turn_duration_ms=5000,
        timing_confidence=1.0,
    )
    cps_blended = engine.update_with_utterance(utt_slow, t_slow)
    assert len(cps_blended) >= 1
    cp_blended = next(c for c in cps_blended if c.feature_name == "speech_rate_wpm")
    assert cp_blended.baseline_source == "blended"

    # 4. First-time caller whose baseline locks must be tagged as "intra_call"
    engine_first_time = BaselineAndChangePointEngine(
        call_sid="call_first_time",
        prospect_id="prospect_new_99",
        store=store,
        intra_call_window_ms=10000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )
    for i in range(3):
        start = i * 4000
        end = start + 3000
        utt = normalize_generic_transcript(
            text=f"First time clean calibration turn number {i} for this prospect.",
            speaker_id="client",
            start_ms=start,
            end_ms=end,
            call_sid="call_first_time",
        )
        t = TimingFeatureSnapshot(
            timestamp_ms=end,
            window_ms=60000,
            speaker_id="client",
            speech_rate_wpm=140.0,
            avg_pause_duration_ms=250.0,
            intra_turn_pause_count=1,
            response_latency_ms=350,
            turn_length_words=9,
            turn_duration_ms=3000,
            timing_confidence=1.0,
        )
        engine_first_time.update_with_utterance(utt, t)

    assert engine_first_time.is_intra_call_locked is True

    # Spike on first-time caller -> "intra_call"
    cps_intra = engine_first_time.update_with_utterance(utt_slow, t_slow)
    assert len(cps_intra) >= 1
    cp_intra = next(c for c in cps_intra if c.feature_name == "speech_rate_wpm")
    assert cp_intra.baseline_source == "intra_call"


def test_recalibrated_variance_floor_prevents_false_anomalies_on_slow_speaker(tmp_path: Path):
    """Verifies that natural micro-variation (e.g. 115 WPM from a 90 WPM baseline speaker)
    and short conversational confirmations do NOT trigger false change-points.
    Under the old min_stddev_ratio=0.08, stddev would be 7.2 WPM, causing 115 WPM to trip |z| = 3.47.
    With recalibrated per-feature floor (ratio=0.18, min_abs=15 WPM), stddev >= 16.2 WPM, keeping |z| < 1.6.
    """
    store = ProspectBaselineStore(store_path=tmp_path / "baselines.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_slow_variance_test",
        prospect_id="prospect_slow_var",
        store=store,
        intra_call_window_ms=60000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )
    timing_engine = DeterministicTimingEngine()

    # Establish steady, deliberate ~90-95 WPM baseline across 65s
    steady_slow_turns = [
        ("We are currently reviewing all vendor proposals with our evaluation board.", 1000, 8000),      # 11 words, 7s -> 94.3 WPM
        ("The main requirement for our team is seamless integration with our database.", 15000, 23000),  # 12 words, 8s -> 90.0 WPM
        ("We need to ensure compliance requirements are strictly met across all systems.", 40000, 48000), # 11 words, 8s -> 82.5 WPM
        ("Our executive committee will make the final selection next month.", 55000, 63000),              # 10 words, 8s -> 75.0 WPM
    ]

    for text, start, end in steady_slow_turns:
        utt = normalize_generic_transcript(text, "client", start, end, "call_slow_variance_test")
        t_snap = timing_engine.process_utterance(utt)
        engine.update_with_utterance(utt, t_snap)

    assert engine.is_intra_call_locked is True
    profile = engine.get_active_profile()
    assert profile is not None
    # Recalibrated stddev floor guarantees stddev >= max(15.0, mean * 0.18)
    assert profile.speech_rate_wpm.stddev >= 15.0

    # Test 1: Natural micro-variation turn (115 WPM - faster sentence, but still natural speech)
    # Under old 0.08 ratio, this would have produced z = (115 - 88) / 7.0 = 3.86 (FALSE ANOMALY)
    natural_faster_turn = normalize_generic_transcript(
        text="That makes total sense and we certainly agree with that approach.", # 11 words, 6s -> 110 WPM
        speaker_id="client",
        start_ms=68000,
        end_ms=74000,
        call_sid="call_slow_variance_test",
    )
    timing_engine.process_utterance(natural_faster_turn)
    t_faster = timing_engine.compute_snapshot_for_window(window_ms=10000, speaker_id="client")
    cps_faster = engine.update_with_utterance(natural_faster_turn, t_faster)

    # Must NOT trigger a false rate surge change-point
    assert not any(cp.feature_name == "speech_rate_wpm" for cp in cps_faster)

    # Test 2: Short conversational confirmation ("Yes, exactly.")
    # Word count is 2 words, duration 1.2s -> raw WPM would be 100 WPM, but short turns (< 4 words)
    # are protected from rate change-point false alerts, pause absence drops, and backchannel length drops.
    short_utt = normalize_generic_transcript(
        text="Yes, exactly.",
        speaker_id="client",
        start_ms=76000,
        end_ms=77200,
        call_sid="call_slow_variance_test",
    )
    timing_engine.process_utterance(short_utt)
    t_short = timing_engine.compute_snapshot_for_window(window_ms=10000, speaker_id="client")
    cps_short = engine.update_with_utterance(short_utt, t_short)

    assert len(cps_short) == 0

    # Test 3: True genuine surge (240 WPM - frantic objection) MUST still trigger
    surge_utt = normalize_generic_transcript(
        text="Wait hold on that is completely ridiculous why would you ever quote that amount to us right now!",
        speaker_id="client",
        start_ms=80000,
        end_ms=85000,
        call_sid="call_slow_variance_test",
    )
    timing_engine.process_utterance(surge_utt)
    t_surge = timing_engine.compute_snapshot_for_window(window_ms=10000, speaker_id="client")
    cps_surge = engine.update_with_utterance(surge_utt, t_surge)

    assert any(cp.feature_name == "speech_rate_wpm" and cp.direction == "surge" and cp.z_score >= 2.0 for cp in cps_surge)


def test_short_utterance_and_pause_gating_behavior(tmp_path: Path):
    """Verifies that:
    1. Utterances with 0 intra-turn pauses do NOT trigger false 'avg_pause_duration_ms' drop change-points.
    2. Utterances with 0 intra-turn pauses do NOT pollute the baseline pause duration sample distribution.
    3. Short utterances (< 4 words) do NOT pollute baseline speech rate with denominator quantization noise.
    4. Legitimate response latency hesitations on short utterances (e.g. 3500ms before 'No.') ARE preserved
       and successfully trigger response_latency_ms change-points.
    """
    store = ProspectBaselineStore(store_path=tmp_path / "baselines_pause_gate.json")
    engine = BaselineAndChangePointEngine(
        call_sid="call_pause_gate",
        prospect_id="prospect_pause_gate",
        store=store,
        intra_call_window_ms=60000,
        min_calibration_turns=3,
        min_cumulative_words=15,
        z_threshold=2.0,
    )

    # Establish baseline with explicit 400ms pause and 600ms latency
    profile = BaselineProfile(
        prospect_id="prospect_pause_gate",
        speech_rate_wpm=SpeakerBaselineStats(mean=120.0, stddev=21.6, sample_count=10),
        avg_pause_duration_ms=SpeakerBaselineStats(mean=400.0, stddev=150.0, sample_count=10),
        response_latency_ms=SpeakerBaselineStats(mean=600.0, stddev=200.0, sample_count=10),
        turn_length_words=SpeakerBaselineStats(mean=12.0, stddev=4.2, sample_count=10),
        call_count=2,
    )
    store.save_baseline("prospect_pause_gate", profile)
    engine = BaselineAndChangePointEngine(
        call_sid="call_pause_gate",
        prospect_id="prospect_pause_gate",
        store=store,
    )

    # 1. Fluent 2-word acknowledgment: "Yes, definitely." (0 intra-turn pauses, normal latency 550ms)
    utt_fluent = normalize_generic_transcript(
        text="Yes, definitely.",
        speaker_id="client",
        start_ms=1000,
        end_ms=2000,
        call_sid="call_pause_gate",
    )
    t_fluent = TimingFeatureSnapshot(
        timestamp_ms=2000,
        window_ms=10000,
        speaker_id="client",
        speech_rate_wpm=120.0,
        avg_pause_duration_ms=0.0,
        intra_turn_pause_count=0,
        response_latency_ms=550,
        turn_length_words=2,
        turn_duration_ms=1000,
        timing_confidence=1.0,
    )
    cps_fluent = engine.update_with_utterance(utt_fluent, t_fluent)
    # Must NOT produce pause drop (val=0) or turn length drop (val=2 backchannel)
    assert len(cps_fluent) == 0

    # Deviations check: pause duration and short-turn rate must NOT be flagged significant
    devs = engine.compute_deviations(t_fluent)
    pause_dev = next(d for d in devs if d.feature_name == "avg_pause_duration_ms")
    assert pause_dev.is_significant is False
    assert pause_dev.z_score == 0.0
    assert pause_dev.is_measured is False

    # 2. Prolonged hesitation before short 1-word response: "No." with 3500ms latency
    utt_hesitant = normalize_generic_transcript(
        text="No.",
        speaker_id="client",
        start_ms=6000,
        end_ms=6500,
        call_sid="call_pause_gate",
    )
    t_hesitant = TimingFeatureSnapshot(
        timestamp_ms=6500,
        window_ms=10000,
        speaker_id="client",
        speech_rate_wpm=120.0,
        avg_pause_duration_ms=0.0,
        intra_turn_pause_count=0,
        response_latency_ms=3500, # (3500 - 600) / 200 = 14.5 -> surge!
        turn_length_words=1,
        turn_duration_ms=500,
        timing_confidence=1.0,
    )
    cps_hesitant = engine.update_with_utterance(utt_hesitant, t_hesitant)
    assert len(cps_hesitant) == 1
    assert cps_hesitant[0].feature_name == "response_latency_ms"
    assert cps_hesitant[0].direction == "surge"
    assert cps_hesitant[0].z_score >= 2.0


