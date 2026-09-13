import pytest
from pathlib import Path
from copilot.behavioral_normalization import NormalizedUtterance, NormalizedWord
from copilot.behavioral_timing import DeterministicTimingEngine
from copilot.behavioral_evidence import MultiWindowAggregator, SQLiteEvidenceLogStore
from copilot.behavioral_baseline import BaselineAndChangePointEngine
from copilot.behavioral_inference import DownstreamInferenceEngine
from copilot.behavioral_semantic import SemanticFeatureSnapshot, SemanticFeatureEngine
from copilot.behavioral_pipeline_service import execute_behavioral_turn_pipeline


def make_test_utterance(
    utterance_id: str,
    speaker_id: str,
    text: str,
    start_ms: int,
    end_ms: int,
    asr_confidence: float = 0.98,
    is_dual_track: bool = True,
    is_estimated_timing: bool = False,
    speaker_separation_confidence: float = None,
) -> NormalizedUtterance:
    words = text.split()
    step = (end_ms - start_ms) // max(1, len(words))
    norm_words = [
        NormalizedWord(
            word=w,
            start_ms=start_ms + i * step,
            end_ms=start_ms + (i + 1) * step if i < len(words) - 1 else end_ms,
            confidence=asr_confidence,
            is_estimated=is_estimated_timing,
        )
        for i, w in enumerate(words)
    ]
    meta = {
        "is_dual_track": is_dual_track,
        "enforce_single_speaker": is_dual_track,
    }
    if speaker_separation_confidence is not None:
        meta["speaker_separation_confidence"] = speaker_separation_confidence
    elif is_dual_track:
        meta["speaker_separation_confidence"] = 1.0
    else:
        meta["speaker_separation_confidence"] = 0.80

    return NormalizedUtterance(
        utterance_id=utterance_id,
        call_sid="call_sprint4_test",
        speaker_id=speaker_id,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        asr_confidence=asr_confidence,
        is_final=True,
        speech_final=True,
        is_estimated_timing=is_estimated_timing,
        words=norm_words,
        metadata=meta,
    )


def test_confidence_drops_on_poor_asr_audio_quality(tmp_path: Path):
    """Sprint 4 DoD: Intentionally poor audio (ASR confidence = 0.62)
    visibly drops overall confidence below 95% and writes a one-line reason.
    """
    db_path = tmp_path / "test_asr_drop.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_asr_test")
    aggregator = MultiWindowAggregator("call_asr_test", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    # Create locked baseline for speaker first to isolate ASR factor
    for i in range(4):
        u_warm = make_test_utterance(
            f"warm_{i}", "client", "Here are several words to calibrate the baseline profile properly today.",
            i * 20000, i * 20000 + 5000, asr_confidence=0.98
        )
        t_snap = timing_engine.process_utterance(u_warm)
        baseline_engine.update_with_utterance(u_warm, t_snap)
    assert baseline_engine.is_locked("client") is True

    # Poor audio turn: ASR confidence = 0.62
    u_poor = make_test_utterance(
        "u_poor", "client", "I think the audio is breaking up slightly.",
        85000, 88000, asr_confidence=0.62
    )
    t_snap_poor = timing_engine.process_utterance(u_poor)
    frame = aggregator.process_turn(u_poor, t_snap_poor)

    curr_snap = frame.windows["current_utterance"]
    # Evidence confidence must match weakest link = 0.62
    assert curr_snap.evidence_confidence == 0.62
    assert "ASR audio transcription quality (62%)" in curr_snap.confidence_driver

    # Downstream inference overall confidence must be 0.62
    inf = inference_engine.compute_inference("call_asr_test", frame)
    assert inf.overall_confidence == 0.62
    assert inf.overall_confidence_reason is not None
    assert "ASR audio transcription quality (62%)" in inf.overall_confidence_reason

    store.close()


def test_confidence_drops_before_baseline_lock(tmp_path: Path):
    """Sprint 4 DoD: A turn before baseline lock shows reduced confidence (0.70),
    not full 95%, with a clear reason explaining uncalibrated baseline.
    """
    db_path = tmp_path / "test_baseline_drop.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_base_test")
    aggregator = MultiWindowAggregator("call_base_test", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    # First turn: baseline is NOT locked (< 3 turns / < 15 words)
    assert baseline_engine.is_locked("client") is False
    u1 = make_test_utterance("u1", "client", "Hello this is our very first turn.", 0, 3000, asr_confidence=0.98)
    t1 = timing_engine.process_utterance(u1)
    frame1 = aggregator.process_turn(u1, t1)

    curr1 = frame1.windows["current_utterance"]
    assert curr1.evidence_confidence == 0.70
    assert "Uncalibrated speaker baseline (client calibrating)" in curr1.confidence_driver

    inf1 = inference_engine.compute_inference("call_base_test", frame1)
    assert inf1.overall_confidence == 0.70
    assert "Uncalibrated speaker baseline" in inf1.overall_confidence_reason

    store.close()


def test_speaker_separation_confidence_dual_vs_single_track(tmp_path: Path):
    """Sprint 4: Dual-track has separation factor 1.0; single-track diarized drops to 0.80."""
    db_path = tmp_path / "test_sep.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_sep_test")
    aggregator = MultiWindowAggregator("call_sep_test", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    # Pre-lock baseline
    for i in range(4):
        u_warm = make_test_utterance(
            f"warm_{i}", "salesperson", "Consistent calibration words across all turns to lock baseline.",
            i * 20000, i * 20000 + 5000, asr_confidence=0.98, is_dual_track=True
        )
        t_snap = timing_engine.process_utterance(u_warm)
        baseline_engine.update_with_utterance(u_warm, t_snap)
    assert baseline_engine.is_locked("salesperson") is True

    # 1. Dual-track turn: separation confidence is 1.0 -> High confidence (0.95)
    u_dual = make_test_utterance(
        "u_dual", "salesperson", "We are using dual channel microphones here.",
        85000, 88000, asr_confidence=0.98, is_dual_track=True
    )
    t_dual = timing_engine.process_utterance(u_dual)
    frame_dual = aggregator.process_turn(u_dual, t_dual)
    assert frame_dual.windows["current_utterance"].evidence_confidence >= 0.95
    assert frame_dual.windows["current_utterance"].confidence_breakdown["speaker_sep"] == 1.0

    # 2. Single-track diarized turn: separation confidence drops to 0.80
    u_single = make_test_utterance(
        "u_single", "salesperson", "This audio comes from a single microphone stream.",
        90000, 93000, asr_confidence=0.98, is_dual_track=False
    )
    t_single = timing_engine.process_utterance(u_single)
    frame_single = aggregator.process_turn(u_single, t_single)

    curr_single = frame_single.windows["current_utterance"]
    assert curr_single.evidence_confidence == 0.80
    assert "diarization" in curr_single.confidence_driver.lower()

    inf_single = inference_engine.compute_inference("call_sep_test", frame_single)
    assert inf_single.overall_confidence == 0.80
    assert "diarization" in inf_single.overall_confidence_reason.lower()

    store.close()


def test_heuristic_extraction_mode_drops_confidence(tmp_path: Path):
    """Sprint 4: Semantic heuristic fallback drops confidence to 0.65 and is not diluted."""
    db_path = tmp_path / "test_sem_drop.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_sem_test")
    aggregator = MultiWindowAggregator("call_sem_test", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    # Pre-lock baseline
    for i in range(4):
        u_warm = make_test_utterance(
            f"warm_{i}", "client", "Consistent words to lock calibration baseline across multiple horizons.",
            i * 20000, i * 20000 + 5000, asr_confidence=0.98
        )
        t_snap = timing_engine.process_utterance(u_warm)
        baseline_engine.update_with_utterance(u_warm, t_snap)
    assert baseline_engine.is_locked("client") is True

    u = make_test_utterance("u_sem", "client", "What is the fee structure?", 85000, 88000, asr_confidence=0.98)
    t_snap = timing_engine.process_utterance(u)

    # Heuristic semantic feature snapshot (mode = heuristic, confidence = 0.65)
    sem_heuristic = SemanticFeatureSnapshot(
        utterance_id=u.utterance_id,
        call_sid="call_sem_test",
        speaker_id="client",
        extraction_mode="heuristic_timeout",
        question_type="transactional",
        specificity_score=0.40,
        future_language_score=0.20,
        boundary_score=0.0,
        agreement_score=0.0,
        semantic_confidence=0.65,
    )

    frame = aggregator.process_turn(u, t_snap, semantic_snapshot=sem_heuristic)
    curr = frame.windows["current_utterance"]
    assert curr.evidence_confidence == 0.65
    assert "Heuristic semantic extraction" in curr.confidence_driver

    inf = inference_engine.compute_inference("call_sem_test", frame)
    assert inf.overall_confidence == 0.65
    assert "Heuristic semantic extraction" in inf.overall_confidence_reason

    store.close()


def test_weakest_link_rule_no_dilution_average(tmp_path: Path):
    """Sprint 4: Strict weakest link rule - lowest factor dominates, never diluted by average.
    Given:
      timing = 1.0
      speaker separation = 1.0
      baseline = 0.70 (calibrating)
      asr = 0.58 (noisy audio)
      semantic = 0.95
    Result must be min(1.0, 1.0, 0.70, 0.58, 0.95) = 0.58.
    Average would have been 0.846 - which is rejected.
    """
    db_path = tmp_path / "test_weakest.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_weakest_test")  # unlocked
    aggregator = MultiWindowAggregator("call_weakest_test", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    u = make_test_utterance("u_weak", "client", "Noisy audio early in the call.", 1000, 3000, asr_confidence=0.58)
    t_snap = timing_engine.process_utterance(u)
    sem = SemanticFeatureSnapshot(
        utterance_id=u.utterance_id,
        call_sid="call_weakest_test",
        speaker_id="client",
        extraction_mode="llm",
        semantic_confidence=0.95,
    )
    frame = aggregator.process_turn(u, t_snap, semantic_snapshot=sem)
    assert frame.windows["current_utterance"].evidence_confidence == 0.58
    assert "ASR audio transcription quality (58%)" in frame.windows["current_utterance"].confidence_driver

    inf = inference_engine.compute_inference("call_weakest_test", frame)
    assert inf.overall_confidence == 0.58

    store.close()
