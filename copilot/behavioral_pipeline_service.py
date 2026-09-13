"""Shared Behavioral Turn Pipeline Execution Service.

Provides a single, canonical pipeline executor for processing normalized utterance turns
through Deterministic Timing, Semantic Features, Baseline & Change Points, Multi-Window Evidence,
and Downstream Behavioral Inference.

Used by both copilot/fastapi_app.py and run_behavioral_signal.py to guarantee zero runtime drift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from .behavioral_normalization import NormalizedUtterance
from .behavioral_timing import DeterministicTimingEngine, TimingFeatureSnapshot
from .behavioral_semantic import SemanticFeatureEngine, SemanticFeatureSnapshot
from .behavioral_baseline import BaselineAndChangePointEngine, BaselineProfile
from .behavioral_evidence import MultiWindowAggregator, MultiWindowEvidenceFrame
from .behavioral_inference import DownstreamInferenceEngine, DownstreamInferenceState

LOGGER = logging.getLogger("behavioral_pipeline_service")


async def execute_behavioral_turn_pipeline(
    segmented_utts: List[NormalizedUtterance],
    call_sid: str,
    prospect_id: str,
    agent_id: str,
    transcript: str,
    timing_engine: DeterministicTimingEngine,
    semantic_engine: SemanticFeatureEngine,
    baseline_engine: BaselineAndChangePointEngine,
    aggregator: MultiWindowAggregator,
    inference_engine: DownstreamInferenceEngine,
    context_history: Optional[List[NormalizedUtterance]] = None,
    turn_index_start: int = 1,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Processes a chronological sequence of normalized utterances through the complete
    behavioral signal pipeline and returns the standard diagnostic payload.
    """
    if context_history is None:
        context_history = []

    last_timing_snap: Optional[TimingFeatureSnapshot] = None
    last_sem_snap: Optional[SemanticFeatureSnapshot] = None
    last_evidence_frame: Optional[MultiWindowEvidenceFrame] = None
    last_inference_state: Optional[DownstreamInferenceState] = None
    play_by_play_turns: List[Dict[str, Any]] = []

    current_turn_index = turn_index_start - 1

    for norm_utt in segmented_utts:
        current_turn_index += 1
        speaker_role = norm_utt.speaker_id  # "salesperson" or "client"
        speaker_label = "Agent" if speaker_role == "salesperson" else "Prospect"

        # 1. Deterministic Timing Snapshot
        timing_snap = timing_engine.process_utterance(norm_utt)
        last_timing_snap = timing_snap

        # 2. Semantic Features
        sem_snap: Optional[SemanticFeatureSnapshot] = None
        try:
            sem_snap = await semantic_engine.analyze_turn_semantic(
                utterance=norm_utt,
                context_history=context_history,
            )
        except Exception as exc:
            LOGGER.warning("Semantic feature extraction failed for turn %d: %s", current_turn_index, exc)
        last_sem_snap = sem_snap
        context_history.append(norm_utt)

        # 3. Per-Speaker Baseline Profile Comparison & Change-Points
        new_cps = baseline_engine.update_with_utterance(norm_utt, timing_snap, sem_snap)

        # 4. Multi-Window Evidence Frame
        evidence_frame = aggregator.process_turn(
            norm_utt, timing_snap, sem_snap, new_change_points=new_cps
        )
        last_evidence_frame = evidence_frame

        # 5. Downstream Inference Scoring
        inference_state = inference_engine.compute_inference(
            call_sid=call_sid,
            current_frame=evidence_frame,
        )
        last_inference_state = inference_state

        # 6. Baseline Profile & Delta Calculations (Strictly Segregated Per Speaker)
        active_prof: Optional[BaselineProfile] = baseline_engine.get_active_profile(speaker_role)
        base_wpm = (
            round(active_prof.speech_rate_wpm.mean, 1)
            if (active_prof and active_prof.speech_rate_wpm.sample_count > 0)
            else None
        )
        curr_wpm = timing_snap.turn_speech_rate_wpm or round(timing_snap.speech_rate_wpm, 1)
        pace_delta = (
            round(((curr_wpm - base_wpm) / base_wpm) * 100.0, 1)
            if (base_wpm and base_wpm > 0)
            else None
        )
        base_pause_ms = (
            round(active_prof.avg_pause_duration_ms.mean, 1)
            if (active_prof and active_prof.avg_pause_duration_ms.sample_count > 0)
            else None
        )
        pause_delta = (
            round(((timing_snap.avg_pause_duration_ms - base_pause_ms) / base_pause_ms) * 100.0, 1)
            if (base_pause_ms and base_pause_ms > 0 and timing_snap.pause_measured)
            else None
        )

        play_by_play_turns.append({
            "turn_index": current_turn_index,
            "speaker_id": speaker_role,
            "speaker_label": speaker_label,
            "start_ms": norm_utt.start_ms,
            "end_ms": norm_utt.end_ms,
            "start_sec": round(norm_utt.start_ms / 1000.0, 1),
            "end_sec": round(norm_utt.end_ms / 1000.0, 1),
            "text": norm_utt.text,
            "words_count": len(norm_utt.words) if norm_utt.words else len(norm_utt.text.split()),
            "turn_duration_ms": timing_snap.turn_duration_ms,
            "speech_rate_wpm": curr_wpm,
            "turn_wpm": curr_wpm,
            "rolling_wpm_60s": round(timing_snap.speech_rate_wpm, 1),
            "baseline_wpm": base_wpm,
            "pace_delta_pct": pace_delta,
            "avg_pause_duration_ms": round(timing_snap.avg_pause_duration_ms, 1),
            "baseline_pause_ms": base_pause_ms,
            "pause_delta_pct": pause_delta,
            "response_latency_ms": timing_snap.response_latency_ms,
            "question_type": sem_snap.question_type if sem_snap else "none",
            "recurrence_type": sem_snap.recurrence_type if sem_snap else "none",
            "recurrence_count": sem_snap.recurrence_count if sem_snap else 0,
            "boundary_score": sem_snap.boundary_score if sem_snap else 0.0,
            "specificity_score": sem_snap.specificity_score if sem_snap else 0.0,
            "future_language_score": sem_snap.future_language_score if sem_snap else 0.0,
            "agreement_score": sem_snap.agreement_score if sem_snap else 0.0,
            "turn_confidence": evidence_frame.windows["current_utterance"].evidence_confidence,
            "confidence_driver": evidence_frame.windows["current_utterance"].confidence_driver,
            "asr_confidence": round(norm_utt.asr_confidence, 2),
            "speaker_separation_confidence": norm_utt.metadata.get(
                "speaker_separation_confidence",
                1.0 if (norm_utt.metadata and norm_utt.metadata.get("is_dual_track", True)) else 0.80,
            ),
            "semantic_confidence": sem_snap.semantic_confidence if sem_snap else 0.50,
            "pacing_score": inference_state.pacing.score,
            "trust_score": inference_state.trust.score,
            "engagement_score": inference_state.engagement.score,
            "momentum_score": inference_state.momentum.score,
            "readiness_score": inference_state.readiness.score,
            "expressed_valence": inference_state.emotion.expressed_valence if inference_state.emotion else 0.0,
            "tension_level": inference_state.emotion.tension_level if inference_state.emotion else 0.20,
            "is_baseline_locked": baseline_engine.is_locked(speaker_role),
        })

    agent_prof = baseline_engine.get_active_profile("salesperson")
    prospect_prof = baseline_engine.get_active_profile("client")
    agent_base_wpm = (
        round(agent_prof.speech_rate_wpm.mean, 1)
        if (agent_prof and agent_prof.speech_rate_wpm.sample_count > 0)
        else None
    )
    prospect_base_wpm = (
        round(prospect_prof.speech_rate_wpm.mean, 1)
        if (prospect_prof and prospect_prof.speech_rate_wpm.sample_count > 0)
        else None
    )

    elapsed_ms = (
        (last_timing_snap.timestamp_ms - (baseline_engine.earliest_sample_ms or 0))
        if last_timing_snap
        else 0
    )
    prospect_turns = len(baseline_engine.speaker_samples["client"]["turn_length_words"])
    prospect_words = int(sum(baseline_engine.speaker_samples["client"]["turn_length_words"]))
    agent_turns = len(baseline_engine.speaker_samples["salesperson"]["turn_length_words"])
    agent_words = int(sum(baseline_engine.speaker_samples["salesperson"]["turn_length_words"]))
    elapsed_sec = round(elapsed_ms / 1000.0, 1)
    target_sec = round(baseline_engine.intra_call_window_ms / 1000.0, 1)

    result_payload = {
        "status": "success",
        "call_sid": call_sid,
        "prospect_id": prospect_id,
        "agent_id": agent_id,
        "turns_processed": len(segmented_utts),
        "turn_index": current_turn_index,
        "is_baseline_locked": baseline_engine.is_locked("client"),
        "agent_baseline_locked": baseline_engine.is_locked("salesperson"),
        "prospect_baseline_locked": baseline_engine.is_locked("client"),
        "agent_baseline_wpm": agent_base_wpm,
        "prospect_baseline_wpm": prospect_base_wpm,
        "calibration_progress": {
            "is_locked": baseline_engine.is_locked("client"),
            "agent_is_locked": baseline_engine.is_locked("salesperson"),
            "prospect_is_locked": baseline_engine.is_locked("client"),
            "agent_baseline_wpm": agent_base_wpm,
            "prospect_baseline_wpm": prospect_base_wpm,
            "elapsed_sec": elapsed_sec,
            "target_sec": target_sec,
            "turn_count": prospect_turns,
            "target_turns": baseline_engine.min_calibration_turns,
            "total_words": prospect_words,
            "target_words": baseline_engine.min_cumulative_words,
            "agent_turn_count": agent_turns,
            "agent_total_words": agent_words,
        },
        "transcript": transcript,
        "utterances": [
            {
                "turn_index": turn_index_start + idx,
                "speaker_id": u.speaker_id,
                "speaker_label": "Agent" if u.speaker_id == "salesperson" else "Prospect",
                "text": u.text,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
                "words_count": len(u.words),
            }
            for idx, u in enumerate(segmented_utts)
        ],
        "play_by_play_turns": play_by_play_turns,
        "words_count": sum(len(u.words) for u in segmented_utts),
        "latest_timing": last_timing_snap.model_dump() if last_timing_snap else {},
        "latest_inference": last_inference_state.model_dump() if last_inference_state else {},
        "latest_evidence_frame": last_evidence_frame.model_dump() if last_evidence_frame else {},
    }

    if reports_dir is not None:
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_filename = f"behavioral_signal_{call_sid}.json"
            report_path = reports_dir / report_filename
            result_payload["report_file"] = f"reports/{report_filename}"
            report_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            LOGGER.info("Saved behavioral signal report to: %s", report_path)
        except Exception as exc:
            LOGGER.warning("Could not save behavioral signal report to disk: %s", exc)

    return result_payload
