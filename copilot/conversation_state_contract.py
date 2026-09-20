from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from .behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from .behavioral_semantic import SemanticFeatureSnapshot
from .behavioral_baseline import ContactPreferenceRecord

LOGGER = logging.getLogger("copilot.conversation_state_contract")


class BehavioralSignalInputBundle(BaseModel):
    """Normalized bundle of per-turn upstream evidence consumed by ConversationState."""
    call_sid: str
    turn_id: int
    speaker_id: Literal["salesperson", "client"]
    utterance_text: str
    timestamp_ms: int

    # Downstream Inference Dimensions (Scores: 0.0 - 1.0, Confidence: 0.0 - 1.0)
    trust: DimensionScore
    emotion: EmotionState
    pacing: DimensionScore
    engagement: DimensionScore
    momentum: DimensionScore
    readiness: DimensionScore
    commitment: Optional[DimensionScore] = None
    inference_confidence: float = Field(..., ge=0.0, le=1.0)
    inference_latency_ms: float = Field(0.0, ge=0.0)

    # Semantic Behavioral Features
    boundary_score: float = Field(0.0, ge=0.0, le=1.0)
    boundary_confidence: float = Field(1.0, ge=0.0, le=1.0)
    contact_preference: Literal["none", "reduced_frequency", "channel_restriction", "timing_restriction"] = "none"
    contact_preference_confidence: float = Field(0.0, ge=0.0, le=1.0)
    contact_preference_details: Optional[str] = None
    recurrence_type: Literal[
        "same_objection_repeated",
        "concern_after_failed_reframe",
        "positive_echo",
        "boundary_repeated",
        "scheduling_detail_repeated",
        "scheduling_repeated",
        "none",
    ] = "none"
    recurrence_id: Optional[str] = None
    recurrence_count: int = 0
    question_type: Literal["evaluation", "transactional", "clarifying", "hostile", "rhetorical", "none"] = "none"
    specificity_score: float = Field(0.0, ge=0.0, le=1.0)
    future_language_score: float = Field(0.0, ge=0.0, le=1.0)
    agreement_score: float = Field(0.0, ge=0.0, le=1.0)
    semantic_confidence: float = Field(1.0, ge=0.0, le=1.0)

    # Strategy & Intervention Tracking (populates ObjectionRecord.attempted_strategies in Phase 3)
    salesperson_strategy_tag: Optional[str] = Field(
        None,
        description="e.g. reframe_cost_of_inaction, hyperlocal_comps, fee_guarantee, social_proof"
    )
    salesperson_strategy_source: Literal["strategic_engine", "salesperson_inference", "none"] = "none"

    # Contributing Evidence IDs for backward traceability
    contributing_evidence_ids: List[str] = Field(default_factory=list)


def extract_behavioral_bundle(
    turn_id: int,
    speaker_id: Literal["salesperson", "client"],
    utterance_text: str,
    inference_state: DownstreamInferenceState,
    semantic_snapshot: SemanticFeatureSnapshot,
    contact_preference_record: Optional[ContactPreferenceRecord] = None,
    salesperson_strategy_tag: Optional[str] = None,
    salesperson_strategy_source: Literal["strategic_engine", "salesperson_inference", "none"] = "none",
) -> BehavioralSignalInputBundle:
    """Safely extracts and validates upstream Behavioral Signal Engine data into an input bundle."""
    # Combine evidence IDs from both inference and semantic snapshots
    evidence_ids = list(inference_state.contributing_evidence_ids)
    if semantic_snapshot.utterance_id and semantic_snapshot.utterance_id not in evidence_ids:
        evidence_ids.append(semantic_snapshot.utterance_id)

    # Determine contact preference (snapshot takes precedence for latest observed turn, falling back to persistent record)
    pref = semantic_snapshot.contact_preference
    pref_conf = semantic_snapshot.contact_preference_confidence
    pref_details = semantic_snapshot.contact_preference_details

    if pref == "none" and contact_preference_record and contact_preference_record.preference != "none":
        pref = contact_preference_record.preference
        pref_conf = contact_preference_record.confidence
        pref_details = contact_preference_record.details

    return BehavioralSignalInputBundle(
        call_sid=inference_state.call_sid,
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=utterance_text,
        timestamp_ms=inference_state.timestamp_ms,
        trust=inference_state.trust,
        emotion=inference_state.emotion,
        pacing=inference_state.pacing,
        engagement=inference_state.engagement,
        momentum=inference_state.momentum,
        readiness=inference_state.readiness,
        commitment=getattr(inference_state, "commitment", None),
        inference_confidence=inference_state.overall_confidence,
        inference_latency_ms=inference_state.inference_latency_ms,
        boundary_score=semantic_snapshot.boundary_score,
        boundary_confidence=semantic_snapshot.boundary_confidence,
        contact_preference=pref,
        contact_preference_confidence=pref_conf,
        contact_preference_details=pref_details,
        recurrence_type=semantic_snapshot.recurrence_type,
        recurrence_id=semantic_snapshot.recurrence_id,
        recurrence_count=semantic_snapshot.recurrence_count,
        question_type=semantic_snapshot.question_type,
        specificity_score=semantic_snapshot.specificity_score,
        future_language_score=semantic_snapshot.future_language_score,
        agreement_score=semantic_snapshot.agreement_score,
        semantic_confidence=semantic_snapshot.semantic_confidence,
        salesperson_strategy_tag=salesperson_strategy_tag,
        salesperson_strategy_source=salesperson_strategy_source,
        contributing_evidence_ids=evidence_ids,
    )
