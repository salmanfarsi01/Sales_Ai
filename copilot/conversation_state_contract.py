from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from .behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from .behavioral_semantic import SemanticFeatureSnapshot
from .behavioral_baseline import ContactPreferenceRecord

LOGGER = logging.getLogger("copilot.conversation_state_contract")


def infer_salesperson_strategy_from_text(text: str) -> Optional[str]:
    """Infers canonical salesperson objection reframe strategy from utterance text when not supplied in metadata."""
    if not text:
        return None
    text_lower = text.lower()

    # 1. Financial Net Proceeds / Fee Breakdown Reframe
    if any(re.search(p, text_lower) for p in [
        r"\b(?:net\s+proceeds|net\s+sheet|what\s+you\s+(?:actually\s+)?walk\s+away\s+with)\b",
        r"\b(?:what\s+(?:that|the)\s+fee\s+covers|break\s+down\s+the\s+fee|covers\s+if\s+that\s+would\s+help)\b",
        r"\bafter\s+all\s+costs\b",
        r"\bsee\s+the\s+real\s+number\b",
    ]):
        return "financial_net_proceeds_reframe"

    # 2. Hyperlocal Marketing Differentiation
    if any(re.search(p, text_lower) for p in [
        r"\b(?:hyperlocal|buyer\s+pipeline|marketing\s+(?:plan|differentiation|strategy)|syndication|staging|professional\s+photos?)\b",
    ]):
        return "hyperlocal_marketing_differentiation"

    # 3. Performance / Fee Guarantee
    if any(re.search(p, text_lower) for p in [
        r"\b(?:guarantee|cancel\s+anytime|days\s+on\s+market\s+guarantee|performance\s+guarantee)\b",
    ]):
        return "fee_performance_guarantee"

    # 4. Market Data / Walkthrough Reframe
    if any(re.search(p, text_lower) for p in [
        r"\b(?:walk\s+through\s+(?:what\s+)?the\s+market|see\s+the\s+actual\s+numbers?|market\s+looks?\s+like|look\s+at\s+the\s+(?:numbers|comps|market)|market\s+data)\b",
        r"\bfeel\s+that\s+way\s+before\s+they\s+see\b",
    ]):
        return "market_data_walkthrough"

    return None


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

    # Quality & Performance Metrics
    inference_confidence: float = Field(1.0, ge=0.0, le=1.0)
    inference_latency_ms: float = 0.0

    # Semantic & Compliance Signals
    boundary_score: float = Field(0.0, ge=0.0, le=1.0)
    boundary_confidence: float = Field(1.0, ge=0.0, le=1.0)
    contact_preference: Literal["none", "time_of_day", "reduced_frequency", "timing_restriction", "channel_restriction", "soft_cadence"] = "none"
    contact_preference_confidence: float = Field(0.0, ge=0.0, le=1.0)
    contact_preference_details: Optional[str] = None
    recurrence_type: Literal[
        "same_objection_repeated",
        "concern_after_failed_reframe",
        "positive_echo",
        "boundary_repeated",
        "scheduling_detail_repeated",
        "scheduling_repeated",
        "clarification_probe",
        "none",
    ] = "none"
    recurrence_id: Optional[str] = None
    recurrence_count: int = 0
    question_type: Literal["evaluation", "transactional", "clarifying", "hostile", "rhetorical", "none"] = "none"
    specificity_score: float = Field(0.0, ge=0.0, le=1.0)
    future_language_score: float = Field(0.0, ge=0.0, le=1.0)
    agreement_score: float = Field(0.0, ge=0.0, le=1.0)
    semantic_confidence: float = Field(1.0, ge=0.0, le=1.0)
    agreement_measured: bool = Field(False)
    specificity_measured: bool = Field(False)
    future_language_measured: bool = Field(False)

    # Strategy & Intervention Tracking (populates ObjectionRecord.attempted_strategies in Phase 3)
    salesperson_strategy_tag: Optional[str] = Field(
        None,
        description="e.g. reframe_cost_of_inaction, hyperlocal_comps, fee_guarantee, social_proof"
    )
    salesperson_strategy_source: Literal["strategic_engine", "salesperson_inference", "none", "speech_heuristic"] = "none"

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
    salesperson_strategy_source: Literal["strategic_engine", "salesperson_inference", "none", "speech_heuristic"] = "none",
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

    # Client Feedback Item 6: Automatically infer strategy tag for salesperson if not explicitly supplied
    strat_tag = salesperson_strategy_tag
    strat_source = salesperson_strategy_source
    if speaker_id == "salesperson" and (not strat_tag or strat_tag == "none"):
        inferred = infer_salesperson_strategy_from_text(utterance_text)
        if inferred and semantic_snapshot.question_type != "clarifying":
            strat_tag = inferred
            strat_source = "speech_heuristic"

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
        agreement_measured=getattr(semantic_snapshot, "agreement_measured", False),
        specificity_measured=getattr(semantic_snapshot, "specificity_measured", False),
        future_language_measured=getattr(semantic_snapshot, "future_language_measured", False),
        salesperson_strategy_tag=strat_tag,
        salesperson_strategy_source=strat_source,
        contributing_evidence_ids=evidence_ids,
    )
