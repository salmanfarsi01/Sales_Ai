from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from .prospect_memory import LoadedProspectMemory

class ObjectionLifecycleState(str, Enum):
    ACTIVE = "active"
    PARTIALLY_ADDRESSED = "partially_addressed"
    DORMANT = "dormant"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"

    # Backward compatibility aliases
    UNRESOLVED = "unresolved"
    CLARIFIED = "clarified"
    PARTIALLY_RESOLVED = "partially_resolved"
    REACTIVATED = "reactivated"
    BOUNDARY = "boundary"

    def __eq__(self, other: Any) -> bool:
        val = getattr(other, "value", other)
        if str(self.value) == str(val):
            return True
        if self.value == "active" and val in ("unresolved", "reactivated", "active"):
            return True
        if val == "active" and self.value in ("unresolved", "reactivated", "active"):
            return True
        if self.value == "partially_addressed" and val in ("partially_resolved", "clarified", "partially_addressed"):
            return True
        if val == "partially_addressed" and self.value in ("partially_resolved", "clarified", "partially_addressed"):
            return True
        return False

    def __hash__(self) -> int:
        return hash(self.value)


class DealDispositionType(str, Enum):
    ACTIVELY_SELLING = "actively_selling"
    RECONSIDERING = "reconsidering"
    DECLINED = "declined"
    REVERSED_DECLINE = "reversed_decline"


class DealDispositionRecord(BaseModel):
    disposition_id: str = Field(default_factory=lambda: f"disp_{uuid.uuid4().hex[:8]}")
    disposition: DealDispositionType = DealDispositionType.ACTIVELY_SELLING
    confidence: float = Field(0.90, ge=0.0, le=1.0)
    source_turn_id: int = 0
    timestamp_ms: int = 0
    rationale: Optional[str] = None
    superseded_by_id: Optional[str] = None
    superseded_at_turn_id: Optional[int] = None


class ConversationStage(str, Enum):
    DISCOVERY = "discovery"
    DECISION_RESOLUTION = "decision_resolution"
    OBJECTION_HANDLING = "objection_handling"
    VALUE_WALKTHROUGH = "value_walkthrough"
    SCHEDULING = "scheduling"
    COMMITMENT_CONFIRMED = "commitment_confirmed"

    def __eq__(self, other: Any) -> bool:
        val = getattr(other, "value", other)
        return str(self.value) == str(val)

    def __hash__(self) -> int:
        return hash(self.value)


class StageHistoryRecord(BaseModel):
    stage: ConversationStage
    entered_turn_id: int
    exited_turn_id: Optional[int] = None
    trigger_reason: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)

PersistentFactCategory = Literal[
    "logistical",
    "decision_maker",
    "timeline",
    "preference",
    "financial",
    "property",
    "general",
]


class DecisionStakeholder(BaseModel):
    name: Optional[str] = None
    role: str = Field(..., description="e.g. spouse, business_partner, attorney, co_owner")
    presence: Literal["on_call", "absent", "confirmed_attending", "unknown"] = "unknown"
    notes: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class DecisionStructure(BaseModel):
    primary_decision_maker: Optional[str] = None
    decision_maker_present: bool = True
    co_decision_required: bool = False
    stakeholders: List[DecisionStakeholder] = Field(default_factory=list)
    timeline_horizon: Optional[str] = None
    urgency_level: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    access_constraints: List[str] = Field(default_factory=list)
    confidence: float = Field(0.70, ge=0.0, le=1.0)


class ObjectionDriverLayer(BaseModel):
    """Client Feedback No. 5: Second-layer underlying driver mapping why an objection exists."""
    surface_objection: str = Field(..., description="Canonical category, e.g. commission_fee")
    underlying_driver: str = Field(..., description="Closed taxonomy driver, e.g. price_resistance, perceived_value_deficit, previous_agent_outcome")
    origin_context: Optional[str] = Field(None, description="Triggering conversational context, e.g. referenced prior agent experience")
    supporting_evidence: List[str] = Field(default_factory=list, description="Turn IDs or quoted phrases justifying the driver classification")
    confidence: float = Field(0.90, ge=0.0, le=1.0)
    strategic_target: str = Field(..., description="Actionable objective the sales response should address")
    classification_source: Literal["llm", "heuristic_pattern", "heuristic_default"] = Field(
        "heuristic_pattern",
        description="Source of classification: 'llm' for Groq LLM inference, 'heuristic_pattern' for offline pattern matching, 'heuristic_default' for category fallback"
    )


# -----------------------------------------------------------------------------
# Canonical Strategy Families for Tactical Aliasing (Client Review Follow-Up)
# Maps near-duplicate tactical tag variations to their canonical strategy family
# -----------------------------------------------------------------------------
STRATEGY_FAMILY_ALIASES: Dict[str, str] = {
    # Net proceeds reframe variations
    "net_proceeds_comparison": "financial_net_proceeds_reframe",
    "net_sheet_breakdown": "financial_net_proceeds_reframe",
    "net_proceeds_reframe": "financial_net_proceeds_reframe",
    "net_sheet_roi": "financial_net_proceeds_reframe",
    # Hyperlocal marketing variations
    "hyperlocal_comps": "hyperlocal_marketing_differentiation",
    "marketing_differentiation": "hyperlocal_marketing_differentiation",
    "hyperlocal_buyer_pipeline": "hyperlocal_marketing_differentiation",
    # Performance guarantee variations
    "fee_guarantee": "fee_performance_guarantee",
    "performance_guarantee": "fee_performance_guarantee",
    "days_on_market_guarantee": "fee_performance_guarantee",
}


def normalize_strategy_tag(strategy_tag: str) -> str:
    """Normalizes strategy tag to its canonical family if recognized, or strips/lowercases."""
    clean = strategy_tag.strip().lower()
    return STRATEGY_FAMILY_ALIASES.get(clean, clean)


class StrategyAttemptOutcome(BaseModel):
    """Client Feedback No. 6: Structured effectiveness feedback tracking prospect reaction to an attempted reframe."""
    strategy_tag: str = Field(..., description="Salesperson strategy tag, e.g. financial_net_proceeds_reframe")
    attempted_at_turn_id: int
    prospect_response_turn_id: int
    prospect_response_summary: str = Field(..., description="Explanation of how prospect reacted, e.g. partial_acceptance_objection_persists")
    effectiveness: Literal["effective", "partial", "insufficient", "rejected", "no_response"]
    evidence: Dict[str, float] = Field(default_factory=dict, description="Behavioral signals from response turn, e.g. agreement_score, specificity_score")
    timestamp_ms: int = 0


class DormancyEvidence(BaseModel):
    evidence_type: Literal[
        "supersession",
        "stage_transition",
        "behavioral_resolution",
        "blocker_supersession",
    ]
    description: str
    turn_id: int
    supporting_signals: Dict[str, Any] = Field(default_factory=dict)


class ObjectionRecord(BaseModel):
    objection_id: str = Field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:8]}")
    recurrence_id: Optional[str] = Field(None, description="Anchored Behavioral Signal Engine recurrence tracking ID")
    canonical_category: str = Field(..., description="e.g. commission_fee, timing_market, representation_broker, price")
    initial_statement: str
    latest_statement: str
    lifecycle_state: ObjectionLifecycleState = ObjectionLifecycleState.ACTIVE
    first_turn_id: int
    last_updated_turn_id: int
    recurrence_count: int = 1
    attempted_strategies: List[str] = Field(default_factory=list)
    resolution_evidence: Optional[str] = None
    confidence: float = Field(0.85, ge=0.0, le=1.0)
    superseded_by_objection_id: Optional[str] = None
    superseded_at_turn_id: Optional[int] = None
    # Client Feedback No. 5 & No. 6 Extensions
    driver_layer: Optional[ObjectionDriverLayer] = None
    strategy_outcomes: List[StrategyAttemptOutcome] = Field(default_factory=list)
    dormancy_evidence: Optional[DormancyEvidence] = None
    deferred_to_meeting: bool = Field(default=False, description="True when prospect explicitly defers discussion of this objection to the scheduled meeting (Point 3)")
    retained_for_followup: bool = Field(default=False, description="True when concern is retained for meeting follow-up (Point 3)")

    def get_failed_strategies(self) -> List[str]:
        """Returns list of strategy tags that failed (insufficient or rejected) on this objection."""
        return [o.strategy_tag for o in self.strategy_outcomes if o.effectiveness in ("insufficient", "rejected")]

    def has_strategy_failed(self, strategy_tag: str, match_family: bool = True) -> bool:
        """Returns True if the specified strategy (or its canonical family) has already failed on this objection.
        If match_family is True (default), checks both exact tag and canonical tactical family aliases.
        """
        target_norm = normalize_strategy_tag(strategy_tag) if match_family else strategy_tag.strip().lower()
        for o in self.strategy_outcomes:
            if o.effectiveness in ("insufficient", "rejected"):
                o_norm = normalize_strategy_tag(o.strategy_tag) if match_family else o.strategy_tag.strip().lower()
                if o.strategy_tag == strategy_tag or o_norm == target_norm:
                    return True
        return False

    def get_latest_strategy_outcome(self) -> Optional[StrategyAttemptOutcome]:
        """Returns the most recent strategy attempt outcome, if any."""
        return self.strategy_outcomes[-1] if self.strategy_outcomes else None

    def get_effective_strategies(self) -> List[str]:
        """Returns list of strategy tags that successfully resolved or advanced this objection."""
        return [o.strategy_tag for o in self.strategy_outcomes if o.effectiveness == "effective"]

    def get_partial_strategies(self) -> List[str]:
        """Returns list of strategy tags that achieved partial agreement."""
        return [o.strategy_tag for o in self.strategy_outcomes if o.effectiveness == "partial"]



class PersistentFactRecord(BaseModel):
    fact_id: str = Field(default_factory=lambda: f"fact_{uuid.uuid4().hex[:8]}")
    category: PersistentFactCategory = "general"
    fact_key: str = Field(..., description="e.g. spouse_involvement, morning_availability, preferred_channel")
    fact_value: str
    status: Literal["active", "superseded"] = "active"
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source_turn_id: int
    timestamp_ms: int
    superseded_by_fact_id: Optional[str] = None
    superseded_at_turn_id: Optional[int] = None
    notes: Optional[str] = None


class DimensionScores(BaseModel):
    trust: float = Field(0.5, ge=0.0, le=1.0)
    trust_confidence: float = Field(0.7, ge=0.0, le=1.0)
    trust_measured: bool = False
    emotion_valence: float = Field(0.0, ge=-1.0, le=1.0)
    emotion_tension: float = Field(0.0, ge=0.0, le=1.0)
    emotion_confidence: float = Field(0.7, ge=0.0, le=1.0)
    engagement: float = Field(0.5, ge=0.0, le=1.0)
    engagement_confidence: float = Field(0.7, ge=0.0, le=1.0)
    momentum: float = Field(0.5, ge=0.0, le=1.0)
    momentum_confidence: float = Field(0.7, ge=0.0, le=1.0)
    readiness: float = Field(0.5, ge=0.0, le=1.0)
    readiness_confidence: float = Field(0.7, ge=0.0, le=1.0)
    commitment: float = Field(0.0, ge=0.0, le=1.0)
    commitment_confidence: float = Field(0.7, ge=0.0, le=1.0)
    pacing: float = Field(0.5, ge=0.0, le=1.0)
    pacing_confidence: float = Field(0.7, ge=0.0, le=1.0)


class ContactPreference(BaseModel):
    """Client Feedback Issue #7: Structured soft or channel-specific contact preference."""
    channel: Literal["sms", "call", "email", "whatsapp", "other"]
    allowed: bool = True
    cadence: Optional[Literal["reduced", "specific_times", "no_preference"]] = None
    prohibited_behavior: Optional[str] = None  # e.g. "daily texting"
    time_restriction: Optional[str] = None  # e.g. "no calls before 10am"
    time_not_before: Optional[str] = None  # e.g. "9am" or "10am"
    time_not_after: Optional[str] = None   # e.g. "6pm" or "5pm"
    boundary_strength: Literal["preference", "hard_restriction"] = "preference"
    source_turn_id: int
    confidence: float = 1.0
    is_historical: bool = False
    source_call_sid: Optional[str] = None
    notes: Optional[str] = None

    def is_time_permitted(self, proposed_time_24h: str) -> Tuple[bool, Optional[str]]:
        """Evaluates whether proposed_time_24h (e.g. '19:00', '08:30') complies with this preference.

        Bare numbers like '6' are read as 18:00 (or '8' as 08:00) via normalize_time_to_24h.
        Text-only windows such as 'early morning' or 'afternoon' return (False, 'Unspecified window requires clarification').
        """
        if not self.allowed:
            return False, f"Channel '{self.channel}' is prohibited"
        if not proposed_time_24h or not re.match(r"^\d{2}:\d{2}$", proposed_time_24h):
            return False, f"Unspecified or text window '{proposed_time_24h}' requires clarification"
        if self.time_not_before and proposed_time_24h < self.time_not_before:
            return False, f"Proposed time {proposed_time_24h} violates restriction: no {self.channel} before {self.time_not_before}"
        if self.time_not_after and proposed_time_24h > self.time_not_after:
            return False, f"Proposed time {proposed_time_24h} violates restriction: no {self.channel} after {self.time_not_after}"
        if self.time_restriction:
            tr_lower = self.time_restriction.lower()
            if "early morning" in tr_lower and proposed_time_24h < "09:00":
                return False, f"Proposed time {proposed_time_24h} violates restriction: no {self.channel} in early morning"
            if ("evening" in tr_lower or "after 6" in tr_lower) and proposed_time_24h >= "18:00":
                return False, f"Proposed time {proposed_time_24h} violates restriction: no {self.channel} after 18:00"
        return True, None


class ContactComplianceState(BaseModel):
    """Client Feedback Issue #7: Contact compliance separating hard legal/stop boundaries from soft preferences."""
    hard_boundary_active: bool = False
    hard_boundary_reason: Optional[str] = None
    hard_boundary_channels: List[str] = Field(default_factory=list)
    contact_preferences: List[ContactPreference] = Field(default_factory=list)
    boundary_suspected: bool = False
    boundary_suspected_reason: Optional[str] = None
    boundary_suspected_turn_id: Optional[int] = None
    hard_boundary_retracted: bool = False
    retraction_turn_id: Optional[int] = None
    contact_not_before: Optional[str] = Field(default=None, description="Temporal hold until specific day/time (e.g. 'Thursday')")
    contact_not_before_turn_id: Optional[int] = None
    # Backward compatibility scalar fields
    contact_preference: Literal["none", "reduced_frequency", "channel_restriction", "timing_restriction"] = "none"
    contact_preference_details: Optional[str] = None
    contact_preference_confidence: float = Field(0.0, ge=0.0, le=1.0)


class ComplianceEvent(BaseModel):
    """ComplianceEvent per Spec 11 §4.6 with clearly labeled inferred extensions."""
    event_id: str = Field(default_factory=lambda: f"comp_{uuid.uuid4().hex[:8]}")
    event_type: Literal[
        "disclosure_detected",
        "consent_issue",
        "stop_ai_prompted",
        "ai_stopped",
        "continued_after_disclosure_issue",
        "hard_boundary_confirmed",
        "boundary_retracted",
    ]
    occurred_at: str
    source_turn_ids: List[int] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    is_inferred_extension: bool = False


ContactCompliance = ContactComplianceState


class StateChangeRecord(BaseModel):
    """Explainability record capturing the exact cause and evidence for every state mutation."""
    change_id: str = Field(default_factory=lambda: f"chg_{uuid.uuid4().hex[:8]}")
    state_version_before: int
    state_version_after: int
    field_path: str = Field(..., description="Target field, e.g. decision_structure.decision_maker_present")
    old_value: Any = None
    new_value: Any = None
    triggering_turn_id: int
    evidence_ids: List[str] = Field(default_factory=list)
    reason: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    timestamp_ms: int


MomentumTrend = Literal["advancing", "stable", "stalling", "regressing"]


class MomentumBreakdown(BaseModel):
    """Structured breakdown of 7-family weighted momentum score and trajectory (Phase 6)."""
    momentum_score: float = Field(..., ge=0.0, le=100.0, description="Overall momentum score on 0 - 100 scale")
    trend: MomentumTrend = "stable"
    trend_delta: float = 0.0
    family_scores: Dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ReadinessBreakdown(BaseModel):
    """Multi-dimensional readiness breakdown with explainable blocker caps (Phase 6)."""
    readiness_score: Optional[float] = Field(default=None, description="Final capped readiness score on 0-100 scale, or None if insufficient evidence or below coverage threshold")
    readiness_partial: Optional[float] = Field(default=None, description="Calculated partial score across measured dimensions, available even when overall score is unpublished due to low coverage")
    coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Proportion of dimension weight measured (0.0 to 1.0)")
    uncapped_score: Optional[float] = Field(default=None, description="Readiness score before applying blocker caps")
    emotional_readiness: Optional[float] = Field(default=None)
    logical_readiness: Optional[float] = Field(default=None)
    logistical_readiness: Optional[float] = Field(default=None)
    decision_readiness: Optional[float] = Field(default=None)
    active_blocker_caps: List[str] = Field(default_factory=list)
    capped_reason: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    insufficient_evidence: bool = Field(default=False, description="True if baseline lacks prospect-originated evidence (unknown readiness) or below coverage threshold")
    evidence_turn_ids: List[int] = Field(default_factory=list)


PushStrengthState = Literal[
    "confirm_and_protect",  # Client Feedback Item 9: "We already won, stop selling" mode
    "protect_and_shorten",
    "resolve_then_ask",
    "direct_ask",
    "two_window_choice",
    "reduce_friction_reask",
    "respect_record_exit",
    "explore_conditional_terms",
]

ConversionType = Literal[
    "in_person_meeting",
    "property_walkthrough",
    "phone_consultation",
    "video_call",
    "document_review",
    "information_send",
    "unspecified",
]

class ConversionEventStatus(str, Enum):
    PROPOSED = "proposed"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    COMPLETED = "completed"
    # Stage-gate and compatibility states
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    NOT_ATTEMPTED = "not_attempted"
    DECLINED = "declined"


ConversionStatus = ConversionEventStatus


class GateConditionResult(BaseModel):
    """Evaluation result for one of the 7 meeting gate conditions."""
    condition_name: str
    status: Literal["met", "not_met", "unknown"] = "unknown"
    met: bool = False
    evidence_turn_ids: List[int] = Field(default_factory=list)
    score_or_value: Any = None
    threshold: Any = None
    reason: str
    is_overridden: bool = Field(default=False, description="True if condition passed via explicit human statement override")

    def model_post_init(self, __context: Any) -> None:
        if self.met and self.status == "unknown":
            self.status = "met"
        elif self.status == "met":
            self.met = True
        else:
            self.met = False


class MeetingConversionGate(BaseModel):
    """Boolean safety gate with explainable condition census (Phase 7)."""
    is_open: bool = Field(False, description="True if and only if all 7 conditions are satisfied simultaneously")
    status: Literal["open", "closed"] = "closed"
    conversion_target: str = Field("appointment", description="Active conversion ask (e.g. appointment, signed_listing_agreement, permission_to_follow_up)")
    conditions: List[GateConditionResult] = Field(default_factory=list)
    failed_conditions: List[str] = Field(default_factory=list)
    unknown_conditions: List[str] = Field(default_factory=list)
    blocking_reasons: List[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    explicit_commitment_detected: bool = Field(default=False, description="True if prospect provided a direct, concrete commitment")
    commitment_slot: Optional[str] = Field(None, description="Extracted concrete commitment slot (e.g. Thursday at 4)")


class PushStrengthRecommendation(BaseModel):
    """Strategic recommendation for how assertive to be when closing (Phase 7 / Point 6).
    Action/posture describes strategy; push strength describes pressure.
    """
    state: PushStrengthState
    rationale: str
    recommended_action: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    pressure: str = Field(default="none", description="Pressure level: none, low, moderate, high (Point 6)")
    strategic_posture: str = Field(default="protect", description="Posture: protect, advance, explore, coordinate (Point 6)")
    strategy: str = Field(default="confirm_and_protect", description="Action/strategy name: confirm_and_protect, resolve_then_ask, etc.")
    legacy_strategy_alias: Optional[str] = Field(default=None, description="Explicit legacy strategy alias (Point 6)")

    def model_post_init(self, __context: Any) -> None:
        st = str(self.state)
        self.strategy = st
        if not self.legacy_strategy_alias:
            self.legacy_strategy_alias = st
        if st in ("confirm_and_protect", "protect_and_shorten", "respect_record_exit"):
            self.pressure = "none"
            self.strategic_posture = "protect"
        elif st in ("resolve_then_ask", "two_window_choice"):
            self.pressure = "moderate"
            self.strategic_posture = "advance"
        elif st == "direct_ask":
            self.pressure = "high"
            self.strategic_posture = "advance"
        elif st in ("reduce_friction_reask", "explore_conditional_terms"):
            self.pressure = "low"
            self.strategic_posture = "explore"
        elif st in ("none", "low", "moderate", "high"):
            self.pressure = st

    @property
    def push_strength(self) -> str:
        """Pressure level describing closing push strength (Point 6)."""
        return self.pressure


class ConversionEventObject(BaseModel):
    """Structured commitment / conversion tracking entity with non-destructive supersession."""
    event_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:8]}")
    conversion_type: ConversionType = "unspecified"
    status: ConversionEventStatus = ConversionEventStatus.NOT_ATTEMPTED
    start_at: Optional[str] = None
    location_or_format: Optional[str] = None
    participants: List[str] = Field(default_factory=list)
    confirmation_confidence: float = Field(0.0, ge=0.0, le=1.0)
    source_turn_ids: List[int] = Field(default_factory=list)
    blocking_items: List[str] = Field(default_factory=list)
    followup_is_conversion: bool = False
    superseded_by_event_id: Optional[str] = None
    supersedes_event_id: Optional[str] = None
    superseded_at_turn_id: Optional[int] = None
    reversal_reason: Optional[str] = None


class ConversationStateSnapshot(BaseModel):
    """Current truth about the conversation at turn N."""
    state_id: str = Field(default_factory=lambda: f"state_{uuid.uuid4().hex[:10]}")
    call_sid: str
    state_version: int = Field(1, ge=1)
    last_updated_turn_id: int = 0
    last_updated_timestamp_ms: int = 0
    prospect_turn_ids: List[int] = Field(default_factory=list)
    decision_structure: DecisionStructure = Field(default_factory=DecisionStructure)
    objections: List[ObjectionRecord] = Field(default_factory=list)
    facts: List[PersistentFactRecord] = Field(default_factory=list)
    dimensions: DimensionScores = Field(default_factory=DimensionScores)
    contact_compliance: ContactComplianceState = Field(default_factory=ContactComplianceState)
    momentum: Optional[MomentumBreakdown] = None
    readiness: Optional[ReadinessBreakdown] = None
    conversion_gate: Optional[MeetingConversionGate] = None
    push_strength: Optional[PushStrengthRecommendation] = None
    conversion_event: Optional[ConversionEventObject] = None
    conversion_events: List[ConversionEventObject] = Field(default_factory=list)
    deal_disposition: Optional[DealDispositionRecord] = None
    deal_dispositions: List[DealDispositionRecord] = Field(default_factory=list)
    conversation_stage: ConversationStage = ConversationStage.DISCOVERY
    stage_history: List[StageHistoryRecord] = Field(default_factory=list)
    compliance_events: List[ComplianceEvent] = Field(default_factory=list)
    overall_confidence: float = Field(0.75, ge=0.0, le=1.0)
    unclassified_material: bool = Field(default=False, description="True if prospect turn was classified as material but no specific structural extractor fired")
    change_history: List[StateChangeRecord] = Field(default_factory=list)
    loaded_prospect_memory: List[LoadedProspectMemory] = Field(
        default_factory=list,
        description="Eligible historical prospect memory records loaded with provenance, kept separate from current-call facts (Point 17).",
    )

    def get_loaded_prospect_memories(self) -> List[LoadedProspectMemory]:
        return list(self.loaded_prospect_memory)

    def get_loaded_prospect_memory(self, key: str) -> Optional[LoadedProspectMemory]:
        for m in reversed(self.loaded_prospect_memory):
            if m.key == key:
                return m
        return None

    def get_active_facts(self) -> List[PersistentFactRecord]:
        return [f for f in self.facts if f.status == "active"]

    def get_active_fact(self, fact_key: str) -> Optional[PersistentFactRecord]:
        for f in reversed(self.facts):
            if f.fact_key == fact_key and f.status == "active":
                return f
        return None

    def get_superseded_facts(self) -> List[PersistentFactRecord]:
        return [f for f in self.facts if f.status == "superseded"]

    def get_unresolved_objections(self) -> List[ObjectionRecord]:
        return [
            o for o in self.objections
            if o.lifecycle_state in (
                ObjectionLifecycleState.ACTIVE,
                ObjectionLifecycleState.PARTIALLY_ADDRESSED,
                "active",
                "partially_addressed",
                "unresolved",
                "reactivated",
                "partially_resolved",
            )
        ]

    def get_active_objections(self) -> List[ObjectionRecord]:
        return [
            o for o in self.objections
            if o.lifecycle_state in (
                ObjectionLifecycleState.ACTIVE,
                ObjectionLifecycleState.PARTIALLY_ADDRESSED,
                "active",
                "partially_addressed",
                "unresolved",
                "clarified",
                "partially_resolved",
                "reactivated",
            )
        ]

    def get_dormant_objections(self) -> List[ObjectionRecord]:
        return [o for o in self.objections if o.lifecycle_state in (ObjectionLifecycleState.DORMANT, "dormant")]

    def get_superseded_objections(self) -> List[ObjectionRecord]:
        return [o for o in self.objections if o.lifecycle_state in (ObjectionLifecycleState.SUPERSEDED, "superseded")]

    def get_deal_disposition_history(self) -> List[DealDispositionRecord]:
        return list(self.deal_dispositions)

    def get_active_deal_disposition(self) -> Optional[DealDispositionRecord]:
        for d in reversed(self.deal_dispositions):
            if d.superseded_by_id is None:
                return d
        return self.deal_disposition

    def get_conversion_event_history(self) -> List[ConversionEventObject]:
        """Returns the full chronological lineage of conversion events."""
        return list(self.conversion_events)

    def get_active_conversion_event(self) -> Optional[ConversionEventObject]:
        """Returns the current active (unsuperseded) conversion event, if any."""
        for ev in reversed(self.conversion_events):
            if ev.superseded_by_event_id is None:
                return ev
        return self.conversion_event

    def get_superseded_conversion_events(self) -> List[ConversionEventObject]:
        """Returns all superseded historical conversion events."""
        return [ev for ev in self.conversion_events if ev.superseded_by_event_id is not None]

    @property
    def compliance_event(self) -> Optional[ComplianceEvent]:
        """Returns the most recent compliance event or None."""
        return self.compliance_events[-1] if self.compliance_events else None

