from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

ObjectionLifecycleState = Literal[
    "unresolved",
    "clarified",
    "partially_resolved",
    "resolved",
    "reactivated",
    "boundary",
]

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
    presence: Literal["on_call", "absent", "unknown"] = "unknown"
    notes: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class DecisionStructure(BaseModel):
    primary_decision_maker: Optional[str] = None
    decision_maker_present: bool = True
    stakeholders: List[DecisionStakeholder] = Field(default_factory=list)
    timeline_horizon: Optional[str] = None
    urgency_level: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    access_constraints: List[str] = Field(default_factory=list)
    confidence: float = Field(0.70, ge=0.0, le=1.0)


class ObjectionRecord(BaseModel):
    objection_id: str = Field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:8]}")
    recurrence_id: Optional[str] = Field(None, description="Anchored Behavioral Signal Engine recurrence tracking ID")
    canonical_category: str = Field(..., description="e.g. commission_fee, timing_market, representation_broker, price")
    initial_statement: str
    latest_statement: str
    lifecycle_state: ObjectionLifecycleState = "unresolved"
    first_turn_id: int
    last_updated_turn_id: int
    recurrence_count: int = 1
    attempted_strategies: List[str] = Field(default_factory=list)
    resolution_evidence: Optional[str] = None
    confidence: float = Field(0.85, ge=0.0, le=1.0)


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
    emotion_valence: float = Field(0.0, ge=-1.0, le=1.0)
    emotion_tension: float = Field(0.0, ge=0.0, le=1.0)
    emotion_confidence: float = Field(0.7, ge=0.0, le=1.0)
    engagement: float = Field(0.5, ge=0.0, le=1.0)
    engagement_confidence: float = Field(0.7, ge=0.0, le=1.0)
    momentum: float = Field(0.5, ge=0.0, le=1.0)
    momentum_confidence: float = Field(0.7, ge=0.0, le=1.0)
    readiness: float = Field(0.5, ge=0.0, le=1.0)
    readiness_confidence: float = Field(0.7, ge=0.0, le=1.0)
    pacing: float = Field(0.5, ge=0.0, le=1.0)
    pacing_confidence: float = Field(0.7, ge=0.0, le=1.0)


class ContactComplianceState(BaseModel):
    hard_boundary_active: bool = False
    hard_boundary_reason: Optional[str] = None
    contact_preference: Literal["none", "reduced_frequency", "channel_restriction", "timing_restriction"] = "none"
    contact_preference_details: Optional[str] = None
    contact_preference_confidence: float = Field(0.0, ge=0.0, le=1.0)


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
    readiness_score: float = Field(..., ge=0.0, le=100.0, description="Final capped readiness score on 0 - 100 scale")
    uncapped_score: float = Field(..., ge=0.0, le=100.0, description="Readiness score before applying blocker caps")
    emotional_readiness: float = Field(..., ge=0.0, le=100.0)
    logical_readiness: float = Field(..., ge=0.0, le=100.0)
    logistical_readiness: float = Field(..., ge=0.0, le=100.0)
    decision_readiness: float = Field(..., ge=0.0, le=100.0)
    active_blocker_caps: List[str] = Field(default_factory=list)
    capped_reason: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ConversationStateSnapshot(BaseModel):
    """Current truth about the conversation at turn N."""
    state_id: str = Field(default_factory=lambda: f"state_{uuid.uuid4().hex[:10]}")
    call_sid: str
    state_version: int = Field(1, ge=1)
    last_updated_turn_id: int = 0
    last_updated_timestamp_ms: int = 0
    decision_structure: DecisionStructure = Field(default_factory=DecisionStructure)
    objections: List[ObjectionRecord] = Field(default_factory=list)
    facts: List[PersistentFactRecord] = Field(default_factory=list)
    dimensions: DimensionScores = Field(default_factory=DimensionScores)
    contact_compliance: ContactComplianceState = Field(default_factory=ContactComplianceState)
    momentum: Optional[MomentumBreakdown] = None
    readiness: Optional[ReadinessBreakdown] = None
    overall_confidence: float = Field(0.75, ge=0.0, le=1.0)
    change_history: List[StateChangeRecord] = Field(default_factory=list)

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
        return [o for o in self.objections if o.lifecycle_state in ("unresolved", "reactivated", "partially_resolved")]
