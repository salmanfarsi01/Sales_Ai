from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from .conversation_state_models import PushStrengthState


class StrategicAction(str, Enum):
    WAIT_SILENCE = "wait_silence"
    ACKNOWLEDGE = "acknowledge"
    CLARIFY = "clarify"
    VALIDATE = "validate"
    MIRROR = "mirror"
    REFRAME = "reframe"
    EDUCATE = "educate"
    QUANTIFY = "quantify"
    DIFFERENTIATE = "differentiate"
    DE_RISK = "de_risk"
    SOCIAL_PROOF = "social_proof"
    CHALLENGE = "challenge"
    FUTURE_PACE = "future_pace"
    QUESTION = "question"
    COMMITMENT_CLOSE = "commitment_close"

    def __eq__(self, other: Any) -> bool:
        val = getattr(other, "value", other)
        return str(self.value) == str(val)

    def __hash__(self) -> int:
        return hash(self.value)


class Spec01ObjectionLadderStage(str, Enum):
    """Spec 01 Section 6 canonical 6-level objection depth ladder."""
    SURFACE_OBJECTION = "surface_objection"
    UNDERLYING_CONCERN = "underlying_concern"
    FIRST_PUSHBACK = "first_pushback"
    REPEATED_RESISTANCE = "repeated_resistance"
    PARTIAL_RESOLUTION = "partial_resolution"
    RESOLVED = "resolved"


class RequiredFactScope(BaseModel):
    """Spec 11 Section 8 scoped, verifiable fact retrieval contract."""
    fact_id: str = Field(default_factory=lambda: f"fact_req_{uuid.uuid4().hex[:8]}")
    topic: str
    entity_scope: Optional[str] = None
    required_evidence_type: str = "general_proof"
    verification_required: bool = True
    min_confidence: float = 0.80


class StrategicDecision(BaseModel):
    decision_id: str = Field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:10]}")
    call_id: str
    source_state_version: int
    should_prompt: bool = True
    strategic_objective: str
    primary_action: StrategicAction
    secondary_action: Optional[StrategicAction] = None
    push_strength: PushStrengthState = "resolve_then_ask"
    reason_codes: List[str] = Field(default_factory=list)
    do_not_do: List[str] = Field(default_factory=list)
    what_to_protect: List[str] = Field(default_factory=list)
    question_allowed: bool = True
    required_facts: List[Union[str, RequiredFactScope]] = Field(default_factory=list)
    retrieval_needed: bool = False
    playbook_influence: Optional[Dict[str, Any]] = None
    calibration_influence: Optional[Dict[str, Any]] = None
    urgency: Literal["immediate", "moderate", "wait"] = "immediate"
    max_prompt_words: int = 24
    expires_on_state_change: bool = True
    confidence: float = Field(0.85, ge=0.0, le=1.0)
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)
    created_at_ms: int = 0


class StrategicInterpretationContext(BaseModel):
    state_version: int
    active_objections: List[str] = Field(default_factory=list)
    failed_strategies_by_objection: Dict[str, List[str]] = Field(default_factory=dict)
    decision_maker_present: bool = True
    hard_boundary_active: bool = False
    meeting_gate_open: bool = False
    conversion_confirmed: bool = False
    push_strength_state: PushStrengthState = "resolve_then_ask"
    readiness_score: float = 0.0
    momentum_trend: str = "stable"
    trust_score: float = 0.0
    cross_metric_penalties: List[str] = Field(default_factory=list)


class DecisionEvaluationResult(BaseModel):
    decision: StrategicDecision
    context: StrategicInterpretationContext
