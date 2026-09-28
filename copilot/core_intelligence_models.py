from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

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
    # Full Trace Attributes (Spec: Evidence -> Interpretation -> Decision -> Gateway Prompt Stub)
    evidence_considered: List[str] = Field(default_factory=list)
    strategic_interpretation: Dict[str, Any] = Field(default_factory=dict)
    gateway_fallback_stub: Optional[str] = Field(default=None, description="Surfaces LLM Gateway's deterministic fallback stub in offline replay trace")
    final_prompt_text: Optional[str] = Field(default=None, description="Alias for gateway_fallback_stub for backward compatibility")
    meeting_gate_open: Optional[bool] = Field(default=None, description="Current meeting gate status for invariant validation")

    @model_validator(mode="after")
    def validate_action_gate_push_invariants(self) -> "StrategicDecision":
        # Extract gate status if provided via field or strategic_interpretation
        gate_open = self.meeting_gate_open
        if gate_open is None and self.strategic_interpretation:
            gate_open = self.strategic_interpretation.get("meeting_gate_open")

        # 1. Closed Gate Constraint: When meeting gate is explicitly closed,
        # close-style pushes (direct_ask, two_window_choice) and COMMITMENT_CLOSE are strictly forbidden.
        if gate_open is False:
            if self.primary_action == StrategicAction.COMMITMENT_CLOSE:
                raise ValueError(
                    f"Invariant violation: primary_action cannot be COMMITMENT_CLOSE when meeting gate is closed (decision_id={self.decision_id})."
                )
            if self.push_strength in ("direct_ask", "two_window_choice"):
                raise ValueError(
                    f"Invariant violation: push_strength cannot be close-style '{self.push_strength}' when meeting gate is closed (decision_id={self.decision_id})."
                )

        # 2. Action & Push Harmony: Close-style push recommendations (two_window_choice, direct_ask)
        # must NOT accompany non-closing actions.
        non_closing_actions = {
            StrategicAction.QUESTION,
            StrategicAction.CLARIFY,
            StrategicAction.ACKNOWLEDGE,
            StrategicAction.VALIDATE,
            StrategicAction.EDUCATE,
            StrategicAction.REFRAME,
            StrategicAction.MIRROR,
            StrategicAction.DIFFERENTIATE,
        }
        if self.primary_action in non_closing_actions and self.push_strength in ("direct_ask", "two_window_choice"):
            raise ValueError(
                f"Invariant violation: push_strength '{self.push_strength}' cannot accompany non-closing primary_action '{self.primary_action.value}'."
            )

        # 3. COMMITMENT_CLOSE requires an affirmative closing push strength or milestone confirmation
        if self.primary_action == StrategicAction.COMMITMENT_CLOSE:
            if self.push_strength in ("respect_record_exit", "protect_and_shorten", "explore_conditional_terms"):
                raise ValueError(
                    f"Invariant violation: primary_action COMMITMENT_CLOSE is incompatible with push_strength '{self.push_strength}'."
                )

        return self


class StrategicInterpretationContext(BaseModel):
    state_version: int
    active_objections: List[str] = Field(default_factory=list)
    failed_strategies_by_objection: Dict[str, List[str]] = Field(default_factory=dict)
    decision_maker_present: bool = True
    hard_boundary_active: bool = False
    boundary_suspected: bool = False
    boundary_retracted: bool = False
    meeting_gate_open: bool = False
    conversion_confirmed: bool = False
    push_strength_state: PushStrengthState = "resolve_then_ask"
    readiness_score: Optional[float] = None
    momentum_trend: str = "stable"
    trust_score: float = 0.0
    cross_metric_penalties: List[str] = Field(default_factory=list)


class DecisionEvaluationResult(BaseModel):
    decision: StrategicDecision
    context: StrategicInterpretationContext
