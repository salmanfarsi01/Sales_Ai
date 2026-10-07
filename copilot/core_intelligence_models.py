from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

from .conversation_state_models import PushStrengthState


class StrategicAction(str, Enum):
    WAIT_SILENCE = "wait_silence"  # Tactical conversational pause (giving prospect 2-3s space to finish speaking/venting)
    HOLD = "hold"                  # Strategic state hold / no new prompt stance (temporal hold, waiting on stakeholder)
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


class PushStrengthValue(str):
    """Encapsulates independent push pressure (none/low/moderate/high) with backward-compatible legacy aliasing."""
    _legacy_alias: Optional[str] = None

    def __new__(cls, value: str, legacy_alias: Optional[str] = None):
        str_val = str(value)
        # If passed legacy name directly, map to canonical pressure level while keeping legacy alias
        if str_val == "confirm_and_protect" and not legacy_alias:
            str_val = "none"
            legacy_alias = "confirm_and_protect"
        elif str_val in ("protect_and_shorten", "respect_record_exit") and not legacy_alias:
            legacy_alias = str_val
            str_val = "none"
        elif str_val == "resolve_then_ask" and not legacy_alias:
            legacy_alias = str_val
            str_val = "moderate"
        elif str_val == "two_window_choice" and not legacy_alias:
            legacy_alias = str_val
            str_val = "moderate"
        elif str_val == "direct_ask" and not legacy_alias:
            legacy_alias = str_val
            str_val = "high"

        obj = super().__new__(cls, str_val)
        obj._legacy_alias = legacy_alias
        return obj

    @property
    def legacy_alias(self) -> Optional[str]:
        return getattr(self, "_legacy_alias", None)

    def __eq__(self, other: Any) -> bool:
        if super().__eq__(other):
            return True
        alias = getattr(self, "_legacy_alias", None)
        if alias and str(other) == str(alias):
            return True
        return False

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, cls) else (
                cls(v["value"], v.get("legacy_alias")) if isinstance(v, dict) else cls(str(v))
            )
        )


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
    call_sid: Optional[str] = None
    source_state_version: int
    source_turn_id: Optional[int] = None
    source_event_id: Optional[str] = None
    utterance_turn_id: Optional[int] = None
    metrics_source_turn_id: Optional[int] = None
    should_prompt: bool = True
    strategic_objective: str
    primary_action: StrategicAction
    strategic_posture: str = Field(default="explore", description="High-level posture: protect, advance, defend, coordinate, explore (Point 6)")
    secondary_action: Optional[StrategicAction] = None
    secondary_action_reason: Optional[str] = Field(default=None, description="Explicit justification for secondary technique (Point 9)")
    push_strength: Union[PushStrengthValue, PushStrengthState, str] = Field(default="resolve_then_ask", description="Push strength pressure: none, low, moderate, high (Point 6)")
    carried_forward_from_decision_id: Optional[str] = Field(default=None, description="Decision ID when strategy is continued from prior turn (Point 13)")
    carried_forward_from_turn_id: Optional[int] = Field(default=None, description="Turn ID when strategy is continued from prior turn (Point 13)")
    referenced_fact_ids: List[str] = Field(default_factory=list, description="IDs of facts influencing this decision (Point 12)")
    referenced_objection_ids: List[str] = Field(default_factory=list, description="IDs of objections influencing this decision (Point 12)")
    referenced_stakeholder_ids: List[str] = Field(default_factory=list, description="IDs of stakeholders influencing this decision (Point 12)")
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
    unclassified_material: bool = Field(default=False, description="True if decision routed from an unclassified material prospect disclosure")
    # Full Trace Attributes (Spec: Evidence -> Interpretation -> Decision -> Gateway Prompt Stub)
    evidence_considered: List[str] = Field(default_factory=list)
    strategic_interpretation: Dict[str, Any] = Field(default_factory=dict)
    gateway_fallback_stub: Optional[str] = Field(default=None, description="Surfaces LLM Gateway's deterministic fallback stub in offline replay trace")
    final_prompt_text: Optional[str] = Field(default=None, description="Alias for gateway_fallback_stub for backward compatibility")
    meeting_gate_open: Optional[bool] = Field(default=None, description="SNAPSHOT-ONLY: Frozen gate status at decision time. Do not use for live reads.")
    conversion_confirmed: Optional[bool] = Field(default=None, description="SNAPSHOT-ONLY: Frozen conversion status at decision time. Do not use for live reads.")
    commitment_slot: Optional[str] = Field(default=None, description="DEPRECATED/SNAPSHOT-ONLY: Frozen slot at decision time. Use resolve_commitment_slot(snapshot) for live reads.")

    def resolve_commitment_slot(self, snapshot: Optional[Any] = None) -> Optional[str]:
        """Dynamically resolves commitment slot from canonical ConversationStateSnapshot via references (Point 12).
        Guarantees that mutating ConversationState doesn't leave divergent stale copies.
        """
        if snapshot is not None:
            if self.referenced_fact_ids:
                for fid in self.referenced_fact_ids:
                    f = snapshot.get_fact_by_id(fid) if hasattr(snapshot, "get_fact_by_id") else next((x for x in getattr(snapshot, "facts", []) if getattr(x, "fact_id", None) == fid), None)
                    if f and getattr(f, "fact_key", None) == "confirmed_meeting_time" and getattr(f, "status", None) == "active":
                        return f.fact_value
            if getattr(snapshot, "conversion_gate", None) and getattr(snapshot.conversion_gate, "commitment_slot", None):
                return snapshot.conversion_gate.commitment_slot
            if hasattr(snapshot, "get_active_conversion_event"):
                conv = snapshot.get_active_conversion_event()
                if conv and getattr(conv, "start_at", None):
                    return conv.start_at
        return self.commitment_slot

    def resolve_meeting_gate_open(self, snapshot: Optional[Any] = None) -> Optional[bool]:
        """Dynamically resolves meeting gate status from canonical ConversationStateSnapshot via references (Point 12)."""
        if snapshot is not None and getattr(snapshot, "conversion_gate", None) is not None:
            return snapshot.conversion_gate.is_open
        return self.meeting_gate_open

    def resolve_conversion_confirmed(self, snapshot: Optional[Any] = None) -> Optional[bool]:
        """Dynamically resolves conversion confirmed status from canonical ConversationStateSnapshot via references (Point 12)."""
        if snapshot is not None and hasattr(snapshot, "get_active_conversion_event"):
            conv = snapshot.get_active_conversion_event()
            if conv is not None:
                return str(conv.status).lower() in ("confirmed", "conversioneventstatus.confirmed")
        return self.conversion_confirmed

    def resolve_fact(self, snapshot: Any, fact_id: str) -> Optional[Any]:
        """Resolves fact by ID reference directly from ConversationStateSnapshot."""
        for f in getattr(snapshot, "facts", []):
            if getattr(f, "fact_id", None) == fact_id:
                return f
        return None

    @model_validator(mode="after")
    def validate_action_gate_push_invariants(self) -> "StrategicDecision":
        import os
        import sys
        import logging

        # Use explicit STRICT_INVARIANT_RAISE setting: in tests raise, in production coerce safely
        strict_flag = os.environ.get("STRICT_INVARIANT_RAISE", "").lower().strip()
        strict_raise = strict_flag in ("1", "true", "yes")

        # Extract gate status if provided via field or strategic_interpretation
        gate_open = self.meeting_gate_open
        if gate_open is None and self.strategic_interpretation:
            gate_open = self.strategic_interpretation.get("meeting_gate_open")

        is_confirmed = self.conversion_confirmed
        if is_confirmed is None and self.strategic_interpretation:
            is_confirmed = self.strategic_interpretation.get("conversion_confirmed")

        violation_reason: Optional[str] = None

        # 1. Closed Gate Constraint: When meeting gate is explicitly closed,
        # close-style pushes (direct_ask, two_window_choice, high) and COMMITMENT_CLOSE are strictly forbidden.
        if gate_open is False:
            if self.primary_action == StrategicAction.COMMITMENT_CLOSE:
                violation_reason = f"primary_action cannot be COMMITMENT_CLOSE when meeting gate is closed (decision_id={self.decision_id})"
            elif self.push_strength in ("direct_ask", "two_window_choice") or (str(self.push_strength) == "high" and self.primary_action == StrategicAction.COMMITMENT_CLOSE):
                violation_reason = f"push_strength cannot be close-style '{self.push_strength}' when meeting gate is closed (decision_id={self.decision_id})"
            elif (self.push_strength == "confirm_and_protect" or self.strategic_posture == "protect") and not is_confirmed and "CONVERSION_CONFIRMED" in self.reason_codes:
                violation_reason = f"push_strength cannot be confirm_and_protect when gate is closed without a confirmed conversion (decision_id={self.decision_id})"

        # 2. Action & Push Harmony: Close-style push recommendations (two_window_choice, direct_ask, high)
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
            StrategicAction.WAIT_SILENCE,
            StrategicAction.HOLD,
        }
        if not violation_reason and self.primary_action in non_closing_actions and (self.push_strength in ("direct_ask", "two_window_choice") or (str(self.push_strength) == "high" and self.primary_action != StrategicAction.COMMITMENT_CLOSE)):
            violation_reason = f"push_strength '{self.push_strength}' cannot accompany non-closing primary_action '{self.primary_action.value}' (decision_id={self.decision_id})"

        # 3. Confirm and Protect requires a confirmed meeting:
        if not violation_reason and (self.push_strength == "confirm_and_protect" or (self.strategic_posture == "protect" and "CONVERSION_CONFIRMED" in self.reason_codes)) and is_confirmed is False:
            violation_reason = f"push_strength 'confirm_and_protect' is invalid without a confirmed appointment (decision_id={self.decision_id})"

        # 4. COMMITMENT_CLOSE requires an affirmative closing push strength or milestone confirmation
        if not violation_reason and self.primary_action == StrategicAction.COMMITMENT_CLOSE:
            if self.push_strength in ("respect_record_exit", "protect_and_shorten", "explore_conditional_terms") or str(self.push_strength) == "none":
                violation_reason = f"primary_action COMMITMENT_CLOSE is incompatible with push_strength '{self.push_strength}' (decision_id={self.decision_id})"

        # 5. Stacking discipline (Point 9): Ensure secondary action always has meaningful justification
        if self.secondary_action is not None:
            if not self.secondary_action_reason:
                self.secondary_action_reason = f"Reinforces '{self.secondary_action.value}' to support primary action '{self.primary_action.value}'"
            else:
                reason = self.secondary_action_reason.strip()
                clean_reason = re.sub(r"[^\w\s]", "", reason.lower()).strip()
                words = clean_reason.split()
                unique_words = set(words)
                weak_placeholders = {
                    "test", "na", "n/a", "none", "secondary", "because", "secondary action",
                    "stacking", "not applicable", "as discussed", "see above", "just because",
                    "no reason", "placeholder reason", "placeholder", "tbd", "todo",
                    "reinforces action", "support primary", "general reason"
                }
                is_placeholder = clean_reason in weak_placeholders or (
                    len(words) <= 4 and any(clean_reason == p or clean_reason.startswith(p + " ") for p in ("not applicable", "placeholder", "as discussed", "see above", "tbd"))
                )
                is_low_entropy = len(set(c for c in clean_reason if not c.isspace())) < 5
                is_repetitive_words = len(words) >= 2 and len(unique_words) < 2
                is_too_short = len(reason) < 15 or len(words) < 2

                if is_too_short or is_placeholder or is_low_entropy or is_repetitive_words:
                    if strict_raise:
                        raise ValueError(
                            f"secondary_action '{self.secondary_action.value}' requires a meaningful justification "
                            f"(at least 15 chars, 2 unique words, non-repetitive, non-placeholder, got: '{reason}')"
                        )
                    else:
                        self.secondary_action_reason = f"Reinforces '{self.secondary_action.value}' to support primary action '{self.primary_action.value}'"

        if violation_reason:
            if strict_raise:
                raise ValueError(f"Invariant violation: {violation_reason}")
            else:
                logging.getLogger(__name__).warning("StrategicDecision invariant violation coerced: %s", violation_reason)
                if self.primary_action == StrategicAction.COMMITMENT_CLOSE:
                    self.primary_action = StrategicAction.QUESTION
                if self.push_strength in ("direct_ask", "two_window_choice", "confirm_and_protect"):
                    self.push_strength = "resolve_then_ask"
                if "INVARIANT_VIOLATION_COERCED" not in self.reason_codes:
                    self.reason_codes.append("INVARIANT_VIOLATION_COERCED")

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
