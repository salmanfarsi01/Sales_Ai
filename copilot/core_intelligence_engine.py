from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .conversation_state_models import (
    ConversationStateSnapshot,
    ConversationStage,
    ConversionEventStatus,
    ObjectionLifecycleState,
    ObjectionRecord,
    PushStrengthState,
)
from .core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
    StrategicInterpretationContext,
    DecisionEvaluationResult,
)

LOGGER = logging.getLogger("copilot.core_intelligence_engine")


class PitchProXCoreIntelligenceEngine:
    """Core intelligence engine that interprets ConversationState and generates StrategicDecision."""

    def __init__(self, lead_type: str = "general"):
        self.lead_type = lead_type

    def evaluate(
        self,
        snapshot: ConversationStateSnapshot,
        turn_speaker: str = "prospect",
        turn_text: str = "",
        turn_timestamp_ms: int = 0,
    ) -> DecisionEvaluationResult:
        context = self._build_context(snapshot)

        # 1. Compliance and Hard Boundary Gate
        if context.hard_boundary_active:
            return self._build_boundary_decision(snapshot, context, turn_timestamp_ms)

        # 2. Confirmed Conversion Protection Gate
        if context.conversion_confirmed or context.push_strength_state == "confirm_and_protect":
            return self._build_confirm_protect_decision(snapshot, context, turn_timestamp_ms)

        # 3. Active Objection Lifecycle and Branching Progression
        unresolved_objections = snapshot.get_unresolved_objections()
        if unresolved_objections:
            return self._build_objection_decision(snapshot, context, unresolved_objections, turn_timestamp_ms)

        # 4. Multi-Stakeholder and Absent Decision Maker Gate
        if not context.decision_maker_present:
            return self._build_absent_stakeholder_decision(snapshot, context, turn_timestamp_ms)

        # 5. Conversion Gate and Push Strength Alignment
        if context.meeting_gate_open:
            return self._build_meeting_gate_decision(snapshot, context, turn_timestamp_ms)

        # 6. Stage-Specific and Discovery Fallback Strategy
        return self._build_stage_default_decision(snapshot, context, turn_timestamp_ms)

    def _build_context(self, snapshot: ConversationStateSnapshot) -> StrategicInterpretationContext:
        unresolved = snapshot.get_unresolved_objections()
        active_obj_names = [o.canonical_category for o in unresolved]
        failed_by_obj = {o.canonical_category: o.get_failed_strategies() for o in unresolved}

        conv_event = snapshot.get_active_conversion_event()
        is_confirmed = False
        if conv_event and conv_event.status == ConversionEventStatus.CONFIRMED:
            is_confirmed = True

        gate_open = False
        if snapshot.conversion_gate and snapshot.conversion_gate.is_open:
            gate_open = True

        push_strength: PushStrengthState = "resolve_then_ask"
        if snapshot.push_strength:
            push_strength = snapshot.push_strength.state

        readiness_val = snapshot.dimensions.readiness * 100.0
        if snapshot.readiness:
            readiness_val = snapshot.readiness.readiness_score

        mom_trend = "stable"
        if snapshot.momentum:
            mom_trend = snapshot.momentum.trend

        return StrategicInterpretationContext(
            state_version=snapshot.state_version,
            active_objections=active_obj_names,
            failed_strategies_by_objection=failed_by_obj,
            decision_maker_present=snapshot.decision_structure.decision_maker_present,
            hard_boundary_active=snapshot.contact_compliance.hard_boundary_active,
            meeting_gate_open=gate_open,
            conversion_confirmed=is_confirmed,
            push_strength_state=push_strength,
            readiness_score=readiness_val,
            momentum_trend=mom_trend,
            trust_score=snapshot.dimensions.trust * 100.0,
        )

    def _build_boundary_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective="Acknowledge boundary, cease persuasion, and gracefully conclude call.",
            primary_action=StrategicAction.ACKNOWLEDGE,
            secondary_action=StrategicAction.WAIT_SILENCE,
            push_strength="respect_record_exit",
            reason_codes=["HARD_BOUNDARY_ACTIVE", "COMPLIANCE_PRIORITY"],
            do_not_do=["persuade", "pitch", "schedule_meeting", "overcome_boundary"],
            what_to_protect=["legal_compliance", "prospect_boundary", "reputation"],
            question_allowed=False,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=18,
            confidence=1.0,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

    def _build_confirm_protect_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective="Protect confirmed appointment, confirm logistics, and avoid reopening settled concerns.",
            primary_action=StrategicAction.ACKNOWLEDGE,
            secondary_action=StrategicAction.DE_RISK,
            push_strength="confirm_and_protect",
            reason_codes=["CONVERSION_CONFIRMED", "CONFIRM_AND_PROTECT_ACTIVE"],
            do_not_do=["reopen_resolved_objections", "push_for_additional_commitments", "oversell", "prolong_call"],
            what_to_protect=["confirmed_appointment", "established_trust", "agreed_logistics"],
            question_allowed=False,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=24,
            confidence=0.95,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

    def _build_objection_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        unresolved_objections: List[ObjectionRecord],
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        primary_obj = unresolved_objections[-1]
        category = primary_obj.canonical_category.lower()
        recurrence = primary_obj.recurrence_count
        failed_strategies = primary_obj.get_failed_strategies()

        do_not_do = ["premature_close", "dismiss_concern"]
        do_not_do.extend(failed_strategies)
        what_to_protect = ["trust", "rapport"]

        if recurrence == 1:
            primary_action = StrategicAction.VALIDATE
            secondary_action = StrategicAction.CLARIFY
            objective = f"Validate concern regarding {category} and clarify underlying intent."
            reason_codes = ["OBJECTION_SURFACE_INITIAL", f"CATEGORY_{category.upper()}"]
            max_words = 22
        elif recurrence == 2:
            if "commission" in category or "fee" in category or "financial" in category:
                primary_action = StrategicAction.REFRAME
                secondary_action = StrategicAction.QUANTIFY
                objective = "Reframe commission cost into net financial proceeds comparison."
                reason_codes = ["OBJECTION_PUSHBACK_SHIFT_ANGLE", "NET_PROCEEDS_REFRAME"]
            elif "timing" in category or "market" in category:
                primary_action = StrategicAction.EDUCATE
                secondary_action = StrategicAction.FUTURE_PACE
                objective = "Educate on market timing dynamics and illustrate future scenario."
                reason_codes = ["OBJECTION_PUSHBACK_SHIFT_ANGLE", "MARKET_TIMING_EDUCATION"]
            else:
                primary_action = StrategicAction.DIFFERENTIATE
                secondary_action = StrategicAction.CLARIFY
                objective = f"Differentiate approach and isolate primary reservation on {category}."
                reason_codes = ["OBJECTION_PUSHBACK_SHIFT_ANGLE", "DIFFERENTIATE_APPROACH"]
            max_words = 26
        else:
            primary_action = StrategicAction.DE_RISK
            if "social_proof" not in failed_strategies:
                secondary_action = StrategicAction.SOCIAL_PROOF
            else:
                secondary_action = StrategicAction.QUESTION

            objective = f"De-risk commitment regarding persistent {category} objection without repeating failed strategies."
            reason_codes = ["OBJECTION_REPEATED_PROGRESSION_BRANCH", "DE_RISK_PERSISTENT_CONCERN"]
            max_words = 24

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=objective,
            primary_action=primary_action,
            secondary_action=secondary_action,
            push_strength="resolve_then_ask",
            reason_codes=reason_codes,
            do_not_do=do_not_do,
            what_to_protect=what_to_protect,
            question_allowed=True,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=max_words,
            confidence=0.88,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

    def _build_absent_stakeholder_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        stakeholder_roles = [s.role for s in snapshot.decision_structure.stakeholders if s.presence != "on_call"]
        role_label = stakeholder_roles[0] if stakeholder_roles else "partner"

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=f"Align on mutual value and position a collaborative walkthrough including absent {role_label}.",
            primary_action=StrategicAction.DE_RISK,
            secondary_action=StrategicAction.FUTURE_PACE,
            push_strength="resolve_then_ask",
            reason_codes=["DECISION_MAKER_ABSENT", "LOGISTICAL_READINESS_CAPPED"],
            do_not_do=["press_for_single_party_commitment", "ignore_absent_decision_maker", "force_immediate_agreement"],
            what_to_protect=["collaborative_buy_in", "stakeholder_harmony"],
            question_allowed=True,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=24,
            confidence=0.90,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

    def _build_meeting_gate_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        push_state = context.push_strength_state

        if push_state == "two_window_choice":
            primary_action = StrategicAction.COMMITMENT_CLOSE
            secondary_action = StrategicAction.QUESTION
            objective = "Propose two specific calendar windows for the walkthrough."
            reason_codes = ["MEETING_GATE_OPEN", "TWO_WINDOW_CHOICE"]
        else:
            primary_action = StrategicAction.COMMITMENT_CLOSE
            secondary_action = None
            objective = "Directly propose and secure confirmed property walkthrough."
            reason_codes = ["MEETING_GATE_OPEN", "DIRECT_ASK"]

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=objective,
            primary_action=primary_action,
            secondary_action=secondary_action,
            push_strength=push_state,
            reason_codes=reason_codes,
            do_not_do=["reopen_discovery", "hesitate_on_logistics", "apply_manipulative_pressure"],
            what_to_protect=["positive_momentum", "readiness_peak"],
            question_allowed=True,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=22,
            confidence=0.92,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

    def _build_stage_default_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        stage = snapshot.conversation_stage
        trust_score = context.trust_score

        if stage == ConversationStage.DISCOVERY:
            primary_action = StrategicAction.QUESTION
            secondary_action = StrategicAction.MIRROR
            objective = "Uncover prospect goals, situation, and core priorities."
            reason_codes = ["STAGE_DISCOVERY", "EXPLORE_PROSPECT_NEEDS"]
            do_not_do = ["premature_close", "pitch_prematurely"]
            max_words = 20
        elif stage == ConversationStage.VALUE_WALKTHROUGH:
            primary_action = StrategicAction.EDUCATE
            secondary_action = StrategicAction.DIFFERENTIATE
            objective = "Demonstrate tailored value proposition and distinguish approach."
            reason_codes = ["STAGE_VALUE_WALKTHROUGH", "DEMONSTRATE_DIFFERENTIATION"]
            do_not_do = ["overwhelm_with_detail", "press_unready_prospect"]
            max_words = 26
        elif stage == ConversationStage.DECISION_RESOLUTION:
            primary_action = StrategicAction.CLARIFY
            secondary_action = StrategicAction.REFRAME
            objective = "Resolve remaining decision criteria and establish consensus."
            reason_codes = ["STAGE_DECISION_RESOLUTION", "CLARIFY_CRITERIA"]
            do_not_do = ["premature_close"]
            max_words = 24
        elif stage in (ConversationStage.SCHEDULING, ConversationStage.COMMITMENT_CONFIRMED):
            primary_action = StrategicAction.COMMITMENT_CLOSE
            secondary_action = StrategicAction.CLARIFY
            objective = "Coordinate logistical details and calendar commitment."
            reason_codes = ["STAGE_SCHEDULING", "FINALIZE_TIME"]
            do_not_do = ["reopen_discovery"]
            max_words = 20
        else:
            primary_action = StrategicAction.QUESTION
            secondary_action = StrategicAction.ACKNOWLEDGE
            objective = "Engage prospect and build conversation foundation."
            reason_codes = ["STAGE_DEFAULT_ENGAGEMENT"]
            do_not_do = ["premature_close"]
            max_words = 20

        if trust_score < 40.0 and primary_action != StrategicAction.VALIDATE:
            secondary_action = primary_action
            primary_action = StrategicAction.VALIDATE
            reason_codes.append("LOW_TRUST_VALIDATION_INJECTED")

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=objective,
            primary_action=primary_action,
            secondary_action=secondary_action,
            push_strength=context.push_strength_state,
            reason_codes=reason_codes,
            do_not_do=do_not_do,
            what_to_protect=["trust", "conversation_flow"],
            question_allowed=True,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=max_words,
            confidence=0.85,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)
