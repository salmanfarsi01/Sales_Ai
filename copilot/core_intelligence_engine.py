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
    Spec01ObjectionLadderStage,
)

LOGGER = logging.getLogger("copilot.core_intelligence_engine")


class PitchProXCoreIntelligenceEngine:
    """Core intelligence engine interpreting ConversationState and generating StrategicDecision."""

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

        # 1. Compliance and Hard Boundary Gate (Spec 01 §10, Spec 09 §7)
        if context.hard_boundary_active:
            eval_res = self._build_boundary_decision(snapshot, context, turn_timestamp_ms)
            self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
            self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
            return eval_res

        # 2. Confirmed Conversion Protection Gate (Item 9: confirm_and_protect)
        if context.conversion_confirmed or context.push_strength_state == "confirm_and_protect":
            eval_res = self._build_confirm_protect_decision(snapshot, context, turn_timestamp_ms)
            self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
            self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
            return eval_res

        # 3. Active Objection Lifecycle and Spec 01 §6 6-Level Depth Ladder
        unresolved_objections = snapshot.get_unresolved_objections()
        if unresolved_objections:
            eval_res = self._build_objection_decision(snapshot, context, unresolved_objections, turn_timestamp_ms)
            self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
            self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
            return eval_res

        # 4. Multi-Stakeholder and Absent Decision Maker Gate (Spec 09 §3)
        if not context.decision_maker_present:
            eval_res = self._build_absent_stakeholder_decision(snapshot, context, turn_timestamp_ms)
            self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
            self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
            return eval_res

        # 5. Conversion Gate and Push Strength Alignment (Spec 09 §6, §7)
        if context.meeting_gate_open:
            eval_res = self._build_meeting_gate_decision(snapshot, context, turn_timestamp_ms)
            self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
            self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
            return eval_res

        # 6. Stage-Specific and Discovery Fallback Strategy
        eval_res = self._build_stage_default_decision(snapshot, context, turn_timestamp_ms)
        self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
        self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
        return eval_res

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

    def _apply_cross_metric_consistency_rules(
        self,
        snapshot: ConversationStateSnapshot,
        decision: StrategicDecision,
        context: StrategicInterpretationContext,
    ) -> None:
        """Enforces Spec 10 Section 8 cross-metric consistency checks."""
        # Rule 1: Trust high but repeated boundary language -> Boundary wins, lower trust confidence
        if context.hard_boundary_active and context.trust_score > 60.0:
            context.cross_metric_penalties.append("BOUNDARY_OVERRIDES_HIGH_TRUST")
            if "COMPLIANCE_PRIORITY" not in decision.reason_codes:
                decision.reason_codes.append("COMPLIANCE_PRIORITY")

        # Rule 2: Positive valence but momentum falling -> Do not treat friendliness as progress
        if snapshot.dimensions.emotion_valence > 0.20 and context.momentum_trend in ("stalling", "regressing"):
            context.cross_metric_penalties.append("FRIENDLINESS_IS_NOT_PROGRESS")
            if "premature_close" not in decision.do_not_do:
                decision.do_not_do.append("premature_close")
            if "CROSS_METRIC_VALENCE_MOMENTUM_MISMATCH" not in decision.reason_codes:
                decision.reason_codes.append("CROSS_METRIC_VALENCE_MOMENTUM_MISMATCH")

        # Rule 3: Engagement high but objection unresolved -> Continue strategy, do not close blindly
        if snapshot.dimensions.engagement > 0.65 and context.active_objections:
            context.cross_metric_penalties.append("HIGH_ENGAGEMENT_WITH_UNRESOLVED_OBJECTION")
            if "premature_close" not in decision.do_not_do:
                decision.do_not_do.append("premature_close")
            if "OBJECTION_PREVENTS_BLIND_CLOSE" not in decision.reason_codes:
                decision.reason_codes.append("OBJECTION_PREVENTS_BLIND_CLOSE")

        # Rule 4: Pacing alignment high but prospect annoyed/negative -> Pacing does not erase negative sentiment
        if snapshot.dimensions.pacing > 0.70 and snapshot.dimensions.emotion_valence < -0.20:
            context.cross_metric_penalties.append("PACING_DOES_NOT_ERASE_NEGATIVE_VALENCE")
            if "ignore_negative_sentiment" not in decision.do_not_do:
                decision.do_not_do.append("ignore_negative_sentiment")

        # Rule 5: Readiness high but authority/logistics blocked -> Apply blocker cap
        if context.readiness_score > 70.0 and not context.decision_maker_present:
            context.cross_metric_penalties.append("READINESS_CAPPED_BY_ABSENT_AUTHORITY")
            if "press_for_single_party_commitment" not in decision.do_not_do:
                decision.do_not_do.append("press_for_single_party_commitment")

    def _finalize_decision_confidence(
        self,
        snapshot: ConversationStateSnapshot,
        decision: StrategicDecision,
        context: StrategicInterpretationContext,
    ) -> None:
        """Dynamically computes decision confidence per Spec 10 Section 2 and Section 6."""
        breakdown: Dict[str, float] = {}

        # Base confidence from relevant state subsystem
        if decision.primary_action in (StrategicAction.COMMITMENT_CLOSE,):
            base = snapshot.conversion_gate.confidence if snapshot.conversion_gate else 0.80
            breakdown["base_source"] = base
        elif context.active_objections:
            primary_obj = snapshot.get_unresolved_objections()[-1]
            base = primary_obj.confidence
            breakdown["base_source"] = base
        elif context.hard_boundary_active:
            base = 1.0
            breakdown["base_source"] = base
        else:
            base = snapshot.overall_confidence
            breakdown["base_source"] = base

        # Trust and engagement confidence weights
        trust_conf = snapshot.dimensions.trust_confidence
        breakdown["trust_confidence"] = trust_conf

        # Cross-metric consistency penalty deduction
        penalties_count = len(context.cross_metric_penalties)
        penalty_deduction = penalties_count * 0.05
        breakdown["cross_metric_penalty"] = -penalty_deduction

        final_conf = max(0.40, min(1.0, (base * 0.70) + (trust_conf * 0.30) - penalty_deduction))
        breakdown["final_confidence"] = round(final_conf, 3)

        decision.confidence = round(final_conf, 3)
        decision.confidence_breakdown = breakdown

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

        # Spec 01 Section 6 Canonical 6-Level Objection Ladder Logic
        # Ladder: Surface -> Underlying -> First pushback -> Repeated resistance -> Partial resolution -> Resolved
        if primary_obj.lifecycle_state in (ObjectionLifecycleState.PARTIALLY_ADDRESSED, "partially_resolved", "clarified"):
            ladder_stage = Spec01ObjectionLadderStage.PARTIAL_RESOLUTION
            primary_action = StrategicAction.ACKNOWLEDGE
            secondary_action = StrategicAction.CLARIFY
            objective = f"Acknowledge partial alignment and narrow remaining concern on {category}."
            reason_codes = ["OBJECTION_PARTIAL_RESOLUTION", f"LADDER_{ladder_stage.value.upper()}"]
            max_words = 22
        elif recurrence == 1:
            ladder_stage = Spec01ObjectionLadderStage.SURFACE_OBJECTION
            primary_action = StrategicAction.VALIDATE
            secondary_action = StrategicAction.CLARIFY
            objective = f"Validate concern regarding {category} and clarify underlying intent."
            reason_codes = ["OBJECTION_SURFACE_INITIAL", f"LADDER_{ladder_stage.value.upper()}", f"CATEGORY_{category.upper()}"]
            max_words = 22
        elif recurrence == 2 and primary_obj.driver_layer:
            ladder_stage = Spec01ObjectionLadderStage.UNDERLYING_CONCERN
            primary_action = StrategicAction.MIRROR
            secondary_action = StrategicAction.REFRAME
            objective = f"Mirror underlying driver ({primary_obj.driver_layer.underlying_driver}) and reframe strategic target."
            reason_codes = ["OBJECTION_UNDERLYING_DRIVER", f"LADDER_{ladder_stage.value.upper()}"]
            max_words = 26
        elif recurrence == 2:
            ladder_stage = Spec01ObjectionLadderStage.FIRST_PUSHBACK
            if "commission" in category or "fee" in category or "financial" in category:
                primary_action = StrategicAction.REFRAME
                secondary_action = StrategicAction.QUANTIFY
                objective = "Reframe commission cost into net financial proceeds comparison."
                reason_codes = ["OBJECTION_FIRST_PUSHBACK_SHIFT_ANGLE", "NET_PROCEEDS_REFRAME"]
            elif "timing" in category or "market" in category:
                primary_action = StrategicAction.EDUCATE
                secondary_action = StrategicAction.FUTURE_PACE
                objective = "Educate on market timing dynamics and illustrate future scenario."
                reason_codes = ["OBJECTION_FIRST_PUSHBACK_SHIFT_ANGLE", "MARKET_TIMING_EDUCATION"]
            else:
                primary_action = StrategicAction.DIFFERENTIATE
                secondary_action = StrategicAction.CLARIFY
                objective = f"Differentiate approach and isolate primary reservation on {category}."
                reason_codes = ["OBJECTION_FIRST_PUSHBACK_SHIFT_ANGLE", "DIFFERENTIATE_APPROACH"]
            reason_codes.append(f"LADDER_{ladder_stage.value.upper()}")
            max_words = 26
        else:
            ladder_stage = Spec01ObjectionLadderStage.REPEATED_RESISTANCE
            primary_action = StrategicAction.DE_RISK
            if "social_proof" not in failed_strategies:
                secondary_action = StrategicAction.SOCIAL_PROOF
            else:
                secondary_action = StrategicAction.QUESTION

            objective = f"De-risk commitment regarding persistent {category} objection without repeating failed strategies."
            reason_codes = ["OBJECTION_REPEATED_RESISTANCE_BRANCH", f"LADDER_{ladder_stage.value.upper()}"]
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
            confidence=primary_obj.confidence,
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
            confidence=snapshot.decision_structure.confidence,
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

        gate_conf = snapshot.conversion_gate.confidence if snapshot.conversion_gate else 0.90

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
            confidence=gate_conf,
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
            confidence=snapshot.overall_confidence,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)
