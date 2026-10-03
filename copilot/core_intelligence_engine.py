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
        turn_id: Optional[int] = None,
    ) -> DecisionEvaluationResult:
        context = self._build_context(snapshot)
        unresolved_objections = snapshot.get_unresolved_objections()

        # 1. Compliance and Hard Boundary Gate (Spec 01 §10, Spec 09 §7)
        if context.hard_boundary_active:
            eval_res = self._build_boundary_decision(snapshot, context, turn_timestamp_ms)
        elif snapshot.contact_compliance and snapshot.contact_compliance.contact_not_before and snapshot.contact_compliance.contact_not_before_turn_id == snapshot.last_updated_turn_id:
            eval_res = self._build_contact_not_before_decision(snapshot, context, turn_timestamp_ms)
        elif context.boundary_suspected:
            eval_res = self._build_boundary_suspected_decision(snapshot, context, turn_timestamp_ms)
        # 2. Confirmed Conversion Protection Gate (Item 9: confirm_and_protect)
        elif context.conversion_confirmed or context.push_strength_state == "confirm_and_protect":
            eval_res = self._build_confirm_protect_decision(snapshot, context, turn_timestamp_ms)
        # 3. Active Objection Lifecycle and Spec 01 §6 6-Level Depth Ladder
        elif unresolved_objections:
            eval_res = self._build_objection_decision(snapshot, context, unresolved_objections, turn_timestamp_ms)
        # 4. Multi-Stakeholder and Absent Decision Maker Gate (Spec 09 §3)
        elif not context.decision_maker_present:
            eval_res = self._build_absent_stakeholder_decision(snapshot, context, turn_timestamp_ms)
        # 5. Conversion Gate and Push Strength Alignment (Spec 09 §6, §7)
        elif context.meeting_gate_open:
            eval_res = self._build_meeting_gate_decision(snapshot, context, turn_timestamp_ms)
        # 6. Stage-Specific and Discovery Fallback Strategy
        else:
            eval_res = self._build_stage_default_decision(snapshot, context, turn_timestamp_ms)

        self._apply_contact_compliance_constraints(snapshot, eval_res.decision, eval_res.context)
        self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
        self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
        self._attach_full_trace(eval_res.decision, eval_res.context, snapshot, turn_speaker, turn_text, turn_id=turn_id)
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
            boundary_suspected=snapshot.contact_compliance.boundary_suspected,
            boundary_retracted=snapshot.contact_compliance.hard_boundary_retracted,
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
        if context.readiness_score is not None and context.readiness_score > 70.0 and not context.decision_maker_present:
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

    def _apply_contact_compliance_constraints(
        self,
        snapshot: ConversationStateSnapshot,
        decision: StrategicDecision,
        context: StrategicInterpretationContext,
    ) -> None:
        """Enforces contact preferences and channel restrictions across all strategic decisions (Spec 01 §10)."""
        comp = snapshot.contact_compliance
        if not comp:
            return

        has_active_preference = False

        # 1. Process structured contact preferences
        for pref in comp.contact_preferences:
            if not pref.allowed or pref.cadence in ("reduced", "specific_times") or pref.prohibited_behavior:
                has_active_preference = True
                if "violating_contact_preference" not in decision.do_not_do:
                    decision.do_not_do.append("violating_contact_preference")
                if "contact_preference" not in decision.what_to_protect:
                    decision.what_to_protect.append("contact_preference")
                if pref.prohibited_behavior:
                    tag = f"prohibited_{pref.prohibited_behavior.replace(' ', '_')}"
                    if tag not in decision.do_not_do:
                        decision.do_not_do.append(tag)
                # Channel specific prohibitions (e.g. email only -> no phone calls)
                if pref.channel == "email":
                    if "prohibited_phone_calls" not in decision.do_not_do:
                        decision.do_not_do.append("prohibited_phone_calls")
                # Timing specific prohibitions (e.g. mornings unavailable / no calls before 10am)
                if pref.time_restriction and any(m in pref.time_restriction.lower() for m in ("morning", "before 10", "before 10am")):
                    if "prohibited_morning_calls" not in decision.do_not_do:
                        decision.do_not_do.append("prohibited_morning_calls")

        # 2. Check facts for contact preferences
        pref_facts = [f for f in snapshot.facts if f.category == "preference" and f.status == "active"]
        if pref_facts:
            has_active_preference = True
        # 3. Check for expired contact friction in prospect memory (Client Audit Item 3)
        # Keeps light constraint: no high-pressure closing, no aggressive scheduling push
        has_past_friction = any(f.fact_key == "past_contact_friction" and f.status == "active" for f in snapshot.facts)
        if has_past_friction:
            has_active_preference = True
            if "high_pressure_closing" not in decision.do_not_do:
                decision.do_not_do.append("high_pressure_closing")
            if "scheduling_push" not in decision.do_not_do:
                decision.do_not_do.append("scheduling_push")
            if "PAST_CONTACT_FRICTION_HONORED" not in decision.reason_codes:
                decision.reason_codes.append("PAST_CONTACT_FRICTION_HONORED")

        # 4. Check time-bounded hold (contact_not_before)
        if comp.contact_not_before:
            has_active_preference = True
            tag = f"prohibited_contact_before_{comp.contact_not_before.lower().replace(' ', '_')}"
            if tag not in decision.do_not_do:
                decision.do_not_do.append(tag)
            if "contact_not_before_hold" not in decision.what_to_protect:
                decision.what_to_protect.append("contact_not_before_hold")
            if "CONTACT_NOT_BEFORE_HONORED" not in decision.reason_codes:
                decision.reason_codes.append("CONTACT_NOT_BEFORE_HONORED")

        if has_active_preference and "CONTACT_PREFERENCE_ENFORCED" not in decision.reason_codes:
            decision.reason_codes.append("CONTACT_PREFERENCE_ENFORCED")

    def _build_contact_not_before_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        """Handles explicit temporal contact holds (e.g. 'until Thursday'): acknowledges hold respectfully."""
        comp = snapshot.contact_compliance
        hold_target = comp.contact_not_before if comp else "requested date"
        clean_tag = hold_target.lower().replace(' ', '_')
        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=f"Acknowledge the prospect's requested contact hold until {hold_target} and gracefully confirm timing.",
            primary_action=StrategicAction.ACKNOWLEDGE,
            secondary_action=None,
            push_strength="protect_and_shorten",
            reason_codes=["CONTACT_NOT_BEFORE_DECLARED", "HOLD_RESPECTED"],
            do_not_do=["press_for_earlier_time", "premature_close", f"prohibited_contact_before_{clean_tag}"],
            what_to_protect=["contact_not_before_hold", "prospect_trust"],
            question_allowed=False,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=20,
            confidence=0.95,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

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
            secondary_action=None,
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

    def _build_boundary_suspected_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        """Handles ambiguous boundary signals: stops persuading and issues a short clarify (Spec 01 §10)."""
        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective="Stop persuading and issue a short clarify to confirm prospect comfort and boundaries.",
            primary_action=StrategicAction.CLARIFY,
            secondary_action=None,
            push_strength="respect_record_exit",
            reason_codes=["BOUNDARY_SUSPECTED", "STOP_PERSUADING_CLARIFY"],
            do_not_do=["persuade", "pitch", "schedule_meeting", "overcome_boundary", "apply_pressure"],
            what_to_protect=["prospect_comfort", "conversation_safety", "legal_compliance"],
            question_allowed=True,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=16,
            confidence=0.85,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

    def _build_confirm_protect_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        slot = None
        if snapshot.conversion_gate and snapshot.conversion_gate.commitment_slot:
            slot = snapshot.conversion_gate.commitment_slot
        elif conv := snapshot.get_active_conversion_event():
            slot = conv.start_at
        if not slot:
            meeting_fact = snapshot.get_active_fact("confirmed_meeting_time")
            if meeting_fact:
                slot = meeting_fact.fact_value

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            commitment_slot=slot,
            should_prompt=True,
            strategic_objective="Protect confirmed appointment, confirm logistics, and avoid reopening settled concerns.",
            primary_action=StrategicAction.ACKNOWLEDGE,
            secondary_action=StrategicAction.DE_RISK,
            push_strength="confirm_and_protect",
            meeting_gate_open=context.meeting_gate_open,
            conversion_confirmed=True,
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
        gate = snapshot.conversion_gate
        unknown_conds = getattr(gate, "unknown_conditions", []) if gate else []
        gate_conf = gate.confidence if gate else 0.0

        # Don't choose COMMITMENT_CLOSE on a gate with low confidence or unknown conditions.
        # Route to QUESTION or CLARIFY to discover the missing piece.
        if unknown_conds or gate_conf < 0.60 or not (gate and gate.is_open):
            missing_cond = unknown_conds[0] if unknown_conds else "readiness_criteria"
            if missing_cond == "plausible_logistics":
                primary_action = StrategicAction.QUESTION
                secondary_action = StrategicAction.CLARIFY
                objective = "Explore prospect scheduling preferences and logistical availability."
                reason_codes = ["GATE_UNKNOWN_LOGISTICS", "DISCOVER_MISSING_GATE_PIECE"]
            elif missing_cond in ("clear_value_reason", "problem_pain_acknowledged"):
                primary_action = StrategicAction.QUESTION
                secondary_action = StrategicAction.EDUCATE
                objective = "Discover prospect priorities and establish clear value before closing."
                reason_codes = ["GATE_UNKNOWN_VALUE_REASON", "DISCOVER_MISSING_GATE_PIECE"]
            elif missing_cond == "decision_maker_aligned":
                primary_action = StrategicAction.CLARIFY
                secondary_action = StrategicAction.QUESTION
                objective = "Clarify stakeholder involvement and decision process."
                reason_codes = ["GATE_UNKNOWN_DECISION_MAKER", "DISCOVER_MISSING_GATE_PIECE"]
            else:
                primary_action = StrategicAction.CLARIFY
                secondary_action = StrategicAction.QUESTION
                objective = f"Clarify missing gate criteria ({missing_cond}) before attempting commitment close."
                reason_codes = ["GATE_LOW_CONFIDENCE_OR_UNKNOWN", "DISCOVER_MISSING_GATE_PIECE"]

            decision = StrategicDecision(
                call_id=snapshot.call_sid,
                source_state_version=snapshot.state_version,
                should_prompt=True,
                strategic_objective=objective,
                primary_action=primary_action,
                secondary_action=secondary_action,
                push_strength="resolve_then_ask",
                reason_codes=reason_codes,
                do_not_do=["premature_close", "blind_commitment_close", "apply_manipulative_pressure"],
                what_to_protect=["trust", "conversation_flow"],
                question_allowed=True,
                retrieval_needed=False,
                urgency="immediate",
                max_prompt_words=20,
                confidence=gate_conf,
                created_at_ms=turn_timestamp_ms,
            )
            return DecisionEvaluationResult(decision=decision, context=context)

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
            gate = snapshot.conversion_gate
            gate_ready = bool(gate and gate.is_open and not getattr(gate, "unknown_conditions", []) and (gate.confidence >= 0.60))
            if not gate_ready:
                primary_action = StrategicAction.QUESTION
                secondary_action = StrategicAction.CLARIFY
                objective = "Discover scheduling preferences and uncover logistical details."
                reason_codes = ["STAGE_SCHEDULING", "GATE_NOT_READY_DISCOVERY"]
                do_not_do = ["premature_close", "blind_commitment_close"]
                max_words = 20
            else:
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

        # Invariant: Non-closing discovery/clarify actions cannot carry a close-style push recommendation
        push_st = context.push_strength_state
        if primary_action in (
            StrategicAction.QUESTION,
            StrategicAction.CLARIFY,
            StrategicAction.VALIDATE,
            StrategicAction.EDUCATE,
            StrategicAction.ACKNOWLEDGE,
            StrategicAction.REFRAME,
            StrategicAction.MIRROR,
            StrategicAction.DIFFERENTIATE,
        ) and push_st in ("two_window_choice", "direct_ask"):
            push_st = "resolve_then_ask"

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=objective,
            primary_action=primary_action,
            secondary_action=secondary_action,
            push_strength=push_st,
            meeting_gate_open=context.meeting_gate_open,
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

    def _attach_full_trace(
        self,
        decision: StrategicDecision,
        context: StrategicInterpretationContext,
        snapshot: ConversationStateSnapshot,
        turn_speaker: str,
        turn_text: str,
        turn_id: Optional[int] = None,
    ) -> None:
        """Attaches full explainability trace to StrategicDecision: Evidence -> Interpretation -> Decision Contract.
        Client Principle: Core Intelligence strictly outputs structural decisions and constraints;
        it does NOT generate spoken teleprompter copy.
        """
        effective_turn_id = turn_id if turn_id is not None else getattr(snapshot, "last_updated_turn_id", 0)
        decision.call_sid = snapshot.call_sid
        decision.call_id = snapshot.call_sid
        decision.source_state_version = snapshot.state_version
        decision.source_turn_id = effective_turn_id
        decision.utterance_turn_id = effective_turn_id
        decision.metrics_source_turn_id = effective_turn_id
        decision.source_event_id = f"ev_turn_{effective_turn_id}_v{snapshot.state_version}"
        if getattr(decision, "commitment_slot", None) is None:
            if snapshot.conversion_gate and snapshot.conversion_gate.commitment_slot:
                decision.commitment_slot = snapshot.conversion_gate.commitment_slot
            elif conv := snapshot.get_active_conversion_event():
                decision.commitment_slot = conv.start_at
            else:
                meeting_fact = snapshot.get_active_fact("confirmed_meeting_time")
                if meeting_fact:
                    decision.commitment_slot = meeting_fact.fact_value

        evidence: List[str] = []
        if turn_text:
            cleaned_text = turn_text.strip().replace("\n", " ")
            evidence.append(f'Turn utterance ({turn_speaker}): "{cleaned_text[:80]}"')

        # Differentiate measured trust from default baseline (Client Group 1 Point 5)
        is_trust_measured = getattr(snapshot.dimensions, "trust_measured", False) or (context.trust_score != 50.0)
        if is_trust_measured:
            evidence.append(f"Trust: {context.trust_score:.0f}% (measured, confidence: {snapshot.dimensions.trust_confidence:.2f})")
        else:
            evidence.append(f"Trust: {context.trust_score:.0f}% (default, unmeasured)")

        # Insufficient evidence display: Show UNKNOWN instead of misleading 0.0%
        has_insufficient_ev = (snapshot.readiness and snapshot.readiness.insufficient_evidence) or (snapshot.readiness and snapshot.readiness.readiness_score is None)
        if has_insufficient_ev:
            evidence.append("Readiness: UNKNOWN (Insufficient Evidence)")
        else:
            evidence.append(f"Readiness: {context.readiness_score:.1f}%")

        evidence.append(f"Momentum: {context.momentum_trend}")
        if context.active_objections:
            evidence.append(f"Active objections: {context.active_objections}")
        if snapshot.conversion_gate:
            gate_st = "OPEN" if snapshot.conversion_gate.is_open else "CLOSED"
            unk = getattr(snapshot.conversion_gate, "unknown_conditions", [])
            evidence.append(f"Gate: {gate_st} (unknown: {unk if unk else 'none'})")
        if snapshot.contact_compliance:
            if snapshot.contact_compliance.hard_boundary_active:
                evidence.append("Hard boundary: ACTIVE")
            elif snapshot.contact_compliance.boundary_suspected:
                evidence.append("Boundary suspected: AMBIGUOUS")
            if snapshot.contact_compliance.contact_preferences:
                evidence.append(f"Contact preferences: {len(snapshot.contact_compliance.contact_preferences)} active")

        decision.evidence_considered = evidence
        decision.meeting_gate_open = context.meeting_gate_open
        decision.conversion_confirmed = context.conversion_confirmed
        decision.strategic_interpretation = {
            "conversation_stage": str(snapshot.conversation_stage.value if hasattr(snapshot.conversation_stage, "value") else snapshot.conversation_stage),
            "meeting_gate_open": context.meeting_gate_open,
            "conversion_confirmed": context.conversion_confirmed,
            "decision_maker_present": context.decision_maker_present,
            "hard_boundary_active": context.hard_boundary_active,
            "boundary_suspected": context.boundary_suspected,
            "push_strength_state": context.push_strength_state,
            "active_objections": context.active_objections,
            "readiness_score": None if has_insufficient_ev else round(context.readiness_score, 1),
            "momentum_trend": context.momentum_trend,
            "cross_metric_penalties": context.cross_metric_penalties,
        }
