from __future__ import annotations

import os
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
    PushStrengthValue,
    StrategicInterpretationContext,
    DecisionEvaluationResult,
    Spec01ObjectionLadderStage,
)

LOGGER = logging.getLogger("copilot.core_intelligence_engine")

DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_UNMEASURED_TRUST_CAP = 0.45
DEFAULT_MIN_CONFIDENCE_FLOOR = 0.40
DEFAULT_MOMENTUM_ADJUSTMENTS: Dict[str, float] = {
    "advancing": 0.04,
    "stable": 0.0,
    "stalling": -0.04,
    "regressing": -0.08,
}


class PitchProXCoreIntelligenceEngine:
    """Core intelligence engine interpreting ConversationState and generating StrategicDecision."""

    def __init__(
        self,
        lead_type: str = "general",
        confidence_threshold: Optional[float] = None,
        unmeasured_trust_cap: Optional[float] = None,
        momentum_adjustments: Optional[Dict[str, float]] = None,
        min_confidence_floor: Optional[float] = None,
    ):
        self.lead_type = lead_type
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)
        else:
            self.confidence_threshold = float(os.environ.get("CORE_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)))
        self.unmeasured_trust_cap = float(unmeasured_trust_cap) if unmeasured_trust_cap is not None else DEFAULT_UNMEASURED_TRUST_CAP
        self.momentum_adjustments = dict(momentum_adjustments) if momentum_adjustments is not None else dict(DEFAULT_MOMENTUM_ADJUSTMENTS)
        self.min_confidence_floor = float(min_confidence_floor) if min_confidence_floor is not None else DEFAULT_MIN_CONFIDENCE_FLOOR

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
        # 1b. Turn-level contact preference or boundary declared on this turn
        elif self._is_contact_preference_turn(snapshot, turn_text, turn_id):
            eval_res = self._build_contact_preference_decision(snapshot, context, turn_timestamp_ms)
        # 2. Confirmed Conversion Protection Gate (Item 9: confirm_and_protect)
        elif context.conversion_confirmed or context.push_strength_state == "confirm_and_protect":
            eval_res = self._build_confirm_protect_decision(snapshot, context, turn_timestamp_ms)
        # 3. Multi-Stakeholder and Absent Decision Maker Gate (Spec 09 §3)
        elif not context.decision_maker_present and (
            snapshot.conversation_stage == ConversationStage.DECISION_RESOLUTION
            or any(w in (turn_text or "").lower() for w in ("wife", "husband", "spouse", "partner", "decision maker", "sign off", "alone", "both of us", "consult", "lawyer", "attorney", "divorce"))
            or (unresolved_objections and any(o.canonical_category == "spouse_authority" for o in unresolved_objections) and not any(o.canonical_category != "spouse_authority" and o.lifecycle_state in (ObjectionLifecycleState.ACTIVE, "active") for o in unresolved_objections))
            or (turn_id is None and not turn_text)
        ):
            eval_res = self._build_absent_stakeholder_decision(snapshot, context, turn_timestamp_ms, turn_text=turn_text)
        # 4. Active Objection Lifecycle and Spec 01 §6 6-Level Depth Ladder
        elif unresolved_objections and (
            any(o.lifecycle_state in (ObjectionLifecycleState.ACTIVE, "active", "reactivated") for o in unresolved_objections)
            or snapshot.conversation_stage == ConversationStage.OBJECTION_HANDLING
            or (
                any(o.lifecycle_state in (ObjectionLifecycleState.PARTIALLY_ADDRESSED, "partially_resolved", "clarified") for o in unresolved_objections)
                and snapshot.conversation_stage not in (ConversationStage.SCHEDULING, ConversationStage.COMMITMENT_CONFIRMED, ConversationStage.VALUE_WALKTHROUGH)
            )
        ):
            eval_res = self._build_objection_decision(snapshot, context, unresolved_objections, turn_timestamp_ms)
        # 5. Conversion Gate and Push Strength Alignment (Spec 09 §6, §7)
        elif context.meeting_gate_open:
            eval_res = self._build_meeting_gate_decision(snapshot, context, turn_timestamp_ms)
        # 6. Stage-Specific and Discovery Fallback Strategy
        else:
            eval_res = self._build_stage_default_decision(snapshot, context, turn_timestamp_ms)

        self._apply_contact_compliance_constraints(snapshot, eval_res.decision, eval_res.context, turn_text=turn_text, turn_id=turn_id)
        self._apply_cross_metric_consistency_rules(snapshot, eval_res.decision, eval_res.context)
        self._finalize_decision_confidence(snapshot, eval_res.decision, eval_res.context)
        self._attach_full_trace(eval_res.decision, eval_res.context, snapshot, turn_speaker, turn_text, turn_id=turn_id)
        if turn_speaker in ("salesperson", "rep", "agent"):
            eval_res.decision.should_prompt = False
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
        elif "CONVERSION_CONFIRMED" in decision.reason_codes or "CONFIRM_AND_PROTECT_ACTIVE" in decision.reason_codes:
            conv = snapshot.get_active_conversion_event()
            if conv and conv.confirmation_confidence > 0.0:
                base = conv.confirmation_confidence
            elif snapshot.conversion_gate and snapshot.conversion_gate.confidence > 0.0:
                base = snapshot.conversion_gate.confidence
            else:
                base = 0.90
            breakdown["base_source"] = base
        elif any("OBJECTION" in r or "LADDER" in r for r in decision.reason_codes) or decision.primary_action in (
            StrategicAction.VALIDATE, StrategicAction.MIRROR, StrategicAction.REFRAME,
            StrategicAction.QUANTIFY, StrategicAction.DIFFERENTIATE, StrategicAction.DE_RISK
        ):
            primary_obj = snapshot.get_unresolved_objections()[-1] if snapshot.get_unresolved_objections() else None
            base = primary_obj.confidence if primary_obj else snapshot.overall_confidence
            if primary_obj and primary_obj.recurrence_count > 1:
                base = max(0.50, base - (0.08 * (primary_obj.recurrence_count - 1)))
            breakdown["base_source"] = base
        elif any("DECISION_MAKER_ABSENT" in r for r in decision.reason_codes):
            base = snapshot.decision_structure.confidence
            breakdown["base_source"] = base
        elif any("CONTACT" in r for r in decision.reason_codes):
            base = 0.85
            breakdown["base_source"] = base
        elif context.hard_boundary_active:
            base = 1.0
            breakdown["base_source"] = base
        else:
            base = snapshot.overall_confidence
            breakdown["base_source"] = base

        # Trust confidence weight: if trust is unmeasured, discount its contribution
        # Exemption: Explicitly confirmed conversions (Point 5 / Point 10 harmony)
        is_confirmed_conversion = (
            context.conversion_confirmed
            or "CONVERSION_CONFIRMED" in decision.reason_codes
            or "CONFIRM_AND_PROTECT_ACTIVE" in decision.reason_codes
        )
        is_trust_measured = getattr(snapshot.dimensions, "trust_measured", False) or (context.trust_score != 50.0) or is_confirmed_conversion
        trust_conf = snapshot.dimensions.trust_confidence if is_trust_measured else min(self.unmeasured_trust_cap, snapshot.dimensions.trust_confidence)
        breakdown["trust_confidence"] = trust_conf

        # Momentum adjustment (configurable via momentum_adjustments dict)
        momentum_adj = self.momentum_adjustments.get(context.momentum_trend, 0.0)
        breakdown["momentum_adjustment"] = momentum_adj

        # Cross-metric consistency penalty deduction
        penalties_count = len(context.cross_metric_penalties)
        penalty_deduction = penalties_count * 0.05
        breakdown["cross_metric_penalty"] = -penalty_deduction

        final_conf = max(self.min_confidence_floor, min(1.0, (base * 0.70) + (trust_conf * 0.30) - penalty_deduction + momentum_adj))
        final_conf = round(final_conf, 4)
        breakdown["final_confidence"] = final_conf

        decision.confidence = final_conf
        decision.confidence_breakdown = breakdown

        # Point 10: Confidence must change behavior, not just be a displayed number.
        # Below confidence threshold (configurable, default 0.65), constrain action selection to lower-risk options.
        if final_conf < (self.confidence_threshold - 1e-9):
            downgraded = False
            if decision.primary_action == StrategicAction.COMMITMENT_CLOSE:
                decision.primary_action = StrategicAction.CLARIFY
                decision.strategic_posture = "explore"
                decision.push_strength = PushStrengthValue("low", legacy_alias="resolve_then_ask")
                decision.secondary_action = StrategicAction.QUESTION
                decision.secondary_action_reason = "Inquire regarding comfort level before advancing."
                decision.strategic_objective = "Clarify prospect alignment and verify comfort before attempting commitment due to lower confidence."
                downgraded = True
            elif "CONFIRM_AND_PROTECT_ACTIVE" in decision.reason_codes:
                # Point 10: In an otherwise 'confirm' scenario with low confidence, downgrade to CLARIFY to verify understanding
                decision.primary_action = StrategicAction.CLARIFY
                decision.strategic_posture = "explore"
                decision.push_strength = PushStrengthValue("low", legacy_alias="resolve_then_ask")
                decision.secondary_action = StrategicAction.QUESTION
                decision.secondary_action_reason = "Verify understanding and confirm appointment details due to low confidence."
                decision.strategic_objective = "Clarify and verify agreed appointment details before locking confirmation due to lower confidence."
                downgraded = True
            elif decision.primary_action == StrategicAction.CHALLENGE:
                decision.primary_action = StrategicAction.QUESTION
                decision.strategic_posture = "explore"
                decision.secondary_action = StrategicAction.CLARIFY
                decision.secondary_action_reason = "Explore prospect viewpoint without confrontational challenge under low confidence."
                decision.strategic_objective = "Inquire gently into prospect perspective rather than challenging under lower confidence."
                downgraded = True

            # Cap high or moderate push strength when confidence is low
            # Only count as downgrade if push strength actually changed from a higher pressure level
            old_push_str = str(decision.push_strength)
            if old_push_str in ("high", "moderate") or decision.push_strength in ("direct_ask", "two_window_choice"):
                if old_push_str not in ("low", "none"):
                    decision.push_strength = PushStrengthValue("low")
                    downgraded = True

            if downgraded:
                if "LOW_CONFIDENCE_ACTION_DOWNGRADE" not in decision.reason_codes:
                    decision.reason_codes.append("LOW_CONFIDENCE_ACTION_DOWNGRADE")
                if "premature_close_on_low_confidence" not in decision.do_not_do:
                    decision.do_not_do.append("premature_close_on_low_confidence")
                if "aggressive_push_on_low_confidence" not in decision.do_not_do:
                    decision.do_not_do.append("aggressive_push_on_low_confidence")

    def _apply_contact_compliance_constraints(
        self,
        snapshot: ConversationStateSnapshot,
        decision: StrategicDecision,
        context: StrategicInterpretationContext,
        turn_text: str = "",
        turn_id: Optional[int] = None,
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

        # Point 11: Only add CONTACT_PREFERENCE_ENFORCED if this turn is actively enforcing/declaring contact compliance
        # Active contact preferences inject constraints into do_not_do and what_to_protect,
        # but reason codes should only flag enforcement when not on a confirmed appointment close (Turn 18).
        is_confirmed_close = (
            context.conversion_confirmed
            or snapshot.conversation_stage == ConversationStage.COMMITMENT_CONFIRMED
            or "CONFIRM_AND_PROTECT_ACTIVE" in decision.reason_codes
            or "CONVERSION_CONFIRMED" in decision.reason_codes
        )
        is_contact_turn = (
            "CONTACT_PREFERENCE_DECLARED" in decision.reason_codes
            or "CONTACT_NOT_BEFORE_HONORED" in decision.reason_codes
            or self._is_contact_preference_turn(snapshot, turn_text, turn_id)
            or (turn_id is None and not turn_text and not snapshot.objections and has_active_preference)
        )
        if has_active_preference and is_contact_turn and not is_confirmed_close and "CONTACT_PREFERENCE_ENFORCED" not in decision.reason_codes:
            decision.reason_codes.append("CONTACT_PREFERENCE_ENFORCED")

    def _is_contact_preference_turn(
        self,
        snapshot: ConversationStateSnapshot,
        turn_text: str,
        turn_id: Optional[int],
    ) -> bool:
        effective_turn_id = turn_id if turn_id is not None else getattr(snapshot, "last_updated_turn_id", 0)
        comp = snapshot.contact_compliance
        if comp and any(
            getattr(p, "source_turn_id", None) == effective_turn_id
            and (not p.allowed or p.cadence != "no_preference" or p.time_restriction or p.prohibited_behavior)
            for p in comp.contact_preferences
        ):
            return True
        text_lower = (turn_text or "").lower()
        if any(w in text_lower for w in ("text", "texting", "call", "calling", "email", "contact")) and any(neg in text_lower for neg in ("don't", "dont", "do not", "please don't", "stop", "never")):
            return True
        return False

    def _build_contact_preference_decision(
        self,
        snapshot: ConversationStateSnapshot,
        context: StrategicInterpretationContext,
        turn_timestamp_ms: int,
    ) -> DecisionEvaluationResult:
        ref_facts = [f.fact_id for f in snapshot.facts if f.category == "preference" and f.status == "active"]
        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective="Acknowledge contact preference respectfully and protect prospect communication boundaries.",
            primary_action=StrategicAction.ACKNOWLEDGE,
            strategic_posture="protect",
            secondary_action=None,
            secondary_action_reason=None,
            push_strength=PushStrengthValue("none", legacy_alias="respect_record_exit"),
            referenced_fact_ids=ref_facts,
            reason_codes=["CONTACT_PREFERENCE_DECLARED", "PROTECT_PROSPECT_PREFERENCE"],
            do_not_do=["violating_contact_preference", "high_pressure_closing", "scheduling_push"],
            what_to_protect=["contact_preference", "prospect_trust"],
            question_allowed=False,
            retrieval_needed=False,
            urgency="immediate",
            max_prompt_words=18,
            confidence=0.90,
            created_at_ms=turn_timestamp_ms,
        )
        return DecisionEvaluationResult(decision=decision, context=context)

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
        ref_facts = [f.fact_id for f in snapshot.facts if f.category == "preference" and f.status == "active"]
        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=f"Acknowledge the prospect's requested contact hold until {hold_target} and gracefully confirm timing.",
            primary_action=StrategicAction.ACKNOWLEDGE,
            strategic_posture="protect",
            secondary_action=None,
            push_strength=PushStrengthValue("none", legacy_alias="protect_and_shorten"),
            referenced_fact_ids=ref_facts,
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
            strategic_posture="defend",
            secondary_action=None,
            push_strength=PushStrengthValue("none", legacy_alias="respect_record_exit"),
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
        """Handles suspected boundary signals. Compliance negative imperatives route to ACKNOWLEDGE / protect / none."""
        suspected_reason = getattr(snapshot.contact_compliance, "boundary_suspected_reason", "") or ""
        is_ambiguous_hesitation = "Ambiguous hesitation detected" in suspected_reason

        if is_ambiguous_hesitation:
            decision = StrategicDecision(
                call_id=snapshot.call_sid,
                source_state_version=snapshot.state_version,
                should_prompt=True,
                strategic_objective="Stop persuading and issue a short clarify to confirm prospect comfort and boundaries.",
                primary_action=StrategicAction.CLARIFY,
                strategic_posture="defend",
                secondary_action=None,
                push_strength=PushStrengthValue("low", legacy_alias="respect_record_exit"),
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
        else:
            decision = StrategicDecision(
                call_id=snapshot.call_sid,
                source_state_version=snapshot.state_version,
                should_prompt=True,
                strategic_objective="Acknowledge boundary or contact friction respectfully and protect prospect communication limits.",
                primary_action=StrategicAction.ACKNOWLEDGE,
                strategic_posture="protect",
                secondary_action=None,
                push_strength=PushStrengthValue("none", legacy_alias="respect_record_exit"),
                reason_codes=["BOUNDARY_SUSPECTED", "COMPLIANCE_PRIORITY", "PROTECT_PROSPECT_PREFERENCE"],
                do_not_do=["persuade", "pitch", "schedule_meeting", "overcome_boundary", "apply_pressure", "violating_contact_preference"],
                what_to_protect=["prospect_comfort", "conversation_safety", "legal_compliance", "contact_preference"],
                question_allowed=False,
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
        meeting_fact = snapshot.get_active_fact("confirmed_meeting_time")
        if not slot and meeting_fact:
            slot = meeting_fact.fact_value

        ref_facts = [meeting_fact.fact_id] if meeting_fact else []

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            commitment_slot=slot,
            should_prompt=True,
            strategic_objective="Protect confirmed appointment, confirm logistics, and avoid reopening settled concerns.",
            primary_action=StrategicAction.ACKNOWLEDGE,
            strategic_posture="protect",
            secondary_action=None,
            secondary_action_reason=None,
            push_strength=PushStrengthValue("none", legacy_alias="confirm_and_protect"),
            meeting_gate_open=context.meeting_gate_open,
            conversion_confirmed=True,
            referenced_fact_ids=ref_facts,
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
        secondary_reason = None
        if primary_obj.lifecycle_state in (ObjectionLifecycleState.PARTIALLY_ADDRESSED, "partially_resolved", "clarified"):
            ladder_stage = Spec01ObjectionLadderStage.PARTIAL_RESOLUTION
            primary_action = StrategicAction.ACKNOWLEDGE
            secondary_action = None
            secondary_reason = None
            objective = f"Acknowledge partial alignment and narrow remaining concern on {category}."
            reason_codes = ["OBJECTION_PARTIAL_RESOLUTION", f"LADDER_{ladder_stage.value.upper()}"]
            max_words = 22
        elif recurrence == 1:
            ladder_stage = Spec01ObjectionLadderStage.SURFACE_OBJECTION
            primary_action = StrategicAction.VALIDATE
            secondary_action = StrategicAction.CLARIFY
            secondary_reason = f"Clarify underlying {category} context while validating prospect perspective."
            objective = f"Validate concern regarding {category} and clarify underlying intent."
            reason_codes = ["OBJECTION_SURFACE_INITIAL", f"LADDER_{ladder_stage.value.upper()}", f"CATEGORY_{category.upper()}"]
            max_words = 22
        elif recurrence == 2 and primary_obj.driver_layer:
            ladder_stage = Spec01ObjectionLadderStage.UNDERLYING_CONCERN
            primary_action = StrategicAction.MIRROR
            secondary_action = StrategicAction.REFRAME
            secondary_reason = f"Reframe strategic perspective after mirroring core {category} concern."
            objective = f"Mirror underlying driver ({primary_obj.driver_layer.underlying_driver}) and reframe strategic target."
            reason_codes = ["OBJECTION_UNDERLYING_DRIVER", f"LADDER_{ladder_stage.value.upper()}"]
            max_words = 26
        elif recurrence == 2:
            ladder_stage = Spec01ObjectionLadderStage.FIRST_PUSHBACK
            if "commission" in category or "fee" in category or "financial" in category:
                primary_action = StrategicAction.REFRAME
                secondary_action = StrategicAction.QUANTIFY
                secondary_reason = "Quantify net financial proceeds to concretely support reframe."
                objective = "Reframe commission cost into net financial proceeds comparison."
                reason_codes = ["OBJECTION_FIRST_PUSHBACK_SHIFT_ANGLE", "NET_PROCEEDS_REFRAME"]
            elif "timing" in category or "market" in category:
                primary_action = StrategicAction.EDUCATE
                secondary_action = StrategicAction.FUTURE_PACE
                secondary_reason = "Illustrate future market timing scenarios."
                objective = "Educate on market timing dynamics and illustrate future scenario."
                reason_codes = ["OBJECTION_FIRST_PUSHBACK_SHIFT_ANGLE", "MARKET_TIMING_EDUCATION"]
            else:
                primary_action = StrategicAction.DIFFERENTIATE
                secondary_action = None
                secondary_reason = None
                objective = f"Differentiate approach and isolate primary reservation on {category}."
                reason_codes = ["OBJECTION_FIRST_PUSHBACK_SHIFT_ANGLE", "DIFFERENTIATE_APPROACH"]
            reason_codes.append(f"LADDER_{ladder_stage.value.upper()}")
            max_words = 26
        else:
            ladder_stage = Spec01ObjectionLadderStage.REPEATED_RESISTANCE
            primary_action = StrategicAction.DE_RISK
            if "social_proof" not in failed_strategies:
                secondary_action = StrategicAction.SOCIAL_PROOF
                secondary_reason = "Provide verified references to de-risk persistent concern."
            else:
                secondary_action = StrategicAction.QUESTION
                secondary_reason = f"Inquire into root blocker on persistent {category} objection."

            objective = f"De-risk commitment regarding persistent {category} objection without repeating failed strategies."
            reason_codes = ["OBJECTION_REPEATED_RESISTANCE_BRANCH", f"LADDER_{ladder_stage.value.upper()}"]
            max_words = 24

        # Point 14: Use strategy history to avoid repeating failed approaches
        failed_strategy_names = [str(s).lower() for s in failed_strategies]
        if primary_action.value.lower() in failed_strategy_names:
            failed_tactic = primary_action.value
            prior_outcome = next((o for o in reversed(primary_obj.strategy_outcomes) if o.strategy_tag.lower() == failed_tactic.lower()), None)

            # Exception: "unless new evidence justifies retrying"
            # If new substantive evidence/facts arrived since the attempt, retrying may be permitted
            has_new_evidence = False
            if prior_outcome and prior_outcome.attempted_at_turn_id:
                new_facts = [f for f in snapshot.facts if f.source_turn_id is not None and f.source_turn_id > prior_outcome.attempted_at_turn_id and f.status == "active"]
                if new_facts:
                    has_new_evidence = True

            if has_new_evidence:
                objective = f"Retrying '{failed_tactic}' with new evidence on {category} from subsequent conversation turns."
                reason_codes.append("RETRY_FAILED_STRATEGY_JUSTIFIED_BY_NEW_EVIDENCE")
                if prior_outcome and prior_outcome.attempted_at_turn_id:
                    reason_codes.append(f"PREVIOUS_ATTEMPT_TURN_{prior_outcome.attempted_at_turn_id}")
            else:
                candidates = [
                    (StrategicAction.QUANTIFY, "Quantify empirical differences and net financial value."),
                    (StrategicAction.DIFFERENTIATE, "Differentiate service model and structural methodology."),
                    (StrategicAction.DE_RISK, "De-risk commitment and remove downside exposure."),
                    (StrategicAction.CLARIFY, "Clarify underlying concern to discover core driver."),
                    (StrategicAction.QUESTION, "Inquire directly regarding remaining hesitations."),
                    (StrategicAction.VALIDATE, "Validate prospect perspective before advancing."),
                ]
                pivoted = False
                for candidate_action, candidate_obj in candidates:
                    if candidate_action.value.lower() not in failed_strategy_names and candidate_action != primary_action:
                        primary_action = candidate_action
                        attempt_str = f"Turn {prior_outcome.attempted_at_turn_id}" if prior_outcome and prior_outcome.attempted_at_turn_id else "earlier attempt"
                        outcome_str = f"outcome: '{prior_outcome.prospect_response_summary}'" if prior_outcome and prior_outcome.prospect_response_summary else "ineffective outcome"
                        objective = f"Pivoted from previously failed '{failed_tactic}' approach ({attempt_str}, {outcome_str}): {candidate_obj}"
                        reason_codes.append(f"AVOIDED_FAILED_STRATEGY_{failed_tactic.upper()}")
                        if prior_outcome and prior_outcome.effectiveness:
                            reason_codes.append(f"FAILED_OUTCOME_{prior_outcome.effectiveness.upper()}")
                        reason_codes.append("PIVOTED_TO_UNTRIED_STRATEGY")
                        if failed_tactic not in do_not_do:
                            do_not_do.append(failed_tactic)
                        pivoted = True
                        break

                if not pivoted:
                    # Edge case: All candidates have failed
                    primary_action = StrategicAction.CLARIFY
                    objective = f"All targeted strategies for {category} previously failed; falling back to open clarifying dialogue without repeating exhausted tactics."
                    reason_codes.append("ALL_OBJECTION_STRATEGIES_EXHAUSTED_FALLBACK_CLARIFY")

        secondary_reason = secondary_reason or (f"Reinforces {secondary_action.value} to support handling of {category}." if secondary_action else None)

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=objective,
            primary_action=primary_action,
            strategic_posture="advance",
            secondary_action=secondary_action,
            secondary_action_reason=secondary_reason,
            push_strength=PushStrengthValue("moderate", legacy_alias="resolve_then_ask"),
            referenced_objection_ids=[primary_obj.objection_id],
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
        turn_text: str = "",
    ) -> DecisionEvaluationResult:
        stakeholder_roles = [s.role for s in snapshot.decision_structure.stakeholders if s.presence != "on_call"]
        role_label = stakeholder_roles[0] if stakeholder_roles else "partner"
        stakeholder_ids = [getattr(s, "stakeholder_id", f"stakeholder_{s.role}") for s in snapshot.decision_structure.stakeholders]

        # Point 7: Stop auto-mapping "decision-maker changed" to DE_RISK.
        # Classify reason into four distinct categories:
        # 1. Relationship conflict -> VALIDATE / posture mediate
        # 2. Trust concern -> VALIDATE / posture reassure
        # 3. Genuine deal risk -> DE_RISK / posture defend
        # 4. Logistics / coordination -> CLARIFY / posture coordinate
        text_lower = (turn_text or "").lower()

        conflict_keywords = [
            "divorce", "fighting", "dispute", "disagree", "we don't agree",
            "not on the same page", "conflict", "opposed while i want", "divided"
        ]
        trust_keywords = [
            "distrust", "skeptical", "suspicious", "doesn't trust", "don't trust",
            "rip off", "ripoff", "rip-off", "taken advantage", "sleazy", "shady", "sales pitch"
        ]
        risk_keywords = [
            "scam", "fraud", "lawyer", "attorney", "legal", "sue",
            "refuse to sign", "hard veto", "forbid", "prohibit", "against selling"
        ]

        if any(k in text_lower for k in conflict_keywords):
            # Category 1: Relationship Conflict
            primary_action = StrategicAction.VALIDATE
            secondary_action = StrategicAction.QUESTION
            secondary_reason = "Explore shared household priorities neutrally without taking sides in the relationship conflict."
            strategic_posture = "mediate"
            push_strength = PushStrengthValue("none")
            objective = f"Validate differing perspectives neutrally and explore shared priorities with absent {role_label}."
            reason_codes = ["STAKEHOLDER_CONCERN", "RELATIONSHIP_CONFLICT", "EXPLORE_SHARED_GOALS"]
        elif any(k in text_lower for k in trust_keywords):
            # Category 2: Trust Concern
            primary_action = StrategicAction.VALIDATE
            secondary_action = StrategicAction.SOCIAL_PROOF
            secondary_reason = "Provide transparent verified references to address stakeholder skepticism."
            strategic_posture = "reassure"
            push_strength = PushStrengthValue("low", legacy_alias="resolve_then_ask")
            objective = f"Validate stakeholder skepticism and offer transparent third-party proof to address {role_label}'s trust concern."
            reason_codes = ["STAKEHOLDER_CONCERN", "STAKEHOLDER_TRUST_CONCERN", "BUILD_THIRD_PARTY_TRUST"]
        elif any(k in text_lower for k in risk_keywords):
            # Category 3: Genuine Deal Risk
            primary_action = StrategicAction.DE_RISK
            secondary_action = StrategicAction.QUESTION
            secondary_reason = "Inquire into specific legal or structural constraints before attempting next steps."
            strategic_posture = "defend"
            push_strength = PushStrengthValue("none")
            objective = f"De-risk deal and address specific legal or structural blocker regarding {role_label} collaboratively."
            reason_codes = ["STAKEHOLDER_CONCERN", "GENUINE_DEAL_RISK", "PROTECT_AGREEMENT_VIABILITY"]
        else:
            # Category 4: Pure Logistics / Coordination (e.g., Turn 10 scenario)
            primary_action = StrategicAction.CLARIFY
            secondary_action = StrategicAction.FUTURE_PACE
            secondary_reason = "Future pace a collaborative conversation once logistical alignment is established."
            strategic_posture = "coordinate"
            push_strength = PushStrengthValue("low", legacy_alias="resolve_then_ask")
            objective = f"Coordinate logistics to include {role_label} collaboratively in the conversation or walkthrough."
            reason_codes = ["DECISION_MAKER_ABSENT", "COORDINATION_LOGISTICS", "COLLABORATIVE_SCHEDULING"]

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=objective,
            primary_action=primary_action,
            strategic_posture=strategic_posture,
            secondary_action=secondary_action,
            secondary_action_reason=secondary_reason,
            push_strength=push_strength,
            referenced_stakeholder_ids=stakeholder_ids,
            reason_codes=reason_codes,
            do_not_do=["press_for_single_party_commitment", "ignore_absent_decision_maker", "force_immediate_agreement", "separate_partners", "isolate_from_spouse"],
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
                strategic_posture="explore",
                secondary_action=secondary_action,
                secondary_action_reason="Explore missing gate criteria before closing.",
                push_strength=PushStrengthValue("low", legacy_alias="resolve_then_ask"),
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
            secondary_reason = "Provide two options to reduce scheduling friction."
            objective = "Propose two specific calendar windows for the walkthrough."
            reason_codes = ["MEETING_GATE_OPEN", "TWO_WINDOW_CHOICE"]
        else:
            primary_action = StrategicAction.COMMITMENT_CLOSE
            secondary_action = None
            secondary_reason = None
            objective = "Directly propose and secure confirmed property walkthrough."
            reason_codes = ["MEETING_GATE_OPEN", "DIRECT_ASK"]

        gate_conf = snapshot.conversion_gate.confidence if snapshot.conversion_gate else 0.90

        decision = StrategicDecision(
            call_id=snapshot.call_sid,
            source_state_version=snapshot.state_version,
            should_prompt=True,
            strategic_objective=objective,
            primary_action=primary_action,
            strategic_posture="advance",
            secondary_action=secondary_action,
            secondary_action_reason=secondary_reason,
            push_strength=PushStrengthValue("high", legacy_alias=push_state),
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
            secondary_action = None
            secondary_reason = None
            objective = "Uncover prospect goals, situation, and core priorities."
            reason_codes = ["STAGE_DISCOVERY", "EXPLORE_PROSPECT_NEEDS"]
            do_not_do = ["premature_close", "pitch_prematurely"]
            max_words = 20
        elif stage == ConversationStage.VALUE_WALKTHROUGH:
            primary_action = StrategicAction.EDUCATE
            secondary_action = None
            secondary_reason = None
            objective = "Demonstrate tailored value proposition and distinguish approach."
            reason_codes = ["STAGE_VALUE_WALKTHROUGH", "DEMONSTRATE_DIFFERENTIATION"]
            do_not_do = ["overwhelm_with_detail", "press_unready_prospect"]
            max_words = 26
        elif stage == ConversationStage.DECISION_RESOLUTION:
            primary_action = StrategicAction.CLARIFY
            secondary_action = None
            secondary_reason = None
            objective = "Resolve remaining decision criteria and establish consensus."
            reason_codes = ["STAGE_DECISION_RESOLUTION", "CLARIFY_CRITERIA"]
            do_not_do = ["premature_close"]
            max_words = 24
        elif stage in (ConversationStage.SCHEDULING, ConversationStage.COMMITMENT_CONFIRMED):
            gate = snapshot.conversion_gate
            gate_ready = bool(gate and gate.is_open and not getattr(gate, "unknown_conditions", []) and (gate.confidence >= 0.60))
            if not gate_ready:
                primary_action = StrategicAction.QUESTION
                secondary_action = None
                secondary_reason = None
                objective = "Discover scheduling preferences and uncover logistical details."
                reason_codes = ["STAGE_SCHEDULING", "GATE_NOT_READY_DISCOVERY"]
                do_not_do = ["premature_close", "blind_commitment_close"]
                max_words = 20
            else:
                primary_action = StrategicAction.COMMITMENT_CLOSE
                secondary_action = None
                secondary_reason = None
                objective = "Coordinate logistical details and calendar commitment."
                reason_codes = ["STAGE_SCHEDULING", "FINALIZE_TIME"]
                do_not_do = ["reopen_discovery"]
                max_words = 20
        else:
            primary_action = StrategicAction.QUESTION
            secondary_action = None
            secondary_reason = None
            objective = "Engage prospect and build conversation foundation."
            reason_codes = ["STAGE_DEFAULT_ENGAGEMENT"]
            do_not_do = ["premature_close"]
            max_words = 20

        # Client Feedback Point 5: Unclassified material turn routing
        # When prospect statement was material but no structured extractor fired,
        # route safely to CLARIFY / VALIDATE rather than an uninformed discovery question.
        if getattr(snapshot, "unclassified_material", False) and primary_action == StrategicAction.QUESTION:
            primary_action = StrategicAction.CLARIFY
            secondary_action = StrategicAction.VALIDATE
            secondary_reason = "Validate and clarify unclassified material disclosure to avoid deaf discovery."
            objective = "Clarify prospect viewpoint and explore details following unclassified material statement."
            reason_codes = ["UNCLASSIFIED_MATERIAL_CONTENT", "SAFE_CLARIFY_ROUTING"]
            do_not_do = ["blind_discovery_question", "premature_close"]
            max_words = 22

        if trust_score < 40.0 and primary_action != StrategicAction.VALIDATE:
            secondary_action = primary_action
            secondary_reason = f"Reinforce {secondary_action.value} after validating prospect."
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
            strategic_posture="explore",
            secondary_action=secondary_action,
            secondary_action_reason=secondary_reason,
            push_strength=PushStrengthValue("low", legacy_alias=push_st),
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
            unclassified_material=getattr(snapshot, "unclassified_material", False),
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
        decision.unclassified_material = bool(getattr(snapshot, "unclassified_material", False))
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

        reasons_set = set(decision.reason_codes)

        # 1. Trust: Visibly tag default/unmeasured trust vs measured trust (Point 5)
        is_trust_measured = getattr(snapshot.dimensions, "trust_measured", False) or (context.trust_score != 50.0)
        if is_trust_measured:
            evidence.append(f"Trust: {context.trust_score:.0f}% (measured, confidence: {snapshot.dimensions.trust_confidence:.2f})")
        else:
            evidence.append(f"Trust: {context.trust_score:.0f}% (default, unmeasured)")

        # 2. Readiness: Include only if readiness/gate is evaluating closing or materially influencing decision
        readiness_relevant = (
            any(
                r.startswith("READINESS_")
                or r.startswith("GATE_")
                or r.startswith("COMMITMENT_")
                or r.startswith("CONVERSION_")
                or r.startswith("CLOSE_")
                for r in reasons_set
            )
            or decision.primary_action == StrategicAction.COMMITMENT_CLOSE
        )
        has_insufficient_ev = (snapshot.readiness and snapshot.readiness.insufficient_evidence) or (snapshot.readiness and snapshot.readiness.readiness_score is None)
        if readiness_relevant:
            if has_insufficient_ev:
                evidence.append("Readiness: UNKNOWN (Insufficient Evidence)")
            else:
                evidence.append(f"Readiness: {context.readiness_score:.1f}%")

        # 3. Momentum: Include only if non-stable or directly influential
        momentum_relevant = context.momentum_trend != "stable" or any(r.startswith("MOMENTUM_") for r in reasons_set)
        if momentum_relevant:
            evidence.append(f"Momentum: {context.momentum_trend}")

        # 4. Active objections: Include only if handling an objection or objection prevents close
        objection_relevant = bool(context.active_objections) and (
            any(
                r.startswith("OBJECTION_")
                or r.startswith("LADDER_")
                or r.startswith("REFRAME_")
                or r.startswith("QUANTIFY_")
                or r.startswith("DIFFERENTIATE_")
                for r in reasons_set
            )
            or decision.primary_action in (
                StrategicAction.VALIDATE, StrategicAction.MIRROR, StrategicAction.REFRAME,
                StrategicAction.QUANTIFY, StrategicAction.DIFFERENTIATE, StrategicAction.DE_RISK
            )
        )
        if objection_relevant:
            evidence.append(f"Active objections: {context.active_objections}")

        # 5. Gate status: Include only if gate influenced the decision
        gate_relevant = bool(snapshot.conversion_gate) and (
            any(
                r.startswith("GATE_")
                or r.startswith("COMMITMENT_")
                or r.startswith("CONVERSION_")
                or r.startswith("CLOSE_")
                or r.startswith("TWO_WINDOW")
                or r.startswith("DIRECT_ASK")
                for r in reasons_set
            )
            or decision.primary_action == StrategicAction.COMMITMENT_CLOSE
        )
        if gate_relevant:
            gate_st = "OPEN" if snapshot.conversion_gate.is_open else "CLOSED"
            unk = getattr(snapshot.conversion_gate, "unknown_conditions", [])
            evidence.append(f"Gate: {gate_st} (unknown: {unk if unk else 'none'})")

        # 6. Contact Compliance: Only include if compliance materially influenced this decision (Point 11)
        material_compliance_reasons = {
            "CONTACT_NOT_BEFORE_DECLARED", "HOLD_RESPECTED",
            "PAST_CONTACT_FRICTION_HONORED", "HARD_BOUNDARY_ACTIVE", "COMPLIANCE_PRIORITY",
            "BOUNDARY_SUSPECTED", "CONTACT_PREFERENCE_DECLARED", "CONTACT_PREFERENCE_ENFORCED"
        }
        is_confirmed_close = (
            context.conversion_confirmed
            or snapshot.conversation_stage == ConversationStage.COMMITMENT_CONFIRMED
            or "CONFIRM_AND_PROTECT_ACTIVE" in decision.reason_codes
        )
        is_compliance_influential = (
            not is_confirmed_close
            and bool(material_compliance_reasons.intersection(reasons_set))
        )

        if snapshot.contact_compliance:
            if snapshot.contact_compliance.hard_boundary_active and ("HARD_BOUNDARY_ACTIVE" in reasons_set or "COMPLIANCE_PRIORITY" in reasons_set):
                evidence.append("Hard boundary: ACTIVE")
            elif snapshot.contact_compliance.boundary_suspected and "BOUNDARY_SUSPECTED" in reasons_set:
                evidence.append("Boundary suspected: AMBIGUOUS")
            if is_compliance_influential and snapshot.contact_compliance.contact_preferences:
                evidence.append(f"Contact preferences: {len(snapshot.contact_compliance.contact_preferences)} active")

        # 7. Fail-safe: If evidence contains 1 or fewer items, include conversation stage so evidence is never empty
        if len(evidence) <= 1:
            stage_str = str(getattr(snapshot.conversation_stage, "value", snapshot.conversation_stage))
            evidence.append(f"Stage: {stage_str.upper()}")

        # 8. Historical Strategy Provenance: Preserve strategy origin separately (Point 1)
        if getattr(decision, "carried_forward_from_turn_id", None) is not None:
            from_tid = decision.carried_forward_from_turn_id
            from_did = getattr(decision, "carried_forward_from_decision_id", None) or "prior"
            prov_str = f"Strategy provenance: carried forward from Turn {from_tid} (decision {from_did})"
            if prov_str not in evidence:
                evidence.append(prov_str)

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
