from __future__ import annotations

import re
import uuid
import logging
from typing import Any, Dict, List, Literal, Optional

from .conversation_scoring_config import ConversationScoringConfig, DEFAULT_CONVERSATION_SCORING_CONFIG
from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
    GateConditionResult,
    MeetingConversionGate,
    PushStrengthState,
    PushStrengthRecommendation,
    ConversionType,
    ConversionStatus,
    ConversionEventObject,
)

LOGGER = logging.getLogger("copilot.conversation_conversion")


class MeetingConversionGateEngine:
    """Phase 7: Evaluates the 7-condition Meeting/Conversion Gate, Push Strength state machine,

    and Conversion Event Object tracking.
    """

    def __init__(self, config: Optional[ConversationScoringConfig] = None):
        self.config = config or DEFAULT_CONVERSATION_SCORING_CONFIG

    def evaluate_gate(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
    ) -> MeetingConversionGate:
        """Evaluates all 7 meeting gate conditions simultaneously.

        The gate is OPEN if and only if all 7 conditions are met.
        If any condition fails, the gate remains CLOSED with full explainability.
        """
        cfg = self.config
        dims = current_state.dimensions
        conditions: List[GateConditionResult] = []

        # ---------------------------------------------------------------------
        # Condition 1: Trust Not Collapsing
        # ---------------------------------------------------------------------
        trust_val = dims.trust
        tension_val = dims.emotion_tension
        trust_ok = (trust_val >= cfg.gate_min_trust) and (tension_val <= cfg.gate_max_tension)
        if not trust_ok:
            if trust_val < cfg.gate_min_trust:
                reason1 = f"Trust score ({trust_val:.2f}) is below minimum threshold ({cfg.gate_min_trust:.2f})"
            else:
                reason1 = f"Emotion tension ({tension_val:.2f}) exceeds maximum allowable threshold ({cfg.gate_max_tension:.2f})"
        else:
            reason1 = f"Trust healthy ({trust_val:.2f} >= {cfg.gate_min_trust:.2f}) and tension contained ({tension_val:.2f} <= {cfg.gate_max_tension:.2f})"

        conditions.append(
            GateConditionResult(
                condition_name="trust_not_collapsing",
                met=trust_ok,
                score_or_value={"trust": trust_val, "tension": tension_val},
                threshold={"min_trust": cfg.gate_min_trust, "max_tension": cfg.gate_max_tension},
                reason=reason1,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 2: Engagement On-Topic
        # ---------------------------------------------------------------------
        eng_val = dims.engagement
        eng_ok = eng_val >= cfg.gate_min_engagement
        if not eng_ok:
            reason2 = f"Engagement score ({eng_val:.2f}) is below on-topic threshold ({cfg.gate_min_engagement:.2f})"
        else:
            reason2 = f"Engagement active and on-topic ({eng_val:.2f} >= {cfg.gate_min_engagement:.2f})"

        conditions.append(
            GateConditionResult(
                condition_name="engagement_on_topic",
                met=eng_ok,
                score_or_value=eng_val,
                threshold=cfg.gate_min_engagement,
                reason=reason2,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 3: Objection State (All raised objections resolved or partial)
        # ---------------------------------------------------------------------
        unresolved_objs = [
            o for o in current_state.objections
            if o.lifecycle_state in ("unresolved", "reactivated", "boundary")
        ]
        obj_ok = len(unresolved_objs) == 0
        if not obj_ok:
            categories = [o.canonical_category for o in unresolved_objs]
            reason3 = f"Active unresolved objection(s) present: {', '.join(categories)}"
        else:
            if current_state.objections:
                reason3 = f"All {len(current_state.objections)} raised objection(s) are resolved or partially resolved"
            else:
                reason3 = "No active objections raised (clean slate)"

        conditions.append(
            GateConditionResult(
                condition_name="objections_resolved_or_partial",
                met=obj_ok,
                score_or_value=len(unresolved_objs),
                threshold=0,
                reason=reason3,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 4: Clear Value Reason
        # ---------------------------------------------------------------------
        val_score = 50.0
        if current_state.momentum and "value_recognition" in current_state.momentum.family_scores:
            val_score = current_state.momentum.family_scores["value_recognition"]
        logical_r = current_state.readiness.logical_readiness if current_state.readiness else 50.0
        agreement_factor = bundle.agreement_score
        has_goal_facts = any(f.category in ("timeline", "financial", "property") for f in current_state.facts if f.status == "active")

        # Friendly-but-vague guard: Polite filler agreement with low specificity (<= 0.45)
        # without concrete goal/timeline facts does NOT establish a clear value reason.
        is_vague_filler = (bundle.specificity_score <= cfg.two_window_choice_max_specificity) and not has_goal_facts

        val_ok = not is_vague_filler and (
            (val_score >= cfg.gate_min_value_recognition)
            or (agreement_factor >= 0.50 and logical_r >= 50.0)
        )
        if not val_ok:
            if is_vague_filler:
                reason4 = f"Vague conversational discourse (specificity={bundle.specificity_score:.2f} <= {cfg.two_window_choice_max_specificity:.2f}) lacks concrete value/problem justification"
            else:
                reason4 = f"Value recognition ({val_score:.1f}) below threshold ({cfg.gate_min_value_recognition:.1f}) without compensating agreement"
        else:
            reason4 = f"Clear value justification established (value_score={val_score:.1f}, logical_readiness={logical_r:.1f})"

        conditions.append(
            GateConditionResult(
                condition_name="clear_value_reason",
                met=val_ok,
                score_or_value=val_score,
                threshold=cfg.gate_min_value_recognition,
                reason=reason4,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 5: Correct Decision-Maker Identified & Aligned
        # ---------------------------------------------------------------------
        dec = current_state.decision_structure
        has_dm_blocker = bool(
            current_state.readiness and ("absent_decision_maker" in current_state.readiness.active_blocker_caps)
        )
        has_absent_stakeholder = any(
            s.role in ("spouse", "partner", "co-owner", "co_owner", "attorney") and s.presence == "absent"
            for s in dec.stakeholders
        )
        dm_ok = dec.decision_maker_present and not has_dm_blocker and not has_absent_stakeholder
        if not dm_ok:
            reason5 = "Decision-maker absent, unaligned, or consultation required with unconfirmed party"
        else:
            reason5 = f"Decision authority present and aligned ({dec.primary_decision_maker or 'self-authorized'})"

        conditions.append(
            GateConditionResult(
                condition_name="decision_maker_aligned",
                met=dm_ok,
                score_or_value=dec.decision_maker_present,
                threshold=True,
                reason=reason5,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 6: Plausible Logistics
        # ---------------------------------------------------------------------
        log_r = current_state.readiness.logistical_readiness if current_state.readiness else 50.0
        comp = current_state.contact_compliance
        has_logistical_blocker = bool(
            current_state.readiness and ("logistical_deficit" in current_state.readiness.active_blocker_caps)
        )
        has_contact_restriction = comp.contact_preference in ("channel_restriction", "timing_restriction")
        has_access_constraints = bool(current_state.decision_structure.access_constraints)

        log_ok = (
            (log_r >= cfg.gate_min_logistical_readiness)
            and not has_logistical_blocker
            and not has_contact_restriction
            and not has_access_constraints
            and not comp.hard_boundary_active
        )
        if not log_ok:
            if has_contact_restriction:
                reason6 = f"Active contact/scheduling restriction ({comp.contact_preference}) impedes meeting logistics"
            elif has_access_constraints:
                reason6 = f"Access constraints ({', '.join(current_state.decision_structure.access_constraints)}) require resolution"
            else:
                reason6 = f"Logistical readiness deficit ({log_r:.1f} < {cfg.gate_min_logistical_readiness:.1f})"
        else:
            reason6 = f"Logistical feasibility confirmed ({log_r:.1f} >= {cfg.gate_min_logistical_readiness:.1f})"

        conditions.append(
            GateConditionResult(
                condition_name="plausible_logistics",
                met=log_ok,
                score_or_value=log_r,
                threshold=cfg.gate_min_logistical_readiness,
                reason=reason6,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 7: No Active Boundary
        # ---------------------------------------------------------------------
        has_boundary_obj = any(o.lifecycle_state == "boundary" for o in current_state.objections)
        boundary_ok = not comp.hard_boundary_active and not has_boundary_obj and bundle.boundary_score < 0.85
        if not boundary_ok:
            reason7 = "Hard compliance boundary active — persuasion and conversion prohibited"
        else:
            reason7 = "No active compliance boundary"

        conditions.append(
            GateConditionResult(
                condition_name="no_active_boundary",
                met=boundary_ok,
                score_or_value=bundle.boundary_score,
                threshold=0.85,
                reason=reason7,
            )
        )

        # Gate Synthesis
        is_open = all(c.met for c in conditions)
        failed_conditions = [c.condition_name for c in conditions if not c.met]
        blocking_reasons = [c.reason for c in conditions if not c.met]
        effective_conf = round(
            min(bundle.inference_confidence, bundle.semantic_confidence, current_state.overall_confidence),
            3,
        )

        return MeetingConversionGate(
            is_open=is_open,
            status="open" if is_open else "closed",
            conditions=conditions,
            failed_conditions=failed_conditions,
            blocking_reasons=blocking_reasons,
            confidence=effective_conf,
        )

    def evaluate_push_strength(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        gate: MeetingConversionGate,
    ) -> PushStrengthRecommendation:
        """Determines the appropriate operational push strength and strategic approach."""
        cfg = self.config
        dims = current_state.dimensions
        comp = current_state.contact_compliance
        conf = gate.confidence

        # 1. Hard Boundary State: Immediate graceful exit
        has_boundary_obj = any(o.lifecycle_state == "boundary" for o in current_state.objections)
        if comp.hard_boundary_active or has_boundary_obj or bundle.boundary_score >= 0.85:
            return PushStrengthRecommendation(
                state="respect_record_exit",
                rationale="Hard boundary statement detected. Persuasion and conversion asks are prohibited.",
                recommended_action="Acknowledge boundary gracefully, record compliance preference, and terminate call cleanly.",
                confidence=conf,
            )

        # 2. Low Trust / High Threat: Protect & shorten
        if dims.trust < cfg.gate_min_trust or dims.emotion_tension > cfg.gate_max_tension:
            return PushStrengthRecommendation(
                state="protect_and_shorten",
                rationale=f"Trust is deteriorating ({dims.trust:.2f}) or tension is elevated ({dims.emotion_tension:.2f}). Pushing for commitment will trigger immediate defensive reactance.",
                recommended_action="Validate prospect perspective, reduce conversational pressure, and shorten call.",
                confidence=conf,
            )

        # 3. Moderate Trust + Unresolved Objection: Resolve then ask
        unresolved_objs = [
            o for o in current_state.objections
            if o.lifecycle_state in ("unresolved", "reactivated")
        ]
        if unresolved_objs:
            lead_obj = unresolved_objs[0]
            return PushStrengthRecommendation(
                state="resolve_then_ask",
                rationale=f"Active objection '{lead_obj.canonical_category}' is unresolved. Asking for a meeting before reframing will be perceived as dismissive.",
                recommended_action=f"Acknowledge and resolve the {lead_obj.canonical_category} concern before proposing next steps.",
                confidence=conf,
            )

        # 4. Gate Open + High Trust: Direct ask
        if gate.is_open and dims.trust >= cfg.direct_ask_min_trust:
            return PushStrengthRecommendation(
                state="direct_ask",
                rationale=f"Meeting gate is open and trust is strong ({dims.trust:.2f}). Conversation state fully supports a confident, direct close.",
                recommended_action="Propose a specific walkthrough or meeting date, time, and format directly.",
                confidence=conf,
            )

        # 5. Agreeable-but-Vague: Two-window choice
        # Prospect is polite/agreeable but specificity is low or non-committal
        is_agreeable = bundle.agreement_score >= cfg.two_window_choice_min_agreement or dims.trust >= 0.60
        is_vague = bundle.specificity_score <= cfg.two_window_choice_max_specificity or not bool(
            any(f.category == "timeline" and f.status == "active" for f in current_state.facts)
        )
        if not gate.is_open and is_agreeable and is_vague:
            return PushStrengthRecommendation(
                state="two_window_choice",
                rationale="Prospect is agreeable but vague on concrete commitments. Open-ended closing requests invite polite brush-offs.",
                recommended_action="Offer a concrete binary choice (e.g. 'Tuesday morning or Thursday afternoon?') to overcome vagueness.",
                confidence=conf,
            )

        # 6. Soft Hesitation / Logistical Friction: Reduce friction, re-ask
        log_r = current_state.readiness.logistical_readiness if current_state.readiness else 50.0
        if not gate.is_open and log_r < cfg.reduce_friction_logistical_upper:
            return PushStrengthRecommendation(
                state="reduce_friction_reask",
                rationale="Prospect displays general interest but logistical friction or schedule congestion is impeding a full meeting.",
                recommended_action="Lower the commitment barrier (e.g. propose a brief 10-minute phone call or virtual overview), then re-ask.",
                confidence=conf,
            )

        # 7. Fallback when Gate is Open
        if gate.is_open:
            return PushStrengthRecommendation(
                state="direct_ask",
                rationale="All 7 meeting gate conditions are satisfied.",
                recommended_action="Directly ask for confirmation on the next step.",
                confidence=conf,
            )

        # Default closed fallback
        return PushStrengthRecommendation(
            state="two_window_choice",
            rationale=f"Meeting gate closed due to: {', '.join(gate.failed_conditions)}.",
            recommended_action="Structure conversation with low-pressure alternative choices to build alignment.",
            confidence=conf,
        )

    def evaluate_conversion_event(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        gate: MeetingConversionGate,
        previous_event: Optional[ConversionEventObject] = None,
    ) -> Optional[ConversionEventObject]:
        """Tracks, creates, or updates the structured Conversion Event Object."""
        text_lower = bundle.utterance_text.lower()

        # Spec Acceptance Test #3: "Send me something" alone != conversion
        send_info_patterns = [
            r"\bsend\s+(me\s+)?(an?\s+)?(email|info|information|brochure|package|something)\b",
            r"\bemail\s+me\s+(the\s+)?(details|pricing|packet|info)\b",
            r"\bmail\s+me\s+(something|it)\b",
            r"\bjust\s+send\s+me\b",
        ]
        is_send_info = any(re.search(pat, text_lower) for pat in send_info_patterns)

        if is_send_info and bundle.speaker_id == "client":
            return ConversionEventObject(
                event_id=previous_event.event_id if previous_event else f"conv_{uuid.uuid4().hex[:8]}",
                conversion_type="information_send",
                status="blocked",
                start_at=None,
                location_or_format="email",
                participants=[bundle.speaker_id],
                confirmation_confidence=0.30,
                source_turn_ids=[bundle.turn_id],
                blocking_items=["brush_off_send_only_not_conversion", "no_confirmed_meeting_time"],
                followup_is_conversion=False,
            )

        # Spec Acceptance Test #2: Walkthrough confirmation
        walkthrough_patterns = [
            r"\b(walkthrough|walk\s*through|walk\s+the\s+property|walk\s+the\s+house)\b",
            r"\bcome\s+(by|over|see)\b",
            r"\bstop\s+by\b",
        ]
        is_walkthrough = any(re.search(pat, text_lower) for pat in walkthrough_patterns)

        # Explicit timing indicators in utterance or facts
        time_patterns = [
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(tomorrow|today|this\s+week|next\s+week)\b",
            r"\b\d{1,2}(:\d{2})?\s*(am|pm|o'?clock)?\b",
        ]
        has_time = any(re.search(pat, text_lower) for pat in time_patterns)

        # Confirming words
        confirm_patterns = [
            r"\b(yes|yeah|sure|that\s+works|sounds\s+good|perfect|see\s+you\s+then|i'll\s+be\s+there)\b",
            r"\blet's\s+do\s+it\b",
            r"\bthursday\s+at\s+\d\b",
        ]
        is_confirming = any(re.search(pat, text_lower) for pat in confirm_patterns)

        if (is_walkthrough or (previous_event and previous_event.conversion_type == "property_walkthrough")) and bundle.speaker_id == "client":
            # If confirmed walkthrough
            if (is_confirming and (has_time or (previous_event and previous_event.start_at))) or (is_walkthrough and has_time and is_confirming):
                extracted_time = None
                time_match = re.search(r"\b(thursday|friday|monday|tuesday|wednesday|tomorrow)\s*(at\s*\d{1,2}(:\d{2})?\s*(am|pm)?)?", text_lower)
                if time_match:
                    extracted_time = time_match.group(0).title()
                elif previous_event and previous_event.start_at:
                    extracted_time = previous_event.start_at

                return ConversionEventObject(
                    event_id=previous_event.event_id if previous_event else f"conv_{uuid.uuid4().hex[:8]}",
                    conversion_type="property_walkthrough",
                    status="confirmed",
                    start_at=extracted_time or "Confirmed Time Slot",
                    location_or_format="Property Address",
                    participants=["Client", "Agent"],
                    confirmation_confidence=0.90,
                    source_turn_ids=sorted(list(set((previous_event.source_turn_ids if previous_event else []) + [bundle.turn_id]))),
                    blocking_items=[],
                    followup_is_conversion=True,
                )

        # If gate is open and prospect agrees to meeting proposal
        meeting_patterns = [
            r"\b(meet|meeting|consultation|call|schedule|calendar)\b",
        ]
        is_meeting_topic = any(re.search(pat, text_lower) for pat in meeting_patterns)

        if gate.is_open and is_confirming and (is_meeting_topic or has_time):
            return ConversionEventObject(
                event_id=previous_event.event_id if previous_event else f"conv_{uuid.uuid4().hex[:8]}",
                conversion_type="in_person_meeting",
                status="confirmed",
                start_at=has_time and "Confirmed Window" or (previous_event.start_at if previous_event else "Confirmed Window"),
                location_or_format="Scheduled Meeting",
                participants=["Client", "Agent"],
                confirmation_confidence=0.85,
                source_turn_ids=sorted(list(set((previous_event.source_turn_ids if previous_event else []) + [bundle.turn_id]))),
                blocking_items=[],
                followup_is_conversion=True,
            )

        # If gate is open but no explicit confirmation turn yet, state is eligible
        if gate.is_open:
            return ConversionEventObject(
                event_id=previous_event.event_id if previous_event else f"conv_{uuid.uuid4().hex[:8]}",
                conversion_type=previous_event.conversion_type if previous_event else "in_person_meeting",
                status="eligible",
                start_at=previous_event.start_at if previous_event else None,
                location_or_format=previous_event.location_or_format if previous_event else None,
                participants=previous_event.participants if previous_event else ["Client", "Agent"],
                confirmation_confidence=0.60,
                source_turn_ids=sorted(list(set((previous_event.source_turn_ids if previous_event else []) + [bundle.turn_id]))),
                blocking_items=[],
                followup_is_conversion=True,
            )

        # Gate is closed
        if previous_event:
            # Update previous event with current gate blockers
            return ConversionEventObject(
                event_id=previous_event.event_id,
                conversion_type=previous_event.conversion_type,
                status="blocked" if previous_event.status != "confirmed" else "confirmed",
                start_at=previous_event.start_at,
                location_or_format=previous_event.location_or_format,
                participants=previous_event.participants,
                confirmation_confidence=previous_event.confirmation_confidence if previous_event.status == "confirmed" else 0.20,
                source_turn_ids=sorted(list(set(previous_event.source_turn_ids + [bundle.turn_id]))),
                blocking_items=gate.blocking_reasons,
                followup_is_conversion=previous_event.followup_is_conversion,
            )

        return None
