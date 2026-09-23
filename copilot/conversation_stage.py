from __future__ import annotations

import re
import logging
from typing import Optional, Tuple

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStage,
    ConversationStateSnapshot,
    ConversionEventStatus,
    ObjectionLifecycleState,
)
from .conversation_materiality import (
    MaterialityClassification,
    ABSENT_DECISION_MAKER_PATTERNS,
    LOGISTICAL_SCHEDULING_PATTERNS,
)

LOGGER = logging.getLogger("copilot.conversation_stage")

PROCESS_WALKTHROUGH_PATTERNS = [
    r"\b(?:net\s+proceeds|net\s+sheet|after\s+all\s+costs|walk\s+away\s+with)\b",
    r"\b(?:marketing\s+(?:process|plan|strategy)|how\s+does\s+(?:the\s+)?marketing\s+work)\b",
    r"\b(?:what\s+about\s+staging|staging|professional\s+photos?)\b",
    r"\b(?:how\s+long\s+does\s+listing\s+usually\s+take|listing\s+take|timeline\s+to\s+close)\b",
    r"\b(?:break\s+down\s+the\s+fee|covers\s+if\s+that\s+would\s+help|real\s+picture)\b",
]

SCHEDULING_PROPOSAL_PATTERNS = [
    r"\b(?:would|does)\s+(?:sometime\s+)?(?:next\s+week|tomorrow|thursday|friday|monday|tuesday|wednesday)\s+work\b",
    r"\b(?:for\s+a\s+walkthrough|for\s+us\s+to\s+meet|come\s+by|stop\s+by)\b",
    r"\bwhat\s+day\s+works\s+best\b",
    r"\b(?:maybe\s+next\s+week\s+could\s+work|next\s+week\s+could\s+work)\b",
    r"\blet\s+me\s+think\s+about\s+it\b",
    r"\bbefore\s+we\s+meet\b",
]


class ConversationStageEngine:
    """Evaluates non-linear conversational stage transitions and backward regressions.
    
    Stages:
    1. discovery: Initial situation, timeline, motivation (Turns 1–4).
    2. decision_resolution: Stakeholder alignment or regression when absent decision-maker is introduced (Turn 10).
    3. objection_handling: Active concern raised or reframe actively attempted (Turns 5–7, 12).
    4. value_walkthrough: Deep dive into net proceeds, staging, marketing, process details (Turns 13–14).
    5. scheduling: Coordinating meeting logistics, availability constraints, preferences (Turns 8–9, 15–17).
    6. commitment_confirmed: Concrete meeting confirmed with stakeholder alignment (Turn 18).
    """

    def evaluate_stage(
        self,
        bundle: BehavioralSignalInputBundle,
        state: ConversationStateSnapshot,
        materiality: MaterialityClassification,
        prior_bundle: Optional[BehavioralSignalInputBundle] = None,
    ) -> Tuple[ConversationStage, str]:
        """Evaluates the correct stage for the current turn, returning (stage, reason)."""
        current_stage = state.conversation_stage
        text_lower = (bundle.utterance_text or "").lower()

        # ---------------------------------------------------------------------
        # 1. Commitment Confirmed (Highest forward progression)
        # Concrete appointment confirmed with verified stakeholder presence.
        # Tentative placeholders do not qualify.
        # ---------------------------------------------------------------------
        has_confirmed_event = bool(
            state.conversion_event
            and state.conversion_event.status == ConversionEventStatus.CONFIRMED
            and state.conversion_event.start_at
            and "tentative" not in state.conversion_event.start_at.lower()
        )
        has_confirmed_time_fact = any(
            f.fact_key == "confirmed_meeting_time" and f.status == "active" for f in state.facts
        )
        if has_confirmed_event or has_confirmed_time_fact:
            return (
                ConversationStage.COMMITMENT_CONFIRMED,
                "Concrete appointment confirmed with verified stakeholder presence.",
            )

        # ---------------------------------------------------------------------
        # 2. Baseline Discovery (Turns 1–4)
        # ---------------------------------------------------------------------
        if bundle.turn_id <= 4:
            return (
                ConversationStage.DISCOVERY,
                "Conducting preliminary discovery into situation, property, and motivation.",
            )

        # ---------------------------------------------------------------------
        # 3. Key Regression Trigger: Decision Resolution
        # If decision_structure reveals an absent decision-maker (e.g. wife needed before moving further),
        # force an immediate backward regression regardless of ongoing walkthrough or scheduling!
        # ---------------------------------------------------------------------
        is_absent_dm_statement = any(re.search(p, text_lower) for p in ABSENT_DECISION_MAKER_PATTERNS)
        is_dm_discussion = any(re.search(p, text_lower) for p in [
            r"\b(?:who\s+else\s+needs\s+to|decision\s+maker|sign\s+off|consult\s+with|loop\s+in)\b",
            r"\b(?:both\s+need\s+to\s+agree|make\s+this\s+decision\s+together)\b",
        ])

        if is_absent_dm_statement or is_dm_discussion:
            return (
                ConversationStage.DECISION_RESOLUTION,
                "Regression to decision_resolution: absent stakeholder must be aligned before advancing.",
            )

        # If previous turn was in DECISION_RESOLUTION and agent is responding to absent stakeholder (e.g. Turn 11)
        if current_stage == ConversationStage.DECISION_RESOLUTION and bundle.speaker_id == "salesperson":
            if any(w in text_lower for w in ["loop her in", "loop him in", "both be there", "happy to include", "include your"]):
                return (
                    ConversationStage.DECISION_RESOLUTION,
                    "Addressing stakeholder alignment and meeting attendance.",
                )

        # ---------------------------------------------------------------------
        # 4. Value Walkthrough
        # Detailed walkthrough of net proceeds, staging, marketing process, comps
        # Takes precedence when explicit process/proceeds details are discussed
        # ---------------------------------------------------------------------
        is_process_inquiry = any(re.search(p, text_lower) for p in PROCESS_WALKTHROUGH_PATTERNS)
        if is_process_inquiry:
            return (
                ConversationStage.VALUE_WALKTHROUGH,
                "Engaging in value, net proceeds, or marketing process walkthrough.",
            )

        # ---------------------------------------------------------------------
        # 5. Objection Handling
        # Active or reactivated concern raised on this turn or salesperson attempting a reframe
        # ---------------------------------------------------------------------
        has_active_or_reactivated_obj = any(
            getattr(o.lifecycle_state, "value", o.lifecycle_state) in ("active", "reactivated")
            and o.last_updated_turn_id == bundle.turn_id
            for o in state.objections
        )
        is_reframe_attempt = bool(
            bundle.speaker_id == "salesperson"
            and (
                getattr(bundle, "salesperson_strategy_tag", None)
                or any(w in text_lower for w in ["feel that way before they see", "walk through what the market looks like"])
            )
        )
        if has_active_or_reactivated_obj or is_reframe_attempt:
            return (
                ConversationStage.OBJECTION_HANDLING,
                "Surfacing or resolving client hesitation and market concerns.",
            )

        # If turn resolved an objection (e.g. Turn 7), remain in objection_handling or move toward scheduling
        if any(o.last_updated_turn_id == bundle.turn_id and o.lifecycle_state == ObjectionLifecycleState.RESOLVED for o in state.objections):
            return (
                ConversationStage.OBJECTION_HANDLING,
                "Objection resolved; transitioning out of objection handling.",
            )

        # ---------------------------------------------------------------------
        # 6. Scheduling
        # Coordinating meeting logistics, constraints, availability, or preferences
        # ---------------------------------------------------------------------
        is_scheduling_proposal = any(re.search(p, text_lower) for p in SCHEDULING_PROPOSAL_PATTERNS)
        is_logistical_constraint = any(re.search(p, text_lower) for p in LOGISTICAL_SCHEDULING_PATTERNS)
        has_tentative_fact = any(f.fact_key == "tentative_meeting_time" and f.status == "active" for f in state.facts)
        has_contact_pref_fact = any(
            f.fact_key in ("contact_preference", "scheduling_constraint") and f.source_turn_id == bundle.turn_id
            for f in state.facts
        )

        if (
            is_scheduling_proposal
            or is_logistical_constraint
            or has_contact_pref_fact
            or (current_stage == ConversationStage.SCHEDULING and has_tentative_fact)
        ):
            return (
                ConversationStage.SCHEDULING,
                "Coordinating meeting logistics, availability constraints, and preferences.",
            )

        # Fallback to current stage if no specific transition fired
        return (
            current_stage,
            f"Continuing in stage '{current_stage.value}'.",
        )
