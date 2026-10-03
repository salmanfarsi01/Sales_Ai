from __future__ import annotations

import re
import uuid
import logging
from typing import Any, Dict, List, Literal, Optional

from .conversation_scoring_config import ConversationScoringConfig, DEFAULT_CONVERSATION_SCORING_CONFIG
from .conversation_conversion_config import ConversionBlockingConfig, DEFAULT_CONVERSION_BLOCKING_CONFIG
from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
    ConversationStage,
    GateConditionResult,
    MeetingConversionGate,
    PushStrengthState,
    PushStrengthRecommendation,
    ConversionType,
    ConversionStatus,
    ConversionEventStatus,
    ConversionEventObject,
    DealDispositionType,
)

def detect_explicit_reversal_in_text(text: str) -> tuple[bool, Optional[str]]:
    """Detects whether explicit text requests to cancel, retract, or call off an appointment/meeting.
    Guarded with negation filters and hypothetical conditionals.
    """
    text_lower = text.lower()

    # Negation guards: Prospect asserting they do NOT want to cancel
    negation_patterns = [
        r"\b(?:don't|do\s+not|won't|will\s+not|not\s+going\s+to|not\s+trying\s+to)\s+cancel\b",
        r"\b(?:not|never)\s+cancelling\b",
        r"\b(?:still\s+want\s+to\s+meet|still\s+planning\s+to|still\s+good\s+for)\b",
        r"\b(?:no\s+need\s+to\s+cancel|don't\s+cancel)\b",
    ]
    # Hypothetical guards: Prospect framing cancellation hypothetically
    hypothetical_patterns = [
        r"\bhypothetical(?:ly)?\b",
        r"\blet's\s+say\b",
        r"\bwhat\s+if\b",
        r"\bsuppose\b",
        r"\bjust\s+pretend\b",
        r"\bmaybe\b",
        r"\bif\s+i\s+(?:had\s+to|needed\s+to|were\s+to\s+cancel)\b",
    ]
    if any(re.search(pat, text_lower) for pat in negation_patterns) or any(re.search(pat, text_lower) for pat in hypothetical_patterns):
        return False, None

    # Reversal keywords & phrases
    reversal_patterns = [
        (r"\b(?:cancel|cancelling|cancelled)\b", "Prospect requested meeting cancellation"),
        (r"\b(?:never\s+mind|nevermind)\b", "Prospect stated 'never mind' to conversion"),
        (r"\b(?:let's\s+not|let\s+us\s+not)\s+(?:meet|do\s+that|do\s+this|schedule)\b", "Prospect requested not to meet/schedule"),
        (r"\bforget\s+(?:about\s+)?(?:it|that|thursday|friday|monday|tuesday|wednesday|tomorrow|the\s+meeting|meeting)\b", "Prospect requested to forget scheduled meeting"),
        (r"\b(?:can't|cannot|couldn't|could\s+not)\s+make\s+it\s+(?:anymore|after\s+all|now)\b", "Prospect stated they cannot make the meeting"),
        (r"\bwon't\s+be\s+able\s+to\s+meet\b", "Prospect unavailable to meet"),
        (r"\b(?:call\s+off|called\s+off)\b", "Prospect called off meeting"),
        (r"\bnot\s+going\s+to\s+work\s+out\b", "Prospect stated meeting will not work out"),
        (r"\bdecided\s+(?:against|not\s+to\s+meet)\b", "Prospect decided not to meet"),
        (r"\btake\s+me\s+off\s+(?:the\s+schedule|your\s+calendar)\b", "Prospect requested removal from calendar"),
    ]

    for pat, desc in reversal_patterns:
        if re.search(pat, text_lower):
            return True, desc

    return False, None


class MeetingConversionGateEngine:
    """Phase 7: Evaluates the 7-condition Meeting/Conversion Gate, Push Strength state machine,
    and Conversion Event Object tracking. Supports target-specific objection filtering and
    deterministic explicit commitment overrides.
    """

    def __init__(
        self,
        config: Optional[ConversationScoringConfig] = None,
        blocking_config: Optional[ConversionBlockingConfig] = None,
    ):
        self.config = config or DEFAULT_CONVERSATION_SCORING_CONFIG
        self.blocking_config = blocking_config or DEFAULT_CONVERSION_BLOCKING_CONFIG

    def detect_explicit_commitment(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        prior_bundle: Optional[BehavioralSignalInputBundle] = None,
    ) -> tuple[bool, Optional[str]]:
        """Detects whether prospect made an explicit, unambiguous commitment to a specific slot or proposal."""
        if bundle.speaker_id != "client":
            return False, None

        text_lower = bundle.utterance_text.lower()

        # Check for explicit negation, hypothetical framing, or hedged phrasing
        negation_patterns = [
            r"\b(?:doesn't|does\s+not|won't|will\s+not|can't|cannot|couldn't|could\s+not)\s+(?:work|make\s+it|do\s+it)\b",
            r"\b(?:not|never)\s+(?:works?|feasible|possible|available|good)\b",
            r"\b(?:not\s+free|unavailable|busy)\b",
            r"\b(?:hard|tough|impossible)\s+to\s+make\b",
            r"\b(?:don't|do\s+not)\s+(?:call|reach|contact|text|bother)\b",
        ]
        hypothetical_patterns = [
            r"\bhypothetical(?:ly)?\b",
            r"\blet's\s+say\b",
            r"\bwhat\s+if\b",
            r"\bsuppose\b",
            r"\bjust\s+pretend\b",
            r"\bmaybe\b",
            r"\bif\s+i\s+(?:could|can|were)\b",
            r"\bunlikely\b",
            r"\bdoubt\s+(?:it|i\s+can)\b",
        ]
        hedged_patterns = [
            r"\b(?:could|might)\s+work\b",
            r"\b(?:could|might)\s+(?:be\s+able\s+to|possibly)\b",
            r"\btentative(?:ly)?\b",
            r"\bpossibly\b",
            r"\blet\s+me\s+(?:check|see|think)\b",
            r"\bnot\s+(?:100%|sure|certain)\b",
            r"\bif\s+(?:that|it)\s+works\b",
        ]
        if (
            any(re.search(pat, text_lower) for pat in negation_patterns)
            or any(re.search(pat, text_lower) for pat in hypothetical_patterns)
            or any(re.search(pat, text_lower) for pat in hedged_patterns)
        ):
            return False, None

        # Day & time indicators (day + clock time, day + period, or day alone)
        time_day_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b(?:\s+(?:at\s+)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)|(?:\s+(?:morning|afternoon|evening|noon)))?",
            text_lower,
        )

        confirm_keywords = [
            r"\b(?:that\s+works|works\s+for\s+me|this\s+works|thursday\s+works|friday\s+works|it\s+works)\b",
            r"\b(?:sounds\s+good|sounds\s+fair|perfect|let's\s+do\s+it|deal|fine\s+with\s+me|see\s+you\s+then|i'll\s+be\s+there)\b",
            r"\b(?:i\s+could\s+do|i\s+can\s+do|i\s+can\s+meet|i\s+could\s+meet|let's\s+meet|we\s+can\s+meet|can\s+meet|could\s+meet|come\s+by|stop\s+by)\b",
            r"\b(?:sure\s+let's\s+meet|sure\s+come\s+by)\b",
            r"(?<!doesn't\s)(?<!does\snot\s)(?<!won't\s)(?<!not\s)\bworks\b",
            r"\b(yes|yeah|sure|definitely|absolutely)\b",
        ]
        has_confirm_keyword = any(re.search(pat, text_lower) for pat in confirm_keywords)

        # Case A: Explicit day/time and affirmative stance in current utterance
        if time_day_match and (has_confirm_keyword or bundle.agreement_score >= 0.50):
            return True, time_day_match.group(0).strip().title()

        # Case B: Salesperson proposed a day/time in prior turn, and prospect explicitly accepted
        if prior_bundle and prior_bundle.speaker_id == "salesperson":
            prior_lower = prior_bundle.utterance_text.lower()
            prop_match = re.search(
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b(?:\s+(?:at\s+)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)|(?:\s+(?:morning|afternoon|evening|noon)))?",
                prior_lower,
            )
            is_affirmative = (
                bundle.agreement_score >= 0.60
                or any(re.search(rf"\b{aff}\b", text_lower) for aff in ["yeah", "yes", "definitely", "sure", "absolutely", "works", "that works", "perfect", "sounds good", "sounds fair"])
            )
            if prop_match and is_affirmative:
                return True, prop_match.group(0).strip().title()

        # Case C: Confirmed meeting time fact already active and affirmed
        confirmed_fact = next((f for f in current_state.facts if f.fact_key == "confirmed_meeting_time" and f.status == "active"), None)
        if confirmed_fact and (bundle.agreement_score >= 0.60 or has_confirm_keyword):
            return True, str(confirmed_fact.fact_value)

        # Case D: Concrete timing keywords + high composite commitment scores
        has_temporal = any(re.search(pat, text_lower) for pat in [
            r"\b(morning|afternoon|evening|noon|calendar|schedule|appointment|meet|meeting|walkthrough)\b",
            r"\b(next\s+week|this\s+week|weekend)\b",
        ])
        if (
            has_temporal
            and bundle.agreement_score >= 0.65
            and bundle.specificity_score >= 0.60
            and bundle.future_language_score >= 0.60
        ):
            return True, bundle.utterance_text.strip()

        return False, None

    def detect_explicit_reversal(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        prior_bundle: Optional[BehavioralSignalInputBundle] = None,
    ) -> tuple[bool, Optional[str]]:
        """Detects whether prospect explicitly requested to cancel, retract, or call off an appointment/meeting."""
        if bundle.speaker_id != "client":
            return False, None

        return detect_explicit_reversal_in_text(bundle.utterance_text)

    def evaluate_gate(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        conversion_target: str = "appointment",
        prior_bundle: Optional[BehavioralSignalInputBundle] = None,
    ) -> MeetingConversionGate:
        """Evaluates all 7 meeting gate conditions simultaneously.

        The gate is OPEN if and only if all 7 conditions are met.
        If any condition fails, the gate remains CLOSED with full explainability.
        Evaluates objections relative to conversion_target blocking rules, and applies
        explicit human statement overrides when unambiguous commitments are observed.
        """
        cfg = self.config
        dims = current_state.dimensions
        conditions: List[GateConditionResult] = []

        # Detect explicit commitment override
        has_explicit_commit, commit_slot = self.detect_explicit_commitment(
            bundle, current_state, prior_bundle
        )

        # Collect prospect evidence turn IDs (strictly prospect turns, never rep lines)
        prospect_turns: List[int] = []
        if hasattr(current_state, "prospect_turn_ids"):
            for pt in current_state.prospect_turn_ids:
                if pt not in prospect_turns:
                    prospect_turns.append(pt)
        if bundle.speaker_id == "client" and bundle.turn_id not in prospect_turns:
            prospect_turns.append(bundle.turn_id)

        def _prospect_only(turns: List[int]) -> List[int]:
            """Filters evidence turn IDs to strictly prospect-spoken turns."""
            return [t for t in turns if t in prospect_turns]

        # Check for substantive prospect-originated evidence
        has_prospect_spoken = len(prospect_turns) > 0 or getattr(current_state, "has_prospect_spoken", False)
        has_substantive_facts = any(
            f.category in ("timeline", "financial", "property", "decision_maker") and f.status == "active"
            for f in current_state.facts
        )
        has_affirmative_evidence = (
            has_explicit_commit
            or (bundle.speaker_id == "client" and (bundle.agreement_score > 0.55 or bundle.future_language_score > 0.40 or bundle.specificity_score > 0.50))
            or bool(current_state.objections)
            or bool(current_state.decision_structure.primary_decision_maker)
            or has_substantive_facts
            or any(f.fact_key == "confirmed_meeting_time" and f.status == "active" for f in current_state.facts)
            or bool(current_state.conversion_event and getattr(current_state.conversion_event, "status", "") in ("confirmed", ConversionEventStatus.CONFIRMED))
        )
        is_clean_slate_defaults = (not has_prospect_spoken) or (not has_affirmative_evidence)

        # ---------------------------------------------------------------------
        # Condition 1: Trust Not Collapsing
        # ---------------------------------------------------------------------
        trust_val = dims.trust
        tension_val = dims.emotion_tension
        has_trust_affirmation = (
            trust_val > 0.50
            or dims.emotion_valence > 0.05
            or any(o.lifecycle_state in ("resolved", "superseded") for o in current_state.objections)
            or (bundle.speaker_id == "client" and bundle.agreement_score > 0.65)
        )
        trust_evidence_turns: List[int] = []
        for o in current_state.objections:
            if o.lifecycle_state in ("resolved", "superseded") and getattr(o, "last_updated_turn_id", 0) > 0:
                trust_evidence_turns.append(o.last_updated_turn_id)
        if bundle.speaker_id == "client" and (trust_val > 0.50 or bundle.agreement_score > 0.65 or dims.emotion_valence > 0.05):
            if bundle.turn_id not in trust_evidence_turns:
                trust_evidence_turns.append(bundle.turn_id)

        if not prospect_turns:
            cond1_status = "unknown"
            reason1 = "No prospect turns observed to evaluate trust dynamics (clean slate)"
            cond1_ev = []
        elif trust_val < cfg.gate_min_trust or tension_val > cfg.gate_max_tension:
            cond1_status = "not_met"
            cond1_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [])
            if trust_val < cfg.gate_min_trust:
                reason1 = f"Trust score ({trust_val:.2f}) is below minimum threshold ({cfg.gate_min_trust:.2f})"
            else:
                reason1 = f"Emotion tension ({tension_val:.2f}) exceeds maximum allowable threshold ({cfg.gate_max_tension:.2f})"
        elif has_trust_affirmation or has_explicit_commit:
            cond1_status = "met"
            reason1 = f"Trust healthy ({trust_val:.2f} >= {cfg.gate_min_trust:.2f}) and tension contained ({tension_val:.2f} <= {cfg.gate_max_tension:.2f})"
            cond1_ev = trust_evidence_turns if trust_evidence_turns else (prospect_turns[-1:] if prospect_turns else [])
        else:
            cond1_status = "unknown"
            reason1 = "Trust dynamics sitting on neutral baseline without affirmative trust evidence"
            cond1_ev = []

        conditions.append(
            GateConditionResult(
                condition_name="trust_not_collapsing",
                status=cond1_status,
                met=(cond1_status == "met"),
                evidence_turn_ids=_prospect_only(cond1_ev),
                score_or_value={"trust": trust_val, "tension": tension_val},
                threshold={"min_trust": cfg.gate_min_trust, "max_tension": cfg.gate_max_tension},
                reason=reason1,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 2: Engagement On-Topic
        # ---------------------------------------------------------------------
        eng_val = dims.engagement
        has_proposal_or_resolution = (
            bool(current_state.conversion_event)
            or any(o.lifecycle_state in ("resolved", "superseded") for o in current_state.objections)
            or any(f.category in ("property", "financial") and f.status == "active" for f in current_state.facts)
            or any(f.category == "timeline" and f.fact_key != "timeline_horizon" and f.status == "active" for f in current_state.facts)
        )
        has_affirmative_engagement = (
            eng_val > 0.50
            or has_proposal_or_resolution
            or (bundle.speaker_id == "client" and bundle.agreement_score > 0.65)
        )
        eng_evidence_turns: List[int] = []
        for o in current_state.objections:
            if o.lifecycle_state in ("resolved", "superseded") and getattr(o, "last_updated_turn_id", 0) > 0:
                if o.last_updated_turn_id in prospect_turns:
                    eng_evidence_turns.append(o.last_updated_turn_id)
        if current_state.conversion_event and getattr(current_state.conversion_event, "source_turn_ids", []):
            for st in current_state.conversion_event.source_turn_ids:
                if st in prospect_turns and st not in eng_evidence_turns:
                    eng_evidence_turns.append(st)
        if bundle.speaker_id == "client" and (eng_val > 0.50 or bundle.agreement_score > 0.65):
            if bundle.turn_id not in eng_evidence_turns and bundle.turn_id in prospect_turns:
                eng_evidence_turns.append(bundle.turn_id)

        is_overridden2 = False
        if has_explicit_commit:
            cond2_status = "met"
            is_overridden2 = True
            reason2 = f"[EXPLICIT COMMITMENT OVERRIDE: '{commit_slot or bundle.utterance_text}'] Engagement active on specific commitment"
            cond2_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [bundle.turn_id])
        elif not prospect_turns:
            cond2_status = "unknown"
            reason2 = "No prospect engagement observed (clean slate)"
            cond2_ev = []
        elif eng_val < cfg.gate_min_engagement:
            cond2_status = "not_met"
            reason2 = f"Engagement score ({eng_val:.2f}) is below on-topic threshold ({cfg.gate_min_engagement:.2f})"
            cond2_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [])
        elif has_affirmative_engagement:
            cond2_status = "met"
            reason2 = f"Engagement active and on-topic ({eng_val:.2f} >= {cfg.gate_min_engagement:.2f})"
            cond2_ev = eng_evidence_turns if eng_evidence_turns else (prospect_turns[-1:] if prospect_turns else [])
        else:
            cond2_status = "unknown"
            reason2 = "Engagement sitting on neutral baseline (awaiting substantive on-topic discussion)"
            cond2_ev = []

        conditions.append(
            GateConditionResult(
                condition_name="engagement_on_topic",
                status=cond2_status,
                met=(cond2_status == "met"),
                evidence_turn_ids=_prospect_only(cond2_ev),
                score_or_value=eng_val,
                threshold=cfg.gate_min_engagement,
                reason=reason2,
                is_overridden=is_overridden2,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 3: Objection State (Target-aware blocking category filter)
        # ---------------------------------------------------------------------
        target_blocking_cats = self.blocking_config.blocking_categories.get(
            conversion_target,
            ["boundary"],
        )
        unresolved_objs = [
            o for o in current_state.objections
            if o.lifecycle_state in ("active", "unresolved", "reactivated", "boundary", "clarified")
        ]
        target_blocking_unresolved = []
        non_blocking_unresolved = []
        escalated_objs = []
        non_esc_cats = getattr(self.blocking_config, "non_escalating_categories", {}).get(conversion_target, ["commission_fee"])
        for o in unresolved_objs:
            if o.canonical_category in target_blocking_cats:
                target_blocking_unresolved.append(o)
            elif (
                getattr(self.blocking_config, "escalate_on_recurrence", True)
                and o.canonical_category not in non_esc_cats
                and getattr(o, "recurrence_count", 1) > getattr(self.blocking_config, "max_non_blocking_recurrence", 2)
            ):
                target_blocking_unresolved.append(o)
                escalated_objs.append(o)
            else:
                non_blocking_unresolved.append(o)

        has_proposal_or_close = (
            bool(current_state.conversion_event)
            or any(
                getattr(t_event, "status", "") in (ConversionEventStatus.PROPOSED, ConversionEventStatus.TENTATIVE, ConversionEventStatus.CONFIRMED)
                for t_event in getattr(current_state, "conversion_event_history", [])
            )
            or getattr(current_state, "conversation_stage", None) in (ConversationStage.SCHEDULING, ConversationStage.COMMITMENT_CONFIRMED)
        )

        if len(target_blocking_unresolved) > 0:
            cond3_status = "not_met"
            cond3_ev = [o.source_turn_id for o in target_blocking_unresolved if getattr(o, "source_turn_id", 0) > 0]
            categories = []
            for o in target_blocking_unresolved:
                if o in escalated_objs:
                    categories.append(f"{o.canonical_category} (escalated due to recurrence >= {o.recurrence_count})")
                else:
                    categories.append(o.canonical_category)
            reason3 = f"Active unresolved objection(s) blocking '{conversion_target}': {', '.join(categories)}"
        elif current_state.objections:
            cond3_status = "met"
            cond3_ev = []
            for o in current_state.objections:
                if getattr(o, "source_turn_id", 0) > 0 and o.source_turn_id not in cond3_ev:
                    cond3_ev.append(o.source_turn_id)
                if getattr(o, "last_updated_turn_id", 0) > 0 and o.last_updated_turn_id not in cond3_ev:
                    cond3_ev.append(o.last_updated_turn_id)
            if non_blocking_unresolved:
                nb_cats = [o.canonical_category for o in non_blocking_unresolved]
                reason3 = f"No blocking objections for '{conversion_target}' (non-blocking active: {', '.join(nb_cats)})"
            else:
                reason3 = f"All {len(current_state.objections)} raised objection(s) are resolved, partially resolved, or superseded"
        elif has_explicit_commit:
            cond3_status = "met"
            cond3_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [bundle.turn_id])
            reason3 = "No active objections raised; prospect explicit commitment confirms alignment"
        elif bundle.speaker_id == "client" and (bundle.agreement_score >= 0.70 or (bundle.agreement_score >= 0.60 and bundle.specificity_score >= 0.50)):
            cond3_status = "met"
            cond3_ev = [bundle.turn_id]
            reason3 = f"No objections raised; prospect affirmative alignment demonstrated at Turn {bundle.turn_id} (agreement={bundle.agreement_score:.2f})"
        elif has_proposal_or_close and (bundle.agreement_score >= 0.60 or any(f.status == "active" for f in current_state.facts)):
            cond3_status = "met"
            cond3_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [])
            reason3 = "No active objections raised following commitment proposal; prospect affirmative dialogue demonstrates alignment"
        else:
            cond3_status = "unknown"
            cond3_ev = []
            reason3 = "No active objections raised yet — prospect has not yet been presented with a commitment proposal to surface concerns"

        conditions.append(
            GateConditionResult(
                condition_name="objections_resolved_or_partial",
                status=cond3_status,
                met=(cond3_status == "met"),
                evidence_turn_ids=_prospect_only(cond3_ev),
                score_or_value=len(target_blocking_unresolved),
                threshold=0,
                reason=reason3,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 4: Clear Value Reason (Single Source of Truth: Logical Readiness)
        # ---------------------------------------------------------------------
        val_score = 50.0
        if current_state.momentum and "value_recognition" in current_state.momentum.family_scores:
            val_score = current_state.momentum.family_scores["value_recognition"]
        logical_r = (
            current_state.readiness.logical_readiness
            if (current_state.readiness and current_state.readiness.logical_readiness is not None)
            else None
        )

        has_resolved_value_objection = any(
            o.lifecycle_state in ("resolved", "superseded") and o.canonical_category in ("commission_fee", "financial_net_proceeds", "perceived_value_deficit")
            for o in current_state.objections
        )
        has_explicit_goal_fact = any(
            f.category in ("financial", "property", "problem") and f.status == "active"
            for f in current_state.facts
        )
        val_evidence_turns: List[int] = []
        for o in current_state.objections:
            if o.lifecycle_state in ("resolved", "superseded") and o.canonical_category in ("commission_fee", "financial_net_proceeds", "perceived_value_deficit") and getattr(o, "last_updated_turn_id", 0) > 0:
                if o.last_updated_turn_id in prospect_turns:
                    val_evidence_turns.append(o.last_updated_turn_id)
        for f in current_state.facts:
            if f.category in ("financial", "property", "problem") and getattr(f, "source_turn_id", 0) > 0:
                if f.source_turn_id in prospect_turns and f.source_turn_id not in val_evidence_turns:
                    val_evidence_turns.append(f.source_turn_id)

        comp = getattr(current_state, "contact_compliance", None)
        has_contact_friction_or_pref = (
            (bundle.contact_preference and bundle.contact_preference != "none")
            or bool(comp and (comp.contact_preferences or comp.contact_preference != "none"))
            or bool(comp and comp.boundary_suspected)
        )
        is_vague_filler = (
            (bundle.specificity_score <= cfg.two_window_choice_max_specificity)
            and not has_explicit_goal_fact
            and not has_resolved_value_objection
            and not has_contact_friction_or_pref
        )

        has_declined_disp = (
            current_state.deal_disposition is not None
            and getattr(current_state.deal_disposition, "disposition", None) in ("declined", DealDispositionType.DECLINED)
        )
        has_reopened_disp = (
            current_state.deal_disposition is not None
            and getattr(current_state.deal_disposition, "disposition", None) in ("reconsidering", "reversed_decline", DealDispositionType.RECONSIDERING, DealDispositionType.REVERSED_DECLINE)
        )
        has_decision_to_stay = has_declined_disp or (
            not has_reopened_disp and (
                any(o.lifecycle_state == "superseded" and o.superseded_by_objection_id == "decision_to_stay" for o in current_state.objections)
                or any(f.fact_key == "decision_to_stay" and f.status == "active" for f in current_state.facts)
            )
        )

        has_insufficient_readiness = (
            current_state.readiness is None
            or current_state.readiness.insufficient_evidence
            or (getattr(current_state.readiness, "coverage", 1.0) < 0.50)
        )
        val_deficient = (not has_insufficient_readiness) and (
            (val_score is not None and val_score < cfg.gate_min_value_recognition)
            and (logical_r is None or logical_r < cfg.gate_min_value_recognition)
        )
        has_affirmative_value_evidence = (
            has_resolved_value_objection
            or has_explicit_goal_fact
            or (not has_insufficient_readiness and logical_r is not None and logical_r >= cfg.gate_min_value_recognition and any(f.category in ("financial", "property", "problem") for f in current_state.facts))
            or (val_score > 55.0)
        )
        val_ok = not has_decision_to_stay and not is_vague_filler and not val_deficient and has_affirmative_value_evidence

        is_overridden4 = False
        if has_explicit_commit and not has_decision_to_stay and not val_deficient:
            cond4_status = "met"
            is_overridden4 = True
            cond4_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [bundle.turn_id])
            reason4 = f"[OVERRIDE: Explicit commitment detected: '{commit_slot or bundle.utterance_text}'] Clear value justification established"
        elif has_decision_to_stay:
            cond4_status = "not_met"
            cond4_ev = prospect_turns[-1:] if prospect_turns else []
            reason4 = "Prospect explicitly decided to stay and not sell; transaction value proposition is void"
        elif val_deficient:
            cond4_status = "not_met"
            cond4_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [])
            def_score = val_score if (val_score is not None and val_score < cfg.gate_min_value_recognition) else (logical_r or 0.0)
            reason4 = f"Value recognition score ({def_score:.1f}) below threshold ({cfg.gate_min_value_recognition:.1f})"
        elif not prospect_turns or is_clean_slate_defaults:
            cond4_status = "unknown"
            cond4_ev = []
            reason4 = "Value recognition not yet demonstrated by prospect (clean-slate baseline lacks prospect evidence)"
        elif is_vague_filler:
            cond4_status = "not_met"
            cond4_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [])
            reason4 = f"Vague conversational discourse (specificity={bundle.specificity_score:.2f} <= {cfg.two_window_choice_max_specificity:.2f}) lacks concrete value/problem justification"
        elif val_ok:
            cond4_status = "met"
            cond4_ev = val_evidence_turns if val_evidence_turns else ([bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else []))
            reason4 = f"Clear value justification established by prospect (logical_readiness={logical_r or val_score:.1f})"
        else:
            cond4_status = "unknown"
            cond4_ev = []
            reason4 = "Value recognition not yet established by prospect (prospect has not articulated problem, financial goals, or transaction value justification)"

        conditions.append(
            GateConditionResult(
                condition_name="clear_value_reason",
                status=cond4_status,
                met=(cond4_status == "met"),
                evidence_turn_ids=_prospect_only(cond4_ev),
                score_or_value=logical_r or val_score,
                threshold=cfg.gate_min_value_recognition,
                reason=reason4,
                is_overridden=is_overridden4,
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
            s.role in ("spouse", "partner", "co-owner", "co_owner", "attorney", "wife", "husband") and s.presence == "absent"
            for s in dec.stakeholders
        )
        dm_facts = [f for f in current_state.facts if (f.category == "decision_maker" or f.fact_key in ("spouse_involvement", "decision_maker_authority", "sole_decision_maker", "decision_maker_role")) and f.status == "active"]
        dm_fact_turns = [f.source_turn_id for f in dm_facts if getattr(f, "source_turn_id", 0) > 0]

        if not dec.decision_maker_present or has_dm_blocker or has_absent_stakeholder:
            cond5_status = "not_met"
            cond5_ev = dm_fact_turns if dm_fact_turns else (prospect_turns[-1:] if prospect_turns else [])
            reason5 = "Decision-maker absent, unaligned, or consultation required with unconfirmed party"
        elif has_explicit_commit:
            cond5_status = "met"
            cond5_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [bundle.turn_id])
            reason5 = f"Decision authority confirmed via explicit commitment ({dec.primary_decision_maker or 'self-authorized'})"
        elif not prospect_turns or is_clean_slate_defaults:
            cond5_status = "unknown"
            cond5_ev = []
            reason5 = "Decision authority unverified with prospect (clean slate — awaiting prospect confirmation)"
        elif dec.primary_decision_maker or dm_facts or any(f.category == "property" and f.status == "active" for f in current_state.facts):
            cond5_status = "met"
            cond5_ev = dm_fact_turns if dm_fact_turns else ([2] if (2 in prospect_turns and "sole" in str(dec.primary_decision_maker).lower()) else ([bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [])))
            reason5 = f"Decision authority present and aligned ({dec.primary_decision_maker or 'self-authorized'})"
        else:
            cond5_status = "unknown"
            cond5_ev = []
            reason5 = "Decision authority unverified with prospect (clean-slate default lacks explicit prospect confirmation)"

        conditions.append(
            GateConditionResult(
                condition_name="decision_maker_aligned",
                status=cond5_status,
                met=(cond5_status == "met"),
                evidence_turn_ids=_prospect_only(cond5_ev),
                score_or_value=dec.decision_maker_present,
                threshold=True,
                reason=reason5,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 6: Plausible Logistics
        # ---------------------------------------------------------------------
        log_r = (
            current_state.readiness.logistical_readiness
            if (current_state.readiness and current_state.readiness.logistical_readiness is not None)
            else 50.0
        )
        comp = current_state.contact_compliance
        has_logistical_blocker = bool(
            current_state.readiness and ("logistical_deficit" in current_state.readiness.active_blocker_caps)
        )
        has_hard_channel_restriction = any(
            p.boundary_strength == "hard_restriction" and not p.allowed
            for p in getattr(comp, "contact_preferences", [])
        )
        has_contact_restriction = (
            has_hard_channel_restriction
            or comp.contact_preference in ("channel_restriction", "timing_restriction")
        )

        raw_constraints = current_state.decision_structure.access_constraints
        unresolved_constraints: List[str] = []
        slot_text = f"{commit_slot or ''} {bundle.utterance_text}".lower()

        for c in raw_constraints:
            c_lower = c.lower()
            if "morning" in c_lower:
                is_morning_slot = bool(re.search(r"\b(?:\d{1,2}\s*am|morning|mornings|10\s*am|11\s*am|9\s*am)\b", slot_text))
                is_afternoon_slot = bool(re.search(r"\b(?:at\s+(?:[1-9]|1[0-2])\s*pm|\b(?:1[2-9]|[2-9])\b|afternoon|pm|evening|at\s+3\b|at\s+three\b)\b", slot_text))
                if has_explicit_commit and (is_afternoon_slot or not is_morning_slot):
                    continue
                unresolved_constraints.append(c)
            elif "afternoon" in c_lower:
                is_afternoon_slot = bool(re.search(r"\b(?:\d{1,2}\s*pm|afternoon|afternoons)\b", slot_text))
                if has_explicit_commit and not is_afternoon_slot:
                    continue
                unresolved_constraints.append(c)
            elif "weekend" in c_lower:
                is_weekend_slot = bool(re.search(r"\b(?:saturday|sunday|weekend)\b", slot_text))
                if has_explicit_commit and not is_weekend_slot:
                    continue
                unresolved_constraints.append(c)
            else:
                unresolved_constraints.append(c)

        has_access_constraints = bool(unresolved_constraints)
        constraints_satisfied = bool(raw_constraints) and not has_access_constraints

        appointment_timing_facts = [
            f for f in current_state.facts
            if f.category in ("timeline", "logistical")
            and f.fact_key != "timeline_horizon"
            and f.status == "active"
        ]
        appointment_timing_fact_turns = [f.source_turn_id for f in appointment_timing_facts if getattr(f, "source_turn_id", 0) > 0]
        has_confirmed_meeting = (
            any(f.fact_key == "confirmed_meeting_time" and f.status == "active" for f in current_state.facts)
            or bool(current_state.conversion_event and getattr(current_state.conversion_event, "status", "") in ("confirmed", ConversionEventStatus.CONFIRMED))
        )
        has_tentative_meeting = any(f.fact_key in ("tentative_meeting_time", "walkthrough_timing") and f.status == "active" for f in current_state.facts)
        is_appointment_target = conversion_target in ("appointment", "property_walkthrough", "initial_consultation", "meeting")
        has_appointment_logistics = bool(
            (not is_appointment_target)
            or has_confirmed_meeting
            or has_tentative_meeting
            or appointment_timing_facts
            or has_explicit_commit
            or (log_r is not None and log_r >= cfg.gate_min_logistical_readiness and (has_confirmed_meeting or has_tentative_meeting or has_access_constraints))
        )

        log_ok = (
            ((log_r is not None and log_r >= cfg.gate_min_logistical_readiness) or not is_appointment_target)
            and has_appointment_logistics
            and not has_logistical_blocker
            and not has_contact_restriction
            and not has_access_constraints
            and not comp.hard_boundary_active
        )

        is_overridden6 = False
        if (
            has_explicit_commit
            and not comp.hard_boundary_active
            and not has_contact_restriction
            and not has_access_constraints
        ):
            cond6_status = "met"
            is_overridden6 = True
            cond6_ev = [bundle.turn_id] if bundle.speaker_id == "client" else (prospect_turns[-1:] if prospect_turns else [bundle.turn_id])
            constraint_note = " (satisfies scheduling constraints)" if constraints_satisfied else ""
            reason6 = f"[OVERRIDE: Explicit commitment detected: '{commit_slot or bundle.utterance_text}'{constraint_note} outranks inferred logistical score] Logistical feasibility confirmed"
        elif comp.hard_boundary_active or has_contact_restriction or has_access_constraints or has_logistical_blocker:
            cond6_status = "not_met"
            cond6_ev = appointment_timing_fact_turns if appointment_timing_fact_turns else (prospect_turns[-1:] if prospect_turns else [])
            if comp.hard_boundary_active:
                reason6 = "Hard compliance boundary active — logistics prohibited"
            elif has_contact_restriction:
                reason6 = f"Active contact/scheduling restriction ({comp.contact_preference}) impedes meeting logistics"
            elif has_access_constraints:
                reason6 = f"Access constraints ({', '.join(unresolved_constraints)}) require resolution"
            else:
                reason6 = f"Logistical readiness deficit ({log_r or 0.0:.1f} < {cfg.gate_min_logistical_readiness:.1f})"
        elif not is_appointment_target and not comp.hard_boundary_active and not has_contact_restriction and not has_access_constraints:
            cond6_status = "met"
            cond6_ev = list(prospect_turns)
            reason6 = f"Logistical feasibility satisfied for {conversion_target}"
        elif not prospect_turns or is_clean_slate_defaults or not has_appointment_logistics or log_r is None:
            cond6_status = "unknown"
            cond6_ev = []
            reason6 = "Logistical availability and scheduling feasibility not yet discussed by prospect"
        elif not log_ok:
            cond6_status = "not_met"
            cond6_ev = appointment_timing_fact_turns if appointment_timing_fact_turns else list(prospect_turns)
            reason6 = f"Logistical readiness deficit ({log_r or 0.0:.1f} < {cfg.gate_min_logistical_readiness:.1f})"
        else:
            cond6_status = "met"
            cond6_ev = appointment_timing_fact_turns if appointment_timing_fact_turns else list(prospect_turns)
            reason6 = f"Logistical feasibility confirmed ({log_r:.1f} >= {cfg.gate_min_logistical_readiness:.1f})"

        is_hedged_logistics = any(
            w in bundle.utterance_text.lower()
            for w in ("maybe", "let me think", "might", "possibly", "think about it", "not sure if")
        )
        if cond6_status == "met" and is_hedged_logistics:
            reason6 = f"Logistical window identified ('{bundle.utterance_text.strip()}'), but qualified by hedge phrase ('maybe' / 'let me think') — tentative feasibility"

        conditions.append(
            GateConditionResult(
                condition_name="plausible_logistics",
                status=cond6_status,
                met=(cond6_status == "met"),
                evidence_turn_ids=_prospect_only(cond6_ev),
                score_or_value=log_r,
                threshold=cfg.gate_min_logistical_readiness,
                reason=reason6,
                is_overridden=is_overridden6,
            )
        )

        # ---------------------------------------------------------------------
        # Condition 7: No Active Boundary
        # ---------------------------------------------------------------------
        has_boundary_obj = any(o.lifecycle_state == "boundary" for o in current_state.objections)
        boundary_ok = not comp.hard_boundary_active and not has_boundary_obj and bundle.boundary_score < 0.85

        if not boundary_ok:
            cond7_status = "not_met"
            cond7_ev = [bundle.turn_id] if bundle.boundary_score >= 0.85 else (prospect_turns[-1:] if prospect_turns else [])
            reason7 = "Hard compliance boundary active — persuasion and conversion prohibited"
        elif not prospect_turns:
            cond7_status = "unknown"
            cond7_ev = []
            reason7 = "Awaiting prospect interaction to verify compliance boundaries (clean slate)"
        else:
            cond7_status = "met"
            cond7_ev = list(prospect_turns)
            reason7 = "No active compliance boundary"

        conditions.append(
            GateConditionResult(
                condition_name="no_active_boundary",
                status=cond7_status,
                met=(cond7_status == "met"),
                evidence_turn_ids=_prospect_only(cond7_ev),
                score_or_value=bundle.boundary_score,
                threshold=0.85,
                reason=reason7,
            )
        )

        # Gate Synthesis
        is_open = all(c.status == "met" for c in conditions)
        failed_conditions = [c.condition_name for c in conditions if c.status == "not_met"]
        unknown_conditions = [c.condition_name for c in conditions if c.status == "unknown"]
        blocking_reasons = [c.reason for c in conditions if c.status != "met"]

        base_conf = min(bundle.inference_confidence, bundle.semantic_confidence)
        if is_hedged_logistics:
            base_conf = round(base_conf * 0.65, 3)
        if unknown_conditions and not is_open:
            base_conf = min(base_conf, 0.40)
        effective_conf = base_conf

        return MeetingConversionGate(
            is_open=is_open,
            status="open" if is_open else "closed",
            conversion_target=conversion_target,
            conditions=conditions,
            failed_conditions=failed_conditions,
            unknown_conditions=unknown_conditions,
            blocking_reasons=blocking_reasons,
            confidence=effective_conf,
            explicit_commitment_detected=has_explicit_commit,
            commitment_slot=commit_slot,
        )

    def evaluate_push_strength(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        gate: MeetingConversionGate,
        conversion_target: Optional[str] = None,
    ) -> PushStrengthRecommendation:
        """Determines the appropriate operational push strength and strategic approach."""
        cfg = self.config
        dims = current_state.dimensions
        comp = current_state.contact_compliance
        conf = gate.confidence
        target = conversion_target or getattr(gate, "conversion_target", "appointment") or "appointment"

        # 1. Hard Boundary State: Immediate graceful exit
        has_boundary_obj = any(o.lifecycle_state == "boundary" for o in current_state.objections)
        has_declined_disp = (
            current_state.deal_disposition is not None
            and getattr(current_state.deal_disposition, "disposition", None) in ("declined", DealDispositionType.DECLINED)
        )
        has_reopened_disp = (
            current_state.deal_disposition is not None
            and getattr(current_state.deal_disposition, "disposition", None) in ("reconsidering", "reversed_decline", DealDispositionType.RECONSIDERING, DealDispositionType.REVERSED_DECLINE)
        )
        has_decision_to_stay = has_declined_disp or (
            not has_reopened_disp and (
                any(o.lifecycle_state == "superseded" and o.superseded_by_objection_id == "decision_to_stay" for o in current_state.objections)
                or any(f.fact_key == "decision_to_stay" and f.status == "active" for f in current_state.facts)
            )
        )
        if comp.hard_boundary_active or has_boundary_obj or bundle.boundary_score >= 0.85:
            return PushStrengthRecommendation(
                state="respect_record_exit",
                rationale="Hard boundary statement detected. Persuasion and conversion asks are prohibited.",
                recommended_action="Acknowledge boundary gracefully, record compliance preference, and terminate call cleanly.",
                confidence=conf,
            )
        if has_decision_to_stay:
            return PushStrengthRecommendation(
                state="respect_record_exit",
                rationale="Prospect explicitly decided to stay and not sell. Persuasion and conversion asks are prohibited.",
                recommended_action="Respect prospect decision to stay, record status in CRM, and terminate call gracefully.",
                confidence=conf,
            )

        # 2. We Already Won (Client Feedback Item 9): Stop selling, confirm & protect
        # If prospect has provided an explicit concrete commitment or confirmed/tentative slot,
        # persuasion is COMPLETE. Stop selling, do NOT ask again ('direct_ask'), do NOT re-open
        # objections ('resolve_then_ask'), and do NOT abort ('protect_and_shorten').
        # CRITICAL ARCHITECTURAL INVARIANT:
        # This operates INDEPENDENTLY of gate.is_open and BEFORE any objection-blocking fallthroughs
        # or threat dampeners. An unresolved objection (e.g. commission_fee) can coexist with,
        # or even be the exact rationale for, the meeting itself:
        # "Your commission is still too expensive, but I can meet Thursday at 4."
        # 2. Confirmed Conversion Gate (Client Item 9: confirm_and_protect)
        # CRITICAL INVARIANT: confirm_and_protect applies ONLY after a confirmed meeting!
        # If an appointment is tentative or unconfirmed, confirm_and_protect is strictly disallowed.
        is_appointment_target = target in ("appointment", "property_walkthrough", "initial_consultation", "meeting")
        is_dm_absent = (not current_state.decision_structure.decision_maker_present) or any(
            s.role in ("spouse", "partner", "co-owner", "co_owner", "attorney", "wife", "husband") and s.presence == "absent"
            for s in current_state.decision_structure.stakeholders
        )
        has_confirmed_event = (
            current_state.conversion_event is not None
            and getattr(current_state.conversion_event, "status", None) in (
                "confirmed",
                ConversionEventStatus.CONFIRMED,
            )
            and (
                getattr(current_state.conversion_event, "event_type", None) == target
                or is_appointment_target
            )
            and not comp.hard_boundary_active
            and not is_dm_absent
        )
        has_appointment_slot = (
            gate.explicit_commitment_detected
            and is_appointment_target
            and not comp.hard_boundary_active
            and not is_dm_absent
        )

        has_explicit_commitment = has_appointment_slot or has_confirmed_event
        if has_explicit_commitment:
            slot_info = gate.commitment_slot or (
                current_state.conversion_event.start_at if current_state.conversion_event else None
            )
            slot_str = f" at {slot_info}" if slot_info else ""
            pref_reassurance = ""
            if comp.contact_preference != "none" or comp.contact_preferences:
                pref_reassurance = " Reassure and respect their stated contact preferences (e.g. reduced texting cadence)."

            return PushStrengthRecommendation(
                state="confirm_and_protect",
                rationale=(
                    f"Milestone secured: Prospect explicitly agreed to the conversion target{slot_str}. "
                    "Persuasion is complete; stop selling, avoid re-opening closed or dormant objections, and protect the commitment."
                ),
                recommended_action=(
                    f"Confirm the scheduled appointment{slot_str}.{pref_reassurance} "
                    "Do not reopen objections, send calendar invite, and exit the call cleanly."
                ),
                confidence=conf,
            )

        # 3. Deal Disposition: Reconsidering conditionality check (distinct operational state)
        if (
            current_state.deal_disposition is not None
            and getattr(current_state.deal_disposition, "disposition", None) in ("reconsidering", DealDispositionType.RECONSIDERING)
        ):
            return PushStrengthRecommendation(
                state="explore_conditional_terms",
                rationale="Prospect is conditionally reconsidering sale ('reconsidering'). Pushing for immediate closing commitment before exploring their criteria triggers defensive reactance.",
                recommended_action="Acknowledge conditional openness, explore specific strategic or financial criteria (e.g. required strategy, net proceeds, or timing), and avoid premature closing asks until alignment is established.",
                confidence=conf,
            )

        # 4. Low Trust / High Threat: Protect & shorten
        if dims.trust < cfg.gate_min_trust or dims.emotion_tension > cfg.gate_max_tension:
            return PushStrengthRecommendation(
                state="protect_and_shorten",
                rationale=f"Trust is deteriorating ({dims.trust:.2f}) or tension is elevated ({dims.emotion_tension:.2f}). Pushing for commitment will trigger immediate defensive reactance.",
                recommended_action="Validate prospect perspective, reduce conversational pressure, and shorten call.",
                confidence=conf,
            )

        # 5. Moderate Trust + Unresolved Target-Blocking Objection: Resolve then ask
        target_blocking_cats = self.blocking_config.blocking_categories.get(target, ["boundary"])
        unresolved_blocking_objs = []
        unresolved_non_blocking_objs = []
        non_esc_cats = getattr(self.blocking_config, "non_escalating_categories", {}).get(target, ["commission_fee"])
        for o in current_state.objections:
            if o.lifecycle_state in ("unresolved", "reactivated", "active"):
                is_blocking_cat = o.canonical_category in target_blocking_cats
                is_escalated = (
                    getattr(self.blocking_config, "escalate_on_recurrence", True)
                    and o.canonical_category not in non_esc_cats
                    and getattr(o, "recurrence_count", 1) > getattr(self.blocking_config, "max_non_blocking_recurrence", 2)
                )
                if is_blocking_cat or is_escalated:
                    unresolved_blocking_objs.append(o)
                else:
                    unresolved_non_blocking_objs.append(o)

        if unresolved_blocking_objs:
            lead_obj = unresolved_blocking_objs[0]
            driver = getattr(lead_obj, "driver_layer", None)
            strat_target = getattr(driver, "strategic_target", None)
            target_str = f" ({strat_target.replace('_', ' ')})" if strat_target else ""
            return PushStrengthRecommendation(
                state="resolve_then_ask",
                rationale=f"Active objection '{lead_obj.canonical_category}' blocks '{target}'. Asking for commitment before reframing will be perceived as dismissive.",
                recommended_action=f"Acknowledge and resolve the {lead_obj.canonical_category} concern{target_str} before proposing next steps.",
                confidence=conf,
            )

        # 5b. Active Non-Blocking Objection (Item 9 Client Feedback Alignment):
        # Even if objection does not structurally block milestone gating (e.g. general_hesitation for appointment),
        # an active unresolved objection requires addressing the underlying driver before a direct close ('direct_ask')
        # to prevent contradictory advice and avoid triggering defensive reactance.
        if unresolved_non_blocking_objs:
            lead_obj = unresolved_non_blocking_objs[0]
            driver = getattr(lead_obj, "driver_layer", None)
            if driver and getattr(driver, "strategic_target", None):
                strat_action = driver.strategic_target.replace("_", " ")
                driver_name = getattr(driver, "underlying_driver", "hesitation")
                if driver_name in ("process_overwhelm", "information_deficit"):
                    return PushStrengthRecommendation(
                        state="reduce_friction_reask",
                        rationale=(
                            f"Active objection '{lead_obj.canonical_category}' is non-blocking for '{target}', "
                            f"but underlying driver '{driver_name}' creates operational drag. "
                            "Pushing for full commitment before lowering friction triggers resistance."
                        ),
                        recommended_action=(
                            f"Execute driver target: {strat_action} before proposing a specific closing commitment."
                        ),
                        confidence=conf,
                    )
                else:
                    return PushStrengthRecommendation(
                        state="resolve_then_ask",
                        rationale=(
                            f"Active objection '{lead_obj.canonical_category}' is non-blocking for '{target}', "
                            f"but underlying driver '{driver_name}' requires tactical exploration. "
                            "Proposing a direct closing ask before surfacing root hesitation triggers defensive reactance."
                        ),
                        recommended_action=(
                            f"Execute driver target: {strat_action} to clarify prospect hesitation before proposing a specific meeting time."
                        ),
                        confidence=conf,
                    )
            else:
                return PushStrengthRecommendation(
                    state="resolve_then_ask",
                    rationale=(
                        f"Active objection '{lead_obj.canonical_category}' is non-blocking for '{target}', "
                        "but unaddressed hesitation will create sales friction if rushed."
                    ),
                    recommended_action=(
                        f"Acknowledge and explore the {lead_obj.canonical_category} concern before directly pressing for an appointment."
                    ),
                    confidence=conf,
                )

        # Check for hedged logistics or agreeable-but-vague non-committal response
        is_hedged = any(
            w in bundle.utterance_text.lower()
            for w in ("maybe", "let me think", "might", "possibly", "think about it", "not sure if")
        )
        is_agreeable = bundle.agreement_score >= cfg.two_window_choice_min_agreement or dims.trust >= 0.60
        is_vague = (bundle.specificity_score <= cfg.two_window_choice_max_specificity) and not bool(
            any(f.category == "timeline" and f.status == "active" for f in current_state.facts)
        )
        if gate.is_open and (is_hedged or (is_agreeable and is_vague)):
            return PushStrengthRecommendation(
                state="two_window_choice",
                rationale="Meeting gate is open, but prospect hedged with conditional hesitation ('let me think about it'). Avoid high-pressure direct close; offer a low-friction binary choice.",
                recommended_action="Offer a concrete binary choice (e.g. 'Would Tuesday morning or Thursday afternoon suit you better?') to reduce decision overhead.",
                confidence=conf,
            )

        # 6. Gate Open without Concrete Slot: Direct close proposal
        if gate.is_open and dims.trust >= cfg.direct_ask_min_trust:
            if comp.contact_preference == "reduced_frequency":
                return PushStrengthRecommendation(
                    state="direct_ask",
                    rationale="Meeting gate is open and prospect requested reduced frequency. Proposed appointment must respect cadence preference.",
                    recommended_action="Propose a specific walkthrough or meeting date, time, and format while reassuring prospect that frequent follow-ups will not be sent.",
                    confidence=conf,
                )
            return PushStrengthRecommendation(
                state="direct_ask",
                rationale=f"Meeting gate is open and trust is strong ({dims.trust:.2f}). Conversation state fully supports a confident, direct close.",
                recommended_action="Propose a specific walkthrough or meeting date, time, and format directly.",
                confidence=conf,
            )
            if comp.contact_preference == "reduced_frequency":
                return PushStrengthRecommendation(
                    state="two_window_choice",
                    rationale="Meeting gate is open. Prospect is agreeable with low contact frequency preference. Respect cadence while offering binary choice.",
                    recommended_action="Offer a low-friction binary choice while reassuring prospect that contact frequency will remain minimal.",
                    confidence=conf,
                )
            return PushStrengthRecommendation(
                state="two_window_choice",
                rationale="Meeting gate is open. Prospect is agreeable but vague on concrete commitments. Open-ended closing requests invite polite brush-offs.",
                recommended_action="Offer a concrete binary choice (e.g. 'Tuesday morning or Thursday afternoon?') to overcome vagueness.",
                confidence=conf,
            )

        # 6. Soft Hesitation / Logistical Friction: Reduce friction, re-ask
        log_r = (
            current_state.readiness.logistical_readiness
            if (current_state.readiness and current_state.readiness.logistical_readiness is not None)
            else None
        )
        if not gate.is_open and log_r is not None and log_r < cfg.reduce_friction_logistical_upper:
            return PushStrengthRecommendation(
                state="reduce_friction_reask",
                rationale="Prospect displays general interest but logistical friction or schedule congestion is impeding a full meeting.",
                recommended_action="Lower the commitment barrier (e.g. propose a brief 10-minute phone call or virtual overview), then re-ask.",
                confidence=conf,
            )

        # 7. Fallback when Gate is Open
        if gate.is_open:
            if comp.contact_preference == "reduced_frequency":
                return PushStrengthRecommendation(
                    state="direct_ask",
                    rationale="All 7 meeting gate conditions satisfied with reduced frequency preference.",
                    recommended_action="Confirm appointment time directly and reassure prospect that frequent messages will not be sent.",
                    confidence=conf,
                )
            return PushStrengthRecommendation(
                state="direct_ask",
                rationale="All 7 meeting gate conditions are satisfied.",
                recommended_action="Directly ask for confirmation on the next step.",
                confidence=conf,
            )

        # Default closed fallback: Must NOT be a close-style push (Spec 09 / Client Audit Fix Item 1)
        reasons = list(gate.failed_conditions) + list(getattr(gate, "unknown_conditions", []))
        reason_str = ", ".join(reasons) if reasons else "unmet gate criteria"
        return PushStrengthRecommendation(
            state="resolve_then_ask",
            rationale=f"Meeting gate closed due to: {reason_str}. Close-style push is strictly disallowed while gate is closed.",
            recommended_action="Discover missing gate details and address prospect priorities before attempting commitment close.",
            confidence=conf,
        )

    def _resolve_conversion_participants(
        self,
        current_state: ConversationStateSnapshot,
        bundle: BehavioralSignalInputBundle,
        previous_event: Optional[ConversionEventObject] = None,
    ) -> List[str]:
        base = ["Client", "Agent"]
        if current_state and current_state.decision_structure:
            for s in current_state.decision_structure.stakeholders:
                label = s.role.title() if s.role else "Stakeholder"
                if s.presence == "confirmed_attending" or (
                    bundle and s.role and s.role.lower() in bundle.utterance_text.lower()
                ):
                    if label not in base:
                        base.append(label)
        if bundle and bundle.speaker_id == "client":
            t_lower = bundle.utterance_text.lower()
            if "wife" in t_lower and any(w in t_lower for w in ["there", "attend", "join", "come", "with me", "both"]):
                if "Wife" not in base:
                    base.append("Wife")
            elif "husband" in t_lower and any(w in t_lower for w in ["there", "attend", "join", "come", "with me", "both"]):
                if "Husband" not in base:
                    base.append("Husband")
            elif "spouse" in t_lower and any(w in t_lower for w in ["there", "attend", "join", "come", "with me", "both"]):
                if "Spouse" not in base:
                    base.append("Spouse")
            elif "partner" in t_lower and any(w in t_lower for w in ["there", "attend", "join", "come", "with me", "both"]):
                if "Partner" not in base:
                    base.append("Partner")
        if previous_event and previous_event.participants:
            for p in previous_event.participants:
                if p not in base:
                    base.append(p)
        return base

    def evaluate_conversion_event(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        gate: MeetingConversionGate,
        previous_event: Optional[ConversionEventObject] = None,
    ) -> Optional[ConversionEventObject]:
        """Tracks, creates, or updates the structured Conversion Event Object with non-destructive supersession."""
        text_lower = bundle.utterance_text.lower()

        # ---------------------------------------------------------------------
        # Trigger B: Explicit Reversal Language (Takes Priority for Explicit Cancellations)
        # ---------------------------------------------------------------------
        is_reversal, reversal_reason = self.detect_explicit_reversal(bundle, current_state)
        if is_reversal:
            if previous_event and previous_event.status in (
                ConversionEventStatus.CONFIRMED,
                ConversionEventStatus.TENTATIVE,
                ConversionEventStatus.PROPOSED,
            ):
                new_event = ConversionEventObject(
                    event_id=f"conv_{uuid.uuid4().hex[:8]}",
                    conversion_type=previous_event.conversion_type,
                    status=ConversionEventStatus.CANCELLED,
                    start_at=previous_event.start_at,
                    location_or_format=previous_event.location_or_format,
                    participants=previous_event.participants,
                    confirmation_confidence=0.90,
                    source_turn_ids=[bundle.turn_id],
                    blocking_items=[reversal_reason or "explicit_cancellation"],
                    followup_is_conversion=False,
                    supersedes_event_id=previous_event.event_id,
                    reversal_reason=reversal_reason or "explicit_cancellation",
                )
                previous_event.superseded_by_event_id = new_event.event_id
                previous_event.superseded_at_turn_id = bundle.turn_id
                return new_event
            elif previous_event and previous_event.status == ConversionEventStatus.CANCELLED:
                return previous_event

        # ---------------------------------------------------------------------
        # Trigger A: Hard Boundary Active Override (Blocks Unconfirmed / Preserves Confirmed)
        # ---------------------------------------------------------------------
        is_hard_boundary = current_state.contact_compliance.hard_boundary_active or bundle.boundary_score >= 0.85
        if is_hard_boundary:
            # Client Requirement: Do NOT auto-cancel a CONFIRMED appointment on a contact boundary.
            # "Stop contacting me" and "cancel the appointment" are different outcomes.
            # Only unconfirmed / tentative / proposed / eligible conversion events are cancelled.
            if previous_event and previous_event.status in (
                ConversionEventStatus.TENTATIVE,
                ConversionEventStatus.PROPOSED,
                ConversionEventStatus.ELIGIBLE,
            ):
                new_event = ConversionEventObject(
                    event_id=f"conv_{uuid.uuid4().hex[:8]}",
                    conversion_type=previous_event.conversion_type,
                    status=ConversionEventStatus.CANCELLED,
                    start_at=previous_event.start_at,
                    location_or_format=previous_event.location_or_format,
                    participants=previous_event.participants,
                    confirmation_confidence=1.0,
                    source_turn_ids=[bundle.turn_id],
                    blocking_items=["hard_boundary"],
                    followup_is_conversion=False,
                    supersedes_event_id=previous_event.event_id,
                    reversal_reason="hard_boundary",
                )
                previous_event.superseded_by_event_id = new_event.event_id
                previous_event.superseded_at_turn_id = bundle.turn_id
                return new_event
            elif previous_event and previous_event.status in (
                ConversionEventStatus.CONFIRMED,
                ConversionEventStatus.CANCELLED,
            ):
                return previous_event

        # If previous event is already cancelled, stay cancelled unless a new affirmative commitment is made
        if previous_event and previous_event.status == ConversionEventStatus.CANCELLED:
            has_commit, _ = self.detect_explicit_commitment(bundle, current_state)
            if not has_commit:
                return previous_event

        # ---------------------------------------------------------------------
        # Walkthrough & Meeting Topic Detection
        # ---------------------------------------------------------------------
        walkthrough_patterns = [
            r"\b(walkthrough|walk\s*through|walk\s+the\s+property|walk\s+the\s+house)\b",
            r"\bcome\s+(by|over|see)\b",
            r"\bstop\s+by\b",
        ]
        is_walkthrough = any(re.search(pat, text_lower) for pat in walkthrough_patterns)

        # ---------------------------------------------------------------------
        # Salesperson Slot Proposal (State: PROPOSED)
        # ---------------------------------------------------------------------
        if bundle.speaker_id == "salesperson":
            prop_match = re.search(
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b(?:\s+(?:at\s+)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)|(?:\s+(?:morning|afternoon|evening|noon)))?|(?:next\s+week|this\s+week|weekend)",
                text_lower,
            )
            if prop_match and (previous_event is None or previous_event.status in (ConversionEventStatus.NOT_ATTEMPTED, ConversionEventStatus.ELIGIBLE)):
                conv_type = "property_walkthrough" if is_walkthrough else "in_person_meeting"
                return ConversionEventObject(
                    event_id=f"conv_{uuid.uuid4().hex[:8]}",
                    conversion_type=conv_type,
                    status=ConversionEventStatus.PROPOSED,
                    start_at=prop_match.group(0).strip().title(),
                    location_or_format="Property Address" if is_walkthrough else "Scheduled Meeting",
                    participants=["Agent", "Client"],
                    confirmation_confidence=0.50,
                    source_turn_ids=[bundle.turn_id],
                    blocking_items=[],
                    followup_is_conversion=True,
                )

        # ---------------------------------------------------------------------
        # Spec Acceptance Test #3: "Send me something" alone != conversion
        # ---------------------------------------------------------------------
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
                status=ConversionEventStatus.BLOCKED,
                start_at=None,
                location_or_format="email",
                participants=[bundle.speaker_id],
                confirmation_confidence=0.30,
                source_turn_ids=[bundle.turn_id],
                blocking_items=["brush_off_send_only_not_conversion", "no_confirmed_meeting_time"],
                followup_is_conversion=False,
            )

        # ---------------------------------------------------------------------
        # Walkthrough & Meeting Confirmation / Tentative Agreements
        # ---------------------------------------------------------------------
        meeting_patterns = [
            r"\b(meet|meeting|consultation|call|schedule|calendar)\b",
        ]
        is_meeting_topic = any(re.search(pat, text_lower) for pat in meeting_patterns)

        # Explicit timing indicators in utterance or facts
        time_patterns = [
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(tomorrow|today|this\s+week|next\s+week)\b",
            r"\b\d{1,2}(:\d{2})?\s*(am|pm|o'?clock)?\b",
        ]
        has_time = any(re.search(pat, text_lower) for pat in time_patterns)

        confirm_patterns = [
            r"\b(?:that\s+works|works\s+for\s+me|this\s+works|it\s+works)\b",
            r"\b(?:sounds\s+good|sounds\s+fair|perfect|let's\s+do\s+it|deal|fine\s+with\s+me|see\s+you\s+then|i'll\s+be\s+there)\b",
            r"\b(?:i\s+could\s+do|i\s+can\s+do|let's\s+meet|we\s+can\s+meet|come\s+by|stop\s+by)\b",
            r"\b(?:sure\s+let's\s+meet|sure\s+come\s+by)\b",
            r"(?<!doesn't\s)(?<!does\snot\s)(?<!won't\s)(?<!not\s)\bworks\b",
            r"\b(yes|yeah|sure|definitely|absolutely)\b",
        ]
        has_explicit_commit, commit_slot = self.detect_explicit_commitment(bundle, current_state)
        is_confirming = any(re.search(pat, text_lower) for pat in confirm_patterns) or has_explicit_commit

        hedged_patterns = [
            r"\b(?:could|might)\s+work\b",
            r"\b(?:could|might)\s+(?:be\s+able\s+to|possibly)\b",
            r"\bmaybe\b",
            r"\bperhaps\b",
            r"\btentative(?:ly)?\b",
            r"\bpossibly\b",
            r"\blet\s+me\s+(?:check|see|think)\b",
            r"\bnot\s+(?:100%|sure|certain)\b",
            r"\bif\s+(?:that|it)\s+works\b",
        ]
        is_hedged = any(re.search(pat, text_lower) for pat in hedged_patterns)

        time_slot_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b(?:\s+(?:at\s+)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)|(?:\s+(?:morning|afternoon|evening|noon)))?|(?:next\s+week|this\s+week|weekend)",
            text_lower,
        )

        # ---------------------------------------------------------------------
        # Case 1: Hedged / Tentative Prospect Response -> TENTATIVE status
        # Requires actual meeting discussion or time indicators
        # ---------------------------------------------------------------------
        if is_hedged and bundle.speaker_id == "client" and (has_time or is_meeting_topic or commit_slot):
            tentative_slot = (
                (time_slot_match.group(0).strip().title() if time_slot_match else None)
                or (previous_event.start_at if previous_event and previous_event.start_at not in ("Tentative Time Slot", "Confirmed Time Slot") else None)
            )
            if tentative_slot:
                conv_type = previous_event.conversion_type if previous_event else ("property_walkthrough" if is_walkthrough else "in_person_meeting")

                if previous_event and (
                    previous_event.status in (ConversionEventStatus.PROPOSED, ConversionEventStatus.ELIGIBLE, ConversionEventStatus.CANCELLED)
                    or (previous_event.status == ConversionEventStatus.TENTATIVE and previous_event.start_at and tentative_slot != previous_event.start_at)
                ):
                    new_event = ConversionEventObject(
                        event_id=f"conv_{uuid.uuid4().hex[:8]}",
                        conversion_type=conv_type,
                        status=ConversionEventStatus.TENTATIVE,
                        start_at=tentative_slot,
                        location_or_format="Property Address" if conv_type == "property_walkthrough" else "Scheduled Meeting",
                        participants=self._resolve_conversion_participants(current_state, bundle, previous_event),
                        confirmation_confidence=0.65,
                        source_turn_ids=sorted(list(set(previous_event.source_turn_ids + [bundle.turn_id]))),
                        blocking_items=[],
                        followup_is_conversion=True,
                        supersedes_event_id=previous_event.event_id,
                        reversal_reason="rescheduled" if (previous_event.status == ConversionEventStatus.TENTATIVE and tentative_slot != previous_event.start_at) else None,
                    )
                    previous_event.superseded_by_event_id = new_event.event_id
                    previous_event.superseded_at_turn_id = bundle.turn_id
                    return new_event
                elif previous_event and previous_event.status == ConversionEventStatus.TENTATIVE:
                    return ConversionEventObject(
                        event_id=previous_event.event_id,
                        conversion_type=conv_type,
                        status=ConversionEventStatus.TENTATIVE,
                        start_at=tentative_slot,
                        location_or_format="Property Address" if conv_type == "property_walkthrough" else "Scheduled Meeting",
                        participants=self._resolve_conversion_participants(current_state, bundle, previous_event),
                        confirmation_confidence=0.65,
                        source_turn_ids=sorted(list(set(previous_event.source_turn_ids + [bundle.turn_id]))),
                        blocking_items=[],
                        followup_is_conversion=True,
                        supersedes_event_id=previous_event.supersedes_event_id,
                        reversal_reason=previous_event.reversal_reason,
                    )
                else:
                    return ConversionEventObject(
                        event_id=f"conv_{uuid.uuid4().hex[:8]}",
                        conversion_type=conv_type,
                        status=ConversionEventStatus.TENTATIVE,
                        start_at=tentative_slot,
                        location_or_format="Property Address" if conv_type == "property_walkthrough" else "Scheduled Meeting",
                        participants=self._resolve_conversion_participants(current_state, bundle, previous_event),
                        confirmation_confidence=0.65,
                        source_turn_ids=[bundle.turn_id],
                        blocking_items=[],
                        followup_is_conversion=True,
                    )

        # ---------------------------------------------------------------------
        # Case 2: Firm / Unhedged Confirmation from Client -> CONFIRMED status
        # ---------------------------------------------------------------------
        if not is_hedged and bundle.speaker_id == "client":
            is_walkthrough_turn = is_walkthrough or (previous_event and previous_event.conversion_type == "property_walkthrough")
            confirmed_fact = next((f for f in current_state.facts if f.fact_key == "confirmed_meeting_time" and f.status == "active"), None)
            has_meeting_confirmation = bool(confirmed_fact) or (is_meeting_topic and (has_time or is_confirming)) or (previous_event and previous_event.start_at and has_time)

            if (gate.is_open or confirmed_fact or has_explicit_commit) and (is_confirming or is_walkthrough_turn or has_meeting_confirmation):
                extracted_time = (
                    commit_slot
                    or (time_slot_match.group(0).strip().title() if time_slot_match else None)
                    or (confirmed_fact.fact_value if confirmed_fact else None)
                    or (previous_event.start_at if previous_event and previous_event.start_at not in ("Confirmed Time Slot", "Tentative Time Slot") else None)
                )
                if extracted_time:
                    conv_type = "property_walkthrough" if is_walkthrough_turn else (previous_event.conversion_type if previous_event else "in_person_meeting")

                    if previous_event and (
                        previous_event.status in (ConversionEventStatus.PROPOSED, ConversionEventStatus.TENTATIVE, ConversionEventStatus.ELIGIBLE, ConversionEventStatus.CANCELLED)
                        or (previous_event.status == ConversionEventStatus.CONFIRMED and previous_event.start_at and extracted_time != previous_event.start_at)
                    ):
                        new_event = ConversionEventObject(
                            event_id=f"conv_{uuid.uuid4().hex[:8]}",
                            conversion_type=conv_type,
                            status=ConversionEventStatus.CONFIRMED,
                            start_at=extracted_time,
                            location_or_format="Property Address" if conv_type == "property_walkthrough" else "Scheduled Meeting",
                            participants=self._resolve_conversion_participants(current_state, bundle, previous_event),
                            confirmation_confidence=0.90,
                            source_turn_ids=sorted(list(set(previous_event.source_turn_ids + [bundle.turn_id]))),
                            blocking_items=[],
                            followup_is_conversion=True,
                            supersedes_event_id=previous_event.event_id,
                            reversal_reason="rescheduled" if (previous_event.status == ConversionEventStatus.CONFIRMED and extracted_time != previous_event.start_at) else None,
                        )
                        previous_event.superseded_by_event_id = new_event.event_id
                        previous_event.superseded_at_turn_id = bundle.turn_id
                        return new_event
                    elif previous_event and previous_event.status == ConversionEventStatus.CONFIRMED:
                        return ConversionEventObject(
                            event_id=previous_event.event_id,
                            conversion_type=conv_type,
                            status=ConversionEventStatus.CONFIRMED,
                            start_at=extracted_time,
                            location_or_format="Property Address" if conv_type == "property_walkthrough" else "Scheduled Meeting",
                            participants=self._resolve_conversion_participants(current_state, bundle, previous_event),
                            confirmation_confidence=0.90,
                            source_turn_ids=sorted(list(set(previous_event.source_turn_ids + [bundle.turn_id]))),
                            blocking_items=[],
                            followup_is_conversion=True,
                            supersedes_event_id=previous_event.supersedes_event_id,
                            reversal_reason=previous_event.reversal_reason,
                        )
                    else:
                        return ConversionEventObject(
                            event_id=f"conv_{uuid.uuid4().hex[:8]}",
                            conversion_type=conv_type,
                            status=ConversionEventStatus.CONFIRMED,
                            start_at=extracted_time,
                            location_or_format="Property Address" if conv_type == "property_walkthrough" else "Scheduled Meeting",
                            participants=self._resolve_conversion_participants(current_state, bundle, previous_event),
                            confirmation_confidence=0.90,
                            source_turn_ids=[bundle.turn_id],
                            blocking_items=[],
                            followup_is_conversion=True,
                        )

        # ---------------------------------------------------------------------
        # If gate is open but no explicit confirmation turn yet, state is eligible
        # ---------------------------------------------------------------------
        if gate.is_open:
            if previous_event and previous_event.status in (ConversionEventStatus.CONFIRMED, ConversionEventStatus.TENTATIVE, ConversionEventStatus.PROPOSED):
                return previous_event
            return ConversionEventObject(
                event_id=previous_event.event_id if previous_event else f"conv_{uuid.uuid4().hex[:8]}",
                conversion_type=previous_event.conversion_type if previous_event else "in_person_meeting",
                status=ConversionEventStatus.ELIGIBLE,
                start_at=previous_event.start_at if previous_event else None,
                location_or_format=previous_event.location_or_format if previous_event else None,
                participants=self._resolve_conversion_participants(current_state, bundle, previous_event),
                confirmation_confidence=0.60,
                source_turn_ids=sorted(list(set((previous_event.source_turn_ids if previous_event else []) + [bundle.turn_id]))),
                blocking_items=[],
                followup_is_conversion=True,
            )

        # ---------------------------------------------------------------------
        # Gate is closed
        # ---------------------------------------------------------------------
        if previous_event:
            if previous_event.status == ConversionEventStatus.CANCELLED:
                return previous_event
            # If previous event was TENTATIVE, PROPOSED, or ELIGIBLE and gate is closed:
            # Transition to BLOCKED because safety gate conditions are failing
            if previous_event.status in (ConversionEventStatus.TENTATIVE, ConversionEventStatus.PROPOSED, ConversionEventStatus.ELIGIBLE):
                return ConversionEventObject(
                    event_id=previous_event.event_id,
                    conversion_type=previous_event.conversion_type,
                    status=ConversionEventStatus.BLOCKED,
                    start_at=previous_event.start_at,
                    location_or_format=previous_event.location_or_format,
                    participants=previous_event.participants,
                    confirmation_confidence=0.20,
                    source_turn_ids=sorted(list(set(previous_event.source_turn_ids + [bundle.turn_id]))),
                    blocking_items=gate.blocking_reasons,
                    followup_is_conversion=previous_event.followup_is_conversion,
                    supersedes_event_id=previous_event.supersedes_event_id,
                    superseded_by_event_id=previous_event.superseded_by_event_id,
                    reversal_reason=previous_event.reversal_reason,
                )

        return None
