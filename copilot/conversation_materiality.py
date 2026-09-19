from __future__ import annotations

import os
import re
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import ConversationStateSnapshot
from .conversation_objections import CANONICAL_OBJECTION_PATTERNS, DECISION_TO_STAY_PATTERNS

LOGGER = logging.getLogger("copilot.conversation_materiality")

ABSENT_DECISION_MAKER_PATTERNS: List[str] = [
    r"\b(?:he['’]?s|she['’]?s|they['’]?re)\s+not\s+going\s+to\s+make\s+a\s+decision\b",
    r"\b(?:won['’]?t|will\s+not|can['’]?t|cannot)\s+make\s+a\s+decision\s+unless\b",
    r"\b(?:my\s+)?(?:business\s+)?(?:husband|wife|spouse|partner|boss|attorney|lawyer)\s+(?:is\s+not\s+here|is\s+out\s+of\s+town|handles?\s+all\s+(?:the\s+)?(?:financial\s+)?decisions?|makes?\s+all\s+(?:the\s+)?(?:financial\s+)?decisions?)\b",
    r"\b(?:husband|wife|spouse|partner)\s+is\s+not\s+here\b",
    r"\b(?:(?:my\s+)?(?:husband|wife|spouse|partner)|we)\b.*?\b(?:decide|make\s+(?:a\s+)?decision)\s+(?:everything\s+)?together\b",
    r"\b(?:consult|check|talk|speak)\s+with\s+(?:my\s+)?(?:husband|wife|spouse|partner|boss|attorney|lawyer)\b",
    r"\b(?:need|have)\s+to\s+(?:speak|talk|consult|check)\s+with\s+(?:my\s+)?(?:husband|wife|spouse|partner)\b",
    r"\b(?:my\s+)?(?:husband|wife|spouse|partner)\b.*?\b(?:part\s+of\s+(?:this|the)\s+conversation|involved|loop(?:ed)?\s+in|present|here)\b",
    r"\b(?:my\s+)?(?:husband|wife|spouse|partner)\b.*?\b(?:needs?|would\s+(?:really\s+)?need|has\s+to|must)\b.*?\b(?:conversation|decision|call|talk|meeting|input|present|here|further)\b",
    r"\b(?:my\s+)?(?:husband|wife|spouse|partner)\b.*?\bbefore\s+(?:we|i)\s+(?:go\s+any\s+further|make\s+a\s+decision|proceed|move\s+forward)\b",
    r"\b(?:our|my)\s+(?:attorney|lawyer)\s+(?:must|needs?\s+to)\s+review\b",
]

MaterialityTarget = Literal[
    "dimensions",
    "facts",
    "objections",
    "decision_structure",
    "contact_compliance",
]

ALL_TARGETS: Set[MaterialityTarget] = {
    "dimensions",
    "facts",
    "objections",
    "decision_structure",
    "contact_compliance",
}


class MaterialityClassification(BaseModel):
    """Result of turn-level materiality evaluation."""
    is_material: bool = False
    affected_targets: Set[MaterialityTarget] = Field(default_factory=set)
    reasoning: str = ""
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class MaterialityFilter:
    """Tier-1 Turn-Level Gating Engine (Phase 5 / Sprint 4).

    Decides which state fields a given turn actually affects, preventing
    gratuitous recomputation and preserving dimension stability (Client Principle #3).

    Core Invariants:
    1. Dimension Stability: A turn that only conveys a factual disclosure without
       acoustic/emotional shifts leaves Trust/Emotion/Pacing provably untouched.
    2. True Negatives: Casual chatter ("Looks like rain", "Sounds good") without
       domain facts, objections, or boundaries is marked non-material.
    3. Embedded Facts: Chit-chat concealing structural facts ("We were having coffee
       talking about how my brother-in-law co-owns the deed") is correctly flagged
       as material for decision_structure and facts.
    4. Weakest-link confidence tracking (Client Principle #8).
    """

    def __init__(self, groq_client: Optional[Any] = None, timeout_seconds: float = 3.0):
        self.groq_client = groq_client
        self.timeout_seconds = timeout_seconds

    def classify_turn(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: Optional[ConversationStateSnapshot] = None,
        has_explicit_fact_updates: bool = False,
        prior_bundle: Optional[BehavioralSignalInputBundle] = None,
    ) -> MaterialityClassification:
        """Evaluates whether a turn is material and which state sub-engines it affects."""
        api_key_groq = os.getenv("GROQ_API_KEY")
        is_testing = bool(os.getenv("PYTEST_CURRENT_TEST"))
        should_use_llm = self.groq_client is not None or (
            api_key_groq
            and not api_key_groq.startswith("mock_")
            and not api_key_groq.startswith("gsk_test")
            and not is_testing
        )
        if should_use_llm:
            try:
                result = self._classify_via_llm(
                    bundle,
                    current_state,
                    has_explicit_fact_updates,
                    api_key_groq or "",
                    prior_bundle=prior_bundle,
                )
            except Exception as exc:
                LOGGER.warning("LLM materiality classification failed, falling back to deterministic heuristic: %s", exc)
                result = self._classify_via_heuristic(
                    bundle,
                    current_state,
                    has_explicit_fact_updates,
                    prior_bundle=prior_bundle,
                )
        else:
            result = self._classify_via_heuristic(
                bundle,
                current_state,
                has_explicit_fact_updates,
                prior_bundle=prior_bundle,
            )

        # ---------------------------------------------------------------------
        # HARD DETERMINISTIC SAFETY OVERRIDES (Invariant Protection)
        # ---------------------------------------------------------------------
        # Under NO circumstances can a classifier (heuristic or LLM) suppress an
        # upstream hard compliance signal, contact preference, or objection recurrence.
        forced_targets: Set[MaterialityTarget] = set()
        forced_reasons: List[str] = []

        # 1. Hard Compliance Boundary Overrides
        # Architectural Note on 0.70 vs 0.80/0.85 Thresholds:
        # Downstream engines enforce specific cutoffs:
        #   - Behavioral Inference collapses Readiness & caps Trust at boundary_score >= 0.80
        #   - State Manager activates hard_boundary at boundary_score >= 0.85
        # As a Tier-1 gatekeeper, the Materiality Filter's override threshold is intentionally
        # set lower (>= 0.70) to serve as a protective funnel buffer. This guarantees that
        # elevated or borderline boundary threat turns are NEVER clipped at the front gate
        # before downstream engines evaluate them against their specific operational thresholds.
        if bundle.boundary_score >= 0.70:
            forced_targets.update(["contact_compliance", "objections", "dimensions"])
            forced_reasons.append("Deterministic Override: boundary_score >= 0.70 (buffer below 0.80/0.85 firing thresholds) requires contact_compliance, objections, and dimensions.")

        # 2. Contact Preference Overrides
        if bundle.contact_preference != "none":
            forced_targets.update(["contact_compliance", "facts"])
            forced_reasons.append("Deterministic Override: upstream contact_preference requires contact_compliance and facts.")

        # 3. Objection Recurrence & Strategy Overrides
        is_sched_recurrence = bool(
            bundle.recurrence_type in ("scheduling_detail_repeated", "scheduling_repeated", "positive_echo")
            or (bundle.recurrence_id and (bundle.recurrence_id.startswith("SCHED_") or "sched" in bundle.recurrence_id.lower()))
        )
        if (bundle.recurrence_id or bundle.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe")) and not is_sched_recurrence:
            forced_targets.update(["objections", "dimensions"])
            forced_reasons.append("Deterministic Override: upstream recurrence_id requires objections and dimensions.")

        if bundle.salesperson_strategy_tag:
            forced_targets.add("objections")
            forced_reasons.append("Deterministic Override: salesperson_strategy_tag requires objections.")

        # 4. Explicit Payload Fact Updates Overrides
        if has_explicit_fact_updates:
            forced_targets.add("facts")
            forced_reasons.append("Deterministic Override: explicit fact updates supplied.")
        # 5. Absent Decision-Maker & External Authority Overrides (Spec Test E / Client Principle #6)
        if bundle.speaker_id == "client" and any(re.search(pat, bundle.utterance_text.lower()) for pat in ABSENT_DECISION_MAKER_PATTERNS):
            forced_targets.update(["decision_structure", "facts"])
            forced_reasons.append("Deterministic Override: client disclosure indicates absent/external decision-maker authority (Spec Test E).")

        # 5b. Decision to Stay / Cancel Sale Supersession Overrides
        if bundle.speaker_id == "client" and any(re.search(p, bundle.utterance_text.lower()) for p in DECISION_TO_STAY_PATTERNS):
            forced_targets.update(["objections", "facts", "dimensions"])
            forced_reasons.append("Deterministic Override: client decision to stay/cancel sale supersedes active objections.")

        # 5c. Conversion Commitment or Cancellation / Reversal Overrides
        reversal_markers = [
            r"\b(?:cancel|cancelling|cancelled)\b",
            r"\b(?:never\s+mind|nevermind)\b",
            r"\b(?:let's\s+not|let\s+us\s+not)\s+(?:meet|do\s+that|do\s+this|schedule)\b",
            r"\bforget\s+(?:about\s+)?(?:it|that|thursday|friday|monday|tuesday|wednesday|tomorrow|the\s+meeting|meeting)\b",
            r"\b(?:can't|cannot|couldn't|could\s+not)\s+make\s+it\b",
            r"\bwon't\s+be\s+able\s+to\s+meet\b",
            r"\b(?:call\s+off|called\s+off)\b",
        ]
        if bundle.speaker_id == "client" and any(re.search(p, bundle.utterance_text.lower()) for p in reversal_markers):
            forced_targets.update(["facts", "dimensions"])
            forced_reasons.append("Deterministic Override: client explicit conversion reversal / cancellation.")

        # 5d. Objection Reversal / Withdrawal Overrides
        objection_reversal_markers = [
            r"\b(?:don't|do\s+not|not)\s+(?:care|worried|concerned)\s+about\s+(?:the\s+)?(?:fee|commission|percentage|rate)\b",
            r"\b(?:commission|fee)\s+(?:is\s+fine|isn't\s+an\s+issue|doesn't\s+matter|is\s+fair|is\s+okay)\b",
            r"\b(?:fine|okay|happy)\s+with\s+(?:the\s+)?(?:commission|fee|rate)\b",
            r"\bnever\s+mind\s+about\s+(?:the\s+)?(?:fee|commission)\b",
            r"\bready\s+to\s+(?:sell|move\s+forward|list|go)\s+now\b",
            r"\bnot\s+worried\s+about\s+(?:timing|the\s+market|rates)\b",
        ]
        if bundle.speaker_id == "client" and any(re.search(p, bundle.utterance_text.lower()) for p in objection_reversal_markers):
            forced_targets.update(["objections", "dimensions"])
            forced_reasons.append("Deterministic Override: client explicit objection reversal / withdrawal.")

        # 6. Behavioral & Acoustic Shifts (Client Principle #3: Dimension Stability & Materiality Gating)
        # Protect dimension sync: If upstream inference in bundle meaningfully diverges
        # from current state dimensions (delta >= 0.08 or client substantive agreement >= 0.70),
        # dimensions MUST be updated, regardless of whether LLM or heuristic classified it.
        if current_state and current_state.dimensions:
            cur_dim = current_state.dimensions
            valence_delta = abs(bundle.emotion.expressed_valence - cur_dim.emotion_valence)
            tension_delta = abs(bundle.emotion.tension_level - cur_dim.emotion_tension)
            trust_delta = abs(bundle.trust.score - cur_dim.trust)
            readiness_delta = abs(bundle.readiness.score - cur_dim.readiness)
            engagement_delta = abs(bundle.engagement.score - cur_dim.engagement)

            has_behavioral_shift = (
                valence_delta >= 0.08
                or tension_delta >= 0.08
                or trust_delta >= 0.08
                or readiness_delta >= 0.08
                or engagement_delta >= 0.08
                or (bundle.speaker_id == "client" and bundle.agreement_score >= 0.70)
            )
            if has_behavioral_shift and (bundle.speaker_id == "client" or result.is_material):
                forced_targets.add("dimensions")
                forced_reasons.append(
                    f"Deterministic Override: behavioral/acoustic shift detected "
                    f"(trust_d={trust_delta:.2f}, ready_d={readiness_delta:.2f}, val_d={valence_delta:.2f})."
                )

        if forced_targets:
            result.affected_targets.update(forced_targets)
            result.is_material = True
            if forced_reasons:
                result.reasoning = f"{result.reasoning} [{'; '.join(forced_reasons)}]".strip()

        return result

    def _classify_via_heuristic(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: Optional[ConversationStateSnapshot] = None,
        has_explicit_fact_updates: bool = False,
        prior_bundle: Optional[BehavioralSignalInputBundle] = None,
    ) -> MaterialityClassification:
        """Deterministic heuristic classifier with adversarial dual-direction support."""
        text = bundle.utterance_text.strip()
        text_lower = text.lower()
        targets: Set[MaterialityTarget] = set()
        reasons: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Initial State Bootstrap: First turn must initialize baseline dimensions
        # ---------------------------------------------------------------------
        is_initial_turn = current_state is None or current_state.state_version == 0
        if is_initial_turn:
            targets.add("dimensions")
            reasons.append("Initial state bootstrap: establishing dimension baseline.")

        # ---------------------------------------------------------------------
        # 2. Contact Compliance & Boundaries (Highest Consequence)
        # ---------------------------------------------------------------------
        boundary_markers = [
            r"\bdo\s+not\s+call\b",
            r"\bdon't\s+call\b",
            r"\bstop\s+calling\b",
            r"\bremove\s+(?:me|my\s+number)\b",
            r"\btake\s+me\s+off\b",
            r"\blawyer\b",
            r"\bsue\b",
            r"\bharassment\b",
            r"\bunsubscribe\b",
            r"\blose\s+my\s+number\b",
        ]
        has_boundary_words = any(re.search(p, text_lower) for p in boundary_markers)
        # Protective gate buffer: triggers at 0.70 so downstream engines can evaluate at 0.80/0.85
        has_boundary_score = bundle.boundary_score >= 0.70

        pref_markers = [
            r"\bdon't\s+(?:start\s+)?text(?:ing)?\b",
            r"\btext\s+or\s+email\b",
            r"\bcall\s+me\s+(?:after|before|in\s+the\s+morning|afternoon)\b",
            r"\bclinic\s+hours\b",
            r"\bprefer\s+(?:text|email|call)\b",
        ]
        has_pref_words = any(re.search(p, text_lower) for p in pref_markers) or bundle.contact_preference != "none"

        if has_boundary_words or has_boundary_score or has_pref_words:
            targets.add("contact_compliance")
            reasons.append("Contains contact boundary or communication channel preference.")
            if has_pref_words:
                targets.add("facts")
                reasons.append("Communication preferences also update persistent channel facts.")
            # Hard boundary also affects dimensions (collapsing readiness)
            if has_boundary_score or has_boundary_words:
                targets.add("dimensions")
                reasons.append("Hard boundary impacts readiness/trust dimensions.")

        # ---------------------------------------------------------------------
        # 3. Decision Structure & Authority (Stakeholders, Title, Legal Signers)
        # ---------------------------------------------------------------------
        # Direct marital decision partners (inherent real estate stakeholders)
        direct_marital_markers = [
            r"\b(?:wife|husband|spouse|partner)\b",
        ]
        # Legal and formal signing authorities
        legal_authority_markers = [
            r"\b(?:attorney|lawyer|power\s+of\s+attorney|executor|probate)\b",
            r"\b(?:sole\s+authority|joint\s+decision|consult\s+with|sign\s+off)\b",
            r"\b(?:decision\s+maker|handles?\s+(?:all\s+)?(?:the\s+)?decisions?)\b",
            r"\b(?:co-owner|co-signer)\b",
            r"\b(?:make\s+(?:a\s+|the\s+)?decision|makes?\s+(?:a\s+|the\s+)?decisions?|making\s+(?:a\s+|the\s+)?decision)\b",
            r"\b(?:decide\s+everything\s+together|we\s+decide\s+together)\b",
        ]
        # Explicit Legal Instruments: Definitive title/deed markers, immune to casual idiom phrases
        explicit_instrument_markers = [
            r"\b(?:on\s+the\s+deed|on\s+the\s+title|holds?\s+(?:the\s+)?title|title\s+and\s+deed|deed\s+and\s+title)\b",
            r"\b(?:on\s+the\s+contract|on\s+the\s+mortgage)\b",
        ]
        # Loose Ownership Language: Subject to figurative joke/hyperbole disqualification
        loose_ownership_markers = [
            r"\b(?:co-owns?\s+(?:the\s+)?(?:title|deed|property|house|home)|co-owns?)\b",
            r"\b(?:owns?\s+(?:the\s+)?(?:house|property|home|place))\b",
        ]

        has_explicit_instrument = any(re.search(p, text_lower) for p in explicit_instrument_markers)
        has_loose_ownership = any(re.search(p, text_lower) for p in loose_ownership_markers)

        # Figurative / hyperbolic idiom filter (e.g. "jokes he practically owns the place")
        figurative_ownership = bool(
            re.search(
                r"\b(?:jokes?|joking|kidding|claims?|thinks?|feels?|acts?)\s+(?:that\s+)?(?:he|she|they)?\s*(?:practically|basically|almost|like\s+he|like\s+she)?\s*owns?\b",
                text_lower,
            )
        )

        has_legal_authority = any(re.search(p, text_lower) for p in legal_authority_markers)
        has_direct_marital = any(re.search(p, text_lower) for p in direct_marital_markers)
        extended_family = bool(re.search(r"\b(?:brother(?:-in-law)?|sister(?:-in-law)?|in-law|cousin|uncle|aunt|parent)\b", text_lower))

        # Real title ownership: Explicit legal instrument ALWAYS holds; loose ownership only holds if not a figurative idiom
        valid_title_ownership = has_explicit_instrument or (has_loose_ownership and not figurative_ownership)
        has_extended_with_role = extended_family and (valid_title_ownership or has_legal_authority)

        is_decision_structure = (
            has_direct_marital
            or has_legal_authority
            or has_extended_with_role
            or valid_title_ownership
        )

        # Inconsistency #3 Fix: Salesperson questions (inquiries about third parties/signers)
        # are exploratory questions opening inquiry, not confirmed factual disclosures.
        is_salesperson_question = (
            bundle.speaker_id == "salesperson"
            and (
                text.endswith("?")
                or bundle.question_type != "none"
                or any(text_lower.startswith(q) for q in ["would you", "do you", "could you", "is there", "who", "what", "can you", "should we", "have you", "are you"])
            )
        )

        if is_decision_structure and not is_salesperson_question:
            targets.add("decision_structure")
            targets.add("facts")  # Structural roles are also persistent facts
            reasons.append("References decision stakeholder, title ownership, or signing authority.")

        # Inconsistency #5 (Turn 2) Fix: Check if client affirmatively confirms a prior salesperson inquiry about stakeholders
        if bundle.speaker_id == "client" and prior_bundle and prior_bundle.speaker_id == "salesperson":
            prior_text_l = prior_bundle.utterance_text.lower()
            asked_stakeholder = any(w in prior_text_l for w in ["him involved", "her involved", "them involved", "husband", "wife", "partner", "spouse", "decision maker", "sign off", "on the title", "on the deed"])
            is_affirmative = (
                bundle.agreement_score >= 0.60
                or any(re.search(rf"\b{aff}\b", text_lower) for aff in ["yeah", "yes", "definitely", "sure", "absolutely", "correct", "of course"])
            )
            if asked_stakeholder and is_affirmative:
                targets.add("decision_structure")
                targets.add("facts")
                reasons.append("Client confirms third-party stakeholder involvement inquired by agent.")

        # ---------------------------------------------------------------------
        # 4. Objections & Reframe Strategies
        # ---------------------------------------------------------------------
        has_active_objections = bool(
            current_state
            and any(
                o.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "active", "partially_addressed")
                for o in current_state.objections
            )
        )
        has_dormancy_aging_due = bool(
            current_state
            and any(
                o.lifecycle_state in ("active", "partially_addressed", "unresolved", "clarified", "partially_resolved", "reactivated")
                and (bundle.turn_id - o.last_updated_turn_id) >= 2
                for o in current_state.objections
            )
        )
        is_sched_rec = bool(
            bundle.recurrence_type in ("scheduling_detail_repeated", "scheduling_repeated", "positive_echo")
            or (bundle.recurrence_id and (bundle.recurrence_id.startswith("SCHED_") or "sched" in bundle.recurrence_id.lower()))
        )
        has_objection_tag = bool(
            (bundle.recurrence_id or bundle.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe"))
            and not is_sched_rec
        )
        has_reframe_tag = bool(bundle.salesperson_strategy_tag)
        has_canonical_match = any(
            re.search(p, text_lower)
            for p_list in CANONICAL_OBJECTION_PATTERNS.values()
            for p in p_list
        )
        has_agent_clarification = (
            bundle.speaker_id == "salesperson"
            and bundle.question_type == "clarifying"
            and has_active_objections
        )

        if (
            has_objection_tag
            or has_reframe_tag
            or has_canonical_match
            or has_agent_clarification
            or has_boundary_score
            or has_boundary_words
            or has_dormancy_aging_due
            or (has_active_objections and (bundle.agreement_score > 0.35 or bundle.salesperson_strategy_tag))
        ):
            targets.add("objections")
            reasons.append("Contains active objection, canonical marker, or salesperson clarification/reframe.")
            # Objections affect readiness and engagement trends
            targets.add("dimensions")
            reasons.append("Objection dynamics impact readiness and momentum dimensions.")

        # ---------------------------------------------------------------------
        # 5. Persistent Facts & Truth Supersession
        # ---------------------------------------------------------------------
        if has_explicit_fact_updates:
            targets.add("facts")
            reasons.append("Explicit fact updates supplied in bundle payload.")

        fact_domain_markers = [
            # Timeline / relocation / selling
            r"\b(?:move|moving|stay|staying|sell|selling|list|listing|market|closing|close)\b",
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(?:spring|summer|fall|winter|holidays|thanksgiving|christmas|semester)\b",
            # Financial expectations
            r"\b(?:price|pricing|budget|dollars|\$|proceeds|netting)\b",
            # Conditional statements
            r"\bif\s+(?:i|we)\s+believed\s+there\s+was\s+a\s+better\b",
            r"\bopen\s+to\s+moving\s+if\b",
        ]
        if any(re.search(p, text_lower) for p in fact_domain_markers) and not is_salesperson_question:
            targets.add("facts")
            reasons.append("Mentions deal parameters (timeline, pricing, or moving decision).")

        # ---------------------------------------------------------------------
        # 6. Behavioral & Acoustic Shifts (Dimensions)
        # ---------------------------------------------------------------------
        # If not already flagged by an objection or boundary, check if acoustic/behavioral
        # scores in bundle meaningfully diverge from current snapshot.
        if not is_initial_turn and current_state and "dimensions" not in targets:
            cur_dim = current_state.dimensions

            # Meaningful deltas
            valence_delta = abs(bundle.emotion.expressed_valence - cur_dim.emotion_valence)
            tension_delta = abs(bundle.emotion.tension_level - cur_dim.emotion_tension)
            trust_delta = abs(bundle.trust.score - cur_dim.trust)
            readiness_delta = abs(bundle.readiness.score - cur_dim.readiness)
            engagement_delta = abs(bundle.engagement.score - cur_dim.engagement)

            has_behavioral_shift = (
                valence_delta >= 0.08
                or tension_delta >= 0.08
                or trust_delta >= 0.08
                or readiness_delta >= 0.08
                or engagement_delta >= 0.08
                or (bundle.speaker_id == "client" and bundle.agreement_score >= 0.70)
            )

            if has_behavioral_shift:
                targets.add("dimensions")
                reasons.append(
                    f"Acoustic/emotional shift detected (valence_d={valence_delta:.2f}, "
                    f"tension_d={tension_delta:.2f}, trust_d={trust_delta:.2f})."
                )

        # ---------------------------------------------------------------------
        # 7. Non-Material Filter (True Negatives: Casual Pleasantries / Weather)
        # ---------------------------------------------------------------------
        # If targets is empty, this turn has no material impact on state!
        is_material = len(targets) > 0
        if not is_material:
            reason_summary = "Non-material conversational chatter or filler: no state targets affected."
        else:
            reason_summary = "; ".join(reasons)

        effective_confidence = round(
            min(bundle.semantic_confidence, bundle.inference_confidence), 3
        )

        return MaterialityClassification(
            is_material=is_material,
            affected_targets=targets,
            reasoning=reason_summary,
            confidence=effective_confidence,
        )

    def _classify_via_llm(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: Optional[ConversationStateSnapshot],
        has_explicit_fact_updates: bool,
        api_key: str,
        prior_bundle: Optional[BehavioralSignalInputBundle] = None,
    ) -> MaterialityClassification:
        """LLM-based comparative classification with conversational context."""
        prior_context = ""
        if prior_bundle:
            prior_context = f"Prior Turn [Speaker: {prior_bundle.speaker_id}]: \"{prior_bundle.utterance_text}\"\n"

        prompt = (
            f"You are an expert sales conversational linguist evaluating turn materiality.\n\n"
            f"{prior_context}"
            f"Current Utterance: \"{bundle.utterance_text}\"\n"
            f"Speaker: {bundle.speaker_id}\n"
            f"Recurrence/Objection ID: {bundle.recurrence_id or 'None'}\n"
            f"Boundary Score: {bundle.boundary_score:.2f}\n\n"
            f"Task: Determine which of the following conversation state targets this turn materially affects:\n"
            f"- dimensions (behavioral trust, emotional valence, readiness, engagement)\n"
            f"- facts (persistent deal facts, timeline, budget, channel, conditionality)\n"
            f"- objections (objection raises, pushbacks, reframes, resolutions)\n"
            f"- decision_structure (stakeholders, authority, spouses, co-owners, title, lawyers)\n"
            f"- contact_compliance (channel restrictions, calling hours, hard boundaries)\n\n"
            f"Important Rules:\n"
            f"1. Casual pleasantries or weather chatter ('Looks like rain', 'Good morning') with no deal facts are NOT material.\n"
            f"2. Statements that superficially look like casual chatter but contain embedded structural facts "
            f"('We were having coffee talking about how my brother-in-law co-owns the deed') MUST be marked for decision_structure and facts.\n"
            f"3. A pure factual disclosure should NOT mark dimensions unless significant emotion/tension is present.\n"
            f"4. AGENT INQUIRIES & PROPOSALS: Questions or scheduling proposals asked by the salesperson (e.g. 'Thursday at three still work?', 'Would you want him involved?', 'Who else makes decisions?') "
            f"are exploratory inquiries/proposals, NOT established deal facts. Do NOT mark decision_structure or facts for an agent inquiry alone; wait for prospect confirmation.\n"
            f"5. PROSPECT CONFIRMATIONS: If the salesperson previously proposed a meeting time or asked about involving a stakeholder, and the prospect affirmatively confirms "
            f"('Yeah. Definitely.', 'Yes', 'Absolutely', 'Thursday works'), this prospect confirmation IS material for facts (and decision_structure if involving stakeholders).\n\n"
            f"Respond with raw JSON containing:\n"
            f"- is_material: boolean\n"
            f"- affected_targets: list of strings from ['dimensions', 'facts', 'objections', 'decision_structure', 'contact_compliance']\n"
            f"- reasoning: concise explanation\n"
            f"- confidence: float 0.0 to 1.0\n"
            f"Output ONLY raw JSON."
        )

        if self.groq_client is not None:
            client = self.groq_client
        else:
            import groq
            client = groq.Groq(api_key=api_key, timeout=self.timeout_seconds)

        groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        if "openai/gpt-oss" in groq_model or "llama" in groq_model:
            groq_model = "qwen/qwen3.8-27b"

        completion = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(completion.choices[0].message.content.strip())

        is_mat = bool(parsed.get("is_material", False))
        raw_targets = parsed.get("affected_targets", [])
        valid_targets: Set[MaterialityTarget] = {t for t in raw_targets if t in ALL_TARGETS}
        if has_explicit_fact_updates:
            valid_targets.add("facts")
            is_mat = True

        return MaterialityClassification(
            is_material=is_mat and len(valid_targets) > 0,
            affected_targets=valid_targets,
            reasoning=parsed.get("reasoning", "LLM materiality classification"),
            confidence=float(parsed.get("confidence", 0.90)),
        )
