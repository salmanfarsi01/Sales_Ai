from __future__ import annotations

import os
import re
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import ConversationStateSnapshot, DormancyEvidence, ObjectionRecord
from .conversation_objections import CANONICAL_OBJECTION_PATTERNS, DECISION_TO_STAY_PATTERNS, classify_objection_label, is_decision_authority_statement
from .behavioral_semantic import HARD_BOUNDARY_PATTERNS, SOFT_PREFERENCE_PATTERNS

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
    r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\s+(?:is|was|feels|seems)?\s*(?:really\s+|just\s+)?(?:thinks|believes|feels|says|told|warned|worried|concerned|skeptical|doubtful|wants|needs)\b",
    r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\s+(?:told\s+me|warned\s+me)\b",
    r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\s+wants\s+to\s+look\s+at\s+it\s+first\b",
    r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\s+(?:is\s+)?(?:really\s+)?(?:skeptical|hesitant|worried)\b",
]

PRESENCE_CONFIRMATION_PATTERNS: List[str] = [
    r"\b(?:my\s+)?(?:wife|husband|partner|spouse)\s+will\s+be\s+there\b",
    r"\b(?:both\s+of\s+us|we\s+will\s+both)\s+be\s+there\b",
    r"\b(?:she|he|they)\s+will\s+be\s+there\b",
    r"\b(?:my\s+)?(?:wife|husband|partner|spouse)\s+(?:can|will)\s+(?:make\s+it|attend|join)\b",
    r"\b(?:bringing|bring)\s+(?:my\s+)?(?:wife|husband|partner|spouse)\b",
    r"\b(?:wife|husband|partner|spouse)\s+is\s+(?:on\s+board|aligned|agreed|available)\b",
]

LOGISTICAL_SCHEDULING_PATTERNS: List[str] = [
    r"\b(?:mornings?|afternoons?|evenings?|weekdays?|weekends?)\s+(?:don['’]?t|do\s+not|aren['’]?t|won['’]?t|can['’]?t|cannot)\s+(?:really\s+)?work\b",
    r"\b(?:mornings?|afternoons?|evenings?|weekdays?|weekends?)\s+(?:only|preferred|work\s+better|suit\s+us\s+better)\b",
    r"\b(?:only\s+free|only\s+available|better\s+for\s+us|doesn['’]?t\s+work|does\s+not\s+work)\b",
    r"\bnot\s+available\s+in\s+the\s+(?:mornings?|afternoons?|evenings?)\b",
    r"\bcan['’]?t\s+do\s+(?:mornings?|afternoons?|evenings?|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b",
    r"\b(?:sometime\s+next\s+year|moving\s+next\s+year|look\s+at\s+moving|next\s+year|nothing\s+urgent)\b",
    r"\b(?:maybe\s+next\s+week|sometime\s+next\s+week|next\s+week\s+could\s+work|think\s+about\s+it)\b",
    r"\b(?:earlier|later)\s+in\s+the\s+(?:day|week|month)\s+(?:works?|is\s+better)\b",
    r"\b(?:thursday|friday|monday|tuesday|wednesday|saturday|sunday)\s+(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|afternoon|morning|evening)\b",
]

EXPLICIT_MEETING_RESISTANCE_PATTERNS: List[str] = [
    r"\b(?:don['’]?t|do\s+not|won['’]?t)\s+(?:want\s+to\s+)?meet\b",
    r"\bno\s+(?:point|need|reason)\s+(?:in\s+)?meeting\b",
    r"\bdon['’]?t\s+(?:come\s+over|bother)\b",
    r"\bnot\s+(?:ready|interested\s+in)\s+meeting\b",
    r"\bstop\s+trying\s+to\s+schedule\b",
    r"\brefuse\s+to\s+meet\b",
    r"\b(?:i['’]?m|we['’]?re)\s+too\s+busy\s+to\s+meet\b",
    r"\bdon['’]?t\s+have\s+time\s+for\s+(?:this|a\s+meeting)\b",
]

SOFT_CONTACT_PREF_PATTERNS: List[str] = [
    r"\bdon['’]?t\s+(?:start\s+)?text(?:ing)?\b",
    r"\btext\s+or\s+email\b",
    r"\bcall\s+me\s+(?:after|before|in\s+the\s+morning|afternoon)\b",
    r"\bclinic\s+hours\b",
    r"\bprefer\s+(?:text|email|call)\b",
    r"\b(?:text|email|call)\s+(?:is\s+better|preferred|instead)\b",
    r"\b(?:no|don['’]?t\s+make|no\s+more)\s+(?:phone\s+)?calls?\b",
    r"\b(?:no\s+(?:phone\s+)?calls?|don['’]?t\s+call)\s+(?:after|before)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
    r"\bemail\s+(?:me\s+)?(?:the\s+)?(?:contract|details|info|information)?\b",
    r"\b(?:email|text)\s+only\b",
    r"\b(?:text|email)\s+is\s+fine\b",
    r"\bdon['’]?t\s+call\s+me\s+before\s+(?:noon|\d{1,2})\b",
    r"\bno\s+phone\s+calls\b",
]

POSITIVE_FILLER_PATTERNS: List[str] = [
    # Short one or two-word conversational acknowledgments
    r"^(?:yeah|yes|okay|ok|sure|right|uh-huh|yep|gotcha|mm-hm|mhm|yup|nope|no|alright|all\s+right|sounds\s+good)[\.\,\!\?]*$",
    # Basic pleasantries / greetings with no domain facts
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|how\s+are\s+you(?:\s+doing)?|doing\s+well|fine\s+thanks)[\.\,\!\?]*$",
    # Polite closings / thanks with no domain constraints
    r"^(?:thanks|thank\s+you|have\s+a\s+good\s+(?:day|one)|take\s+care|bye|goodbye)[\.\,\!\?]*$",
    # Pure weather / superficial chatter without deal tokens
    r".*\b(?:looks\s+like\s+(?:it\s+might\s+)?rain|weather|cold\s+outside|hot\s+today|enjoy\s+the\s+weather)\b.*",
    # Casual idioms / jokes / banter with no structural decision meaning
    r".*\b(?:jokes?\s+he(?:\s+practically)?|just\s+joking|kidding\s+around|figure\s+of\s+speech)\b.*",
]


def is_positive_filler(text: str) -> bool:
    """True if utterance matches known non-material conversational filler/pleasantry whitelist."""
    t = text.strip().lower()
    deal_tokens = [
        "scam", "fraud", "fee", "fees", "contract", "closing", "deposit", "money",
        "disappeared", "vanished", "price", "property", "house", "mortgage",
        "email me", "text me", "don't call", "no calls", "call me", "text only"
    ]
    if any(k in t for k in deal_tokens):
        return False
    return any(re.search(p, t) for p in POSITIVE_FILLER_PATTERNS)


def is_soft_contact_preference(text: str, bundle: BehavioralSignalInputBundle) -> bool:
    """Detects soft boundary or communication cadence/channel preference (mutually exclusive with objection)."""
    text_lower = text.lower()
    if bundle.contact_preference != "none":
        return True
    if any(re.search(p, text_lower) for p in SOFT_CONTACT_PREF_PATTERNS):
        return True
    if isinstance(SOFT_PREFERENCE_PATTERNS, dict):
        for pat_list in SOFT_PREFERENCE_PATTERNS.values():
            if any(re.search(p, text_lower) for p in pat_list):
                return True
    elif isinstance(SOFT_PREFERENCE_PATTERNS, (list, set, tuple)):
        if any(re.search(p, text_lower) for p in SOFT_PREFERENCE_PATTERNS):
            return True
    return False


def has_explicit_meeting_resistance(text: str) -> bool:
    """Detects active pushback/refusal against meeting (transforms scheduling constraint into objection)."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in EXPLICIT_MEETING_RESISTANCE_PATTERNS)


def is_logistical_scheduling_constraint(text: str) -> bool:
    """Detects logistical availability / scheduling constraints without explicit meeting resistance."""
    text_lower = text.lower()
    if has_explicit_meeting_resistance(text):
        return False
    return any(re.search(p, text_lower) for p in LOGISTICAL_SCHEDULING_PATTERNS)

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
        if bundle.contact_preference != "none" or (bundle.speaker_id == "client" and is_soft_contact_preference(bundle.utterance_text, bundle)):
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

        # 3b. Client Canonical Objection Expression Overrides
        if bundle.speaker_id == "client" and not is_decision_authority_statement(bundle.utterance_text):
            obj_label = classify_objection_label(bundle.utterance_text)
            if obj_label:
                forced_targets.update(["objections", "dimensions"])
                forced_reasons.append(f"Deterministic Override: client expressed canonical objection ({obj_label}).")

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
            forced_targets.add("facts")
            if current_state and getattr(current_state.dimensions, "commitment", 0.0) > 0.0:
                forced_targets.add("dimensions")
            forced_reasons.append("Deterministic Override: client explicit conversion reversal / cancellation.")

        # 5d. Objection Reversal / Withdrawal Overrides
        objection_reversal_markers = [
            r"\b(?:don't|do\s+not|not)\s+(?:care|worried|concerned)\s+about\s+(?:the\s+)?(?:fee|commission|percentage|rate)\b",
            r"\b(?:commission|fee)\s+(?:is\s+fine|isn't\s+an\s+issue|doesn't\s+matter|is\s+fair|is\s+okay)\b",
            r"\b(?:fine|okay|happy)\s+with\s+(?:the\s+)?(?:commission|fee|rate)\b",
            r"\bnever\s+mind\s+about\s+(?:the\s+)?(?:fee|commission)\b",
            r"\bready\s+to\s+(?:sell|move\s+forward|list|go)\s+now\b",
            r"\bnot\s+worried\s+about\s+(?:timing|the\s+market|rates)\b",
            r"\b(?:timing|market)\s+(?:isn't|is\s+not)\s+(?:the\s+biggest\s+|an?\s+)?issue\b",
            r"\b(?:makes?\s+sense|fair\s+enough)\b.*?\b(?:timing|wait|now)\b",
            r"\btiming\s+isn't\s+the\s+biggest\s+issue\b",
        ]
        if bundle.speaker_id == "client" and any(re.search(p, bundle.utterance_text.lower()) for p in objection_reversal_markers):
            forced_targets.update(["objections", "dimensions"])
            forced_reasons.append("Deterministic Override: client explicit objection reversal / withdrawal.")

        # 5e. Client Tentative Timeline & Scheduling Disclosures
        tentative_timeline_patterns = [
            r"\b(?:sometime\s+next\s+year|moving\s+next\s+year|look\s+at\s+moving|next\s+year|nothing\s+urgent)\b",
            r"\b(?:maybe\s+next\s+week|sometime\s+next\s+week|next\s+week\s+could\s+work|think\s+about\s+it)\b",
            r"\b(?:mornings?\s+(?:don['’]?t|do\s+not)\s+(?:really\s+)?work|afternoons?\s+(?:only|preferred|work\s+better))\b",
            r"\bnot\s+available\s+in\s+the\s+mornings?\b",
        ]
        if bundle.speaker_id == "client" and (
            any(re.search(p, bundle.utterance_text.lower()) for p in tentative_timeline_patterns)
            or is_logistical_scheduling_constraint(bundle.utterance_text)
        ):
            forced_targets.update(["decision_structure", "facts"])
            forced_reasons.append("Deterministic Override: client timeline or scheduling availability disclosure.")

        # 5f. Stakeholder Presence Confirmation Overrides (Client Principle #1)
        if bundle.speaker_id == "client" and any(re.search(p, bundle.utterance_text.lower()) for p in PRESENCE_CONFIRMATION_PATTERNS):
            forced_targets.update(["decision_structure", "facts"])
            forced_reasons.append("Deterministic Override: client confirmed stakeholder presence / attendance.")

        # 6. Behavioral & Acoustic Shifts (Client Principle #3: Dimension Stability & Materiality Gating)
        # Protect dimension sync: If upstream inference in bundle meaningfully diverges
        # from current state dimensions (delta >= 0.08),
        # dimensions MUST be updated, regardless of whether LLM or heuristic classified it.
        if current_state and current_state.dimensions:
            cur_dim = current_state.dimensions
            valence_delta = abs(bundle.emotion.expressed_valence - cur_dim.emotion_valence)
            tension_delta = abs(bundle.emotion.tension_level - cur_dim.emotion_tension)
            trust_delta = abs(bundle.trust.score - cur_dim.trust)
            readiness_delta = abs(bundle.readiness.score - cur_dim.readiness)
            engagement_delta = abs(bundle.engagement.score - cur_dim.engagement)

            deltas: Dict[str, float] = {
                "trust_d": trust_delta,
                "ready_d": readiness_delta,
                "val_d": valence_delta,
                "ten_d": tension_delta,
                "eng_d": engagement_delta,
            }
            if bundle.commitment is not None:
                cur_commit = getattr(cur_dim, "commitment", 0.0)
                deltas["commit_d"] = abs(bundle.commitment.score - cur_commit)

            # Strictly require at least one delta >= 0.08
            active_shifts = {k: v for k, v in deltas.items() if v >= 0.08}
            if active_shifts and (bundle.speaker_id == "client" or result.is_material):
                forced_targets.add("dimensions")
                shift_desc = ", ".join(f"{k}={v:.2f}" for k, v in active_shifts.items())
                forced_reasons.append(
                    f"Deterministic Override: behavioral/acoustic shift detected ({shift_desc})."
                )

        if forced_targets:
            result.affected_targets.update(forced_targets)
            result.is_material = True
            if forced_reasons:
                result.reasoning = f"{result.reasoning} [{'; '.join(forced_reasons)}]".strip()

        # Enforce mutual exclusivity for soft contact preferences and logistical scheduling constraints (Client Items #2 & #3)
        if bundle.speaker_id == "client":
            text_raw = bundle.utterance_text
            text_l = text_raw.lower()
            canonical_obj = classify_objection_label(text_raw)
            hard_bound_words = any(re.search(p, text_l) for p in HARD_BOUNDARY_PATTERNS)
            hard_bound = bundle.boundary_score >= 0.70 or hard_bound_words
            has_resistance = has_explicit_meeting_resistance(text_raw)

            pure_soft_pref = is_soft_contact_preference(text_raw, bundle) and not hard_bound and not canonical_obj
            pure_sched_constraint = is_logistical_scheduling_constraint(text_raw) and not has_resistance and not canonical_obj

            if pure_soft_pref or pure_sched_constraint:
                # Ensure objections is purged from affected targets
                result.affected_targets.discard("objections")

                # If no genuine acoustic shift occurred (deltas < 0.08), purge spurious dimensions trigger
                cur_dim = current_state.dimensions if current_state else None
                if cur_dim:
                    val_d = abs(bundle.emotion.expressed_valence - cur_dim.emotion_valence)
                    ten_d = abs(bundle.emotion.tension_level - cur_dim.emotion_tension)
                    tru_d = abs(bundle.trust.score - cur_dim.trust)
                    rea_d = abs(bundle.readiness.score - cur_dim.readiness)
                    eng_d = abs(bundle.engagement.score - cur_dim.engagement)
                    if not any(d >= 0.08 for d in [val_d, ten_d, tru_d, rea_d, eng_d]):
                        result.affected_targets.discard("dimensions")
                        result.reasoning = re.sub(r"Acoustic/emotional shift detected[^;\]]+[;\]]?", "", result.reasoning).strip()

                # Sanitize reasoning string: strip objection-flavored phrases
                reason_clean = result.reasoning
                objection_phrases = [
                    "Contains active objection, canonical marker, or salesperson clarification/reframe.",
                    "Objection dynamics impact readiness and momentum dimensions.",
                    "Contains active objection",
                    "canonical marker",
                    "Objection dynamics impact readiness and momentum dimensions",
                ]
                for phrase in objection_phrases:
                    reason_clean = reason_clean.replace(phrase, "")
                reason_clean = re.sub(r"\s+", " ", reason_clean).strip()
                reason_clean = re.sub(r"\[\s*;\s*", "[", reason_clean)
                reason_clean = re.sub(r";\s*\]", "]", reason_clean)
                reason_clean = re.sub(r"\[\s*\]", "", reason_clean).strip()
                result.reasoning = reason_clean

                if pure_soft_pref:
                    result.affected_targets.add("contact_compliance")
                    result.affected_targets.add("facts")
                    if "Soft communication preference" not in result.reasoning:
                        result.reasoning = (result.reasoning + " Soft communication preference: prospect specified contact cadence or channel preference.").strip()

                if pure_sched_constraint:
                    result.affected_targets.add("decision_structure")
                    result.affected_targets.add("facts")
                    if "Logistical scheduling constraint" not in result.reasoning:
                        result.reasoning = (result.reasoning + " Logistical scheduling constraint: prospect declared availability constraint to coordinate meeting logistics.").strip()

                result.reasoning = re.sub(r"[\s;]+$", "", result.reasoning.strip()).strip()

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
        has_boundary_words = any(re.search(p, text_lower) for p in boundary_markers) or any(re.search(p, text_lower) for p in HARD_BOUNDARY_PATTERNS)
        # Protective gate buffer: triggers at 0.70 so downstream engines can evaluate at 0.80/0.85
        has_boundary_score = bundle.boundary_score >= 0.70
        is_hard_boundary = has_boundary_words or has_boundary_score

        is_soft_pref = is_soft_contact_preference(text, bundle)
        is_sched_constraint = is_logistical_scheduling_constraint(text)
        has_canonical_obj = bool(bundle.speaker_id == "client" and classify_objection_label(text))
        has_meeting_res = has_explicit_meeting_resistance(text)

        skip_objection_classification = False

        # Boundary retraction check
        retraction_patterns = [
            r"\b(?:actually\b\s*,?\s*)?(?:you\s+can|feel\s+free\s+to|go\s+ahead\s+and)\s+(?:call|reach\s+out|contact|text)\b",
            r"\b(?:i\s+changed\s+my\s+mind|never\s+mind|changed\s+mind)\b.*?\b(?:call|reach\s+out|contact|talk)\b",
            r"\b(?:go\s+ahead\s+and\s+call|call\s+me\s+back|call\s+me\s+tomorrow|call\s+me\s+later)\b",
            r"\b(?:it['’]?s\s+okay\s+to|fine\s+to|you\s+may)\s+(?:call|contact|reach\s+out)\b",
        ]
        is_retraction = (
            bundle.speaker_id == "client"
            and current_state is not None
            and getattr(current_state, "contact_compliance", None) is not None
            and current_state.contact_compliance.hard_boundary_active
            and any(re.search(p, text.lower()) for p in retraction_patterns)
        )

        if is_hard_boundary:
            targets.add("contact_compliance")
            reasons.append("Contains contact boundary (stop contact / legal threat).")
            targets.add("dimensions")
            reasons.append("Hard boundary impacts readiness/trust dimensions.")
        elif is_retraction:
            targets.add("contact_compliance")
            reasons.append("Boundary retraction: prospect explicitly invited contact after prior hard boundary.")
        elif is_soft_pref:
            targets.add("contact_compliance")
            targets.add("facts")
            reasons.append("Soft communication preference: prospect specified contact cadence or channel preference.")
            if not has_canonical_obj:
                skip_objection_classification = True

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

        # Check if prospect stated a logistical scheduling constraint (Client Feedback #3)
        if is_sched_constraint and not has_meeting_res:
            targets.add("decision_structure")
            targets.add("facts")
            reasons.append("Logistical scheduling constraint: prospect declared availability constraint to coordinate meeting logistics.")
            if not has_canonical_obj:
                skip_objection_classification = True

        # ---------------------------------------------------------------------
        # 4. Objections & Reframe Strategies
        # ---------------------------------------------------------------------
        if not skip_objection_classification:
            has_active_objections = bool(
                current_state
                and any(
                    o.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "active", "partially_addressed")
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
                or is_hard_boundary
                or (has_meeting_res and bundle.speaker_id == "client")
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
            bundle_commit = bundle.commitment.score if bundle.commitment is not None else 0.0
            cur_commit = getattr(cur_dim, "commitment", 0.0)
            commitment_delta = abs(bundle_commit - cur_commit)

            has_behavioral_shift = (
                valence_delta >= 0.08
                or tension_delta >= 0.08
                or trust_delta >= 0.08
                or readiness_delta >= 0.08
                or engagement_delta >= 0.08
                or commitment_delta >= 0.08
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
        # Client Principle: Filler is an explicit whitelist (short acknowledgments / pleasantries).
        # Any substantive statement by the client that does not match the filler whitelist
        # must default to material, rather than dropping domain disclosures as filler.
        if len(targets) == 0:
            if bundle.speaker_id == "client" and not is_positive_filler(text):
                disclaimers = [
                    r"\bnot\s+why\s+i['’]?m\s+(?:calling|reaching\s+out)\b",
                    r"\bnot\s+(?:an?\s+issue|a\s+problem|worried)\b",
                    r"\bdisappeared\s+into\s+(?:the|a)\b",
                ]
                has_disclaimer = any(re.search(d, text_lower) for d in disclaimers)
                trust_or_risk_keywords = [
                    "scam", "fraud", "upfront fee", "deposit", "burned", "ripoff",
                    "rip-off", "distrust", "skeptical", "caution"
                ]
                has_compound_loss = ("fee" in text_lower or "money" in text_lower) and ("disappeared" in text_lower or "vanished" in text_lower)
                if not has_disclaimer and (any(k in text_lower for k in trust_or_risk_keywords) or has_compound_loss):
                    targets.add("objections")
                    reasons.append("Substantive prospect disclosure containing trust, scam, fee, or risk context.")
                else:
                    reasons.append("Substantive prospect disclosure (non-filler).")

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
            f"3. A turn should ONLY mark dimensions if explicit emotional cues, hostility, or trust shifts are present. Do NOT mark dimensions for cancellations, boundaries, or factual disclosures unless accompanied by a shift in trust, tension, or commitment.\n"
            f"4. AGENT INQUIRIES & PROPOSALS: Questions or scheduling proposals asked by the salesperson (e.g. 'Thursday at three still work?', 'Would you want him involved?', 'Who else makes decisions?') "
            f"are exploratory inquiries/proposals, NOT established deal facts. Do NOT mark decision_structure or facts for an agent inquiry alone; wait for prospect confirmation.\n"
            f"5. PROSPECT CONFIRMATIONS: If the salesperson previously proposed a meeting time or asked about involving a stakeholder, and the prospect affirmatively confirms "
            f"('Yeah. Definitely.', 'Yes', 'Absolutely', 'Thursday works'), this prospect confirmation IS material for facts (and decision_structure if involving stakeholders).\n"
            f"6. SOFT CONTACT PREFERENCES: Communication channel/cadence preferences (e.g. 'Please don't text me every day', 'Call me after 5pm', 'Prefer email') "
            f"MUST be marked for contact_compliance and facts, and MUST NEVER be marked for objections.\n"
            f"7. LOGISTICAL SCHEDULING CONSTRAINTS: Statements expressing time-of-day, day-of-week, or availability constraints to coordinate meeting logistics "
            f"(e.g. 'Mornings don't really work for us', 'Afternoons are better') MUST be marked for decision_structure and facts, and MUST NEVER be marked for objections "
            f"unless accompanied by explicit refusal or resistance to meeting (e.g. 'I don't have time for this', 'Refuse to meet').\n\n"
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
            client = groq.Groq(api_key=api_key, timeout=self.timeout_seconds, max_retries=0)

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


def _get_dormancy_evidence(
    objection: ObjectionRecord,
    state: ConversationStateSnapshot,
    current_turn_id: int,
    recent_bundles: Optional[List[BehavioralSignalInputBundle]] = None,
) -> Optional[DormancyEvidence]:
    """Evaluates whether corroborating evidence exists to age an unaddressed objection into DORMANT.
    Per Client Principle 'Silence is not resolution', Δturns alone is a necessary pre-filter, but
    NEVER sufficient alone without one of 4 corroborating evidence types (checked in order):
    1. Supersession: o.superseded_by_objection_id is not None.
    2. Stage-based supersession: state.conversation_stage moved past the objection's domain.
    3. Behavioral resolution evidence: turns between o.last_updated_turn_id and current_turn_id
       exhibit positive forward signals correlated with resolution (e.g. rising agreement,
       future language, high readiness, or cooperative next steps).
    4. Blocker supersession: another objection or blocker became primary (newer entry with higher
       severity/recurrence, or dominant failed condition in conversion gate).

    Returns None if no corroborating evidence exists -> objection stays in current lifecycle state.
    """
    if objection is None:
        return None

    # 1. Supersession — o.superseded_by_objection_id is not None.
    # Strongest signal, immediate dormancy.
    if objection.superseded_by_objection_id:
        return DormancyEvidence(
            evidence_type="supersession",
            description=f"superseded by '{objection.superseded_by_objection_id}'",
            turn_id=objection.superseded_at_turn_id or current_turn_id,
            supporting_signals={"superseded_by_objection_id": objection.superseded_by_objection_id},
        )

    # 2. Stage-based supersession — check whether current stage has moved past the objection domain.
    # (e.g. general_hesitation belongs to discovery/qualification; if stage is now scheduling
    # or commitment_confirmed, that is structural evidence the conversation has moved on).
    # Note: Per Client Principle 'An appointment booking is not an objection resolution',
    # a partially_resolved objection (e.g. hesitation discussed in depth) is NOT superseded
    # simply by moving into scheduling or commitment_confirmed. The meeting was booked to address it!
    # Therefore, stage-based supersession only applies to unaddressed/active objections.
    obj_state_val = getattr(objection.lifecycle_state, "value", objection.lifecycle_state)
    if obj_state_val not in ("partially_resolved", "partially_addressed"):
        stage_val = getattr(state, "conversation_stage", None)
        if stage_val is not None:
            stage_str = stage_val.value if hasattr(stage_val, "value") else str(stage_val).lower()
            DISCOVERY_DOMAINS = {
                "general_hesitation",
                "decision_to_stay",
                "timing_market",
                "discovery",
                "qualification",
            }
            ADVANCED_STAGES = {
                "scheduling",
                "commitment_confirmed",
                "next_steps",
                "closed",
                "contract",
            }
            FEE_DOMAINS = {
                "commission_fee",
                "price",
                "representation_broker",
            }
            POST_NEGOTIATION_STAGES = {
                "commitment_confirmed",
                "next_steps",
                "closed",
                "contract",
            }

            cat = (objection.canonical_category or "").lower()
            if (cat in DISCOVERY_DOMAINS and stage_str in ADVANCED_STAGES) or \
               (cat in FEE_DOMAINS and stage_str in POST_NEGOTIATION_STAGES) or \
               (stage_str in ("commitment_confirmed", "closed")):
                return DormancyEvidence(
                    evidence_type="stage_transition",
                    description=f"superseded by stage transition into '{stage_str}'",
                    turn_id=current_turn_id,
                    supporting_signals={"stage": stage_str, "objection_category": objection.canonical_category},
                )

    # 3. Behavioral resolution evidence since last activity — look for signals correlated with
    # resolution of the objection or its driver_layer.underlying_driver in turns between
    # o.last_updated_turn_id and current_turn_id.
    relevant_bundles: List[BehavioralSignalInputBundle] = []
    if recent_bundles:
        relevant_bundles = [
            b for b in recent_bundles
            if objection.last_updated_turn_id < b.turn_id <= current_turn_id and b.speaker_id == "client"
        ]

    for b in relevant_bundles:
        # Check rising agreement / high agreement
        if b.agreement_score >= 0.65:
            return DormancyEvidence(
                evidence_type="behavioral_resolution",
                description=f"corroborated by prospect behavioral agreement (agreement_score {b.agreement_score:.2f} at turn {b.turn_id})",
                turn_id=b.turn_id,
                supporting_signals={"agreement_score": b.agreement_score, "turn_id": b.turn_id},
            )
        # Check strong future language (forward engagement)
        if b.future_language_score >= 0.50:
            return DormancyEvidence(
                evidence_type="behavioral_resolution",
                description=f"corroborated by forward future language (future_language_score {b.future_language_score:.2f} at turn {b.turn_id})",
                turn_id=b.turn_id,
                supporting_signals={"future_language_score": b.future_language_score, "turn_id": b.turn_id},
            )
        # Check high readiness on turn
        r_score = getattr(b.readiness, "score", b.readiness) if hasattr(b, "readiness") else 0.0
        if isinstance(r_score, (int, float)) and r_score >= 0.65:
            return DormancyEvidence(
                evidence_type="behavioral_resolution",
                description=f"corroborated by elevated prospect readiness (readiness {r_score:.2f} at turn {b.turn_id})",
                turn_id=b.turn_id,
                supporting_signals={"readiness": float(r_score), "turn_id": b.turn_id},
            )

    # Also inspect state-level readiness/momentum if significantly elevated
    if state is not None:
        readiness_val = getattr(state.dimensions, "readiness", 0.0) if state.dimensions else 0.0
        trust_val = getattr(state.dimensions, "trust", 0.0) if state.dimensions else 0.0
        if readiness_val >= 0.70 and trust_val >= 0.60:
            return DormancyEvidence(
                evidence_type="behavioral_resolution",
                description=f"corroborated by sustained high readiness ({readiness_val:.2f}) and trust ({trust_val:.2f})",
                turn_id=current_turn_id,
                supporting_signals={"readiness": readiness_val, "trust": trust_val},
            )

    # 4. Blocker supersession — another objection or blocker became primary (does not age partially_resolved concerns)
    if state is not None and obj_state_val not in ("partially_resolved", "partially_addressed"):
        # Check if a newer objection was raised that has active status and higher/equal recurrence or priority
        for other_o in getattr(state, "objections", []):
            if other_o.objection_id != objection.objection_id:
                if other_o.first_turn_id > objection.last_updated_turn_id:
                    other_state = getattr(other_o.lifecycle_state, "value", other_o.lifecycle_state)
                    if other_state in ("active", "partially_addressed", "unresolved", "reactivated"):
                        if other_o.recurrence_count >= objection.recurrence_count:
                            return DormancyEvidence(
                                evidence_type="blocker_supersession",
                                description=f"superseded by newer primary objection '{other_o.canonical_category}'",
                                turn_id=other_o.first_turn_id,
                                supporting_signals={
                                    "newer_objection_id": other_o.objection_id,
                                    "newer_category": other_o.canonical_category,
                                    "turn_id": other_o.first_turn_id,
                                },
                            )

        # Check whether an active blocking condition dominates conversion_gate (must be explicit 'not_met', not neutral 'unknown')
        if state.conversion_gate and state.conversion_gate.conditions:
            for cond in state.conversion_gate.conditions:
                if cond.status == "not_met" and cond.condition_name != "objections_resolved_or_partial":
                    if cond.condition_name in ("hard_boundary_clear", "decision_maker_confirmed", "no_active_boundary", "decision_maker_aligned"):
                        return DormancyEvidence(
                            evidence_type="blocker_supersession",
                            description=f"superseded by primary blocker condition '{cond.condition_name}'",
                            turn_id=current_turn_id,
                            supporting_signals={"condition_name": cond.condition_name, "reason": cond.reason},
                        )

    # No corroborating evidence -> returns None. Silence is not resolution.
    return None


get_dormancy_evidence = _get_dormancy_evidence

