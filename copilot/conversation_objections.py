from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple, Literal

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ObjectionRecord,
    ObjectionLifecycleState,
    StateChangeRecord,
    ObjectionDriverLayer,
    StrategyAttemptOutcome,
)
from .conversation_objection_driver import ObjectionDriverClassifier

LOGGER = logging.getLogger("copilot.conversation_objections")

DEFAULT_DORMANCY_TURN_WINDOW: int = 3

CANONICAL_OBJECTION_PATTERNS: Dict[str, List[str]] = {
    "commission_fee": [
        r"\b(?:6|5|4|7)\s*(?:percent\b|%)",
        r"\bcommission(?:s)?\b",
        r"\b(?:your|the|listing|broker|agent)\s+fee(?:s)?\b",
        r"\bfee(?:s)?\s+(?:is\s+|are\s+)?(?:too\s+)?(?:high|steep|much|expensive|crazy)\b",
        r"\b(?:goes\s+to|much\s+in|cut\s+into|portion\s+to|percentage\s+to)\s+fees?\b",
        r"\bcut\s+(?:your\s+)?commission\b",
        r"\bcost\s+to\s+list\b",
        r"\btoo\s+expensive\b",
        r"\bdeserves?\s+(?:that|so)\s+much\b",
        r"\bputting\s+a\s+sign\b",
    ],
    "market_timing": [
        r"\bmarket\s+timing\b",
        r"\b(?:market\s+)?timing\s+(?:is\s+|isn't\s+|is\s+not\s+)?(?:not\s+)?(?:right|good|off|bad)\b",
        r"\bwait\s+(?:until|for)\s+(?:spring|next\s+year|summer|the\s+market)\b",
        r"\bmarket\s+(?:is\s+bad|crash|dropping|slow)\b",
        r"\binterest\s+rates?\s+(?:are\s+)?too\s+high\b",
        r"\bbad\s+time\s+to\s+sell\b",
        r"\bnot\s+(?:a\s+good|the\s+right)\s+time\b",
        r"\bwait\s+and\s+see\b",
    ],
    "broker_representation": [
        r"\balready\s+(?:have|working\s+with)\s+an?\s+(?:agent|realtor|broker)\b",
        r"\bmy\s+(?:friend|relative|cousin|nephew|brother|sister)\s+is\s+a\s+(?:realtor|agent|broker)\b",
        r"\bsigned\s+(?:an?\s+)?(?:agreement|contract)\s+with\b",
        r"\bexclusive\s+(?:listing|agreement)\b",
    ],
    "pricing_value": [
        r"\bworth\s+more\s+than\s+that\b",
        r"\bprice\s+is\s+too\s+low\b",
        r"\bwon't\s+take\s+less\s+than\b",
        r"\blowball\b",
        r"\bneed\s+to\s+net\b",
    ],
    "spouse_authority": [
        r"\b(?:need\s+to|have\s+to|must|want\s+to)\s+talk\s+to\s+my\s+(?:wife|husband|spouse|partner)\b",
        r"\bmy\s+(?:wife|husband|spouse|partner)\s+(?:handles|decides|makes\s+the\s+decisions|isn't\s+on\s+board|wants|needs|is\s+hesitant|doesn't\s+want)\b",
        r"\bnot\s+my\s+decision\s+alone\b",
        r"\bdiscuss\s+(?:it\s+)?with\s+my\s+(?:wife|husband|spouse|partner)\b",
        r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\b.*?\b(?:part\s+of\s+(?:this|the)\s+conversation|loop(?:ed)?\s+in|needs?\s+to\s+be\s+(?:part|present|here|involved)|would\s+(?:really\s+)?need\s+to\s+be)\b",
        r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\b.*?\b(?:needs?|would\s+(?:really\s+)?need|has\s+to|must)\b.*?\b(?:conversation|decision|call|talk|meeting|input|further)\b",
        r"\b(?:need\s+to|have\s+to|must|want\s+to|should|would\s+need\s+to)\s+(?:talk|speak|discuss|check|consult)\s+(?:to|with)\s+my\s+(?:wife|husband|spouse|partner)\b",
        r"\b(?:wife|husband|spouse|partner)\b.*?\bbefore\s+(?:we|i)\s+(?:go|make|decide|move)\b",
        r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\s+(?:thinks|believes|feels|says|warned|worried|concerned)\b",
        r"\b(?:my\s+)?(?:wife|husband|spouse|partner)\s+wants\s+to\s+look\s+at\s+it\s+first\b",
    ],
    "trust_credibility": [
        r"\b(?:companies|people|folks|places|outfits)\s+like\s+yours\s+(?:are\s+)?(?:just\s+)?scams?\b",
        r"\b(?:companies|people|folks|places|outfits)\s+like\s+yours\b.*?\bscams?\b",
        r"\b(?:looks?|sounds?)\s+like\s+a\s+(?:total\s+)?scam\b",
        r"\b(?:is|are)\s+(?:just\s+)?(?:a\s+)?(?:total\s+)?scams?\b",
        r"\b(?:think|thinks|thought|believes?)\s+(?:it['’]?s|this\s+is|they['’]?re|companies\s+like\s+yours\s+are|outfits\s+like\s+yours\s+are)\s+(?:just\s+)?(?:a\s+)?scams?\b",
        r"\b(?:a\s+)?neighbor\s+paid\s+(?:a\s+)?(?:big\s+)?(?:upfront\s+)?fee\b",
        r"\b(?:upfront\s+fee|deposit)\b.*?\b(?:disappeared|vanished|ran\s+off|took\s+off|left)\b",
        r"\b(?:agent|broker|company)\s+(?:disappeared|vanished|ran\s+off|took\s+off)\s+(?:with\s+(?:the|our|their|my|an?)\s+(?:money|deposit|fee|cash)|after\s+(?:getting|taking)\s+paid|and\s+stole)\b",
        r"\b(?:took|paid)\s+(?:a\s+)?(?:big\s+)?(?:deposit|fee)\s+and\s+(?:vanished|disappeared)\b",
        r"\b(?:been|got)\s+burned\s+(?:before|by\s+an?\s+agent|in\s+the\s+past)\b",
        r"\b(?:don['’]?t|doesn['’]?t)\s+trust\s+(?:agents?|realtors?|brokers?|companies\s+like\s+yours)\b",
        r"\b(?:taken|got)\s+advantage\s+of\b",
        r"\bupfront\s+fee\b",
    ],
    "general_hesitation": [
        r"\bnot\s+ready\s+(?:yet|to\s+sell|to\s+commit|at\s+this\s+time|now|right\s+now)?\b",
        r"\bjust\s+(?:looking|browsing|curious)\b",
        r"\bneed\s+(?:more\s+)?time\s+to\s+think\b",
        r"\bthinking\s+it\s+over\b",
        r"\b(?:just\s+)?not\s+sure\b.*?\b(?:right\s+time|ready|good\s+time|now)\b",
        r"\b(?:just\s+)?not\s+sure\s+(?:this|if|about|whether|it['’]?s)\b",
        r"\b(?:just\s+)?not\s+(?:completely\s+)?sure\b",
        r"\b(?:don['’]?t\s+think|not\s+thinking)\s+(?:it['’]?s\s+|it\s+is\s+)?(?:the\s+)?(?:right|good)\s+time\b",
        r"\b(?:too\s+much\s+(?:stress|clutter|work|hassle|packing)|overwhelm(?:ed|ing)?)\b",
        r"\b(?:worried|concerned|nervous)\b.*?\b(?:right\s+move|good\s+idea|financially|sell(?:ing)?|mov(?:e|ing))\b",
        r"\bnot\s+(?:really\s+)?(?:the\s+)?right\s+move\b",
        r"\bworried\s+(?:this\s+)?(?:isn['’]?t|is\s+not)\b",
        r"\bworried\b.*?\bfinancially\b",
    ],
}

DECISION_TO_STAY_PATTERNS: List[str] = [
    r"\b(?:decided|chosen|opting)\s+(?:to\s+)?(?:just\s+)?(?:stay|remain|stay\s+put|keep)\b",
    r"\b(?:not\s+(?:going\s+to\s+|gonna\s+)?sell(?:ing)?|won['’]?t\s+be\s+selling)\b",
    r"\b(?:taking|pulling)\s+(?:it\s+)?off\s+(?:the\s+)?market\b",
    r"\b(?:stay|remain|staying)\s+in\s+(?:the|our)\s+(?:house|home|place)\b",
    r"\b(?:not\s+(?:moving|relocating)|staying\s+put)\b",
    r"\b(?:cancel(?:ling)?|call(?:ing)?\s+off)\s+(?:the\s+)?(?:sale|listing)\b",
]


def is_decision_authority_statement(utterance_text: str) -> bool:
    """Detects whether an utterance is a purely procedural decision-authority constraint (e.g. spouse, co-owner, legal authority),
    which must be routed to DecisionStructure rather than spawning an ObjectionRecord.
    """
    clean_text = utterance_text.lower().strip()
    if any(k in clean_text for k in ("scam", "fraud", "ripoff", "rip-off", "disappeared", "vanished", "upfront fee")):
        return False
    from .conversation_materiality import ABSENT_DECISION_MAKER_PATTERNS
    if any(re.search(pat, clean_text) for pat in ABSENT_DECISION_MAKER_PATTERNS):
        return True
    authority_patterns = [
        r"\b(?:my\s+)?(?:wife|husband|spouse|partner|brother-in-law|sister-in-law|co-owner|attorney)\b.*?\b(?:decide|decision|sign|board|involved|say|call|consult|talk|conversation)\b",
        r"\bnot\s+my\s+decision\s+alone\b",
        r"\bwe\s+decide\s+together\b",
        r"\bneeds?\s+to\s+be\s+(?:part|involved|present|here)\b",
    ]
    return any(re.search(pat, clean_text) for pat in authority_patterns)


def is_objection_concession_statement(utterance_text: str) -> bool:
    """Detects whether an utterance is an explicit concession, resolution, or withdrawal of an objection,
    which must not be misclassified as repeating or re-raising the objection.
    """
    clean = utterance_text.lower().strip()
    concession_patterns = [
        r"\b(?:don't|do\s+not|not)\s+(?:care|worried|concerned)\s+about\s+(?:the\s+)?(?:fee|commission|percentage|rate|price|timing)\b",
        r"\b(?:commission|fee|pricing|timing|market)\s+(?:is\s+fine|isn't\s+(?:really\s+)?(?:an?|the)\s+issue|doesn't\s+matter|is\s+fair|is\s+okay|makes\s+sense)\b",
        r"\b(?:fine|okay|happy)\s+with\s+(?:the\s+)?(?:commission|fee|rate|price|timing)\b",
        r"\bnever\s+mind\s+about\s+(?:the\s+)?(?:fee|commission)\b",
        r"\b(?:commission|fee)\s+isn't\s+(?:really\s+)?(?:an?|the)\s+issue\b",
        r"\btiming\s+isn't\s+(?:the\s+biggest\s+|an?\s+)?issue\b",
        r"\b(?:makes?\s+sense|fair\s+enough|okay\s+that\s+makes\s+sense)\b.*?\b(?:commission|timing|fee|pricing|rate|not\s+an\s+issue|isn't\s+the\s+biggest\s+issue|not\s+really\s+the\s+issue|isn't\s+really\s+the\s+issue)\b",
        r"\bcommission\s+is\s+fair\b",
        r"\bnot\s+a\s+problem\s+anymore\b",
    ]
    return any(re.search(p, clean) for p in concession_patterns)


def classify_objection_label(utterance_text: str) -> Optional[str]:
    """Classifies the semantic category label for an objection utterance.

    Note: This provides a descriptive category label. Objection identity itself
    is anchored directly to the Behavioral Signal Engine's recurrence_id.
    """
    clean_text = utterance_text.lower().strip()
    for category, patterns in CANONICAL_OBJECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, clean_text, re.IGNORECASE):
                return category
    return None


class ObjectionLifecycleEngine:
    """Manages the 6-state lifecycle for tracked conversation objections.

    Key Architectural Principles:
    1. Identity Anchoring: Objection identity is anchored directly to the Behavioral Signal Engine's
       recurrence_id and recurrence_type signals, preventing drift between independent systems.
    2. Prospect Acceptance Required for PARTIALLY_RESOLVED: A salesperson attempting a reframe logs the
       strategy and sets a pending reframe state, but the transition to PARTIALLY_RESOLVED requires
       prospect-side evidence of partial acceptance (agreement >= 0.40 without full advance).
    3. Objection-Scoped Resolution: RESOLVED requires forward behavioral advance specifically addressing
       the targeted objection, not global call-wide scalars that would accidentally resolve unrelated concerns.
    4. Speaker-Filtered Clarification: CLARIFIED only triggers on Agent turns with question_type == 'clarifying'.
    5. Identity Preservation on Reactivation (Client Principle #5): Resolved objections that return retain
       their stable objection_id, increment recurrence_count, and retain their full attempted strategy history.
    """

    def __init__(
        self,
        initial_objections: Optional[List[ObjectionRecord]] = None,
        dormancy_turn_threshold: int = DEFAULT_DORMANCY_TURN_WINDOW,
    ):
        self._objections: List[ObjectionRecord] = list(initial_objections) if initial_objections else []
        self.dormancy_turn_threshold: int = dormancy_turn_threshold
        self.pending_reframe_objection_id: Optional[str] = None
        self.pending_reframe_strategy: Optional[str] = None
        self.pending_reframe_turn_id: Optional[int] = None
        self.active_focus_objection_id: Optional[str] = None
        self.driver_classifier = ObjectionDriverClassifier()

    @property
    def objections(self) -> List[ObjectionRecord]:
        return list(self._objections)

    def get_objection_by_id(self, objection_id: str) -> Optional[ObjectionRecord]:
        return next((o for o in self._objections if o.objection_id == objection_id), None)

    def get_objection_by_recurrence_id(self, recurrence_id: str) -> Optional[ObjectionRecord]:
        return next((o for o in self._objections if o.recurrence_id == recurrence_id), None)

    def get_active_objection_by_category(self, category: str) -> Optional[ObjectionRecord]:
        for o in reversed(self._objections):
            if o.canonical_category == category and o.lifecycle_state in (
                ObjectionLifecycleState.ACTIVE,
                ObjectionLifecycleState.PARTIALLY_ADDRESSED,
                "active",
                "partially_addressed",
                "unresolved",
                "clarified",
                "partially_resolved",
                "reactivated",
            ):
                return o
        return None

    def get_active_objections(self) -> List[ObjectionRecord]:
        """Returns all currently active (active, partially_addressed, unresolved, clarified, partially_resolved, reactivated) objections."""
        return [
            o for o in self._objections
            if o.lifecycle_state in (
                ObjectionLifecycleState.ACTIVE,
                ObjectionLifecycleState.PARTIALLY_ADDRESSED,
                "active",
                "partially_addressed",
                "unresolved",
                "clarified",
                "partially_resolved",
                "reactivated",
            )
        ]

    def get_dormant_objections(self) -> List[ObjectionRecord]:
        """Returns all dormant objections."""
        return [o for o in self._objections if o.lifecycle_state in (ObjectionLifecycleState.DORMANT, "dormant")]

    def get_superseded_objections(self) -> List[ObjectionRecord]:
        """Returns all superseded objections."""
        return [o for o in self._objections if o.lifecycle_state in (ObjectionLifecycleState.SUPERSEDED, "superseded")]

    def _record_strategy_outcome(
        self,
        target_obj: ObjectionRecord,
        bundle: BehavioralSignalInputBundle,
        effectiveness: Literal["effective", "partial", "insufficient", "rejected", "no_response"],
        summary: str,
        next_version: int,
        changes: List[StateChangeRecord],
    ) -> int:
        """Client Feedback No. 6: Records structured strategy effectiveness outcome on the target objection."""
        if not self.pending_reframe_strategy:
            return next_version

        outcome = StrategyAttemptOutcome(
            strategy_tag=self.pending_reframe_strategy,
            attempted_at_turn_id=self.pending_reframe_turn_id or (bundle.turn_id - 1),
            prospect_response_turn_id=bundle.turn_id,
            prospect_response_summary=summary,
            effectiveness=effectiveness,
            evidence={
                "agreement_score": round(bundle.agreement_score, 3),
                "specificity_score": round(bundle.specificity_score, 3),
                "future_language_score": round(bundle.future_language_score, 3),
                "emotion_tension": round(bundle.emotion.tension_level, 3),
            },
            timestamp_ms=bundle.timestamp_ms,
        )
        target_obj.strategy_outcomes.append(outcome)
        next_version += 1
        changes.append(
            StateChangeRecord(
                state_version_before=next_version - 1,
                state_version_after=next_version,
                field_path=f"objections.{target_obj.objection_id}.strategy_outcomes",
                old_value=[o.model_dump() for o in target_obj.strategy_outcomes[:-1]],
                new_value=[o.model_dump() for o in target_obj.strategy_outcomes],
                triggering_turn_id=bundle.turn_id,
                evidence_ids=bundle.contributing_evidence_ids,
                reason=f"Strategy '{outcome.strategy_tag}' outcome on '{target_obj.canonical_category}': {outcome.effectiveness.upper()} - {summary}",
                timestamp_ms=bundle.timestamp_ms,
            )
        )
        self.pending_reframe_objection_id = None
        self.pending_reframe_strategy = None
        self.pending_reframe_turn_id = None
        return next_version

    def supersede_objection(
        self,
        old_objection_id: Optional[str] = None,
        superseded_by_id: Optional[str] = None,
        turn_id: int = 0,
        reason: str = "Objection superseded by newer evidence",
        current_version: int = 1,
        evidence_ids: Optional[List[str]] = None,
        timestamp_ms: int = 0,
        *,
        objection_id: Optional[str] = None,
        superseded_by_objection_id: Optional[str] = None,
        superseded_at_turn_id: Optional[int] = None,
    ) -> Tuple[Optional[ObjectionRecord], Optional[StateChangeRecord], int]:
        """Explicitly supersedes an active objection, linking lineage and recording state change."""
        target_id = objection_id or old_objection_id
        sup_by = superseded_by_objection_id or superseded_by_id
        effective_turn = superseded_at_turn_id if superseded_at_turn_id is not None else turn_id

        obj = self.get_objection_by_id(target_id) if target_id else None
        if not obj or obj.lifecycle_state in ("resolved", "superseded", "boundary"):
            return None, None, current_version

        old_state = obj.lifecycle_state
        obj.lifecycle_state = "superseded"
        obj.superseded_by_objection_id = sup_by
        obj.superseded_at_turn_id = effective_turn
        obj.last_updated_turn_id = effective_turn

        next_version = current_version + 1
        change = StateChangeRecord(
            state_version_before=current_version,
            state_version_after=next_version,
            field_path=f"objections.{obj.objection_id}.lifecycle_state",
            old_value=old_state,
            new_value="superseded",
            triggering_turn_id=turn_id,
            evidence_ids=evidence_ids or [],
            reason=reason,
            timestamp_ms=timestamp_ms,
        )
        return obj, change, next_version

    def evaluate_turn(
        self,
        bundle: BehavioralSignalInputBundle,
        current_version: int,
    ) -> Tuple[List[ObjectionRecord], List[StateChangeRecord], int]:
        """Evaluates a turn bundle against the objection registry, returning updated records,

        explainability change records, and next state version.
        """
        changes: List[StateChangeRecord] = []
        next_version = current_version

        # -------------------------------------------------------------------------
        # 1. Hard Compliance Boundary Override (Immediate cutoff across all active objections)
        # -------------------------------------------------------------------------
        if bundle.boundary_score >= 0.85 or bundle.recurrence_type == "boundary_repeated":
            for obj in self._objections:
                if obj.lifecycle_state not in ("boundary", "resolved", "superseded"):
                    old_state = obj.lifecycle_state
                    obj.lifecycle_state = "boundary"
                    obj.last_updated_turn_id = bundle.turn_id
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{obj.objection_id}.lifecycle_state",
                            old_value=old_state,
                            new_value="boundary",
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason="Hard boundary detected; transitioning active objection to compliance cutoff.",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )
            return self._objections, changes, next_version

        # -------------------------------------------------------------------------
        # 2. Salesperson Turn: Clarification and Reframe Strategy Logging
        # -------------------------------------------------------------------------
        if bundle.speaker_id == "salesperson":
            # Check if salesperson explicitly addresses a specific objection category
            agent_target_cat = classify_objection_label(bundle.utterance_text)
            target_obj = None
            if agent_target_cat:
                target_obj = self.get_active_objection_by_category(agent_target_cat)
            if not target_obj and self.active_focus_objection_id:
                target_obj = self.get_objection_by_id(self.active_focus_objection_id)
            if not target_obj:
                target_obj = next((o for o in reversed(self._objections) if o.lifecycle_state in ("active", "unresolved", "clarified", "partially_resolved", "reactivated", ObjectionLifecycleState.ACTIVE)), None)

            if target_obj:
                self.active_focus_objection_id = target_obj.objection_id

            # Strategy Attempted by Agent: Record strategy, but DO NOT prematurely mark partially_resolved!
            if bundle.salesperson_strategy_tag and target_obj:
                if bundle.salesperson_strategy_tag not in target_obj.attempted_strategies:
                    target_obj.attempted_strategies.append(bundle.salesperson_strategy_tag)
                
                self.pending_reframe_objection_id = target_obj.objection_id
                self.pending_reframe_strategy = bundle.salesperson_strategy_tag
                self.pending_reframe_turn_id = bundle.turn_id
                target_obj.last_updated_turn_id = bundle.turn_id

                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path=f"objections.{target_obj.objection_id}.attempted_strategies",
                        old_value=[s for s in target_obj.attempted_strategies if s != bundle.salesperson_strategy_tag],
                        new_value=list(target_obj.attempted_strategies),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=(
                            f"Agent attempted reframe strategy '{bundle.salesperson_strategy_tag}' "
                            f"({bundle.salesperson_strategy_source}) on objection '{target_obj.canonical_category}'. "
                            f"Awaiting prospect acceptance before changing lifecycle state."
                        ),
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )

            # Clarifying Question Asked by Agent (Strictly speaker-filtered)
            if bundle.question_type == "clarifying" and target_obj:
                if target_obj.lifecycle_state == "unresolved":
                    target_obj.lifecycle_state = "clarified"
                    target_obj.last_updated_turn_id = bundle.turn_id
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{target_obj.objection_id}.lifecycle_state",
                            old_value="unresolved",
                            new_value="clarified",
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason="Agent asked clarifying question to define objection scope.",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )

            return self._objections, changes, next_version

        # -------------------------------------------------------------------------
        # 3. Client Turn: Evaluate Objection Expression, Reframe Reaction, and Resolution
        # -------------------------------------------------------------------------
        is_concession = is_objection_concession_statement(bundle.utterance_text)
        detected_category = None if is_concession else classify_objection_label(bundle.utterance_text)

        # Primary Identity Resolution: Anchor to Behavioral Signal Engine's recurrence_id
        matched_obj: Optional[ObjectionRecord] = None
        if not is_concession:
            if bundle.recurrence_id:
                matched_obj = self.get_objection_by_recurrence_id(bundle.recurrence_id)

            # Secondary Identity Resolution: Match by detected category or active focus
            if not matched_obj and detected_category:
                matched_obj = self.get_active_objection_by_category(detected_category)
                if not matched_obj:
                    matched_obj = next(
                        (o for o in reversed(self._objections) if o.canonical_category == detected_category and o.lifecycle_state in (ObjectionLifecycleState.DORMANT, "dormant", "resolved")),
                        None,
                    )

            # Handle explicit recurrence signal from Behavioral Signal Engine
            if not matched_obj and bundle.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe"):
                if self.active_focus_objection_id:
                    matched_obj = self.get_objection_by_id(self.active_focus_objection_id)
                elif self._objections:
                    matched_obj = next((o for o in reversed(self._objections) if o.lifecycle_state != "boundary"), None)

        # Decision-Authority Ingestion Check:
        # Statements expressing decision-authority constraints (spouse, co-owner, signing authority)
        # belong in DecisionStructure and PersistentFactRecord, NOT in ObjectionRecord!
        is_authority_constraint = is_decision_authority_statement(bundle.utterance_text)
        if is_authority_constraint and detected_category == "spouse_authority" and not bundle.recurrence_id:
            detected_category = None

        if detected_category or matched_obj:
            category = detected_category or (matched_obj.canonical_category if matched_obj else "general_hesitation")

            if matched_obj:
                self.active_focus_objection_id = matched_obj.objection_id
                # Anchor recurrence_id if not yet bound
                if bundle.recurrence_id and not matched_obj.recurrence_id:
                    matched_obj.recurrence_id = bundle.recurrence_id

                # Evidence-Gated Objection Supersession / Reversal check:
                from .conversation_supersession import TruthSupersessionDetector
                detector = TruthSupersessionDetector()
                dec = detector.evaluate_objection_supersession(bundle.utterance_text, matched_obj)
                if dec.has_supersession and dec.relation == "REVERSES":
                    old_s = matched_obj.lifecycle_state
                    matched_obj.lifecycle_state = ObjectionLifecycleState.SUPERSEDED
                    matched_obj.superseded_by_objection_id = f"turn_{bundle.turn_id}"
                    matched_obj.superseded_at_turn_id = bundle.turn_id
                    matched_obj.last_updated_turn_id = bundle.turn_id
                    matched_obj.resolution_evidence = dec.new_truth_value or dec.reasoning
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{matched_obj.objection_id}.lifecycle_state",
                            old_value=old_s,
                            new_value="superseded",
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Objection '{matched_obj.canonical_category}' superseded with evidence (REVERSES): {dec.reasoning}",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )
                    self.pending_reframe_objection_id = None
                    self.pending_reframe_strategy = None
                    return self._objections, changes, next_version

                # Dormant concern resurfacing -> Reactivated to ACTIVE
                if matched_obj.lifecycle_state in (ObjectionLifecycleState.DORMANT, "dormant"):
                    old_state = getattr(matched_obj.lifecycle_state, "value", matched_obj.lifecycle_state)
                    matched_obj.lifecycle_state = ObjectionLifecycleState.ACTIVE
                    matched_obj.latest_statement = bundle.utterance_text
                    matched_obj.recurrence_count += 1
                    matched_obj.last_updated_turn_id = bundle.turn_id
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{matched_obj.objection_id}.lifecycle_state",
                            old_value=old_state,
                            new_value=ObjectionLifecycleState.ACTIVE.value,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Dormant objection '{matched_obj.canonical_category}' reactivated to active by prospect (recurrence #{matched_obj.recurrence_count}).",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )
                # Client Principle #5: Previously RESOLVED objection returns -> REACTIVATED
                elif matched_obj.lifecycle_state in (ObjectionLifecycleState.RESOLVED, "resolved"):
                    old_state = getattr(matched_obj.lifecycle_state, "value", matched_obj.lifecycle_state)
                    matched_obj.lifecycle_state = ObjectionLifecycleState.REACTIVATED
                    matched_obj.latest_statement = bundle.utterance_text
                    matched_obj.recurrence_count += 1
                    matched_obj.last_updated_turn_id = bundle.turn_id
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{matched_obj.objection_id}.lifecycle_state",
                            old_value=old_state,
                            new_value=ObjectionLifecycleState.REACTIVATED.value,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Resolved objection '{matched_obj.canonical_category}' reactivated by prospect (recurrence #{matched_obj.recurrence_count}).",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )
                else:
                    # Objection repeats while in ACTIVE, PARTIALLY_ADDRESSED
                    matched_obj.recurrence_count += 1
                    matched_obj.latest_statement = bundle.utterance_text
                    matched_obj.last_updated_turn_id = bundle.turn_id

                    # If concern repeats after a failed reframe, it remains/reverts to ACTIVE
                    if bundle.recurrence_type == "concern_after_failed_reframe" and matched_obj.lifecycle_state == "partially_resolved":
                        matched_obj.lifecycle_state = ObjectionLifecycleState.ACTIVE

                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{matched_obj.objection_id}.recurrence_count",
                            old_value=matched_obj.recurrence_count - 1,
                            new_value=matched_obj.recurrence_count,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Objection '{matched_obj.canonical_category}' repeated by prospect.",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )

                # Client Feedback No. 5: Update driver layer if fresh context available
                fresh_driver = self.driver_classifier.classify_driver(
                    canonical_category=matched_obj.canonical_category,
                    utterance_text=bundle.utterance_text,
                )
                if fresh_driver and (not matched_obj.driver_layer or matched_obj.driver_layer.underlying_driver != fresh_driver.underlying_driver):
                    old_d = matched_obj.driver_layer.model_dump() if matched_obj.driver_layer else None
                    matched_obj.driver_layer = fresh_driver
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{matched_obj.objection_id}.driver_layer",
                            old_value=old_d,
                            new_value=fresh_driver.model_dump(),
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Objection '{matched_obj.canonical_category}' driver refined to '{fresh_driver.underlying_driver}': {fresh_driver.strategic_target}",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )
            else:
                # Brand new objection raised -> ACTIVE
                new_driver = self.driver_classifier.classify_driver(
                    canonical_category=category,
                    utterance_text=bundle.utterance_text,
                )
                new_obj = ObjectionRecord(
                    recurrence_id=bundle.recurrence_id,
                    canonical_category=category,
                    initial_statement=bundle.utterance_text,
                    latest_statement=bundle.utterance_text,
                    lifecycle_state=ObjectionLifecycleState.ACTIVE,
                    first_turn_id=bundle.turn_id,
                    last_updated_turn_id=bundle.turn_id,
                    recurrence_count=1,
                    confidence=min(0.90, bundle.semantic_confidence),
                    driver_layer=new_driver,
                )
                self._objections.append(new_obj)
                self.active_focus_objection_id = new_obj.objection_id
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path=f"objections.{new_obj.objection_id}",
                        old_value=None,
                        new_value=new_obj.model_dump(),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"New objection registered in canonical category '{category}'.",
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )

            # Evidence-Gated Objection Supersession:
            # Route against active objections using TruthSupersessionDetector to verify if the new utterance
            # specifically REVERSES or UPDATES an existing concern. Unrelated concerns remain independent!
            from .conversation_supersession import TruthSupersessionDetector
            detector = TruthSupersessionDetector()
            for active_o in list(self._objections):
                if active_o.lifecycle_state in (
                    ObjectionLifecycleState.ACTIVE,
                    ObjectionLifecycleState.PARTIALLY_ADDRESSED,
                    "active",
                    "partially_addressed",
                    "unresolved",
                    "clarified",
                    "partially_resolved",
                    "reactivated",
                ):
                    if matched_obj and active_o.objection_id == matched_obj.objection_id:
                        continue
                    dec = detector.evaluate_objection_supersession(bundle.utterance_text, active_o)
                    if dec.has_supersession and dec.relation in ("REVERSES", "UPDATES"):
                        old_s = active_o.lifecycle_state
                        active_o.lifecycle_state = ObjectionLifecycleState.SUPERSEDED
                        superseding_id = new_obj.objection_id if 'new_obj' in locals() and new_obj else f"turn_{bundle.turn_id}"
                        active_o.superseded_by_objection_id = superseding_id
                        active_o.superseded_at_turn_id = bundle.turn_id
                        active_o.last_updated_turn_id = bundle.turn_id
                        active_o.resolution_evidence = dec.new_truth_value or dec.reasoning
                        next_version += 1
                        changes.append(
                            StateChangeRecord(
                                state_version_before=next_version - 1,
                                state_version_after=next_version,
                                field_path=f"objections.{active_o.objection_id}.lifecycle_state",
                                old_value=old_s,
                                new_value="superseded",
                                triggering_turn_id=bundle.turn_id,
                                evidence_ids=bundle.contributing_evidence_ids,
                                reason=f"Objection '{active_o.canonical_category}' superseded with evidence ({dec.relation}): {dec.reasoning}",
                                timestamp_ms=bundle.timestamp_ms,
                            )
                        )

            # If a reframe was pending and prospect reiterated/raised an objection, record as rejected
            if self.pending_reframe_strategy:
                reframe_target = self.get_objection_by_id(self.pending_reframe_objection_id) if self.pending_reframe_objection_id else (matched_obj or (new_obj if 'new_obj' in locals() else None))
                if reframe_target:
                    next_version = self._record_strategy_outcome(
                        target_obj=reframe_target,
                        bundle=bundle,
                        effectiveness="rejected",
                        summary=f"Prospect rejected reframe '{self.pending_reframe_strategy}' and reiterated concern (agreement={bundle.agreement_score:.2f})",
                        next_version=next_version,
                        changes=changes,
                    )
            self.pending_reframe_objection_id = None
            self.pending_reframe_strategy = None
            self.pending_reframe_turn_id = None
            return self._objections, changes, next_version

        # Check for Evidence-Gated Objection Supersession / Reversal on client turns even without new objection
        if bundle.speaker_id == "client":
            from .conversation_supersession import TruthSupersessionDetector
            detector = TruthSupersessionDetector()
            for active_o in list(self._objections):
                if active_o.lifecycle_state in (
                    ObjectionLifecycleState.ACTIVE,
                    ObjectionLifecycleState.PARTIALLY_ADDRESSED,
                    ObjectionLifecycleState.DORMANT,
                    "active",
                    "partially_addressed",
                    "dormant",
                    "unresolved",
                    "clarified",
                    "partially_resolved",
                    "reactivated",
                ):
                    dec = detector.evaluate_objection_supersession(bundle.utterance_text, active_o)
                    if dec.has_supersession and dec.relation in ("REVERSES", "UPDATES"):
                        old_s = active_o.lifecycle_state
                        active_o.lifecycle_state = ObjectionLifecycleState.SUPERSEDED
                        active_o.superseded_by_objection_id = f"turn_{bundle.turn_id}"
                        active_o.superseded_at_turn_id = bundle.turn_id
                        active_o.last_updated_turn_id = bundle.turn_id
                        active_o.resolution_evidence = dec.new_truth_value or dec.reasoning
                        next_version += 1
                        changes.append(
                            StateChangeRecord(
                                state_version_before=next_version - 1,
                                state_version_after=next_version,
                                field_path=f"objections.{active_o.objection_id}.lifecycle_state",
                                old_value=old_s,
                                new_value="superseded",
                                triggering_turn_id=bundle.turn_id,
                                evidence_ids=bundle.contributing_evidence_ids,
                                reason=f"Objection '{active_o.canonical_category}' superseded with evidence ({dec.relation}): {dec.reasoning}",
                                timestamp_ms=bundle.timestamp_ms,
                            )
                        )

        # Check for Decision to Stay / Cancellation of Sale that supersedes all sales/transactional objections
        is_stay_or_cancel = any(re.search(p, bundle.utterance_text.lower()) for p in DECISION_TO_STAY_PATTERNS)
        if is_stay_or_cancel:
            for active_o in list(self._objections):
                if active_o.lifecycle_state in ("active", "unresolved", "clarified", "partially_resolved", "reactivated", ObjectionLifecycleState.ACTIVE):
                    old_s = active_o.lifecycle_state
                    active_o.lifecycle_state = "superseded"
                    active_o.superseded_by_objection_id = "decision_to_stay"
                    active_o.superseded_at_turn_id = bundle.turn_id
                    active_o.last_updated_turn_id = bundle.turn_id
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"objections.{active_o.objection_id}.lifecycle_state",
                            old_value=old_s,
                            new_value="superseded",
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Prospect decided not to sell/move; objection '{active_o.canonical_category}' rendered moot and superseded.",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )

        # -------------------------------------------------------------------------
        # 4. Prospect Response to Pending Reframe / Targeted Objection
        # -------------------------------------------------------------------------
        target_obj = (
            self.get_objection_by_id(self.pending_reframe_objection_id)
            if self.pending_reframe_objection_id
            else self.get_objection_by_id(self.active_focus_objection_id)
        )

        if target_obj and target_obj.lifecycle_state in ("active", "unresolved", "clarified", "partially_resolved", "reactivated", ObjectionLifecycleState.ACTIVE):
            # Check for FULL RESOLUTION (Scoped specifically to this targeted objection)
            # Requires forward behavioral commitment on this topic (agreement >= 0.75 or future_language >= 0.60)
            # OR explicit concession/reversal by prospect
            concession_detected = is_concession or bool(re.search(
                r"\b(?:timing|the\s+timing|fee|commission)\s+(?:isn't|is\s+not)\s+(?:the\s+biggest\s+|an?\s+)?issue\b",
                bundle.utterance_text.lower(),
            )) or bool(re.search(
                r"\b(?:that\s+makes\s+sense|fair\s+enough|okay\s+that\s+makes\s+sense)\b.*?\b(?:timing|market|fee|not\s+an\s+issue|isn't\s+the\s+biggest\s+issue)\b",
                bundle.utterance_text.lower(),
            ))

            is_scoped_resolution = (
                bundle.agreement_score >= 0.75
                or bundle.future_language_score >= 0.60
                or (bundle.readiness.score >= 0.70 and bundle.agreement_score >= 0.65)
                or concession_detected
            ) and bundle.boundary_score < 0.20

            if is_scoped_resolution:
                old_state = getattr(target_obj.lifecycle_state, "value", target_obj.lifecycle_state)
                target_obj.lifecycle_state = ObjectionLifecycleState.RESOLVED
                target_obj.last_updated_turn_id = bundle.turn_id
                target_obj.resolution_evidence = (
                    f"Turn {bundle.turn_id} showed prospect behavioral advance responding to '{target_obj.canonical_category}' "
                    f"(agreement={bundle.agreement_score:.2f}, future_lang={bundle.future_language_score:.2f}, readiness={bundle.readiness.score:.2f})."
                )
                if self.pending_reframe_strategy:
                    next_version = self._record_strategy_outcome(
                        target_obj=target_obj,
                        bundle=bundle,
                        effectiveness="effective",
                        summary=f"Objection resolved following reframe '{self.pending_reframe_strategy}': {target_obj.resolution_evidence}",
                        next_version=next_version,
                        changes=changes,
                    )
                self.pending_reframe_objection_id = None
                self.pending_reframe_strategy = None
                self.pending_reframe_turn_id = None
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path=f"objections.{target_obj.objection_id}.lifecycle_state",
                        old_value=old_state,
                        new_value=ObjectionLifecycleState.RESOLVED.value,
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Objection '{target_obj.canonical_category}' resolved: {target_obj.resolution_evidence}",
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )
                return self._objections, changes, next_version

            # Check for PARTIAL RESOLUTION (Prospect Acceptance Required!)
            # Requires positive prospect acknowledgement (agreement >= 0.40) following an agent reframe,
            # or process curiosity/openness, but without full forward resolution commitment.
            is_partial_acceptance = (
                self.pending_reframe_objection_id == target_obj.objection_id
                and (
                    bundle.agreement_score >= 0.40
                    or any(w in bundle.utterance_text.lower() for w in ["really helpful", "that's helpful", "tell me more", "how does the", "makes sense"])
                )
                and bundle.boundary_score < 0.20
            )

            if is_partial_acceptance:
                old_state = getattr(target_obj.lifecycle_state, "value", target_obj.lifecycle_state)
                target_obj.lifecycle_state = ObjectionLifecycleState.PARTIALLY_RESOLVED
                target_obj.last_updated_turn_id = bundle.turn_id
                strategy_used = self.pending_reframe_strategy or "agent reframe"
                if self.pending_reframe_strategy:
                    next_version = self._record_strategy_outcome(
                        target_obj=target_obj,
                        bundle=bundle,
                        effectiveness="partial",
                        summary=f"Partial acceptance (agreement={bundle.agreement_score:.2f}) to reframe '{strategy_used}', but concern persists",
                        next_version=next_version,
                        changes=changes,
                    )
                self.pending_reframe_objection_id = None
                self.pending_reframe_strategy = None
                self.pending_reframe_turn_id = None
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path=f"objections.{target_obj.objection_id}.lifecycle_state",
                        old_value=old_state,
                        new_value=ObjectionLifecycleState.PARTIALLY_RESOLVED.value,
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=(
                            f"Prospect showed partial acceptance (agreement={bundle.agreement_score:.2f}) "
                            f"to reframe '{strategy_used}', but same issue remains."
                        ),
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )
                return self._objections, changes, next_version

        # Record outcome if strategy was pending but prospect did not accept
        if self.pending_reframe_strategy:
            pending_obj = self.get_objection_by_id(self.pending_reframe_objection_id) if self.pending_reframe_objection_id else target_obj
            if pending_obj:
                eff = "insufficient" if bundle.agreement_score < 0.40 else "no_response"
                next_version = self._record_strategy_outcome(
                    target_obj=pending_obj,
                    bundle=bundle,
                    effectiveness=eff,
                    summary=f"Reframe '{self.pending_reframe_strategy}' resulted in {eff} response (agreement={bundle.agreement_score:.2f})",
                    next_version=next_version,
                    changes=changes,
                )
            self.pending_reframe_objection_id = None
            self.pending_reframe_strategy = None
            self.pending_reframe_turn_id = None

        # -------------------------------------------------------------------------
        # 5. DORMANT Aging Evaluation
        # Active or partially addressed concerns unmentioned for >= 3 turns move to DORMANT
        # -------------------------------------------------------------------------
        updated_objs, dormant_changes, next_version = self.check_dormancy_aging(
            current_turn_id=bundle.turn_id,
            timestamp_ms=bundle.timestamp_ms,
            current_version=next_version,
            contributing_evidence_ids=bundle.contributing_evidence_ids,
        )
        changes.extend(dormant_changes)
        return self._objections, changes, next_version

    def check_dormancy_aging(
        self,
        current_turn_id: int,
        timestamp_ms: int,
        current_version: int,
        contributing_evidence_ids: List[str],
        state: Optional[Any] = None,
        recent_bundles: Optional[List[Any]] = None,
    ) -> tuple[List[ObjectionRecord], List[StateChangeRecord], int]:
        """Ages active or partially addressed concerns unmentioned for >= 3 turns into DORMANT
        ONLY IF corroborating evidence exists (supersession, stage transition, behavioral resolution,
        or blocker supersession). Silence alone is not resolution.
        """
        from .conversation_materiality import _get_dormancy_evidence
        changes: List[StateChangeRecord] = []
        next_version = current_version
        for o in self._objections:
            if o.lifecycle_state in (
                ObjectionLifecycleState.ACTIVE,
                ObjectionLifecycleState.PARTIALLY_ADDRESSED,
                "active",
                "partially_addressed",
                "unresolved",
                "clarified",
                "partially_resolved",
                "reactivated",
            ):
                delta = current_turn_id - o.last_updated_turn_id
                # Pre-filter: delta >= threshold (unless superseded)
                if delta < self.dormancy_turn_threshold and not getattr(o, "superseded_by_objection_id", None):
                    continue

                evidence = _get_dormancy_evidence(
                    objection=o,
                    state=state,
                    current_turn_id=current_turn_id,
                    recent_bundles=recent_bundles,
                )
                if evidence is None:
                    # Silence is not resolution: stays in current lifecycle state
                    continue

                old_s = getattr(o.lifecycle_state, "value", o.lifecycle_state)
                o.lifecycle_state = ObjectionLifecycleState.DORMANT
                o.dormancy_evidence = evidence
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path=f"objections.{o.objection_id}.lifecycle_state",
                        old_value=old_s,
                        new_value=ObjectionLifecycleState.DORMANT.value,
                        triggering_turn_id=current_turn_id,
                        evidence_ids=contributing_evidence_ids,
                        reason=f"Concern '{o.canonical_category}' transitioned to DORMANT: {evidence.description} ({delta} turns since last mention).",
                        timestamp_ms=timestamp_ms,
                    )
                )
        return self._objections, changes, next_version
