from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple, Literal

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ObjectionRecord,
    ObjectionLifecycleState,
    StateChangeRecord,
)

LOGGER = logging.getLogger("copilot.conversation_objections")

CANONICAL_OBJECTION_PATTERNS: Dict[str, List[str]] = {
    "commission_fee": [
        r"\b(?:6|5|4|7)\s*(?:percent|%)\b",
        r"\bcommission(?:s)?\b",
        r"\b(?:your|the|listing|broker|agent)\s+fee(?:s)?\b",
        r"\bfee(?:s)?\s+(?:too\s+)?(?:high|steep|much|expensive)\b",
        r"\bcut\s+(?:your\s+)?commission\b",
        r"\bcost\s+to\s+list\b",
        r"\btoo\s+expensive\b",
    ],
    "market_timing": [
        r"\bwait\s+(?:until|for)\s+(?:spring|next\s+year|summer|the\s+market)\b",
        r"\bmarket\s+(?:is\s+bad|crash|dropping|slow)\b",
        r"\binterest\s+rates?\s+(?:are\s+)?too\s+high\b",
        r"\bbad\s+time\s+to\s+sell\b",
        r"\bnot\s+a\s+good\s+time\b",
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
    ],
    "general_hesitation": [
        r"\bnot\s+ready\s+(?:yet|to\s+sell|to\s+commit)\b",
        r"\bjust\s+(?:looking|browsing|curious)\b",
        r"\bneed\s+(?:more\s+)?time\s+to\s+think\b",
        r"\bthinking\s+it\s+over\b",
        r"\b(?:just\s+)?not\s+sure\b.*?\b(?:right\s+time|ready|good\s+time|now)\b",
        r"\b(?:just\s+)?not\s+sure\s+(?:this|if|about|whether|it['’]?s)\b",
        r"\b(?:just\s+)?not\s+(?:completely\s+)?sure\b",
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

    def __init__(self, initial_objections: Optional[List[ObjectionRecord]] = None):
        self._objections: List[ObjectionRecord] = list(initial_objections) if initial_objections else []
        self.pending_reframe_objection_id: Optional[str] = None
        self.pending_reframe_strategy: Optional[str] = None
        self.pending_reframe_turn_id: Optional[int] = None
        self.active_focus_objection_id: Optional[str] = None

    @property
    def objections(self) -> List[ObjectionRecord]:
        return list(self._objections)

    def get_objection_by_id(self, objection_id: str) -> Optional[ObjectionRecord]:
        return next((o for o in self._objections if o.objection_id == objection_id), None)

    def get_objection_by_recurrence_id(self, recurrence_id: str) -> Optional[ObjectionRecord]:
        return next((o for o in self._objections if o.recurrence_id == recurrence_id), None)

    def get_active_objection_by_category(self, category: str) -> Optional[ObjectionRecord]:
        for o in reversed(self._objections):
            if o.canonical_category == category and o.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "reactivated"):
                return o
        return None

    def get_active_objections(self) -> List[ObjectionRecord]:
        """Returns all currently active (unresolved, clarified, partially_resolved, reactivated) objections."""
        return [o for o in self._objections if o.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "reactivated")]

    def get_superseded_objections(self) -> List[ObjectionRecord]:
        """Returns all superseded objections."""
        return [o for o in self._objections if o.lifecycle_state == "superseded"]

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
                target_obj = next((o for o in reversed(self._objections) if o.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "reactivated")), None)

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
        detected_category = classify_objection_label(bundle.utterance_text)

        # Primary Identity Resolution: Anchor to Behavioral Signal Engine's recurrence_id
        matched_obj: Optional[ObjectionRecord] = None
        if bundle.recurrence_id:
            matched_obj = self.get_objection_by_recurrence_id(bundle.recurrence_id)

        # Secondary Identity Resolution: Match by detected category or active focus
        if not matched_obj and detected_category:
            matched_obj = self.get_active_objection_by_category(detected_category)

        # Handle explicit recurrence signal from Behavioral Signal Engine
        if not matched_obj and bundle.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe"):
            if self.active_focus_objection_id:
                matched_obj = self.get_objection_by_id(self.active_focus_objection_id)
            elif self._objections:
                matched_obj = next((o for o in reversed(self._objections) if o.lifecycle_state != "boundary"), None)

        if detected_category or matched_obj:
            category = detected_category or (matched_obj.canonical_category if matched_obj else "general_hesitation")

            if matched_obj:
                self.active_focus_objection_id = matched_obj.objection_id
                # Anchor recurrence_id if not yet bound
                if bundle.recurrence_id and not matched_obj.recurrence_id:
                    matched_obj.recurrence_id = bundle.recurrence_id

                # Client Principle #5: Previously RESOLVED objection returns -> REACTIVATED
                if matched_obj.lifecycle_state == "resolved":
                    old_state = matched_obj.lifecycle_state
                    matched_obj.lifecycle_state = "reactivated"
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
                            new_value="reactivated",
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Resolved objection '{matched_obj.canonical_category}' reactivated by prospect (recurrence #{matched_obj.recurrence_count}).",
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )
                else:
                    # Objection repeats while in UNRESOLVED, CLARIFIED, or PARTIALLY_RESOLVED
                    matched_obj.recurrence_count += 1
                    matched_obj.latest_statement = bundle.utterance_text
                    matched_obj.last_updated_turn_id = bundle.turn_id

                    # If concern repeats after a failed reframe, it remains/reverts to UNRESOLVED
                    if bundle.recurrence_type == "concern_after_failed_reframe" and matched_obj.lifecycle_state == "partially_resolved":
                        matched_obj.lifecycle_state = "unresolved"

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
            else:
                # Brand new objection raised -> UNRESOLVED
                new_obj = ObjectionRecord(
                    recurrence_id=bundle.recurrence_id,
                    canonical_category=category,
                    initial_statement=bundle.utterance_text,
                    latest_statement=bundle.utterance_text,
                    lifecycle_state="unresolved",
                    first_turn_id=bundle.turn_id,
                    last_updated_turn_id=bundle.turn_id,
                    recurrence_count=1,
                    confidence=bundle.semantic_confidence,
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

                # Client Principle #5 / Supersession: If this new objection is concrete
                # (spouse_authority, broker_representation, pricing_value), it supersedes
                # any active generic hesitation ("general_hesitation")!
                if category in ("spouse_authority", "broker_representation", "pricing_value"):
                    for active_o in list(self._objections):
                        if (
                            active_o.objection_id != new_obj.objection_id
                            and active_o.canonical_category == "general_hesitation"
                            and active_o.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "reactivated")
                        ):
                            old_s = active_o.lifecycle_state
                            active_o.lifecycle_state = "superseded"
                            active_o.superseded_by_objection_id = new_obj.objection_id
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
                                    reason=f"Generic hesitation superseded by concrete root objection '{category}'.",
                                    timestamp_ms=bundle.timestamp_ms,
                                )
                            )

            # Clear any pending reframe since prospect raised/repeated a concern
            self.pending_reframe_objection_id = None
            self.pending_reframe_strategy = None
            return self._objections, changes, next_version

        # Check for Decision to Stay / Cancellation of Sale that supersedes all sales/transactional objections
        is_stay_or_cancel = any(re.search(p, bundle.utterance_text.lower()) for p in DECISION_TO_STAY_PATTERNS)
        if is_stay_or_cancel:
            for active_o in list(self._objections):
                if active_o.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "reactivated"):
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

        if target_obj and target_obj.lifecycle_state in ("unresolved", "clarified", "partially_resolved", "reactivated"):
            # Check for FULL RESOLUTION (Scoped specifically to this targeted objection)
            # Requires forward behavioral commitment on this topic (agreement >= 0.75 or future_language >= 0.60)
            is_scoped_resolution = (
                bundle.agreement_score >= 0.75
                or bundle.future_language_score >= 0.60
                or (bundle.readiness.score >= 0.70 and bundle.agreement_score >= 0.65)
            ) and bundle.boundary_score < 0.20

            if is_scoped_resolution:
                old_state = target_obj.lifecycle_state
                target_obj.lifecycle_state = "resolved"
                target_obj.last_updated_turn_id = bundle.turn_id
                target_obj.resolution_evidence = (
                    f"Turn {bundle.turn_id} showed prospect behavioral advance responding to '{target_obj.canonical_category}' "
                    f"(agreement={bundle.agreement_score:.2f}, future_lang={bundle.future_language_score:.2f}, readiness={bundle.readiness.score:.2f})."
                )
                self.pending_reframe_objection_id = None
                self.pending_reframe_strategy = None
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path=f"objections.{target_obj.objection_id}.lifecycle_state",
                        old_value=old_state,
                        new_value="resolved",
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Objection '{target_obj.canonical_category}' resolved: {target_obj.resolution_evidence}",
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )
                return self._objections, changes, next_version

            # Check for PARTIAL RESOLUTION (Prospect Acceptance Required!)
            # Requires positive prospect acknowledgement (agreement >= 0.40) following an agent reframe,
            # but without full forward resolution commitment.
            is_partial_acceptance = (
                self.pending_reframe_objection_id == target_obj.objection_id
                and bundle.agreement_score >= 0.40
                and bundle.boundary_score < 0.20
            )

            if is_partial_acceptance:
                old_state = target_obj.lifecycle_state
                target_obj.lifecycle_state = "partially_resolved"
                target_obj.last_updated_turn_id = bundle.turn_id
                strategy_used = self.pending_reframe_strategy or "agent reframe"
                self.pending_reframe_objection_id = None
                self.pending_reframe_strategy = None
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path=f"objections.{target_obj.objection_id}.lifecycle_state",
                        old_value=old_state,
                        new_value="partially_resolved",
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

        # Clear pending reframe if prospect spoke without accepting
        if self.pending_reframe_objection_id:
            self.pending_reframe_objection_id = None
            self.pending_reframe_strategy = None

        return self._objections, changes, next_version
