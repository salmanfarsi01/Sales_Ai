from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStructure,
    DimensionScores,
    ContactComplianceState,
    StateChangeRecord,
)
from .conversation_facts import PersistentFactsManager
from .conversation_objections import ObjectionLifecycleEngine

LOGGER = logging.getLogger("copilot.conversation_state_manager")


class StaleStateUpdateError(ValueError):
    """Raised when an asynchronous or delayed update attempts to mutate state from an older turn."""
    pass


class ConversationStateManager:
    """Core state engine coordinating snapshots, explainability tracking, and monotonic versioning."""

    def __init__(self, call_sid: str, initial_snapshot: Optional[ConversationStateSnapshot] = None):
        self.call_sid = call_sid
        self.current_state = initial_snapshot or ConversationStateSnapshot(call_sid=call_sid)
        self.facts_manager = PersistentFactsManager(initial_facts=self.current_state.facts)
        self.objections_engine = ObjectionLifecycleEngine(initial_objections=self.current_state.objections)

    def process_turn_bundle(
        self,
        bundle: BehavioralSignalInputBundle,
        decision_updates: Optional[Dict[str, Any]] = None,
        fact_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> ConversationStateSnapshot:
        """Applies a verified Behavioral Signal Engine turn bundle to update the conversation truth."""
        # Stale-write protection (Client Principle #9)
        # 1. Strictly reject turns with turn_id older than latest processed turn
        if bundle.turn_id < self.current_state.last_updated_turn_id:
            raise StaleStateUpdateError(
                f"Stale state rejection: bundle turn {bundle.turn_id} is older than current turn "
                f"{self.current_state.last_updated_turn_id}."
            )
        # 2. If turn_id equals latest turn, allow intra-turn refinement (e.g. late LLM enrichment or
        # continuation fragment) ONLY IF its timestamp is >= current state timestamp.
        # Reject if timestamp is strictly earlier (stale out-of-order packet).
        if (
            bundle.turn_id == self.current_state.last_updated_turn_id
            and bundle.timestamp_ms < self.current_state.last_updated_timestamp_ms
        ):
            raise StaleStateUpdateError(
                f"Stale state rejection: bundle turn {bundle.turn_id} has earlier timestamp "
                f"({bundle.timestamp_ms}ms) than current processed turn ({self.current_state.last_updated_timestamp_ms}ms)."
            )

        changes: List[StateChangeRecord] = []
        old_version = self.current_state.state_version
        next_version = old_version

        # 1. Update Dimensions (independent scoring, no auto-coupling)
        new_dims = DimensionScores(
            trust=bundle.trust.score,
            trust_confidence=bundle.trust.confidence,
            emotion_valence=bundle.emotion.expressed_valence,
            emotion_tension=bundle.emotion.tension_level,
            emotion_confidence=bundle.emotion.confidence,
            engagement=bundle.engagement.score,
            engagement_confidence=bundle.engagement.confidence,
            momentum=bundle.momentum.score,
            momentum_confidence=bundle.momentum.confidence,
            readiness=bundle.readiness.score,
            readiness_confidence=bundle.readiness.confidence,
            pacing=bundle.pacing.score,
            pacing_confidence=bundle.pacing.confidence,
        )
        if new_dims != self.current_state.dimensions:
            next_version += 1
            changes.append(
                StateChangeRecord(
                    state_version_before=old_version,
                    state_version_after=next_version,
                    field_path="dimensions",
                    old_value=self.current_state.dimensions.model_dump(),
                    new_value=new_dims.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason="Updated downstream inference dimension scores from Behavioral Signal Engine.",
                    timestamp_ms=bundle.timestamp_ms,
                )
            )
            self.current_state.dimensions = new_dims

        # 2. Update Contact/Compliance State
        hard_boundary = bundle.boundary_score >= 0.85
        boundary_reason = "Triggered by upstream compliance boundary detection" if hard_boundary else None
        new_compliance = ContactComplianceState(
            hard_boundary_active=hard_boundary,
            hard_boundary_reason=boundary_reason,
            contact_preference=bundle.contact_preference,
            contact_preference_details=bundle.contact_preference_details,
            contact_preference_confidence=bundle.contact_preference_confidence,
        )
        if new_compliance != self.current_state.contact_compliance:
            next_version += 1
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="contact_compliance",
                    old_value=self.current_state.contact_compliance.model_dump(),
                    new_value=new_compliance.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Updated compliance state (boundary={hard_boundary}, pref={bundle.contact_preference}).",
                    timestamp_ms=bundle.timestamp_ms,
                )
            )
            self.current_state.contact_compliance = new_compliance

        # 3. Apply Decision Structure updates if provided
        if decision_updates:
            next_version += 1
            old_dec = self.current_state.decision_structure.model_dump()
            for k, v in decision_updates.items():
                if hasattr(self.current_state.decision_structure, k):
                    setattr(self.current_state.decision_structure, k, v)
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="decision_structure",
                    old_value=old_dec,
                    new_value=self.current_state.decision_structure.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason="Updated decision structure fields.",
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        # 4. Evaluate Objection Lifecycle (Sprint 2 - Phase 3)
        updated_objs, obj_changes, next_version = self.objections_engine.evaluate_turn(
            bundle=bundle,
            current_version=next_version,
        )
        self.current_state.objections = updated_objs
        changes.extend(obj_changes)

        # 5. Process Fact Updates if provided
        if fact_updates:
            for fu in fact_updates:
                self.facts_manager.record_fact(
                    category=fu.get("category", "general"),
                    fact_key=fu["fact_key"],
                    fact_value=fu["fact_value"],
                    source_turn_id=bundle.turn_id,
                    timestamp_ms=bundle.timestamp_ms,
                    confidence=fu.get("confidence", 1.0),
                    notes=fu.get("notes"),
                )
            next_version += 1
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="facts",
                    old_value=[f.model_dump() for f in self.current_state.facts],
                    new_value=[f.model_dump() for f in self.facts_manager.get_all_facts()],
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Recorded {len(fact_updates)} persistent fact update(s).",
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        # 6. Sync facts snapshot
        self.current_state.facts = self.facts_manager.get_all_facts()

        # Update metadata
        self.current_state.last_updated_turn_id = bundle.turn_id
        self.current_state.last_updated_timestamp_ms = bundle.timestamp_ms
        self.current_state.state_version = next_version
        self.current_state.overall_confidence = min(
            bundle.inference_confidence,
            bundle.semantic_confidence,
            self.current_state.decision_structure.confidence,
        )
        self.current_state.change_history.extend(changes)

        return self.current_state
