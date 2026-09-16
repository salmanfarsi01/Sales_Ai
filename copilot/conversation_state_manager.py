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
from .conversation_supersession import TruthSupersessionDetector
from .conversation_materiality import MaterialityFilter
from .conversation_scoring import ConversationScoringEngine
from .conversation_scoring_config import ConversationScoringConfig
from .conversation_conversion import MeetingConversionGateEngine

LOGGER = logging.getLogger("copilot.conversation_state_manager")


class StaleStateUpdateError(ValueError):
    """Raised when an asynchronous or delayed update attempts to mutate state from an older turn."""
    pass


class ConversationStateManager:
    """Core state engine coordinating snapshots, explainability tracking, and monotonic versioning."""

    def __init__(
        self,
        call_sid: str,
        initial_snapshot: Optional[ConversationStateSnapshot] = None,
        scoring_config: Optional[ConversationScoringConfig] = None,
    ):
        self.call_sid = call_sid
        self.current_state = initial_snapshot or ConversationStateSnapshot(call_sid=call_sid)
        self.facts_manager = PersistentFactsManager(initial_facts=self.current_state.facts)
        self.objections_engine = ObjectionLifecycleEngine(initial_objections=self.current_state.objections)
        self.supersession_detector = TruthSupersessionDetector()
        self.materiality_filter = MaterialityFilter()
        self.scoring_engine = ConversationScoringEngine(config=scoring_config)
        self.conversion_engine = MeetingConversionGateEngine(config=scoring_config)

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

        # Tier-1 Turn-Level Gating: Materiality Filter (Phase 5 / Sprint 4)
        materiality = self.materiality_filter.classify_turn(
            bundle=bundle,
            current_state=self.current_state,
            has_explicit_fact_updates=bool(fact_updates),
        )

        changes: List[StateChangeRecord] = []
        old_version = self.current_state.state_version
        next_version = old_version

        # If turn is non-material (no targets affected and no manual decision/fact updates),
        # preserve state version and dimension stability entirely without mutating state.
        if not materiality.is_material and not decision_updates and not fact_updates:
            self.current_state.last_updated_turn_id = bundle.turn_id
            self.current_state.last_updated_timestamp_ms = bundle.timestamp_ms
            return self.current_state

        # 1. Update Dimensions (Gated by Materiality)
        if "dimensions" in materiality.affected_targets:
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

        # 2. Update Contact/Compliance State (Gated by Materiality)
        if "contact_compliance" in materiality.affected_targets:
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

        # 4. Evaluate Objection Lifecycle (Gated by Materiality)
        if "objections" in materiality.affected_targets:
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

        # 6. Evaluate Truth Supersession (Gated by Materiality)
        if "facts" in materiality.affected_targets and bundle.speaker_id == "client":
            supersession_decisions = self.supersession_detector.evaluate_turn(
                candidate_text=bundle.utterance_text,
                active_facts=self.facts_manager.get_active_facts(),
                decision_structure=self.current_state.decision_structure,
            )
            for decision in supersession_decisions:
                if decision.has_supersession and decision.target_fact_id:
                    effective_conf = round(min(decision.confidence, bundle.semantic_confidence, bundle.inference_confidence), 3)
                    old_f, new_f = self.facts_manager.supersede_fact(
                        old_fact_id=decision.target_fact_id,
                        new_fact_value=decision.new_truth_value or bundle.utterance_text,
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        confidence=effective_conf,
                        notes=decision.reasoning,
                    )
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"facts.{decision.target_fact_key}",
                            old_value=old_f.fact_value,
                            new_value=new_f.fact_value,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Truth supersession ({decision.relation}): {decision.reasoning}",
                            confidence=effective_conf,
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )

        # 7. Sync facts snapshot
        self.current_state.facts = self.facts_manager.get_all_facts()

        # 8. Compute Momentum & Readiness Scoring (Phase 6)
        momentum_res = self.scoring_engine.compute_momentum(bundle, self.current_state)
        readiness_res = self.scoring_engine.compute_readiness(bundle, self.current_state)
        self.current_state.momentum = momentum_res
        self.current_state.readiness = readiness_res

        # 9. Evaluate Meeting/Conversion Gate & Push Strength (Phase 7 / Sprint 6)
        prev_gate = self.current_state.conversion_gate
        prev_conv = self.current_state.conversion_event
        gate_res = self.conversion_engine.evaluate_gate(bundle, self.current_state)
        push_res = self.conversion_engine.evaluate_push_strength(bundle, self.current_state, gate_res)
        conv_res = self.conversion_engine.evaluate_conversion_event(
            bundle, self.current_state, gate_res, previous_event=prev_conv
        )

        # Explainability tracking for conversion state changes
        if prev_gate is None or prev_gate.is_open != gate_res.is_open:
            next_version += 1
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="conversion_gate",
                    old_value=prev_gate.model_dump() if prev_gate else None,
                    new_value=gate_res.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Meeting gate transitioned to {'OPEN' if gate_res.is_open else 'CLOSED'}. Failed conditions: {gate_res.failed_conditions or 'None'}.",
                    confidence=gate_res.confidence,
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        if conv_res and (prev_conv is None or prev_conv.status != conv_res.status):
            next_version += 1
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="conversion_event",
                    old_value=prev_conv.model_dump() if prev_conv else None,
                    new_value=conv_res.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Conversion event updated: {conv_res.conversion_type} status={conv_res.status} (followup_is_conversion={conv_res.followup_is_conversion}).",
                    confidence=conv_res.confirmation_confidence,
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        self.current_state.conversion_gate = gate_res
        self.current_state.push_strength = push_res
        self.current_state.conversion_event = conv_res

        # Update metadata
        self.current_state.last_updated_turn_id = bundle.turn_id
        self.current_state.last_updated_timestamp_ms = bundle.timestamp_ms
        self.current_state.state_version = next_version
        self.current_state.overall_confidence = min(
            bundle.inference_confidence,
            bundle.semantic_confidence,
            self.current_state.decision_structure.confidence,
            readiness_res.confidence,
            gate_res.confidence,
        )
        self.current_state.change_history.extend(changes)

        return self.current_state
