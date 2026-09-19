from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStructure,
    DecisionStakeholder,
    DimensionScores,
    ContactComplianceState,
    PushStrengthRecommendation,
    StateChangeRecord,
    ConversionEventStatus,
    ConversionEventObject,
    DealDispositionType,
    DealDispositionRecord,
    ObjectionLifecycleState,
)
from .conversation_facts import PersistentFactsManager
from .conversation_objections import ObjectionLifecycleEngine, DECISION_TO_STAY_PATTERNS
from .conversation_supersession import TruthSupersessionDetector
from .conversation_materiality import MaterialityFilter, ABSENT_DECISION_MAKER_PATTERNS
from .conversation_scoring import ConversationScoringEngine
from .conversation_scoring_config import ConversationScoringConfig
from .conversation_conversion_config import ConversionBlockingConfig
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
        blocking_config: Optional[ConversionBlockingConfig] = None,
        conversion_target: str = "appointment",
    ):
        self.call_sid = call_sid
        self.conversion_target = conversion_target
        self.current_state = initial_snapshot or ConversationStateSnapshot(call_sid=call_sid)
        self.facts_manager = PersistentFactsManager(initial_facts=self.current_state.facts)
        dormancy_thresh = scoring_config.dormancy_turn_threshold if scoring_config else 3
        self.objections_engine = ObjectionLifecycleEngine(
            initial_objections=self.current_state.objections,
            dormancy_turn_threshold=dormancy_thresh,
        )
        self.supersession_detector = TruthSupersessionDetector()
        self.materiality_filter = MaterialityFilter()
        self.scoring_engine = ConversationScoringEngine(config=scoring_config)
        self.conversion_engine = MeetingConversionGateEngine(
            config=scoring_config,
            blocking_config=blocking_config,
        )
        self.prior_bundle: Optional[BehavioralSignalInputBundle] = None
        self.has_prospect_spoken: bool = False
        self._conversion_events: List[ConversionEventObject] = list(self.current_state.conversion_events)
        if self.current_state.conversion_event and not any(e.event_id == self.current_state.conversion_event.event_id for e in self._conversion_events):
            self._conversion_events.append(self.current_state.conversion_event)
        self.deal_disposition: Optional[DealDispositionRecord] = self.current_state.deal_disposition or DealDispositionRecord(disposition=DealDispositionType.ACTIVELY_SELLING)
        self.current_state.deal_disposition = self.deal_disposition
        self._deal_dispositions: List[DealDispositionRecord] = list(self.current_state.deal_dispositions)
        if self.deal_disposition and not any(d.disposition_id == self.deal_disposition.disposition_id for d in self._deal_dispositions):
            self._deal_dispositions.append(self.deal_disposition)
        self.current_state.deal_dispositions = list(self._deal_dispositions)

    def process_turn_bundle(
        self,
        bundle: BehavioralSignalInputBundle,
        decision_updates: Optional[Dict[str, Any]] = None,
        fact_updates: Optional[List[Dict[str, Any]]] = None,
        conversion_target: Optional[str] = None,
    ) -> ConversationStateSnapshot:
        """Applies a verified Behavioral Signal Engine turn bundle to update the conversation truth."""
        if conversion_target is not None:
            self.conversion_target = conversion_target
        if bundle.speaker_id == "client":
            self.has_prospect_spoken = True
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
            prior_bundle=self.prior_bundle,
        )

        changes: List[StateChangeRecord] = []
        old_version = self.current_state.state_version
        next_version = old_version

        dormancy_thresh = getattr(self.objections_engine, "dormancy_turn_threshold", 3)
        dormant_due = any(
            o.lifecycle_state in ("active", "partially_addressed", "unresolved", "reactivated", ObjectionLifecycleState.ACTIVE, ObjectionLifecycleState.PARTIALLY_ADDRESSED)
            and (bundle.turn_id - o.last_updated_turn_id) >= dormancy_thresh
            for o in self.current_state.objections
        )

        # If turn is non-material (no targets affected, no manual decision/fact updates, and no dormancy aging due),
        # preserve state version and dimension stability entirely without mutating state.
        if not materiality.is_material and not decision_updates and not fact_updates and not dormant_due:
            self.current_state.last_updated_turn_id = bundle.turn_id
            self.current_state.last_updated_timestamp_ms = bundle.timestamp_ms
            self.prior_bundle = bundle
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

        # 3. Apply Decision Structure updates (Explicit or Autonomous from bundle)
        auto_decision_updates = None
        if not decision_updates and bundle.speaker_id == "client":
            auto_decision_updates = self._extract_autonomous_decision_updates(bundle, materiality)

        effective_dec_updates = decision_updates or auto_decision_updates
        if effective_dec_updates:
            next_version += 1
            old_dec = self.current_state.decision_structure.model_dump()
            for k, v in effective_dec_updates.items():
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
                    reason=f"Updated decision structure ({', '.join(effective_dec_updates.keys())}).",
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        # 4. Evaluate Objection Lifecycle (Gated by Materiality)
        if "objections" in materiality.affected_targets:
            # Sync any out-of-band added objections into the lifecycle engine
            engine_ids = {o.objection_id for o in self.objections_engine._objections}
            for o in self.current_state.objections:
                if o.objection_id not in engine_ids:
                    self.objections_engine._objections.append(o)

            updated_objs, obj_changes, next_version = self.objections_engine.evaluate_turn(
                bundle=bundle,
                current_version=next_version,
            )
            self.current_state.objections = updated_objs
            changes.extend(obj_changes)

        # 4b. Evaluate Dormancy Aging for unaddressed concerns across turns
        updated_objs, dormant_changes, next_version = self.objections_engine.check_dormancy_aging(
            current_turn_id=bundle.turn_id,
            timestamp_ms=bundle.timestamp_ms,
            current_version=next_version,
            contributing_evidence_ids=bundle.contributing_evidence_ids,
        )
        self.current_state.objections = updated_objs
        changes.extend(dormant_changes)

        # 5. Process Fact Updates if provided or autonomously extracted from client disclosures
        autonomous_fact_updates = None
        if not fact_updates and bundle.speaker_id == "client":
            autonomous_fact_updates = self._extract_autonomous_fact_updates(bundle, materiality)
        all_fact_updates = (fact_updates or []) + (autonomous_fact_updates or [])
        if all_fact_updates:
            for fu in all_fact_updates:
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
                    reason=f"Recorded {len(all_fact_updates)} persistent fact update(s).",
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

        # 7b. Evaluate Deal Disposition (Phase 8 / Fix 2)
        if bundle.speaker_id == "client":
            new_disp = self.supersession_detector.evaluate_deal_disposition(
                candidate_text=bundle.utterance_text,
                current_disposition=self.deal_disposition,
                turn_id=bundle.turn_id,
                timestamp_ms=bundle.timestamp_ms,
            )
            if new_disp is not None and (self.deal_disposition is None or self.deal_disposition.disposition != new_disp.disposition or self.deal_disposition.disposition_id != new_disp.disposition_id):
                if self.deal_disposition is not None and self.deal_disposition.disposition_id != new_disp.disposition_id:
                    self.deal_disposition.superseded_by_id = new_disp.disposition_id
                    self.deal_disposition.superseded_at_turn_id = bundle.turn_id

                if not any(d.disposition_id == new_disp.disposition_id for d in self._deal_dispositions):
                    self._deal_dispositions.append(new_disp)
                prev_disp = self.deal_disposition
                self.deal_disposition = new_disp
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path="deal_disposition",
                        old_value=prev_disp.model_dump() if prev_disp else None,
                        new_value=new_disp.model_dump(),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Deal disposition transitioned to {new_disp.disposition}: {new_disp.rationale or ''}".strip(),
                        confidence=new_disp.confidence,
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )

        self.current_state.deal_disposition = self.deal_disposition
        self.current_state.deal_dispositions = list(self._deal_dispositions)

        # 8. Compute Momentum & Readiness Scoring (Phase 6)
        momentum_res = self.scoring_engine.compute_momentum(bundle, self.current_state)
        readiness_res = self.scoring_engine.compute_readiness(bundle, self.current_state)
        self.current_state.momentum = momentum_res
        self.current_state.readiness = readiness_res

        # 9. Evaluate Meeting/Conversion Gate & Push Strength (Phase 7 / Sprint 6)
        prev_gate = self.current_state.conversion_gate
        prev_conv = self.current_state.conversion_event

        # Speaker Blindspot Protection: Gate evaluation is held pending until the prospect speaks,
        # preventing evaluating agent questions against prospect conversion criteria.
        if not self.has_prospect_spoken and bundle.speaker_id == "salesperson":
            gate_res = None
            push_res = PushStrengthRecommendation(
                state="two_window_choice",
                rationale="Awaiting prospect opening response.",
                recommended_action="Discover prospect availability or offer two window choices once prospect responds.",
                confidence=0.50,
            )
            conv_res = None
        else:
            gate_res = self.conversion_engine.evaluate_gate(
                bundle,
                self.current_state,
                conversion_target=self.conversion_target,
                prior_bundle=self.prior_bundle,
            )
            push_res = self.conversion_engine.evaluate_push_strength(
                bundle,
                self.current_state,
                gate_res,
                conversion_target=self.conversion_target,
            )
            conv_res = self.conversion_engine.evaluate_conversion_event(
                bundle, self.current_state, gate_res, previous_event=prev_conv
            )

        # Explainability tracking for conversion state changes
        if gate_res is not None and (
            prev_gate is None
            or prev_gate.is_open != gate_res.is_open
            or prev_gate.conversion_target != gate_res.conversion_target
            or (gate_res.explicit_commitment_detected and not prev_gate.explicit_commitment_detected)
        ):
            next_version += 1
            override_note = " (explicit commitment override applied)" if gate_res.explicit_commitment_detected else ""
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="conversion_gate",
                    old_value=prev_gate.model_dump() if prev_gate else None,
                    new_value=gate_res.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Meeting gate [{gate_res.conversion_target}] transitioned to {'OPEN' if gate_res.is_open else 'CLOSED'}{override_note}. Failed conditions: {gate_res.failed_conditions or 'None'}.",
                    confidence=gate_res.confidence,
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        if conv_res is not None:
            # Sync into _conversion_events
            existing_idx = next((i for i, e in enumerate(self._conversion_events) if e.event_id == conv_res.event_id), None)
            if existing_idx is not None:
                self._conversion_events[existing_idx] = conv_res
            else:
                self._conversion_events.append(conv_res)

            # If it supersedes an event, ensure that event in self._conversion_events is marked superseded
            if conv_res.supersedes_event_id:
                for ev in self._conversion_events:
                    if ev.event_id == conv_res.supersedes_event_id:
                        ev.superseded_by_event_id = conv_res.event_id
                        ev.superseded_at_turn_id = bundle.turn_id

            if prev_conv is None or prev_conv.status != conv_res.status or prev_conv.event_id != conv_res.event_id:
                next_version += 1
                supersede_info = f" (supersedes {conv_res.supersedes_event_id}, reversal_reason: {conv_res.reversal_reason})" if conv_res.supersedes_event_id else ""
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path="conversion_event",
                        old_value=prev_conv.model_dump() if prev_conv else None,
                        new_value=conv_res.model_dump(),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Conversion event updated: {conv_res.conversion_type} status={conv_res.status}{supersede_info}.",
                        confidence=conv_res.confirmation_confidence,
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )

            # Deterministic Cascade: Conversion Event status directly governs confirmed_meeting_time fact
            active_meeting_fact = next(
                (f for f in self.facts_manager.get_active_facts() if f.fact_key == "confirmed_meeting_time"),
                None,
            )
            if conv_res.status == ConversionEventStatus.CANCELLED and active_meeting_fact:
                old_f, new_f = self.facts_manager.supersede_fact(
                    old_fact_id=active_meeting_fact.fact_id,
                    new_fact_value="Cancelled",
                    source_turn_id=bundle.turn_id,
                    timestamp_ms=bundle.timestamp_ms,
                    notes=conv_res.reversal_reason or "Conversion event cancelled",
                )
                self.current_state.facts = self.facts_manager.get_all_facts()
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path="facts.confirmed_meeting_time",
                        old_value=old_f.fact_value,
                        new_value=new_f.fact_value,
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Deterministic cascade from conversion event cancellation: {conv_res.reversal_reason or 'Cancelled'}",
                        confidence=conv_res.confirmation_confidence,
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )
            elif conv_res.status == ConversionEventStatus.CONFIRMED and conv_res.start_at:
                if active_meeting_fact and active_meeting_fact.fact_value.strip().lower() != conv_res.start_at.strip().lower():
                    old_f, new_f = self.facts_manager.supersede_fact(
                        old_fact_id=active_meeting_fact.fact_id,
                        new_fact_value=conv_res.start_at,
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        notes=conv_res.reversal_reason or f"Meeting rescheduled to '{conv_res.start_at}'",
                    )
                    self.current_state.facts = self.facts_manager.get_all_facts()
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path="facts.confirmed_meeting_time",
                            old_value=old_f.fact_value,
                            new_value=new_f.fact_value,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Deterministic cascade from conversion event confirmation: '{conv_res.start_at}'",
                            confidence=conv_res.confirmation_confidence,
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )

        self.current_state.conversion_gate = gate_res
        self.current_state.push_strength = push_res
        self.current_state.conversion_event = conv_res
        self.current_state.conversion_events = list(self._conversion_events)

        # Update metadata
        self.current_state.last_updated_turn_id = bundle.turn_id
        self.current_state.last_updated_timestamp_ms = bundle.timestamp_ms
        self.current_state.state_version = next_version
        self.current_state.overall_confidence = min(
            bundle.inference_confidence,
            bundle.semantic_confidence,
            self.current_state.decision_structure.confidence,
            readiness_res.confidence,
            gate_res.confidence if gate_res else 0.50,
        )
        self.current_state.change_history.extend(changes)
        self.prior_bundle = bundle

        return self.current_state

    def _extract_autonomous_decision_updates(
        self,
        bundle: BehavioralSignalInputBundle,
        materiality: Any,
    ) -> Optional[Dict[str, Any]]:
        """Autonomously extracts decision structure updates from client disclosures when explicit updates are not provided."""
        if bundle.speaker_id != "client":
            return None

        text = bundle.utterance_text.lower().strip()

        # 1. Direct absent decision maker patterns
        if any(re.search(pat, text) for pat in ABSENT_DECISION_MAKER_PATTERNS):
            role = "co_decision_maker"
            if "husband" in text:
                role = "husband"
            elif "wife" in text:
                role = "wife"
            elif "partner" in text or "spouse" in text:
                role = "partner"
            elif "attorney" in text or "lawyer" in text:
                role = "attorney"
            elif "he" in text:
                role = "male_decision_maker"
            elif "she" in text:
                role = "female_decision_maker"

            stakeholder = DecisionStakeholder(
                stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
                role=role,
                is_decision_maker=True,
                presence="absent",
                notes=f"Identified as absent decision maker from client disclosure: '{bundle.utterance_text[:60]}'",
                confidence=0.85,
            )

            existing = list(self.current_state.decision_structure.stakeholders)
            if not any(s.role == role and s.presence == "absent" for s in existing):
                existing.append(stakeholder)

            return {
                "decision_maker_present": False,
                "stakeholders": existing,
            }

        # 2. Affirmative confirmation of prior salesperson inquiry about third-party stakeholder
        if self.prior_bundle and self.prior_bundle.speaker_id == "salesperson":
            prior_text_l = self.prior_bundle.utterance_text.lower()
            asked_stakeholder = any(w in prior_text_l for w in ["him involved", "her involved", "them involved", "husband", "wife", "partner", "spouse", "decision maker", "sign off"])
            is_affirmative = (
                bundle.agreement_score >= 0.60
                or any(re.search(rf"\b{aff}\b", text) for aff in ["yeah", "yes", "definitely", "sure", "absolutely", "correct", "of course"])
            )
            if asked_stakeholder and is_affirmative:
                role = "male_decision_maker" if "him" in prior_text_l else ("female_decision_maker" if "her" in prior_text_l else "co_decision_maker")
                stakeholder = DecisionStakeholder(
                    stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
                    role=role,
                    is_decision_maker=True,
                    presence="absent",
                    notes=f"Confirmed stakeholder involvement in response to agent inquiry: '{bundle.utterance_text[:60]}'",
                    confidence=0.80,
                )
                existing = list(self.current_state.decision_structure.stakeholders)
                if not any(s.role == role and s.presence == "absent" for s in existing):
                    existing.append(stakeholder)

                return {
                    "decision_maker_present": False,
                    "stakeholders": existing,
                }

        return None

    def _extract_autonomous_fact_updates(
        self,
        bundle: BehavioralSignalInputBundle,
        materiality: Any,
    ) -> List[Dict[str, Any]]:
        """Autonomously extracts persistent facts (meeting confirmations, contact preferences, etc.) from dialogue evidence."""
        if bundle.speaker_id != "client":
            return []

        updates: List[Dict[str, Any]] = []
        text_lower = bundle.utterance_text.lower().strip()
        existing_keys = {f.fact_key for f in self.current_state.facts if f.status == "active"}

        # 1. Contact preference
        if bundle.contact_preference != "none" and "contact_preference" not in existing_keys:
            updates.append({
                "category": "preference",
                "fact_key": "contact_preference",
                "fact_value": bundle.contact_preference,
                "confidence": bundle.contact_preference_confidence or 0.85,
                "notes": f"Communication preference declared: {bundle.contact_preference_details or bundle.contact_preference}",
            })

        # 2. Confirmed meeting / appointment time
        # Case A: Prospect explicitly mentions day and time
        time_day_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b.*?\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            text_lower,
        )
        if time_day_match:
            new_val = time_day_match.group(0).strip().title()
            if "confirmed_meeting_time" not in existing_keys:
                updates.append({
                    "category": "timeline",
                    "fact_key": "confirmed_meeting_time",
                    "fact_value": new_val,
                    "confidence": 0.85,
                    "notes": f"Prospect confirmed meeting time directly: '{bundle.utterance_text[:60]}'",
                })
            else:
                active_f = next((f for f in self.current_state.facts if f.fact_key == "confirmed_meeting_time" and f.status == "active"), None)
                if active_f and active_f.fact_value.strip().lower() != new_val.strip().lower():
                    self.facts_manager.supersede_fact(
                        old_fact_id=active_f.fact_id,
                        new_fact_value=new_val,
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        notes=f"Meeting rescheduled to '{new_val}'",
                    )
        elif self.prior_bundle and self.prior_bundle.speaker_id == "salesperson":
            # Case B: Salesperson proposed a day/time and prospect confirmed affirmatively
            prior_text_lower = self.prior_bundle.utterance_text.lower()
            prop_match = re.search(
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b.*?\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
                prior_text_lower,
            )
            is_affirmative = (
                bundle.agreement_score >= 0.60
                or any(re.search(rf"\b{aff}\b", text_lower) for aff in ["yeah", "yes", "definitely", "sure", "absolutely", "works", "perfect", "sounds good"])
            )
            if prop_match and is_affirmative and "confirmed_meeting_time" not in existing_keys:
                raw_time_str = prop_match.group(0).title()
                norm_time = raw_time_str
                for word, num in [("One", "1:00 PM"), ("Two", "2:00 PM"), ("Three", "3:00 PM"), ("Four", "4:00 PM"), ("Five", "5:00 PM")]:
                    if f"At {word}" in norm_time:
                        norm_time = norm_time.replace(f"At {word}", f"at {num}")
                updates.append({
                    "category": "timeline",
                    "fact_key": "confirmed_meeting_time",
                    "fact_value": norm_time,
                    "confidence": 0.85,
                    "notes": f"Prospect confirmed proposed time '{raw_time_str}' from salesperson proposal.",
                })



        # 3. Decision to Stay / Cancellation of sale
        if any(re.search(p, text_lower) for p in DECISION_TO_STAY_PATTERNS) and "decision_to_stay" not in existing_keys:
            updates.append({
                "category": "timeline",
                "fact_key": "decision_to_stay",
                "fact_value": "Prospect decided to stay in home / cancel sale",
                "confidence": 0.90,
                "notes": f"Decision to stay declared by prospect: '{bundle.utterance_text[:60]}'",
            })

        # 4. Spousal / Absent Decision Maker Involvement
        if any(re.search(p, text_lower) for p in ABSENT_DECISION_MAKER_PATTERNS) and "spouse_involvement" not in existing_keys:
            updates.append({
                "category": "decision_maker",
                "fact_key": "spouse_involvement",
                "fact_value": "Spouse/wife involvement declared for decisions",
                "confidence": 0.85,
                "notes": f"Spouse involvement declared by prospect: '{bundle.utterance_text[:60]}'",
            })

        return updates

    def get_conversion_event_history(self) -> List[ConversionEventObject]:
        """Returns the complete chronological history of conversion events (both superseded and active)."""
        return list(self._conversion_events)

    def get_active_conversion_event(self) -> Optional[ConversionEventObject]:
        """Returns the current active (unsuperseded) conversion event, if any."""
        for ev in reversed(self._conversion_events):
            if ev.superseded_by_event_id is None:
                return ev
        return self.current_state.conversion_event

    def get_deal_disposition_history(self) -> List[DealDispositionRecord]:
        """Returns the complete chronological history of deal disposition records."""
        return list(self._deal_dispositions)

    def get_active_deal_disposition(self) -> Optional[DealDispositionRecord]:
        """Returns the current active (unsuperseded) deal disposition record."""
        for d in reversed(self._deal_dispositions):
            if d.superseded_by_id is None:
                return d
        return self.deal_disposition
