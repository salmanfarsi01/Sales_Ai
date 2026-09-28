from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from .conversation_scoring_config import ConversationScoringConfig, DEFAULT_CONVERSATION_SCORING_CONFIG
from .conversation_conversion_config import ConversionBlockingConfig, DEFAULT_CONVERSION_BLOCKING_CONFIG
from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
    ConversationStage,
    ConversionEventStatus,
    DecisionStructure,
    DimensionScores,
    ContactComplianceState,
    ObjectionRecord,
    MomentumTrend,
    MomentumBreakdown,
    ReadinessBreakdown,
)

LOGGER = logging.getLogger("copilot.conversation_scoring")


class ConversationScoringEngine:
    """Phase 6: Computes Momentum & Readiness composite scores with deterministic blocker caps."""

    def __init__(self, config: Optional[ConversationScoringConfig] = None):
        self.config = config or DEFAULT_CONVERSATION_SCORING_CONFIG
        self._momentum_history: List[float] = []

    def compute_momentum(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
    ) -> MomentumBreakdown:
        """Computes the 7-family weighted momentum score (0-100) and trend trajectory."""
        cfg = self.config
        dims = current_state.dimensions

        # 1. Problem / Goal Clarity (15%)
        # Combines prospect utterance specificity with recorded timeline/moving facts
        spec_factor = bundle.specificity_score * 100.0 if bundle.speaker_id == "client" else 50.0
        has_goal_facts = any(f.category in ("timeline", "financial") for f in current_state.facts if f.status == "active")
        problem_goal_score = round(0.6 * spec_factor + (40.0 if has_goal_facts else 15.0), 1)
        problem_goal_score = min(100.0, max(0.0, problem_goal_score))

        # 2. Value Recognition (15%)
        # Upstream agreement score and absence of heavy value skepticism
        agreement_factor = bundle.agreement_score * 100.0
        value_score = round(0.7 * agreement_factor + 0.3 * (dims.trust * 100.0), 1)
        value_score = min(100.0, max(0.0, value_score))

        # 3. Objection Movement (20%)
        # Evaluates resolution trajectory of raised objections
        if not current_state.objections:
            objection_score = 80.0  # Clean slate
        else:
            obj_states = [o.lifecycle_state for o in current_state.objections]
            # Client Principle / Behavioral Correctness: If an objection was superseded
            # by a deal cancellation ("decision_to_stay"), it is a deal dead-end, NOT a progressive resolution.
            has_dead_end_supersession = any(
                o.lifecycle_state == "superseded" and o.superseded_by_objection_id == "decision_to_stay"
                for o in current_state.objections
            ) or any(
                f.fact_key == "decision_to_stay" and f.status == "active"
                for f in current_state.facts
            )
            if "boundary" in obj_states or has_dead_end_supersession:
                objection_score = 0.0
            elif all(s in ("resolved", "superseded") for s in obj_states):
                objection_score = 100.0
            elif any(s == "partially_resolved" for s in obj_states):
                objection_score = 70.0
            elif any(s == "clarified" for s in obj_states):
                objection_score = 55.0
            else:
                objection_score = 30.0

        # 4. Trust / Engagement Trend (15%)
        trust_pts = dims.trust * 100.0
        eng_pts = dims.engagement * 100.0
        tension_penalty = dims.emotion_tension * 30.0
        trust_eng_score = round(0.5 * trust_pts + 0.5 * eng_pts - tension_penalty, 1)
        trust_eng_score = min(100.0, max(0.0, trust_eng_score))

        # 5. Decision Structure Clarity (10%)
        dec = current_state.decision_structure
        has_primary = bool(dec.primary_decision_maker)
        has_absent_stakeholder = any(s.presence == "absent" for s in dec.stakeholders)
        if has_primary and dec.decision_maker_present and not has_absent_stakeholder:
            dec_score = 95.0
        elif has_primary and not has_absent_stakeholder:
            dec_score = 75.0
        elif has_primary:
            dec_score = 55.0
        else:
            dec_score = 40.0

        # 6. Future / Operational Behavior (15%)
        # Evaluates calendar language, scheduled appointments, and operational timeline targets
        future_pts = bundle.future_language_score * 100.0
        has_closing_target = any(f.fact_key in ("target_closing", "move_in_date") for f in current_state.facts if f.status == "active")
        has_confirmed_meeting = (
            bool(current_state.conversion_event and getattr(current_state.conversion_event, "status", "") in ("confirmed", ConversionEventStatus.CONFIRMED))
            or any(f.fact_key == "confirmed_meeting_time" and f.status == "active" for f in current_state.facts)
        )
        has_tentative_meeting = any(f.fact_key == "tentative_meeting_time" and f.status == "active" for f in current_state.facts)

        if has_confirmed_meeting:
            future_score = round(80.0 + (0.2 * future_pts), 1)
        elif has_tentative_meeting or getattr(current_state, "conversation_stage", None) == ConversationStage.SCHEDULING:
            future_score = round(45.0 + (0.35 * future_pts), 1)
        elif has_closing_target:
            future_score = round(0.6 * future_pts + 40.0, 1)
        else:
            future_score = round(0.6 * future_pts + 15.0, 1)
        future_score = min(100.0, max(0.0, future_score))

        # 7. Commitment Behavior (10%)
        # Consumes explicit commitment dimension (what they have agreed to, 0-100)
        # or confirmed meeting fact, blended with conversational pacing and turn agreement.
        pacing_pts = dims.pacing * 100.0
        effective_commitment = max(dims.commitment, 1.0 if has_confirmed_meeting else 0.0)
        commit_pts = effective_commitment * 100.0
        if commit_pts > 0.0:
            commit_score = round(0.6 * commit_pts + 0.25 * agreement_factor + 0.15 * pacing_pts, 1)
        else:
            commit_score = round(0.6 * agreement_factor + 0.4 * pacing_pts, 1)
        commit_score = min(100.0, max(0.0, commit_score))

        # Composite Weighted Momentum Score
        composite_momentum = (
            problem_goal_score * cfg.problem_goal_clarity_weight
            + value_score * cfg.value_recognition_weight
            + objection_score * cfg.objection_movement_weight
            + trust_eng_score * cfg.trust_engagement_trend_weight
            + dec_score * cfg.decision_structure_clarity_weight
            + future_score * cfg.future_operational_behavior_weight
            + commit_score * cfg.commitment_behavior_weight
        )
        composite_momentum = round(min(100.0, max(0.0, composite_momentum)), 1)

        # Trend Determination
        trend_delta = 0.0
        if self._momentum_history:
            prev_momentum = self._momentum_history[-1]
            trend_delta = round(composite_momentum - prev_momentum, 1)

        if trend_delta >= cfg.trend_advancing_delta:
            trend = "advancing"
        elif trend_delta <= cfg.trend_regressing_delta:
            trend = "regressing"
        elif objection_score <= 35.0 or dims.readiness <= 0.35:
            trend = "stalling"
        else:
            trend = "stable"

        self._momentum_history.append(composite_momentum)

        families = {
            "problem_goal_clarity": problem_goal_score,
            "value_recognition": value_score,
            "objection_movement": objection_score,
            "trust_engagement_trend": trust_eng_score,
            "decision_structure_clarity": dec_score,
            "future_operational_behavior": future_score,
            "commitment_behavior": commit_score,
        }

        effective_conf = round(min(bundle.inference_confidence, bundle.semantic_confidence), 3)

        return MomentumBreakdown(
            momentum_score=composite_momentum,
            trend=trend,
            trend_delta=trend_delta,
            family_scores=families,
            confidence=effective_conf,
        )

    def compute_readiness(
        self,
        bundle: BehavioralSignalInputBundle,
        current_state: ConversationStateSnapshot,
        conversion_target: str = "appointment",
        blocking_config: Optional[ConversionBlockingConfig] = None,
    ) -> ReadinessBreakdown:
        """Computes multi-dimensional readiness with deterministic blocker caps."""
        cfg = self.config
        dims = current_state.dimensions

        # ---------------------------------------------------------------------
        # 1. Sub-Dimensions (0 - 100)
        # ---------------------------------------------------------------------
        # A. Emotional Readiness: High trust, positive valence, absence of tension
        norm_val = 0.5 + (0.5 * dims.emotion_valence)
        inv_tension = 1.0 - dims.emotion_tension
        emotional = round((0.4 * dims.trust + 0.3 * norm_val + 0.3 * inv_tension) * 100.0, 1)
        emotional = min(100.0, max(0.0, emotional))

        # B. Logical Readiness: Value alignment, agreement, problem clarity
        agreement_pts = bundle.agreement_score * 100.0
        logical = round(0.5 * agreement_pts + 0.5 * (dims.engagement * 100.0), 1)
        logical = min(100.0, max(0.0, logical))

        # C. Logistical Readiness: Timeline feasibility, lack of scheduling/channel/access restriction
        comp = current_state.contact_compliance
        has_active_timeline = any(f.category == "timeline" and f.status == "active" for f in current_state.facts)
        has_channel_block = comp.contact_preference == "channel_restriction"
        has_timing_block = comp.contact_preference == "timing_restriction"
        has_access_constraints = bool(current_state.decision_structure.access_constraints)

        # Collect prospect evidence turn IDs
        prospect_ev_turn_ids: List[int] = []
        if bundle.speaker_id == "client":
            prospect_ev_turn_ids.append(bundle.turn_id)
        if hasattr(current_state, "prospect_turn_ids"):
            for pt in current_state.prospect_turn_ids:
                if pt not in prospect_ev_turn_ids:
                    prospect_ev_turn_ids.append(pt)
        for f in current_state.facts:
            if getattr(f, "source_turn_id", 0) > 0 and f.source_turn_id not in prospect_ev_turn_ids:
                prospect_ev_turn_ids.append(f.source_turn_id)
        for o in current_state.objections:
            if getattr(o, "source_turn_id", 0) > 0 and o.source_turn_id not in prospect_ev_turn_ids:
                prospect_ev_turn_ids.append(o.source_turn_id)

        has_prospect_spoken = len(prospect_ev_turn_ids) > 0 or getattr(current_state, "has_prospect_spoken", False)

        # ---------------------------------------------------------------------
        # 1. Per-Dimension Evidence & Scoring (0 - 100 or None if unmeasured)
        # ---------------------------------------------------------------------
        comp = current_state.contact_compliance
        dec = current_state.decision_structure

        # A. Emotional Readiness: High trust, positive valence, absence of tension
        # Evaluated strictly on observed sub-components with evidence (no constant 50 defaults)
        emotional_sub_components = []
        has_trust_ev = (
            getattr(bundle.trust, "is_measured", False)
            or bool(bundle.trust.drivers)
            or bool(bundle.trust.contributing_evidence_ids)
            or abs(dims.trust - 0.50) > 0.05
        )
        if has_trust_ev:
            emotional_sub_components.append((dims.trust * 100.0, 0.40))

        has_valence_ev = (
            getattr(bundle.emotion, "is_measured", False)
            or bool(bundle.emotion.observable_signals)
            or abs(dims.emotion_valence) > 0.05
        )
        if has_valence_ev:
            norm_val = (0.5 + 0.5 * dims.emotion_valence) * 100.0
            emotional_sub_components.append((norm_val, 0.30))

        has_tension_ev = (
            getattr(bundle.emotion, "is_measured", False)
            or bool(bundle.emotion.observable_signals)
            or dims.emotion_tension > 0.25
            or dims.emotion_tension < 0.15
            or bool(current_state.objections)
        )
        if has_tension_ev:
            inv_tension = (1.0 - dims.emotion_tension) * 100.0
            emotional_sub_components.append((inv_tension, 0.30))

        if has_prospect_spoken and emotional_sub_components:
            total_w = sum(w for _, w in emotional_sub_components)
            emotional: Optional[float] = round(sum(val * (w / total_w) for val, w in emotional_sub_components), 1)
            emotional = min(100.0, max(0.0, emotional))
        else:
            emotional = None

        # B. Logical Readiness: Value alignment, agreement, problem clarity
        # Evaluated strictly on observed sub-components with evidence (no constant 50 defaults)
        logical_sub_components = []
        has_agreement_ev = (
            getattr(bundle, "agreement_measured", False)
            or abs(bundle.agreement_score - 0.50) > 0.05
        )
        if has_agreement_ev:
            logical_sub_components.append((bundle.agreement_score * 100.0, 0.40))

        has_engagement_ev = (
            getattr(bundle.engagement, "is_measured", False)
            or bool(bundle.engagement.drivers)
            or abs(dims.engagement - 0.50) > 0.05
        )
        if has_engagement_ev:
            logical_sub_components.append((dims.engagement * 100.0, 0.30))

        has_spec_or_future_ev = (
            getattr(bundle, "specificity_measured", False)
            or getattr(bundle, "future_language_measured", False)
            or abs(bundle.specificity_score - 0.50) > 0.05
            or bundle.future_language_score > 0.20
            or any(f.category in ("problem", "financial", "property") and f.status == "active" for f in current_state.facts)
        )
        if has_spec_or_future_ev:
            spec_val = bundle.specificity_score * 100.0
            if bundle.future_language_score > 0.20:
                spec_val = (bundle.specificity_score * 0.50 + bundle.future_language_score * 0.50) * 100.0
            logical_sub_components.append((spec_val, 0.30))

        if has_prospect_spoken and logical_sub_components:
            total_w = sum(w for _, w in logical_sub_components)
            logical: Optional[float] = round(sum(val * (w / total_w) for val, w in logical_sub_components), 1)
            logical = min(100.0, max(0.0, logical))
        else:
            logical = None

        # C. Logistical Readiness: Meeting/Appointment feasibility & scheduling specificity
        # Measured only when prospect has engaged on appointment scheduling, timings, or constraints
        has_confirmed_meeting = (
            any(f.fact_key == "confirmed_meeting_time" and f.status == "active" for f in current_state.facts)
            or bool(current_state.conversion_event and getattr(current_state.conversion_event, "status", "") in ("confirmed", ConversionEventStatus.CONFIRMED))
        )
        has_tentative_meeting = any(f.fact_key in ("tentative_meeting_time", "walkthrough_timing") and f.status == "active" for f in current_state.facts)
        has_access_constraints = bool(current_state.decision_structure.access_constraints)
        has_scheduling_constraint = any(f.fact_key == "scheduling_constraint" and f.status == "active" for f in current_state.facts)
        has_channel_block = comp.contact_preference == "channel_restriction"
        has_timing_block = comp.contact_preference == "timing_restriction"

        has_logistical_facts = any(
            f.category in ("timeline", "logistical")
            and f.fact_key != "timeline_horizon"
            and f.status == "active"
            for f in current_state.facts
        )
        has_logistical_utterance = (
            bundle.speaker_id == "client"
            and any(k in bundle.utterance_text.lower() for k in ("schedule", "appointment", "walkthrough", "meet", "calendar", "coordinate", "tomorrow at", "thursday at", "friday at", "morning at", "afternoon at"))
            and not any(k in bundle.utterance_text.lower() for k in ("next year", "sometime next year", "doesn't work", "does not work", "won't work", "cannot work", "can't make it", "hypothetical"))
        )

        # Explicit appointment logistics indicator (selling timeline 'timeline_horizon' is NOT appointment logistics; contact preferences are not appointment logistics)
        has_logistical_evidence = has_prospect_spoken and (
            has_confirmed_meeting
            or has_tentative_meeting
            or has_access_constraints
            or has_scheduling_constraint
            or comp.hard_boundary_active
            or has_channel_block
            or has_timing_block
            or has_logistical_facts
            or has_logistical_utterance
        )

        if has_logistical_evidence:
            if has_confirmed_meeting:
                logistical_pts = 90.0
            elif has_tentative_meeting:
                logistical_pts = 80.0
            else:
                logistical_pts = 70.0
            if has_channel_block:
                logistical_pts -= 35.0
            if has_timing_block:
                logistical_pts -= 25.0
            if has_access_constraints and not has_confirmed_meeting:
                logistical_pts -= 15.0
            if comp.hard_boundary_active:
                logistical_pts = 0.0
            logistical: Optional[float] = round(min(100.0, max(0.0, logistical_pts)), 1)
        else:
            logistical = None

        # D. Decision Readiness: Authority identified, present, aligned
        has_absent_spouse_or_stakeholder = any(
            s.role in ("spouse", "partner", "co-owner", "co_owner", "attorney", "wife", "husband") and s.presence == "absent"
            for s in dec.stakeholders
        )
        spouse_facts = [f for f in current_state.facts if f.fact_key in ("spouse_involvement", "decision_maker_authority") and f.status == "active"]
        for sf in spouse_facts:
            val_lower = sf.fact_value.lower()
            if any(w in val_lower for w in ["must be present", "handles the decisions", "consult", "wife handles", "husband handles", "talk to my wife"]):
                spouse_on_call = any(s.role in ("spouse", "partner", "wife", "husband") and s.presence in ("on_call", "confirmed_attending") for s in dec.stakeholders)
                if not spouse_on_call:
                    has_absent_spouse_or_stakeholder = True

        has_decision_evidence = has_prospect_spoken and (
            dec.primary_decision_maker is not None
            or has_absent_spouse_or_stakeholder
            or bool(dec.stakeholders)
            or bool(spouse_facts)
            or any(f.category == "decision_maker" and f.status == "active" for f in current_state.facts)
        )

        if has_decision_evidence:
            if not dec.decision_maker_present or has_absent_spouse_or_stakeholder:
                decision_pts = 40.0
            elif dec.primary_decision_maker:
                decision_pts = 90.0
            else:
                decision_pts = 50.0
            decision_readiness: Optional[float] = round(min(100.0, max(0.0, decision_pts)), 1)
        else:
            decision_readiness = None

        # ---------------------------------------------------------------------
        # 2. Composite Uncapped Readiness Mean (Measured Dimensions Only)
        # ---------------------------------------------------------------------
        dim_weights = {
            "emotional": cfg.emotional_readiness_weight,
            "logical": cfg.logical_readiness_weight,
            "logistical": cfg.logistical_readiness_weight,
            "decision": cfg.decision_readiness_weight,
        }
        measured = [
            (name, val, dim_weights[name])
            for name, val in [
                ("emotional", emotional),
                ("logical", logical),
                ("logistical", logistical),
                ("decision", decision_readiness),
            ]
            if val is not None
        ]

        total_measured_weight = sum(w for _, _, w in measured)
        coverage = round(total_measured_weight / 1.0, 3)

        is_insufficient = (not has_prospect_spoken) or (len(measured) == 0)
        active_blockers: List[str] = []
        applicable_ceilings: Dict[str, float] = {}
        blocker_descriptions: List[str] = []

        if is_insufficient:
            uncapped = None
            capped = None
            readiness_partial = None
            readiness_score_val = None
            effective_conf = 0.0
            binding_note = "Readiness unknown (clean slate baseline lacks prospect-originated evidence)."
            active_blockers.append("insufficient_evidence")
            blocker_descriptions.append("Insufficient prospect-originated evidence for readiness assessment (readiness unknown)")
        else:
            uncapped = round(sum(val * (w / total_measured_weight) for _, val, w in measured), 1)
            uncapped = min(100.0, max(0.0, uncapped))
            readiness_partial = uncapped
            # Confidence directly reflects proportion of measured dimensions
            raw_conf = min(bundle.inference_confidence, bundle.semantic_confidence)
            effective_conf = round(raw_conf * (total_measured_weight / 1.0), 3)

            # -----------------------------------------------------------------
            # 3. Deterministic Blocker Caps (Full Census + Strictest-Wins Monotonic Min)
            # -----------------------------------------------------------------
            # Blocker 1: Hard Compliance Boundary
            if comp.hard_boundary_active:
                active_blockers.append("hard_boundary")
                applicable_ceilings["hard_boundary"] = 0.0
                blocker_descriptions.append("Hard compliance boundary active (ceiling 0.0)")

            # Blocker 2: Absent / Unaligned Decision Maker
            is_dm_absent = (not dec.decision_maker_present) or has_absent_spouse_or_stakeholder
            if not comp.hard_boundary_active and is_dm_absent:
                active_blockers.append("absent_decision_maker")
                applicable_ceilings["absent_decision_maker"] = cfg.absent_decision_maker_ceiling
                blocker_descriptions.append(f"Absent/unconfirmed decision maker (ceiling {cfg.absent_decision_maker_ceiling:.0f})")

            # Blocker 3: Logistical Deficit Blocker (only applies if logistical readiness was measured)
            if not comp.hard_boundary_active and logistical is not None and logistical <= cfg.logistical_deficit_threshold:
                active_blockers.append("logistical_deficit")
                applicable_ceilings["logistical_deficit"] = cfg.logistical_deficit_ceiling
                blocker_descriptions.append(
                    f"Logistical readiness deficit ({logistical:.0f} <= {cfg.logistical_deficit_threshold:.0f}, ceiling {cfg.logistical_deficit_ceiling:.0f})"
                )

            # Blocker 4: Active Target-Blocking Objection Blocker (Goal-Aware Gating Alignment)
            b_cfg = blocking_config or DEFAULT_CONVERSION_BLOCKING_CONFIG
            target_blocking_cats = b_cfg.blocking_categories.get(conversion_target, ["boundary"])
            non_esc_cats = getattr(b_cfg, "non_escalating_categories", {}).get(conversion_target, ["commission_fee"])

            blocking_objs = []
            for o in current_state.objections:
                if o.lifecycle_state in ("unresolved", "reactivated", "active"):
                    if o.canonical_category in target_blocking_cats:
                        blocking_objs.append(o)
                    elif (
                        getattr(b_cfg, "escalate_on_recurrence", True)
                        and o.canonical_category not in non_esc_cats
                        and getattr(o, "recurrence_count", 1) > getattr(b_cfg, "max_non_blocking_recurrence", 2)
                    ):
                        blocking_objs.append(o)

            if not comp.hard_boundary_active and blocking_objs:
                active_blockers.append("unresolved_objection")
                applicable_ceilings["unresolved_objection"] = cfg.unresolved_objection_ceiling
                blocker_descriptions.append(f"Active unresolved objection (ceiling {cfg.unresolved_objection_ceiling:.0f})")

            # Strictest-Wins Resolution:
            if applicable_ceilings:
                strictest_blocker = min(applicable_ceilings, key=applicable_ceilings.get)
                strictest_ceiling = applicable_ceilings[strictest_blocker]
                capped = min(uncapped, strictest_ceiling)
                binding_note = f"Readiness capped at {capped:.0f} by strictest blocker ({strictest_blocker}). Active blockers: {', '.join(blocker_descriptions)}."
            else:
                capped = uncapped
                binding_note = None

            if capped is not None:
                capped = round(min(100.0, max(0.0, capped)), 1)

            # Minimum Coverage Rule (Client Feedback Item 2):
            # If fewer than 2 dimensions measured (coverage < 0.50), do NOT publish overall readiness_score.
            # Expose it as readiness_partial with coverage metadata to avoid overstating from a single isolated dimension.
            if coverage < 0.50:
                readiness_score_val = None
                is_insufficient = True
                coverage_note = f"Readiness unconfirmed: low dimensional coverage ({len(measured)} of 4 dimensions measured, coverage {coverage:.2f} < 0.50). Partial score: {uncapped:.1f}."
                binding_note = f"{binding_note} | {coverage_note}" if binding_note else coverage_note
                active_blockers.append("low_coverage")
            else:
                readiness_score_val = capped

        return ReadinessBreakdown(
            readiness_score=readiness_score_val,
            readiness_partial=readiness_partial,
            coverage=coverage,
            uncapped_score=uncapped,
            emotional_readiness=emotional,
            logical_readiness=logical,
            logistical_readiness=logistical,
            decision_readiness=decision_readiness,
            active_blocker_caps=active_blockers,
            capped_reason=binding_note,
            confidence=effective_conf,
            insufficient_evidence=is_insufficient,
            evidence_turn_ids=prospect_ev_turn_ids,
        )
