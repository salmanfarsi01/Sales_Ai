from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from .conversation_scoring_config import ConversationScoringConfig, DEFAULT_CONVERSATION_SCORING_CONFIG
from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
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
            if "boundary" in obj_states:
                objection_score = 0.0
            elif all(s == "resolved" for s in obj_states):
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
        elif has_primary:
            dec_score = 65.0
        else:
            dec_score = 40.0

        # 6. Future / Operational Behavior (15%)
        future_pts = bundle.future_language_score * 100.0
        has_closing_target = any(f.fact_key in ("target_closing", "move_in_date") for f in current_state.facts if f.status == "active")
        future_score = round(0.6 * future_pts + (40.0 if has_closing_target else 15.0), 1)
        future_score = min(100.0, max(0.0, future_score))

        # 7. Commitment Behavior (10%)
        pacing_pts = dims.pacing * 100.0
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

        logistical_pts = 70.0
        if has_active_timeline:
            logistical_pts += 20.0
        if has_channel_block:
            logistical_pts -= 35.0
        if has_timing_block:
            logistical_pts -= 25.0
        if has_access_constraints:
            logistical_pts -= 15.0
        if comp.hard_boundary_active:
            logistical_pts = 0.0
        logistical = round(min(100.0, max(0.0, logistical_pts)), 1)

        # D. Decision Readiness: Authority identified, present, aligned
        dec = current_state.decision_structure
        has_absent_spouse_or_stakeholder = any(
            s.role in ("spouse", "partner", "co-owner", "co_owner", "attorney") and s.presence == "absent"
            for s in dec.stakeholders
        )

        # Check for spouse / stakeholder requirement in facts or decision structure
        spouse_facts = [f for f in current_state.facts if f.fact_key in ("spouse_involvement", "decision_maker_authority") and f.status == "active"]
        for sf in spouse_facts:
            val_lower = sf.fact_value.lower()
            if any(w in val_lower for w in ["must be present", "handles the decisions", "consult", "wife handles", "husband handles", "talk to my wife"]):
                spouse_on_call = any(s.role in ("spouse", "partner") and s.presence == "on_call" for s in dec.stakeholders)
                if not spouse_on_call:
                    has_absent_spouse_or_stakeholder = True

        if not dec.decision_maker_present or has_absent_spouse_or_stakeholder:
            decision_pts = 30.0
        elif dec.primary_decision_maker:
            decision_pts = 90.0
        else:
            decision_pts = 50.0
        decision_readiness = round(min(100.0, max(0.0, decision_pts)), 1)

        # ---------------------------------------------------------------------
        # 2. Base Composite Uncapped Readiness Mean
        # ---------------------------------------------------------------------
        uncapped = (
            emotional * cfg.emotional_readiness_weight
            + logical * cfg.logical_readiness_weight
            + logistical * cfg.logistical_readiness_weight
            + decision_readiness * cfg.decision_readiness_weight
        )
        uncapped = round(min(100.0, max(0.0, uncapped)), 1)

        # ---------------------------------------------------------------------
        # 3. Deterministic Blocker Caps (Full Census + Strictest-Wins Monotonic Min)
        # ---------------------------------------------------------------------
        active_blockers: List[str] = []
        applicable_ceilings: Dict[str, float] = {}
        blocker_descriptions: List[str] = []

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

        # Blocker 3: Logistical Deficit Blocker
        if not comp.hard_boundary_active and logistical <= cfg.logistical_deficit_threshold:
            active_blockers.append("logistical_deficit")
            applicable_ceilings["logistical_deficit"] = cfg.logistical_deficit_ceiling
            blocker_descriptions.append(
                f"Logistical readiness deficit ({logistical:.0f} <= {cfg.logistical_deficit_threshold:.0f}, ceiling {cfg.logistical_deficit_ceiling:.0f})"
            )

        # Blocker 4: Active Unresolved Objection Blocker
        has_unresolved_obj = any(o.lifecycle_state == "unresolved" for o in current_state.objections)
        if not comp.hard_boundary_active and has_unresolved_obj:
            active_blockers.append("unresolved_objection")
            applicable_ceilings["unresolved_objection"] = cfg.unresolved_objection_ceiling
            blocker_descriptions.append(f"Active unresolved objection (ceiling {cfg.unresolved_objection_ceiling:.0f})")

        # Strictest-Wins Resolution:
        # Monotonic minimum across uncapped readiness and all active ceilings
        if applicable_ceilings:
            strictest_blocker = min(applicable_ceilings, key=applicable_ceilings.get)
            strictest_ceiling = applicable_ceilings[strictest_blocker]
            capped = min(uncapped, strictest_ceiling)
            binding_note = f"Readiness capped at {capped:.0f} by strictest blocker ({strictest_blocker}). Active blockers: {', '.join(blocker_descriptions)}."
        else:
            capped = uncapped
            binding_note = None

        capped = round(min(100.0, max(0.0, capped)), 1)
        effective_conf = round(min(bundle.inference_confidence, bundle.semantic_confidence), 3)

        return ReadinessBreakdown(
            readiness_score=capped,
            uncapped_score=uncapped,
            emotional_readiness=emotional,
            logical_readiness=logical,
            logistical_readiness=logistical,
            decision_readiness=decision_readiness,
            active_blocker_caps=active_blockers,
            capped_reason=binding_note,
            confidence=effective_conf,
        )
