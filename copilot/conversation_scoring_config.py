from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ConversationScoringConfig(BaseModel):
    """Named, versioned, documented configuration for ConversationState scoring (Phase 6).

    Reuses the InferenceScoringConfig pattern from the Behavioral Signal Engine.
    All weights are normalized and explicitly documented.
    """
    config_version: str = "1.0.0"

    # -------------------------------------------------------------------------
    # 1. Momentum Evidence Family Weights (Spec Document: Sum = 100%)
    # -------------------------------------------------------------------------
    problem_goal_clarity_weight: float = Field(0.15, ge=0.0, le=1.0, description="15%: Clarity on goals/problems")
    value_recognition_weight: float = Field(0.15, ge=0.0, le=1.0, description="15%: Value and strategic fit acknowledgement")
    objection_movement_weight: float = Field(0.20, ge=0.0, le=1.0, description="20%: Resolution progress on objections")
    trust_engagement_trend_weight: float = Field(0.15, ge=0.0, le=1.0, description="15%: Trajectory of trust and topic engagement")
    decision_structure_clarity_weight: float = Field(0.10, ge=0.0, le=1.0, description="10%: Clarity of decision makers & authority")
    future_operational_behavior_weight: float = Field(0.15, ge=0.0, le=1.0, description="15%: Operational planning, calendar language")
    commitment_behavior_weight: float = Field(0.10, ge=0.0, le=1.0, description="10%: Agreement, forward commitments, next steps")

    # -------------------------------------------------------------------------
    # 2. Readiness Dimension Weights (Base Composite Mean: Sum = 100%)
    # -------------------------------------------------------------------------
    emotional_readiness_weight: float = Field(0.25, ge=0.0, le=1.0, description="Emotional safety, trust, low tension")
    logical_readiness_weight: float = Field(0.25, ge=0.0, le=1.0, description="Value alignment, clear rationale, problem urgency")
    logistical_readiness_weight: float = Field(0.25, ge=0.0, le=1.0, description="Timeline viability, scheduling feasibility")
    decision_readiness_weight: float = Field(0.25, ge=0.0, le=1.0, description="Decision authority present and aligned")

    # -------------------------------------------------------------------------
    # 3. Deterministic Blocker Caps (Ceilings on 0 - 100 Scale)
    # -------------------------------------------------------------------------
    absent_decision_maker_ceiling: float = Field(
        55.0,
        ge=0.0,
        le=100.0,
        description="Caps overall readiness if key decision maker (spouse, attorney) is absent/unaligned"
    )
    logistical_deficit_threshold: float = Field(
        40.0,
        ge=0.0,
        le=100.0,
        description="Logistical readiness score below which logistical deficit blocker applies"
    )
    logistical_deficit_ceiling: float = Field(
        60.0,
        ge=0.0,
        le=100.0,
        description="Caps overall readiness when logistical feasibility is deficient"
    )
    unresolved_objection_ceiling: float = Field(
        55.0,
        ge=0.0,
        le=100.0,
        description="Caps overall readiness when active major objection remains unresolved"
    )

    # -------------------------------------------------------------------------
    # 4. Momentum Trend Delta Thresholds (Point movement across rolling window)
    # -------------------------------------------------------------------------
    trend_advancing_delta: float = Field(4.0, ge=0.0, description="Rolling momentum delta >= +4.0 classifies advancing")
    trend_regressing_delta: float = Field(-4.0, le=0.0, description="Rolling momentum delta <= -4.0 classifies regressing")

    # -------------------------------------------------------------------------
    # 5. Meeting/Conversion Gate & Push Strength Thresholds (Phase 7 / Sprint 6)
    # -------------------------------------------------------------------------
    gate_min_trust: float = Field(0.45, ge=0.0, le=1.0, description="Condition 1: Minimum trust level for open gate")
    gate_max_tension: float = Field(0.65, ge=0.0, le=1.0, description="Condition 1: Maximum emotion tension before trust collapses")
    gate_min_engagement: float = Field(0.50, ge=0.0, le=1.0, description="Condition 2: Minimum engagement on-topic")
    gate_min_value_recognition: float = Field(50.0, ge=0.0, le=100.0, description="Condition 4: Minimum value recognition score")
    gate_min_logistical_readiness: float = Field(40.0, ge=0.0, le=100.0, description="Condition 6: Minimum logistical readiness score")
    direct_ask_min_trust: float = Field(0.65, ge=0.0, le=1.0, description="Minimum trust level for direct_ask push strength")
    two_window_choice_min_agreement: float = Field(0.60, ge=0.0, le=1.0, description="Minimum agreement for agreeable-but-vague classification")
    two_window_choice_max_specificity: float = Field(0.45, ge=0.0, le=1.0, description="Maximum specificity for agreeable-but-vague classification")
    reduce_friction_logistical_upper: float = Field(55.0, ge=0.0, le=100.0, description="Upper bound for soft logistical hesitation before direct ask")

    # -------------------------------------------------------------------------
    # 6. Concern Lifecycle & Dormancy Thresholds (Provisional v1 Calibration)
    # -------------------------------------------------------------------------
    dormancy_turn_threshold: int = Field(
        3,
        ge=1,
        le=10,
        description="Provisional v1 threshold: unaddressed conversation turns before an active/partially-addressed objection transitions to DORMANT. Calibrated for standard ~1.5-2 min dialogue cadence."
    )

    @model_validator(mode="after")
    def validate_weights(self) -> ConversationScoringConfig:
        momentum_sum = (
            self.problem_goal_clarity_weight
            + self.value_recognition_weight
            + self.objection_movement_weight
            + self.trust_engagement_trend_weight
            + self.decision_structure_clarity_weight
            + self.future_operational_behavior_weight
            + self.commitment_behavior_weight
        )
        if abs(momentum_sum - 1.0) > 1e-4:
            raise ValueError(f"Momentum family weights must sum to 1.0, got {momentum_sum:.4f}")

        readiness_sum = (
            self.emotional_readiness_weight
            + self.logical_readiness_weight
            + self.logistical_readiness_weight
            + self.decision_readiness_weight
        )
        if abs(readiness_sum - 1.0) > 1e-4:
            raise ValueError(f"Readiness dimension weights must sum to 1.0, got {readiness_sum:.4f}")

        return self


DEFAULT_CONVERSATION_SCORING_CONFIG = ConversationScoringConfig()
