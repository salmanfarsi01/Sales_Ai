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
