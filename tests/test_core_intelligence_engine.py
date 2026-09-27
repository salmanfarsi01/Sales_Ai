import pytest

from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversationStage,
    ContactComplianceState,
    ConversionEventObject,
    ConversionEventStatus,
    DecisionStructure,
    DecisionStakeholder,
    DimensionScores,
    MeetingConversionGate,
    ObjectionRecord,
    ObjectionLifecycleState,
    ObjectionDriverLayer,
    PushStrengthRecommendation,
    StrategyAttemptOutcome,
)
from copilot.core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
    RequiredFactScope,
    Spec01ObjectionLadderStage,
)
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_strategy_modifiers import StrategyModifierPipeline
from copilot.core_decision_manager import CoreDecisionManager


def test_hard_boundary_priority():
    engine = PitchProXCoreIntelligenceEngine()
    snapshot = ConversationStateSnapshot(
        call_sid="call_test_01",
        state_version=10,
        contact_compliance=ContactComplianceState(
            hard_boundary_active=True,
            hard_boundary_reason="Prospect requested not to be called",
        ),
    )

    result = engine.evaluate(snapshot)
    decision = result.decision

    assert decision.primary_action == StrategicAction.ACKNOWLEDGE
    assert decision.secondary_action == StrategicAction.WAIT_SILENCE
    assert decision.push_strength == "respect_record_exit"
    assert not decision.question_allowed
    assert "HARD_BOUNDARY_ACTIVE" in decision.reason_codes
    assert "persuade" in decision.do_not_do


def test_confirm_and_protect_preserves_confirmed_appointment():
    engine = PitchProXCoreIntelligenceEngine()
    conv_event = ConversionEventObject(
        conversion_type="property_walkthrough",
        status=ConversionEventStatus.CONFIRMED,
        start_at="Thursday at 3:00 PM",
        confirmation_confidence=0.95,
    )
    snapshot = ConversationStateSnapshot(
        call_sid="call_test_02",
        state_version=25,
        conversion_event=conv_event,
        conversion_events=[conv_event],
        push_strength=PushStrengthRecommendation(
            state="confirm_and_protect",
            rationale="Appointment confirmed",
            recommended_action="confirm_and_protect",
        ),
    )

    result = engine.evaluate(snapshot)
    decision = result.decision

    assert decision.primary_action == StrategicAction.ACKNOWLEDGE
    assert decision.push_strength == "confirm_and_protect"
    assert "CONVERSION_CONFIRMED" in decision.reason_codes
    assert "reopen_resolved_objections" in decision.do_not_do
    assert "confirmed_appointment" in decision.what_to_protect


def test_absent_decision_maker_blocks_premature_close():
    engine = PitchProXCoreIntelligenceEngine()
    stakeholder = DecisionStakeholder(role="spouse", presence="absent")
    snapshot = ConversationStateSnapshot(
        call_sid="call_test_03",
        state_version=15,
        conversation_stage=ConversationStage.VALUE_WALKTHROUGH,
        decision_structure=DecisionStructure(
            primary_decision_maker="Prospect and Spouse",
            decision_maker_present=False,
            stakeholders=[stakeholder],
        ),
        dimensions=DimensionScores(trust=0.75, engagement=0.8, readiness=0.5),
    )

    result = engine.evaluate(snapshot)
    decision = result.decision

    assert decision.primary_action == StrategicAction.DE_RISK
    assert "DECISION_MAKER_ABSENT" in decision.reason_codes
    assert "press_for_single_party_commitment" in decision.do_not_do
    assert "collaborative_buy_in" in decision.what_to_protect


def test_spec01_objection_ladder_canonical_stages():
    """Validates Spec 01 Section 6 canonical 6-stage objection depth ladder."""
    engine = PitchProXCoreIntelligenceEngine()

    # 1. Surface objection
    obj_surface = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your commission is too high.",
        latest_statement="Your commission is too high.",
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        first_turn_id=2,
        last_updated_turn_id=2,
        recurrence_count=1,
    )
    res_surface = engine.evaluate(ConversationStateSnapshot(call_sid="call_ladder", state_version=3, objections=[obj_surface]))
    assert "LADDER_SURFACE_OBJECTION" in res_surface.decision.reason_codes
    assert res_surface.decision.primary_action == StrategicAction.VALIDATE

    # 2. Underlying concern (with driver layer)
    driver = ObjectionDriverLayer(
        surface_objection="commission_fee",
        underlying_driver="previous_agent_outcome",
        strategic_target="demonstrate_accountability",
    )
    obj_driver = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your commission is too high.",
        latest_statement="Last agent charged 6% and did nothing.",
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        first_turn_id=2,
        last_updated_turn_id=4,
        recurrence_count=2,
        driver_layer=driver,
    )
    res_driver = engine.evaluate(ConversationStateSnapshot(call_sid="call_ladder", state_version=5, objections=[obj_driver]))
    assert "LADDER_UNDERLYING_CONCERN" in res_driver.decision.reason_codes
    assert res_driver.decision.primary_action == StrategicAction.MIRROR

    # 3. First pushback (without driver, shifts angle to reframe/quantify)
    obj_pushback = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your commission is too high.",
        latest_statement="I still think 6% is too much.",
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        first_turn_id=2,
        last_updated_turn_id=6,
        recurrence_count=2,
    )
    res_pushback = engine.evaluate(ConversationStateSnapshot(call_sid="call_ladder", state_version=7, objections=[obj_pushback]))
    assert "LADDER_FIRST_PUSHBACK" in res_pushback.decision.reason_codes
    assert res_pushback.decision.primary_action == StrategicAction.REFRAME

    # 4. Repeated resistance (Recurrence 3+)
    obj_repeat = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your commission is too high.",
        latest_statement="I won't pay 6%, period.",
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        first_turn_id=2,
        last_updated_turn_id=8,
        recurrence_count=3,
        strategy_outcomes=[
            StrategyAttemptOutcome(
                strategy_tag="financial_net_proceeds_reframe",
                attempted_at_turn_id=6,
                prospect_response_turn_id=7,
                prospect_response_summary="Rejected net sheet comparison",
                effectiveness="rejected",
            )
        ],
    )
    res_repeat = engine.evaluate(ConversationStateSnapshot(call_sid="call_ladder", state_version=9, objections=[obj_repeat]))
    assert "LADDER_REPEATED_RESISTANCE" in res_repeat.decision.reason_codes
    assert res_repeat.decision.primary_action == StrategicAction.DE_RISK
    assert "financial_net_proceeds_reframe" in res_repeat.decision.do_not_do

    # 5. Partial resolution
    obj_partial = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your commission is too high.",
        latest_statement="Okay I see the net sheet makes sense, but what about cancellation?",
        lifecycle_state=ObjectionLifecycleState.PARTIALLY_ADDRESSED,
        first_turn_id=2,
        last_updated_turn_id=10,
        recurrence_count=3,
    )
    res_partial = engine.evaluate(ConversationStateSnapshot(call_sid="call_ladder", state_version=11, objections=[obj_partial]))
    assert "LADDER_PARTIAL_RESOLUTION" in res_partial.decision.reason_codes
    assert res_partial.decision.primary_action == StrategicAction.ACKNOWLEDGE


def test_spec10_no_false_precision_dynamic_confidence():
    """Validates Spec 10 Section 2 and 7 dynamic confidence calculation without hardcoded static default."""
    engine = PitchProXCoreIntelligenceEngine()
    obj = ObjectionRecord(
        canonical_category="price",
        initial_statement="Price is too high",
        latest_statement="Price is too high",
        confidence=0.72,
        first_turn_id=1,
        last_updated_turn_id=1,
    )
    snapshot = ConversationStateSnapshot(
        call_sid="call_conf",
        state_version=2,
        objections=[obj],
        dimensions=DimensionScores(trust=0.60, trust_confidence=0.78),
    )

    result = engine.evaluate(snapshot)
    decision = result.decision

    assert decision.confidence != 0.85
    assert "base_source" in decision.confidence_breakdown
    assert decision.confidence_breakdown["base_source"] == 0.72
    assert decision.confidence_breakdown["trust_confidence"] == 0.78
    assert decision.confidence == round((0.72 * 0.70) + (0.78 * 0.30), 3)


def test_spec10_cross_metric_consistency_rules():
    """Validates Spec 10 Section 8 cross-metric consistency checks."""
    engine = PitchProXCoreIntelligenceEngine()

    # Case A: Friendly prospect (high valence) but momentum falling
    snapshot_friendly = ConversationStateSnapshot(
        call_sid="call_cross_metric",
        state_version=8,
        dimensions=DimensionScores(trust=0.70, emotion_valence=0.45, momentum=0.30),
        momentum=None,
    )
    from copilot.conversation_state_models import MomentumBreakdown
    snapshot_friendly.momentum = MomentumBreakdown(momentum_score=30.0, trend="regressing")
    res_friendly = engine.evaluate(snapshot_friendly)

    assert "CROSS_METRIC_VALENCE_MOMENTUM_MISMATCH" in res_friendly.decision.reason_codes
    assert "premature_close" in res_friendly.decision.do_not_do
    assert "FRIENDLINESS_IS_NOT_PROGRESS" in res_friendly.context.cross_metric_penalties

    # Case B: High engagement with unresolved objection
    obj = ObjectionRecord(
        canonical_category="representation_broker",
        initial_statement="I already have an agent friend.",
        latest_statement="I already have an agent friend.",
        first_turn_id=2,
        last_updated_turn_id=2,
    )
    snapshot_engaged = ConversationStateSnapshot(
        call_sid="call_cross_metric_2",
        state_version=10,
        objections=[obj],
        dimensions=DimensionScores(engagement=0.85, trust=0.60),
    )
    res_engaged = engine.evaluate(snapshot_engaged)

    assert "premature_close" in res_engaged.decision.do_not_do
    assert "OBJECTION_PREVENTS_BLIND_CLOSE" in res_engaged.decision.reason_codes
    assert "HIGH_ENGAGEMENT_WITH_UNRESOLVED_OBJECTION" in res_engaged.context.cross_metric_penalties


def test_spec11_two_tiered_staleness_guard():
    """Validates Spec 11 Section 5 and 10 two-tiered staleness checkpoints."""
    manager = CoreDecisionManager(call_sid="call_staleness")
    snapshot_v1 = ConversationStateSnapshot(
        call_sid="call_staleness",
        state_version=5,
        conversation_stage=ConversationStage.DISCOVERY,
    )

    eval_result = manager.evaluate_state(snapshot_v1)
    decision = eval_result.decision
    assert decision.source_state_version == 5

    # Checkpoint 1 (Pre-dispatch): state version matches
    assert not manager.is_decision_stale(decision, snapshot_v1)

    # Checkpoint 2 (Pre-display): valid at current state
    is_valid, reason = manager.validate_for_teleprompter_display(decision.source_state_version, snapshot_v1)
    assert is_valid
    assert reason is None

    # Advance state with material update (e.g. hard boundary)
    snapshot_v2 = snapshot_v1.model_copy(deep=True)
    snapshot_v2.state_version = 6
    snapshot_v2.contact_compliance.hard_boundary_active = True

    # Checkpoint 1 flags stale
    assert manager.is_decision_stale(decision, snapshot_v2)

    # Checkpoint 2 cancels display
    is_valid_disp, cancel_reason = manager.validate_for_teleprompter_display(decision.source_state_version, snapshot_v2)
    assert not is_valid_disp
    assert cancel_reason == "HARD_BOUNDARY_SUPERSEDED_PROMPT"


def test_spec11_required_facts_granularity():
    """Validates Spec 11 Section 8 structured RequiredFactScope generation."""
    modifier = StrategyModifierPipeline()
    base_decision = StrategicDecision(
        call_id="call_facts",
        source_state_version=10,
        strategic_objective="Quantify financial net sheet ROI",
        primary_action=StrategicAction.QUANTIFY,
        max_prompt_words=24,
    )

    modified = modifier.apply_modifiers(base_decision)
    assert modified.retrieval_needed
    assert len(modified.required_facts) == 2
    assert all(isinstance(f, RequiredFactScope) for f in modified.required_facts)
    assert modified.required_facts[0].topic == "market_comps"
    assert modified.required_facts[0].verification_required is True
    assert modified.required_facts[1].required_evidence_type == "roi_calculator"
