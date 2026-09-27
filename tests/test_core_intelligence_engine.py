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
    PushStrengthRecommendation,
    StrategyAttemptOutcome,
)
from copilot.core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
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


def test_objection_recurrence_branching_and_no_repetition():
    engine = PitchProXCoreIntelligenceEngine()

    # Recurrence 1: Initial surface objection
    obj_rec1 = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your fee is too high.",
        latest_statement="Your fee is too high.",
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        first_turn_id=3,
        last_updated_turn_id=3,
        recurrence_count=1,
    )
    snapshot1 = ConversationStateSnapshot(
        call_sid="call_test_04",
        state_version=4,
        objections=[obj_rec1],
    )
    result1 = engine.evaluate(snapshot1)
    assert result1.decision.primary_action == StrategicAction.VALIDATE
    assert result1.decision.secondary_action == StrategicAction.CLARIFY
    assert "OBJECTION_SURFACE_INITIAL" in result1.decision.reason_codes

    # Recurrence 2: Shift angle to net proceeds reframe
    obj_rec2 = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your fee is too high.",
        latest_statement="I still think 6 percent is too much.",
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        first_turn_id=3,
        last_updated_turn_id=7,
        recurrence_count=2,
    )
    snapshot2 = ConversationStateSnapshot(
        call_sid="call_test_04",
        state_version=8,
        objections=[obj_rec2],
    )
    result2 = engine.evaluate(snapshot2)
    assert result2.decision.primary_action == StrategicAction.REFRAME
    assert result2.decision.secondary_action == StrategicAction.QUANTIFY
    assert "OBJECTION_PUSHBACK_SHIFT_ANGLE" in result2.decision.reason_codes

    # Recurrence 3: Previous strategy failed; branch to de-risk without reusing failed strategy
    failed_outcome = StrategyAttemptOutcome(
        strategy_tag="financial_net_proceeds_reframe",
        attempted_at_turn_id=8,
        prospect_response_turn_id=9,
        prospect_response_summary="Prospect remained resistant to net sheet breakdown",
        effectiveness="insufficient",
    )
    obj_rec3 = ObjectionRecord(
        canonical_category="commission_fee",
        initial_statement="Your fee is too high.",
        latest_statement="I don't care about your net sheet, 6 percent is too high.",
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        first_turn_id=3,
        last_updated_turn_id=11,
        recurrence_count=3,
        strategy_outcomes=[failed_outcome],
    )
    snapshot3 = ConversationStateSnapshot(
        call_sid="call_test_04",
        state_version=12,
        objections=[obj_rec3],
    )
    result3 = engine.evaluate(snapshot3)
    assert result3.decision.primary_action == StrategicAction.DE_RISK
    assert "OBJECTION_REPEATED_PROGRESSION_BRANCH" in result3.decision.reason_codes
    assert "financial_net_proceeds_reframe" in result3.decision.do_not_do


def test_meeting_gate_open_direct_ask_and_two_window():
    engine = PitchProXCoreIntelligenceEngine()

    snapshot_two_window = ConversationStateSnapshot(
        call_sid="call_test_05",
        state_version=20,
        conversation_stage=ConversationStage.SCHEDULING,
        conversion_gate=MeetingConversionGate(is_open=True, status="open"),
        push_strength=PushStrengthRecommendation(
            state="two_window_choice",
            rationale="Gate open, prospect agreeable",
            recommended_action="two_window_choice",
        ),
    )
    res_two_window = engine.evaluate(snapshot_two_window)
    assert res_two_window.decision.primary_action == StrategicAction.COMMITMENT_CLOSE
    assert res_two_window.decision.secondary_action == StrategicAction.QUESTION
    assert "TWO_WINDOW_CHOICE" in res_two_window.decision.reason_codes

    snapshot_direct = ConversationStateSnapshot(
        call_sid="call_test_05",
        state_version=21,
        conversation_stage=ConversationStage.SCHEDULING,
        conversion_gate=MeetingConversionGate(is_open=True, status="open"),
        push_strength=PushStrengthRecommendation(
            state="direct_ask",
            rationale="High readiness, clear path",
            recommended_action="direct_ask",
        ),
    )
    res_direct = engine.evaluate(snapshot_direct)
    assert res_direct.decision.primary_action == StrategicAction.COMMITMENT_CLOSE
    assert res_direct.decision.secondary_action is None
    assert "DIRECT_ASK" in res_direct.decision.reason_codes


def test_playbook_and_calibration_modifiers():
    modifier = StrategyModifierPipeline()
    base_decision = StrategicDecision(
        call_id="call_test_06",
        source_state_version=14,
        strategic_objective="Demonstrate market differentiation",
        primary_action=StrategicAction.REFRAME,
        secondary_action=None,
        max_prompt_words=24,
    )

    playbook = {"id": "pb_challenger_01", "methodology": "challenger"}
    calibration = {"brevity_mode": "concise", "target_wpm": 125}

    modified = modifier.apply_modifiers(base_decision, playbook=playbook, calibration=calibration)

    assert modified.secondary_action == StrategicAction.CHALLENGE
    assert "PLAYBOOK_CHALLENGER_EMPHASIS" in modified.reason_codes
    assert modified.max_prompt_words <= 16
    assert modified.calibration_influence["brevity_mode"] == "concise"


def test_core_decision_manager_stale_invalidation():
    manager = CoreDecisionManager(call_sid="call_test_07")
    snapshot_v1 = ConversationStateSnapshot(
        call_sid="call_test_07",
        state_version=5,
        conversation_stage=ConversationStage.DISCOVERY,
    )

    eval_result = manager.evaluate_state(snapshot_v1)
    decision_v1 = eval_result.decision
    assert decision_v1.source_state_version == 5

    # Same state version is not stale
    assert not manager.is_decision_stale(decision_v1, snapshot_v1)

    # State version advances with material change (e.g. boundary triggered)
    snapshot_v2 = snapshot_v1.model_copy(deep=True)
    snapshot_v2.state_version = 6
    snapshot_v2.contact_compliance.hard_boundary_active = True

    assert manager.is_decision_stale(decision_v1, snapshot_v2)
