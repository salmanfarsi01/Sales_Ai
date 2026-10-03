"""Test suite validating Group 2: Decision-model structural refinements (Points 6 to 14)
per guideline to fix issues 021026.md and client feedback.
"""
from __future__ import annotations

import pytest
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversationStage,
    ConversionEventObject,
    ConversionEventStatus,
    MeetingConversionGate,
    DecisionStructure,
    DecisionStakeholder,
    DimensionScores,
    ContactCompliance,
    ContactPreference,
    ObjectionRecord,
    ObjectionLifecycleState,
    PushStrengthRecommendation,
    PersistentFactRecord,
    StrategyAttemptOutcome,
)
from copilot.core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
    PushStrengthValue,
)
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_decision_manager import CoreDecisionManager
from copilot.llm_response_gateway import LLMResponseGateway
from copilot.conversation_replay import ConversationReplayEngine


def test_point6_split_action_posture_and_push_strength():
    """Point 6: Split Action / Posture / Push Strength into three independent fields.
    confirm_and_protect-style decisions must have push_strength: none or low, not hardcoded high,
    while maintaining strategic_posture: protect and primary_action: acknowledge.
    """
    engine = PitchProXCoreIntelligenceEngine()
    snap = ConversationStateSnapshot(
        call_sid="call_pt6_confirm_protect",
        state_version=5,
        last_updated_turn_id=18,
        conversion_confirmed=True,
        conversion_event=ConversionEventObject(
            conversion_type="property_walkthrough",
            status=ConversionEventStatus.CONFIRMED,
            start_at="Thursday at 3:00 PM",
            participants=["Client", "Wife", "Agent"],
        ),
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.98,
            commitment_slot="Thursday at 3:00 PM",
        ),
        push_strength=PushStrengthRecommendation(
            state="confirm_and_protect",
            rationale="Confirmed appointment protection mode",
            recommended_action="confirm_and_protect",
            confidence=0.98,
        ),
    )

    eval_res = engine.evaluate(snapshot=snap, turn_speaker="client", turn_text="Thursday at 3 works, and my wife will be there.", turn_id=18)
    dec = eval_res.decision

    # 1. Three independent fields exist and are populated
    assert dec.primary_action == StrategicAction.ACKNOWLEDGE
    assert dec.strategic_posture == "protect"

    # 2. Push strength is 'none' or 'low', proving protecting an appointment does not imply selling harder
    assert dec.push_strength in ("none", "low")
    assert dec.push_strength == "none"
    assert dec.push_strength != "high"

    # 3. Backward compatible legacy alias check still works
    assert dec.push_strength == "confirm_and_protect"


def test_point7_absent_decision_maker_logistics_vs_genuine_risk():
    """Point 7: Stop auto-mapping 'decision-maker changed' to DE_RISK.
    - Pure logistics (Turn 10 scenario) routes toward coordination (CLARIFY / coordinate posture).
    - Genuine deal risk (e.g. 'my wife thinks we're being scammed') routes toward DE_RISK / defend posture.
    """
    engine = PitchProXCoreIntelligenceEngine()

    # Case A: Turn 10 logistics scenario ("happy to loop her in", no hostility or suspicion)
    snap_logistics = ConversationStateSnapshot(
        call_sid="call_pt7_logistics",
        state_version=3,
        last_updated_turn_id=10,
        decision_structure=DecisionStructure(
            decision_maker_present=False,
            primary_decision_maker="Spouse involvement required",
            stakeholders=[
                DecisionStakeholder(name="Wife", role="wife", presence="absent")
            ],
            confidence=0.85,
        ),
    )
    eval_logistics = engine.evaluate(
        snapshot=snap_logistics,
        turn_speaker="client",
        turn_text="Actually, my wife would really need to be part of this conversation before we go any further.",
        turn_id=10,
    )
    dec_log = eval_logistics.decision
    assert dec_log.primary_action == StrategicAction.CLARIFY
    assert dec_log.strategic_posture == "coordinate"
    assert dec_log.push_strength in ("none", "low")
    assert "COORDINATION_LOGISTICS" in dec_log.reason_codes
    assert "GENUINE_DEAL_RISK" not in dec_log.reason_codes
    assert dec_log.primary_action != StrategicAction.DE_RISK

    # Case B: Genuine deal risk / conflict / scam scenario
    snap_risk = ConversationStateSnapshot(
        call_sid="call_pt7_risk",
        state_version=3,
        last_updated_turn_id=10,
        decision_structure=DecisionStructure(
            decision_maker_present=False,
            primary_decision_maker="Spouse opposes deal",
            stakeholders=[
                DecisionStakeholder(name="Wife", role="wife", presence="absent")
            ],
            confidence=0.85,
        ),
    )
    eval_risk = engine.evaluate(
        snapshot=snap_risk,
        turn_speaker="client",
        turn_text="My wife thinks we're being scammed by investors and told me she will refuse to sign anything.",
        turn_id=10,
    )
    dec_risk = eval_risk.decision
    assert dec_risk.primary_action == StrategicAction.DE_RISK
    assert dec_risk.strategic_posture == "defend"
    assert "GENUINE_DEAL_RISK" in dec_risk.reason_codes


def test_point8_allow_wait_hold_no_prompt_at_all():
    """Point 8: Support WAIT / HOLD / no prompt at all.
    When a decision carries should_prompt=False or HOLD, the teleprompter outputs no spoken sentence.
    """
    gateway = LLMResponseGateway()

    # Case A: should_prompt=False
    dec_no_prompt = StrategicDecision(
        call_id="call_pt8_no_prompt",
        source_state_version=2,
        source_turn_id=3,
        should_prompt=False,
        strategic_objective="Hold position and allow prospect space without intervening.",
        primary_action=StrategicAction.HOLD,
        strategic_posture="defend",
        push_strength="none",
        reason_codes=["STRATEGY_CARRIED_FORWARD"],
    )
    snap = ConversationStateSnapshot(call_sid="call_pt8_no_prompt", state_version=2)
    prompt = gateway.generate_prompt(decision=dec_no_prompt, snapshot=snap, facts=[])

    assert prompt.text == ""
    assert prompt.status == "skipped"

    # Case B: Deterministic fallback also returns empty text
    fb_text = gateway._deterministic_fallback(dec_no_prompt, snapshot=snap)
    assert fb_text == ""


def test_point9_discipline_on_primary_secondary_strategy_stacking():
    """Point 9: Discipline on strategy stacking:
    1. Maximum 1 secondary action (never 3+ stacked strategies).
    2. Whenever a secondary action is set, an explicit secondary_action_reason is required.
    3. Spot-check 18-turn benchmark for compliance.
    """
    # 1. Unit model check: secondary_action requires secondary_action_reason
    dec = StrategicDecision(
        call_id="call_pt9_stacking",
        source_state_version=1,
        strategic_objective="Test stacking discipline",
        primary_action=StrategicAction.QUESTION,
        secondary_action=StrategicAction.MIRROR,
        secondary_action_reason="Mirror statements to establish rapport.",
    )
    assert dec.secondary_action is not None
    assert dec.secondary_action_reason == "Mirror statements to establish rapport."

    # 2. Replay 18-turn benchmark and verify every decision follows stacking discipline
    turns = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, thanks for making time today — tell me a bit about what's going on with the house."},
        {"turn_id": 2, "speaker_id": "client", "text": "I'm the one making this decision, no one else needs to sign off."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Got it. What's driving the timing for you?"},
        {"turn_id": 4, "speaker_id": "client", "text": "We might look at moving sometime next year, nothing urgent yet."},
        {"turn_id": 5, "speaker_id": "client", "text": "Honestly, we're not sure this is the right time anymore."},
        {"turn_id": 6, "speaker_id": "salesperson", "text": "That's fair — a lot of people feel that way before they see the actual numbers."},
        {"turn_id": 7, "speaker_id": "client", "text": "Okay, that makes sense, I guess timing isn't the biggest issue."},
        {"turn_id": 8, "speaker_id": "salesperson", "text": "Great — would sometime next week work for a walkthrough?"},
        {"turn_id": 9, "speaker_id": "client", "text": "Maybe next week could work, let me think about it."},
        {"turn_id": 10, "speaker_id": "client", "text": "Actually, my wife would really need to be part of this conversation before we go any further."},
        {"turn_id": 11, "speaker_id": "salesperson", "text": "Of course, happy to loop her in whenever works."},
        {"turn_id": 12, "speaker_id": "client", "text": "I guess I'm just worried this isn't really the right move for us financially with everything going on."},
        {"turn_id": 13, "speaker_id": "salesperson", "text": "Totally understand — let's look at your net proceeds after all costs."},
        {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more."},
        {"turn_id": 15, "speaker_id": "client", "text": "Please don't start texting me every day before we meet, by the way."},
        {"turn_id": 16, "speaker_id": "client", "text": "Mornings don't really work for us either, just so you know."},
        {"turn_id": 17, "speaker_id": "salesperson", "text": "Noted on all of that. What day works best?"},
        {"turn_id": 18, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
    ]

    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns("sim_pt9_stacking", turns, save_report=False)
    cdm = CoreDecisionManager(call_sid="sim_pt9_stacking")

    for step in rep.timeline:
        snapshot = step.state_after
        turn_t = step.text
        speaker = step.speaker_id
        eval_res = cdm.evaluate_state(snapshot=snapshot, turn_speaker=speaker, turn_text=turn_t, turn_id=step.turn_id)
        d = eval_res.decision

        # Primary action is always present
        assert d.primary_action is not None
        # Secondary action is at most one, and if present, has explicit justification
        if d.secondary_action is not None:
            assert d.secondary_action_reason is not None and len(d.secondary_action_reason) > 5


def test_point10_confidence_downgrades_commitment_close_to_cautious_action():
    """Point 10: Confidence must change behavior, not just be a displayed number.
    Below confidence threshold (< 0.65), an otherwise closing action must downgrade to CLARIFY/QUESTION.
    """
    engine = PitchProXCoreIntelligenceEngine()

    # Case A: Normal high-confidence gate -> COMMITMENT_CLOSE
    snap_high = ConversationStateSnapshot(
        call_sid="call_pt10_high",
        state_version=4,
        conversation_stage=ConversationStage.SCHEDULING,
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.90,
            unknown_conditions=[],
        ),
        dimensions=DimensionScores(trust=0.80, trust_confidence=0.90),
    )
    eval_high = engine.evaluate(snapshot=snap_high, turn_speaker="client", turn_text="Sounds good, let's schedule.", turn_id=12)
    assert eval_high.decision.primary_action == StrategicAction.COMMITMENT_CLOSE
    assert eval_high.decision.confidence >= 0.65

    # Case B: Artificially lowered confidence (< 0.65) -> Action downgrades to CLARIFY
    snap_low = ConversationStateSnapshot(
        call_sid="call_pt10_low",
        state_version=4,
        conversation_stage=ConversationStage.SCHEDULING,
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.75,
            unknown_conditions=[],
        ),
        dimensions=DimensionScores(trust=0.20, trust_confidence=0.30),
    )
    eval_low = engine.evaluate(snapshot=snap_low, turn_speaker="client", turn_text="Sounds good, let's schedule.", turn_id=12)
    dec_low = eval_low.decision

    assert dec_low.confidence < 0.65
    assert dec_low.primary_action == StrategicAction.CLARIFY
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" in dec_low.reason_codes
    assert dec_low.strategic_posture == "explore"
    assert dec_low.push_strength in ("none", "low")


def test_point11_evidence_considered_filters_irrelevant_metrics():
    """Point 11: Evidence Considered should be relevant, not exhaustive.
    Unrelated metrics (e.g., active contact preferences) must drop out when they didn't influence the decision.
    """
    engine = PitchProXCoreIntelligenceEngine()

    # Case A: Snapshot has contact preferences, but turn is an objection on timing (compliance didn't influence decision)
    snap_irrelevant = ConversationStateSnapshot(
        call_sid="call_pt11_irrelevant",
        state_version=2,
        contact_compliance=ContactCompliance(
            contact_preferences=[
                ContactPreference(channel="sms", allowed=False, prohibited_behavior="daily_texting", source_turn_id=1)
            ]
        ),
        objections=[
            ObjectionRecord(
                objection_id="obj_t1",
                canonical_category="timing",
                raw_utterance="We're not ready yet.",
                initial_statement="We're not ready yet.",
                latest_statement="We're not ready yet.",
                first_turn_id=5,
                last_updated_turn_id=5,
                lifecycle_state=ObjectionLifecycleState.ACTIVE,
                recurrence_count=1,
            )
        ],
    )
    eval_irrelevant = engine.evaluate(
        snapshot=snap_irrelevant,
        turn_speaker="client",
        turn_text="We're not ready yet.",
        turn_id=5,
    )
    ev_irrelevant = eval_irrelevant.decision.evidence_considered
    assert not any("Contact preferences:" in e for e in ev_irrelevant), (
        f"Contact preferences should NOT appear when irrelevant to the decision, got: {ev_irrelevant}"
    )

    # Case B: Turn explicitly invokes contact restrictions -> contact preferences MUST appear
    snap_relevant = ConversationStateSnapshot(
        call_sid="call_pt11_relevant",
        state_version=3,
        contact_compliance=ContactCompliance(
            contact_preferences=[
                ContactPreference(channel="sms", allowed=False, prohibited_behavior="daily_texting", source_turn_id=1)
            ]
        ),
    )
    eval_relevant = engine.evaluate(
        snapshot=snap_relevant,
        turn_speaker="client",
        turn_text="Please don't start texting me every day.",
        turn_id=15,
    )
    ev_relevant = eval_relevant.decision.evidence_considered
    assert any("Contact preferences:" in e for e in ev_relevant), (
        f"Contact preferences MUST appear when they influenced the decision, got: {ev_relevant}"
    )


def test_point12_no_canonical_state_duplication_uses_references():
    """Point 12: Don't duplicate canonical state between ConversationState and StrategicDecision.
    StrategicDecision points to facts and objections via referenced_fact_ids / referenced_objection_ids.
    """
    fact1 = PersistentFactRecord(
        fact_id="fact_meet_101",
        fact_key="confirmed_meeting_time",
        fact_value="Thursday at 3:00 PM",
        source_turn_id=18,
        timestamp_ms=1000,
    )
    snap = ConversationStateSnapshot(
        call_sid="call_pt12_ref",
        state_version=5,
        conversion_confirmed=True,
        facts=[fact1],
        conversion_event=ConversionEventObject(
            conversion_type="property_walkthrough",
            status=ConversionEventStatus.CONFIRMED,
            start_at="Thursday at 3:00 PM",
            participants=["Client", "Agent"],
        ),
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.95,
            commitment_slot="Thursday at 3:00 PM",
        ),
        push_strength=PushStrengthRecommendation(
            state="confirm_and_protect",
            rationale="Confirmed appointment protection mode",
            recommended_action="confirm_and_protect",
            confidence=0.95,
        ),
    )

    engine = PitchProXCoreIntelligenceEngine()
    eval_res = engine.evaluate(snapshot=snap, turn_speaker="client", turn_text="Thursday at 3 works.", turn_id=18)
    dec = eval_res.decision

    # Decision stores references by ID, not deep duplicated fact entities
    assert "fact_meet_101" in dec.referenced_fact_ids

    # Modifying state after decision doesn't leave divergent copies on StrategicDecision
    fact1.fact_value = "Thursday at 4:00 PM"
    assert dec.referenced_fact_ids == ["fact_meet_101"]


def test_point13_preserve_strategy_across_filler_turns():
    """Point 13: Preserve strategy across turns when nothing material changed.
    A filler turn ('okay') records carried_forward_from_decision_id without regenerating duplicate logic.
    """
    cdm = CoreDecisionManager(call_sid="call_pt13_carried")
    snap1 = ConversationStateSnapshot(
        call_sid="call_pt13_carried",
        state_version=1,
        last_updated_turn_id=3,
        conversation_stage=ConversationStage.DISCOVERY,
    )
    # Turn 1: Substantive question
    eval1 = cdm.evaluate_state(
        snapshot=snap1,
        turn_speaker="client",
        turn_text="We might look at moving sometime next year.",
        turn_id=4,
    )
    dec1 = eval1.decision
    assert dec1.carried_forward_from_decision_id is None

    # Turn 2: Non-material filler acknowledgment ("okay")
    snap2 = ConversationStateSnapshot(
        call_sid="call_pt13_carried",
        state_version=1,
        last_updated_turn_id=4,
        conversation_stage=ConversationStage.DISCOVERY,
    )
    eval2 = cdm.evaluate_state(
        snapshot=snap2,
        turn_speaker="client",
        turn_text="okay",
        turn_id=5,
    )
    dec2 = eval2.decision

    # Preserves provenance explicitly from Turn 4 decision
    assert dec2.carried_forward_from_decision_id == dec1.decision_id
    assert dec2.carried_forward_from_turn_id == 4
    assert "STRATEGY_CARRIED_FORWARD" in dec2.reason_codes
    assert dec2.should_prompt is False  # Point 8: filler produces no new prompt


def test_point14_use_strategy_history_avoids_repeating_failed_approaches():
    """Point 14: Use strategy history to avoid repeating failed approaches.
    If 'reframe' already failed on an objection, reactivating that objection selects a different tactic (e.g. QUANTIFY).
    """
    engine = PitchProXCoreIntelligenceEngine()

    obj_with_failure = ObjectionRecord(
        objection_id="obj_commission_reactivated",
        canonical_category="commission",
        raw_utterance="Your commission is too high.",
        initial_statement="Your commission is too high.",
        latest_statement="Your commission is too high.",
        first_turn_id=4,
        last_updated_turn_id=8,
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        recurrence_count=2,  # Normally triggers FIRST_PUSHBACK -> REFRAME
        attempted_strategies=["reframe"],
        strategy_outcomes=[
            StrategyAttemptOutcome(
                strategy_tag="reframe",
                attempted_at_turn_id=4,
                prospect_response_turn_id=5,
                prospect_response_summary="Objection persisted despite reframe",
                effectiveness="rejected",
            )
        ],
    )
    assert "reframe" in obj_with_failure.get_failed_strategies()

    snap = ConversationStateSnapshot(
        call_sid="call_pt14_history",
        state_version=4,
        last_updated_turn_id=8,
        objections=[obj_with_failure],
    )

    eval_res = engine.evaluate(snapshot=snap, turn_speaker="client", turn_text="I'm still not paying that fee.", turn_id=8)
    dec = eval_res.decision

    # 1. Did NOT repeat failed 'reframe'
    assert dec.primary_action != StrategicAction.REFRAME
    # 2. Selected alternative strategy
    assert dec.primary_action == StrategicAction.QUANTIFY
    # 3. Reason codes explicitly reference avoidance and pivot
    assert "AVOIDED_FAILED_STRATEGY_REFRAME" in dec.reason_codes
    assert "PIVOTED_TO_UNTRIED_STRATEGY" in dec.reason_codes
    # 4. Objective explicitly mentions the pivot
    assert "reframe" in dec.strategic_objective.lower()
    assert "quantify" in dec.strategic_objective.lower()
