import pytest
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle, DimensionScore, EmotionState
from copilot.conversation_state_models import (
    ConversationStage,
    ConversationStateSnapshot,
    ConversionEventStatus,
    ConversionEventObject,
    StageHistoryRecord,
)


def _make_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    agreement: float = 0.0,
    strategy_tag: str = "",
) -> BehavioralSignalInputBundle:
    return BehavioralSignalInputBundle(
        call_sid="CA_stage_test",
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        timestamp_ms=turn_id * 1000,
        trust=DimensionScore(score=0.6, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.1, confidence=0.85),
        pacing=DimensionScore(score=0.5, confidence=0.85, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.6, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_60_90s"),
        agreement_score=agreement,
        salesperson_strategy_tag=strategy_tag,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )


def test_turn_10_regression_to_decision_resolution():
    """Client Requirement #1:
    Assert that turn 10 moves the stage backward to decision_resolution
    even though turn 9 was in scheduling / appointment negotiation.
    """
    manager = ConversationStateManager(call_sid="CA_regress_test")

    # Turns 1-4: Discovery
    for t, (spk, txt) in enumerate([
        ("salesperson", "Tell me a bit about what's going on with the house."),
        ("client", "I'm the one making this decision, no one else needs to sign off."),
        ("salesperson", "Got it. What's driving the timing for you?"),
        ("client", "We might look at moving sometime next year, nothing urgent yet."),
    ], start=1):
        s = manager.process_turn_bundle(_make_bundle(t, spk, txt))
        assert s.conversation_stage == ConversationStage.DISCOVERY

    # Turns 5-7: Objection handling
    s5 = manager.process_turn_bundle(_make_bundle(5, "client", "Honestly, we're not sure this is the right time anymore."))
    assert s5.conversation_stage == ConversationStage.OBJECTION_HANDLING

    s6 = manager.process_turn_bundle(_make_bundle(6, "salesperson", "That's fair – a lot of people feel that way before they see the actual numbers. Would it help to walk through what the market looks like right now?"))
    assert s6.conversation_stage == ConversationStage.OBJECTION_HANDLING

    s7 = manager.process_turn_bundle(_make_bundle(7, "client", "Okay, that makes sense, I guess timing isn't the biggest issue."))
    assert s7.conversation_stage == ConversationStage.OBJECTION_HANDLING

    # Turns 8-9: Scheduling
    s8 = manager.process_turn_bundle(_make_bundle(8, "salesperson", "Great – would sometime next week work for a walkthrough?"))
    assert s8.conversation_stage == ConversationStage.SCHEDULING

    s9 = manager.process_turn_bundle(_make_bundle(9, "client", "Maybe next week could work, let me think about it."))
    assert s9.conversation_stage == ConversationStage.SCHEDULING

    # Turn 10: Client reveals absent wife -> FORCES BACKWARD REGRESSION TO DECISION_RESOLUTION
    s10 = manager.process_turn_bundle(_make_bundle(10, "client", "Actually, my wife would really need to be part of this conversation before we go any further."))
    assert s10.conversation_stage == ConversationStage.DECISION_RESOLUTION

    # Verify audit record in change_history for the regression
    regress_record = next(
        c for c in s10.change_history
        if c.field_path == "conversation_stage" and c.triggering_turn_id == 10
    )
    assert regress_record.old_value == "scheduling"
    assert regress_record.new_value == "decision_resolution"
    assert "Regression to decision_resolution" in regress_record.reason

    # Turn 11: Salesperson aligns on looping in the absent spouse
    s11 = manager.process_turn_bundle(_make_bundle(11, "salesperson", "Of course, happy to loop her in whenever works."))
    assert s11.conversation_stage == ConversationStage.DECISION_RESOLUTION


def test_full_stage_progression_and_turn_18_commitment_confirmed():
    """Verifies the complete 18-turn trajectory and confirms that Turn 18
    lands in COMMITMENT_CONFIRMED.
    """
    manager = ConversationStateManager(call_sid="CA_full_traj")

    turns = [
        (1, "salesperson", "Hi, thanks for making time today – tell me a bit about what's going on with the house."),
        (2, "client", "I'm the one making this decision, no one else needs to sign off."),
        (3, "salesperson", "Got it. What's driving the timing for you?"),
        (4, "client", "We might look at moving sometime next year, nothing urgent yet."),
        (5, "client", "Honestly, we're not sure this is the right time anymore."),
        (6, "salesperson", "That's fair – a lot of people feel that way before they see the actual numbers. Would it help to walk through what the market looks like right now?"),
        (7, "client", "Okay, that makes sense, I guess timing isn't the biggest issue."),
        (8, "salesperson", "Great – would sometime next week work for a walkthrough?"),
        (9, "client", "Maybe next week could work, let me think about it."),
        (10, "client", "Actually, my wife would really need to be part of this conversation before we go any further."),
        (11, "salesperson", "Of course, happy to loop her in whenever works."),
        (12, "client", "I guess I'm just worried this isn't really the right move for us financially with everything going on."),
        (13, "salesperson", "Totally understand – let's look at your net proceeds after all costs, so you can see the real picture."),
        (14, "client", "That's actually really helpful, tell me more – how does the marketing process work, what about staging, how long does listing usually take?"),
        (15, "client", "Please don't start texting me every day before we meet, by the way."),
        (16, "client", "Mornings don't really work for us either, just so you know."),
        (17, "salesperson", "Noted on all of that. What day works best?"),
        (18, "client", "Thursday at 3 works, and my wife will be there."),
    ]

    expected_stages = {
        1: ConversationStage.DISCOVERY,
        2: ConversationStage.DISCOVERY,
        3: ConversationStage.DISCOVERY,
        4: ConversationStage.DISCOVERY,
        5: ConversationStage.OBJECTION_HANDLING,
        6: ConversationStage.OBJECTION_HANDLING,
        7: ConversationStage.OBJECTION_HANDLING,
        8: ConversationStage.SCHEDULING,
        9: ConversationStage.SCHEDULING,
        10: ConversationStage.DECISION_RESOLUTION,
        11: ConversationStage.DECISION_RESOLUTION,
        12: ConversationStage.OBJECTION_HANDLING,
        13: ConversationStage.VALUE_WALKTHROUGH,
        14: ConversationStage.VALUE_WALKTHROUGH,
        15: ConversationStage.SCHEDULING,
        16: ConversationStage.SCHEDULING,
        17: ConversationStage.SCHEDULING,
        18: ConversationStage.COMMITMENT_CONFIRMED,
    }

    for tid, spk, txt in turns:
        b = _make_bundle(tid, spk, txt)
        # Give Turn 18 agreement to trigger concrete commitment confirmation
        if tid == 18:
            b.agreement_score = 0.85
        s = manager.process_turn_bundle(b)
        assert s.conversation_stage == expected_stages[tid], f"Turn {tid} expected {expected_stages[tid]}, got {s.conversation_stage}"

    # Verify final stage is COMMITMENT_CONFIRMED
    assert manager.current_state.conversation_stage == ConversationStage.COMMITMENT_CONFIRMED


def test_stage_history_preserves_non_linear_reentry_without_overwriting():
    """Verifies that re-entering a stage (e.g. scheduling at Turn 8-9 and again at Turn 15-17)
    appends separate chronological records in stage_history rather than overwriting.
    """
    manager = ConversationStateManager(call_sid="CA_reentry_history")

    # Start in discovery
    b1 = _make_bundle(1, "salesperson", "Tell me about the house.")
    manager.process_turn_bundle(b1)

    # First scheduling span
    b8 = _make_bundle(8, "salesperson", "Would next week work for a walkthrough?")
    manager.process_turn_bundle(b8)
    assert manager.current_state.conversation_stage == ConversationStage.SCHEDULING

    # Regression to decision resolution
    b10 = _make_bundle(10, "client", "Actually, my wife would really need to be part of this conversation before we go any further.")
    manager.process_turn_bundle(b10)
    assert manager.current_state.conversation_stage == ConversationStage.DECISION_RESOLUTION

    # Second scheduling span
    b15 = _make_bundle(15, "client", "Please don't start texting me every day before we meet, by the way.")
    manager.process_turn_bundle(b15)
    assert manager.current_state.conversation_stage == ConversationStage.SCHEDULING

    # Inspect stage_history
    history = manager.current_state.stage_history
    assert len(history) >= 4

    scheduling_spans = [h for h in history if h.stage == ConversationStage.SCHEDULING]
    assert len(scheduling_spans) == 2, "Expected exactly two separate scheduling spans in history"

    first_sched = scheduling_spans[0]
    second_sched = scheduling_spans[1]

    assert first_sched.entered_turn_id == 8
    assert first_sched.exited_turn_id == 10

    assert second_sched.entered_turn_id == 15
    assert second_sched.exited_turn_id is None  # Still active


def test_state_change_audit_records_for_stage_transitions():
    """Verifies that every stage transition creates a StateChangeRecord with
    field_path == 'conversation_stage', old_value, new_value, and descriptive reason.
    """
    manager = ConversationStateManager(call_sid="CA_audit_test")

    # Initial turn (remains in discovery, no stage transition)
    manager.process_turn_bundle(_make_bundle(1, "salesperson", "Hello there."))
    stage_changes_t1 = [c for c in manager.current_state.change_history if c.field_path == "conversation_stage"]
    assert len(stage_changes_t1) == 0

    # Turn 5: Objection raised -> transition to objection_handling
    manager.process_turn_bundle(_make_bundle(5, "client", "Honestly, we're not sure this is the right time anymore."))
    stage_changes = [c for c in manager.current_state.change_history if c.field_path == "conversation_stage"]
    assert len(stage_changes) == 1

    change = stage_changes[0]
    assert change.old_value == "discovery"
    assert change.new_value == "objection_handling"
    assert change.triggering_turn_id == 5
    assert "Conversation stage transitioned from 'discovery' to 'objection_handling'" in change.reason
    assert change.state_version_after > change.state_version_before
