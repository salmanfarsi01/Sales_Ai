import pytest
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import (
    BehavioralSignalInputBundle,
    DimensionScore,
    EmotionState,
    DownstreamInferenceState,
    SemanticFeatureSnapshot,
    extract_behavioral_bundle,
)
from copilot.conversation_state_models import DecisionStakeholder
from copilot.conversation_replay import ConversationReplayEngine


def _make_upstream(call_sid: str, turn_id: int, text: str, timestamp_ms: int = 1000) -> BehavioralSignalInputBundle:
    inf = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=timestamp_ms,
        trust=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.3, tension_level=0.2, confidence=0.85),
        pacing=DimensionScore(score=0.60, confidence=0.80, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.80, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.70, confidence=0.80, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.70, confidence=0.80, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )
    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id}",
        call_sid=call_sid,
        speaker_id="client",
        contact_preference="none",
        specificity_score=0.80,
        agreement_score=0.80,
        future_language_score=0.70,
        boundary_score=0.0,
    )
    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id="client",
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=sem,
    )


def test_stakeholder_presence_confirmation_flips_presence_and_dm_present():
    """Client Principle #1: Disclosing that an absent stakeholder will attend updates presence to confirmed_attending and sets decision_maker_present=True."""
    mgr = ConversationStateManager(call_sid="test_presence_confirm")

    # Step 1: Client declares wife needs to be part of conversation (absent stakeholder)
    bundle1 = _make_upstream("test_presence_confirm", 1, "Actually, my wife would really need to be part of this conversation before we go any further.", 1000)
    s1 = mgr.process_turn_bundle(bundle1)
    assert not s1.decision_structure.decision_maker_present
    assert any(s.role == "wife" and s.presence == "absent" for s in s1.decision_structure.stakeholders)

    # Step 2: Client declares wife will be there at meeting
    bundle2 = _make_upstream("test_presence_confirm", 2, "Thursday at 3 works, and my wife will be there.", 2000)
    s2 = mgr.process_turn_bundle(bundle2)

    # Assertions: wife presence is updated to confirmed_attending and decision_maker_present is restored to True
    assert s2.decision_structure.decision_maker_present
    wife_stk = next((s for s in s2.decision_structure.stakeholders if s.role == "wife"), None)
    assert wife_stk is not None
    assert wife_stk.presence == "confirmed_attending"
    assert "Confirmed attending" in wife_stk.notes

    # Assert Condition 5 (decision_maker_aligned) in conversion gate is met
    assert s2.conversion_gate is not None
    dm_cond = next((c for c in s2.conversion_gate.conditions if c.condition_name == "decision_maker_aligned"), None)
    assert dm_cond is not None
    assert dm_cond.met is True


def test_logistics_morning_constraint_satisfied_by_afternoon_slot():
    """Client Principles #1 & #6: Afternoon appointment satisfies earlier morning constraint, unblocking plausible_logistics."""
    mgr = ConversationStateManager(call_sid="test_logistics_constraint")

    # Step 1: Client declares mornings don't work
    bundle1 = _make_upstream("test_logistics_constraint", 1, "Mornings don't really work for us either, just so you know.", 1000)
    s1 = mgr.process_turn_bundle(bundle1)
    assert "Mornings unavailable" in s1.decision_structure.access_constraints

    # Step 2: Client books Thursday at 3 PM
    bundle2 = _make_upstream("test_logistics_constraint", 2, "Thursday at 3 works, that's fine.", 2000)
    s2 = mgr.process_turn_bundle(bundle2)

    # Plausible logistics condition must pass (satisfied constraint outranked by explicit slot)
    log_cond = next((c for c in s2.conversion_gate.conditions if c.condition_name == "plausible_logistics"), None)
    assert log_cond is not None
    assert log_cond.met is True
    assert log_cond.is_overridden is True
    assert "satisfies scheduling constraints" in log_cond.reason


def test_full_18_turn_dialogue_gate_and_push_strength():
    """Client Principles #1, #6, #9: 18-turn test dialogue opens gate and triggers confirm_and_protect."""
    dialogue = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, thanks for making time today — tell me a bit about what's going on with the house."},
        {"turn_id": 2, "speaker_id": "client", "text": "I'm the one making this decision, no one else needs to sign off."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Got it. What's driving the timing for you?"},
        {"turn_id": 4, "speaker_id": "client", "text": "We might look at moving sometime next year, nothing urgent yet."},
        {"turn_id": 5, "speaker_id": "client", "text": "Honestly, we're not sure this is the right time anymore."},
        {"turn_id": 6, "speaker_id": "salesperson", "text": "That's fair — a lot of people feel that way before they see the actual numbers. Would it help to walk through what the market looks like right now?"},
        {"turn_id": 7, "speaker_id": "client", "text": "Okay, that makes sense, I guess timing isn't the biggest issue."},
        {"turn_id": 8, "speaker_id": "salesperson", "text": "Great — would sometime next week work for a walkthrough?"},
        {"turn_id": 9, "speaker_id": "client", "text": "Maybe next week could work, let me think about it."},
        {"turn_id": 10, "speaker_id": "client", "text": "Actually, my wife would really need to be part of this conversation before we go any further."},
        {"turn_id": 11, "speaker_id": "salesperson", "text": "Of course, happy to loop her in whenever works."},
        {"turn_id": 12, "speaker_id": "client", "text": "I guess I'm just worried this isn't really the right move for us financially with everything going on."},
        {"turn_id": 13, "speaker_id": "salesperson", "text": "Totally understand — let's look at your net proceeds after all costs, so you can see the real picture."},
        {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more — how does the marketing process work, what about staging, how long does listing usually take?"},
        {"turn_id": 15, "speaker_id": "client", "text": "Please don't start texting me every day before we meet, by the way."},
        {"turn_id": 16, "speaker_id": "client", "text": "Mornings don't really work for us either, just so you know."},
        {"turn_id": 17, "speaker_id": "salesperson", "text": "Noted on all of that. What day works best?"},
        {"turn_id": 18, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
    ]

    engine = ConversationReplayEngine()
    report = engine.replay_dialogue_turns(call_sid="sim_18turn_test", raw_turns=dialogue, save_report=False)

    final = report.final_state
    # 1. Gate is OPEN with 0 failed conditions
    assert final.conversion_gate.is_open is True
    assert final.conversion_gate.status == "open"
    assert final.conversion_gate.failed_conditions == []
    assert final.conversion_gate.blocking_reasons == []

    # 2. Push strength is confirm_and_protect
    assert final.push_strength.state == "confirm_and_protect"

    # 3. Decision maker present is True and wife is confirmed attending
    assert final.decision_structure.decision_maker_present is True
    wife = next((s for s in final.decision_structure.stakeholders if s.role == "wife"), None)
    assert wife is not None
    assert wife.presence == "confirmed_attending"

    # 4. Client Principle #10: Intermediate steps have change_history pruned to avoid bloat
    for step in report.timeline:
        assert step.state_before.change_history == []
        assert step.state_after.change_history == []
        # But turn changes are captured
        assert isinstance(step.state_changes, list)

    # 5. Complete audit trail is preserved intact on final_state
    assert len(final.change_history) > 10
