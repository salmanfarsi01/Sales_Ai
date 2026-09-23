import pytest
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle, DimensionScore, EmotionState
from copilot.conversation_state_models import (
    ConversationStage,
    ConversationStateSnapshot,
    ObjectionLifecycleState,
    ObjectionRecord,
    DormancyEvidence,
)
from copilot.conversation_scoring_config import ConversationScoringConfig
from copilot.conversation_materiality import _get_dormancy_evidence


def _create_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    agreement: float = 0.0,
    future_lang: float = 0.0,
    readiness: float = 0.5,
    trust: float = 0.5,
) -> BehavioralSignalInputBundle:
    return BehavioralSignalInputBundle(
        call_sid="CA_test_dormancy",
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        timestamp_ms=turn_id * 1000,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.1, confidence=0.85),
        pacing=DimensionScore(score=0.5, confidence=0.85, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.6, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.85, primary_horizon="last_60_90s"),
        agreement_score=agreement,
        future_language_score=future_lang,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )


def test_silence_alone_does_not_trigger_dormancy():
    """Client Principle 'Silence is not resolution':
    A 3-turn gap with zero corroborating evidence must NOT age an active or
    partially resolved objection into DORMANT. It must preserve its current state.
    """
    manager = ConversationStateManager(call_sid="CA_silence_test")

    # Turn 1: Client raises objection
    b1 = _create_bundle(1, "client", "Your commission fee of 6 percent is way too high.")
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    obj = s1.objections[0]
    assert obj.lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 2: Non-corroborating question
    b2 = _create_bundle(2, "salesperson", "What is your timeline for the move?")
    s2 = manager.process_turn_bundle(b2)
    assert s2.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 3: Neutral response
    b3 = _create_bundle(3, "client", "We might look at moving next year.", agreement=0.0, future_lang=0.0)
    s3 = manager.process_turn_bundle(b3)
    assert s3.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 4: Turn 4 arrives (3 turns elapsed since Turn 1: 4 - 1 = 3 >= 3).
    # Since there is NO corroborating evidence, objection MUST STAY ACTIVE!
    b4 = _create_bundle(4, "salesperson", "Understood.", agreement=0.0, future_lang=0.0)
    s4 = manager.process_turn_bundle(b4)
    assert s4.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert len(s4.get_dormant_objections()) == 0
    assert not any("dormant" in c.field_path.lower() for c in s4.change_history)

    # Turn 5: 4 turns elapsed, still zero corroborating evidence -> stays ACTIVE!
    b5 = _create_bundle(5, "salesperson", "Let me know if you need anything else.")
    s5 = manager.process_turn_bundle(b5)
    assert s5.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE


def test_partially_resolved_objection_stays_partially_resolved_under_silence():
    """Verifies that partially_resolved objections stay partially_resolved
    when elapsed turns >= 3 but no corroborating evidence exists.
    """
    manager = ConversationStateManager(call_sid="CA_partially_resolved_silence")

    # Initialize state with an existing partially_resolved objection at turn 14
    obj = ObjectionRecord(
        objection_id="obj_hesitation",
        canonical_category="general_hesitation",
        initial_statement="Not sure this is the right time.",
        latest_statement="Not sure this is the right time.",
        lifecycle_state=ObjectionLifecycleState.PARTIALLY_RESOLVED,
        first_turn_id=5,
        last_updated_turn_id=14,
    )
    manager.current_state.objections.append(obj)
    manager.current_state.last_updated_turn_id = 14
    manager.objections_engine._objections = [obj]

    # Turn 15: Soft contact preference
    b15 = _create_bundle(15, "client", "Please don't text me daily before we meet.")
    manager.process_turn_bundle(b15)

    # Turn 16: Logistical scheduling constraint
    b16 = _create_bundle(16, "client", "Mornings don't really work for us.")
    manager.process_turn_bundle(b16)

    # Turn 17: Salesperson asks what day works best (delta = 17 - 14 = 3 >= 3)
    b17 = _create_bundle(17, "salesperson", "Noted on all of that. What day works best?")
    s17 = manager.process_turn_bundle(b17)

    # Invariant: Objection stays PARTIALLY_RESOLVED, NOT DORMANT!
    active_obj = next(o for o in s17.objections if o.objection_id == "obj_hesitation")
    assert active_obj.lifecycle_state == ObjectionLifecycleState.PARTIALLY_RESOLVED
    assert active_obj.dormancy_evidence is None
    assert len(s17.get_dormant_objections()) == 0


def test_supersession_evidence_triggers_immediate_dormancy():
    """Signal 1: Supersession — o.superseded_by_objection_id is not None.
    Immediate dormancy regardless of turn count.
    """
    manager = ConversationStateManager(call_sid="CA_supersession_dormancy")

    b1 = _create_bundle(1, "client", "Honestly, we're not sure this is the right time anymore.")
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    obj = s1.objections[0]

    # Explicitly mark superseded at turn 2 (delta = 2 - 1 = 1 < 3)
    obj.superseded_by_objection_id = "obj_decision_to_stay"
    obj.superseded_at_turn_id = 2

    b2 = _create_bundle(2, "salesperson", "Understood, let me know what you decide.")
    s2 = manager.process_turn_bundle(b2)

    # Should transition to DORMANT immediately due to supersession evidence
    dormant_objs = s2.get_dormant_objections()
    assert len(dormant_objs) == 1
    assert dormant_objs[0].dormancy_evidence is not None
    assert dormant_objs[0].dormancy_evidence.evidence_type == "supersession"

    # Verify reason string format
    change = next(c for c in s2.change_history if c.field_path == f"objections.{obj.objection_id}.lifecycle_state" and c.new_value == "dormant")
    assert "transitioned to DORMANT: superseded by 'obj_decision_to_stay'" in change.reason
    assert "1 turns since last mention" in change.reason
    assert "after 1 turns without mention" not in change.reason


def test_stage_based_supersession_corroborates_dormancy():
    """Signal 2: Stage-based supersession — current_state.conversation_stage
    has moved past the domain the objection belongs to.
    """
    manager = ConversationStateManager(call_sid="CA_stage_dormancy")

    b1 = _create_bundle(1, "client", "Honestly, we're not sure this is the right time anymore.")
    manager.process_turn_bundle(b1)

    b2 = _create_bundle(2, "salesperson", "Let's explore options.")
    manager.process_turn_bundle(b2)

    b3 = _create_bundle(3, "client", "Okay, what do you suggest?")
    manager.process_turn_bundle(b3)

    # At Turn 4, stage transitions to 'scheduling'
    manager.current_state.conversation_stage = "scheduling"
    b4 = _create_bundle(4, "salesperson", "Let's get together Thursday to look at market comps.")
    s4 = manager.process_turn_bundle(b4)

    # 3 turns elapsed (4 - 1 = 3 >= 3) and stage has moved past general_hesitation -> DORMANT!
    dormant_objs = s4.get_dormant_objections()
    assert len(dormant_objs) == 1
    d_obj = dormant_objs[0]
    assert d_obj.lifecycle_state == ObjectionLifecycleState.DORMANT
    assert d_obj.dormancy_evidence is not None
    assert d_obj.dormancy_evidence.evidence_type == "stage_transition"
    assert "scheduling" in d_obj.dormancy_evidence.description

    change = next(c for c in s4.change_history if "lifecycle_state" in c.field_path and c.new_value == "dormant")
    assert "Concern 'general_hesitation' transitioned to DORMANT: superseded by stage transition into 'scheduling' (3 turns since last mention)." in change.reason


def test_stage_based_supersession_corroborates_dormancy_natural_progression():
    """Signal 2: Stage-based supersession — end-to-end natural progression.
    Verifies that an unaddressed ACTIVE objection transitions to DORMANT when
    conversation_stage naturally moves past its domain (into scheduling) through
    normal turns, with zero manual state overrides.
    """
    manager = ConversationStateManager(call_sid="CA_natural_stage_dormancy")

    # Turn 1: Client raises early timing hesitation -> ACTIVE objection
    b1 = _create_bundle(1, "client", "Honestly, we are not sure this is the right time anymore.")
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    assert s1.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 2: Non-reframe neutral question (no strategy attempted)
    b2 = _create_bundle(2, "salesperson", "Tell me a bit about the house condition.")
    s2 = manager.process_turn_bundle(b2)
    assert s2.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 3: Neutral property disclosure
    b3 = _create_bundle(3, "client", "It has three bedrooms and two baths.")
    s3 = manager.process_turn_bundle(b3)
    assert s3.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 4: General walkthrough discussion
    b4 = _create_bundle(4, "salesperson", "Let us look at your net proceeds after all costs.")
    s4 = manager.process_turn_bundle(b4)
    assert s4.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 5: Salesperson proposes scheduling walkthrough -> stage transitions to SCHEDULING
    b5 = _create_bundle(5, "salesperson", "Would sometime next week work for a walkthrough?")
    s5 = manager.process_turn_bundle(b5)
    assert s5.conversation_stage == ConversationStage.SCHEDULING

    # Turn 6: Client responds to scheduling ask (delta = 6 - 1 = 5 >= 3)
    # Since stage is now 'scheduling', the unaddressed ACTIVE discovery objection
    # is corroborated for dormancy by stage-based supersession!
    b6 = _create_bundle(6, "client", "Maybe next week could work, let me think about it.")
    s6 = manager.process_turn_bundle(b6)

    dormant_objs = s6.get_dormant_objections()
    assert len(dormant_objs) == 1
    d_obj = dormant_objs[0]
    assert d_obj.lifecycle_state == ObjectionLifecycleState.DORMANT
    assert d_obj.dormancy_evidence is not None
    assert d_obj.dormancy_evidence.evidence_type == "stage_transition"
    assert "scheduling" in d_obj.dormancy_evidence.description

    change = next(c for c in s6.change_history if "lifecycle_state" in c.field_path and c.new_value == "dormant")
    assert "Concern 'general_hesitation' transitioned to DORMANT: superseded by stage transition into 'scheduling'" in change.reason


def test_behavioral_resolution_corroborates_dormancy():
    """Signal 3: Behavioral resolution evidence since last activity —
    rising agreement, forward future language, or elevated readiness.
    """
    manager = ConversationStateManager(call_sid="CA_behavioral_dormancy")

    # Turn 1: Canonical fee objection
    b1 = _create_bundle(1, "client", "Your commission fee is 6 percent, which feels way too expensive.")
    manager.process_turn_bundle(b1)

    # Turn 2: Neutral question
    b2 = _create_bundle(2, "salesperson", "What timeline are you thinking?")
    manager.process_turn_bundle(b2)

    # Turn 3: Prospect demonstrates forward future language without directly resolving the objection
    b3 = _create_bundle(
        3,
        "client",
        "We want to plan ahead for the listing in November.",
        agreement=0.50,
        future_lang=0.55,
        readiness=0.55,
    )
    manager.process_turn_bundle(b3)

    # Turn 4: Turn 4 arrives (4 - 1 = 3 turns elapsed)
    b4 = _create_bundle(4, "salesperson", "Understood.")
    s4 = manager.process_turn_bundle(b4)

    dormant_objs = s4.get_dormant_objections()
    assert len(dormant_objs) == 1
    d_obj = dormant_objs[0]
    assert d_obj.lifecycle_state == ObjectionLifecycleState.DORMANT
    assert d_obj.dormancy_evidence is not None
    assert d_obj.dormancy_evidence.evidence_type == "behavioral_resolution"

    change = next(c for c in s4.change_history if "lifecycle_state" in c.field_path and c.new_value == "dormant")
    assert "transitioned to DORMANT: corroborated by forward future language" in change.reason
    assert "3 turns since last mention" in change.reason


def test_blocker_supersession_corroborates_dormancy():
    """Signal 4: Blocker supersession — another objection or blocker became primary.
    """
    manager = ConversationStateManager(call_sid="CA_blocker_supersession")

    # Turn 1: General hesitation concern
    b1 = _create_bundle(1, "client", "Honestly, we're not sure this is the right time anymore.")
    manager.process_turn_bundle(b1)

    # Turn 2: Agent response
    b2 = _create_bundle(2, "salesperson", "Understood.")
    manager.process_turn_bundle(b2)

    # Turn 3: A major blocker objection arises (commission fee resistance)
    b3 = _create_bundle(3, "client", "Your commission fee is 6 percent, which feels way too expensive.")
    manager.process_turn_bundle(b3)

    # Turn 4: Turn 4 arrives (4 - 1 = 3 turns since general_hesitation)
    b4 = _create_bundle(4, "salesperson", "Let's discuss the fee breakdown.")
    s4 = manager.process_turn_bundle(b4)

    # general_hesitation should be dormant due to blocker supersession by commission_fee
    dormant_objs = s4.get_dormant_objections()
    assert len(dormant_objs) == 1
    assert dormant_objs[0].canonical_category == "general_hesitation"
    assert dormant_objs[0].dormancy_evidence.evidence_type == "blocker_supersession"
    assert "commission_fee" in dormant_objs[0].dormancy_evidence.description

    change = next(c for c in s4.change_history if c.field_path == f"objections.{dormant_objs[0].objection_id}.lifecycle_state" and c.new_value == "dormant")
    assert "superseded by newer primary objection 'commission_fee'" in change.reason


def test_reason_string_never_emits_bare_turns_without_mention():
    """Audits all emitted reason strings to guarantee the old phrasing
    'after N turns without mention' is NEVER emitted alone.
    """
    manager = ConversationStateManager(call_sid="CA_reason_audit")

    b1 = _create_bundle(1, "client", "I'm unsure about this.")
    manager.process_turn_bundle(b1)
    b2 = _create_bundle(2, "salesperson", "Tell me more.")
    manager.process_turn_bundle(b2)
    b3 = _create_bundle(3, "client", "I see what you mean, that sounds good.", agreement=0.75)
    manager.process_turn_bundle(b3)
    b4 = _create_bundle(4, "salesperson", "Great.")
    s4 = manager.process_turn_bundle(b4)

    for c in s4.change_history:
        if "lifecycle_state" in c.field_path and c.new_value == "dormant":
            assert "without mention" not in c.reason
            assert "transitioned to DORMANT:" in c.reason
            assert "turns since last mention" in c.reason
