import pytest
from copilot.conversation_objections import classify_objection_label, ObjectionLifecycleEngine
from copilot.conversation_state_contract import infer_salesperson_strategy_from_text, extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_replay import ConversationReplayEngine
from copilot.behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from copilot.behavioral_semantic import SemanticFeatureSnapshot


def test_classify_objection_label_turns_5_and_12():
    t5 = "Honestly, we're not sure this is the right time anymore."
    t12 = "I guess I'm just worried this isn't really the right move for us financially with everything going on."

    assert classify_objection_label(t5) == "general_hesitation"
    assert classify_objection_label(t12) == "general_hesitation"


def test_infer_salesperson_strategy_turns_6_and_13():
    t6 = "That's fair — a lot of people feel that way before they see the actual numbers. Would it help to walk through what the market looks like right now?"
    t13 = "Totally understand — let's look at your net proceeds after all costs, so you can see the real picture."

    assert infer_salesperson_strategy_from_text(t6) == "market_data_walkthrough"
    assert infer_salesperson_strategy_from_text(t13) == "financial_net_proceeds_reframe"


def test_objection_lifecycle_reactivation_and_supersession():
    sm = ConversationStateManager(call_sid="sim_test_lifecycle")

    def _make_bundle(turn_id, speaker_id, text, agreement=0.5, ready=0.5, eng=0.5):
        inf = DownstreamInferenceState(
            call_sid="sim_test_lifecycle",
            timestamp_ms=turn_id * 3000,
            trust=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_20_30s"),
            emotion=EmotionState(expressed_valence=0.0, tension_level=0.2, confidence=0.85),
            pacing=DimensionScore(score=0.6, confidence=0.8, primary_horizon="current_utterance"),
            engagement=DimensionScore(score=eng, confidence=0.85, primary_horizon="last_20_30s"),
            momentum=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
            readiness=DimensionScore(score=ready, confidence=0.8, primary_horizon="last_60_90s"),
            overall_confidence=0.85,
            contributing_evidence_ids=[f"ev_{turn_id}"],
        )
        sem = SemanticFeatureSnapshot(
            utterance_id=f"utt_{turn_id}",
            call_sid="sim_test_lifecycle",
            speaker_id=speaker_id,
            boundary_score=0.0,
            specificity_score=0.6,
            agreement_score=agreement,
        )
        b = extract_behavioral_bundle(
            turn_id=turn_id,
            speaker_id=speaker_id,
            utterance_text=text,
            inference_state=inf,
            semantic_snapshot=sem,
        )
        b.timestamp_ms = turn_id * 3000
        return b

    # Turn 4: Client expresses tentative timeline horizon
    sm.process_turn_bundle(_make_bundle(4, "client", "We might look at moving sometime next year, nothing urgent yet."))
    facts = {f.fact_key: f for f in sm.current_state.facts if f.status == "active"}
    assert "timeline_horizon" in facts
    assert facts["timeline_horizon"].fact_value == "Sometime next year"
    assert sm.current_state.decision_structure.timeline_horizon == "sometime next year"
    assert sm.current_state.decision_structure.urgency_level == "low"

    # Turn 5: Client raises general hesitation objection
    sm.process_turn_bundle(_make_bundle(5, "client", "Honestly, we're not sure this is the right time anymore."))
    assert len(sm.current_state.objections) == 1
    obj = sm.current_state.objections[0]
    assert obj.canonical_category == "general_hesitation"
    assert obj.lifecycle_state in ("active", "unresolved")
    assert obj.recurrence_count == 1

    # Turn 6: Salesperson reframes using market data
    sm.process_turn_bundle(_make_bundle(6, "salesperson", "That's fair — a lot of people feel that way before they see the actual numbers. Would it help to walk through what the market looks like right now?"))
    obj = sm.current_state.objections[0]
    assert "market_data_walkthrough" in obj.attempted_strategies

    # Turn 7: Client concedes timing isn't the biggest issue -> objection resolves
    sm.process_turn_bundle(_make_bundle(7, "client", "Okay, that makes sense, I guess timing isn't the biggest issue."))
    obj = sm.current_state.objections[0]
    assert obj.lifecycle_state == "resolved"
    assert len(obj.strategy_outcomes) == 1
    assert obj.strategy_outcomes[0].strategy_tag == "market_data_walkthrough"
    assert obj.strategy_outcomes[0].effectiveness == "effective"

    # Turn 9: Client expresses tentative meeting timing
    sm.process_turn_bundle(_make_bundle(9, "client", "Maybe next week could work, let me think about it."))
    facts = {f.fact_key: f for f in sm.current_state.facts if f.status == "active"}
    assert "tentative_meeting_time" in facts
    assert "next week" in facts["tentative_meeting_time"].fact_value.lower()

    # Turn 12: Client expresses financial move anxiety -> reactivates resolved objection
    sm.process_turn_bundle(_make_bundle(12, "client", "I guess I'm just worried this isn't really the right move for us financially with everything going on."))
    obj = sm.current_state.objections[0]
    assert obj.lifecycle_state in ("active", "reactivated")
    assert obj.recurrence_count == 2
    assert "financially" in obj.latest_statement

    # Turn 13: Salesperson reframes with net proceeds
    sm.process_turn_bundle(_make_bundle(13, "salesperson", "Totally understand — let's look at your net proceeds after all costs, so you can see the real picture."))
    obj = sm.current_state.objections[0]
    assert "financial_net_proceeds_reframe" in obj.attempted_strategies

    # Turn 14: Client accepts reframe with process interest -> partially resolved
    sm.process_turn_bundle(_make_bundle(14, "client", "That's actually really helpful, tell me more — how does the marketing process work, what about staging, how long does listing usually take?", eng=0.85, ready=0.35))
    obj = sm.current_state.objections[0]
    assert obj.lifecycle_state in ("partially_resolved", "dormant")
    assert len(obj.strategy_outcomes) == 2
    assert obj.strategy_outcomes[1].strategy_tag == "financial_net_proceeds_reframe"
    assert obj.strategy_outcomes[1].effectiveness == "partial"

    # Turn 15: Client sets soft contact preference
    sm.process_turn_bundle(_make_bundle(15, "client", "Please don't start texting me every day before we meet, by the way."))
    facts = {f.fact_key: f for f in sm.current_state.facts if f.status == "active"}
    assert "contact_preference" in facts
    assert "texting" in facts["contact_preference"].fact_value.lower()

    # Turn 16: Client sets scheduling constraint
    sm.process_turn_bundle(_make_bundle(16, "client", "Mornings don't really work for us either, just so you know."))
    facts = {f.fact_key: f for f in sm.current_state.facts if f.status == "active"}
    assert "scheduling_constraint" in facts
    assert "mornings" in facts["scheduling_constraint"].fact_value.lower()
    assert "Mornings unavailable" in sm.current_state.decision_structure.access_constraints

    # Turn 18: Client confirms concrete meeting -> supersedes tentative meeting time
    sm.process_turn_bundle(_make_bundle(18, "client", "Thursday at 3 works, and my wife will be there."))
    all_facts = sm.current_state.facts
    active_facts = {f.fact_key: f for f in all_facts if f.status == "active"}

    assert "confirmed_meeting_time" in active_facts
    assert "Thursday At 3" in active_facts["confirmed_meeting_time"].fact_value
    assert any(f.fact_key == "tentative_meeting_time" and f.status == "superseded" for f in all_facts)


def test_full_sim_muc323di_replay():
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
        {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more — how does the marketing process work, what about staging, how long does listing usually take?", "engagement": 0.85, "readiness": 0.35},
        {"turn_id": 15, "speaker_id": "client", "text": "Please don't start texting me every day before we meet, by the way."},
        {"turn_id": 16, "speaker_id": "client", "text": "Mornings don't really work for us either, just so you know."},
        {"turn_id": 17, "speaker_id": "salesperson", "text": "Noted on all of that. What day works best?"},
        {"turn_id": 18, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
    ]

    engine = ConversationReplayEngine()
    report = engine.replay_dialogue_turns(call_sid="sim_muc323di_unit_test", raw_turns=dialogue, save_report=False)

    # 1. Objection lifecycle fired and reactivated
    assert len(report.final_state.objections) == 1
    obj = report.final_state.objections[0]
    assert obj.canonical_category == "general_hesitation"
    assert obj.recurrence_count == 2
    assert len(obj.strategy_outcomes) == 2
    assert obj.strategy_outcomes[0].strategy_tag == "market_data_walkthrough"
    assert obj.strategy_outcomes[0].effectiveness == "effective"
    assert obj.strategy_outcomes[1].strategy_tag == "financial_net_proceeds_reframe"
    assert obj.strategy_outcomes[1].effectiveness == "partial"
    assert obj.lifecycle_state in ("partially_resolved", "dormant")

    # Verify chronological lifecycle transitions in change_history
    lifecycle_changes = [
        ch for ch in report.final_state.change_history
        if ch.field_path.endswith(".lifecycle_state")
    ]
    assert [ch.triggering_turn_id for ch in lifecycle_changes] == [7, 12, 14, 17]
    assert [(ch.old_value, ch.new_value) for ch in lifecycle_changes] == [
        ("active", "resolved"),
        ("resolved", "reactivated"),
        ("reactivated", "partially_resolved"),
        ("partially_resolved", "dormant"),
    ]

    # 2. Timeline truth supersession verified
    all_facts = report.final_state.facts
    assert any(f.fact_key == "tentative_meeting_time" and f.status == "superseded" for f in all_facts)
    assert any(f.fact_key == "confirmed_meeting_time" and f.status == "active" and "Thursday At 3" in f.fact_value for f in all_facts)

    # 3. Persistent facts verified
    active_keys = {f.fact_key for f in all_facts if f.status == "active"}
    assert "timeline_horizon" in active_keys
    assert "spouse_involvement" in active_keys
    assert "contact_preference" in active_keys
    assert "scheduling_constraint" in active_keys
    assert "confirmed_meeting_time" in active_keys

    # 4. Decision structure verified
    ds = report.final_state.decision_structure
    assert not ds.decision_maker_present
    assert any(s.role == "wife" and s.presence == "absent" for s in ds.stakeholders)
    assert ds.timeline_horizon == "sometime next year"
    assert ds.urgency_level == "low"
    assert "Mornings unavailable" in ds.access_constraints
