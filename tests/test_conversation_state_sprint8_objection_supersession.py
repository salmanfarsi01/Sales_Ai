import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_objections import ObjectionLifecycleEngine
from copilot.conversation_state_models import (
    ObjectionRecord,
    ObjectionLifecycleState,
    ConversationStateSnapshot,
    DimensionScores,
)
from copilot.conversation_conversion import MeetingConversionGateEngine
from copilot.conversation_scoring import ConversationScoringEngine


def _create_turn_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    agreement: float = 0.0,
    readiness: float = 0.5,
    future_lang: float = 0.0,
    boundary: float = 0.0,
    question_type: str = "none",
    strategy_tag: str = None,
    strategy_source: str = "none",
    recurrence_type: str = "none",
    recurrence_id: str = None,
):
    inference = DownstreamInferenceState(
        call_sid="CA_sprint8_supersede_call",
        timestamp_ms=2500 * turn_id,
        trust=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.2, confidence=0.8),
        pacing=DimensionScore(score=0.60, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.65, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_turn_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid="CA_sprint8_supersede_call",
        speaker_id=speaker_id,
        question_type=question_type,
        agreement_score=agreement,
        boundary_score=boundary,
        future_language_score=future_lang,
        recurrence_type=recurrence_type,
        recurrence_id=recurrence_id,
    )

    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
        salesperson_strategy_tag=strategy_tag,
        salesperson_strategy_source=strategy_source,
    )


def test_objection_lifecycle_state_superseded_enum_and_lineage():
    """Verify 'superseded' is a valid ObjectionLifecycleState and fields store lineage."""
    engine = ObjectionLifecycleEngine()

    bundle1 = _create_turn_bundle(1, "client", "Just looking around right now, not ready")
    engine.evaluate_turn(bundle1, current_version=0)

    active_objs = engine.get_active_objections()
    assert len(active_objs) == 1
    obj1 = active_objs[0]
    assert obj1.lifecycle_state == "unresolved"
    assert obj1.canonical_category == "general_hesitation"

    # Explicitly supersede
    superseded_record, change, next_ver = engine.supersede_objection(
        objection_id=obj1.objection_id,
        superseded_by_objection_id="obj_root_spouse",
        superseded_at_turn_id=2,
        reason="Higher-order decision maker objection takes precedence",
    )

    assert superseded_record is not None
    assert superseded_record.lifecycle_state == "superseded"
    assert superseded_record.superseded_by_objection_id == "obj_root_spouse"
    assert superseded_record.superseded_at_turn_id == 2

    # Query active vs superseded
    assert len(engine.get_active_objections()) == 0
    superseded_list = engine.get_superseded_objections()
    assert len(superseded_list) == 1
    assert superseded_list[0].objection_id == obj1.objection_id

    # Check StateChangeRecord
    assert change is not None
    assert change.field_path == f"objections.{obj1.objection_id}.lifecycle_state"
    assert change.old_value == "unresolved"
    assert change.new_value == "superseded"
    assert change.reason == "Higher-order decision maker objection takes precedence"


def test_autonomous_supersession_general_hesitation_by_root_authority():
    """Verify generic hesitation is superseded when a concrete root objection (spouse_authority) is raised."""
    manager = ConversationStateManager(call_sid="CA_sprint8_supersede_hesitation")

    # Turn 1: Generic hesitation
    b1 = _create_turn_bundle(1, "client", "Just looking around right now, not ready")
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    assert s1.objections[0].canonical_category == "general_hesitation"
    assert s1.objections[0].lifecycle_state == "unresolved"

    # Turn 2: Root objection: spouse authority
    b2 = _create_turn_bundle(
        2, "client", "I need to talk to my wife first. She makes the decisions."
    )
    s2 = manager.process_turn_bundle(b2)

    # Should have 2 objections tracked: 1 superseded, 1 active
    assert len(s2.objections) == 2
    hesitation = next(o for o in s2.objections if o.canonical_category == "general_hesitation")
    spouse = next(o for o in s2.objections if o.canonical_category == "spouse_authority")

    assert hesitation.lifecycle_state == "superseded"
    assert hesitation.superseded_by_objection_id == spouse.objection_id
    assert hesitation.superseded_at_turn_id == 2

    assert spouse.lifecycle_state == "unresolved"
    assert s2.get_active_objections() == [spouse]
    assert s2.get_superseded_objections() == [hesitation]


def test_autonomous_supersession_transactional_by_decision_to_stay():
    """Verify transactional fee objection is superseded when prospect decides to stay / take off market."""
    manager = ConversationStateManager(call_sid="CA_sprint8_supersede_stay")

    # Turn 1: Commission fee objection
    b1 = _create_turn_bundle(1, "client", "Your 6 percent commission is too high")
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    assert s1.objections[0].canonical_category == "commission_fee"
    assert s1.objections[0].lifecycle_state == "unresolved"

    # Turn 2: Prospect decides to stay
    b2 = _create_turn_bundle(
        2, "client", "Look, we talked it over and we have decided to stay and not selling anymore."
    )
    s2 = manager.process_turn_bundle(b2)

    assert len(s2.objections) == 1
    fee_obj = s2.objections[0]
    assert fee_obj.lifecycle_state == "superseded"
    assert fee_obj.superseded_by_objection_id == "decision_to_stay"
    assert fee_obj.superseded_at_turn_id == 2
    assert len(s2.get_active_objections()) == 0
    assert len(s2.get_superseded_objections()) == 1


def test_superseded_objection_does_not_fail_conversion_gate():
    """Verify that superseded objections do NOT fail Gate Condition 3 (objections_resolved_or_partial)."""
    snapshot = ConversationStateSnapshot(
        call_sid="CA_sprint8_gate_test",
        state_version=3,
        dimensions=DimensionScores(
            trust=0.75,
            emotion_tension=0.20,
            emotion_valence=0.30,
            engagement=0.80,
            momentum=0.70,
            readiness=0.85,
        ),
        objections=[
            ObjectionRecord(
                objection_id="obj_001",
                canonical_category="general_hesitation",
                initial_statement="I need time to think",
                latest_statement="I need time to think",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="superseded",
                superseded_by_objection_id="decision_to_stay",
                superseded_at_turn_id=2,
            )
        ],
    )

    gate_engine = MeetingConversionGateEngine()
    bundle = _create_turn_bundle(2, "client", "We have decided to stay.", agreement=0.6, readiness=0.85)
    gate = gate_engine.evaluate_gate(bundle=bundle, current_state=snapshot)

    cond3 = next(c for c in gate.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3.met is True, f"Condition 3 failed: {cond3.reason}"
    assert "superseded" in cond3.reason


def test_superseded_objection_does_not_trigger_readiness_blocker_cap():
    """Verify ReadinessBreakdown Blocker 4 (unresolved_objection) does not cap readiness for superseded objections."""
    snapshot = ConversationStateSnapshot(
        call_sid="CA_sprint8_readiness_test",
        state_version=2,
        dimensions=DimensionScores(
            trust=0.85,
            emotion_tension=0.15,
            emotion_valence=0.50,
            engagement=0.80,
            momentum=0.75,
            readiness=0.90,
        ),
        objections=[
            ObjectionRecord(
                objection_id="obj_fee",
                canonical_category="commission_fee",
                initial_statement="6 percent is too high",
                latest_statement="6 percent is too high",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="superseded",
                superseded_by_objection_id="decision_to_stay",
                superseded_at_turn_id=2,
            )
        ],
    )

    gate_engine = MeetingConversionGateEngine()
    scoring_engine = ConversationScoringEngine()
    bundle = _create_turn_bundle(2, "client", "Okay, sounds good.", agreement=0.8, readiness=0.90)
    gate = gate_engine.evaluate_gate(bundle=bundle, current_state=snapshot)

    readiness_breakdown = scoring_engine.compute_readiness(
        bundle=bundle,
        current_state=snapshot,
    )

    assert "unresolved_objection" not in readiness_breakdown.active_blocker_caps


def test_momentum_family_3_awards_full_score_when_all_superseded():
    """Verify Momentum Family 3 (Objection Movement) awards 100.0 when all objections are resolved or superseded."""
    snapshot = ConversationStateSnapshot(
        call_sid="CA_sprint8_momentum_test",
        state_version=2,
        dimensions=DimensionScores(trust=0.7, emotion_tension=0.2, emotion_valence=0.3, engagement=0.7, momentum=0.7, readiness=0.7),
        objections=[
            ObjectionRecord(
                objection_id="obj_old",
                canonical_category="general_hesitation",
                initial_statement="I am not sure",
                latest_statement="I am not sure",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="superseded",
                superseded_by_objection_id="obj_new",
                superseded_at_turn_id=2,
            )
        ],
    )

    scoring_engine = ConversationScoringEngine()
    bundle = _create_turn_bundle(2, "salesperson", "Understood.", agreement=0.5)
    mom = scoring_engine.compute_momentum(
        bundle=bundle,
        current_state=snapshot,
    )

    # Family 3 is objection movement. When all are superseded or resolved, score must be 100.0
    assert mom.family_scores.get("objection_movement") == 100.0


def test_momentum_family_3_collapses_when_superseded_by_decision_to_stay():
    """Verify Momentum Family 3 collapses to 0.0 and push strength exits when superseded by decision_to_stay."""
    snapshot = ConversationStateSnapshot(
        call_sid="CA_sprint8_stay_momentum_test",
        state_version=3,
        dimensions=DimensionScores(trust=0.7, emotion_tension=0.2, emotion_valence=0.3, engagement=0.7, momentum=0.7, readiness=0.7),
        objections=[
            ObjectionRecord(
                objection_id="obj_fee",
                canonical_category="commission_fee",
                initial_statement="6 percent is too high",
                latest_statement="6 percent is too high",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="superseded",
                superseded_by_objection_id="decision_to_stay",
                superseded_at_turn_id=2,
            )
        ],
    )

    scoring_engine = ConversationScoringEngine()
    gate_engine = MeetingConversionGateEngine()
    bundle = _create_turn_bundle(2, "client", "We have decided to stay and not sell anymore.", agreement=0.5)

    # 1. Momentum Family 3 must collapse to 0.0 (deal dead-end, not a successful resolution)
    mom = scoring_engine.compute_momentum(bundle=bundle, current_state=snapshot)
    assert mom.family_scores.get("objection_movement") == 0.0

    # 2. Conversion Gate Condition 4 (clear_value_reason) must fail because transaction value is void
    gate = gate_engine.evaluate_gate(bundle=bundle, current_state=snapshot)
    cond4 = next(c for c in gate.conditions if c.condition_name == "clear_value_reason")
    assert cond4.met is False
    assert "decided to stay" in cond4.reason
    assert gate.is_open is False

    # 3. Push Strength must select respect_record_exit
    push = gate_engine.evaluate_push_strength(bundle=bundle, current_state=snapshot, gate=gate)
    assert push.state == "respect_record_exit"
    assert "decided to stay" in push.rationale


def test_sim_mu5du135_full_8_turn_supersession_flow():
    """End-to-end regression verifying the exact 8-turn dialogue from sim_mu5du135.

    Guarantees:
    - Turn 2: 'general_hesitation' registered from natural hesitation phrasing.
    - Turn 4: 'spouse_authority' registered, decision_structure populated with absent wife,
              and Trigger 1 fires (general_hesitation superseded by spouse_authority).
    - Turn 6: 'commission_fee' registered (now 2 active objections: spouse_authority & commission_fee).
    - Turn 8: 'decision_to_stay' fires, simultaneously superseding BOTH active objections,
              recording decision_to_stay fact, collapsing Momentum Family 3 to 0.0,
              closing the gate with explicit void reason, and setting push strength to respect_record_exit.
    """
    manager = ConversationStateManager(call_sid="sim_mu5du135_test")

    # Turn 1: Agent intro
    b1 = _create_turn_bundle(
        1, "salesperson", "Hi Daniel, I noticed your home was listed previously — how's everything going with the sale?"
    )
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 0

    # Turn 2: Prospect generic hesitation
    b2 = _create_turn_bundle(
        2, "client", "Yeah, honestly we're just not sure this is the right time anymore."
    )
    s2 = manager.process_turn_bundle(b2)
    assert len(s2.objections) == 1
    assert s2.objections[0].canonical_category == "general_hesitation"
    assert s2.objections[0].lifecycle_state == "unresolved"

    # Turn 3: Agent clarifying question
    b3 = _create_turn_bundle(
        3, "salesperson", "That's totally understandable. Can you tell me more about what's giving you pause?"
    )
    s3 = manager.process_turn_bundle(b3)
    assert len(s3.objections) == 1

    # Turn 4: Prospect spousal disclosure & authority objection
    b4 = _create_turn_bundle(
        4, "client", "Well, my wife would really need to be part of this conversation too before we go any further."
    )
    s4 = manager.process_turn_bundle(b4)

    # 1. Decision structure must be populated with absent spouse
    assert s4.decision_structure.decision_maker_present is False
    assert len(s4.decision_structure.stakeholders) == 1
    stakeholder = s4.decision_structure.stakeholders[0]
    assert stakeholder.role == "wife"
    assert stakeholder.presence == "absent"

    # 2. Objections: spouse_authority registered, general_hesitation superseded
    assert len(s4.objections) == 2
    hesitation = next(o for o in s4.objections if o.canonical_category == "general_hesitation")
    spouse = next(o for o in s4.objections if o.canonical_category == "spouse_authority")
    assert hesitation.lifecycle_state == "superseded"
    assert hesitation.superseded_by_objection_id == spouse.objection_id
    assert hesitation.superseded_at_turn_id == 4
    assert spouse.lifecycle_state == "unresolved"
    assert len(s4.get_active_objections()) == 1

    # Turn 5: Agent inquiry
    b5 = _create_turn_bundle(
        5, "salesperson", "Of course, happy to loop her in. In the meantime, what commission structure were you expecting to pay?"
    )
    s5 = manager.process_turn_bundle(b5)

    # Turn 6: Prospect fee objection
    b6 = _create_turn_bundle(
        6, "client", "Honestly the commission fee last time felt way too high for what we got out of it."
    )
    s6 = manager.process_turn_bundle(b6)

    # Now 3 tracked objections: 1 superseded (hesitation), 2 active (spouse_authority & commission_fee)
    assert len(s6.objections) == 3
    fee = next(o for o in s6.objections if o.canonical_category == "commission_fee")
    assert fee.lifecycle_state == "unresolved"
    active_objs = s6.get_active_objections()
    assert len(active_objs) == 2
    assert {o.canonical_category for o in active_objs} == {"spouse_authority", "commission_fee"}

    # Turn 7: Agent explanation
    b7 = _create_turn_bundle(
        7, "salesperson", "I hear you — we can walk through exactly what that fee covers if that would help."
    )
    s7 = manager.process_turn_bundle(b7)

    # Turn 8: Prospect decision to stay in the house and cancel sale
    b8 = _create_turn_bundle(
        8, "client", "Actually, you know what, we've decided to just stay in the house. We're not going to sell after all."
    )
    s8 = manager.process_turn_bundle(b8)

    # 1. Fact recorded for decision_to_stay
    stay_facts = [f for f in s8.facts if f.fact_key == "decision_to_stay" and f.status == "active"]
    assert len(stay_facts) == 1

    # 2. BOTH active objections (spouse_authority AND commission_fee) must be superseded by decision_to_stay!
    assert len(s8.get_active_objections()) == 0
    superseded_objs = s8.get_superseded_objections()
    assert len(superseded_objs) == 3

    spouse_final = next(o for o in s8.objections if o.canonical_category == "spouse_authority")
    fee_final = next(o for o in s8.objections if o.canonical_category == "commission_fee")
    assert spouse_final.lifecycle_state == "superseded"
    assert spouse_final.superseded_by_objection_id == "decision_to_stay"
    assert fee_final.lifecycle_state == "superseded"
    assert fee_final.superseded_by_objection_id == "decision_to_stay"

    # 3. Momentum Family 3 (Objection Movement) must collapse to 0.0
    assert s8.momentum.family_scores["objection_movement"] == 0.0

    # 4. Conversion Gate Condition 4 (clear_value_reason) must fail with explicit reason
    cond4 = next(c for c in s8.conversion_gate.conditions if c.condition_name == "clear_value_reason")
    assert cond4.met is False
    assert "decided to stay and not sell" in cond4.reason
    assert s8.conversion_gate.is_open is False

    # 5. Push Strength must transition to respect_record_exit
    assert s8.push_strength.state == "respect_record_exit"
    assert "decided to stay" in s8.push_strength.rationale

    # 6. Change history must include Turn 8 state changes!
    turn_8_changes = [ch for ch in s8.change_history if ch.triggering_turn_id == 8]
    assert len(turn_8_changes) >= 2

