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
