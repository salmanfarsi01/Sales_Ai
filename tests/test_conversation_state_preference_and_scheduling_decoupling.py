import pytest
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_materiality import MaterialityFilter
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_models import ObjectionLifecycleState


def _make_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    boundary_score: float = 0.0,
    contact_pref: str = "none",
    agreement_score: float = 0.50,
    strategy_tag: str = None,
):
    inference = DownstreamInferenceState(
        call_sid="CA_test_decoupling",
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=0.80, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.2, confidence=0.8),
        pacing=DimensionScore(score=0.60, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.65, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.50, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_turn_{turn_id}"],
    )
    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid="CA_test_decoupling",
        speaker_id=speaker_id,
        question_type="none",
        agreement_score=agreement_score,
        boundary_score=boundary_score,
        recurrence_type="none",
    )
    bundle = extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
        salesperson_strategy_tag=strategy_tag,
    )
    bundle.contact_preference = contact_pref
    return bundle


def test_turn_15_soft_contact_preference_pure_decoupling():
    """Client Item #2:
    Soft contact preference must be classified as contact_compliance + facts,
    and MUST NEVER emit 'objections' or objection-flavored reasoning string.
    """
    filter_engine = MaterialityFilter()
    b15 = _make_bundle(
        turn_id=15,
        speaker_id="client",
        text="Please don't start texting me every day before we meet, by the way.",
    )

    res = filter_engine.classify_turn(b15)
    assert res.is_material is True
    assert "contact_compliance" in res.affected_targets
    assert "facts" in res.affected_targets
    assert "objections" not in res.affected_targets
    assert "objection" not in res.reasoning.lower()
    assert "canonical marker" not in res.reasoning.lower()
    assert "Soft communication preference" in res.reasoning


def test_turn_16_logistical_scheduling_constraint_pure_decoupling():
    """Client Item #3:
    Logistical scheduling availability constraint must be classified as
    decision_structure + facts, and MUST NEVER emit 'objections' or objection-flavored reasoning.
    """
    filter_engine = MaterialityFilter()
    b16 = _make_bundle(
        turn_id=16,
        speaker_id="client",
        text="Mornings don't really work for us either, just so you know.",
    )

    res = filter_engine.classify_turn(b16)
    assert res.is_material is True
    assert "decision_structure" in res.affected_targets
    assert "facts" in res.affected_targets
    assert "objections" not in res.affected_targets
    assert "objection" not in res.reasoning.lower()
    assert "canonical marker" not in res.reasoning.lower()
    assert "Logistical scheduling constraint" in res.reasoning


def test_explicit_meeting_resistance_remains_objection():
    """Client Item #3 Guard:
    If a scheduling statement is accompanied by explicit meeting refusal or resistance,
    it MUST map to objections.
    """
    filter_engine = MaterialityFilter()
    b_resist = _make_bundle(
        turn_id=16,
        speaker_id="client",
        text="I don't want to meet with you, mornings or afternoons.",
    )

    res = filter_engine.classify_turn(b_resist)
    assert res.is_material is True
    assert "objections" in res.affected_targets


def test_mixed_canonical_objection_with_soft_preference():
    """If a client turn expresses both an explicit canonical objection and a soft boundary,
    both objections and contact_compliance should be recorded.
    """
    filter_engine = MaterialityFilter()
    b_mixed = _make_bundle(
        turn_id=7,
        speaker_id="client",
        text="Your 6 percent commission fee is way too high. And please don't text me constantly.",
    )

    res = filter_engine.classify_turn(b_mixed)
    assert res.is_material is True
    assert "objections" in res.affected_targets
    assert "contact_compliance" in res.affected_targets
    assert "facts" in res.affected_targets


def test_state_manager_turn_15_and_16_lifecycle_safety():
    """Full lifecycle integration test:
    Demonstrates that in a conversation with an existing active objection (from Turn 12),
    Turn 15 and Turn 16 do NOT trigger objection updates, do NOT add 'objections' to materiality,
    and correctly update contact compliance and decision structure/facts.
    """
    manager = ConversationStateManager(call_sid="CA_decoupling_sim")

    # Turn 12: Client raises hesitation/objection
    b12 = _make_bundle(
        turn_id=12,
        speaker_id="client",
        text="I guess I'm just worried this isn't really the right move for us financially with everything going on.",
    )
    s12 = manager.process_turn_bundle(b12)
    assert len(s12.objections) >= 1
    initial_obj_count = len(s12.objections)

    # Turn 13: Salesperson reframe
    b13 = _make_bundle(
        turn_id=13,
        speaker_id="salesperson",
        text="That makes total sense — this is just an initial conversation to see if it even makes sense.",
        strategy_tag="perspective_taking",
    )
    s13 = manager.process_turn_bundle(b13)

    # Turn 14: Client agrees
    b14 = _make_bundle(
        turn_id=14,
        speaker_id="client",
        text="Okay, that seems fair enough.",
        agreement_score=0.85,
    )
    s14 = manager.process_turn_bundle(b14)

    # Turn 15: Client expresses soft contact preference
    b15 = _make_bundle(
        turn_id=15,
        speaker_id="client",
        text="Please don't start texting me every day before we meet, by the way.",
    )
    s15 = manager.process_turn_bundle(b15)

    # Verify Turn 15 materiality
    mat15 = manager.last_materiality
    assert mat15 is not None
    assert "contact_compliance" in mat15.affected_targets
    assert "facts" in mat15.affected_targets
    assert "objections" not in mat15.affected_targets
    assert "objection" not in mat15.reasoning.lower()

    # Verify contact compliance updated
    assert len(s15.contact_compliance.contact_preferences) >= 1
    assert any(f.fact_key == "contact_preference" for f in s15.facts)
    # Verify no new objection created
    assert len(s15.objections) == initial_obj_count

    # Turn 16: Client expresses logistical scheduling constraint
    b16 = _make_bundle(
        turn_id=16,
        speaker_id="client",
        text="Mornings don't really work for us either, just so you know.",
    )
    s16 = manager.process_turn_bundle(b16)

    # Verify Turn 16 materiality
    mat16 = manager.last_materiality
    assert mat16 is not None
    assert "decision_structure" in mat16.affected_targets
    assert "facts" in mat16.affected_targets
    assert "objections" not in mat16.affected_targets
    assert "objection" not in mat16.reasoning.lower()

    # Verify decision structure access constraints updated
    assert "Mornings unavailable" in s16.decision_structure.access_constraints
    # Verify scheduling constraint fact recorded
    assert any(f.fact_key == "scheduling_constraint" for f in s16.facts)
    # Verify no new objection created
    assert len(s16.objections) == initial_obj_count
