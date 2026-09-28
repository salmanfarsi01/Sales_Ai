from __future__ import annotations

import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStakeholder,
    ConversionEventStatus,
)
from copilot.conversation_conversion import MeetingConversionGateEngine
from copilot.conversation_scoring import ConversationScoringEngine


def _create_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.50,
    emotion_valence: float = 0.0,
    emotion_tension: float = 0.20,
    engagement: float = 0.50,
    agreement: float = 0.50,
    specificity: float = 0.50,
    future_lang: float = 0.50,
    boundary: float = 0.0,
    contact_preference: str = "none",
    call_sid: str = "CA_gate_tristate_test",
):
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=0.8),
        pacing=DimensionScore(score=0.50, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.50, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.50, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        agreement_score=agreement,
        specificity_score=specificity,
        future_language_score=future_lang,
        contact_preference=contact_preference,
    )

    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
    )


def test_empty_call_gate_is_closed_and_conditions_unknown():
    """Validates that a clean slate / empty call shows the conversion gate closed,
    with conditions lacking prospect-originated evidence marked as unknown.
    """
    engine = MeetingConversionGateEngine()
    snapshot = ConversationStateSnapshot(call_sid="CA_empty_call")
    bundle = _create_bundle(
        turn_id=1,
        speaker_id="salesperson",
        text="Would Thursday work for a quick call to go over some options?",
    )

    gate = engine.evaluate_gate(bundle=bundle, current_state=snapshot)

    assert gate.is_open is False
    assert gate.status == "closed"
    assert len(gate.unknown_conditions) > 0
    assert "clear_value_reason" in gate.unknown_conditions
    assert "plausible_logistics" in gate.unknown_conditions
    assert "decision_maker_aligned" in gate.unknown_conditions
    assert "objections_resolved_or_partial" in gate.unknown_conditions
    assert gate.confidence <= 0.40


def test_empty_or_early_call_readiness_insufficient_evidence_not_57_2():
    """Validates that readiness built from defaults is flagged as insufficient_evidence
    with 0.0 confidence and capped at 0.0, avoiding the unearned 57.2 score.
    """
    engine = ConversationScoringEngine()
    snapshot = ConversationStateSnapshot(call_sid="CA_scoring_empty")
    bundle = _create_bundle(
        turn_id=1,
        speaker_id="salesperson",
        text="Hi, this is Alex with Apex Realty.",
    )

    readiness = engine.compute_readiness(bundle=bundle, current_state=snapshot)

    assert readiness.insufficient_evidence is True
    assert readiness.confidence == 0.0
    assert readiness.readiness_score is None
    assert "insufficient_evidence" in readiness.active_blocker_caps
    assert readiness.readiness_score != 57.2


def test_early_call_soft_preference_gate_remains_closed_and_readiness_insufficient():
    """Replicates Turn 2 of sim_mukmxsle:
    Prospect says 'Please don't start texting me every day.'
    Gate must be closed, and readiness must have insufficient evidence (0.0, not 57.2).
    """
    manager = ConversationStateManager(call_sid="CA_sim_mukmxsle_check")

    # Turn 1: Salesperson pitch
    t1 = _create_bundle(
        turn_id=1,
        speaker_id="salesperson",
        text="Would Thursday work for a quick call to go over some options?",
    )
    manager.process_turn_bundle(t1)

    # Turn 2: Prospect soft contact preference
    t2 = _create_bundle(
        turn_id=2,
        speaker_id="client",
        text="Please don't start texting me every day.",
        contact_preference="reduced_frequency",
        agreement=0.40,
        specificity=0.30,
        future_lang=0.10,
    )
    s2 = manager.process_turn_bundle(t2)

    assert s2.conversion_gate is not None
    assert s2.conversion_gate.is_open is False
    assert s2.conversion_gate.status == "closed"
    assert "clear_value_reason" in s2.conversion_gate.unknown_conditions
    assert "plausible_logistics" in s2.conversion_gate.unknown_conditions

    assert s2.readiness is not None
    assert s2.readiness.insufficient_evidence is True
    assert s2.readiness.readiness_score is None
    assert s2.readiness.readiness_score != 57.2


def test_condition_met_requires_prospect_originated_evidence_turn_ids():
    """Validates that a 'met' condition carries prospect turn IDs as evidence."""
    engine = MeetingConversionGateEngine()
    snapshot = ConversationStateSnapshot(
        call_sid="CA_evidence_test",
        prospect_turn_ids=[2],
    )
    bundle = _create_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me, let's meet then.",
        agreement=0.85,
        specificity=0.80,
        future_lang=0.75,
    )

    gate = engine.evaluate_gate(bundle=bundle, current_state=snapshot)

    assert gate.is_open is True
    for cond in gate.conditions:
        assert cond.status == "met"
        assert cond.met is True
        assert len(cond.evidence_turn_ids) > 0
        assert 2 in cond.evidence_turn_ids
