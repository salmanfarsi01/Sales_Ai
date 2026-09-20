from __future__ import annotations

import pytest
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_conversion import MeetingConversionGateEngine
from copilot.conversation_replay import ConversationReplayEngine
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle


def _make_turn_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.50,
    emotion_valence: float = 0.0,
    emotion_tension: float = 0.20,
    engagement: float = 0.50,
    readiness: float = 0.50,
    momentum: float = 0.50,
    agreement: float = 0.50,
    specificity: float = 0.60,
    call_sid: str = "sim_harmony_test",
):
    ts_ms = 3000 * turn_id
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=ts_ms,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=0.85),
        pacing=DimensionScore(score=0.60, confidence=0.80, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=momentum, confidence=0.80, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )
    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=0.0,
        agreement_score=agreement,
        specificity_score=specificity,
        future_language_score=0.50,
        contact_preference="none",
    )
    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
    )


def test_push_strength_consults_objection_driver_strategic_target_item_9():
    """Validates Item 9 Fix:
    When a non-blocking objection ('general_hesitation') is active on an appointment target:
    - Gate Condition 3 passes (met=True, 'No blocking objections for appointment')
    - evaluate_push_strength() MUST NOT recommend 'direct_ask'!
    - It must recommend 'resolve_then_ask' with actionable guidance derived directly
      from lead_obj.driver_layer.strategic_target ('deploy gentle diagnostic inquiry...').
    """
    manager = ConversationStateManager(call_sid="CA_item9_driver_push", conversion_target="appointment")

    # Turn 1: Agent intro
    b1 = _make_turn_bundle(1, "salesperson", "Hi Daniel, how is everything going with the home?")
    manager.process_turn_bundle(b1)

    # Turn 2: Prospect hesitation
    # Use healthy trust (0.70) so that gate conditions would otherwise allow direct_ask if blind
    b2 = _make_turn_bundle(
        2,
        "client",
        "Yeah, honestly we're just not sure this is the right time anymore.",
        trust=0.70,
        emotion_tension=0.15,
        engagement=0.70,
        readiness=0.60,
    )
    snap2 = manager.process_turn_bundle(b2)

    # 1. Objection was captured with driver layer
    assert len(snap2.objections) == 1
    obj = snap2.objections[0]
    assert obj.canonical_category == "general_hesitation"
    assert obj.driver_layer is not None
    assert obj.driver_layer.underlying_driver == "unexpressed_underlying_concern"
    assert obj.driver_layer.strategic_target == "deploy_gentle_diagnostic_inquiry_to_surface_root_constraint"

    # 2. Gate condition 3 passes (non-blocking for appointment)
    cond3 = next(c for c in snap2.conversion_gate.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3.met is True
    assert "No blocking objections for 'appointment'" in cond3.reason
    assert "general_hesitation" in cond3.reason

    # 3. CRITICAL: Push strength must NOT recommend direct_ask!
    # It must recommend resolve_then_ask and quote the driver's strategic target
    push = snap2.push_strength
    assert push is not None
    assert push.state == "resolve_then_ask", f"Expected 'resolve_then_ask', got '{push.state}'"
    assert "deploy gentle diagnostic inquiry to surface root constraint" in push.recommended_action
    assert "unexpressed_underlying_concern" in push.rationale
    assert "defensive reactance" in push.rationale


def test_process_overwhelm_driver_yields_reduce_friction_reask():
    """Validates that logistical/overwhelm drivers recommend reduce_friction_reask."""
    manager = ConversationStateManager(call_sid="CA_overwhelm_driver", conversion_target="appointment")

    b1 = _make_turn_bundle(1, "salesperson", "Would you be open to an initial walkthrough?")
    manager.process_turn_bundle(b1)

    b2 = _make_turn_bundle(
        2,
        "client",
        "Honestly it is just way too much stress, packing and clutter to deal with right now.",
        trust=0.70,
        emotion_tension=0.20,
        engagement=0.70,
        readiness=0.60,
    )
    snap2 = manager.process_turn_bundle(b2)

    assert len(snap2.objections) == 1
    obj = snap2.objections[0]
    assert obj.driver_layer is not None
    assert obj.driver_layer.underlying_driver == "process_overwhelm"
    assert obj.driver_layer.strategic_target == "decompose_process_into_low_friction_bite_sized_milestones"

    push = snap2.push_strength
    assert push is not None
    assert push.state == "reduce_friction_reask"
    assert "decompose process into low friction bite sized milestones" in push.recommended_action


def test_synthetic_dialogue_replay_dimension_stability_item_8():
    """Validates Item 8 Fix:
    Synthetic dialogue turns without explicit dimension metrics maintain previous turn's
    dimension scores and do NOT artificially jump +0.20 on hesitation turns.
    """
    engine = ConversationReplayEngine()
    raw_turns = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi Daniel, checking in on the property."},
        {"turn_id": 2, "speaker_id": "client", "text": "Yeah, honestly we're just not sure this is the right time anymore."},
    ]
    report = engine.replay_dialogue_turns(call_sid="sim_dim_stability_test", raw_turns=raw_turns, save_report=False)

    step1 = report.timeline[0]
    step2 = report.timeline[1]

    # Turn 1 baseline trust is 0.50
    assert step1.state_after.dimensions.trust == 0.50

    # Turn 2 without explicit trust input must NOT jump to 0.70
    assert step2.state_after.dimensions.trust == 0.50

    # Turn 2 materiality reasoning must NOT include artificial acoustic shift override
    mat_reason = step2.materiality.reasoning
    assert "Deterministic Override: behavioral/acoustic shift detected" not in mat_reason


def test_recurrence_escalation_for_non_blocking_objections_item_1():
    """Validates Item 1 Severity Rule:
    Non-blocking objections (e.g. general_hesitation for appointment) remain non-blocking
    for up to max_non_blocking_recurrence (2), but escalate to blocking on recurrence >= 3.
    """
    manager = ConversationStateManager(call_sid="CA_recurrence_escalation", conversion_target="appointment")

    # Turn 1: Initial hesitation (recurrence 1) -> Non-blocking
    b1 = _make_turn_bundle(1, "client", "Honestly, we're not sure this is the right time.")
    s1 = manager.process_turn_bundle(b1)
    cond_t1 = next(c for c in s1.conversion_gate.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond_t1.met is True
    assert "No blocking objections" in cond_t1.reason

    # Turn 2: Agent responds
    b2 = _make_turn_bundle(2, "salesperson", "Understood. What is making you hesitate?")
    manager.process_turn_bundle(b2)

    # Turn 3: Repeated hesitation (recurrence 2) -> Still allowed under max_non_blocking_recurrence=2
    b3 = _make_turn_bundle(3, "client", "We really just don't think it is the right time right now.")
    s3 = manager.process_turn_bundle(b3)
    cond_t3 = next(c for c in s3.conversion_gate.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond_t3.met is True

    # Turn 4: Agent responds again
    b4 = _make_turn_bundle(4, "salesperson", "I completely understand your concern.")
    manager.process_turn_bundle(b4)

    # Turn 5: 3rd occurrence of hesitation (recurrence 3) -> ESCALATES TO BLOCKING!
    b5 = _make_turn_bundle(5, "client", "Again, we are definitely not ready at this time.")
    s5 = manager.process_turn_bundle(b5)
    cond_t5 = next(c for c in s5.conversion_gate.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond_t5.met is False
    assert "escalated due to recurrence >= 3" in cond_t5.reason
