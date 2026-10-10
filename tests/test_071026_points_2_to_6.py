"""Regression tests for Client Feedback 071026 Points 2 to 6:
Core Intelligence and Strategic Decision Component.

Points Verified:
- Point 2: Non-material turns, active strategy provenance preservation, HOLD action, prompt suppression.
- Point 3: Unresolved concerns after appointment agreement (Turns 6-7 fee objection, appointment protection,
           failed strategy tracking in do_not_do, anti-repetition, Turn 8 meeting deferral).
- Point 4: Appointment lifecycle (identity preservation across turns, AM/PM clarification not a reschedule).
- Point 5: Unmeasured signals (gate distinguishes defaults from measured evidence, appointment agreement does not fabricate trust).
- Point 6: Pressure-field consistency (ConversationState, PushStrengthRecommendation, StrategicDecision, and summaries report pressure).
- Sole-decision-maker statement cannot override later spouse involvement requirement.
"""

import json
from pathlib import Path
import pytest

from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversionEventStatus,
    ObjectionLifecycleState,
    PushStrengthRecommendation,
)
from copilot.core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
    PushStrengthValue,
)
from copilot.conversation_state_manager import ConversationStateManager
from copilot.core_decision_manager import CoreDecisionManager
from copilot.conversation_replay import ConversationReplayEngine
from copilot.conversation_conversion import is_ampm_clarification


@pytest.fixture(scope="module")
def sim_muy04bh0_replay_data():
    """Runs replay on the fee complaint script sim_muy04bh0 and provides timeline and final report."""
    script_path = Path(__file__).resolve().parent.parent / "data" / "script_sim_muy04bh0_fee_complaint.json"
    assert script_path.exists(), f"Missing replay script: {script_path}"
    raw_turns = json.loads(script_path.read_text(encoding="utf-8"))

    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns(
        call_sid="sim_muy04bh0",
        raw_turns=raw_turns,
        save_report=True,
        run_semantic_analysis=False,
    )
    return rep


# =============================================================================
# Point 2: Non-Material Turns and Prompt Repetition
# =============================================================================

def test_point2_non_material_turn_carry_forward_and_prompt_suppression(sim_muy04bh0_replay_data):
    """Point 2: Turn 5 is classified as non-material.
    It must preserve the active strategy and its original provenance,
    select HOLD, and emit no new prompt. Carry-forward references must NOT be empty.
    """
    step5 = [s for s in sim_muy04bh0_replay_data.timeline if s.turn_id == 5][0]
    m5 = step5.materiality
    d5 = step5.strategic_decision

    # 1. Materiality classification
    assert m5.is_material is False, "Turn 5 ('That helps a little.') must be classified as non-material"
    assert len(m5.affected_targets) == 0

    # 2. Strategy selection: HOLD, protect posture, no push
    assert d5.primary_action == StrategicAction.HOLD
    assert d5.strategic_posture == "protect"
    assert str(d5.push_strength) == "none"

    # 3. Prompt suppression: no new prompt emitted
    assert d5.should_prompt is False, "Turn 5 must suppress prompt (should_prompt=False)"
    assert not d5.final_prompt_text
    assert not d5.gateway_fallback_stub

    # 4. Strategy provenance: carry-forward references must NOT be empty
    assert d5.carried_forward is True
    assert d5.carried_forward_from_turn_id == 3, "Turn 5 must reference original strategy turn (Turn 3)"
    assert d5.carried_forward_from_decision_id is not None
    assert "HOLD_NON_MATERIAL_TURN" in d5.reason_codes
    assert "STRATEGY_CARRIED_FORWARD" in d5.reason_codes


# =============================================================================
# Point 3: Unresolved Concerns After Appointment Agreement
# =============================================================================

def test_point3_persistent_fee_concern_protects_appointment(sim_muy04bh0_replay_data):
    """Point 3: Turns 6-7 record a persistent fee objection after Thursday at 4 was agreed.
    The decision must protect the appointment while responding to the current concern,
    reference the objection ID, exclude failed strategies in do_not_do, and avoid repeating
    the appointment confirmation sentence.
    """
    step6 = [s for s in sim_muy04bh0_replay_data.timeline if s.turn_id == 6][0]
    step7 = [s for s in sim_muy04bh0_replay_data.timeline if s.turn_id == 7][0]

    d6 = step6.strategic_decision
    d7 = step7.strategic_decision

    # Both turns protect the appointment without push pressure
    assert d6.strategic_posture == "protect"
    assert str(d6.push_strength) == "none"
    assert d7.strategic_posture == "protect"
    assert str(d7.push_strength) == "none"

    # Reference the active fee objection
    assert len(d6.referenced_objection_ids) > 0
    assert len(d7.referenced_objection_ids) > 0
    assert d6.referenced_objection_ids == d7.referenced_objection_ids

    # do_not_do includes failed strategy (hyperlocal_marketing_differentiation)
    assert "hyperlocal_marketing_differentiation" in d6.do_not_do or "repeat_rejected_explanation" in d6.do_not_do
    assert "hyperlocal_marketing_differentiation" in d7.do_not_do or "repeat_rejected_explanation" in d7.do_not_do

    # Prompts address the fee concern instead of repeating generic appointment confirmation
    prompt6 = d6.final_prompt_text or d6.gateway_fallback_stub or ""
    prompt7 = d7.final_prompt_text or d7.gateway_fallback_stub or ""
    assert "Perfect, I've noted Thursday at 4 for us." not in prompt6
    assert "Perfect, I've noted Thursday at 4 for us." not in prompt7
    assert any(term in prompt6.lower() for term in ("investment", "obligation", "commission", "fee", "numbers"))
    assert any(term in prompt7.lower() for term in ("investment", "obligation", "commission", "fee", "numbers"))


def test_point3_explicit_deferral_retains_concern_for_meeting(sim_muy04bh0_replay_data):
    """Point 3: On Turn 8, client explicitly defers fee discussion to the meeting
    ('ok leave it we will talk about it in our meeting Thursday at 4 pm').
    The system must retain that concern for follow-up and tag OBJECTION_DEFERRED_TO_MEETING.
    """
    step8 = [s for s in sim_muy04bh0_replay_data.timeline if s.turn_id == 8][0]
    d8 = step8.strategic_decision
    state8 = step8.state_after

    assert "OBJECTION_DEFERRED_TO_MEETING" in d8.reason_codes
    assert "RETAINED_FOR_FOLLOWUP" in d8.reason_codes

    # Objection record has deferred_to_meeting and retained_for_followup
    fee_objs = [o for o in state8.objections if "fee" in o.canonical_category.lower()]
    assert len(fee_objs) > 0
    fee_obj = fee_objs[0]
    assert fee_obj.deferred_to_meeting is True
    assert fee_obj.retained_for_followup is True


# =============================================================================
# Point 4: Appointment Lifecycle
# =============================================================================

def test_point4_appointment_identity_preserved_across_turns(sim_muy04bh0_replay_data):
    """Point 4: Confirmed appointment must NOT disappear on Turns 4-5,
    must NOT be recreated on Turn 6, and clarifying AM/PM must update existing event.
    """
    tl = sim_muy04bh0_replay_data.timeline
    step3 = [s for s in tl if s.turn_id == 3][0]
    step4 = [s for s in tl if s.turn_id == 4][0]
    step5 = [s for s in tl if s.turn_id == 5][0]
    step6 = [s for s in tl if s.turn_id == 6][0]
    step8 = [s for s in tl if s.turn_id == 8][0]

    # Turn 3 creates initial confirmed event
    ce3 = step3.state_after.conversion_event
    assert ce3 is not None
    assert ce3.status == ConversionEventStatus.CONFIRMED
    assert ce3.start_at == "Thursday At 4"
    event_id = ce3.event_id

    # Turn 4: appointment does NOT disappear
    ce4 = step4.state_after.conversion_event
    assert ce4 is not None
    assert ce4.event_id == event_id, "Turn 4 must preserve original event_id"
    assert ce4.status == ConversionEventStatus.CONFIRMED

    # Turn 5: appointment does NOT disappear
    ce5 = step5.state_after.conversion_event
    assert ce5 is not None
    assert ce5.event_id == event_id, "Turn 5 must preserve original event_id"
    assert ce5.status == ConversionEventStatus.CONFIRMED

    # Turn 6: fee complaint does NOT recreate a new event ID
    ce6 = step6.state_after.conversion_event
    assert ce6 is not None
    assert ce6.event_id == event_id, "Turn 6 must preserve original event_id, not recreate"
    assert ce6.status == ConversionEventStatus.CONFIRMED

    # Turn 8: AM/PM clarification updates start_at on existing event without creating a reschedule
    ce8 = step8.state_after.conversion_event
    assert ce8 is not None
    assert ce8.event_id == event_id, "Turn 8 must update existing appointment without recreating"
    assert ce8.start_at == "Thursday At 4 Pm"
    assert ce8.reversal_reason is None, "AM/PM clarification must not set reversal_reason='rescheduled'"

    # Only 1 unique event exists across the call
    events = step8.state_after.conversion_events
    assert len(events) == 1
    assert events[0].event_id == event_id


def test_point4_is_ampm_clarification_helper():
    """Point 4 helper validation: distinguishes unspecified AM/PM clarification from actual reschedules."""
    assert is_ampm_clarification("Thursday At 4", "Thursday At 4 Pm") is True
    assert is_ampm_clarification("Thursday 4", "Thursday 4 pm") is True
    assert is_ampm_clarification("Friday at 10", "Friday at 10 am") is True
    assert is_ampm_clarification("Thursday At 4", "Thursday At 4") is True

    # Real reschedules must return False
    assert is_ampm_clarification("Thursday At 4", "Friday At 4") is False
    assert is_ampm_clarification("Thursday At 4", "Thursday At 5") is False
    assert is_ampm_clarification("Thursday At 4", "Thursday At 10") is False


# =============================================================================
# Point 5: Unmeasured Signals and Trust Handling
# =============================================================================

def test_point5_gate_condition1_distinguishes_defaults_from_measured_trust(sim_muy04bh0_replay_data):
    """Point 5: Distinguish defaults from measured evidence throughout gate evaluation.
    When trust is unmeasured (default 0.50), gate condition 1 must describe trust as default,
    not 'healthy (0.50 >= 0.45)'.
    """
    tl = sim_muy04bh0_replay_data.timeline
    step3 = [s for s in tl if s.turn_id == 3][0]
    gate3 = step3.state_after.conversion_gate
    assert gate3 is not None
    assert len(gate3.conditions) > 0

    c1 = gate3.conditions[0]
    assert c1.condition_name == "trust_not_collapsing"
    assert "unmeasured default" in c1.reason.lower()
    assert "trust healthy" not in c1.reason.lower()


def test_point5_appointment_agreement_does_not_fabricate_measured_trust(sim_muy04bh0_replay_data):
    """Point 5: An explicit appointment agreement can support the appointment without
    establishing measured trust, fee acceptance, or decision authority.
    """
    step3 = [s for s in sim_muy04bh0_replay_data.timeline if s.turn_id == 3][0]
    state3 = step3.state_after

    # Trust dimension remains unmeasured
    assert state3.dimensions.trust_measured is False, "Appointment agreement must not set trust_measured=True"
    assert state3.dimensions.trust == 0.50

    # Fee objection is NOT resolved by appointment agreement
    fee_objs = [o for o in state3.objections if "fee" in o.canonical_category.lower()]
    assert len(fee_objs) > 0
    assert fee_objs[0].lifecycle_state == ObjectionLifecycleState.ACTIVE


# =============================================================================
# Point 6: Pressure-Field Consistency
# =============================================================================

def test_point6_push_strength_reports_pressure_consistently(sim_muy04bh0_replay_data):
    """Point 6: Apply separation consistently: action/posture describes strategy;
    push strength describes pressure ('none', 'low', 'moderate', 'high').
    """
    rep = sim_muy04bh0_replay_data
    cs = rep.conversion_summary

    # Summary push_strength reports pressure level
    assert cs["push_strength"] == "none"
    assert cs["strategic_posture"] == "protect"
    assert cs["closing_strategy"] == "confirm_and_protect"

    # PushStrengthRecommendation model
    final_ps = rep.timeline[-1].state_after.push_strength
    assert isinstance(final_ps, PushStrengthRecommendation)
    assert final_ps.pressure == "none"
    assert final_ps.push_strength == "none"
    assert final_ps.strategic_posture == "protect"
    assert final_ps.strategy == "confirm_and_protect"

    # StrategicDecision uses pressure
    final_dec = rep.timeline[-1].strategic_decision
    assert str(final_dec.push_strength) == "none"
    assert final_dec.strategic_posture == "protect"


# =============================================================================
# Multi-Stakeholder Invariant: Sole Decision Maker Cannot Override Later Requirement
# =============================================================================

def test_prior_sole_decision_maker_cannot_override_later_spouse_involvement():
    """Verification item: confirm that earlier sole-decision-maker statement cannot
    override later requirement for spouse involvement.
    """
    manager = ConversationStateManager(call_sid="sim_stakeholder_override_test")

    # Turn 1: Prospect claims to be sole decision maker
    from copilot.conversation_state_contract import (
        extract_behavioral_bundle,
        DownstreamInferenceState,
        SemanticFeatureSnapshot,
        DimensionScore,
        EmotionState,
    )

    def make_bundle(turn_id, speaker, text):
        inf = DownstreamInferenceState(
            call_sid="sim_stakeholder_override_test",
            timestamp_ms=turn_id * 1000,
            trust=DimensionScore(score=0.5, confidence=0.7, primary_horizon="last_20_30s"),
            emotion=EmotionState(expressed_valence=0.0, tension_level=0.1, confidence=0.7),
            pacing=DimensionScore(score=0.6, confidence=0.8, primary_horizon="current_utterance"),
            momentum=DimensionScore(score=0.6, confidence=0.8, primary_horizon="last_60_90s"),
            readiness=DimensionScore(score=0.6, confidence=0.7, primary_horizon="last_60_90s"),
            engagement=DimensionScore(score=0.6, confidence=0.7, primary_horizon="last_20_30s"),
            overall_confidence=0.75,
        )
        sem = SemanticFeatureSnapshot(
            utterance_id=f"utt_{turn_id}",
            call_sid="sim_stakeholder_override_test",
            speaker_id=speaker,
            agreement_score=0.6,
        )
        return extract_behavioral_bundle(turn_id=turn_id, speaker_id=speaker, utterance_text=text, inference_state=inf, semantic_snapshot=sem)

    b1 = make_bundle(1, "client", "I'm the one making this decision, no one else needs to sign off.")
    s1 = manager.process_turn_bundle(b1)
    assert s1.decision_structure.primary_decision_maker == "sole_decision_maker"
    assert s1.decision_structure.co_decision_required is False
    assert s1.decision_structure.decision_maker_present is True

    # Turn 2: Prospect clarifies wife must be involved before signing
    b2 = make_bundle(2, "client", "Actually, my wife would really need to be part of this conversation before we go any further.")
    s2 = manager.process_turn_bundle(b2)
    assert s2.decision_structure.co_decision_required is True, "Later spouse requirement must override earlier sole decision maker statement"
    assert s2.decision_structure.decision_maker_present is False, "Spouse is absent so decision maker is not fully present"
