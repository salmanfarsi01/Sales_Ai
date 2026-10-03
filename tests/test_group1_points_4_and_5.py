import re
import pytest
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStructure,
    DecisionStakeholder,
    ConversionEventObject,
    ConversionEventStatus,
    MeetingConversionGate,
    DimensionScores,
)
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_conversion import MeetingConversionGateEngine
from copilot.conversation_replay import ConversationReplayEngine
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_intelligence_models import StrategicDecision, StrategicAction
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.llm_response_gateway import LLMResponseGateway


def _make_bundle(call_sid: str, turn_id: int, speaker: str, text: str, trust_score: float = 0.5, trust_conf: float = 0.7):
    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_{call_sid}_{turn_id}",
        call_sid=call_sid,
        speaker_id=speaker,
        boundary_score=0.0,
    )
    inf = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=turn_id * 1000,
        trust=DimensionScore(score=trust_score, confidence=trust_conf, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.2, confidence=0.85),
        pacing=DimensionScore(score=0.6, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
    )
    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker,
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=sem,
    )


def test_point4_decision_maker_reconciliation_and_participant_inclusion():
    """Point 4:
    - Turn 2: 'sole_decision_maker' (co_decision_required=False)
    - Turn 10: Spouse requirement reconciles to 'sole decision maker, spouse required for final approval' (co_decision_required=True)
    - Turn 18: Spouse confirmed attending reconciles to 'sole decision maker, spouse confirmed attending' (co_decision_required=True, decision_maker_present=True)
    - Confirmed conversion event participants includes 'Wife'
    """
    mgr = ConversationStateManager(call_sid="call_reconcile_test")

    # Turn 2: Sole decision maker declaration
    b2 = _make_bundle("call_reconcile_test", 2, "client", "I'm the one making this decision, no one else needs to sign off.")
    s2 = mgr.process_turn_bundle(b2)
    assert s2.decision_structure.primary_decision_maker == "sole_decision_maker"
    assert s2.decision_structure.decision_maker_present is True
    assert s2.decision_structure.co_decision_required is False

    # Turn 10: Spouse involvement declared
    b10 = _make_bundle("call_reconcile_test", 10, "client", "Actually, my wife would really need to be part of this conversation before we go any further.")
    s10 = mgr.process_turn_bundle(b10)
    assert s10.decision_structure.co_decision_required is True
    assert s10.decision_structure.decision_maker_present is False
    assert s10.decision_structure.primary_decision_maker == "sole decision maker, spouse required for final approval"
    assert any(s.role == "wife" and s.presence == "absent" for s in s10.decision_structure.stakeholders)

    # Turn 18: Spouse attendance confirmed
    b18 = _make_bundle("call_reconcile_test", 18, "client", "Thursday at 3 works, and my wife will be there.")
    s18 = mgr.process_turn_bundle(b18)
    assert s18.decision_structure.co_decision_required is True
    assert s18.decision_structure.decision_maker_present is True
    assert s18.decision_structure.primary_decision_maker == "sole decision maker, spouse confirmed attending"
    assert any(s.role == "wife" and s.presence == "confirmed_attending" for s in s18.decision_structure.stakeholders)


def test_point4_canonical_18_turn_final_state_participants_and_reconciliation():
    """Verify that in the canonical 18-turn dialogue replay, Turn 18's final state shows:
    1. 'Wife' in conversion_event.participants
    2. co_decision_required is True
    3. primary_decision_maker reflects spouse confirmation without contradictory sole claim.
    """
    turns = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, thanks for making time today — tell me a bit about what's going on with the house."},
        {"turn_id": 2, "speaker_id": "client", "text": "I'm the one making this decision, no one else needs to sign off."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Got it. What's driving the timing for you?"},
        {"turn_id": 4, "speaker_id": "client", "text": "We might look at moving sometime next year, nothing urgent yet."},
        {"turn_id": 5, "speaker_id": "client", "text": "Honestly, we're not sure this is the right time anymore."},
        {"turn_id": 6, "speaker_id": "salesperson", "text": "That's fair — a lot of people feel that way before they see the actual numbers."},
        {"turn_id": 7, "speaker_id": "client", "text": "Okay, that makes sense, I guess timing isn't the biggest issue."},
        {"turn_id": 8, "speaker_id": "salesperson", "text": "Great — would sometime next week work for a walkthrough?"},
        {"turn_id": 9, "speaker_id": "client", "text": "Maybe next week could work, let me think about it."},
        {"turn_id": 10, "speaker_id": "client", "text": "Actually, my wife would really need to be part of this conversation before we go any further."},
        {"turn_id": 11, "speaker_id": "salesperson", "text": "Of course, happy to loop her in whenever works."},
        {"turn_id": 12, "speaker_id": "client", "text": "I guess I'm just worried this isn't really the right move for us financially with everything going on."},
        {"turn_id": 13, "speaker_id": "salesperson", "text": "Totally understand — let's look at your net proceeds after all costs."},
        {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more."},
        {"turn_id": 15, "speaker_id": "client", "text": "Please don't start texting me every day before we meet, by the way."},
        {"turn_id": 16, "speaker_id": "client", "text": "Mornings don't really work for us either, just so you know."},
        {"turn_id": 17, "speaker_id": "salesperson", "text": "Noted on all of that. What day works best?"},
        {"turn_id": 18, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
    ]

    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns("sim_pt4_canonical", turns, save_report=False)
    final_st = rep.final_state

    # 1. Final conversion event has Wife as participant
    assert final_st.conversion_event is not None
    assert final_st.conversion_event.status == ConversionEventStatus.CONFIRMED
    assert "Wife" in final_st.conversion_event.participants
    assert "Client" in final_st.conversion_event.participants
    assert "Agent" in final_st.conversion_event.participants

    # 2. Decision structure reconciled
    assert final_st.decision_structure.co_decision_required is True
    assert final_st.decision_structure.decision_maker_present is True
    assert "spouse confirmed attending" in final_st.decision_structure.primary_decision_maker


def test_point5_evidence_considered_distinguishes_default_from_measured_trust():
    """Point 5: Evidence Considered must visibly tag default/unmeasured trust vs measured trust."""
    from copilot.core_decision_manager import CoreDecisionManager

    # Case A: Default / unmeasured baseline (trust=0.5, trust_measured=False)
    cdm_default = CoreDecisionManager(call_sid="call_trust_default")
    snap_default = ConversationStateSnapshot(
        call_sid="call_trust_default",
        state_version=1,
        dimensions=DimensionScores(trust=0.50, trust_confidence=0.70, trust_measured=False),
    )
    eval_default = cdm_default.evaluate_state(snapshot=snap_default, turn_speaker="client", turn_text="Hello", turn_id=1)
    dec_default = eval_default.decision

    trust_ev_default = [e for e in dec_default.evidence_considered if e.startswith("Trust:")]
    assert len(trust_ev_default) == 1
    assert "Trust: 50% (default, unmeasured)" in trust_ev_default[0]

    # Case B: Measured trust (trust=0.65, trust_measured=True)
    cdm_measured = CoreDecisionManager(call_sid="call_trust_measured")
    snap_measured = ConversationStateSnapshot(
        call_sid="call_trust_measured",
        state_version=2,
        dimensions=DimensionScores(trust=0.65, trust_confidence=0.85, trust_measured=True),
    )
    eval_measured = cdm_measured.evaluate_state(snapshot=snap_measured, turn_speaker="client", turn_text="I trust your numbers.", turn_id=2)
    dec_measured = eval_measured.decision

    trust_ev_measured = [e for e in dec_measured.evidence_considered if e.startswith("Trust:")]
    assert len(trust_ev_measured) == 1
    assert "Trust: 65% (measured, confidence: 0.85)" in trust_ev_measured[0]


def test_point5_deterministic_fallback_no_unconfirmed_calendar_claims():
    """Point 5: Fallback prompt must not claim an appointment is 'confirmed on my calendar' without an actual write."""
    gateway = LLMResponseGateway()

    dec = StrategicDecision(
        call_id="call_slot_test",
        source_state_version=18,
        primary_action=StrategicAction.ACKNOWLEDGE,
        reason_codes=["CONVERSION_CONFIRMED"],
        strategic_objective="Protect confirmed appointment",
        commitment_slot="Thursday at 3",
    )
    snap = ConversationStateSnapshot(call_sid="call_slot_test", state_version=18)

    prompt = gateway.generate_prompt(decision=dec, snapshot=snap, facts=[])

    # 1. Assert prompt does not claim calendar confirmation
    assert "confirmed on my calendar" not in prompt.text.lower()
    # 2. Assert accurate phrasing
    assert prompt.text == "Perfect, I've noted Thursday at 3 for us. I will see you both then."
