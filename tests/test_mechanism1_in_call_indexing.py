import pytest
from copilot.conversation_state_models import ConversationStateSnapshot
from copilot.core_decision_manager import CoreDecisionManager
from copilot.conversation_replay import ConversationReplayEngine
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from copilot.behavioral_semantic import SemanticFeatureSnapshot

def _make_bundle(turn_id: int, speaker: str, text: str):
    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id}",
        call_sid="test_call_mech1",
        speaker_id=speaker,
        boundary_score=0.0,
    )
    inf = DownstreamInferenceState(
        call_sid="test_call_mech1",
        timestamp_ms=turn_id * 1000,
        trust=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_20_30s"),
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

def test_mechanism1_core_decision_manager_turn_indexing():
    """Validates that CoreDecisionManager indexes decisions by turn and version and protects against mutations."""
    cdm = CoreDecisionManager(call_sid="test_call_mech1")
    snap1 = ConversationStateSnapshot(call_sid="test_call_mech1", state_version=3, last_updated_turn_id=3)
    res1 = cdm.evaluate_state(snapshot=snap1, turn_speaker="salesperson", turn_text="What is your timeline?", turn_id=3)
    
    assert cdm.get_decision_for_turn(3) is not None
    assert cdm.get_decision_for_turn(3).source_state_version == 3
    assert cdm.get_decision_for_version(3) is not None

    # Process turn 18 (e.g. terminal commitment)
    snap2 = ConversationStateSnapshot(call_sid="test_call_mech1", state_version=38, last_updated_turn_id=18)
    res2 = cdm.evaluate_state(snapshot=snap2, turn_speaker="client", turn_text="Thursday at 3 works", turn_id=18)

    assert cdm.get_decision_for_turn(18).source_state_version == 38
    # Critical invariant: Turn 3 lookup must NOT be overwritten by Turn 18
    dec_t3 = cdm.get_decision_for_turn(3)
    assert dec_t3.source_state_version == 3
    assert dec_t3.decision_id != cdm.latest_decision.decision_id

def test_mechanism1_replay_timeline_step_immutability():
    """Validates that ConversationReplayEngine preserves immutable point-in-time strategic decisions across turns."""
    engine = ConversationReplayEngine()
    bundles = [
        _make_bundle(1, "salesperson", "Hi there"),
        _make_bundle(2, "client", "I might move next year"),
        _make_bundle(3, "salesperson", "What is driving the timing?"),
        _make_bundle(4, "client", "Thursday at 3 works, and my wife will be there."),
    ]
    report = engine.replay_call(call_sid="sim_test_mech1_immutability", bundles=bundles, save_report=False)
    
    step_t3 = report.timeline[2]
    step_t4 = report.timeline[3]

    assert step_t3.turn_id == 3
    assert step_t4.turn_id == 4

    # Turn 3 strategic decision must NOT reflect Turn 4's explicit commitment or state
    assert step_t3.strategic_decision is not None
    assert step_t4.strategic_decision is not None
    assert step_t3.strategic_decision.source_state_version < step_t4.strategic_decision.source_state_version
    assert step_t3.strategic_decision.decision_id != step_t4.strategic_decision.decision_id
