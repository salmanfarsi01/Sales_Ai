import pytest
from pathlib import Path
from copilot.conversation_replay import ConversationReplayEngine, TurnReplayStep
from copilot.conversation_state_models import ConversationStateSnapshot
from copilot.conversation_materiality import MaterialityClassification
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.core_decision_manager import CoreDecisionManager
from copilot.core_intelligence_models import StrategicDecision, StrategicAction


def _make_test_bundle(call_sid: str, turn_id: int, speaker: str, text: str):
    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_{call_sid}_{turn_id}",
        call_sid=call_sid,
        speaker_id=speaker,
        boundary_score=0.0,
    )
    inf = DownstreamInferenceState(
        call_sid=call_sid,
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


def test_point4_pending_state_when_no_decision_recorded():
    """Point 4: If a selected turn has no valid CoreDecision for its exact version,
    the system must return None / pending state — NEVER silently substitute the final or latest decision.
    """
    cdm = CoreDecisionManager(call_sid="call_pending_test")

    # Evaluate Turn 1 (V2)
    snap1 = ConversationStateSnapshot(call_sid="call_pending_test", state_version=2, last_updated_turn_id=1)
    eval1 = cdm.evaluate_state(snapshot=snap1, turn_speaker="client", turn_text="Hello", turn_id=1)

    # Evaluate Turn 18 (V38)
    snap18 = ConversationStateSnapshot(call_sid="call_pending_test", state_version=38, last_updated_turn_id=18)
    eval18 = cdm.evaluate_state(snapshot=snap18, turn_speaker="client", turn_text="Thursday at 3 works", turn_id=18)

    assert cdm.latest_decision.source_state_version == 38

    # Query Turn 5 (which was never evaluated)
    dec_turn5 = cdm.get_decision_for_turn(5)
    # Must be None — MUST NOT silently fall back to Turn 18 or latest_decision!
    assert dec_turn5 is None
    assert dec_turn5 != cdm.latest_decision

    # Query version 15 (which was never evaluated)
    dec_v15 = cdm.get_decision_for_version(15)
    assert dec_v15 is None
    assert dec_v15 != cdm.latest_decision

    # Check TurnReplayStep with no decision
    bundle = _make_test_bundle("call_pending_test", 5, "client", "I am thinking about it.")
    mat = MaterialityClassification(is_material=False, reasoning="Pleasantry", confidence=1.0)
    step_pending = TurnReplayStep(
        turn_id=5,
        speaker_id="client",
        text="I am thinking about it.",
        timestamp_ms=5000,
        evidence_bundle=bundle,
        materiality=mat,
        state_after=ConversationStateSnapshot(call_sid="call_pending_test", state_version=10, last_updated_turn_id=5),
        strategic_decision=None,
    )
    assert step_pending.strategic_decision is None

    # Verify conversation_state_replay.html enforces pending state banner and never displays final state
    html_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"
    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert "Pending / No Decision Recorded" in html_content
    assert "hasValidDecision = sd && (sd.source_state_version === sAfter.state_version)" in html_content
    assert "Silent fallback to later turns or final outcome is strictly prohibited" in html_content


def test_point5_evidence_consistency_single_coherent_snapshot():
    """Point 5: The Evidence Considered block must be assembled from exactly one
    ConversationState object at one source_state_version:
    assert evidence.utterance_turn_id == evidence.metrics_source_turn_id == decision.source_state_version's turn.
    """
    turns = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, thanks for making time today."},
        {"turn_id": 2, "speaker_id": "client", "text": "I'm the one making this decision."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Got it. What's driving the timing for you?"},
        {"turn_id": 4, "speaker_id": "client", "text": "We might look at moving sometime next year, nothing urgent yet."},
        {"turn_id": 5, "speaker_id": "client", "text": "Honestly, we're not sure this is the right time anymore."},
        {"turn_id": 6, "speaker_id": "salesperson", "text": "That's fair — a lot of people feel that way before they see the actual numbers."},
        {"turn_id": 7, "speaker_id": "client", "text": "Okay, that makes sense, I guess timing isn't the biggest issue."},
        {"turn_id": 8, "speaker_id": "salesperson", "text": "Great — would sometime next week work for a walkthrough?"},
        {"turn_id": 9, "speaker_id": "client", "text": "Maybe next week could work, let me think about it."},
        {"turn_id": 10, "speaker_id": "client", "text": "Actually, my wife would really need to be part of this conversation."},
        {"turn_id": 11, "speaker_id": "salesperson", "text": "Of course, happy to loop her in whenever works."},
        {"turn_id": 12, "speaker_id": "client", "text": "I guess I'm just worried this isn't really the right move for us financially."},
        {"turn_id": 13, "speaker_id": "salesperson", "text": "Totally understand — let's look at your net proceeds after all costs."},
        {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more."},
        {"turn_id": 15, "speaker_id": "client", "text": "Please don't start texting me every day before we meet, by the way."},
        {"turn_id": 16, "speaker_id": "client", "text": "Mornings don't really work for us either, just so you know."},
        {"turn_id": 17, "speaker_id": "salesperson", "text": "Noted on all of that. What day works best?"},
        {"turn_id": 18, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
    ]

    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns("sim_test_point5_consistency", turns, save_report=False)

    assert len(rep.timeline) == 18

    # Invariant assertion across EVERY turn
    for step in rep.timeline:
        sd = step.strategic_decision
        assert sd is not None, f"Turn {step.turn_id} missing StrategicDecision"
        sa = step.state_after

        # Point 5 core assertion: single coherent source turn and version
        assert sd.utterance_turn_id == step.turn_id, f"Turn {step.turn_id} utterance_turn_id mismatch: {sd.utterance_turn_id}"
        assert sd.metrics_source_turn_id == step.turn_id, f"Turn {step.turn_id} metrics_source_turn_id mismatch: {sd.metrics_source_turn_id}"
        assert sd.source_turn_id == step.turn_id, f"Turn {step.turn_id} source_turn_id mismatch: {sd.source_turn_id}"
        assert sd.source_state_version == sa.state_version, f"Turn {step.turn_id} state_version mismatch: {sd.source_state_version} vs {sa.state_version}"

        # Assert utterance text in evidence comes strictly from this turn
        if sd.evidence_considered:
            first_ev = sd.evidence_considered[0]
            clean_step_text = step.text.strip().replace("\n", " ")[:60]
            assert clean_step_text in first_ev, f"Turn {step.turn_id} evidence contained wrong utterance: {first_ev}"


def test_point6_traceability_fields_present_and_unique_turn_by_turn():
    """Point 6: Expose call_sid, decision_id, source_state_version, source_event_id/turn_id
    directly on every StrategicDecision, changing cleanly turn-by-turn.
    """
    turns = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hello there."},
        {"turn_id": 2, "speaker_id": "client", "text": "Hi, I want to sell my property."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Wonderful, when can we meet?"},
        {"turn_id": 4, "speaker_id": "client", "text": "Tomorrow afternoon works."}
    ]

    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns("sim_test_point6_traceability", turns, save_report=False)

    decision_ids = set()
    prev_version = 0

    for step in rep.timeline:
        sd = step.strategic_decision
        assert sd is not None

        # 1. call_sid is present and matches
        assert sd.call_sid == "sim_test_point6_traceability"
        assert sd.call_id == "sim_test_point6_traceability"

        # 2. decision_id is present and distinct per turn
        assert sd.decision_id.startswith("dec_")
        assert sd.decision_id not in decision_ids, f"Duplicate decision_id: {sd.decision_id}"
        decision_ids.add(sd.decision_id)

        # 3. source_state_version is present and monotonically advances
        assert sd.source_state_version >= prev_version
        prev_version = sd.source_state_version

        # 4. source_event_id / source_turn_id is present and matches turn
        assert sd.source_turn_id == step.turn_id
        assert sd.source_event_id == f"ev_turn_{step.turn_id}_v{sd.source_state_version}"
