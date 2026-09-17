import asyncio
import os
import sys
import json
import dotenv
sys.path.insert(0, ".")
dotenv.load_dotenv()

from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_baseline import ContactPreferenceRecord
from copilot.behavioral_normalization import normalize_generic_transcript
from copilot.behavioral_semantic import SemanticFeatureEngine

async def test_replay_063():
    semantic_engine = SemanticFeatureEngine()
    manager = ConversationStateManager(call_sid="test_call_063_verification")

    # Turn 1: Agent Turn
    u1 = normalize_generic_transcript(
        text="I can prepare exactly that. Thursday at three still work?",
        speaker_id="salesperson",
        start_ms=800,
        end_ms=2500,
        call_sid="test_call_063_verification",
    )
    s1 = await semantic_engine.analyze_turn_semantic(u1, [])
    print(f"Turn 1 extraction_mode: {s1.extraction_mode}")
    print(f"Turn 1 specificity_score: {s1.specificity_score}")

    inf1 = DownstreamInferenceState(
        call_sid="test_call_063_verification",
        timestamp_ms=2500,
        trust=DimensionScore(score=0.50, confidence=0.70, primary_horizon="current_utterance"),
        readiness=DimensionScore(score=0.50, confidence=0.70, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.50, confidence=0.70, primary_horizon="current_utterance"),
        momentum=DimensionScore(score=0.50, confidence=0.70, primary_horizon="current_utterance"),
        pacing=DimensionScore(score=0.50, confidence=0.70, primary_horizon="current_utterance"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.0, confidence=0.70),
        overall_confidence=0.70,
        contributing_evidence_ids=["ev_turn_1"],
    )
    pref1 = ContactPreferenceRecord(call_sid="test_call_063_verification", preference="none", confidence=0.0)
    b1 = extract_behavioral_bundle(
        turn_id=1,
        speaker_id="salesperson",
        utterance_text=u1.text,
        inference_state=inf1,
        semantic_snapshot=s1,
        contact_preference_record=pref1,
    )

    state1 = manager.process_turn_bundle(b1)
    print(f"Turn 1 State Version: {state1.state_version}")
    print(f"Turn 1 Conversion Gate: {state1.conversion_gate}")
    print(f"Turn 1 Recorded Facts: {len(state1.facts)} facts")

    # Turn 2: Prospect Turn with reduced frequency preference
    u2 = normalize_generic_transcript(
        text="Yeah. Definitely. Just don't text me every single day.",
        speaker_id="client",
        start_ms=3000,
        end_ms=4500,
        call_sid="test_call_063_verification",
    )
    s2 = await semantic_engine.analyze_turn_semantic(u2, [u1])
    print(f"\nTurn 2 extraction_mode: {s2.extraction_mode}")
    print(f"Turn 2 contact_preference: {s2.contact_preference}")
    print(f"Turn 2 agreement_score: {s2.agreement_score}")

    inf2 = DownstreamInferenceState(
        call_sid="test_call_063_verification",
        timestamp_ms=4500,
        trust=DimensionScore(score=0.75, confidence=0.80, primary_horizon="current_utterance"),
        readiness=DimensionScore(score=0.85, confidence=0.80, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.75, confidence=0.80, primary_horizon="current_utterance"),
        momentum=DimensionScore(score=0.75, confidence=0.80, primary_horizon="current_utterance"),
        pacing=DimensionScore(score=0.60, confidence=0.80, primary_horizon="current_utterance"),
        emotion=EmotionState(expressed_valence=0.4, tension_level=0.1, confidence=0.80),
        overall_confidence=0.80,
        contributing_evidence_ids=["ev_turn_2"],
    )
    pref2 = ContactPreferenceRecord(
        call_sid="test_call_063_verification",
        preference=s2.contact_preference,
        details="don't text me every single day",
        confidence=0.90,
    )
    b2 = extract_behavioral_bundle(
        turn_id=2,
        speaker_id="client",
        utterance_text=u2.text,
        inference_state=inf2,
        semantic_snapshot=s2,
        contact_preference_record=pref2,
    )

    state2 = manager.process_turn_bundle(b2)
    print(f"Turn 2 State Version: {state2.state_version}")
    print(f"Turn 2 Gate Is Open: {state2.conversion_gate.is_open if state2.conversion_gate else None}")
    if state2.conversion_gate:
        print(f"Turn 2 Gate Status: {state2.conversion_gate.status}, Failed: {state2.conversion_gate.failed_conditions}")
    print(f"Turn 2 Push Strength State: {state2.push_strength.state if state2.push_strength else None}")
    print(f"Turn 2 Push Strength Action: {state2.push_strength.recommended_action if state2.push_strength else None}")
    print(f"Turn 2 Conversion Event: {state2.conversion_event.status if state2.conversion_event else None}, Slot: {state2.conversion_event.start_at if state2.conversion_event else None}")
    print("Turn 2 Recorded Facts:")
    for f in state2.facts:
        print(f"  [{f.category}] {f.fact_key} = {f.fact_value}")

    print(f"\nTurn 2 Raw Readiness (Dimension Prior): {state2.dimensions.readiness}")
    print(f"Turn 2 Composite Readiness: {state2.readiness.readiness_score}% (Uncapped: {state2.readiness.uncapped_score}%)")
    print(f"Turn 2 Active Blocker Caps: {state2.readiness.active_blocker_caps}")

if __name__ == "__main__":
    asyncio.run(test_replay_063())
