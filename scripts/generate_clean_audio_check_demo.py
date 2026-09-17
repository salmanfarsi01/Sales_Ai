"""Generate clean audio check and non-material backup demonstration call.

Demonstrates:
Turn 1: Clean, complete question: "Hi, just checking, can you hear me okay?" -> NON-MATERIAL (0 changes)
Turn 2: Casual acknowledgment: "Yeah, I hear you fine." -> NON-MATERIAL (0 changes)
Turn 3: Real material disclosure: "We might consider selling next summer if we get the right price." -> MATERIAL (facts, dimensions updated)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.conversation_replay import ConversationReplayEngine

def main():
    engine = ConversationReplayEngine()
    turns = [
        {
            "turn_id": 1,
            "speaker_id": "salesperson",
            "text": "Hi, just checking, can you hear me okay?",
            "trust": 0.50,
            "tension": 0.0,
            "valence": 0.0,
            "engagement": 0.50,
            "readiness": 0.50,
            "momentum": 0.50,
            "specificity": 0.1,
            "agreement": 0.5,
        },
        {
            "turn_id": 2,
            "speaker_id": "client",
            "text": "Yeah, I hear you fine.",
            "trust": 0.50,
            "tension": 0.0,
            "valence": 0.0,
            "engagement": 0.50,
            "readiness": 0.50,
            "momentum": 0.50,
            "specificity": 0.1,
            "agreement": 0.5,
        },
        {
            "turn_id": 3,
            "speaker_id": "client",
            "text": "We might consider selling next summer if we get the right price.",
            "trust": 0.65,
            "tension": 0.10,
            "valence": 0.20,
            "engagement": 0.70,
            "readiness": 0.60,
            "momentum": 0.65,
            "specificity": 0.75,
            "agreement": 0.60,
        },
    ]

    call_sid = "sim_clean_audio_check_demo"
    report = engine.replay_dialogue_turns(call_sid=call_sid, raw_turns=turns, save_report=True)
    print(f"Generated replay report: {call_sid} (total_turns={report.total_turns})")
    for step in report.timeline:
        print(f"Turn {step.turn_id} ({step.speaker_id}): '{step.text}'")
        print(f"  Materiality: {step.materiality.is_material} (Targets: {list(step.materiality.affected_targets)})")
        print(f"  Reasoning: {step.materiality.reasoning}")
        print(f"  Changes: {len(step.state_changes)}")

if __name__ == "__main__":
    main()
