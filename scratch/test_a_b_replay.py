import sys
sys.path.insert(0, '.')
from copilot.conversation_replay import ConversationReplayEngine

engine = ConversationReplayEngine()
# Run Test A (3 turns with Thursday at 3)
turns_A = [
    {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi Daniel"},
    {"turn_id": 2, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
]
rep_A = engine.replay_dialogue_turns("sim_test_A_check", turns_A)
print("=== TEST A ===")
for s in rep_A.timeline:
    sd = s.strategic_decision
    print(f"Turn {s.turn_id}: text='{s.text}' sd_v={sd.source_state_version if sd else None}")
    if sd:
        print("  ev:", sd.evidence_considered)

# Run Test B (new clean test with 2 turns)
turns_B = [
    {"turn_id": 1, "speaker_id": "salesperson", "text": "Hello are you there?"},
    {"turn_id": 2, "speaker_id": "client", "text": "Yes I am here."}
]
rep_B = engine.replay_dialogue_turns("sim_test_B_check", turns_B)
print("\n=== TEST B ===")
for s in rep_B.timeline:
    sd = s.strategic_decision
    print(f"Turn {s.turn_id}: text='{s.text}' sd_v={sd.source_state_version if sd else None}")
    if sd:
        print("  ev:", sd.evidence_considered)
