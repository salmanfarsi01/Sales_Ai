import json, sys
sys.path.insert(0, ".")
from copilot.conversation_replay import ConversationReplayEngine

engine = ConversationReplayEngine()
raw_turns = json.loads(open("data/script_sim_muy04bh0_fee_complaint.json", encoding="utf-8").read())
rep = engine.replay_dialogue_turns(call_sid="sim_muy04bh0", raw_turns=raw_turns, save_report=True)

for tid in (3, 8):
    step = [s for s in rep.timeline if s.turn_id == tid][0]
    gate = step.state_after.conversion_gate
    print(f"=== Turn {tid} Gate ===")
    print(f"is_open: {gate.is_open}, status: {gate.status}")
    for c in gate.conditions:
        print(f"  {c.condition_name:<30} status={c.status:<8} met={str(c.met):<5} overridden={str(c.is_overridden):<5} reason={c.reason}")
