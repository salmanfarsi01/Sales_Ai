import json, sys
sys.path.insert(0, ".")
from copilot.conversation_replay import ConversationReplayEngine

engine = ConversationReplayEngine()
raw_turns = json.loads(open("data/script_sim_muy04bh0_fee_complaint.json", encoding="utf-8").read())
rep = engine.replay_dialogue_turns(call_sid="test_muy", raw_turns=raw_turns, save_report=False)
for tid in (3, 8):
    step = [s for s in rep.timeline if s.turn_id == tid][0]
    print(f"=== Facts at Turn {tid} ===")
    for f in step.state_after.facts:
        print(f"  {f.category}: {f.fact_key} = {f.fact_value} (status={f.status})")
    print(f"Decision structure: present={step.state_after.decision_structure.decision_maker_present}, primary={step.state_after.decision_structure.primary_decision_maker}, stakeholders={len(step.state_after.decision_structure.stakeholders)}")
