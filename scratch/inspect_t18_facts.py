import json, sys
sys.path.insert(0, ".")
from copilot.conversation_replay import ConversationReplayEngine

engine = ConversationReplayEngine()
raw_turns = json.loads(open("data/script_canonical.json", encoding="utf-8").read())
rep = engine.replay_dialogue_turns(call_sid="test_canon", raw_turns=raw_turns, save_report=False)
t18 = rep.timeline[17]
state = t18.state_after
print("Facts at Turn 18:")
for f in state.facts:
    print(f"  {f.category}: {f.fact_key} = {f.fact_value} (status={f.status})")
