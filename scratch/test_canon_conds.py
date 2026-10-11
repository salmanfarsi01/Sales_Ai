import json, sys
sys.path.insert(0, ".")
from copilot.conversation_replay import ConversationReplayEngine

engine = ConversationReplayEngine()
raw_turns = json.loads(open("data/script_canonical.json", encoding="utf-8").read())
rep = engine.replay_dialogue_turns(call_sid="test_canon", raw_turns=raw_turns, save_report=False)
t18 = rep.timeline[17]
gate = t18.state_after.conversion_gate
for c in gate.conditions:
    print(f"{c.condition_name}: status={c.status}, reason={c.reason}")
