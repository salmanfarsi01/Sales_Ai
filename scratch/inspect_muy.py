import json

d = json.load(open("reports/synthetic/conversation_state_sim_muy04bh0.json", encoding="utf-8"))
for step in d["timeline"]:
    tid = step["turn_id"]
    spk = step.get("speaker_id")
    sd = step.get("strategic_decision") or {}
    sa = step.get("state_after") or {}
    gate = sa.get("conversion_gate") or {}
    ce = sa.get("conversion_event") or {}
    objs = [o.get("objection_id") + ":" + o.get("lifecycle_state") for o in sa.get("objections", [])]
    text = step.get("text", "")[:45]
    print(f"Turn {tid:<2} ({spk:<11}): Gate={str(gate.get('is_open')):<5} (st={gate.get('status')}) EvStatus={ce.get('status')} Objs={objs} | '{text}'")
