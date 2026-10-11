import json

g = json.load(open("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json", encoding="utf-8"))
t18 = g["timeline"][17]
gate = t18["state_snapshot"]["conversion_gate"]
for c in gate["conditions"]:
    print(f"{c['condition_name']}: status={c['status']}, is_overridden={c.get('is_overridden')}, reason={c.get('reason')}")
