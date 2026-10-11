import json

golden = json.load(open("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json", encoding="utf-8"))
tl = golden["timeline"]
print(f"Total turns: {len(tl)}")
print(f"{'Turn':<5} {'Speaker':<12} {'Action':<15} {'Conf':<8} {'Gate':<8} {'Codes'}")
for s in tl:
    tid = s["turn_id"]
    spk = s.get("speaker_id")
    sd = s.get("strategic_decision") or {}
    act = sd.get("primary_action", "")
    conf = sd.get("confidence", 0.0)
    gate = sd.get("meeting_gate_open", False)
    codes = sd.get("reason_codes", [])
    print(f"{tid:<5} {spk:<12} {act:<15} {conf:<8.3f} {str(gate):<8} {codes[:3]}")
