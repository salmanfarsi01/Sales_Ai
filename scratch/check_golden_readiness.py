import json

with open("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json") as f:
    d = json.load(f)

for step in d["timeline"]:
    t = step["turn_id"]
    sp = step["speaker_id"]
    sa = step.get("state_snapshot") or step.get("state_after") or {}
    r = sa.get("readiness") or {}
    print(f"Turn {t:2d} ({sp:11s}): score={r.get('readiness_score')} partial={r.get('readiness_partial')} cov={r.get('coverage')} insuff={r.get('insufficient_evidence')}")
