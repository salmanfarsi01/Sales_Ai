import json

with open("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json") as f:
    d = json.load(f)

for step in d["timeline"]:
    t = step["turn_id"]
    sp = step["speaker_id"]
    text = step.get("utterance_text") or step.get("text")
    strat = step.get("strategic_decision")
    sa = step.get("state_snapshot") or step.get("state_after") or {}
    ps = sa.get("push_strength") or {}
    gate = sa.get("conversion_gate") or {}
    r = sa.get("readiness") or {}
    ce = sa.get("conversion_event")
    
    strat_str = f"action={strat.get('primary_action')} posture={strat.get('strategic_posture')} push={strat.get('push_strength')} conf={strat.get('confidence')}" if strat else "None"
    print(f"Turn {t:2d} ({sp:11s}): strat=[{strat_str}] ps_posture={ps.get('strategic_posture')} ps_pressure={ps.get('pressure')} ps_state={ps.get('state')} gate_open={gate.get('is_open')} gate_status={gate.get('status')} ce={ce.get('status') if ce else None}")
