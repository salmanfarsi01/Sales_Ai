import json

with open('reports/synthetic/conversation_state_sim_mucj0p5s.json') as f:
    base = json.load(f)
with open('reports/synthetic/conversation_state_sim_validation_test.json') as f:
    curr = json.load(f)

for b, c in zip(base['timeline'], curr['timeline']):
    tid = b.get('turn_id')
    br = b.get('state_after', {}).get('readiness', {}) or {}
    cr = c.get('state_after', {}).get('readiness', {}) or {}
    print(f"T{tid:02d}: base={br.get('readiness_score')} curr={cr.get('readiness_score')} | "
          f"B[emo={br.get('emotional_readiness')}, log={br.get('logical_readiness')}, logist={br.get('logistical_readiness')}, dec={br.get('decision_readiness')}, uncapped={br.get('uncapped_score')}, caps={br.get('active_blocker_caps')}] | "
          f"C[emo={cr.get('emotional_readiness')}, log={cr.get('logical_readiness')}, logist={cr.get('logistical_readiness')}, dec={cr.get('decision_readiness')}, uncapped={cr.get('uncapped_score')}, caps={cr.get('active_blocker_caps')}]")
