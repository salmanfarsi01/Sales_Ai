import json
with open('reports/synthetic/conversation_state_sim_mucj0p5s.json') as f:
    base = json.load(f)
for t in base['timeline'][:8]:
    print(f"T{t['turn_id']}: [{t.get('speaker_id')}] {t.get('text')}")
