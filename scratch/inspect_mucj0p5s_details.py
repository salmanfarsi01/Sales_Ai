import json

data = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))
for t in data['timeline']:
    sa = t.get('state_after', {})
    sd = t.get('strategic_decision')
    gate = sa.get('conversion_gate') or {}
    dims = sa.get('dimensions') or {}
    act = sd.get('primary_action') if sd else None
    push = sd.get('push_strength') if sd else None
    print(f"T{t['turn_id']:02d} ({t['speaker_id']}): act={act}, push={push}, gate={gate.get('is_open')}, commit={dims.get('commitment')}, v={sa.get('state_version')}")
