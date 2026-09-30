import json

data = json.load(open('reports/synthetic/conversation_state_sim_mufc4lsh.json', encoding='utf-8'))
for t in data['timeline']:
    sa = t.get('state_after') or {}
    sd = t.get('strategic_decision') or {}
    gate = sa.get('conversion_gate') or {}
    dims = sa.get('dimensions') or {}
    print(f"T{t['turn_id']:02d} ({t['speaker_id'][:3]}): v={sa.get('state_version')}, act={sd.get('primary_action')}, push={sd.get('push_strength')}, gate={gate.get('is_open')}, commit={dims.get('commitment')}, slot={gate.get('commitment_slot')}, text=\"{t['text']}\"")
