import json

data = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))
for s in data['timeline']:
    for sc in s.get('state_changes', []):
        if 'objection' in sc.get('field_path', ''):
            print(f"Turn {s['turn_id']}: {sc['field_path']} -> {sc['new_value']} | {sc['reason']}")
    for o in s.get('state_after', {}).get('objections', []):
        if s['turn_id'] in (13, 14, 15, 16, 17, 18):
            print(f"  Turn {s['turn_id']} Obj: id={o['objection_id']} state={o['lifecycle_state']} last_updated={o['last_updated_turn_id']}")
