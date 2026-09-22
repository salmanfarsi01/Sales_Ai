import json

with open('reports/synthetic/conversation_state_sim_mucfqdlo.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for step in data['timeline']:
    tid = step['turn_id']
    spk = step['speaker_id']
    txt = step['text']
    mat = step['materiality']['is_material']
    tgts = list(step['materiality']['affected_targets'])
    chgs = [c['field_path'] for c in step.get('state_changes', [])]
    dec = step['state_after']['decision_structure']
    dm_pres = dec.get('decision_maker_present')
    stks = [s.get('role') + ':' + s.get('presence') for s in dec.get('stakeholders', [])]
    gate = step['state_after'].get('conversion_gate')
    g_open = gate.get('is_open') if gate else None
    failed = gate.get('failed_conditions', []) if gate else []
    push_obj = step['state_after'].get('push_strength')
    push = push_obj.get('state') if push_obj else None
    read_obj = step['state_after'].get('readiness')
    readiness = read_obj.get('readiness_score') if read_obj else None
    caps = read_obj.get('active_blocker_caps') if read_obj else []
    objs = [o['canonical_category'] + '=' + o['lifecycle_state'] for o in step['state_after'].get('objections', [])]
    print(f"T{tid:02d} [{spk:11s}]: \"{txt[:45]:45s}\" | mat={mat} tgts={tgts} | dm={dm_pres} stks={stks} | gate={g_open} fail={failed} push={push} | read={readiness} caps={caps} | objs={objs}")
