import json

rep = json.load(open('reports/synthetic/conversation_state_sim_muy04bh0.json', encoding='utf-8'))
print(f"Call SID: {rep.get('call_sid')}")
for t in rep['timeline']:
    tid = t['turn_id']
    spk = t['evidence_bundle']['speaker_id']
    txt = t['evidence_bundle']['utterance_text']
    v_b = t['state_before']['state_version']
    v_a = t['state_after']['state_version']
    dec = t.get('strategic_decision', {})
    d_v = dec.get('source_state_version')
    d_cid = dec.get('call_id') or dec.get('call_sid')
    d_gate = dec.get('meeting_gate_open')
    g_a = t['state_after']['conversion_gate']['is_open'] if t['state_after'].get('conversion_gate') else None
    r_a = t['state_after']['readiness']['readiness_score'] if t['state_after'].get('readiness') else None
    print(f"Turn {tid} ({spk}): '{txt}'")
    print(f"   state_before v={v_b}, state_after v={v_a}")
    print(f"   decision: v={d_v}, call_id={d_cid}, gate_open={d_gate}")
    print(f"   state_after gate_open={g_a}, readiness={r_a}")
    print(f"   decision evidence_considered: {dec.get('evidence_considered')}")
    print(f"   decision action={dec.get('primary_action')}, posture={dec.get('strategic_posture')}, push={dec.get('push_strength')}")
    print(f"   carried_from_turn={dec.get('carried_forward_from_turn_id')}, carried_from_dec={dec.get('carried_forward_from_decision_id')}")
    print()
