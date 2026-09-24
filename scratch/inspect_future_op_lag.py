import json

with open(r'reports/synthetic/conversation_state_sim_mueyx6wy.json', 'r') as f:
    data = json.load(f)

for entry in data.get('timeline', []):
    t_num = entry.get('turn_id')
    if t_num not in [6, 7, 8, 9, 10, 11, 17, 18]:
        continue
    spk = entry.get('speaker_id')
    utt = entry.get('text', '')
    
    state_before = entry.get('state_before') or {}
    ce_before = state_before.get('conversion_event')
    ce_b_status = ce_before.get('status') if ce_before else None
    
    state_after = entry.get('state_after') or {}
    ce_after = state_after.get('conversion_event')
    ce_a_status = ce_after.get('status') if ce_after else None
    ce_id = ce_after.get('event_id') if ce_after else None
    
    mom = state_after.get('momentum') or {}
    fam = mom.get('family_scores') or {}
    fob = fam.get('future_operational_behavior')
    dsc = fam.get('decision_structure_clarity')
    overall = mom.get('momentum_score')
    
    facts = [f.get('fact_key') for f in (state_after.get('facts') or []) if f.get('status') == 'active']
    
    state_changes = entry.get('state_changes', [])
    ce_changes = [c for c in state_changes if 'conversion_event' in str(c)]
    
    print(f"=== Turn {t_num} [{spk}] ===")
    print(f"Utterance: \"{utt}\"")
    print(f"CE Before: {ce_b_status} -> CE After: {ce_a_status} (id={ce_id})")
    print(f"CE Changes in state_changes: {ce_changes}")
    print(f"future_operational_behavior: {fob}")
    print(f"decision_structure_clarity:  {dsc}")
    print(f"overall momentum:            {overall}")
    print(f"active facts: {facts}")
    print()
