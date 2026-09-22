import json

with open('reports/synthetic/conversation_state_sim_mucfqdlo.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== CALL SUMMARY ===")
print("Call SID:", data['call_sid'])
print("Total turns:", data['total_turns'])
print("Final state version:", data['final_state']['state_version'])
print("Conversion summary:", data.get('conversion_summary'))

print("\n=== TIMELINE TURNS ===")
for step in data['timeline']:
    tid = step['turn_id']
    spk = step['speaker_id']
    txt = step['text']
    mat = step['materiality']
    is_mat = mat.get('is_material')
    tgts = list(mat.get('affected_targets', []))
    reason = mat.get('reasoning', '')
    chgs = step.get('state_changes', [])
    
    # Check decision structure
    dec = step['state_after']['decision_structure']
    dm_pres = dec.get('decision_maker_present')
    stks = [s.get('role') + '(' + s.get('presence') + ')' for s in dec.get('stakeholders', [])]
    
    # Check gate
    gate = step['state_after'].get('conversion_gate')
    g_open = gate.get('is_open') if gate else None
    failed_conds = gate.get('failed_conditions', []) if gate else []
    
    # Check push
    push = step['state_after'].get('push_strength')
    push_st = push.get('state') if push else None
    
    # Check facts
    facts = [f.get('fact_key') + '=' + f.get('fact_value') + '(' + f.get('status') + ')' for f in step['state_after'].get('facts', [])]
    
    # Check objections
    objs = [o.get('canonical_category') + ':' + o.get('lifecycle_state') for o in step['state_after'].get('objections', [])]
    
    print(f"\n--- Turn {tid} [{spk}]: \"{txt}\" ---")
    print(f"  Material: {is_mat} | Targets: {tgts}")
    print(f"  Reason: {reason}")
    print(f"  Changes ({len(chgs)}): {[c['field_path'] + ' (' + c['reason'][:40] + ')' for c in chgs]}")
    print(f"  Decision Structure: DM Present={dm_pres}, Stakeholders={stks}")
    print(f"  Gate Open: {g_open} | Failed: {failed_conds} | Push: {push_st}")
    print(f"  Active Objections: {objs}")
    print(f"  Facts: {facts}")

print("\n=== FINAL STATE DIAGNOSTICS ===")
fin = data['final_state']
print("Decision Structure:", json.dumps(fin['decision_structure'], indent=2))
print("Objections:", json.dumps(fin['objections'], indent=2))
print("Facts:", json.dumps(fin['facts'], indent=2))
print("Dimensions:", json.dumps(fin['dimensions'], indent=2))
print("Contact Compliance:", json.dumps(fin['contact_compliance'], indent=2))
print("Readiness:", json.dumps(fin['readiness'], indent=2))
print("Conversion Gate:", json.dumps(fin['conversion_gate'], indent=2))
print("Push Strength:", json.dumps(fin['push_strength'], indent=2))
print("Conversion Event:", json.dumps(fin['conversion_event'], indent=2))
