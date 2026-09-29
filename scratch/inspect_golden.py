import json

for fname in ['sim_validation_test', 'sim_mucj0p5s_golden']:
    path = f'reports/synthetic/conversation_state_{fname}.json'
    d = json.load(open(path, encoding='utf-8'))
    print(f"=== {fname} ===")
    print("Call SID:", d.get('call_sid'))
    tl = d.get('timeline', [])
    for step in tl[:4]:
        tid = step.get('turn_id')
        sa = step.get('state_after') or {}
        sd = step.get('strategic_decision') or {}
        print(f"  Turn {tid}: text='{step.get('text', '')[:30]}' sa_v={sa.get('state_version')} sd_v={sd.get('source_state_version')} act={sd.get('primary_action')}")
        if sd.get('evidence_considered'):
            print(f"     ev: {sd.get('evidence_considered')[:2]}")
    step18 = tl[-1]
    sa18 = step18.get('state_after') or {}
    sd18 = step18.get('strategic_decision') or {}
    print(f"  Turn {step18.get('turn_id')}: text='{step18.get('text', '')[:30]}' sa_v={sa18.get('state_version')} sd_v={sd18.get('source_state_version')} act={sd18.get('primary_action')}")
    if sd18.get('evidence_considered'):
        print(f"     ev: {sd18.get('evidence_considered')[:2]}")
