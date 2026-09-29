import json

for fname in ['reports/synthetic/conversation_state_sim_mufc4lsh.json', 'reports/synthetic/conversation_state_sim_mukmxsle.json', 'reports/synthetic/conversation_state_sim_mucj0p5s_golden.json']:
    try:
        d = json.load(open(fname, encoding='utf-8'))
        print(f"=== {fname} ===")
        tl = d.get('timeline', [])
        for i in [2, 3, 4, 17]:
            if i < len(tl):
                step = tl[i]
                sa = step.get('state_after', {})
                sd = step.get('strategic_decision', {})
                tid = step.get('turn_id')
                sa_ver = sa.get('state_version')
                gate_open = sa.get('conversion_gate', {}).get('is_open')
                commit = sa.get('dimensions', {}).get('commitment')
                sd_ver = sd.get('source_state_version') if sd else None
                action = sd.get('primary_action') if sd else None
                stub = sd.get('gateway_fallback_stub') if sd else None
                ev = sd.get('evidence_considered', []) if sd else []
                print(f"Turn {tid}: sa_version={sa_ver}, gate_open={gate_open}, commitment={commit}")
                print(f"   sd_version={sd_ver}, action={action}")
                print(f"   stub={stub}")
                print(f"   ev={ev[:2]}")
    except Exception as e:
        print(f"Error on {fname}: {e}")
