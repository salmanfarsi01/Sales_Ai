import json, glob

for arch in glob.glob('reports/synthetic/archive/conversation_state_sim_mukmxsle_*.json'):
    try:
        d = json.load(open(arch, encoding='utf-8'))
        print(f"=== {arch} ===")
        tl = d.get('timeline', [])
        print(f"Total turns: {len(tl)}")
        for step in tl:
            sd = step.get('strategic_decision')
            tid = step.get('turn_id')
            spk = step.get('speaker_id')
            text = step.get('text', '')[:25]
            sd_v = sd.get('source_state_version') if sd else None
            ev0 = sd.get('evidence_considered')[0] if (sd and sd.get('evidence_considered')) else 'None'
            print(f"Turn {tid}: {spk} '{text}' sd_v={sd_v} ev0={ev0}")
    except Exception as e:
        print(f"Error: {e}")
