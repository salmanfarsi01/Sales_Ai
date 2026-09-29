import json, glob

files = glob.glob('reports/**/conversation_state_sim_mufc4lsh*.json', recursive=True)
print("Found files:", files)
for f in files:
    try:
        d = json.load(open(f, encoding='utf-8'))
        tl = d.get('timeline', [])
        print(f"\n=== FILE: {f} (timeline len={len(tl)}) ===")
        for step in tl[:5]:
            tid = step.get('turn_id')
            sa = step.get('state_after') or {}
            sd = step.get('strategic_decision') or {}
            dim = sa.get('dimensions') or {}
            gate = sa.get('conversion_gate') or {}
            print(f"  Turn {tid}: sa_version={sa.get('state_version')} sd_version={sd.get('source_state_version')} action={sd.get('primary_action')} commit={dim.get('commitment')} gate_open={gate.get('is_open')}")
        if len(tl) >= 18:
            step18 = tl[17]
            sa18 = step18.get('state_after') or {}
            sd18 = step18.get('strategic_decision') or {}
            dim18 = sa18.get('dimensions') or {}
            gate18 = sa18.get('conversion_gate') or {}
            print(f"  Turn 18: sa_version={sa18.get('state_version')} sd_version={sd18.get('source_state_version')} action={sd18.get('primary_action')} commit={dim18.get('commitment')} gate_open={gate18.get('is_open')}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
