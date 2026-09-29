import json, glob

for f in glob.glob('reports/**/*.json', recursive=True):
    try:
        d = json.load(open(f, encoding='utf-8'))
        tl = d.get('timeline', [])
        for step in tl:
            sd = step.get('strategic_decision')
            if sd and sd.get('evidence_considered'):
                ev = sd.get('evidence_considered')
                ev_str = " ".join(ev)
                if "Thursday at 3" in ev_str and "Contact preferences" in ev_str:
                    print(f"FOUND MATCH in {f} Turn {step.get('turn_id')}:")
                    print("  call_sid in file:", d.get('call_sid'))
                    print("  step text:", step.get('text'))
                    print("  decision call_id:", sd.get('call_id'))
                    print("  source_state_version:", sd.get('source_state_version'))
                    print("  evidence:")
                    for e in ev:
                        print("    -", e)
    except Exception:
        pass
