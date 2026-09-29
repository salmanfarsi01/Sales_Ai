import json, glob

for f in sorted(glob.glob('reports/conversation_state_test_*.json'), key=lambda x: -json.load(open(x, encoding='utf-8')).get('total_turns', 0)):
    try:
        d = json.load(open(f, encoding='utf-8'))
        tl = d.get('timeline', [])
        for step in tl:
            sd = step.get('strategic_decision')
            if sd:
                ev = sd.get('evidence_considered', [])
                for e in ev:
                    if 'Thursday' in e or 'wife' in e:
                        print(f"FOUND LEAK in {f} Turn {step.get('turn_id')}:")
                        print("  Text of turn:", step.get('text'))
                        print("  Evidence considered:", ev)
                        print("  Strategic decision call_id:", sd.get('call_id'))
                        print("  Source state version:", sd.get('source_state_version'))
                        break
    except Exception as e:
        pass
print("Search done.")
