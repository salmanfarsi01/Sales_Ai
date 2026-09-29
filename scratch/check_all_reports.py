import json, glob

for f in glob.glob('reports/synthetic/*.json'):
    try:
        d = json.load(open(f, encoding='utf-8'))
        tl = d.get('timeline', [])
        has_sd = sum(1 for s in tl if s.get('strategic_decision') is not None)
        print(f"{f}: total_turns={len(tl)} turns_with_sd={has_sd}")
    except Exception as e:
        print(f"{f}: Error {e}")
