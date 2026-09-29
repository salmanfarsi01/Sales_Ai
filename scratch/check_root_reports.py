import json, glob

for f in glob.glob('reports/*.json'):
    try:
        d = json.load(open(f, encoding='utf-8'))
        tl = d.get('timeline', [])
        has_sd = sum(1 for s in tl if s.get('strategic_decision') is not None)
        print(f"{f}: turns={len(tl)} sd_count={has_sd}")
    except Exception as e:
        print(f"Error {f}: {e}")
