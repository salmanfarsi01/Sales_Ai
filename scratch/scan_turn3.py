import json, glob

for f in glob.glob('reports/**/*.json', recursive=True):
    try:
        d = json.load(open(f, encoding='utf-8'))
        tl = d.get('timeline', [])
        if len(tl) >= 3:
            s3 = tl[2]
            sd3 = s3.get('strategic_decision') or {}
            sa3 = s3.get('state_after') or {}
            sd_v = sd3.get('source_state_version')
            sa_v = sa3.get('state_version')
            act = sd3.get('primary_action')
            gate = sa3.get('conversion_gate', {}).get('is_open') if sa3.get('conversion_gate') else None
            # check if sd_v > 10 or gate is True or commitment == 1.0
            dim = sa3.get('dimensions') or {}
            comm = dim.get('commitment')
            if sd_v and sd_v > 10:
                print(f"FOUND LEAK in {f} Turn 3: sa_v={sa_v} sd_v={sd_v} act={act} gate={gate} comm={comm}")
            if comm == 1.0:
                print(f"FOUND COMMIT 1.0 in {f} Turn 3: sa_v={sa_v} sd_v={sd_v} act={act}")
    except Exception as e:
        pass
print("Scan complete.")
