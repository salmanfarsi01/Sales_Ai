import json, glob, os

for root, dirs, files in os.walk(r'c:\dev'):
    if '.git' in root or 'node_modules' in root:
        continue
    for f in files:
        if f.endswith('.json'):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    content = fh.read()
                    if "Thursday at 3 works, and my wife will be there." in content:
                        print("MATCH:", p)
            except Exception:
                pass
