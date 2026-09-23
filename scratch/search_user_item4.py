import json

p = r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript_full.jsonl'
with open(p, encoding='utf-8') as f:
    for i, line in enumerate(f):
        step = json.loads(line)
        c = step.get('content', '')
        if '4.' in c and ('dormancy' in c.lower() or 'dormant' in c.lower() or 'aging' in c.lower() or '3 turns' in c.lower()):
            if 'USER' in step.get('source', '') or step.get('type') == 'USER_INPUT':
                print(f"Step {i}, source={step.get('source')}, type={step.get('type')}")
                with open(f"scratch/client_req_step_{i}.txt", "w", encoding="utf-8") as out:
                    out.write(c)
                print(f"Saved scratch/client_req_step_{i}.txt, len={len(c)}")
