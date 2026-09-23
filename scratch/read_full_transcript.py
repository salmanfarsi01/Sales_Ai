import json

p = r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript_full.jsonl'
with open(p, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 4150 and i <= 4175:
            step = json.loads(line)
            if step.get('type') == 'USER_INPUT':
                print(f"=== STEP {i} IN TRANSCRIPT_FULL ===")
                with open("scratch/full_client_requirements.txt", "w", encoding="utf-8") as out:
                    out.write(step.get('content', ''))
                print("Saved scratch/full_client_requirements.txt, len =", len(step.get('content', '')))
