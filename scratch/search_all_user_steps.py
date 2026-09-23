import json

p = r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript.jsonl'
with open(p, encoding='utf-8') as f:
    for i, line in enumerate(f):
        step = json.loads(line)
        if step.get('type') == 'USER_INPUT':
            content = step.get('content', '')
            if 'dormancy' in content.lower() or '4.' in content or '#4' in content:
                print(f"Step {i}:")
                with open(f"scratch/user_step_{i}.txt", "w", encoding="utf-8") as out:
                    out.write(content)
                print(f"Saved scratch/user_step_{i}.txt, len={len(content)}")
