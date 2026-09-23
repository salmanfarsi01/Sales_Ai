import json

p = r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript.jsonl'
with open(p, encoding='utf-8') as f:
    for line in f:
        step = json.loads(line)
        if step.get('type') == 'USER_INPUT':
            content = step.get('content', '')
            if '4.' in content and ('dormancy' in content.lower() or 'objection' in content.lower()):
                with open('scratch/found_user_prompt.txt', 'w', encoding='utf-8') as out:
                    out.write(content)
                print("Wrote user prompt to scratch/found_user_prompt.txt")
                break
