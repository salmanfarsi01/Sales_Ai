import json

p = r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript.jsonl'
with open(p, encoding='utf-8') as f:
    for i, line in enumerate(f):
        step = json.loads(line)
        content = step.get('content', '')
        if step.get('type') == 'USER_INPUT':
            if '4.' in content or 'dormancy' in content.lower():
                print(f"Step {i}:")
                for line in content.split('\n'):
                    if any(k in line.lower() for k in ['1.', '2.', '3.', '4.', '5.', '6.', 'dormancy', 'momentum']):
                        print("  ", line[:120])
