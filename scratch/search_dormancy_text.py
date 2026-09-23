import json

p = r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript.jsonl'
with open(p, encoding='utf-8') as f:
    for i, line in enumerate(f):
        step = json.loads(line)
        content = step.get('content', '')
        if 'Replace automatic objection dormancy' in content or 'dormancy aging based solely' in content or '4. Replace' in content:
            print(f"Step {i}:")
            print(content[:2000])
            print("="*60)
