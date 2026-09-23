import json

with open(r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT' and data.get('step_index') > 3850 and data.get('step_index') < 4228:
            print(f"=== STEP {data.get('step_index')} ===")
            print(data.get('content')[:1500])
