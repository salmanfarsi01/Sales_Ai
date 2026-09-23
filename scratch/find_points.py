import json

with open(r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            content = data.get('content', '')
            if '5.' in content or '6.' in content or 'appointment' in content:
                print(f"--- Step {data.get('step_index')} ---")
                lines = content.split('\n')
                for l in lines:
                    if any(l.strip().startswith(x) for x in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '#1', '#2', '#3', '#4', '#5', '#6']):
                        print(l[:120])
