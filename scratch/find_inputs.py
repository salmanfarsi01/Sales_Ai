import json

with open(r'C:\Users\Softvence\.gemini\antigravity-ide\brain\d0d8e2fa-25b0-4a87-be35-ffb1a35371b4\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            step = data.get('step_index')
            content = data.get('content', '')
            safe_content = content.encode('ascii', errors='replace').decode('ascii')
            print(f"=== STEP {step} ===")
            print(safe_content[:300])
            print("-----------------------")
