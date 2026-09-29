import json, sys

with open(r'C:\Users\Softvence\.gemini\antigravity-ide\brain\d0d8e2fa-25b0-4a87-be35-ffb1a35371b4\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index')
        if idx in (3367, 3368, 3369, 3370):
            print(f"=== STEP {idx} (type={d.get('type')}) ===")
            content = d.get('content', '')
            if content:
                sys.stdout.buffer.write(content[:1500].encode('utf-8'))
                print("\n")
