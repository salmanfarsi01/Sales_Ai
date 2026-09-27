import json
with open(r'C:\Users\Softvence\.gemini\antigravity-ide\brain\4fce2710-0f4d-4984-9ac9-9db77fbf6178\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('step_index') == 4610:
            with open('scratch/step_4610_content.txt', 'w', encoding='utf-8') as out:
                out.write(d.get('content', ''))
            print("Wrote step 4610 to scratch/step_4610_content.txt")
