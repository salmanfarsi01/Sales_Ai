import json

with open(r'C:\Users\Softvence\.gemini\antigravity-ide\brain\d0d8e2fa-25b0-4a87-be35-ffb1a35371b4\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('step_index') == 3367:
            print("FOUND 3367:")
            with open('scratch/step3367_response.txt', 'w', encoding='utf-8') as out:
                out.write(d.get('content', ''))
            print("Wrote to scratch/step3367_response.txt, len:", len(d.get('content', '')))
            break
