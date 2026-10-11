import json

path = r"C:\Users\Softvence\.gemini\antigravity-ide\brain\b987c4a3-61b3-4d2b-9f24-0facb1303465\.system_generated\logs\transcript.jsonl"
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i in (1282, 1406):
            obj = json.loads(line)
            print(f"--- Line {i} Step {obj.get('step_index')} {obj.get('type')} ---")
            print("Tool calls:", obj.get("tool_calls"))
