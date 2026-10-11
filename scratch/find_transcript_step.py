import json

path = r"C:\Users\Softvence\.gemini\antigravity-ide\brain\b987c4a3-61b3-4d2b-9f24-0facb1303465\.system_generated\logs\transcript.jsonl"
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "0.905" in line:
            obj = json.loads(line)
            print(f"Line {i}, Step {obj.get('step_index')}, Type {obj.get('type')}")
            content = str(obj.get("content", ""))
            # print snippet around 0.905
            idx = content.find("0.905")
            if idx != -1:
                print("Snippet:", content[max(0, idx-100): min(len(content), idx+100)])
