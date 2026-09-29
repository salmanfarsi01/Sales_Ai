import os

for root, dirs, files in os.walk(r"c:\dev"):
    if ".git" in root or "node_modules" in root or "__pycache__" in root:
        continue
    for f in files:
        if f.endswith((".md", ".txt")):
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                    if "Issue A" in text or "Point 0" in text or "Test A" in text or "Turn 3" in text:
                        print(f"MATCH: {path}")
            except Exception as e:
                pass
