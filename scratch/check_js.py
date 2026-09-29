with open(r'c:\dev\Sales_Ai\web\conversation_state_replay.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx, line in enumerate(lines, 1):
    if 'activeTurnIndex' in line or 'currentReport' in line:
        print(f"{idx}: {line.strip()[:100]}")
