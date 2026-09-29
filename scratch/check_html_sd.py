with open(r'c:\dev\Sales_Ai\web\conversation_state_replay.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's check how col4 is populated or if any global variable holds strategic_decision
for line_no, line in enumerate(text.splitlines(), 1):
    if any(k in line for k in ['step.strategic_decision', 'strategic_decision', 'col4Body']):
        print(f"{line_no}: {line.strip()[:100]}")
