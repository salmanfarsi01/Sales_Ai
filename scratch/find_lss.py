with open(r'c:\dev\Sales_Ai\web\conversation_state_replay.html', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'loadSampleScript' in line:
            print(f"{i}: {line.strip()}")
