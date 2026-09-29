with open(r'c:\dev\Sales_Ai\copilot\core_intelligence_engine.py', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if 'source_state_version' in line or 'state_version' in line:
            print(f"{idx}: {line.strip()}")
