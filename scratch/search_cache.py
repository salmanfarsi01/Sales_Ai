import ast, os

for root, dirs, files in os.walk(r'c:\dev\Sales_Ai\copilot'):
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            with open(p, 'r', encoding='utf-8') as fh:
                text = fh.read()
                for line_no, line in enumerate(text.splitlines(), 1):
                    if any(term in line.lower() for term in ['latest_decision', 'latest_', 'cache', 'global ']):
                        if not line.strip().startswith('#'):
                            print(f"{f}:{line_no}: {line.strip()[:100]}")
