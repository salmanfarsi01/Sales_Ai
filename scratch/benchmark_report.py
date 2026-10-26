import sys
sys.path.insert(0, ".")
from starlette.testclient import TestClient
from copilot.fastapi_app import app
import json

client = TestClient(app)
client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
scripts = client.get("/api/conditional-retrieval/scripts").json()

script_ids = [
    "canonical_8_turn",
    "messy_unstructured_5_turn",
    "adversarial_security_traps_5_turn",
    "repeated_objection_3_turn",
]

report = {}
for sid in script_ids:
    s = next(x for x in scripts if x["id"] == sid)
    req = {
        "user_id": "rep_john",
        "call_id": f"bench_{sid}",
        "script_name": s["name"],
        "prospect_lines": s["lines"],
        "use_real_llm": False,
    }
    res = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req).json()
    report[sid] = res

with open("scratch/bench_output.json", "w") as f:
    json.dump(report, f, indent=2)

print("Benchmark simulation completed successfully.")
