from fastapi import FastAPI
from fastapi.testclient import TestClient
from copilot.conditional_retrieval_router import get_conditional_retrieval_router

app = FastAPI()
app.include_router(get_conditional_retrieval_router())
client = TestClient(app)

client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
scripts = client.get("/api/conditional-retrieval/scripts").json()

for sid in ["messy_unstructured_5_turn", "adversarial_security_traps_5_turn"]:
    s = next(x for x in scripts if x["id"] == sid)
    res = client.post("/api/conditional-retrieval/multi-turn-simulate", json={
        "user_id": "rep_john",
        "call_id": f"call_{sid}",
        "script_name": s["name"],
        "prospect_lines": s["lines"],
    })
    data = res.json()
    print(f"\n=== Script: {sid} | Passed: {data['passed_turns_count']}/{data['total_turns']} ===")
    for t in data["turns"]:
        if t["score"] != "PASS" or t.get("score_reasons"):
            print(f"Turn {t['turn_index']}: {t['prospect_line']}")
            print(f"  Line: {t['teleprompter_line']}")
            print(f"  Score: {t['score']} | Reasons: {t.get('score_reasons')}")
            print(f"  Violations: {t.get('hallucination_violations')}")
