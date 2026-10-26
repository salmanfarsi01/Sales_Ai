from fastapi import FastAPI
from fastapi.testclient import TestClient
from copilot.conditional_retrieval_router import get_conditional_retrieval_router

app = FastAPI()
app.include_router(get_conditional_retrieval_router())
client = TestClient(app)

canonical_lines = [
    "Hi John, thanks for calling. We're thinking about selling.",
    "What is your actual sales experience and track record in Westlake?",
    "Okay, makes sense. That sounds pretty good.",
    "What is your standard commission fee to list our home?",
    "Another agent told me Sarah offers a $5,000 cash back rebate. Can you match that?",
    "Do you guarantee you can sell it in 7 days? I heard someone claim that.",
    "What are your consultation hours? Can we meet tomorrow afternoon at 3:00 PM?",
    "How long does the residential home inspection contingency take?",
]

res = client.post("/api/conditional-retrieval/multi-turn-simulate", json={
    "script_name": "canonical_8_turn",
    "prospect_lines": canonical_lines,
    "user_id": "rep_john",
    "use_real_llm": False
})

data = res.json()
print("Passed:", data.get("passed_turns_count"), "/", data.get("total_turns"))
for t in data["turns"]:
    print(f"\n--- Turn {t['turn_index']}: {t['prospect_line']} ---")
    print(f"  Action: {t['core_action']} | Trigger: {t['core_trigger']}")
    print(f"  Facts ({len(t['facts_delivered'])}): {[f['knowledge_id'] for f in t['facts_delivered']]}")
    print(f"  Teleprompter Line: {t['teleprompter_line']}")
    print(f"  Score: {t['score']}")
    if t.get("score_reasons"):
        print(f"  Score Reasons: {t['score_reasons']}")
