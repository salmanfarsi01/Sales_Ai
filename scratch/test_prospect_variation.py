from fastapi import FastAPI
from fastapi.testclient import TestClient
from copilot.conditional_retrieval_router import get_conditional_retrieval_router

app = FastAPI()
app.include_router(get_conditional_retrieval_router())
client = TestClient(app)

res = client.post("/api/conditional-retrieval/multi-turn-simulate", json={
    "script_name": "canonical_8_turn",
    "prospect_lines": [
        "What are your consultation hours? Can we meet Thursday at 4?"
    ],
    "user_id": "rep_john",
    "use_real_llm": False
})

data = res.json()
print("Teleprompter Line:", data["turns"][0]["teleprompter_line"])
