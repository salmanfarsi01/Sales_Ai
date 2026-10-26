import sys
import os
import json
import time

sys.path.insert(0, ".")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starlette.testclient import TestClient
from copilot.fastapi_app import app

client = TestClient(app)
client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
scripts = client.get("/api/conditional-retrieval/scripts").json()

script_ids = [
    "canonical_8_turn",
    "messy_unstructured_5_turn",
    "adversarial_security_traps_5_turn",
    "repeated_objection_3_turn",
]

comparison_results = {}

for mode_name, use_llm in [("deterministic", False), ("live_llm", True)]:
    comparison_results[mode_name] = {}
    print(f"\n=======================================================")
    print(f"RUNNING MODE: {mode_name.upper()} (use_real_llm={use_llm})")
    print(f"=======================================================")
    
    for sid in script_ids:
        s = next(x for x in scripts if x["id"] == sid)
        req = {
            "user_id": "rep_john",
            "call_id": f"bench_{mode_name}_{sid}",
            "script_name": s["name"],
            "prospect_lines": s["lines"],
            "use_real_llm": use_llm,
        }
        t0 = time.perf_counter()
        res = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req).json()
        sim_duration = round((time.perf_counter() - t0) * 1000, 2)
        comparison_results[mode_name][sid] = res
        
        print(f"\n--- Script: {s['name']} [{res.get('passed_turns_count')}/{res.get('total_turns')} PASSED] ({sim_duration}ms total) ---")
        for t in res["turns"]:
            idx = t["turn_index"]
            prospect = t["prospect_line"]
            spoken = t["teleprompter_line"]
            action = t["core_action"]
            trigger = t["core_trigger"]
            facts_count = len(t["facts_delivered"])
            gen_src = t.get("generation_source", "unknown")
            score = t.get("score", "UNKNOWN")
            lat = t["latency_ms"]
            print(f"  T{idx} [{score}] ({gen_src}, {lat:.1f}ms):")
            print(f"     PROSPECT: \"{prospect}\"")
            print(f"     SPOKEN:   \"{spoken}\"")
            print(f"     ACTION: {action} | TRIGGER: {trigger} | FACTS: {facts_count}")
            if t["hallucination_violations"]:
                print(f"     VIOLATIONS: {t['hallucination_violations']}")
            if score == "FAIL":
                print(f"     REASONS: {t['score_reasons']}")

with open("scratch/scripts_comparison_output.json", "w", encoding="utf-8") as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

print("\nSaved full comparison to scratch/scripts_comparison_output.json")
