import json

with open("reports/synthetic/conversation_state_sim_muy04bh0.json") as f:
    data = json.load(f)

print(f"Call SID: {data.get('call_sid')}")
timeline = data.get("timeline", [])
for t in timeline:
    tid = t.get("turn_id")
    speaker = t.get("speaker")
    text = t.get("text", "")
    dec = t.get("strategic_decision", {})
    sa = t.get("state_after", {})
    facts = sa.get("facts", [])
    objs = sa.get("objections", [])
    print(f"Turn {tid} [{speaker}]: \"{text}\"")
    print(f"   action={dec.get('primary_action')}, push={dec.get('push_strength')}, carried={dec.get('carried_forward')}")
    print(f"   reasons={dec.get('reason_codes')}")
    print(f"   dec_facts={dec.get('referenced_fact_ids')}, dec_objs={dec.get('referenced_objection_ids')}")
    print(f"   state_facts={[(f.get('fact_id'), f.get('status'), f.get('fact_key'), f.get('superseded_by_fact_id')) for f in facts]}")
    print(f"   state_objs={[(o.get('objection_id'), o.get('lifecycle_state'), o.get('canonical_category')) for o in objs]}")
    print("-" * 60)
