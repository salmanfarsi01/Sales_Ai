import json

with open("reports/synthetic/conversation_state_sim_mu5du135.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total turns in report:", data.get("total_turns"))
print("Final state last_updated_turn_id:", data["final_state"].get("last_updated_turn_id"))
print("Final state changes count:", len(data["final_state"].get("change_history", [])))
print("\nFinal state change_history:")
for ch in data["final_state"].get("change_history", []):
    print(f"  Turn {ch.get('triggering_turn_id')}: {ch.get('field_path')} - {ch.get('reason')}")

print("\nPer-turn state changes:")
for t in data.get("timeline", []):
    tid = t.get("turn_id")
    spk = t.get("speaker_id")
    txt = t.get("text")
    mat = t.get("materiality", {})
    changes = t.get("state_changes", [])
    print(f"\nTurn {tid} ({spk}): \"{txt}\"")
    print(f"  Material: {mat.get('is_material')} Targets: {mat.get('affected_targets')} Reason: {mat.get('reason')}")
    print(f"  Changes ({len(changes)}):")
    for c in changes:
        print(f"    - {c.get('target')}: {c.get('reason')}")

print("\nFinal State Objections:")
for obj in data["final_state"].get("objections", []):
    print(f"  Category: {obj.get('canonical_category')}, State: {obj.get('lifecycle_state')}, Superseded By: {obj.get('superseded_by_objection_id')}")

print("\nFinal State Decision Structure:")
print(json.dumps(data["final_state"].get("decision_structure"), indent=2))

print("\nFinal State Facts:")
print(json.dumps(data["final_state"].get("facts"), indent=2))
