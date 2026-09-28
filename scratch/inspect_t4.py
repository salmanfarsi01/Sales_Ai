import json

with open('reports/synthetic/conversation_state_sim_validation_test.json') as f:
    curr = json.load(f)

for step in curr['timeline']:
    if step['turn_id'] == 4:
        print("TURN 4 DETAILS:")
        print("Text:", step.get('text'))
        print("Speaker:", step.get('speaker_id'))
        gate = step['state_after'].get('conversion_gate', {})
        print("Gate is_open:", gate.get('is_open'))
        print("Gate status:", gate.get('status'))
        print("Push strength:", step['state_after'].get('push_strength'))
        print("Conditions:")
        for c in gate.get('conditions', []):
            print(f"  {c.get('condition_name')}: met={c.get('met')} val={c.get('score_or_value')} reason={c.get('reason')} ev={c.get('evidence_turn_ids')}")
        print("Facts:")
        for fact in step['state_after'].get('facts', []):
            print(f"  {fact.get('fact_key')}: {fact.get('fact_value')} (turn {fact.get('source_turn_id')})")
        print("Decision structure:", step['state_after'].get('decision_structure'))
        print("Readiness:", step['state_after'].get('readiness'))
