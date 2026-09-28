import json
with open('reports/synthetic/conversation_state_sim_validation_test.json') as f:
    curr = json.load(f)

for step in curr['timeline'][:7]:
    t = step['turn_id']
    st = step['state_after']
    ev = st.get('conversion_event')
    gate = st.get('conversion_gate')
    push = st.get('push_strength')
    print(f"T{t} [{step.get('speaker_id')}]: '{step.get('text')}' -> ev={ev.get('status') if ev else None}, push={push.get('state') if push else None}, gate_open={gate.get('is_open') if gate else None}")
