import json

with open('reports/synthetic/conversation_state_sim_validation_test.json') as f:
    data = json.load(f)

print(f"{'Turn':<5} | {'Speaker':<7} | {'Gate':<6} | {'Push Strength':<20} | {'Readiness':<10} | {'Dims (emo,log,logist,dec)':<26} | {'Event':<10}")
print("-" * 100)

for step in data['timeline']:
    t = step['turn_id']
    sp = step.get('speaker_id', '')
    gate = step['state_after'].get('conversion_gate') or {}
    gate_open = gate.get('is_open')
    push_obj = step['state_after'].get('push_strength') or {}
    push = push_obj.get('state', 'None')
    read = step['state_after'].get('readiness') or {}
    r_score = read.get('readiness_score')
    r_str = f"{r_score:.1f}" if r_score is not None else "None"
    dims_str = f"({read.get('emotional_readiness')}, {read.get('logical_readiness')}, {read.get('logistical_readiness')}, {read.get('decision_readiness')})"
    ev = step['state_after'].get('conversion_event')
    ev_str = ev.get('status') if ev else "None"
    print(f"T{t:<4} | {sp:<7} | {str(gate_open):<6} | {push:<20} | {r_str:<10} | {dims_str:<26} | {ev_str:<10}")
