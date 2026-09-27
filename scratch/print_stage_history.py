import json

d = json.load(open('reports/synthetic/conversation_state_sim_mufc4lsh.json', encoding='utf-8'))
print("=== CONVERSATION STAGE HISTORY IN SIM_MUFC4LSH.JSON ===")
for r in d['final_state']['stage_history']:
    entered = r.get('entered_turn_id')
    exited = r.get('exited_turn_id') or 'current'
    print(f"[{r.get('stage')}] turns {entered}..{exited}: {r.get('trigger_reason')}")
