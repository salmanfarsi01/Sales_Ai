import json
from starlette.testclient import TestClient
from copilot.fastapi_app import app

client = TestClient(app)
client.post('/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john')
res_s = client.get('/api/conditional-retrieval/scripts').json()
c8 = next(s for s in res_s if s['id'] == 'canonical_8_turn')

req = {
    'user_id': 'rep_john',
    'call_id': 'inspect_8_turn',
    'script_name': c8['name'],
    'prospect_lines': c8['lines'],
    'use_real_llm': False
}
res = client.post('/api/conditional-retrieval/multi-turn-simulate', json=req).json()
for t in res['turns']:
    print(f"=== Turn {t['turn_index']} ===")
    print("PROSPECT:    ", t['prospect_line'])
    print("ACTION/TRIG: ", t['core_action'], "|", t['core_trigger'])
    print("GATE:        ", t['gate_result'])
    print("FACTS:       ", [f['knowledge_id'] for f in t['facts_delivered']])
    print("WITHHELD:    ", [(w['knowledge_id'], w['reason']) for w in t['withheld']])
    print("TELEPROMPTER:", t['teleprompter_line'])
    print("LATENCIES:   ", t['stage_latencies'])
    print("SCORE:       ", t['score'], t['score_reasons'])
    print()
