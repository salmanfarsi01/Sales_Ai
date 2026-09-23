import sys, os
sys.path.insert(0, os.path.abspath("."))
import json

d = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))
for t in d['timeline']:
    if t['turn_id'] in (14, 15, 16, 17, 18):
        print(f"Turn {t['turn_id']}: trust={t['evidence_bundle']['trust']['score']}, eng={t['evidence_bundle']['engagement']['score']}, tension={t['evidence_bundle']['emotion']['tension_level']}")
