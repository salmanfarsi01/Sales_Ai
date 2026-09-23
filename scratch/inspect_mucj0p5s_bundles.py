import json

d = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))
print('Created at:', d.get('created_at'))
print('Source:', d.get('source'))
print('Report file:', d.get('report_file'))
print('Total turns:', len(d['timeline']))
for t in d['timeline']:
    b = t['evidence_bundle']
    trust = b['trust']['score']
    eng = b['engagement']['score']
    tension = b['emotion']['tension_level']
    mom = b['momentum']['score']
    print(f"Turn {t['turn_id']:02d} ({t['speaker_id'][:6]}): trust={trust:.2f}, eng={eng:.2f}, tension={tension:.2f}, mom={mom:.2f} | text=\"{t['text'][:35]}...\"")
