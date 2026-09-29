import json

d = json.load(open('reports/synthetic/conversation_state_sim_mukmxsle.json', encoding='utf-8'))
for s in d['timeline']:
    print(f"Turn {s['turn_id']}: spk={s['speaker_id']}, text=\"{s['text']}\"")
    sd = s.get('strategic_decision')
    if sd:
        print('  sd.call_id:', sd.get('call_id'))
        print('  sd.source_state_version:', sd.get('source_state_version'))
        print('  sd.primary_action:', sd.get('primary_action'))
        print('  sd.evidence_considered:', sd.get('evidence_considered'))
