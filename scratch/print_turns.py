import json

data = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))
for s in data['timeline']:
    print(f"Turn {s['turn_id']:02d} ({s['speaker_id']}): \"{s['text']}\"")

