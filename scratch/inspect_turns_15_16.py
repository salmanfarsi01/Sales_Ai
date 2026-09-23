import json

d = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))
for t in d['timeline']:
    if t['turn_id'] in (15, 16):
        print(f"=== TURN {t['turn_id']} ({t['speaker_id']}) ===")
        print("Text:", t['text'])
        print("Evidence recurrence_id:", t['evidence_bundle'].get('recurrence_id'))
        print("Evidence recurrence_type:", t['evidence_bundle'].get('recurrence_type'))
        print("Evidence contact_pref:", t['evidence_bundle'].get('contact_preference'))
        print("Materiality is_material:", t['materiality'].get('is_material'))
        print("Materiality affected_targets:", t['materiality'].get('affected_targets'))
        print("Materiality reasoning:", t['materiality'].get('reasoning'))
        print("State changes:", [c.get('field_path') for c in t.get('state_changes', [])])
        print()
