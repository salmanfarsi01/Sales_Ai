import json

data = json.load(open('reports/synthetic/conversation_state_sim_mvdoiaih.json', encoding='utf-8'))
print(f"Total turns: {len(data['timeline'])}")
for s in data['timeline']:
    t = s['turn_id']
    spk = s['speaker_id']
    txt = s['text']
    sa = s['state_after']
    ds = sa.get('decision_structure', {})
    ce = sa.get('conversion_event')
    parts = ce.get('participants') if ce else None
    stks = ds.get('stakeholders', [])
    print(f"T{t:02d} [{spk}]: {txt}")
    if stks:
        print(f"     Stakeholders: {stks}")
    if ce:
        print(f"     ConversionEvent: type={ce.get('event_type')}, time={ce.get('scheduled_time')}, parts={parts}")
