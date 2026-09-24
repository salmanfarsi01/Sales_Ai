import json

path = r'reports/synthetic/conversation_state_sim_mueyx6wy.json'
with open(path, encoding='utf-8') as f:
    d = json.load(f)

print('Call SID:', d.get('call_sid'))
print('Total turns:', d.get('total_turns'))
print('Source:', d.get('source'))
print('Created at:', d.get('created_at'))
print('\n--- CONVERSION SUMMARY ---')
print(json.dumps(d.get('conversion_summary', {}), indent=2))

timeline = d.get('timeline', [])
print(f'\nTimeline length: {len(timeline)}')

print('\n--- TURN-BY-TURN SUMMARY ---')
for step in timeline:
    tid = step['turn_id']
    spk = step['speaker_id']
    txt = step['text'][:60]
    sa = step['state_after']
    mat = step.get('materiality', {})
    targets = mat.get('affected_targets', [])
    stage = sa.get('conversation_stage', 'none')
    mom = sa.get('momentum', {}) or {}
    m_score = mom.get('momentum_score', 0)
    m_trend = mom.get('trend', 'none')
    m_delta = mom.get('trend_delta', 0)
    fs = mom.get('family_scores', {})
    objs = sa.get('objections', [])
    obj_str = '; '.join([f"{o['canonical_category']}:{o['lifecycle_state']}" for o in objs]) if objs else 'none'
    print(f"T{tid:02d} [{spk:11s}] stage={stage:22s} targets={str(targets):35s} mom={m_score:4.1f} (d={m_delta:+4.1f}, {m_trend:10s}) dec={fs.get('decision_structure_clarity', 0):2.0f} fut={fs.get('future_operational_behavior', 0):2.0f} com={fs.get('commitment_behavior', 0):2.0f} obj_mov={fs.get('objection_movement', 0):2.0f} | objs={obj_str}")

print('\n--- FINAL STATE INSPECTION ---')
fs = d.get('final_state', {})
print('deal_milestone_status:', fs.get('deal_milestone_status'))
print('conversation_stage:', fs.get('conversation_stage'))
print('stage_history:', fs.get('stage_history'))
print('deal_disposition:', fs.get('deal_disposition'))
print('conversion_event:', fs.get('conversion_event'))
print('objections count:', len(fs.get('objections', [])))
for o in fs.get('objections', []):
    print('  -', o.get('canonical_category'), '| state:', o.get('lifecycle_state'), '| recurrence:', o.get('recurrence_count'), '| statement:', o.get('latest_statement'))

print('\n--- TURN 15 & 16 DETAILED CHECK ---')
for tid in (15, 16, 17, 18):
    if tid <= len(timeline):
        st = timeline[tid - 1]
        print(f"Turn {tid}:")
        print("  Speaker:", st['speaker_id'])
        print("  Text:", st['text'])
        print("  Materiality is_material:", st.get('materiality', {}).get('is_material'))
        print("  Materiality affected_targets:", st.get('materiality', {}).get('affected_targets'))
        print("  Materiality reasoning:", st.get('materiality', {}).get('reasoning'))
        sa = st['state_after']
        print("  Stage:", sa.get('conversation_stage'))
        print("  Contact Compliance:", sa.get('contact_compliance', {}).get('contact_preference'))
        print("  Access constraints:", sa.get('decision_structure', {}).get('access_constraints'))
        print("  Objections:", [(o.get('canonical_category'), o.get('lifecycle_state')) for o in sa.get('objections', [])])
