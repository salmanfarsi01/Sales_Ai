import json

data = json.load(open('reports/synthetic/conversation_state_sim_mueyx6wy.json', encoding='utf-8'))

for step in data['timeline']:
    tid = step['turn_id']
    if tid in [4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18]:
        sa = step['state_after']
        read = sa.get('readiness', {}) or {}
        print(f"Turn {tid:02d} [{step['speaker_id'][:6]}]: \"{step['text'][:40]}\"")
        print(f"   readiness_score={read.get('readiness_score')}, uncapped={read.get('uncapped_score')}")
        print(f"   emotional={read.get('emotional_readiness')}, logical={read.get('logical_readiness')}, logistical={read.get('logistical_readiness')}, decision={read.get('decision_readiness')}")
        print(f"   blocker_caps={read.get('active_blocker_caps')}, capped_reason={read.get('capped_reason')}")
        print()
