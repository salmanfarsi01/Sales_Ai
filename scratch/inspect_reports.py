import json

for p in [
    'reports/synthetic/archive/conversation_state_sim_muy04bh0_2026-10-07T11-05-46-884859+00-00.json',
    'reports/synthetic/conversation_state_sim_muy04bh0.json'
]:
    try:
        data = json.load(open(p, encoding='utf-8'))
        print('=== FILE:', p)
        for s in data['timeline']:
            d = s.get('strategic_decision', {})
            tid = s['turn_id']
            spk = s['speaker_id']
            act = d.get('primary_action')
            post = d.get('strategic_posture')
            push = d.get('push_strength')
            prompt = d.get('should_prompt')
            cf_turn = d.get('carried_forward_from_turn_id')
            p_txt = d.get('final_prompt_text') or d.get('gateway_fallback_stub')
            objs = [o.get('canonical_category') for o in s.get('state_after', {}).get('objections', [])]
            cev = s.get('state_after', {}).get('conversion_events', [])
            cev_stat = [e.get('status') for e in cev]
            print(f"Turn {tid} ({spk}): act={act}, post={post}, push={push}, prompt={prompt}, cf_turn={cf_turn}, events={cev_stat}, objs={objs}")
            print(f"   Prompt: {p_txt}")
    except Exception as e:
        print(f"Error {p}: {e}")
