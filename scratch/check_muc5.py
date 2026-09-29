import json

p = r'c:\dev\Sales_Ai\reports\synthetic\archive\conversation_state_sim_muc5lyux_2026-09-22T04-11-36-167506+00-00.json'
d = json.load(open(p, encoding='utf-8'))
print('Call SID:', d.get('call_sid'))
print('Total turns:', len(d.get('timeline', [])))
for i, s in enumerate(d.get('timeline', [])):
    s_str = json.dumps(s)
    if 'Thursday at 3' in s_str or 'my wife will be there' in s_str:
        print(f"Match in Turn {s.get('turn_id')}:")
        print('  text:', s.get('text'))
        sd = s.get('strategic_decision')
        if sd:
            print('  sd:', sd)
        sa = s.get('state_after', {})
        print('  sa state_version:', sa.get('state_version'))
