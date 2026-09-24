import json

f_old = 'reports/synthetic/conversation_state_sim_mucj0p5s.json'
f_new = 'reports/synthetic/conversation_state_sim_mueyx6wy.json'

with open(f_old, encoding='utf-8') as f:
    d_old = json.load(f)
with open(f_new, encoding='utf-8') as f:
    d_new = json.load(f)

print(f"Old file: {f_old} | New file: {f_new}")
print(f"Turns: Old={len(d_old['timeline'])} vs New={len(d_new['timeline'])}")

print("\n--- TURN COMPARISON ---")
print("| Turn | Old Stage -> New Stage | Old Mom -> New Mom | Old Trend -> New Trend | New Live Objections |")
print("|:---|:---|:---:|:---:|:---|")

for i in range(min(len(d_old['timeline']), len(d_new['timeline']))):
    t_old = d_old['timeline'][i]
    t_new = d_new['timeline'][i]
    tid = t_new['turn_id']
    
    st_old = t_old['state_after'].get('conversation_stage', 'none')
    st_new = t_new['state_after'].get('conversation_stage', 'none')
    
    m_old = t_old['state_after'].get('momentum', {}).get('momentum_score', 0) if t_old['state_after'].get('momentum') else 0
    m_new = t_new['state_after'].get('momentum', {}).get('momentum_score', 0) if t_new['state_after'].get('momentum') else 0
    
    tr_old = t_old['state_after'].get('momentum', {}).get('trend', 'none') if t_old['state_after'].get('momentum') else 'none'
    tr_new = t_new['state_after'].get('momentum', {}).get('trend', 'none') if t_new['state_after'].get('momentum') else 'none'
    
    objs_new = [f"{o['canonical_category']}:{o['lifecycle_state']}" for o in t_new['state_after'].get('objections', []) if o['lifecycle_state'] not in ('resolved', 'superseded', 'dormant')]
    objs_str = ", ".join(objs_new) if objs_new else "none"
    
    print(f"| T{tid:02d} | `{st_old}` -> `{st_new}` | {m_old:4.1f} -> **{m_new:4.1f}** | `{tr_old}` -> `{tr_new}` | {objs_str} |")

print("\n--- CONVERSION SUMMARY COMPARISON ---")
cs_old = d_old.get('conversion_summary', {})
cs_new = d_new.get('conversion_summary', {})

for k in ['gate_is_open', 'gate_status', 'push_strength', 'momentum_score', 'momentum_trend', 'readiness_score', 'deal_milestone_status', 'milestone_label', 'open_concerns_summary']:
    print(f"{k:25s} : Old={cs_old.get(k)} | New={cs_new.get(k)}")
