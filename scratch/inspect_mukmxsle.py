import json

with open("reports/synthetic/conversation_state_sim_mukmxsle.json", encoding="utf-8") as f:
    d = json.load(f)

print(f"Call SID: {d['call_sid']}")
print(f"Total Turns: {len(d['timeline'])}\n")

for t in d['timeline']:
    sd = t.get('strategic_decision') or {}
    sa = t.get('state_after') or {}
    cc = sa.get('contact_compliance') or {}
    gate = sa.get('conversion_gate') or {}
    read = sa.get('readiness') or {}
    
    print(f"=== TURN {t['turn_id']} ({t['speaker_id']}) ===")
    print(f"Spoken: \"{t['text']}\"")
    print(f"State Version: V{sa.get('state_version')}")
    print(f"Core Decision ID: {sd.get('decision_id')}")
    print(f"  Primary Action:     {sd.get('primary_action')}")
    print(f"  Secondary Action:   {sd.get('secondary_action')}")
    print(f"  Push Strength:      {sd.get('push_strength')}")
    print(f"  Objective:          {sd.get('strategic_objective')}")
    print(f"  Reason Codes:       {sd.get('reason_codes')}")
    print(f"  Do Not Do:          {sd.get('do_not_do')}")
    print(f"  What To Protect:    {sd.get('what_to_protect')}")
    print(f"  Conversion Gate:    {gate.get('status')}")
    print(f"  Readiness Score:    {read.get('readiness_score')} (Blockers: {read.get('active_blocker_caps')})")
    print(f"  Hard Boundary:      {cc.get('hard_boundary_active')} (Reason: {cc.get('hard_boundary_reason')})")
    print(f"  Contact Prefs:      {[p.get('prohibited_behavior') for p in cc.get('contact_preferences', [])]}")
    print()
