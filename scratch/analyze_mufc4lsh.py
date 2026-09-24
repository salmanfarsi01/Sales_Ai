import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_presentation import build_conversion_presentation

report_path = r'reports/synthetic/conversation_state_sim_mufc4lsh.json'
with open(report_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Call SID: {data.get('call_sid')}")
print(f"Total turns: {data.get('total_turns')}")
print(f"Source: {data.get('source')}")
print(f"Created at: {data.get('created_at')}")

timeline = data.get('timeline', [])
print(f"Timeline entries: {len(timeline)}")

print("\n" + "="*80)
print("TURN-BY-TURN AUDIT OF SIM_MUFC4LSH.JSON")
print("="*80)
print(f"| Turn | Spk | Stage | Targets | CE Status | FOB | Commit | Mom | Delta | Trend | Read | Logist | Decis | Blockers |")
print(f"|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

for entry in timeline:
    t = entry.get('turn_id')
    spk = entry.get('speaker_id', '')[:6]
    txt = entry.get('text', '')
    sa = entry.get('state_after', {})
    
    stage = sa.get('conversation_stage', '')
    mat = entry.get('materiality', {})
    targets = mat.get('affected_targets', [])
    targets_str = ",".join(targets) if targets else "none"
    
    ce = sa.get('conversion_event')
    ce_stat = ce.get('status') if ce else 'None'
    
    mom = sa.get('momentum', {}) or {}
    fs = mom.get('family_scores', {})
    fob = fs.get('future_operational_behavior', 0.0)
    com = fs.get('commitment_behavior', 0.0)
    m_score = mom.get('momentum_score', 0.0)
    delta = mom.get('trend_delta')
    d_str = f"{delta:+5.1f}" if delta is not None else "  N/A"
    trend = mom.get('trend', '')
    
    read = sa.get('readiness', {}) or {}
    r_score = read.get('readiness_score', 0.0)
    r_log = read.get('logistical_readiness', 0.0)
    r_dec = read.get('decision_readiness', 0.0)
    blockers = ",".join(read.get('active_blocker_caps', [])) or "none"
    
    print(f"| T{t:02d} | {spk:6s} | {stage:20s} | {targets_str:22s} | {ce_stat:9s} | {fob:4.1f} | {com:4.1f} | {m_score:5.1f} | {d_str} | {trend:10s} | {r_score:4.1f} | {r_log:4.1f} | {r_dec:4.1f} | {blockers} |")

print("\n" + "="*80)
print("AUDITING THE 6 CORE REQUIREMENTS ON SIM_MUFC4LSH.JSON")
print("="*80)

# Check Point 1: Stages
stages = [s['state_after'].get('conversation_stage') for s in timeline]
print("Point 1 (Stage Tracking):")
print(f"  Turns 1-6:   {stages[:6]}")
print(f"  Turns 7-9:   {stages[6:9]}")
print(f"  Turns 10-14: {stages[9:14]}")
print(f"  Turns 15-17: {stages[14:17]}")
print(f"  Turn 18:     {stages[17]}")

# Check Point 2: Turn 15 Soft Boundary
t15 = timeline[14]
mat15 = t15.get('materiality', {})
print("\nPoint 2 (Turn 15 Soft Contact Preference):")
print(f"  affected_targets: {mat15.get('affected_targets')}")
print(f"  reasoning: {mat15.get('reasoning')}")
print(f"  contact_preference: {t15['state_after'].get('contact_compliance', {}).get('contact_preference')}")

# Check Point 3: Turn 16 Scheduling Constraint
t16 = timeline[15]
mat16 = t16.get('materiality', {})
print("\nPoint 3 (Turn 16 Scheduling Constraint):")
print(f"  affected_targets: {mat16.get('affected_targets')}")
print(f"  reasoning: {mat16.get('reasoning')}")
print(f"  access_constraints: {t16['state_after'].get('decision_structure', {}).get('access_constraints')}")

# Check Point 4: Objection Dormancy & Lifecycle (turns 17 & 18 obj_movement)
fs17 = timeline[16]['state_after'].get('momentum', {}).get('family_scores', {})
fs18 = timeline[17]['state_after'].get('momentum', {}).get('family_scores', {})
print("\nPoint 4 (Objection Lifecycle & Preservation):")
print(f"  Turn 17 objection_movement: {fs17.get('objection_movement')}")
print(f"  Turn 18 objection_movement: {fs18.get('objection_movement')}")
objs18 = timeline[17]['state_after'].get('objections', [])
print(f"  Turn 18 live objections count: {len(objs18)}")
for o in objs18:
    print(f"    - {o.get('canonical_category')} : {o.get('lifecycle_state')} (recurr={o.get('recurrence_count')})")

# Check Point 5: Momentum & 7-family scoring at Turn 18, and same-turn updates at Turns 7, 9, 18
print("\nPoint 5 & Audited Lag Check:")
print(f"  Turn 7 FOB: {timeline[6]['state_after'].get('momentum', {}).get('family_scores', {}).get('future_operational_behavior')} (expected 80.0)")
print(f"  Turn 7 Commit: {timeline[6]['state_after'].get('momentum', {}).get('family_scores', {}).get('commitment_behavior')} (expected 81.5)")
print(f"  Turn 7 Trend: {timeline[6]['state_after'].get('momentum', {}).get('trend')} (delta={timeline[6]['state_after'].get('momentum', {}).get('trend_delta')})")
print(f"  Turn 9 FOB: {timeline[8]['state_after'].get('momentum', {}).get('family_scores', {}).get('future_operational_behavior')} (expected 45.0)")
print(f"  Turn 9 Commit: {timeline[8]['state_after'].get('momentum', {}).get('family_scores', {}).get('commitment_behavior')} (expected 63.5)")
print(f"  Turn 9 Trend: {timeline[8]['state_after'].get('momentum', {}).get('trend')} (delta={timeline[8]['state_after'].get('momentum', {}).get('trend_delta')})")
print(f"  Turn 18 FOB: {fs18.get('future_operational_behavior')} (expected 80.0)")
print(f"  Turn 18 Commit: {fs18.get('commitment_behavior')} (expected 81.5)")
print(f"  Turn 18 Dec Clarity: {fs18.get('decision_structure_clarity')} (expected 95.0)")
print(f"  Turn 18 Mom Score: {timeline[17]['state_after'].get('momentum', {}).get('momentum_score')}")
print(f"  Turn 18 Trend: {timeline[17]['state_after'].get('momentum', {}).get('trend')} (delta={timeline[17]['state_after'].get('momentum', {}).get('trend_delta')})")

# Check Point 6: Conversion Presentation & Header
final_state = data.get('final_state', {})
from copilot.conversation_state_models import ConversationStateSnapshot
snap_final = ConversationStateSnapshot.model_validate(final_state)
pres = build_conversion_presentation(snap_final)
print("\nPoint 6 (Presentation & Milestone Status):")
print(f"  deal_milestone_status: {pres['deal_milestone_status']}")
print(f"  milestone_label:       {pres['milestone_label']}")
print(f"  open_concerns:         {pres['open_concerns']}")
print(f"  open_concerns_summary: {pres['open_concerns_summary']}")
