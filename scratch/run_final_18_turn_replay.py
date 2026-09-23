import sys
sys.path.insert(0, ".")
import json
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_presentation import build_conversion_presentation

with open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8') as f:
    data = json.load(f)

mgr = ConversationStateManager(call_sid='sim_mucj0p5s_final_verification')

print("=== 18-TURN END-TO-END REPLAY VERIFICATION ===")
print("| Turn | Spk | Stage | Affected Targets | Live Objections | Momentum | Delta | Trend | dec_clar | fut_op | commit |")
print("|:---|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

for t in data['timeline']:
    eb = BehavioralSignalInputBundle(**t['evidence_bundle'])
    snap = mgr.process_turn_bundle(eb)
    fs = snap.momentum.family_scores
    live_objs = [
        f"{o.canonical_category} ({getattr(o.lifecycle_state, 'value', str(o.lifecycle_state))})"
        for o in snap.objections
        if o.lifecycle_state not in ('resolved', 'superseded', 'dormant')
    ]
    objs_str = ", ".join(live_objs) if live_objs else "None"
    targets = ", ".join(mgr.last_materiality.affected_targets) if mgr.last_materiality else "none"
    d_sign = f"+{snap.momentum.trend_delta:.1f}" if snap.momentum.trend_delta >= 0 else f"{snap.momentum.trend_delta:.1f}"
    print(f"| T{eb.turn_id:02d} | {eb.speaker_id[:3]} | `{snap.conversation_stage.value}` | `[{targets}]` | {objs_str} | **{snap.momentum.momentum_score:.1f}** | {d_sign} | `{snap.momentum.trend}` | {fs.get('decision_structure_clarity', 0):.0f} | {fs.get('future_operational_behavior', 0):.0f} | {fs.get('commitment_behavior', 0):.0f} |")

final_pres = build_conversion_presentation(mgr.current_state)
print("\n=== FINAL CONVERSION HEADER & PRESENTATION ===")
print("Milestone Status :", final_pres['deal_milestone_status'])
print("Milestone Label  :", final_pres['milestone_label'])
print("Open Concerns    :", final_pres['open_concerns'])
print("Open Concerns Line:", final_pres['open_concerns_summary'])
