import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_presentation import build_conversion_presentation

# Replay using the exact evidence bundles from sim_mueyx6wy.json
data = json.load(open('reports/synthetic/conversation_state_sim_mueyx6wy.json', encoding='utf-8'))
mgr = ConversationStateManager(call_sid="audit_6_points_check")

snaps = []
materialities = []

for step in data['timeline']:
    eb = BehavioralSignalInputBundle(**step['evidence_bundle'])
    snap = mgr.process_turn_bundle(eb)
    snaps.append(snap.model_copy(deep=True))
    materialities.append(mgr.last_materiality)

print("=" * 60)
print("COMPREHENSIVE AUDIT OF CLIENT'S 6 POINTS")
print("=" * 60)

# Point 1: Stage Tracking
stages = [s.conversation_stage.value for s in snaps]
print(f"Point 1 (Stage Tracking):")
print(f"  Turns 1-6:   {stages[:6]}")
print(f"  Turns 7-9:   {stages[6:9]}")
print(f"  Turns 10-14: {stages[9:14]}")
print(f"  Turns 15-17: {stages[14:17]}")
print(f"  Turn 18:     {stages[17]}")
# Verify stage progression
assert stages[0] == "discovery"
assert stages[14] in ("discovery", "scheduling")
assert stages[17] in ("scheduling", "closed_won", "commitment_confirmed")

# Point 2: Turn 15 Soft contact preference not tagged as objection
mat15 = materialities[14]
snap15 = snaps[14]
print(f"\nPoint 2 (Turn 15 Soft Boundary):")
print(f"  affected_targets: {mat15.affected_targets}")
print(f"  reasoning: {mat15.reasoning}")
assert "objections" not in mat15.affected_targets, "FAIL: objections in Turn 15 affected_targets"
assert "active objection" not in mat15.reasoning.lower(), "FAIL: objection language in Turn 15 reasoning"
assert snap15.contact_compliance.contact_preference == "timing_restriction" or len(snap15.contact_compliance.contact_preferences) > 0

# Point 3: Turn 16 Scheduling constraint not tagged as objection-adjacent
mat16 = materialities[15]
snap16 = snaps[15]
print(f"\nPoint 3 (Turn 16 Scheduling Constraint):")
print(f"  affected_targets: {mat16.affected_targets}")
print(f"  reasoning: {mat16.reasoning}")
assert "objections" not in mat16.affected_targets, "FAIL: objections in Turn 16 affected_targets"
assert "active objection" not in mat16.reasoning.lower(), "FAIL: objection language in Turn 16 reasoning"
assert "Mornings unavailable" in snap16.decision_structure.access_constraints

# Point 4: Objection Dormancy & Lifecycle Preservation (obj_movement at 70.0 at T17-18)
fs17 = snaps[16].momentum.family_scores
fs18 = snaps[17].momentum.family_scores
print(f"\nPoint 4 (Objection Lifecycle & Preservation):")
print(f"  Turn 17 objection_movement: {fs17['objection_movement']}")
print(f"  Turn 18 objection_movement: {fs18['objection_movement']}")
assert fs17['objection_movement'] == 70.0, f"FAIL: T17 obj_movement={fs17['objection_movement']}"
assert fs18['objection_movement'] == 70.0, f"FAIL: T18 obj_movement={fs18['objection_movement']}"
objs18 = snaps[17].objections
assert len(objs18) == 1
assert objs18[0].canonical_category == "general_hesitation"
assert objs18[0].lifecycle_state == "partially_resolved"

# Point 5: Momentum Scoring at Turn 18 (and no lag at Turns 7, 9, 18)
print(f"\nPoint 5 & Audited Issue (Momentum Scoring & No-Lag Verification):")
fs6 = snaps[5].momentum.family_scores
fs7 = snaps[6].momentum.family_scores
fs8 = snaps[7].momentum.family_scores
fs9 = snaps[8].momentum.family_scores
fs10 = snaps[9].momentum.family_scores

print(f"  Turn 6:  CE={snaps[5].conversion_event.status.value if snaps[5].conversion_event else None} | FOB={fs6['future_operational_behavior']} | Commit={fs6['commitment_behavior']} | Mom={snaps[5].momentum.momentum_score}")
print(f"  Turn 7:  CE={snaps[6].conversion_event.status.value if snaps[6].conversion_event else None} | FOB={fs7['future_operational_behavior']} | Commit={fs7['commitment_behavior']} | Mom={snaps[6].momentum.momentum_score}")
print(f"  Turn 8:  CE={snaps[7].conversion_event.status.value if snaps[7].conversion_event else None} | FOB={fs8['future_operational_behavior']} | Commit={fs8['commitment_behavior']} | Mom={snaps[7].momentum.momentum_score}")
print(f"  Turn 9:  CE={snaps[8].conversion_event.status.value if snaps[8].conversion_event else None} | FOB={fs9['future_operational_behavior']} | Commit={fs9['commitment_behavior']} | Mom={snaps[8].momentum.momentum_score}")
print(f"  Turn 10: CE={snaps[9].conversion_event.status.value if snaps[9].conversion_event else None} | FOB={fs10['future_operational_behavior']} | Commit={fs10['commitment_behavior']} | Mom={snaps[9].momentum.momentum_score}")
print(f"  Turn 18: CE={snaps[17].conversion_event.status.value if snaps[17].conversion_event else None} | FOB={fs18['future_operational_behavior']} | Commit={fs18['commitment_behavior']} | Mom={snaps[17].momentum.momentum_score} | Trend={snaps[17].momentum.trend} (delta={snaps[17].momentum.trend_delta:+4.1f})")

# Assert no lag on Turn 7 upgrade:
assert fs7['future_operational_behavior'] >= 80.0, f"FAIL: T7 FOB={fs7['future_operational_behavior']}"
assert fs7['commitment_behavior'] >= 80.0, f"FAIL: T7 Commit={fs7['commitment_behavior']}"

# Assert no lag on Turn 9 downgrade:
assert fs9['future_operational_behavior'] == 45.0, f"FAIL: T9 FOB={fs9['future_operational_behavior']}"
assert fs9['commitment_behavior'] <= 65.0, f"FAIL: T9 Commit={fs9['commitment_behavior']}"

# Assert Turn 18 scores:
assert fs18['decision_structure_clarity'] == 95.0
assert fs18['future_operational_behavior'] >= 80.0
assert fs18['commitment_behavior'] >= 80.0
assert snaps[17].momentum.momentum_score >= 68.0
assert snaps[17].momentum.trend == "advancing"

# Point 6: Conversion Presentation & Header
pres = build_conversion_presentation(snaps[17])
print(f"\nPoint 6 (Presentation & Milestone Status):")
print(f"  deal_milestone_status: {pres['deal_milestone_status']}")
print(f"  milestone_label:       {pres['milestone_label']}")
print(f"  gate_status:           {snaps[17].conversion_gate.status}")
print(f"  open_concerns_summary: {pres['open_concerns_summary']}")
assert pres['deal_milestone_status'] == "appointment_confirmed"
assert "APPOINTMENT CONFIRMED" in pres['milestone_label']
assert snaps[17].conversion_gate.is_open is True

print("\n" + "=" * 60)
print("ALL 6 POINTS VERIFIED AND PASSING WITH ZERO REGRESSIONS!")
print("=" * 60)
