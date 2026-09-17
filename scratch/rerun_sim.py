import json
from copilot.conversation_replay import ConversationReplayEngine

raw_turns = [
    {
        "turn_id": 1,
        "speaker_id": "salesperson",
        "text": "Hi Daniel, I noticed your home was listed previously — how's everything going with the sale?",
    },
    {
        "turn_id": 2,
        "speaker_id": "client",
        "text": "Yeah, honestly we're just not sure this is the right time anymore.",
    },
    {
        "turn_id": 3,
        "speaker_id": "salesperson",
        "text": "That's totally understandable. Can you tell me more about what's giving you pause?",
    },
    {
        "turn_id": 4,
        "speaker_id": "client",
        "text": "Well, my wife would really need to be part of this conversation too before we go any further.",
    },
    {
        "turn_id": 5,
        "speaker_id": "salesperson",
        "text": "Of course, happy to loop her in. In the meantime, what commission structure were you expecting to pay?",
    },
    {
        "turn_id": 6,
        "speaker_id": "client",
        "text": "Honestly the commission fee last time felt way too high for what we got out of it.",
    },
    {
        "turn_id": 7,
        "speaker_id": "salesperson",
        "text": "I hear you — we can walk through exactly what that fee covers if that would help.",
    },
    {
        "turn_id": 8,
        "speaker_id": "client",
        "text": "Actually, you know what, we've decided to just stay in the house. We're not going to sell after all.",
    },
]

engine = ConversationReplayEngine()
report = engine.replay_dialogue_turns(
    call_sid="sim_mu5du135",
    raw_turns=raw_turns,
    save_report=True,
)

print(f"=== SIMULATION REPORT FOR {report.call_sid} ===")
print(f"Total Turns: {report.total_turns}")
print(f"Run Count: {report.run_count}")
print(f"Timeline steps: {len(report.timeline)}")

print("\n--- TURN BY TURN ANALYSIS ---")
for step in report.timeline:
    tid = step.turn_id
    spk = step.speaker_id
    txt = step.text
    changes = step.state_changes
    mat = step.materiality
    stk = step.state_after.decision_structure.stakeholders
    objs = step.state_after.objections
    dm_pres = step.state_after.decision_structure.decision_maker_present
    print(f"\n[Turn {tid}] ({spk}): \"{txt}\"")
    print(f"  Material: {mat.is_material}, Targets: {mat.affected_targets}")
    print(f"  Decision Maker Present: {dm_pres}, Stakeholders: {[s.role + ' (' + s.presence + ')' for s in stk]}")
    print(f"  Objections ({len(objs)}): {[(o.canonical_category, o.lifecycle_state, 'superseded_by: ' + str(o.superseded_by_objection_id)) for o in objs]}")
    print(f"  State Changes ({len(changes)}):")
    for c in changes:
        print(f"    * {c.field_path}: {c.reason}")

final = report.final_state
print("\n=== FINAL STATE SUMMARY ===")
print(f"Last Updated Turn ID: {final.last_updated_turn_id}")
print(f"Total Change History Count: {len(final.change_history)}")
print(f"Decision Structure: DM Present={final.decision_structure.decision_maker_present}, Stakeholders={[s.role + ' (' + s.presence + ')' for s in final.decision_structure.stakeholders]}")
print(f"Facts ({len(final.facts)}): {[(f.fact_key, f.fact_value) for f in final.facts]}")
print(f"Objections ({len(final.objections)}):")
for o in final.objections:
    print(f"  - Category: {o.canonical_category}, State: {o.lifecycle_state}, Superseded By: {o.superseded_by_objection_id}, Turn: {o.superseded_at_turn_id}")
print(f"Active Objections Count: {len(final.get_active_objections())}")
print(f"Superseded Objections Count: {len(final.get_superseded_objections())}")
print(f"Momentum Score: {final.momentum.momentum_score}, Trend: {final.momentum.trend}")
print(f"  Family Scores: {final.momentum.family_scores}")
print(f"Readiness Score: {final.readiness.readiness_score}, Blockers: {final.readiness.active_blocker_caps}")
print(f"Conversion Gate: Open={final.conversion_gate.is_open}, Status={final.conversion_gate.status}")
for c in final.conversion_gate.conditions:
    if not c.met:
        print(f"  Failed Condition: {c.condition_name} -> {c.reason}")
print(f"Push Strength Recommendation: {final.push_strength.state}")
print(f"  Rationale: {final.push_strength.rationale}")
print(f"  Action: {final.push_strength.recommended_action}")
