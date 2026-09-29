import sys
import json
from pathlib import Path

sys.path.insert(0, ".")
from copilot.conversation_replay import ConversationReplayEngine

# 18-turn benchmark (Test A)
turns_A = [
    {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, thanks for making time today — tell me a bit about what's going on with the house."},
    {"turn_id": 2, "speaker_id": "client", "text": "I'm the one making this decision, no one else needs to sign off."},
    {"turn_id": 3, "speaker_id": "salesperson", "text": "Got it. What's driving the timing for you?"},
    {"turn_id": 4, "speaker_id": "client", "text": "We might look at moving sometime next year, nothing urgent yet."},
    {"turn_id": 5, "speaker_id": "client", "text": "Honestly, we're not sure this is the right time anymore."},
    {"turn_id": 6, "speaker_id": "salesperson", "text": "That's fair — a lot of people feel that way before they see the actual numbers. Would it help to walk through what the market looks like right now?"},
    {"turn_id": 7, "speaker_id": "client", "text": "Okay, that makes sense, I guess timing isn't the biggest issue."},
    {"turn_id": 8, "speaker_id": "salesperson", "text": "Great — would sometime next week work for a walkthrough?"},
    {"turn_id": 9, "speaker_id": "client", "text": "Maybe next week could work, let me think about it."},
    {"turn_id": 10, "speaker_id": "client", "text": "Actually, my wife would really need to be part of this conversation before we go any further."},
    {"turn_id": 11, "speaker_id": "salesperson", "text": "Of course, happy to loop her in whenever works."},
    {"turn_id": 12, "speaker_id": "client", "text": "I guess I'm just worried this isn't really the right move for us financially with everything going on."},
    {"turn_id": 13, "speaker_id": "salesperson", "text": "Totally understand — let's look at your net proceeds after all costs, so you can see the real picture."},
    {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more — how does the marketing process work, what about staging, how long does listing usually take?"},
    {"turn_id": 15, "speaker_id": "client", "text": "Please don't start texting me every day before we meet, by the way."},
    {"turn_id": 16, "speaker_id": "client", "text": "Mornings don't really work for us either, just so you know."},
    {"turn_id": 17, "speaker_id": "salesperson", "text": "Noted on all of that. What day works best?"},
    {"turn_id": 18, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
]

# Fresh independent call (Test B)
turns_B = [
    {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, this is Alex with Premier Realty. How are you doing today?"},
    {"turn_id": 2, "speaker_id": "client", "text": "I'm doing well, but I'm just curious about current home values in our neighborhood."},
    {"turn_id": 3, "speaker_id": "salesperson", "text": "I'd be glad to share recent neighborhood sales. Have you considered selling anytime soon?"},
    {"turn_id": 4, "speaker_id": "client", "text": "Maybe in a couple of years, no immediate plans."}
]

engine = ConversationReplayEngine()

print("================================================================================")
print("RUNNING TEST A: Full 18-Turn Call (sim_test_call_A)")
print("================================================================================")
rep_A = engine.replay_dialogue_turns(
    call_sid="sim_test_call_A",
    raw_turns=turns_A,
    save_report=True,
    conversion_target="appointment",
)

step18_A = rep_A.timeline[17]
sd18_A = step18_A.strategic_decision
sa18_A = step18_A.state_after

print(f"Test A Final Turn 18:")
print(f"  Utterance: \"{step18_A.text}\"")
print(f"  State Version: V{sa18_A.state_version}")
print(f"  Gate Status: {sa18_A.conversion_gate.status} (is_open={sa18_A.conversion_gate.is_open})")
print(f"  Commitment Score: {sa18_A.dimensions.commitment:.2f}")
print(f"  Stage: {sa18_A.conversation_stage}")
print(f"  Facts ({len(sa18_A.facts)}): {[f.fact_value for f in sa18_A.facts]}")
print(f"  Objections ({len(sa18_A.objections)}): {[o.canonical_category for o in sa18_A.objections]}")
print(f"  Contact Preferences ({len(sa18_A.contact_compliance.contact_preferences)}): {[p.prohibited_behavior for p in sa18_A.contact_compliance.contact_preferences]}")
print(f"  Strategic Decision Action: {sd18_A.primary_action if sd18_A else 'None'}")
print(f"  Evidence Considered (Turn 18):")
if sd18_A:
    for ev in sd18_A.evidence_considered:
        print(f"    - {ev}")

print("\n================================================================================")
print("RUNNING TEST B: Back-to-Back Independent Call (sim_test_call_B)")
print("================================================================================")
rep_B = engine.replay_dialogue_turns(
    call_sid="sim_test_call_B",
    raw_turns=turns_B,
    save_report=True,
    conversion_target="appointment",
)

print("\nINSPECTING ALL TURNS OF TEST B FOR CROSS-CALL CONTAMINATION:\n")
for step in rep_B.timeline:
    sa = step.state_after
    sd = step.strategic_decision
    print(f"--- Test B Turn {step.turn_id} ({step.speaker_id.upper()}): \"{step.text}\" ---")
    print(f"  State Version: V{sa.state_version}")
    print(f"  Stage: {sa.conversation_stage}")
    gate_st = sa.conversion_gate.status if sa.conversion_gate else "closed"
    gate_op = sa.conversion_gate.is_open if sa.conversion_gate else False
    print(f"  Gate: status='{gate_st}', is_open={gate_op}")
    print(f"  Commitment: {sa.dimensions.commitment:.2f}")
    print(f"  Facts Count: {len(sa.facts)} -> {[f.fact_value for f in sa.facts]}")
    print(f"  Objections Count: {len(sa.objections)} -> {[o.canonical_category for o in sa.objections]}")
    print(f"  Contact Preferences: {len(sa.contact_compliance.contact_preferences)}")
    if sd:
        print(f"  Decision Action: {sd.primary_action}")
        print(f"  Decision Version: V{sd.source_state_version}")
        print(f"  Evidence Considered ({len(sd.evidence_considered)} items):")
        for ev in sd.evidence_considered:
            print(f"    * {ev}")
    print()

# Rigorous Assertions
final_B = rep_B.final_state
assert final_B.call_sid == "sim_test_call_B"

# 1. Assert zero facts from Test A
for f in final_B.facts:
    assert "Thursday" not in f.fact_value, f"Leaked meeting time in facts: {f.fact_value}"
    assert "wife" not in f.fact_value.lower(), f"Leaked spouse in facts: {f.fact_value}"
    assert "texting" not in f.fact_value.lower(), f"Leaked texting preference in facts: {f.fact_value}"

# 2. Assert zero objections leaked from Test A (all objections in Test B must be originating in Test B's turns)
test_a_obj_statements = {o.initial_statement for o in sa18_A.objections}
for o in final_B.objections:
    assert o.initial_statement not in test_a_obj_statements, f"Leaked objection statement from Test A: {o.initial_statement}"
    assert "financially" not in o.initial_statement.lower()
    assert "sure this is the right time" not in o.initial_statement.lower()

# 3. Assert zero contact preferences from Test A
assert len(final_B.contact_compliance.contact_preferences) == 0, f"Leaked contact preferences: {final_B.contact_compliance.contact_preferences}"
assert final_B.contact_compliance.contact_preference == "none"

# 4. Assert gate is closed and commitment is 0.0
assert final_B.conversion_gate.status == "closed"
assert final_B.conversion_gate.is_open is False
assert final_B.dimensions.commitment == 0.0

# 5. Assert zero evidence from Test A in ANY turn of Test B
for step in rep_B.timeline:
    sd = step.strategic_decision
    if sd:
        for ev in sd.evidence_considered:
            assert "Thursday at 3" not in ev, f"Leaked utterance in evidence on Turn {step.turn_id}: {ev}"
            assert "wife" not in ev.lower(), f"Leaked spouse in evidence on Turn {step.turn_id}: {ev}"
            assert "texting" not in ev.lower(), f"Leaked texting boundary on Turn {step.turn_id}: {ev}"

print("================================================================================")
print("POINT 8 DEMONSTRATION VERDICT: ZERO LEAKAGE CONFIRMED ACROSS ALL DIMENSIONS!")
print("================================================================================")
