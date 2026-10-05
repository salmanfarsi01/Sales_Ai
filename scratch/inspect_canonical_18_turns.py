import sys
sys.path.insert(0, r"c:\dev\Sales_Ai")

from copilot.conversation_replay import ConversationReplayEngine
from copilot.core_decision_manager import CoreDecisionManager

turns = [
    {'turn_id': 1, 'speaker_id': 'salesperson', 'text': "Hi, thanks for making time today — tell me a bit about what's going on with the house."},
    {'turn_id': 2, 'speaker_id': 'client', 'text': "I'm the one making this decision, no one else needs to sign off."},
    {'turn_id': 3, 'speaker_id': 'salesperson', 'text': "Got it. What's driving the timing for you?"},
    {'turn_id': 4, 'speaker_id': 'client', 'text': "We might look at moving sometime next year, nothing urgent yet."},
    {'turn_id': 5, 'speaker_id': 'client', 'text': "Honestly, we're not sure this is the right time anymore."},
    {'turn_id': 6, 'speaker_id': 'salesperson', 'text': "That's fair — a lot of people feel that way before they see the actual numbers."},
    {'turn_id': 7, 'speaker_id': 'client', 'text': "Okay, that makes sense, I guess timing isn't the biggest issue."},
    {'turn_id': 8, 'speaker_id': 'salesperson', 'text': "Great — would sometime next week work for a walkthrough?"},
    {'turn_id': 9, 'speaker_id': 'client', 'text': "Maybe next week could work, let me think about it."},
    {'turn_id': 10, 'speaker_id': 'client', 'text': "Actually, my wife would really need to be part of this conversation before we go any further."},
    {'turn_id': 11, 'speaker_id': 'salesperson', 'text': "Of course, happy to loop her in whenever works."},
    {'turn_id': 12, 'speaker_id': 'client', 'text': "I guess I'm just worried this isn't really the right move for us financially with everything going on."},
    {'turn_id': 13, 'speaker_id': 'salesperson', 'text': "Totally understand — let's look at your net proceeds after all costs."},
    {'turn_id': 14, 'speaker_id': 'client', 'text': "That's actually really helpful, tell me more."},
    {'turn_id': 15, 'speaker_id': 'client', 'text': "Please don't start texting me every day before we meet, by the way."},
    {'turn_id': 16, 'speaker_id': 'client', 'text': "Mornings don't really work for us either, just so you know."},
    {'turn_id': 17, 'speaker_id': 'salesperson', 'text': "Noted on all of that. What day works best?"},
    {'turn_id': 18, 'speaker_id': 'client', 'text': "Thursday at 3 works, and my wife will be there."}
]

engine = ConversationReplayEngine()
rep = engine.replay_dialogue_turns('sim_canonical_audit', turns, save_report=False)

print("="*80)
print("CANONICAL 18-TURN REPLAY INSPECTION")
print("="*80)

for step in rep.timeline:
    sd = step.strategic_decision
    prompt_txt = sd.final_prompt_text or ""
    prompt_disp = f'"{prompt_txt[:35]}..."' if prompt_txt else "[SUPPRESSED/NO PROMPT]"
    
    print(f"\nTurn {step.turn_id:02d} ({step.speaker_id}): \"{step.text[:40]}...\"")
    print(f"  Primary: {sd.primary_action.value} | Posture: {sd.strategic_posture} | Push: {sd.push_strength} (raw={repr(sd.push_strength)})")
    print(f"  Secondary: {sd.secondary_action.value if sd.secondary_action else None} | Reason: {sd.secondary_action_reason}")
    print(f"  Confidence: {sd.confidence:.4f} | Breakdown: {sd.confidence_breakdown}")
    print(f"  Should Prompt: {sd.should_prompt} | Prompt: {prompt_disp}")
    print(f"  Reason Codes: {sd.reason_codes}")
    print(f"  Evidence: {sd.evidence_considered}")
    print(f"  Carried From Turn: {sd.carried_forward_from_turn_id} | Carried Dec: {sd.carried_forward_from_decision_id}")
    if step.state_after.contact_compliance:
        cp = step.state_after.contact_compliance.contact_preferences
        print(f"  Underlying State Contact Preferences ({len(cp)}): {[(p.channel, p.prohibited_behavior or p.time_restriction) for p in cp]}")

print("\n" + "="*80)
