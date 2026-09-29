import json
import sys
from pathlib import Path
sys.path.insert(0, '.')
from copilot.conversation_replay import ConversationReplayEngine

turns = [
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

engine = ConversationReplayEngine()
call_sid = "sim_validation_test_rerun"
report = engine.replay_dialogue_turns(
    call_sid=call_sid,
    raw_turns=turns,
    save_report=True,
    conversion_target="appointment",
)

target_turns = [3, 4, 5, 6, 18]
print("=================== 18-TURN BENCHMARK INSPECTION ===================")
for t_id in target_turns:
    step = report.timeline[t_id - 1]
    sa = step.state_after
    sd = step.strategic_decision
    print(f"\n--- TURN {step.turn_id} ({step.speaker_id.upper()}): \"{step.text}\" ---")
    print(f"STATE AFTER:")
    print(f"  state_version: V{sa.state_version}")
    print(f"  stage: {sa.conversation_stage}")
    gate_st = sa.conversion_gate.status if sa.conversion_gate else "closed"
    gate_open = sa.conversion_gate.is_open if sa.conversion_gate else False
    print(f"  gate: status='{gate_st}', is_open={gate_open}")
    print(f"  commitment: {sa.dimensions.commitment:.2f}")
    readiness_display = f"{sa.dimensions.readiness:.2f}" if sa.dimensions.readiness is not None else "None"
    print(f"  raw readiness: {readiness_display}")
    
    print(f"STRATEGIC DECISION:")
    if sd:
        print(f"  source_state_version: V{sd.source_state_version}")
        print(f"  primary_action: {sd.primary_action}")
        print(f"  push_strength: {sd.push_strength}")
        print(f"  strategic_objective: \"{sd.strategic_objective}\"")
        prompt = sd.gateway_fallback_stub or sd.final_prompt_text or ""
        print(f"  prompt text: \"{prompt}\"")
        print(f"  evidence_considered: {sd.evidence_considered}")
    else:
        print("  NONE")
