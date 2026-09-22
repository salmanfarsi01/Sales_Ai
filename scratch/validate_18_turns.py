import json
from pathlib import Path
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

def main():
    engine = ConversationReplayEngine()
    call_sid = "sim_validation_test"
    report = engine.replay_dialogue_turns(
        call_sid=call_sid,
        raw_turns=turns,
        save_report=True,
        conversion_target="appointment",
    )
    
    final = report.final_state
    print("=== FINAL CONVERSATION STATE RESULTS ===")
    print("Decision Structure:")
    print("  decision_maker_present:", final.decision_structure.decision_maker_present)
    print("  stakeholders:")
    for s in final.decision_structure.stakeholders:
        print(f"    - role: {s.role}, presence: {s.presence}, confidence: {s.confidence}, notes: {s.notes}")
    print("  access_constraints:", final.decision_structure.access_constraints)
    
    print("\nConversion Gate:")
    gate = final.conversion_gate
    print("  is_open:", gate.is_open)
    print("  status:", gate.status)
    print("  failed_conditions:", gate.failed_conditions)
    print("  blocking_reasons:", gate.blocking_reasons)
    for c in gate.conditions:
        print(f"    - {c.condition_name}: met={c.met}, overridden={c.is_overridden}, reason='{c.reason}'")
        
    print("\nPush Strength:")
    print("  state:", final.push_strength.state)
    print("  rationale:", final.push_strength.rationale)
    
    print("\nConversion Event:")
    print("  status:", final.conversion_event.status if final.conversion_event else None)
    print("  start_at:", final.conversion_event.start_at if final.conversion_event else None)

    # File size check
    report_path = Path("reports/synthetic") / f"conversation_state_{call_sid}.json"
    if report_path.exists():
        size_kb = report_path.stat().st_size / 1024
        print(f"\nReport JSON file size: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
