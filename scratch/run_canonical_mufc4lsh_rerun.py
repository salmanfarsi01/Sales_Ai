import json
import sys
from pathlib import Path
sys.path.insert(0, '.')
from copilot.conversation_replay import ConversationReplayEngine

canonical_turns = [
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
    call_sid = "sim_mufc4lsh"
    report = engine.replay_dialogue_turns(
        call_sid=call_sid,
        raw_turns=canonical_turns,
        save_report=True,
        conversion_target="appointment",
    )

    print(f"=== CANONICAL BENCHMARK REPLAY ({call_sid}) ===")
    print(f"Total Steps: {len(report.timeline)}")
    print(f"Final State Version: V{report.final_state.state_version}")

    rows = []
    for step in report.timeline:
        sa = step.state_after
        sd = step.strategic_decision
        gate = sa.conversion_gate if sa else None
        
        row = {
            "turn": step.turn_id,
            "speaker": step.speaker_id.upper(),
            "utterance": step.text,
            "version": f"V{sa.state_version}" if sa else "—",
            "call_sid": sd.call_sid if sd else "—",
            "decision_id": sd.decision_id if sd else "—",
            "source_event": sd.source_event_id if sd else f"ev_t{step.turn_id}",
            "action": str(sd.primary_action).replace("StrategicAction.", "") if sd else "PENDING",
            "push": str(sd.push_strength) if sd else "—",
            "gate": "OPEN" if (gate and gate.is_open) else "CLOSED",
            "commitment": f"{sa.dimensions.commitment:.2f}" if sa else "—",
            "slot": gate.commitment_slot if (gate and gate.commitment_slot) else "None",
            "prompt": (sd.gateway_fallback_stub or sd.final_prompt_text or "") if sd else "",
        }
        rows.append(row)

    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
