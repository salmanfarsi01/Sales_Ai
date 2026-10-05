import sys
from pathlib import Path

sys.path.insert(0, ".")
from copilot.conversation_replay import ConversationReplayEngine

# We take Turn 13, Turn 14, and add Turn 14b ("okay") as a client filler turn
# after salesperson or client educational walkthrough where no question is pending.
turns = [
    {"turn_id": 13, "speaker_id": "salesperson", "text": "Totally understand — let's look at your net proceeds after all costs, so you can see the real picture."},
    {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more — how does the marketing process work, what about staging, how long does listing usually take?"},
    {"turn_id": 15, "speaker_id": "salesperson", "text": "Our marketing includes professional staging and photography to maximize list price within 3 weeks."},
    {"turn_id": 16, "speaker_id": "client", "text": "Okay."},  # Pure filler acknowledgment!
]

engine = ConversationReplayEngine()
report = engine.replay_dialogue_turns(
    call_sid="sim_filler_demonstration",
    raw_turns=turns,
    save_report=False,
    conversion_target="appointment",
)

print("\n=== FILLER CARRY-FORWARD DEMONSTRATION ===")
for step in report.timeline:
    dec = step.strategic_decision
    t_id = step.turn_id
    spk = step.speaker_id
    txt = step.text
    p_act = dec.primary_action.value if dec and dec.primary_action else "none"
    carried = "YES" if ("STRATEGY_CARRIED_FORWARD" in (dec.reason_codes or [])) else "NO"
    prompt_skip = "YES" if (dec and not dec.should_prompt) else "NO"
    print(f"Turn {t_id} ({spk}): \"{txt}\"")
    print(f"  Primary: {p_act} | Carried: {carried} | Prompt Skipped: {prompt_skip}")
    print(f"  Reason Codes: {dec.reason_codes}")
    print(f"  Provenance Decision ID: {dec.carried_forward_from_decision_id}")
