import sys
import json
from pathlib import Path

sys.path.insert(0, ".")
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
report = engine.replay_dialogue_turns(
    call_sid="sim_18_turn_detailed_audit",
    raw_turns=turns,
    save_report=True,
    conversion_target="appointment",
)

print("\n=== 18-TURN AUDIT TABLE WITH NEW COLUMNS ===")
header = "| Turn | Speaker | Text Snippet | Primary Action | Posture | Push | Secondary Action | Secondary Reason | Conf | Stack | Downgrade | Carried | Prompt Skipped |"
sep = "|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|"
print(header)
print(sep)

for step in report.timeline:
    dec = step.strategic_decision
    t_id = step.turn_id
    spk = "Sales" if step.speaker_id == "salesperson" else "Client"
    txt = step.text[:35] + "..." if len(step.text) > 35 else step.text
    txt = txt.replace('"', '\\"')
    p_act = dec.primary_action.value if dec and dec.primary_action else "none"
    post = dec.strategic_posture if dec else "explore"
    push = str(dec.push_strength) if dec else "low"
    s_act = dec.secondary_action.value if dec and dec.secondary_action else "None"
    s_reas = dec.secondary_action_reason if dec and dec.secondary_action_reason else "None"
    conf = f"{dec.confidence:.3f}" if dec and dec.confidence is not None else "0.500"
    stack = 2 if s_act != "None" else 1
    downgrade = "YES" if ("LOW_CONFIDENCE_ACTION_DOWNGRADE" in (dec.reason_codes or [])) else "NO"
    carried = "YES" if ("STRATEGY_CARRIED_FORWARD" in (dec.reason_codes or [])) else "NO"
    prompt_skip = "YES" if (dec and not dec.should_prompt) else "NO"

    print(f"| {t_id} | {spk} | \"{txt}\" | `{p_act}` | `{post}` | `{push}` | `{s_act}` | {s_reas} | {conf} | {stack} | {downgrade} | {carried} | {prompt_skip} |")

print("\n=== TURN DETAILS: REASON CODES & EVIDENCE CONSIDERED ===")
for step in report.timeline:
    dec = step.strategic_decision
    t_id = step.turn_id
    print(f"\n--- Turn {t_id} ({step.speaker_id}): \"{step.text}\" ---")
    print(f"  Primary: {dec.primary_action.value if dec else None} | Posture: {dec.strategic_posture} | Push: {dec.push_strength} | Conf: {dec.confidence}")
    print(f"  Secondary: {dec.secondary_action.value if dec and dec.secondary_action else None} | Reason: {dec.secondary_action_reason}")
    print(f"  Reason Codes: {dec.reason_codes}")
    print(f"  Evidence Considered:")
    for ev in dec.evidence_considered:
        print(f"    - {ev}")
