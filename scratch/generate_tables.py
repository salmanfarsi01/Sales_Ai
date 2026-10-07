import sys
sys.path.insert(0, r"c:\dev\Sales_Ai")
from copilot.conversation_replay import ConversationReplayEngine

dialogue = [
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

for mode_name, flag in [("BASELINE (SEMANTIC LAYER OFF)", False), ("FLAG-ON (LIVE SEMANTIC LAYER ON)", True)]:
    eng = ConversationReplayEngine()
    rep = eng.replay_dialogue_turns(f"table_gen_{flag}", dialogue, save_report=False, run_semantic_analysis=flag)
    print(f"=== {mode_name} ===")
    print("| Turn | Speaker | Primary Action | Secondary Action | Posture | Push | Conf | Stack | Downgrade | Carried | Spoken Prompt / Guidance |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s in rep.timeline:
        tid = s.turn_id
        spk = s.speaker_id
        dec = s.strategic_decision
        act = dec.primary_action.value
        sec = dec.secondary_action.value if dec.secondary_action else "None"
        post = dec.strategic_posture
        push = str(dec.push_strength)
        conf_val = getattr(dec, "confidence", 0.0) or 0.0
        conf = f"{conf_val:.3f}"
        stack = "2" if dec.secondary_action else "1"
        downgrade = "YES" if "LOW_CONFIDENCE_ACTION_DOWNGRADE" in dec.reason_codes else "NO"
        carried = "YES" if ("STRATEGY_CARRIED_FORWARD" in dec.reason_codes or dec.carried_forward_from_turn_id is not None) else "NO"
        prompt = dec.final_prompt_text if dec.should_prompt else "[SUPPRESSED / NO PROMPT]"
        prompt_clean = prompt.replace("\n", " ").strip() if prompt else "None"
        print(f"| {tid} | {spk} | {act} | {sec} | {post} | {push} | {conf} | {stack} | {downgrade} | {carried} | {prompt_clean} |")
    print()
