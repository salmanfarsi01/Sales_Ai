import json
from copilot.conversation_replay import ConversationReplayEngine

raw_turns = [
    {"turn_id": 1, "speaker_id": "salesperson", "text": "Would Thursday work for a quick call to go over some options?"},
    {"turn_id": 2, "speaker_id": "client", "text": "Please don't start texting me every day."},
    {"turn_id": 3, "speaker_id": "salesperson", "text": "Understood, I'll keep it light."},
    {"turn_id": 4, "speaker_id": "client", "text": "Actually, don't call me again."}
]

engine = ConversationReplayEngine()
report = engine.replay_dialogue_turns(
    call_sid="sim_mukmxsle",
    raw_turns=raw_turns,
    save_report=True,
)

print(f"=== SIMULATION REPORT FOR {report.call_sid} ===")
for step in report.timeline:
    t = step.turn_id
    sp = step.speaker_id
    txt = step.text
    gate = step.state_after.conversion_gate
    gate_open = gate.is_open if gate else False
    unknown_conds = getattr(gate, "unknown_conditions", []) if gate else []
    readiness = step.state_after.readiness.readiness_score if step.state_after.readiness else None
    readiness_str = f"{readiness:.1f}" if readiness is not None else "UNKNOWN (Insufficient Evidence)"
    strat = step.strategic_decision
    action = strat.primary_action if strat else "NONE"
    prompt = strat.final_prompt_text if strat else "NONE"
    comp = step.state_after.contact_compliance
    bnd_active = comp.hard_boundary_active if comp else False
    bnd_suspect = comp.boundary_suspected if comp else False
    prefs = [p.model_dump() for p in getattr(comp, "contact_preferences", [])] if comp else []
    events = [e.event_type for e in getattr(step.state_after, "compliance_events", [])]
    print(f"Turn {t} [{sp}]: '{txt}'")
    print(f"  Gate Open: {gate_open} | Unknown: {unknown_conds} | Readiness: {readiness_str}")
    print(f"  Boundary: Active={bnd_active}, Suspected={bnd_suspect} | Events: {events}")
    print(f"  Preferences: {prefs}")
    print(f"  Strategy: Action={action}, Push={strat.push_strength if strat else 'NONE'}")
    print(f"  DoNotDo: {strat.do_not_do if strat else []}")
    print(f"  Prompt: '{prompt}'")
    print("-" * 60)
