import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from pathlib import Path
from copilot.conversation_replay import ConversationReplayEngine

script_path = Path("data/script_sim_muy04bh0_fee_complaint.json")
raw_turns = json.loads(script_path.read_text(encoding="utf-8"))

engine = ConversationReplayEngine()
rep = engine.replay_dialogue_turns(
    call_sid="sim_muy04bh0_test_run",
    raw_turns=raw_turns,
    save_report=False,
    run_semantic_analysis=False,
)

print(f"Replay total turns: {len(rep.timeline)}")
for s in rep.timeline:
    d = s.strategic_decision
    m = s.materiality
    print(f"Turn {s.turn_id} ({s.speaker_id}): '{s.text}'")
    print(f"   Materiality: is_mat={m.is_material if m else None}, targets={m.affected_targets if m else None}")
    print(f"   Action: {d.primary_action}, Posture: {d.strategic_posture}, Push: {d.push_strength}")
    print(f"   Should prompt: {d.should_prompt}, cf_turn: {d.carried_forward_from_turn_id}, cf_dec: {d.carried_forward_from_decision_id}")
    print(f"   Reason codes: {d.reason_codes}")
    print(f"   Prompt text: {d.final_prompt_text or d.gateway_fallback_stub}")
    print(f"   Ref objections: {d.referenced_objection_ids}")
    ce = s.state_after.conversion_event
    print(f"   conv_event: {ce.status if ce else None} start={ce.start_at if ce else None}")
    cev = s.state_after.conversion_events or []
    print(f"   Events: {[(e.event_id, e.status, e.start_at, e.superseded_by_event_id) for e in cev]}")
    gate = s.state_after.conversion_gate
    if gate and gate.conditions:
        c1 = gate.conditions[0]
        print(f"   Gate Cond 1: {c1.condition_name} status={c1.status} reason='{c1.reason}'")
    ps = s.state_after.push_strength
    if ps:
        print(f"   PushStrength: pressure='{ps.pressure}' posture='{ps.strategic_posture}' strategy='{ps.strategy}' state='{ps.state}'")
    facts = s.state_after.facts or []
    print(f"   Facts: {[(f.fact_id, f.fact_value, f.notes) for f in facts if 'meeting' in f.category.lower() or 'meeting' in f.fact_value.lower() or 'thursday' in f.fact_value.lower()]}")
    objs = s.state_after.objections or []
    print(f"   Objections: {[(o.canonical_category, o.lifecycle_state, o.recurrence_count, o.attempted_strategies, [out.effectiveness for out in o.strategy_outcomes], getattr(o, 'deferred_to_meeting', False), getattr(o, 'retained_for_followup', False)) for o in objs]}")
    print()
