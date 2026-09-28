import json, sys
sys.path.insert(0, ".")
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager

with open("reports/synthetic/conversation_state_sim_mucj0p5s.json", encoding="utf-8") as f:
    data = json.load(f)

mgr = ConversationStateManager(call_sid="sim_mucj0p5s_audit")
snaps = {}
bundles = {}

for t in data["timeline"]:
    eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
    snap = mgr.process_turn_bundle(eb)
    snaps[eb.turn_id] = snap.model_copy(deep=True)
    bundles[eb.turn_id] = eb

for tid in [4, 9]:
    snap = snaps[tid]
    gate = snap.conversion_gate
    print(f"\n==================== TURN {tid:02d} GATE AUDIT ====================")
    print(f"Utterance ({bundles[tid].speaker_id}): \"{bundles[tid].utterance_text}\"")
    print(f"Gate Open: {gate.is_open}, Status: {gate.status}")
    print(f"Facts: {[(f.category, f.fact_key, f.source_turn_id) for f in snap.facts]}")
    print(f"Decision structure: {snap.decision_structure.primary_decision_maker}, stakeholders={[s.model_dump() for s in snap.decision_structure.stakeholders]}")
    print(f"Readiness Score: {snap.readiness.readiness_score}, Partial: {snap.readiness.readiness_partial}, Coverage: {snap.readiness.coverage}")
    for c in gate.conditions:
        ev_turns = c.evidence_turn_ids
        print(f"  - Condition: {c.condition_name}")
        print(f"    Status: {c.status} (met={c.met})")
        print(f"    Evidence turns: {ev_turns}")
        print(f"    Reason: {c.reason}")
        if ev_turns:
            for et in ev_turns:
                spk = bundles[et].speaker_id
                txt = bundles[et].utterance_text
                print(f"      Turn {et} ({spk}): \"{txt}\"")
