import json, sys
sys.path.insert(0, ".")
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_presentation import build_conversion_presentation

with open("reports/synthetic/conversation_state_sim_mucj0p5s.json", encoding="utf-8") as f:
    data = json.load(f)

mgr = ConversationStateManager(call_sid="sim_mucj0p5s_check")
for i, t in enumerate(data["timeline"]):
    eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
    snap = mgr.process_turn_bundle(eb)
    if eb.turn_id in (16, 17):
        print(f"Turn {eb.turn_id} obj: {snap.objections[0].lifecycle_state}, evidence: {snap.objections[0].dormancy_evidence}")
    else:
        print(f"Turn {eb.turn_id}: objections={[ (o.canonical_category, o.lifecycle_state) for o in snap.objections ]}")

pres = build_conversion_presentation(mgr.current_state)
print("open_concerns:", pres["open_concerns"])
