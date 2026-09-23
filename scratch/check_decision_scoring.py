import sys
sys.path.insert(0, '.')
import json
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle

with open('reports/synthetic/conversation_state_sim_mucj0p5s.json') as f:
    data = json.load(f)

mgr_test = ConversationStateManager(call_sid='test')
b_t2 = BehavioralSignalInputBundle(**data['timeline'][1]['evidence_bundle'])
mat_t2 = data['timeline'][1]['materiality']
res_t2 = mgr_test._extract_autonomous_decision_updates(b_t2, mat_t2)
print("Turn 2 autonomous extract:", res_t2)

mgr = ConversationStateManager(call_sid='test_call')
for t in data['timeline']:
    eb = BehavioralSignalInputBundle(**t['evidence_bundle'])
    snap = mgr.process_turn_bundle(eb)
    if t['turn_id'] == 2:
        print("T2 snap.decision_structure:", snap.decision_structure)
        print("T2 snap.change_history:")
        for c in snap.change_history:
            print("  -", c.field_path, c.reason)
    dec = snap.decision_structure
    tid = t['turn_id']
    st_list = [{s.role: s.presence} for s in dec.stakeholders]
    mom = snap.momentum
    fs = mom.family_scores
    print(f"T{tid:02d} | Stage: {snap.conversation_stage.value if snap.conversation_stage else 'None':20s} | dec_clarity: {fs.get('decision_structure_clarity'):4.1f} | future_op: {fs.get('future_operational_behavior'):4.1f} | commit: {fs.get('commitment_behavior'):4.1f} | mom: {mom.momentum_score:4.1f} ({mom.trend}) | dm_present: {dec.decision_maker_present} | primary: {dec.primary_decision_maker} | st: {st_list}")
