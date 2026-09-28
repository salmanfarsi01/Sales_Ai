import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle

d = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json'))
mgr = ConversationStateManager(call_sid='sim_mucj0p5s_corrected')

print(f"{'Turn':<5} | {'Speaker':<11} | {'Old Gate':<9} | {'New Gate':<9} | {'Old Push':<20} | {'New Push':<20} | {'Old Read':<8} | {'New Read':<8} | {'Old Status':<10} | {'New Status':<10}")
print("-" * 125)

for i, t in enumerate(d['timeline']):
    tid = t['evidence_bundle']['turn_id']
    spk = t['evidence_bundle']['speaker_id']
    old_state = t['state_after']
    old_gate = old_state['conversion_gate']['is_open'] if old_state.get('conversion_gate') else False
    old_push = old_state['push_strength']['state'] if old_state.get('push_strength') else 'None'
    old_read = old_state['readiness']['readiness_score'] if old_state.get('readiness') else 'None'
    old_ev = old_state['conversion_event']['status'] if old_state.get('conversion_event') else 'None'
    
    b = BehavioralSignalInputBundle(**t['evidence_bundle'])
    snap = mgr.process_turn_bundle(b)
    
    new_gate = snap.conversion_gate.is_open if snap.conversion_gate else False
    new_push = snap.push_strength.state if snap.push_strength else 'None'
    new_read = snap.readiness.readiness_score if snap.readiness else 'None'
    new_ev = snap.conversion_event.status if snap.conversion_event else 'None'
    
    print(f"T{tid:02d}  | {spk:<11} | {str(old_gate):<9} | {str(new_gate):<9} | {old_push:<20} | {new_push:<20} | {str(old_read):<8} | {str(new_read):<8} | {str(old_ev):<10} | {str(new_ev):<10}")
