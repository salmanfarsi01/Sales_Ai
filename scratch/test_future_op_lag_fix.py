import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager

report = json.load(open('reports/synthetic/conversation_state_sim_mueyx6wy.json', encoding='utf-8'))
manager = ConversationStateManager(call_sid="sim_mueyx6wy_test")

print("=== REPLAYING WITH CURRENT CODE ===")
for step in report['timeline']:
    bundle_dict = step['evidence_bundle']
    bundle = BehavioralSignalInputBundle.model_validate(bundle_dict)
    state = manager.process_turn_bundle(bundle)
    
    t = bundle.turn_id
    if t in [6, 7, 8, 9, 10, 11, 17, 18]:
        ce_status = state.conversion_event.status if state.conversion_event else None
        fob = state.momentum.family_scores.get('future_operational_behavior')
        dsc = state.momentum.family_scores.get('decision_structure_clarity')
        commit = state.momentum.family_scores.get('commitment_behavior')
        overall = state.momentum.momentum_score
        print(f"Turn {t:02d} [{bundle.speaker_id[:6]}]: \"{bundle.utterance_text[:40]}\"")
        print(f"   CE: {ce_status} | FOB: {fob} | DSC: {dsc} | Commit: {commit} | Overall: {overall}")
