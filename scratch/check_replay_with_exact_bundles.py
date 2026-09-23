import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_contract import extract_behavioral_bundle, BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager

old_report = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))

manager = ConversationStateManager(call_sid="sim_mucj0p5s_bundle_check")

print("=== REPLAYING WITH EXACT BUNDLES FROM OLD REPORT ===")
for step in old_report['timeline']:
    bundle_dict = step['evidence_bundle']
    # Build bundle
    bundle = BehavioralSignalInputBundle.model_validate(bundle_dict)
    state = manager.process_turn_bundle(bundle)
    
    if step['turn_id'] >= 14:
        old_mom = step.get('state_after', {}).get('momentum', {})
        new_mom = state.momentum
        old_obj_mov = old_mom.get('family_scores', {}).get('objection_movement')
        new_obj_mov = new_mom.family_scores.get('objection_movement') if new_mom else None
        
        old_score = old_mom.get('momentum_score')
        new_score = new_mom.momentum_score if new_mom else None
        
        print(f"Turn {step['turn_id']}:")
        print(f"  objection_movement: old={old_obj_mov} -> new={new_obj_mov}")
        print(f"  momentum_score:     old={old_score} -> new={new_score}")
        all_keys = sorted(list(new_mom.family_scores.keys()))
        for k in all_keys:
            ov = old_mom.get('family_scores', {}).get(k)
            nv = new_mom.family_scores.get(k)
            if ov != nv:
                print(f"    diff in {k}: old={ov} -> new={nv}")
