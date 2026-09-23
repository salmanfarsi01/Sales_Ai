import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_contract import extract_behavioral_bundle, BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager

old_report = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))

manager = ConversationStateManager(call_sid="sim_mucj0p5s_stage_check")

print("=== REPLAYING WITH EXACT BUNDLES TO CHECK CONVERSATION_STAGE ===")
for step in old_report['timeline']:
    bundle_dict = step['evidence_bundle']
    bundle = BehavioralSignalInputBundle.model_validate(bundle_dict)
    state = manager.process_turn_bundle(bundle)
    stage_changes = [c for c in state.change_history if c.field_path == "conversation_stage" and c.triggering_turn_id == bundle.turn_id]
    stage_info = f" -> CHANGED: {stage_changes[0].reason}" if stage_changes else ""
    print(f"Turn {bundle.turn_id:02d} ({bundle.speaker_id[:6]}): stage={state.conversation_stage.value}{stage_info}")

print("\n=== FINAL STAGE HISTORY ===")
for rec in manager.current_state.stage_history:
    print(f"  [{rec.stage.value}] turns {rec.entered_turn_id}..{rec.exited_turn_id} : {rec.trigger_reason}")

