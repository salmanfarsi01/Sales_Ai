import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_replay import ConversationReplayEngine

old_report = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))

print("--- METHOD A: EXACT BUNDLES THROUGH ConversationStateManager ---")
mgr = ConversationStateManager(call_sid='check_exact_commit_58ceb7e')
for step in old_report['timeline']:
    bundle = BehavioralSignalInputBundle.model_validate(step['evidence_bundle'])
    s = mgr.process_turn_bundle(bundle)
    if bundle.turn_id in (17, 18):
        print(f"Turn {bundle.turn_id}: mom={s.momentum.momentum_score}, read={s.readiness.readiness_score}, obj_mov={s.momentum.family_scores.get('objection_movement')}, trust_eng={s.momentum.family_scores.get('trust_engagement_trend')}")
        print(f"   family_scores: {s.momentum.family_scores}")

print("\n--- METHOD B: RAW TURNS THROUGH ConversationReplayEngine ---")
engine = ConversationReplayEngine()
raw_turns = [{'turn_id': t['turn_id'], 'speaker_id': t['speaker_id'], 'text': t['text']} for t in old_report['timeline']]
rep = engine.replay_dialogue_turns(call_sid='check_raw_commit_58ceb7e', raw_turns=raw_turns, save_report=False)
for step in rep.timeline:
    if step.turn_id in (17, 18):
        m = step.state_after.momentum
        r = step.state_after.readiness
        print(f"Turn {step.turn_id}: mom={m.momentum_score}, read={r.readiness_score}, obj_mov={m.family_scores.get('objection_movement')}, trust_eng={m.family_scores.get('trust_engagement_trend')}")
        print(f"   family_scores: {m.family_scores}")
