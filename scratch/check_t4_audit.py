import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle

d = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json'))
mgr = ConversationStateManager(call_sid='CA_check_t4')
turns = {}
for t in d['timeline'][:4]:
    b = BehavioralSignalInputBundle(**t['evidence_bundle'])
    turns[b.turn_id] = b.utterance_text
    snap = mgr.process_turn_bundle(b)

gate4 = snap.conversion_gate
print('=== T04 CONVERSION GATE AUDIT ===')
print('Gate is_open:', gate4.is_open)
print('Gate status:', gate4.status)
print('Push recommendation:', snap.push_strength.state)
print('Failed conditions:', gate4.failed_conditions)
print('Unknown conditions:', getattr(gate4, 'unknown_conditions', []))
print('Readiness score:', snap.readiness.readiness_score if snap.readiness else None)
print('Readiness breakdown:', {
    'emotional': snap.readiness.emotional_readiness if snap.readiness else None,
    'logical': snap.readiness.logical_readiness if snap.readiness else None,
    'logistical': snap.readiness.logistical_readiness if snap.readiness else None,
    'decision': snap.readiness.decision_readiness if snap.readiness else None,
})
print('\n--- Conditions Breakdown ---')
for c in gate4.conditions:
    ev_texts = [f"T{tid}: '{turns.get(tid, 'unknown')}'" for tid in c.evidence_turn_ids]
    print(f"Condition: {c.condition_name}")
    print(f"  Met: {c.met} (Status: {c.status})")
    print(f"  Evidence Turn IDs: {c.evidence_turn_ids}")
    print(f"  Evidence Text: {ev_texts}")
    print(f"  Reason: {c.reason}\n")
