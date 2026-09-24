import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_replay import ConversationReplayEngine

report_path = 'reports/synthetic/conversation_state_sim_mueyx6wy.json'
with open(report_path, encoding='utf-8') as f:
    old_data = json.load(f)

bundles = []
for step in old_data['timeline']:
    eb = BehavioralSignalInputBundle(**step['evidence_bundle'])
    bundles.append(eb)

engine = ConversationReplayEngine()
new_report = engine.replay_call(
    call_sid="sim_mueyx6wy",
    bundles=bundles,
    save_report=True,
    source="synthetic_simulation",
    conversion_target="appointment",
)

print(f"Replay completed and report saved to {report_path}")

# Now inspect turns 6, 7, 8, 9, 10, 18 from the saved file to verify:
with open(report_path, encoding='utf-8') as f:
    saved_data = json.load(f)

for step in saved_data['timeline']:
    tid = step['turn_id']
    if tid in [6, 7, 8, 9, 10, 18]:
        spk = step['speaker_id']
        txt = step['text'][:40]
        sa = step['state_after']
        ce = sa.get('conversion_event')
        ce_stat = ce.get('status') if ce else None
        mom = sa.get('momentum', {}) or {}
        fs = mom.get('family_scores', {})
        fob = fs.get('future_operational_behavior')
        dsc = fs.get('decision_structure_clarity')
        com = fs.get('commitment_behavior')
        m_score = mom.get('momentum_score')
        print(f"Turn {tid:02d} [{spk:6s}]: \"{txt}\" -> CE={ce_stat} | FOB={fob} | DSC={dsc} | Com={com} | Mom={m_score}")
