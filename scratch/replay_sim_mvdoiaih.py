import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from copilot.conversation_replay import ConversationReplayEngine
from copilot.core_decision_manager import StrategicDecisionCache
from copilot.conversation_state_models import ConversationStateSnapshot
from copilot.core_intelligence_models import StrategicDecision

old_report_path = Path("reports/synthetic/conversation_state_sim_mvdoiaih.json")
old_data = json.loads(old_report_path.read_text(encoding="utf-8"))

raw_turns = []
for s in old_data["timeline"]:
    raw_turns.append({
        "turn_id": s["turn_id"],
        "speaker_id": s["speaker_id"],
        "text": s["text"],
    })

print(f"Replaying {len(raw_turns)} turns...")
engine = ConversationReplayEngine()
report = engine.replay_dialogue_turns(
    call_sid="sim_mvdoiaih",
    raw_turns=raw_turns,
    save_report=True,
)

print(f"Replay complete! Total timeline steps: {len(report.timeline)}")

# Audit Turn 2, Turn 13, and Final State
for step in report.timeline:
    t = step.turn_id
    spk = step.speaker_id
    txt = step.text
    sa = step.state_after
    ds = sa.decision_structure
    ce = sa.conversion_event
    
    if t in (2, 8, 11, 13, 14):
        print("="*80)
        print(f"Turn {t} [{spk}]: '{txt}'")
        print(f"  Decision Structure:")
        print(f"    - decision_maker_present: {ds.decision_maker_present}")
        print(f"    - co_decision_required: {ds.co_decision_required}")
        print(f"    - primary_decision_maker: {ds.primary_decision_maker}")
        print(f"    - stakeholders: {[s.model_dump() for s in ds.stakeholders]}")
        if ce:
            print(f"  Conversion Event:")
            print(f"    - status: {ce.status}")
            print(f"    - start_at: {ce.start_at}")
            print(f"    - participants: {ce.participants}")
        else:
            print("  Conversion Event: None")
        
        # Check strategic decision evidence consistency
        if step.strategic_decision:
            ok, reason = StrategicDecisionCache.validate_evidence_consistency(step.strategic_decision, sa, require_active=True)
            print(f"  Strategic Decision Evidence Consistency: {'PASS' if ok else 'FAIL (' + reason + ')'}")

print("="*80)
print("Checking All Turns 4-Key Binding and Evidence Consistency:")
all_ok = True
for step in report.timeline:
    if step.strategic_decision:
        ok, r = StrategicDecisionCache.validate_evidence_consistency(step.strategic_decision, step.state_after, require_active=True)
        if not ok:
            print(f"Turn {step.turn_id} consistency FAIL: {r}")
            all_ok = False
        b_ok, b_r = StrategicDecisionCache.validate_binding(
            decision=step.strategic_decision,
            call_sid="sim_mvdoiaih",
            turn_id=step.strategic_decision.source_turn_id,
            source_state_version=step.strategic_decision.source_state_version,
            source_event_id=step.strategic_decision.source_event_id,
            state_call_sid=step.state_after.call_sid,
            snapshot=step.state_after,
            require_active=True,
        )
        if not b_ok:
            print(f"Turn {step.turn_id} binding FAIL: {b_r}")
            all_ok = False

if all_ok:
    print("ALL TURNS PASSED EVIDENCE CONSISTENCY AND 4-KEY BINDING!")
