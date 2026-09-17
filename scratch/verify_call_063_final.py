import sys
sys.path.insert(0, ".")
import dotenv
dotenv.load_dotenv()
from pathlib import Path
from copilot.conversation_replay import ConversationReplayEngine

engine = ConversationReplayEngine(reports_dir=Path('reports'))
rep = engine.replay_behavioral_signal_report(Path('reports/behavioral_signal_test_call_063.json'), save_report=True)

print(f"=== REPLAY REPORT FOR {rep.call_sid} ===")
print(f"Total turns: {len(rep.timeline)}")
print(f"Run count: {rep.run_count}")
print(f"Report file: {rep.report_file}")

for t in rep.timeline:
    print(f"\n--- TURN {t.turn_id} ({t.speaker_id.upper()}) ---")
    print(f"Text: '{t.text}'")
    print(f"Materiality is_material: {t.materiality.is_material}")
    print(f"Materiality targets: {sorted(list(t.materiality.affected_targets))}")
    print(f"Materiality reasoning: {t.materiality.reasoning}")
    print(f"State Before: V{t.state_before.state_version}")
    print(f"State Changes: {len(t.state_changes)}")
    for c in t.state_changes:
        print(f"  * [{c.field_path}] V{c.state_version_before} -> V{c.state_version_after}: {c.reason}")
    print(f"State After: V{t.state_after.state_version}")
    print(f"  - Raw Readiness: {t.state_after.dimensions.readiness:.2f}")
    print(f"  - Raw Trust: {t.state_after.dimensions.trust:.2f}")
    print(f"  - Raw Valence: {t.state_after.dimensions.emotion_valence:.2f}")
    print(f"  - Raw Tension: {t.state_after.dimensions.emotion_tension:.2f}")
    gate_str = t.state_after.conversion_gate.status if t.state_after.conversion_gate else "None (Pending)"
    print(f"  - Conversion Gate: {gate_str}")
    push_str = t.state_after.push_strength.state if t.state_after.push_strength else "None"
    print(f"  - Push Strength: {push_str}")
    read_str = f"{t.state_after.readiness.readiness_score}%" if t.state_after.readiness else "None"
    print(f"  - Composite Readiness: {read_str}")
    print(f"  - Recorded Facts: {len(t.state_after.facts)}")
    for f in t.state_after.facts:
        print(f"    [{f.category}] {f.fact_key} = {f.fact_value}")

print("\n=== FINAL SUMMARY ===")
print("Gate is open:", rep.conversion_summary['gate_is_open'])
print("Gate status:", rep.conversion_summary['gate_status'])
print("Push strength:", rep.conversion_summary['push_strength'])
print("Recommended action:", rep.conversion_summary['recommended_action'])
print("Conversion event:", rep.conversion_summary['conversion_event'])
print("Active blockers:", rep.conversion_summary['active_blockers'])
