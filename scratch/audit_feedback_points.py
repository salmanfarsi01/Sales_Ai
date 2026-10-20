import sys
sys.path.insert(0, ".")
import json
from pathlib import Path

path = Path("reports/synthetic/conversation_state_sim_mvdoiaih.json")
if not path.exists():
    print("Report file not found")
    sys.exit(1)

data = json.loads(path.read_text(encoding="utf-8"))

print("=== TIMELINE AUDIT: SIM_MVDOIAIH ALL TURNS ===")
for s in data["timeline"]:
    tid = s["turn_id"]
    spk = s["speaker_id"]
    txt = s["text"]
    sa = s["state_after"]
    gate = sa.get("conversion_gate") or {}
    ce = sa.get("conversion_event") or {}
    stage = sa.get("conversation_stage")
    ds = sa.get("decision_structure") or {}
    mat = s.get("materiality") or {}
    changes = [c.get("field_path") for c in s.get("state_changes", [])]
    v = sa.get("state_version")
    
    print(f"Turn {tid:02d} [{spk:11s}]: v={v} | changes={changes}")
    print(f"    text: '{txt}'")
    print(f"    gate: open={gate.get('is_open')}, status={gate.get('status')}, failed={gate.get('failed_conditions')}")
    print(f"    ce: status={ce.get('status')}, start_at={ce.get('start_at')}, participants={ce.get('participants')}, sources={ce.get('source_turn_ids')}")
    print(f"    stage: {stage}")
    print(f"    ds: dm_present={ds.get('decision_maker_present')}, co_dec={ds.get('co_decision_required')}, primary={ds.get('primary_decision_maker')}, stks={ds.get('stakeholders')}")
    print(f"    mat: is_material={mat.get('is_material')}, targets={mat.get('affected_targets')}, unclassified={mat.get('unclassified_material')}")
    dec = s.get("strategic_decision") or {}
    if dec:
        print(f"    decision: action={dec.get('primary_action')}, posture={dec.get('strategic_posture')}, push={dec.get('push_strength')}, reasons={dec.get('reason_codes')}")
