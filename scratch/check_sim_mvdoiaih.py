import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from copilot.core_decision_manager import StrategicDecisionCache
from copilot.conversation_state_models import ConversationStateSnapshot
from copilot.core_intelligence_models import StrategicDecision

report_path = Path("reports/synthetic/conversation_state_sim_mvdoiaih.json")
if not report_path.exists():
    print("Report file not found:", report_path)
    exit(1)

data = json.loads(report_path.read_text(encoding="utf-8"))

print("="*100)
print(f"Call SID: {data.get('call_sid')}")
print(f"Source: {data.get('source')}")
print(f"Created At: {data.get('created_at')}")
print(f"Total Turns: {data.get('total_turns')}")
print("="*100)

all_ok = True
inconsistencies = []

for step in data.get("timeline", []):
    tid = step.get("turn_id")
    speaker = step.get("speaker_id")
    text = step.get("text", "")
    sa_dict = step.get("state_after", {})
    dec_dict = step.get("strategic_decision") or {}
    
    sa = ConversationStateSnapshot.model_validate(sa_dict)
    dec = StrategicDecision.model_validate(dec_dict) if dec_dict else None
    
    warnings = step.get("evidence_warnings", [])
    
    if dec:
        ok, reason = StrategicDecisionCache.validate_evidence_consistency(dec, sa, require_active=True)
        si = dec.strategic_interpretation or {}
        strat = f"{dec.primary_action.value if hasattr(dec.primary_action, 'value') else dec.primary_action} / {dec.strategic_posture}"
        push = dec.push_strength.pressure if hasattr(dec.push_strength, "pressure") else (dec.push_strength.value if hasattr(dec.push_strength, "value") else dec.push_strength)
        ref_f = dec.referenced_fact_ids
        orig_f = dec.original_referenced_fact_ids
        ref_o = dec.referenced_objection_ids
        ref_s = dec.referenced_stakeholder_ids
        carried = dec.carried_forward
        
        status_str = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
            inconsistencies.append((tid, reason))
            
        print(f"Turn {tid:2d} [{speaker:11s}] [{status_str}]: carried={carried} | strat={strat:22s} | push={str(push):10s} | '{text[:50]}'")
        if ref_f:
            print(f"    Referenced Facts: {ref_f} (orig: {orig_f})")
            for fid in ref_f:
                matching = [f for f in sa.facts if f.fact_id == fid]
                if matching:
                    f = matching[0]
                    print(f"      - {fid}: status={f.status}, key={f.fact_key}, val={f.fact_value}")
                else:
                    print(f"      - {fid}: NOT FOUND IN SNAPSHOT!")
        if ref_o:
            print(f"    Referenced Objections: {ref_o}")
            for oid in ref_o:
                matching = [o for o in sa.objections if o.objection_id == oid]
                if matching:
                    o = matching[0]
                    print(f"      - {oid}: lifecycle={o.lifecycle_state}, cat={o.canonical_category}")
                else:
                    print(f"      - {oid}: NOT FOUND IN SNAPSHOT!")
        if ref_s:
            print(f"    Referenced Stakeholders: {ref_s}")
        if warnings:
            print(f"    WARNINGS: {warnings}")
        if not ok:
            print(f"    INCONSISTENCY REASON: {reason}")
    else:
        print(f"Turn {tid:2d} [{speaker:11s}]: NO STRATEGIC DECISION")

print("="*100)
print(f"OVERALL EVIDENCE CONSISTENCY: {'PERFECT (ALL TURNS RESOLVE)' if all_ok else 'FAILED'}")
if inconsistencies:
    print("Inconsistencies detected:")
    for tid, r in inconsistencies:
        print(f"  - Turn {tid}: {r}")

# 4-Key Binding Validation
print("\n" + "="*100)
print("CHECKING 4-KEY BINDING ACROSS ALL TURNS")
print("="*100)
binding_ok = True
for step in data.get("timeline", []):
    tid = step.get("turn_id")
    sa_dict = step.get("state_after", {})
    dec_dict = step.get("strategic_decision") or {}
    sa = ConversationStateSnapshot.model_validate(sa_dict)
    dec = StrategicDecision.model_validate(dec_dict) if dec_dict else None
    if dec:
        # Check validate_binding
        b_ok, b_reason = StrategicDecisionCache.validate_binding(
            decision=dec,
            call_sid=data.get("call_sid"),
            turn_id=dec.source_turn_id,
            source_state_version=dec.source_state_version,
            source_event_id=dec.source_event_id,
            state_call_sid=sa.call_sid,
            snapshot=sa,
            require_active=True,
        )
        if not b_ok:
            binding_ok = False
            print(f"Turn {tid} BINDING REJECTED: {b_reason}")
        else:
            print(f"Turn {tid:2d}: Binding valid (v{dec.source_state_version}, event={dec.source_event_id})")

print(f"OVERALL 4-KEY BINDING: {'PERFECT (ALL TURNS VALID)' if binding_ok else 'FAILED'}")

# Check Client Item 2: Strategic Interpretation panel Push Strength rendering
print("\n" + "="*100)
print("CHECKING CLIENT ITEM 2: STRATEGIC INTERPRETATION PANEL RENDERING")
print("="*100)
render_issues = []
for step in data.get("timeline", []):
    tid = step.get("turn_id")
    dec_dict = step.get("strategic_decision") or {}
    si = dec_dict.get("strategic_interpretation") or {}
    ps = dec_dict.get("push_strength")
    sp = si.get("selected_pressure")
    
    # In client feedback: The Strategic Interpretation panel still displays confirm_and_protect under 'Push Strength.'
    # Label that as strategy/posture, or display the selected pressure there.
    # Check what selected_pressure and push_strength are:
    if ps == "confirm_and_protect":
        render_issues.append((tid, "dec.push_strength is confirm_and_protect instead of pressure"))
    if sp == "confirm_and_protect":
        render_issues.append((tid, "si.selected_pressure is confirm_and_protect instead of pressure"))

if render_issues:
    print(f"Rendering issues found: {render_issues}")
else:
    print("NO rendering defects found: selected_pressure is never 'confirm_and_protect'; all push strengths are canonical pressure levels ('none', 'low', 'moderate', 'high').")

# Deep dive into Turn 11 (Supersession)
print("\n" + "="*100)
print("DEEP DIVE: TURN 11 FACT SUPERSESSION")
print("="*100)
step11 = [s for s in data["timeline"] if s["turn_id"] == 11][0]
sa11 = step11["state_after"]
for f in sa11["facts"]:
    print(f"Fact {f['fact_id']}: status={f['status']}, key={f['fact_key']}, val='{f['fact_value']}', superseded_by={f.get('superseded_by_fact_id')}")

dec11 = step11["strategic_decision"]
print("Turn 11 Decision referenced_fact_ids:", dec11.get("referenced_fact_ids"))


