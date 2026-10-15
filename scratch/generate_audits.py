import json
import time
from pathlib import Path
from copilot.conversation_replay import ConversationReplayEngine
from copilot.core_decision_manager import StrategicDecisionCache

def audit_replay(call_sid: str, script_file: str, run_semantic: bool = False):
    StrategicDecisionCache.clear()
    engine = ConversationReplayEngine()
    script_path = Path("data") / script_file
    raw_turns = json.loads(script_path.read_text(encoding="utf-8"))
    rep = engine.replay_dialogue_turns(
        call_sid=call_sid,
        raw_turns=raw_turns,
        save_report=True,
        run_semantic_analysis=run_semantic,
    )
    
    report_path = Path("reports") / "synthetic" / f"conversation_state_{call_sid}.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    
    rows = []
    for step in data.get("timeline", []):
        tid = step.get("turn_id")
        speaker = step.get("speaker")
        text = step.get("text", "")
        sa = step.get("state_after", {})
        dec = step.get("strategic_decision") or {}
        
        # Build maps of snapshot facts and objections
        facts_map = {f["fact_id"]: f for f in sa.get("facts", [])}
        objs_map = {o["objection_id"]: o for o in sa.get("objections", [])}
        
        ref_facts = dec.get("referenced_fact_ids", [])
        ref_objs = dec.get("referenced_objection_ids", [])
        carried = dec.get("carried_forward", False)
        primary_action = dec.get("primary_action", "—")
        push = dec.get("push_strength", "—")
        if isinstance(push, dict):
            push = push.get("pressure", "—")
        
        fact_audit = []
        for fid in ref_facts:
            if fid in facts_map:
                st = facts_map[fid].get("status", "active")
                succ = facts_map[fid].get("superseded_by_fact_id")
                succ_str = f" (succ: {succ})" if succ else ""
                fact_audit.append(f"`{fid}` (Resolves: YES, Status: {st}{succ_str})")
            else:
                fact_audit.append(f"`{fid}` (Resolves: NO - MISSING)")
                
        obj_audit = []
        for oid in ref_objs:
            if oid in objs_map:
                st = objs_map[oid].get("lifecycle_state", "active")
                obj_audit.append(f"`{oid}` (Resolves: YES, Lifecycle: {st})")
            else:
                obj_audit.append(f"`{oid}` (Resolves: NO - MISSING)")
                
        all_resolved = all("Resolves: YES" in fa for fa in fact_audit) and all("Resolves: YES" in oa for oa in obj_audit)
        
        rows.append({
            "turn_id": tid,
            "speaker": speaker,
            "text": text,
            "carried": carried,
            "action": primary_action,
            "push": push,
            "ref_facts": ref_facts,
            "fact_audit": fact_audit,
            "ref_objs": ref_objs,
            "obj_audit": obj_audit,
            "all_resolved": all_resolved,
        })
    return data, rows


def export_runtime_inputs(call_sid: str, script_file: str):
    StrategicDecisionCache.clear()
    engine = ConversationReplayEngine()
    script_path = Path("data") / script_file
    raw_turns = json.loads(script_path.read_text(encoding="utf-8"))
    rep = engine.replay_dialogue_turns(
        call_sid=call_sid,
        raw_turns=raw_turns,
        save_report=True,
    )
    
    report_path = Path("reports") / "synthetic" / f"conversation_state_{call_sid}.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    
    runtime_inputs = []
    for step in data.get("timeline", []):
        tid = step.get("turn_id")
        speaker = step.get("speaker")
        text = step.get("text", "")
        sb = step.get("state_before") or {}
        sa = step.get("state_after") or {}
        sd = step.get("strategic_decision") or {}
        
        # Decision engine inputs for this turn
        gate = sa.get("conversion_gate") or {}
        readiness = sa.get("readiness") or {}
        dims = sa.get("dimensions") or {}
        inputs = {
            "turn_id": tid,
            "speaker": speaker,
            "utterance": text,
            "source_state_version": sa.get("state_version"),
            "source_event_id": sd.get("source_event_id"),
            "conversation_stage": sa.get("conversation_stage"),
            "conversion_gate": {
                "is_open": gate.get("is_open"),
                "status": gate.get("status"),
                "target": gate.get("conversion_target"),
            },
            "conversion_confirmed": sa.get("conversion_confirmed"),
            "readiness_score": readiness.get("readiness_score"),
            "dimensions": {
                "trust": dims.get("trust"),
                "tension": dims.get("emotion_tension"),
                "engagement": dims.get("engagement"),
                "momentum": dims.get("momentum"),
            },
            "unresolved_objection_ids": [o.get("objection_id") for o in sa.get("objections", []) if o.get("lifecycle_state") in ("active", "reactivated", "partially_resolved")],
            "active_fact_ids": [f.get("fact_id") for f in sa.get("facts", []) if f.get("status") == "active"],
            "decision_output": {
                "primary_action": sd.get("primary_action"),
                "strategic_posture": sd.get("strategic_posture"),
                "push_pressure": sd.get("push_strength", {}).get("pressure") if isinstance(sd.get("push_strength"), dict) else sd.get("push_strength"),
                "carried_forward": sd.get("carried_forward"),
                "referenced_fact_ids": sd.get("referenced_fact_ids"),
                "referenced_objection_ids": sd.get("referenced_objection_ids"),
                "reason_codes": sd.get("reason_codes"),
            }
        }
        runtime_inputs.append(inputs)
        
    return {
        "call_sid": call_sid,
        "run_id": f"run_{call_sid}_{int(time.time())}",
        "created_at": data.get("created_at"),
        "total_turns": len(runtime_inputs),
        "runtime_inputs": runtime_inputs
    }

if __name__ == "__main__":
    # 1. sim_muy04bh0
    data_muy, rows_muy = audit_replay("sim_muy04bh0", "script_sim_muy04bh0_fee_complaint.json")
    print("=== SIM_MUY04BH0 AUDIT ===")
    for r in rows_muy:
        print(f"Turn {r['turn_id']}: Carried={r['carried']}, Action={r['action']}, Push={r['push']}")
        print(f"   Facts: {r['fact_audit']}")
        print(f"   Objs:  {r['obj_audit']}")
        print(f"   All Resolved: {r['all_resolved']}")

    # 2. sim_mucj0p5s (canonical 18-turn) - Semantic OFF
    data_18_off, rows_18_off = audit_replay("sim_mucj0p5s", "script_canonical.json", run_semantic=False)
    print("\n=== SIM_MUCJ0P5S AUDIT (Semantic OFF) ===")
    for r in rows_18_off:
        print(f"Turn {r['turn_id']}: Carried={r['carried']}, Action={r['action']}, Push={r['push']}, All Resolved={r['all_resolved']}, Facts={r['fact_audit']}, Objs={r['obj_audit']}")

    # 3. sim_mucj0p5s (canonical 18-turn) - Semantic ON
    data_18_on, rows_18_on = audit_replay("sim_mucj0p5s_sem", "script_canonical.json", run_semantic=True)
    print("\n=== SIM_MUCJ0P5S AUDIT (Semantic ON) ===")
    for r in rows_18_on:
        print(f"Turn {r['turn_id']}: Carried={r['carried']}, Action={r['action']}, Push={r['push']}, All Resolved={r['all_resolved']}, Facts={r['fact_audit']}, Objs={r['obj_audit']}")

    # 4. Export runtime inputs
    inputs_18 = export_runtime_inputs("sim_mucj0p5s", "script_canonical.json")
    with open("scratch/runtime_inputs_18_turns.json", "w", encoding="utf-8") as f:
        json.dump(inputs_18, f, indent=2)
    print("\n=== EXPORTED RUNTIME INPUTS TO scratch/runtime_inputs_18_turns.json ===")
