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
        speaker = step.get("speaker_id") or step.get("speaker") or "client"
        text = step.get("text", "")
        sa = step.get("state_after", {})
        dec = step.get("strategic_decision") or {}
        
        # Build maps of snapshot facts and objections
        facts_map = {f["fact_id"]: f for f in sa.get("facts", [])}
        objs_map = {o["objection_id"]: o for o in sa.get("objections", [])}
        
        # Build set of snapshot stakeholders
        stk_set = set()
        for s in sa.get("stakeholders", []):
            if s.get("stakeholder_id"):
                stk_set.add(s["stakeholder_id"])
            if s.get("role"):
                stk_set.add(s["role"])
                stk_set.add(f"stakeholder_{s['role']}")
        if sa.get("decision_structure") and sa["decision_structure"].get("stakeholders"):
            for s in sa["decision_structure"]["stakeholders"]:
                if s.get("stakeholder_id"):
                    stk_set.add(s["stakeholder_id"])
                if s.get("role"):
                    stk_set.add(s["role"])
                    stk_set.add(f"stakeholder_{s['role']}")
        
        ref_facts = dec.get("referenced_fact_ids", [])
        orig_facts = dec.get("original_referenced_fact_ids", [])
        ref_objs = dec.get("referenced_objection_ids", [])
        ref_stks = dec.get("referenced_stakeholder_ids", [])
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
                orig_str = f" [orig: {orig_facts}]" if orig_facts and orig_facts != ref_facts else ""
                fact_audit.append(f"`{fid}` (Resolves: YES, Status: {st}{succ_str}{orig_str})")
            else:
                fact_audit.append(f"`{fid}` (Resolves: NO - MISSING)")
                
        obj_audit = []
        for oid in ref_objs:
            if oid in objs_map:
                st = objs_map[oid].get("lifecycle_state", "active")
                obj_audit.append(f"`{oid}` (Resolves: YES, Lifecycle: {st})")
            else:
                obj_audit.append(f"`{oid}` (Resolves: NO - MISSING)")

        stk_audit = []
        for sid in ref_stks:
            if sid in stk_set:
                stk_audit.append(f"`{sid}` (Resolves: YES)")
            else:
                stk_audit.append(f"`{sid}` (Resolves: NO - MISSING)")
                
        all_resolved = (
            all("Resolves: YES" in fa for fa in fact_audit)
            and all("Resolves: YES" in oa for oa in obj_audit)
            and all("Resolves: YES" in sa_item for sa_item in stk_audit)
        )
        
        rows.append({
            "turn_id": tid,
            "speaker": speaker,
            "text": text,
            "carried": carried,
            "action": primary_action,
            "push": push,
            "ref_facts": ref_facts,
            "orig_facts": orig_facts,
            "fact_audit": fact_audit,
            "ref_objs": ref_objs,
            "obj_audit": obj_audit,
            "ref_stks": ref_stks,
            "stk_audit": stk_audit,
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
        speaker = step.get("speaker_id") or step.get("speaker")
        text = step.get("text", "")
        sb = step.get("state_before") or {}
        sa = step.get("state_after") or {}
        sd = step.get("strategic_decision") or {}
        bundle = step.get("evidence_bundle") or {}
        
        # Decision engine inputs for this turn
        gate = sa.get("conversion_gate") or {}
        readiness = sa.get("readiness") or {}
        dims = sa.get("dimensions") or {}
        
        # Dimension measured flags from state and inference bundle
        bundle_trust = bundle.get("trust") or {}
        bundle_emotion = bundle.get("emotion") or {}
        bundle_eng = bundle.get("engagement") or {}
        bundle_mom = bundle.get("momentum") or {}

        trust_measured = bool(dims.get("trust_measured", False) or bundle_trust.get("is_measured", False))
        tension_measured = bool(bundle_emotion.get("is_measured", False))
        engagement_measured = bool(bundle_eng.get("is_measured", False))
        momentum_measured = bool(bundle_mom.get("is_measured", False))

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
            "readiness_partial": readiness.get("readiness_partial"),
            "dimensions": {
                "trust": dims.get("trust"),
                "trust_measured": trust_measured,
                "tension": dims.get("emotion_tension"),
                "tension_measured": tension_measured,
                "engagement": dims.get("engagement"),
                "engagement_measured": engagement_measured,
                "momentum": dims.get("momentum"),
                "momentum_measured": momentum_measured,
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

def format_table(rows, filter_turns=None, title=""):
    res = [f"### {title}\n"]
    res.append("| Turn | Speaker | Utterance | Carried? | Action / Strategy | Push | Referenced Facts | Fact Status & Resolution | Referenced Objections | Objection Lifecycle & Resolution | Referenced Stakeholders | Stakeholder Resolution | Audit Result |")
    res.append("|:---:|:---:|:---|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|")
    sorted_rows = list(rows)
    if filter_turns:
        # Put filter_turns first
        head = [r for r in sorted_rows if r["turn_id"] in filter_turns]
        tail = [r for r in sorted_rows if r["turn_id"] not in filter_turns]
        sorted_rows = head + tail
        
    for r in sorted_rows:
        tid = r["turn_id"]
        spk = r["speaker"]
        utt = r["text"].replace("|", "\\|")
        carried = "**Yes**" if r["carried"] else "No"
        act = r["action"]
        push = r["push"]
        facts = ", ".join(f"`{f}`" for f in r["ref_facts"]) or "*None*"
        facts_res = "<br>".join(r["fact_audit"]) or "*N/A*"
        objs = ", ".join(f"`{o}`" for o in r["ref_objs"]) or "*None*"
        objs_res = "<br>".join(r["obj_audit"]) or "*N/A*"
        stks = ", ".join(f"`{s}`" for s in r["ref_stks"]) or "*None*"
        stks_res = "<br>".join(r["stk_audit"]) or "*N/A*"
        status = "**PASSED (100% Resolves)**" if r["all_resolved"] else "**FAILED**"
        res.append(f"| **{tid}** | {spk} | \"{utt}\" | {carried} | `{act}` | `{push}` | {facts} | {facts_res} | {objs} | {objs_res} | {stks} | {stks_res} | {status} |")
    return "\n".join(res)

if __name__ == "__main__":
    # 1. sim_muy04bh0
    data_muy, rows_muy = audit_replay("sim_muy04bh0", "script_sim_muy04bh0_fee_complaint.json")

    # 2. sim_mucj0p5s (canonical 18-turn) - Semantic OFF
    data_18_off, rows_18_off = audit_replay("sim_mucj0p5s", "script_canonical.json", run_semantic=False)

    # 3. sim_mucj0p5s (canonical 18-turn) - Semantic ON
    data_18_on, rows_18_on = audit_replay("sim_mucj0p5s_sem", "script_canonical.json", run_semantic=True)

    print("\n" + "="*80)
    print(format_table(rows_muy, filter_turns=[3,4,5,6,7,8], title="Audit: sim_muy04bh0 (Fee Objection Script) - Turns 3 to 8 First"))
    print("\n" + "="*80)
    print(format_table(rows_18_off, title="Audit: sim_mucj0p5s (Canonical 18-Turn Benchmark - Semantic Layer OFF)"))
    print("\n" + "="*80)
    print(format_table(rows_18_on, title="Audit: sim_mucj0p5s (Canonical 18-Turn Benchmark - Semantic Layer ON)"))

    # 4. Export runtime inputs
    inputs_18 = export_runtime_inputs("sim_mucj0p5s", "script_canonical.json")
    with open("scratch/runtime_inputs_18_turns.json", "w", encoding="utf-8") as f:
        json.dump(inputs_18, f, indent=2)
    with open("reports/runtime_inputs_18_turns.json", "w", encoding="utf-8") as f:
        json.dump(inputs_18, f, indent=2)
    print("\n=== EXPORTED RUNTIME INPUTS TO reports/runtime_inputs_18_turns.json ===")
