import json
from pathlib import Path

golden_path = Path("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json")
with open(golden_path, "r", encoding="utf-8") as f:
    golden = json.load(f)

rows = []
for step in golden["timeline"]:
    tid = step["turn_id"]
    sp = step["speaker_id"]
    strat = step.get("strategic_decision") or {}
    sa = step.get("state_snapshot") or step.get("state_after") or {}
    ps = sa.get("push_strength") or {}
    gate = sa.get("conversion_gate") or {}
    r = sa.get("readiness") or {}
    ce = sa.get("conversion_event") or {}

    action = strat.get("primary_action")
    posture = strat.get("strategic_posture")
    push = strat.get("push_strength")
    conf = strat.get("confidence")

    r_score = r.get("readiness_score")
    r_cov = r.get("coverage", 0.0)
    is_insuff = r.get("insufficient_evidence", True)

    gate_open = gate.get("is_open")
    gate_status = gate.get("status")

    # Readiness description
    if r_score is None:
        readiness_desc = f"`null` (insufficient, cov={r_cov:.2f})"
    else:
        status_label = "partial" if is_insuff or r_cov < 0.50 else "confirmed"
        if r.get("active_blocker_caps"):
            status_label += ", capped"
        readiness_desc = f"{r_score:.1f} ({status_label}), cov={r_cov:.2f}"

    gate_desc = gate_status if gate_status is not None else "closed"

    rows.append({
        "turn": tid,
        "speaker": sp,
        "action": action,
        "posture": posture,
        "push": push,
        "confidence": conf,
        "readiness": readiness_desc,
        "readiness_score": r_score,
        "coverage": r_cov,
        "gate_is_open": gate_open,
        "gate_status": gate_status,
        "conversion_event_status": ce.get("status") if ce else None
    })

# Save exported JSON
out_json_path = Path("reports/synthetic/canonical_18_turn_baseline_exported.json")
out_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Exported {len(rows)} turns to {out_json_path}")

# Print markdown table
print("| Turn | Speaker | Primary Action | Strategic Posture | Push Pressure | Confidence | Readiness Score & Status | Gate Status |")
print("|:---:|:---|:---|:---:|:---:|:---:|:---|:---:|")
for r in rows:
    conf_str = f"{r['confidence']:.3f}" if r['confidence'] is not None else "—"
    print(f"| **{r['turn']}** | {r['speaker']} | `{r['action']}` | `{r['posture']}` | `{r['push']}` | {conf_str} | {r['readiness']} | `{r['gate_status'] or 'closed'}` |")
