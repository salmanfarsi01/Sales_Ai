import json

data = json.load(open("reports/synthetic/conversation_state_call_canonical_replay_golden_071026.json", encoding="utf-8"))
tl = data["timeline"]
print(f"=== Canonical 18-Turn Replay Export (Call ID: {data['call_sid']}) ===")
for s in tl:
    tid = s["turn_id"]
    spk = s["speaker_id"]
    txt = s["text"]
    sa = s.get("state_after") or {}
    sd = s.get("strategic_decision") or {}
    act = sd.get("primary_action", "")
    posture = sd.get("strategic_posture", "")
    push = sd.get("push_strength", "")
    conf = sd.get("confidence", 0.0)
    
    # Readiness
    readiness = sa.get("readiness") or {}
    r_score = readiness.get("readiness_score")
    r_str = f"{r_score:.1f}" if r_score is not None else "None"
    
    # Momentum
    momentum = sa.get("momentum") or {}
    m_score = momentum.get("momentum_score")
    m_trend = momentum.get("trend")
    m_str = f"{m_score:.1f} ({m_trend})" if m_score is not None else "None"
    
    # Trust
    dims = sa.get("dimensions") or {}
    trust_val = dims.get("trust", 0.5)
    t_meas = dims.get("trust_measured", False)
    t_str = f"{trust_val:.2f} ({'measured' if t_meas else 'unmeasured'})"
    
    # Gate
    gate = sa.get("conversion_gate") or {}
    g_open = gate.get("is_open", False)
    g_stat = gate.get("status", "closed")
    g_str = f"{'OPEN' if g_open else 'CLOSED'} ({g_stat})"
    
    codes = sd.get("reason_codes", [])
    
    print(f"Turn {tid:02d} [{spk:<11}] Act={act:<12} Conf={conf:.3f} | Read={r_str:<6} Mom={m_str:<18} Trust={t_str:<22} Gate={g_str:<14} | Codes={codes}")
