import json

with open("reports/synthetic/conversation_state_sim_muy04bh0.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

print(f"Report call_sid: {rep.get('call_sid')} | timeline count: {len(rep.get('timeline', []))}")
for t in rep.get("timeline", []):
    tid = t.get("turn_id")
    spk = t.get("speaker_id")
    txt = t.get("text")
    st = t.get("state_after") or {}
    cg = st.get("conversion_gate") or {}
    ce = st.get("conversion_events") or []
    objs = st.get("objections") or []
    mat = t.get("materiality") or {}
    unclass = t.get("unclassified_material", False)
    dec = t.get("strategic_decision") or {}
    carried = dec.get("carried_forward", False)
    cf_dec_id = dec.get("carried_forward_from_decision_id")
    cf_turn_id = dec.get("carried_forward_from_turn_id")
    action = dec.get("primary_action")
    posture = dec.get("strategic_posture")
    prompt = dec.get("should_prompt")
    prompt_txt = dec.get("prompt_text") or dec.get("strategic_objective")
    ref_objs = dec.get("referenced_objection_ids")
    codes = dec.get("reason_codes")
    print(f"--- Turn {tid} ({spk}): {txt}")
    print(f"    Material: {mat.get('classification')} / score={mat.get('score')} | unclassified_mat={unclass}")
    print(f"    Gate: {cg.get('state')} | slot: {cg.get('commitment_slot')} | events: {len(ce)} | objs: {[o.get('canonical_category') for o in objs]}")
    print(f"    Decision: action={action} posture={posture} prompt={prompt} carried={carried} cf_turn={cf_turn_id} cf_dec={cf_dec_id}")
    print(f"    Prompt/Obj: {prompt_txt}")
    print(f"    Ref Objs: {ref_objs} | Codes: {codes}")
