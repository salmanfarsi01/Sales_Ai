"""Integration tests for Conditional Retrieval Test Console UI and API Router."""

import pytest
from fastapi.testclient import TestClient

from copilot.fastapi_app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_serve_conditional_retrieval_ui(client):
    """Verifies that GET /conditional-test and /conditional-retrieval serve the UI HTML file."""
    res = client.get("/conditional-test")
    assert res.status_code == 200
    assert "PitchProX | Conditional Knowledge Retrieval Test Console" in res.text
    assert "Acceptance Presets:" in res.text
    assert "Multi-Turn Script Studio" in res.text
    assert "Multi-Turn Execution Matrix" in res.text

    res_alias = client.get("/conditional-retrieval")
    assert res_alias.status_code == 200
    assert "PitchProX | Conditional Knowledge Retrieval Test Console" in res_alias.text


def test_get_acceptance_presets(client):
    """Verifies that all 14 acceptance presets are returned by the catalog endpoint."""
    res = client.get("/api/conditional-retrieval/presets")
    assert res.status_code == 200
    presets = res.json()
    assert len(presets) >= 14
    ids = [p["id"] for p in presets]
    assert "scenario_1" in ids
    assert "scenario_2" in ids
    assert "scenario_4" in ids
    assert "scenario_6" in ids
    assert "scenario_13" in ids
    assert "scenario_14" in ids


def test_list_user_records_and_load_dummies(client):
    """Verifies listing records and seeding canonical dummy docs."""
    # Seed dummies
    seed_res = client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
    assert seed_res.status_code == 200
    assert seed_res.json()["status"] == "success"

    # List records
    list_res = client.get("/api/conditional-retrieval/records?user_id=rep_john")
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["user_id"] == "rep_john"
    assert data["total_records"] >= 5
    kids = [r["knowledge_id"] for r in data["records"]]
    assert "chunk_westlake_sales" in kids
    assert "chunk_commission_v2" in kids


def test_ingest_and_delete_chunk(client):
    """Verifies ingesting a custom chunk and soft-deleting it."""
    payload = {
        "knowledge_id": "chunk_custom_test_001",
        "owner_user_id": "rep_john",
        "owner_scope": "user",
        "knowledge_class": "performance_past_sales",
        "source_title": "Custom Test Doc",
        "text": "Closed 10 homes in Cedar Park.",
        "verification_status": "company_approved",
    }
    res = client.post("/api/conditional-retrieval/records", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Soft delete it
    del_res = client.post(
        "/api/conditional-retrieval/records/delete",
        json={"user_id": "rep_john", "knowledge_id": "chunk_custom_test_001", "soft_tombstone": True},
    )
    assert del_res.status_code == 200
    assert del_res.json()["action"] == "soft_tombstoned"


def test_simulate_pipeline_relevant_upload(client):
    """Verifies full end-to-end simulation of Scenario 2 (Westlake past sales)."""
    payload = {
        "user_id": "rep_john",
        "conversation_evidence": "Do you have any actual sales experience in Westlake?",
        "strategic_action": "quantify",
        "trigger": "credibility_objection",
        "need_statement": "verified sales experience in Westlake",
        "topic_tags": ["westlake", "sales"],
        "state_version": 1,
    }
    res = client.post("/api/conditional-retrieval/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Gate passed
    assert data["gate_evaluation"]["decision"] == "proceed"
    # Retrieved 1 fact
    assert data["retrieval_result"]["status"] == "retrieved"
    assert len(data["retrieval_result"]["facts"]) == 1
    assert "42 homes in Westlake" in data["retrieval_result"]["facts"][0]["text"]
    # Context assembled with 6 blocks
    assert "### BLOCK 1: STRATEGIC DECISION CONTRACT" in data["context_assembled"]
    assert "### BLOCK 3: VERIFIED KNOWLEDGE" in data["context_assembled"]
    # Teleprompter prompt generated
    assert data["prompt_result"]["status"] == "generated"
    assert len(data["prompt_result"]["text"]) > 0
    # Hallucination scan safe
    assert data["hallucination_scanner"]["safe"] is True


def test_simulate_pipeline_no_trigger_fast_exit(client):
    """Verifies fast short-circuit (0ms vector calls) when trigger is none."""
    payload = {
        "user_id": "rep_john",
        "conversation_evidence": "Yeah, that sounds okay.",
        "strategic_action": "acknowledge",
        "trigger": "none",
        "need_statement": "",
        "topic_tags": [],
        "state_version": 1,
    }
    res = client.post("/api/conditional-retrieval/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["gate_evaluation"]["decision"] == "none_needed"
    assert data["retrieval_result"]["status"] == "none_needed"
    assert len(data["retrieval_result"]["facts"]) == 0


def test_simulate_pipeline_interruption_stale_discard(client):
    """Verifies that mid-generation interruption marks the prompt stale_discarded."""
    payload = {
        "user_id": "rep_john",
        "conversation_evidence": "Do you have sales experience in Westlake?",
        "strategic_action": "quantify",
        "trigger": "credibility_objection",
        "need_statement": "sales experience in Westlake",
        "topic_tags": ["westlake"],
        "state_version": 1,
        "simulate_interruption": True,  # Causes state_version to advance during generation
    }
    res = client.post("/api/conditional-retrieval/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prompt_result"]["status"] == "stale_discarded"
    assert data["prompt_result"]["is_stale_discarded"] is True


def test_latency_metrics_endpoint(client):
    """Verifies that GET /api/conditional-retrieval/latency-metrics returns statistical p50/p95."""
    res = client.get("/api/conditional-retrieval/latency-metrics")
    assert res.status_code == 200
    metrics = res.json()
    assert "stage_percentiles" in metrics or "percentiles" in metrics
    assert "total_calls" in metrics
    assert "deadline_alert" in metrics


def test_preset_executes_live_engine_not_canned_document_mutation(client):
    """Proves that presets execute the live retrieval and LLM engine with zero canned responses.

    1. Runs Scenario 2 (Westlake past sales).
       Initial fact in store: "I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market."
       Verifies returned fact cites 42 homes and generated prompt cites 42 homes.
    2. Dynamically mutates chunk_westlake_sales in the authoritative store:
       New text: "I have closed 88 homes in Westlake over the past 5 years with an average of 12 days on market."
    3. Re-runs the exact same simulation request.
    4. Asserts that the returned fact is updated to 88 homes and the generated prompt immediately cites 88 homes,
       conclusively proving the engine runs live and does not return canned/mocked output.
    5. Restores canonical dummies to preserve test isolation.
    """
    # Ensure clean starting store
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")

    # Step 1: Run Scenario 2 initially
    payload = {
        "user_id": "rep_john",
        "conversation_evidence": "Do you have any actual sales experience in Westlake?",
        "strategic_action": "quantify",
        "trigger": "credibility_objection",
        "need_statement": "verified sales experience in Westlake",
        "topic_tags": ["westlake", "sales"],
        "state_version": 1,
    }
    res_1 = client.post("/api/conditional-retrieval/simulate", json=payload)
    assert res_1.status_code == 200
    data_1 = res_1.json()
    assert data_1["retrieval_result"]["status"] == "retrieved"
    assert "42 homes in Westlake" in data_1["retrieval_result"]["facts"][0]["text"]
    assert "42" in data_1["prompt_result"]["text"]

    # Step 2: Mutate the document in the live store
    mutated_text = "I have closed 88 homes in Westlake over the past 5 years with an average of 12 days on market."
    ingest_payload = {
        "knowledge_id": "chunk_westlake_sales",
        "owner_user_id": "rep_john",
        "owner_scope": "user",
        "knowledge_class": "performance_past_sales",
        "source_title": "Westlake Production Record (Updated)",
        "source_version": "2.0",
        "topic_tags": ["westlake", "experience", "sales", "track_record"],
        "text": mutated_text,
        "verification_status": "company_approved",
    }
    update_res = client.post("/api/conditional-retrieval/records", json=ingest_payload)
    assert update_res.status_code == 200

    # Step 3: Rerun the exact same preset request
    res_2 = client.post("/api/conditional-retrieval/simulate", json=payload)
    assert res_2.status_code == 200
    data_2 = res_2.json()

    # Step 4: Verify the result changed dynamically (ZERO canned responses!)
    assert data_2["retrieval_result"]["status"] == "retrieved"
    assert len(data_2["retrieval_result"]["facts"]) == 1
    assert "88 homes in Westlake" in data_2["retrieval_result"]["facts"][0]["text"]
    assert "42 homes" not in data_2["retrieval_result"]["facts"][0]["text"]
    assert "88" in data_2["prompt_result"]["text"]

    # Step 5: Restore canonical dummies
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")


def test_simulate_pipeline_autonomous_core_decision(client):
    """Verifies that setting trigger='auto' invokes Core Intelligence autonomously."""
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
    payload = {
        "user_id": "rep_john",
        "conversation_evidence": "Do you have any actual sales experience in Westlake?",
        "strategic_action": "quantify",
        "trigger": "auto",  # Autonomous Core Mode
        "need_statement": "",
        "topic_tags": [],
        "state_version": 1,
    }
    res = client.post("/api/conditional-retrieval/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["core_decision"] is not None
    assert data["core_decision"]["autonomous"] is True
    assert data["core_decision"]["trigger_chosen"] == "credibility_objection"
    assert data["core_decision"]["retrieval_needed"] is True
    assert data["retrieval_result"]["status"] == "retrieved"
    assert "42 homes in Westlake" in data["retrieval_result"]["facts"][0]["text"]


def test_get_benchmark_scripts(client):
    """Verifies that GET /api/conditional-retrieval/scripts returns the 3 canonical benchmark scripts."""
    res = client.get("/api/conditional-retrieval/scripts")
    assert res.status_code == 200
    scripts = res.json()
    assert len(scripts) >= 3
    ids = [s["id"] for s in scripts]
    assert "canonical_8_turn" in ids
    assert "pricing_state_continuity_5_turn" in ids
    assert "speed_filler_4_turn" in ids

    c8 = next(s for s in scripts if s["id"] == "canonical_8_turn")
    assert len(c8["lines"]) == 8


def test_multi_turn_simulate_canonical_8_turn(client):
    """Verifies full execution of the 8-turn benchmark script through the real Core and retrieval engine."""
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")

    res_scripts = client.get("/api/conditional-retrieval/scripts")
    scripts = res_scripts.json()
    c8 = next(s for s in scripts if s["id"] == "canonical_8_turn")

    req_payload = {
        "user_id": "rep_john",
        "call_id": "bench_call_8_turn",
        "script_name": c8["name"],
        "prospect_lines": c8["lines"],
    }
    res = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["total_turns"] == 8
    assert data["total_violations"] == 0
    turns = data["turns"]
    assert len(turns) == 8

    # Turn 1: Greeting -> Core decides no retrieval needed (0ms)
    t1 = turns[0]
    assert t1["turn_index"] == 1
    assert t1["retrieval_needed"] is False
    assert t1["gate_result"] == "none_needed"
    assert len(t1["facts_delivered"]) == 0
    assert t1["latency_ms"] >= 0.0

    # Turn 2: Westlake track record -> Core picks credibility_objection -> Retrieves 42 homes
    t2 = turns[1]
    assert t2["turn_index"] == 2
    assert t2["retrieval_needed"] is True
    assert t2["core_trigger"] == "credibility_objection"
    assert t2["gate_result"] == "proceed"
    assert len(t2["facts_delivered"]) == 1
    assert "42 homes in Westlake" in t2["facts_delivered"][0]["text"]
    assert t2["hallucination_safe"] is True

    # Turn 3: "Okay, makes sense" -> Core stays quiet (0ms)
    t3 = turns[2]
    assert t3["turn_index"] == 3
    assert t3["retrieval_needed"] is False
    assert t3["gate_result"] == "none_needed"
    assert len(t3["facts_delivered"]) == 0

    # Turn 4: Commission fee -> Core picks policy_wording -> Delivers 5%, withholds 6% as stale
    t4 = turns[3]
    assert t4["turn_index"] == 4
    assert t4["retrieval_needed"] is True
    assert t4["core_trigger"] == "policy_wording"
    assert len(t4["facts_delivered"]) == 1
    assert "5.0%" in t4["facts_delivered"][0]["text"]
    # Verify legacy 6% was withheld specifically as stale (superseded), NOT disputed
    v1_withheld = next(w for w in t4["withheld"] if w["knowledge_id"] == "chunk_commission_v1")
    assert v1_withheld["reason"] == "stale"

    # Turn 5: Sarah rebate trap -> rep_john isolation prevents Sarah's doc from leaking (Constraint C6)
    t5 = turns[4]
    assert t5["turn_index"] == 5
    delivered_ids_t5 = [f["knowledge_id"] for f in t5["facts_delivered"]]
    assert "chunk_sarah_rebate" not in delivered_ids_t5
    assert len(t5["facts_delivered"]) == 0
    # Verify rep does not claim company rebate policy
    assert "we don't offer cash back" not in t5["teleprompter_line"].lower()
    assert "net proceeds" in t5["teleprompter_line"].lower()

    # Turn 6: 7-day guarantee trap -> unverified pitch deck is withheld, 0 facts delivered, declines guarantee
    t6 = turns[5]
    assert t6["turn_index"] == 6
    delivered_ids_t6 = [f["knowledge_id"] for f in t6["facts_delivered"]]
    assert "chunk_old_pitch_deck" not in delivered_ids_t6
    assert len(t6["facts_delivered"]) == 0, f"Expected 0 facts delivered on guarantee trap, got {t6['facts_delivered']}"
    assert "42 homes" not in t6["teleprompter_line"], "Should not recite past sales on guarantee inquiry"
    # Verify rep does not state unsupported company stance
    assert "we don't make artificial" not in t6["teleprompter_line"].lower()
    assert "can't promise a specific timeline" in t6["teleprompter_line"].lower()

    # Turn 7: Afternoon 3 PM schedule -> Live prospect constraint overrides stored morning 10 AM doc
    t7 = turns[6]
    assert t7["turn_index"] == 7
    # Verify stored 10 AM morning chunk was withheld due to conflict with live prospect constraint
    withheld_ids_t7 = [w["knowledge_id"] for w in t7["withheld"]]
    assert "chunk_stored_schedule" in withheld_ids_t7
    # Verify rep checks availability without leaking withheld weekday mornings chunk or falsely agreeing to 3:00 PM
    assert "weekday mornings" not in t7["teleprompter_line"].lower(), "Withheld morning schedule must NOT leak into teleprompter line"
    assert "3:00" in t7["teleprompter_line"] or "3 pm" in t7["teleprompter_line"].lower()
    assert any(w in t7["teleprompter_line"].lower() for w in ("calendar", "check", "verify", "possible", "availability"))
    assert "works great" not in t7["teleprompter_line"].lower()
    assert "tuesday" not in t7["teleprompter_line"].lower()
    assert "thursday" not in t7["teleprompter_line"].lower()

    # Turn 8: Inspection contingency -> Educational inquiry routes to EDUCATE, 0 facts delivered, generic grounded line
    t8 = turns[7]
    assert t8["turn_index"] == 8
    assert t8["core_action"] == "EDUCATE", f"Expected EDUCATE for educational inquiry, got {t8['core_action']}"
    delivered_ids_t8 = [f["knowledge_id"] for f in t8["facts_delivered"]]
    assert "chunk_unrelated_zoning" not in delivered_ids_t8
    assert "chunk_salesperson_profile" not in delivered_ids_t8, "Marginal profile bio must NOT be delivered on inspection contingency inquiry"
    assert len(t8["facts_delivered"]) == 0, f"Low confidence must deliver 0 facts, got {t8['facts_delivered']}"
    assert "john miller has" not in t8["teleprompter_line"].lower(), "Must not recite third-person bio on inspection contingency inquiry"
    assert "10 to 14 days" not in t8["teleprompter_line"], "Must not invent 10 to 14 days inspection timeline"

    # Verify Core actions vary across turns (not stuck on QUESTION for every turn)
    actions = [t["core_action"] for t in turns]
    assert len(set(actions)) >= 3, f"Core actions did not vary across turns: {actions}"

    # Verify per-stage latencies breakdown per Doc 01 §13 and PASS score
    for t in turns:
        assert "context_assembled" in t
        assert "context_blocks" in t
        assert len(t["context_assembled"]) > 0
        lat = t["stage_latencies"]
        assert "state_analysis_ms" in lat
        assert "core_decision_ms" in lat
        assert "retrieval_ms" in lat
        assert "generation_ms" in lat
        assert "guardrail_ms" in lat
        assert "total_turn_ms" in lat
        assert t["score"] == "PASS", f"Turn {t['turn_index']} failed: {t.get('score_reasons')}"


def test_multi_turn_replay_mutation(client):
    """Verifies that mutating a seeded document and replaying the script immediately updates the result."""
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")

    script_lines = [
        "Good morning John.",
        "What is your standard commission fee to list our home?",
    ]
    req = {
        "user_id": "rep_john",
        "call_id": "replay_test_call",
        "script_name": "Replay Mutation Test",
        "prospect_lines": script_lines,
    }

    # Initial run -> Turn 2 delivers 5.0%
    res1 = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req)
    assert res1.status_code == 200
    t2_run1 = res1.json()["turns"][1]
    assert "5.0%" in t2_run1["facts_delivered"][0]["text"]
    assert "5.0%" in t2_run1["teleprompter_line"]

    # Mutate chunk_commission_v2 in live store to 4.25%
    mutated_doc = {
        "knowledge_id": "chunk_commission_v2",
        "owner_user_id": "rep_john",
        "owner_scope": "user",
        "knowledge_class": "pricing_packaging",
        "source_title": "Listing Agreement Schedule 2026 (Special Promotion)",
        "source_version": "2.1",
        "topic_tags": ["commission", "fees", "pricing"],
        "text": "Standard residential listing commission rate is 4.25% of gross purchase price.",
        "verification_status": "company_approved",
    }
    client.post("/api/conditional-retrieval/records", json=mutated_doc)

    # Replay identical script
    res2 = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req)
    assert res2.status_code == 200
    t2_run2 = res2.json()["turns"][1]
    assert "4.25%" in t2_run2["facts_delivered"][0]["text"]
    assert "4.25%" in t2_run2["teleprompter_line"]

    # Restore canonical dummies
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")


def test_upload_file_endpoint(client):
    """Verifies that uploading a document file extracts text and indexes discrete chunks."""
    doc_content = (
        "Westlake Premium Escrow Guarantee.\n\n"
        "All Westlake transactions close within 14 business days with zero escrow cancellation fees.\n\n"
        "Contact the Westlake regional director for emergency wire transfers."
    )
    files = {
        "file": ("escrow_policy.txt", doc_content.encode("utf-8"), "text/plain")
    }
    data = {
        "user_id": "rep_john",
        "knowledge_class": "product_specs",
        "verification_status": "company_approved",
        "topic_tags": "escrow, guarantee, westlake",
        "source_title": "Westlake Escrow Policy Doc",
    }
    res = client.post("/api/conditional-retrieval/upload-file", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["success"] is True
    assert res_data["chunks_created"] >= 2
    assert "escrow_policy.txt" in res_data["file_name"]

    # Verify chunks are now listed in user records
    list_res = client.get("/api/conditional-retrieval/records?user_id=rep_john")
    assert list_res.status_code == 200
    records = list_res.json()["records"]
    matching = [r for r in records if "Westlake Premium Escrow" in r["text"]]
    assert len(matching) > 0


def test_timing_precision_and_context_blocks(client):
    """Verifies that latencies are strictly > 0 ms and all 6 context blocks are populated."""
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
    req = {
        "user_id": "rep_john",
        "call_id": "timing_test_call",
        "script_name": "Timing Precision Test",
        "prospect_lines": ["What is your standard commission fee to list our home?"],
        "use_real_llm": False,
    }
    res = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req)
    assert res.status_code == 200
    turn = res.json()["turns"][0]

    # Verify timing precision is > 0 ms
    assert turn["latency_ms"] > 0.0, f"Turn latency was {turn['latency_ms']} ms"

    # Verify all 6 context blocks are present with non-empty descriptions/content
    blocks = turn["context_blocks"]
    assert "block_1" in blocks or "block_1_operational_mode" in blocks
    assert "block_2" in blocks or "block_2_strategic_intent" in blocks
    assert "block_3" in blocks or "block_3_live_evidence" in blocks
    assert "block_4" in blocks or "block_4_verified_facts" in blocks
    assert "block_5" in blocks or "block_5_boundary_constraints" in blocks
    assert "block_6" in blocks or "block_6_response_format" in blocks

    # Verify teleprompter source badge field
    assert "teleprompter_source" in turn
    assert turn["teleprompter_source"] == "deterministic"


def test_multi_turn_messy_unstructured_script(client):
    """Verifies that the Messy & Unstructured script executes with independent pass criteria."""
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
    scripts = client.get("/api/conditional-retrieval/scripts").json()
    script = next(s for s in scripts if s["id"] == "messy_unstructured_5_turn")
    req = {
        "user_id": "rep_john",
        "call_id": "call_messy_test",
        "script_name": script["name"],
        "prospect_lines": script["lines"],
    }
    res = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["total_turns"] == 5
    assert data["total_violations"] == 0
    assert data["passed_turns_count"] == 5
    for t in data["turns"]:
        assert t["score"] == "PASS"


def test_multi_turn_adversarial_security_script(client):
    """Verifies prompt injection neutralization, C6 isolation, and hard boundary compliance."""
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
    scripts = client.get("/api/conditional-retrieval/scripts").json()
    script = next(s for s in scripts if s["id"] == "adversarial_security_traps_5_turn")
    req = {
        "user_id": "rep_john",
        "call_id": "call_adversarial_test",
        "script_name": script["name"],
        "prospect_lines": script["lines"],
    }
    res = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["total_turns"] == 5
    assert data["total_violations"] == 0
    assert data["passed_turns_count"] == 5
    # Turn 5: Hard boundary exit verified
    t5 = data["turns"][4]
    assert "respect that" in t5["teleprompter_line"].lower()
    assert "call" not in t5["teleprompter_line"].lower() or "not" in t5["teleprompter_line"].lower()


def test_multi_turn_repeated_objection_script(client):
    """Verifies objection ladder progression and non-repetition across 3 recurrences."""
    client.post("/api/conditional-retrieval/load-canonical-dummies?user_id=rep_john")
    scripts = client.get("/api/conditional-retrieval/scripts").json()
    script = next(s for s in scripts if s["id"] == "repeated_objection_3_turn")
    req = {
        "user_id": "rep_john",
        "call_id": "call_repeated_test",
        "script_name": script["name"],
        "prospect_lines": script["lines"],
    }
    res = client.post("/api/conditional-retrieval/multi-turn-simulate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["total_turns"] == 3
    assert data["passed_turns_count"] == 3
    turns = data["turns"]
    # All three spoken lines must be distinct
    lines = [t["teleprompter_line"] for t in turns]
    assert len(set(lines)) == 3, f"Core repeated lines across recurrences: {lines}"


def test_sentence_level_hallucination_grounding_checks():
    """Doc 02 §9: Verifies that broad business claims (contract terms, market claims, staging) are flagged."""
    from copilot.conditional_retrieval_models import RetrievedFact
    from copilot.llm_response_gateway import validate_prompt_hallucinations

    verified_fact = RetrievedFact(
        knowledge_id="f1",
        knowledge_class="pricing_packaging",
        source_title="Pricing Policy",
        text="Standard residential listing commission rate is 5.0% of gross purchase price.",
    )

    # 1. Contract claim without document support
    is_safe, violations = validate_prompt_hallucinations(
        "There is zero obligation—you can walk away anytime if you don't love our service.",
        [verified_fact],
    )
    assert not is_safe
    assert any("contract claim" in v.lower() for v in violations)

    # 2. Market speed claim without document support
    is_safe, violations = validate_prompt_hallucinations(
        "Current market inventory is moving faster than last quarter, which directly impacts your pricing window.",
        [verified_fact],
    )
    assert not is_safe
    assert any("market claim" in v.lower() for v in violations)

    # 3. Service inclusion / staging claim without document support
    is_safe, violations = validate_prompt_hallucinations(
        "Our standard residential listing commission rate is 5.0%, which includes full professional marketing and staging.",
        [verified_fact],
    )
    assert not is_safe
    assert any("service claim" in v.lower() for v in violations)

    # 4. Scheduling assertion without verified consultation doc
    is_safe, violations = validate_prompt_hallucinations(
        "Our consultations are usually weekday mornings, so let me check whether tomorrow at 3:00 PM is possible.",
        [verified_fact],
    )
    assert not is_safe
    assert any("scheduling assertion" in v.lower() for v in violations)

    # 5. Clean, strictly grounded prompt passes
    is_safe, violations = validate_prompt_hallucinations(
        "Our standard residential listing commission rate is 5.0% of gross purchase price.",
        [verified_fact],
    )
    assert is_safe
    assert len(violations) == 0



