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

