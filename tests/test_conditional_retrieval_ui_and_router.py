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
