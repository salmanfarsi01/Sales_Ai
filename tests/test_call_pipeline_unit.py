"""Unit & Pipeline Integration Test Suite for Pre-Call Folder & Parallel Calling Architecture.

Tests:
1. Pre-Call Folder Initialization and In-Memory Caching (call_folder.py)
2. All 16 Standardized Lead Types Resolution
3. Custom Playbook Methodology Lens Integration
4. Zero-Allocation Prompt Assembly Structure (<25 words rule, labeled sections)
5. FastAPI Endpoints (/api/call/init-folder, /api/call/folder/{sid}, /api/calibration/latest)
6. Parallel Concurrency Pipeline Simulation (asyncio.gather for RAG + transcript logging)
"""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from copilot.call_folder import (
    PreCallFolder,
    build_pre_call_folder,
    get_or_create_pre_call_folder,
    CALL_FOLDERS,
)
from copilot.playbook import (
    STANDARDIZED_LEAD_TYPES,
    Playbook,
    PlaybookBasicInfo,
    PlaybookStyle,
    PlaybookObjection,
    ResponseSequenceStep,
    PlaybookStore,
)
from copilot.fastapi_app import app


@pytest.fixture(autouse=True)
def clean_call_folders():
    """Ensure clean state before and after each test."""
    CALL_FOLDERS.clear()
    yield
    CALL_FOLDERS.clear()


def test_all_16_standardized_lead_types_valid():
    """Verify all 16 standardized lead types are recognized and mapped."""
    assert len(STANDARDIZED_LEAD_TYPES) == 16
    for item in STANDARDIZED_LEAD_TYPES:
        lead_type = item["name"]
        desc = item["desc"]
        assert len(lead_type) > 0
        assert len(desc) > 0

        # Build folder for this lead type
        folder = build_pre_call_folder(call_sid=f"call_{lead_type[:4]}", lead_type=lead_type)
        assert folder.lead_type == lead_type
        assert folder.lead_type_desc == desc


def test_pre_call_folder_creation_and_caching():
    """Verify pre-call folder is assembled and cached in memory."""
    call_sid = "call_test_12345"
    folder = build_pre_call_folder(
        call_sid=call_sid,
        lead_type="Expired Listings",
        playbook_id=None,
        salesman_id="agent_alpha",
    )

    assert folder.call_sid == call_sid
    assert folder.lead_type == "Expired Listings"
    assert "Expired Listings" in folder.lead_type_desc or "expired" in folder.lead_type_desc.lower()
    assert len(folder.playbook_title) > 0
    assert "ACTIVE PLAYBOOK METHODOLOGY LENS" in folder.playbook_prompt_lens
    assert "WPM" in folder.calibration_summary_text
    assert folder.salesman_id == "agent_alpha"

    # Verify cached in memory
    assert call_sid in CALL_FOLDERS
    retrieved = get_or_create_pre_call_folder(call_sid)
    assert retrieved is folder


def test_pre_call_folder_with_custom_playbook(tmp_path):
    """Verify pre-call folder incorporates custom playbook rules and objection matrix."""
    store = PlaybookStore(storage_dir=tmp_path / "playbooks")
    custom_pb = Playbook(
        title="The Closer Lens",
        basic_info=PlaybookBasicInfo(
            name="The Closer Lens",
            short_description="Direct, high-efficiency objection handling",
            industry="Real Estate",
            lead_types=["FSBO (For Sale By Owner)"],
            tone_descriptor="Assertive & Consultative",
            philosophy="Validate the homeowner's desire to save equity, then differentiate on net proceeds.",
        ),
        style=PlaybookStyle(
            primary_tones=["Direct", "Empathetic"],
            pacing_words_per_minute=145,
            energy_level="High",
            forbidden_words=["Maybe", "Obviously", "No problem"],
        ),
        response_sequence=[
            ResponseSequenceStep(order=1, name="Validate", icon="heart"),
            ResponseSequenceStep(order=2, name="Educate", icon="file"),
            ResponseSequenceStep(order=3, name="Direct", icon="target"),
        ],
        objections=[
            PlaybookObjection(
                objection="I'm selling it myself to save the 6% commission",
                lead_types=["FSBO (For Sale By Owner)"],
                category="Financial",
                response_style="Reframe",
                ai_suggestion="Validate equity goal, then contrast limited buyer pool with private qualified buyer list.",
            )
        ],
    )
    store.save(custom_pb)

    call_sid = "call_custom_pb_123"
    folder = build_pre_call_folder(
        call_sid=call_sid,
        lead_type="FSBO (For Sale By Owner)",
        playbook_id=custom_pb.id,
        store=store,
    )

    assert folder.playbook_id == custom_pb.id
    assert folder.playbook_title == custom_pb.basic_info.name
    assert "THE CLOSER LENS" in folder.playbook_prompt_lens
    assert "Validate -> Educate -> Direct" in folder.playbook_prompt_lens
    assert "Energy=High" in folder.playbook_prompt_lens
    assert "save equity" in folder.playbook_prompt_lens


def test_prompt_assembly_structure():
    """Verify assemble_prompt constructs exact required message sequence."""
    folder = build_pre_call_folder(
        call_sid="call_prompt_test",
        lead_type="Probate",
    )

    history = [
        {"role": "user", "content": "Hello, who is this?"},
        {"role": "assistant", "content": "Hi there, I noticed the probate filing on Elm Street."},
    ]
    current_utterance = "We're not interested in selling right now, thanks."
    rag_evidence = "Company policy offers 30-day no-commission consultation for estate executors."

    messages = folder.assemble_prompt(
        conversation_context=history,
        current_utterance=current_utterance,
        rag_evidence=rag_evidence,
    )

    # 1. System instruction
    assert len(messages) == 4  # system + 2 history turns + 1 current prompt turn
    assert messages[0]["role"] == "system"
    assert "=== 1. [PRE-CALL FOLDER: IMMUTABLE BASELINE] ===" in messages[0]["content"]
    assert "Probate" in messages[0]["content"]

    # 2. History turns
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, who is this?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there, I noticed the probate filing on Elm Street."

    # 3. Final Turn with RAG
    assert messages[3]["role"] == "user"
    assert "Client just said: \"We're not interested in selling right now, thanks.\"" in messages[3]["content"]
    assert "[VERIFIED COMPANY KNOWLEDGE]:" in messages[3]["content"]
    assert "30-day no-commission consultation" in messages[3]["content"]


def test_api_init_call_folder_and_get_folder():
    """Verify POST /api/call/init-folder and GET /api/call/folder/{call_sid} endpoints."""
    client = TestClient(app)

    # 1. POST init-folder
    res = client.post("/api/call/init-folder", data={
        "call_sid": "api_call_sid_777",
        "lead_type": "Pre-Foreclosure",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["call_sid"] == "api_call_sid_777"
    assert data["lead_type"] == "Pre-Foreclosure"
    assert "distress" in data["lead_type_desc"].lower()

    # 2. GET folder info
    res_get = client.get("/api/call/folder/api_call_sid_777")
    assert res_get.status_code == 200
    folder_data = res_get.json()
    assert folder_data["call_sid"] == "api_call_sid_777"
    assert folder_data["lead_type"] == "Pre-Foreclosure"


def test_api_calibration_latest():
    """Verify GET /api/calibration/latest returns valid calibration summary."""
    client = TestClient(app)
    res = client.get("/api/calibration/latest")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("success", "default")
    assert "wpm" in data or "report" in data


@pytest.mark.asyncio
async def test_parallel_rag_concurrency_simulation():
    """Simulate parallel execution of RAG search and transcript logging via asyncio.gather."""
    history_log = []
    rag_mock_results = []

    async def mock_log_transcript(utterance: str):
        await asyncio.sleep(0.005)  # simulate async DB / memory append
        history_log.append({"role": "user", "content": utterance})
        return True

    async def mock_rag_search(query: str):
        await asyncio.sleep(0.010)  # simulate Pinecone async vector search
        rag_mock_results.append(f"Matching knowledge for: {query}")
        return "Special promotion available for expired listings."

    utterance = "Why should I re-list with you when my last agent failed?"

    # Execute simultaneously via asyncio.gather
    results = await asyncio.gather(
        mock_log_transcript(utterance),
        mock_rag_search(utterance),
    )

    assert results[0] is True
    assert "Special promotion" in results[1]
    assert len(history_log) == 1
    assert history_log[0]["content"] == utterance
