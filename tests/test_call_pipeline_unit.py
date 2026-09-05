"""Unit & Pipeline Integration Test Suite for Pre-Call Folder & Parallel Calling Architecture.

Tests:
1. Core Intelligence Engine Permanency & 4-Step Diagnostic Reasoning Pattern
2. Pre-Call Folder Initialization and In-Memory Caching (call_folder.py)
3. All 16 Standardized Lead Types Resolution
4. Custom Playbook Methodology Lens Integration
5. Zero-Allocation Prompt Assembly Structure (<25 words rule, labeled sections, hierarchy ordering)
6. FastAPI Endpoints (/api/call/init-folder, /api/call/folder/{sid}, /api/calibration/latest)
7. Parallel Concurrency Pipeline Simulation (asyncio.gather for RAG + transcript logging)
"""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from copilot.call_folder import (
    PreCallFolder,
    build_pre_call_folder,
    get_or_create_pre_call_folder,
    CORE_INTELLIGENCE_INSTRUCTION,
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


def test_core_intelligence_engine_permanent_and_unconditional():
    """Verify that Core Intelligence Engine is permanently at the top of the prompt on every turn."""
    # 1. Build bare minimal folder with default settings (no custom playbook, no custom training)
    folder = build_pre_call_folder(call_sid="call_core_test_1", lead_type="Expired Listings")
    
    assert folder.core_engine_name == "PitchProX Core Intelligence (Permanent)"
    
    messages = folder.assemble_prompt(
        conversation_context=[],
        current_utterance="Why are you calling me?",
        rag_evidence="",
    )

    system_prompt = messages[0]["content"]

    # Verify Core Intelligence Header
    assert "=== [CORE INTELLIGENCE — ALWAYS ACTIVE] ===" in system_prompt

    # Verify the 4-Step Core Diagnostic Reasoning Pattern
    assert "1. Identify literally what the prospect just said." in system_prompt
    assert "2. Determine the underlying concern or intent behind it" in system_prompt
    assert "3. Decide the single best strategic move available right now" in system_prompt
    assert "4. Produce ONE exact sentence the rep should say" in system_prompt

    # Verify Guardrails
    assert "Never become defensive" in system_prompt
    assert "Never simply list facts in response to a challenge" in system_prompt
    assert "Always ground the move in what would actually move this specific conversation forward" in system_prompt

    # Verify Core instruction is placed FIRST (before Lead Type, Playbook, Calibration)
    core_idx = system_prompt.find("[CORE INTELLIGENCE — ALWAYS ACTIVE]")
    lead_idx = system_prompt.find("[LEAD TYPE: Expired Listings]")
    playbook_idx = system_prompt.find("ACTIVE PLAYBOOK METHODOLOGY LENS")
    calib_idx = system_prompt.find("[CALIBRATION PROFILE]")

    assert core_idx != -1
    assert lead_idx != -1
    assert playbook_idx != -1
    assert calib_idx != -1

    # Strict Hierarchy Ordering: Core (1st) -> Lead Type (2nd) -> Playbook (3rd) -> Calibration (4th)
    assert core_idx < lead_idx < playbook_idx < calib_idx


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
    assert "Core Intelligence" in folder.core_engine_name

    # Verify cached in memory
    assert call_sid in CALL_FOLDERS
    retrieved = get_or_create_pre_call_folder(call_sid)
    assert retrieved is folder


def test_pre_call_folder_with_custom_playbook(tmp_path):
    """Verify pre-call folder incorporates custom playbook rules without modifying permanent Core logic."""
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

    # Ensure Core Engine is still at the top
    messages = folder.assemble_prompt([], "I'll do it on my own.")
    assert "=== [CORE INTELLIGENCE — ALWAYS ACTIVE] ===" in messages[0]["content"]


def test_prompt_assembly_structure():
    """Verify assemble_prompt constructs exact required message sequence with Core Brain, transcript, and RAG."""
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

    # 1. System instruction with Core Engine first
    assert len(messages) == 4  # system + 2 history turns + 1 current prompt turn
    assert messages[0]["role"] == "system"
    assert "=== [CORE INTELLIGENCE — ALWAYS ACTIVE] ===" in messages[0]["content"]
    assert "Probate" in messages[0]["content"]

    # 2. History turns
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, who is this?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there, I noticed the probate filing on Elm Street."

    # 3. Final Turn with RAG & Core execution directive
    assert messages[3]["role"] == "user"
    assert "Prospect just said: \"We're not interested in selling right now, thanks.\"" in messages[3]["content"]
    assert "[AI TRAINING / COMPANY KNOWLEDGE]:" in messages[3]["content"]
    assert "30-day no-commission consultation" in messages[3]["content"]
    assert "Core Reasoning & Response:" in messages[3]["content"]


def test_api_init_call_folder_and_get_folder():
    """Verify POST /api/call/init-folder and GET /api/call/folder/{call_sid} endpoints include Core Engine."""
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
    assert "Core Intelligence" in data["core_engine"]
    assert data["lead_type"] == "Pre-Foreclosure"
    assert "distress" in data["lead_type_desc"].lower()

    # 2. GET folder info
    res_get = client.get("/api/call/folder/api_call_sid_777")
    assert res_get.status_code == 200
    folder_data = res_get.json()
    assert folder_data["call_sid"] == "api_call_sid_777"
    assert "Core Intelligence" in folder_data["core_engine"]
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
