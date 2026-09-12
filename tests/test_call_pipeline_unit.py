"""Unit & Pipeline Integration Test Suite for Pre-Call Folder & Parallel Calling Architecture.

Tests:
1. Core Intelligence Engine Permanency & 4-Step Diagnostic Reasoning Pattern
2. Stacked Plain-Text Prompt Builder (build_prompt)
3. Deterministic "Respond Now or Wait" Suppression Gate (should_generate_now)
4. Pre-Call Folder Initialization and In-Memory Caching (call_folder.py)
5. All 16 Standardized Lead Types Resolution
6. Custom Playbook Methodology Lens Integration & Runtime Cooldown Settings
7. Zero-Allocation Prompt Assembly Structure (<25 words rule, labeled sections, hierarchy ordering)
8. FastAPI Endpoints (/api/call/init-folder, /api/call/folder/{sid}, /api/calibration/latest)
9. Parallel Concurrency Pipeline Simulation (asyncio.gather for RAG + transcript logging)
"""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from copilot.call_folder import (
    PreCallFolder,
    build_pre_call_folder,
    get_or_create_pre_call_folder,
    build_prompt,
    should_generate_now,
    CORE_INSTRUCTIONS,
    CORE_INTELLIGENCE_INSTRUCTION,
    CALL_FOLDERS,
)
from copilot.playbook import (
    STANDARDIZED_LEAD_TYPES,
    Playbook,
    PlaybookBasicInfo,
    PlaybookStyle,
    PlaybookObjection,
    PlaybookRuntimeSettings,
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


def test_stacked_build_prompt_function():
    """Verify build_prompt stacks plain text pieces starting with permanent Core."""
    # 1. Bare minimal prompt (only Core + Utterance)
    raw = build_prompt(current_utterance="Hello")
    assert CORE_INSTRUCTIONS in raw
    assert 'Prospect just said: "Hello"' in raw
    assert "[LEAD TYPE" not in raw
    assert "[AI TRAINING" not in raw

    # 2. Fully populated prompt
    full_raw = build_prompt(
        lead_type="Expired Listings",
        lead_type_desc="Listings that recently expired without selling.",
        playbook_prompt_lens="### ACTIVE PLAYBOOK METHODOLOGY LENS: CONSULTATIVE ###",
        calibration_summary="Pacing: 140 WPM",
        training_snippet="Brokerage closed 40 homes in this zip code last year.",
        conversation_turns=[
            {"role": "user", "content": "Who is this?"},
            {"role": "assistant", "content": "Hi, I am calling about your property."},
        ],
        current_utterance="Why should I list with you?",
    )

    # Verify exact stacked order: Core -> Lead -> Playbook -> Calibration -> Training -> Conversation
    p_core = full_raw.find("CORE INTELLIGENCE")
    p_lead = full_raw.find("[LEAD TYPE: Expired Listings]")
    p_play = full_raw.find("ACTIVE PLAYBOOK METHODOLOGY LENS")
    p_calib = full_raw.find("[CALIBRATION PROFILE]")
    p_train = full_raw.find("[AI TRAINING / COMPANY KNOWLEDGE]")
    p_conv = full_raw.find("[CONVERSATION SO FAR]")

    assert p_core != -1
    assert p_lead != -1
    assert p_play != -1
    assert p_calib != -1
    assert p_train != -1
    assert p_conv != -1

    assert p_core < p_lead < p_play < p_calib < p_train < p_conv
    assert "Brokerage closed 40 homes" in full_raw
    assert 'Prospect just said: "Why should I list with you?"' in full_raw


def test_should_generate_now_deterministic_gate():
    """Verify deterministic code-level suppression rules before LLM invocation."""
    current_time = 100.0

    # 1. Non-critical short junk -> Suppress
    ok, reason = should_generate_now(
        utterance="zz", is_final=True, speech_final=True,
        last_generation_time=0.0, current_time=current_time,
    )
    assert not ok
    assert "insufficient_length" in reason

    # 2. Critical short objections like "Why?", "No.", "Cost?" -> Allowed
    for short_obj in ["Why?", "No.", "How?", "Cost?", "Pass."]:
        ok, reason = should_generate_now(
            utterance=short_obj, is_final=True, speech_final=True,
            last_generation_time=0.0, current_time=current_time,
        )
        assert ok, f"Expected '{short_obj}' to be allowed but got {reason}"
        assert reason == "ready_to_respond"

    # 3. Passive filler acknowledgments -> Suppress
    ok, reason = should_generate_now(
        utterance="yeah, okay", is_final=True, speech_final=True,
        last_generation_time=0.0, current_time=current_time,
    )
    assert not ok
    assert "passive_acknowledgment" in reason

    # 4. Mid-thought dangling conjunction without speech_final -> Suppress
    ok, reason = should_generate_now(
        utterance="I want to sell my house but", is_final=True, speech_final=False,
        last_generation_time=0.0, current_time=current_time,
    )
    assert not ok
    assert "mid_thought_dangling_connector" in reason

    # 5. Mid-thought trailing ellipsis without speech_final -> Suppress
    ok, reason = should_generate_now(
        utterance="I mean, I guess my concern is...", is_final=True, speech_final=False,
        last_generation_time=0.0, current_time=current_time,
    )
    assert not ok
    assert "mid_thought_trailing_ellipsis" in reason

    # 6. Deepgram is_final=True without speech_final on mid-sentence phrase -> Suppress (awaiting speech final pause)
    ok, reason = should_generate_now(
        utterance="I called your office yesterday to check", is_final=True, speech_final=False,
        last_generation_time=0.0, current_time=current_time,
    )
    assert not ok
    assert "awaiting_speech_final_pause" in reason

    # 7. Playbook Prompt Cooldown active (e.g. last prompt 2s ago, cooldown is 5s) -> Suppress
    ok, reason = should_generate_now(
        utterance="What is your commission rate?", is_final=True, speech_final=True,
        last_generation_time=98.0, current_time=100.0, cooldown_seconds=5.0,
    )
    assert not ok
    assert "cooldown_active" in reason

    # 8. Cooldown expired (last prompt 6s ago, cooldown is 5s) -> Allow
    ok, reason = should_generate_now(
        utterance="What is your commission rate?", is_final=True, speech_final=True,
        last_generation_time=94.0, current_time=100.0, cooldown_seconds=5.0,
    )
    assert ok
    assert reason == "ready_to_respond"


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
    assert folder.prompt_cooldown_seconds >= 1.0

    # Verify cached in memory
    assert call_sid in CALL_FOLDERS
    retrieved = get_or_create_pre_call_folder(call_sid)
    assert retrieved is folder


def test_pre_call_folder_with_custom_playbook(tmp_path):
    """Verify pre-call folder incorporates custom playbook rules and runtime cooldown settings."""
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
            runtime_settings=PlaybookRuntimeSettings(
                prompt_cooldown_seconds=6,
                prompt_timing_sensitivity="Relaxed",
            ),
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
    assert folder.prompt_cooldown_seconds == 6.0
    assert folder.prompt_timing_sensitivity == "Relaxed"
    assert "THE CLOSER LENS" in folder.playbook_prompt_lens
    assert "Validate -> Educate -> Direct" in folder.playbook_prompt_lens
    assert "Energy=High" in folder.playbook_prompt_lens
    assert "save equity" in folder.playbook_prompt_lens

    # Test gate on folder instance
    should_run, reason = folder.should_respond_now(
        utterance="Why did my last agent fail?",
        is_final=True,
        speech_final=True,
        last_generation_time=95.0,
        current_time=98.0,  # 3s elapsed vs 6s cooldown
    )
    assert not should_run
    assert "cooldown_active" in reason


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


def test_behavioral_test_dashboard_and_analyze_recording(monkeypatch):
    """Verify GET /behavioral-test serves the UI and POST /api/test/analyze-recording processes audio."""
    client = TestClient(app)

    # 1. UI route serves the dashboard
    resp = client.get("/behavioral-test")
    assert resp.status_code == 200
    assert "Behavioral Signal Test Console" in resp.text
    assert "#0EADAB" in resp.text

    # 2. Mock STT to return deterministic transcript and word timestamps
    from copilot.calibration import SpeechToTextEngine
    async def mock_transcribe(*args, **kwargs):
        return (
            "We need to review this pricing proposal with our board next week.",
            [
                {"word": "We", "start": 0.1, "end": 0.3},
                {"word": "need", "start": 0.35, "end": 0.6},
                {"word": "to", "start": 0.65, "end": 0.8},
                {"word": "review", "start": 0.85, "end": 1.2},
                {"word": "this", "start": 1.25, "end": 1.45},
                {"word": "pricing", "start": 1.5, "end": 1.9},
                {"word": "proposal", "start": 2.2, "end": 2.7}, # intra-turn pause 300ms
                {"word": "with", "start": 2.75, "end": 2.95},
                {"word": "our", "start": 3.0, "end": 3.2},
                {"word": "board", "start": 3.25, "end": 3.6},
                {"word": "next", "start": 3.65, "end": 3.9},
                {"word": "week.", "start": 3.95, "end": 4.3},
            ]
        )
    monkeypatch.setattr(SpeechToTextEngine, "transcribe_with_timestamps", mock_transcribe)

    # POST dummy audio to /api/test/analyze-recording
    files = {"audio": ("test.webm", b"RIFFFAKEAUDIOBYTES", "audio/webm")}
    data = {"prospect_id": "prospect_ui_test"}
    post_resp = client.post("/api/test/analyze-recording", files=files, data=data)
    assert post_resp.status_code == 200
    res_data = post_resp.json()

    assert res_data["status"] == "success"
    assert "pricing proposal" in res_data["transcript"]
    assert "latest_timing" in res_data
    assert "latest_inference" in res_data
    assert res_data["latest_timing"]["turn_length_words"] == 12
    assert res_data["latest_inference"]["acoustic_evidence"] == "unavailable"
    assert "trust" in res_data["latest_inference"]
    assert "pacing" in res_data["latest_inference"]
    # With 1 single turn, baseline sufficiency gate (>=3) is NOT satisfied -> sources is honestly []
    assert res_data["latest_inference"]["baseline_sources"] == []
    assert res_data["is_baseline_locked"] is False


def test_behavioral_batch_recording_multi_turn_segmentation(monkeypatch):
    """Verify that multi-segment batch recording splits into sequential turns,
    strictly enforces the production 60s/3-turn/15-word sufficiency gate,
    locks the baseline once all three conditions are met, and honestly populates baseline_sources.
    """
    client = TestClient(app)
    from copilot.calibration import SpeechToTextEngine

    # Realistic 65-second multi-turn sequence: 4 turns, 20 words across 67 seconds
    async def mock_multi_turn_transcribe(*args, **kwargs):
        return (
            "We want to sell. It was listed for four months. My two year old needs space. Can we meet tomorrow?",
            [
                # Turn 1: 1.0 - 4.0s (4 words)
                {"word": "We", "start": 1.0, "end": 1.5},
                {"word": "want", "start": 1.8, "end": 2.2},
                {"word": "to", "start": 2.4, "end": 2.8},
                {"word": "sell.", "punctuated_word": "sell.", "start": 3.0, "end": 4.0},
                # pause until 15s -> boundary 1
                # Turn 2: 15.0 - 20.0s (6 words)
                {"word": "It", "start": 15.0, "end": 15.5},
                {"word": "was", "start": 15.8, "end": 16.3},
                {"word": "listed", "start": 16.6, "end": 17.2},
                {"word": "for", "start": 17.5, "end": 18.0},
                {"word": "four", "start": 18.3, "end": 19.0},
                {"word": "months.", "punctuated_word": "months.", "start": 19.2, "end": 20.0},
                # pause until 40s -> boundary 2
                # Turn 3: 40.0 - 46.0s (6 words)
                {"word": "My", "start": 40.0, "end": 40.5},
                {"word": "two", "start": 40.8, "end": 41.5},
                {"word": "year", "start": 41.8, "end": 42.6},
                {"word": "old", "start": 43.0, "end": 43.7},
                {"word": "needs", "start": 44.0, "end": 44.8},
                {"word": "space.", "punctuated_word": "space.", "start": 45.1, "end": 46.0},
                # pause until 62s -> boundary 3
                # Turn 4: 62.0 - 67.0s (4 words) -> elapsed 67s - 1s = 66s >= 60s, words=20 >= 15
                {"word": "Can", "start": 62.0, "end": 62.8},
                {"word": "we", "start": 63.2, "end": 64.0},
                {"word": "meet", "start": 64.4, "end": 65.2},
                {"word": "tomorrow?", "punctuated_word": "tomorrow?", "start": 65.6, "end": 67.0},
            ]
        )
    monkeypatch.setattr(SpeechToTextEngine, "transcribe_with_timestamps", mock_multi_turn_transcribe)

    files = {"audio": ("test_multi.webm", b"RIFFFAKEMULTIAUDIO", "audio/webm")}
    data = {"prospect_id": "prospect_multi_turn_test"}
    post_resp = client.post("/api/test/analyze-recording", files=files, data=data)
    assert post_resp.status_code == 200
    res_data = post_resp.json()

    assert res_data["status"] == "success"
    assert res_data["turns_processed"] == 4
    assert len(res_data["utterances"]) == 4
    # With 4 clean turns, 20 words, and 66s elapsed, all 3 sufficiency conditions are satisfied!
    assert res_data["is_baseline_locked"] is True
    assert "intra_call" in res_data["latest_inference"]["baseline_sources"]
    assert len(res_data["latest_evidence_frame"]["windows"]["last_5_10s"]["deviations"]) > 0


def test_batch_recording_sufficiency_gate_rejects_insufficient_time_and_words(monkeypatch):
    """Verify that 3 short turns totaling only 20 seconds and 12 words STRICTLY REFUSE
    to lock the baseline under the production gate (requires >=60s elapsed AND >=15 cumulative words).
    """
    client = TestClient(app)
    from copilot.calibration import SpeechToTextEngine

    # 3 short turns across only 20 seconds, totaling 12 words
    async def mock_short_multi_turn(*args, **kwargs):
        return (
            "Hello who is calling. I am not sure. Call me next week.",
            [
                # Turn 1: 1.0 - 2.5s (4 words)
                {"word": "Hello", "start": 1.0, "end": 1.3},
                {"word": "who", "start": 1.4, "end": 1.7},
                {"word": "is", "start": 1.8, "end": 2.0},
                {"word": "calling.", "punctuated_word": "calling.", "start": 2.1, "end": 2.5},
                # Turn 2: 8.0 - 10.0s (4 words)
                {"word": "I", "start": 8.0, "end": 8.3},
                {"word": "am", "start": 8.4, "end": 8.7},
                {"word": "not", "start": 8.8, "end": 9.2},
                {"word": "sure.", "punctuated_word": "sure.", "start": 9.3, "end": 10.0},
                # Turn 3: 17.0 - 20.0s (4 words) -> total elapsed 20s - 1s = 19s (<60s), total words = 12 (<15)
                {"word": "Call", "start": 17.0, "end": 17.5},
                {"word": "me", "start": 17.8, "end": 18.2},
                {"word": "next", "start": 18.5, "end": 19.0},
                {"word": "week.", "punctuated_word": "week.", "start": 19.3, "end": 20.0},
            ]
        )
    monkeypatch.setattr(SpeechToTextEngine, "transcribe_with_timestamps", mock_short_multi_turn)

    files = {"audio": ("test_insufficient.webm", b"RIFFINSUFFICIENTBYTES", "audio/webm")}
    data = {"prospect_id": "prospect_insufficient_test"}
    post_resp = client.post("/api/test/analyze-recording", files=files, data=data)
    assert post_resp.status_code == 200
    res_data = post_resp.json()

    assert res_data["status"] == "success"
    assert res_data["turns_processed"] == 3
    assert len(res_data["utterances"]) == 3
    # Gate fails both time (<60s) and word count (<15) checks -> MUST NOT LOCK!
    assert res_data["is_baseline_locked"] is False
    assert res_data["latest_inference"]["baseline_sources"] == []
    assert len(res_data["latest_evidence_frame"]["windows"]["last_5_10s"]["deviations"]) == 0


