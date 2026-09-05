"""Unit tests for Custom Playbook Creator module (5-Step Wizard).

Verifies:
1. Pydantic models for Stage, BasicInfo, Style, Objections, ResponseSequence, and Playbook.
2. Playbook quality score calculation.
3. PlaybookStore persistence layer (save, get, list_all, delete).
4. FastAPI router endpoints (/templates/default, /list, /save, /{id}, /preview-prompt, delete).
5. Error cases and validation.
"""

import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot.playbook import (
    Playbook,
    PlaybookBasicInfo,
    PlaybookStyle,
    PlaybookStage,
    PlaybookObjection,
    ResponseSequenceStep,
    PlaybookStore,
    get_playbook_router,
    calculate_playbook_quality_score,
    DEFAULT_STAGES,
    DEFAULT_OBJECTIONS,
)


@pytest.fixture
def temp_store(tmp_path):
    """Fixture providing an isolated PlaybookStore using a temporary directory."""
    return PlaybookStore(storage_dir=tmp_path / "playbooks_test")


@pytest.fixture
def client(temp_store):
    """Fixture providing FastAPI test client connected to the isolated store."""
    app = FastAPI()
    router = get_playbook_router(temp_store)
    app.include_router(router)
    return TestClient(app)


def test_playbook_models_default_initialization():
    """Verify that default models initialize with valid default fields across 5 steps."""
    info = PlaybookBasicInfo(
        name="Daniel G. Method",
        short_description="My approach to building trust and closing more high-value deals.",
        industry="Real Estate",
        lead_types=["Expired, FSBO, Pre-Expired"],
        experience_level="Intermediate to Advanced",
        voice_phrases="Here's the thing\nBottom line",
        philosophy="1. Establish trust through diagnostic questioning. 2. Frame price in context of total value and net ROI.",
        goals=["Build stronger trust", "Increase appointments", "Close more deals"],
    )
    assert info.name == "Daniel G. Method"
    assert len(info.goals) == 3

    style = PlaybookStyle()
    assert style.overall_tone == "Confident"
    assert style.energy_level == "Medium"
    assert style.humor_level == "Light"

    pb = Playbook(basic_info=info, style=style)
    assert pb.id.startswith("pb_")
    assert pb.title == "Untitled Playbook"
    assert len(pb.stages) == 6
    assert len(pb.objections) == 8
    assert len(pb.response_sequence) == 4
    assert pb.stages[0].name == "Opening"
    assert pb.stages[0].icon == "hand"
    assert pb.objections[0].objection == "I don't want to pay commission."


def test_quality_score_calculation():
    """Verify the quality score calculation algorithm."""
    info = PlaybookBasicInfo(
        name="Daniel G. Method",
        short_description="My approach to building trust and closing more high-value deals.",
        industry="Real Estate",
        lead_types=["Expired, FSBO, Pre-Expired"],
        experience_level="Intermediate to Advanced",
        voice_phrases="Here's the thing\nBottom line",
        philosophy="1. Establish trust through diagnostic questioning. 2. Frame price in context of total value and net ROI.",
        goals=["Build stronger trust", "Increase appointments", "Close more deals"],
    )
    pb = Playbook(basic_info=info)
    score = calculate_playbook_quality_score(pb)
    assert score >= 90
    assert score <= 100


def test_playbook_store_save_get_and_list(temp_store):
    """Test saving a playbook, retrieving it, and listing all playbooks."""
    info = PlaybookBasicInfo(
        name="Enterprise Closer",
        industry="B2B Enterprise Services",
    )
    pb = Playbook(id="pb_test_123", basic_info=info)
    
    saved = temp_store.save(pb)
    assert saved.id == "pb_test_123"
    assert saved.title == "Enterprise Closer"
    assert saved.quality_score >= 20

    # Retrieve by ID
    retrieved = temp_store.get("pb_test_123")
    assert retrieved is not None
    assert retrieved.id == "pb_test_123"
    assert retrieved.basic_info.name == "Enterprise Closer"
    assert len(retrieved.stages) == 6
    assert len(retrieved.objections) == 8

    # List all
    all_pbs = temp_store.list_all()
    assert len(all_pbs) == 1
    assert all_pbs[0]["id"] == "pb_test_123"
    assert all_pbs[0]["title"] == "Enterprise Closer"
    assert all_pbs[0]["objections_count"] == 8


def test_playbook_store_delete(temp_store):
    """Test deleting a playbook."""
    info = PlaybookBasicInfo(name="Temporary Playbook")
    pb = Playbook(id="pb_delete_me", basic_info=info)
    temp_store.save(pb)

    assert temp_store.get("pb_delete_me") is not None
    deleted = temp_store.delete("pb_delete_me")
    assert deleted is True
    assert temp_store.get("pb_delete_me") is None
    assert temp_store.delete("pb_non_existent") is False


def test_api_templates_default(client):
    """Test the /api/playbook/templates/default endpoint."""
    res = client.get("/api/playbook/templates/default")
    assert res.status_code == 200
    data = res.json()
    assert "stages" in data
    assert len(data["stages"]) == 6
    assert "objections" in data
    assert len(data["objections"]) == 8
    assert "response_sequence" in data
    assert len(data["response_sequence"]) == 4
    assert "sample_voices" in data
    assert "industries" in data
    assert "default_tones" in data


def test_api_preview_prompt(client):
    """Test dynamic live coaching prompt preview generation."""
    payload = {
        "name": "Daniel",
        "overall_tone": "Confident",
        "communication_style": "Consultative",
        "sentence_style": "Short & Conversational",
        "formality": "Professional",
    }
    res = client.post("/api/playbook/preview-prompt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "preview_text" in data
    assert "Hey Daniel" in data["preview_text"]
    assert "confident" in data["preview_text"]
    assert "consultative" in data["preview_text"]


def test_api_save_and_retrieve_playbook(client):
    """Test saving a new playbook via API with objections and getting it."""
    payload = {
        "basic_info": {
            "name": "Daniel G. Method",
            "short_description": "Convert inbound enterprise demos and expireds.",
            "industry": "Real Estate",
            "lead_types": ["Expired", "FSBO"],
            "experience_level": "Intermediate to Advanced",
            "voice_phrases": "Here's the thing",
            "philosophy": "Deliver value upfront and net more equity.",
            "goals": ["Build stronger trust", "Increase appointments", "Close more deals"],
        },
        "style": {
            "overall_tone": "Confident",
            "communication_style": "Consultative",
            "energy_level": "High",
            "sentence_style": "Short & Conversational",
            "formality": "Professional",
            "humor_level": "Light",
        },
        "stages": [
            {
                "id": "stg_1",
                "order": 1,
                "name": "Opening",
                "goal": "Grab attention in 15 seconds.",
                "description": "Establish immediate credibility.",
                "icon": "hand",
            }
        ],
        "objections": [
            {
                "id": "obj_custom_1",
                "objection": "I don't want to pay commission.",
                "category": "Pricing Objection",
                "lead_types": ["FSBO"],
                "response_style": "Value & ROI Focus",
                "ai_suggestion": "Highlight net proceeds.",
                "last_updated": "May 2, 2024",
            }
        ],
    }

    # Save
    res = client.post("/api/playbook/save", json=payload)
    assert res.status_code == 200
    save_data = res.json()
    assert save_data["status"] == "success"
    pb_id = save_data["playbook"]["id"]
    assert pb_id.startswith("pb_")
    assert save_data["playbook"]["title"] == "Daniel G. Method"

    # Get
    res_get = client.get(f"/api/playbook/{pb_id}")
    assert res_get.status_code == 200
    got_pb = res_get.json()["playbook"]
    assert got_pb["id"] == pb_id
    assert len(got_pb["objections"]) == 1
    assert got_pb["objections"][0]["objection"] == "I don't want to pay commission."

    # List
    res_list = client.get("/api/playbook/list")
    assert res_list.status_code == 200
    assert len(res_list.json()["playbooks"]) == 1
    assert res_list.json()["playbooks"][0]["objections_count"] == 1


def test_serialize_playbook_prompt_section():
    """Verify that structured playbook data gets serialized into the labeled prompt block."""
    from copilot.playbook import serialize_playbook_prompt_section
    info = PlaybookBasicInfo(
        name="Daniel G. Method",
        industry="Real Estate",
        lead_types=["Expired Listings", "FSBO (For Sale By Owner)"],
        philosophy="1. Establish trust. 2. Frame price as net ROI.",
        voice_phrases="Here's the thing\nBottom line",
    )
    pb = Playbook(title="Daniel G. Method", basic_info=info)

    prompt_section = serialize_playbook_prompt_section(
        pb,
        current_lead_type="Expired Listings",
        detected_objection="I don't want to pay commission."
    )

    assert "### ACTIVE PLAYBOOK METHODOLOGY LENS: DANIEL G. METHOD ###" in prompt_section
    assert "Coaching Style: Tone=Confident" in prompt_section
    assert "Objection Response Sequence: Acknowledge -> Validate -> Reframe -> Guide Forward" in prompt_section
    assert "Matched Objection Rule" in prompt_section
    assert "I don't want to pay commission." in prompt_section
    assert "Quantify" in prompt_section
    assert "Do / Don't Boundaries" in prompt_section
