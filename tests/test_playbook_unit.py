"""Unit tests for Custom Playbook Creator module.

Verifies:
1. Pydantic models for Stage, BasicInfo, Style, and Playbook.
2. PlaybookStore persistence layer (save, get, list_all, delete).
3. FastAPI router endpoints (/templates/default, /list, /save, /{id}, delete).
4. Error cases and validation.
"""

import json
import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot.playbook import (
    Playbook,
    PlaybookBasicInfo,
    PlaybookStyle,
    PlaybookStage,
    PlaybookStore,
    get_playbook_router,
    DEFAULT_STAGES,
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
    """Verify that default models initialize with valid default fields."""
    info = PlaybookBasicInfo(
        name="Daniel G. Method",
        short_description="My approach to building trust and closing more high-value deals.",
        industry="SaaS / Software",
        lead_types=["Inbound Qualified Leads"],
        experience_level="Intermediate (2-4 yrs)",
        voice_phrases="Here's the thing\nBottom line",
        philosophy="Help first, sell second.",
        goals=["Build stronger trust", "Close more deals"],
    )
    assert info.name == "Daniel G. Method"
    assert len(info.goals) == 2

    style = PlaybookStyle()
    assert style.overall_tone == "Confident"
    assert style.energy_level == "Medium"
    assert style.humor_level == "Light"

    pb = Playbook(basic_info=info, style=style)
    assert pb.id.startswith("pb_")
    assert pb.title == "Untitled Playbook"
    assert len(pb.stages) == 6
    assert pb.stages[0].name == "Opening"
    assert pb.stages[0].icon == "hand"


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

    # Retrieve by ID
    retrieved = temp_store.get("pb_test_123")
    assert retrieved is not None
    assert retrieved.id == "pb_test_123"
    assert retrieved.basic_info.name == "Enterprise Closer"
    assert len(retrieved.stages) == 6

    # List all
    all_pbs = temp_store.list_all()
    assert len(all_pbs) == 1
    assert all_pbs[0]["id"] == "pb_test_123"
    assert all_pbs[0]["title"] == "Enterprise Closer"


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
    assert "sample_voices" in data
    assert "industries" in data
    assert "default_tones" in data


def test_api_save_and_retrieve_playbook(client):
    """Test saving a new playbook via API and getting it."""
    payload = {
        "basic_info": {
            "name": "High Ticket Closer",
            "short_description": "Convert inbound enterprise demos.",
            "industry": "SaaS / Software",
            "lead_types": ["Inbound Qualified Leads"],
            "experience_level": "Senior Closer (5+ yrs)",
            "voice_phrases": "Here's the thing",
            "philosophy": "Deliver value upfront.",
            "goals": ["Build stronger trust", "Close more deals"],
        },
        "style": {
            "overall_tone": "Direct",
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
                "name": "Hook & Intro",
                "goal": "Grab attention in 15 seconds.",
                "description": "Establish immediate credibility.",
                "icon": "hand",
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
    assert save_data["playbook"]["title"] == "High Ticket Closer"

    # Get
    res_get = client.get(f"/api/playbook/{pb_id}")
    assert res_get.status_code == 200
    got_pb = res_get.json()["playbook"]
    assert got_pb["id"] == pb_id
    assert got_pb["style"]["overall_tone"] == "Direct"
    assert len(got_pb["stages"]) == 1
    assert got_pb["stages"][0]["name"] == "Hook & Intro"

    # List
    res_list = client.get("/api/playbook/list")
    assert res_list.status_code == 200
    assert len(res_list.json()["playbooks"]) == 1

    # Delete
    res_del = client.delete(f"/api/playbook/{pb_id}")
    assert res_del.status_code == 200
    assert client.get(f"/api/playbook/{pb_id}").status_code == 404
