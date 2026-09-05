"""Custom Playbook Module for Sales AI.

Provides data models, JSON file persistence, and FastAPI router for
creating and managing custom sales playbooks based on a 3-step wizard workflow:
1. Basic Information (Industry, lead types, voice, core philosophy, goals)
2. Your Style (Tone, communication style, energy, sentence style, formality, humor)
3. Conversation Flow (Configurable stage timeline with Add/Edit Stage modal)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("copilot.playbook")
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PLAYBOOKS_DIR = WORKSPACE_DIR / "playbooks"
STATIC_DIR = WORKSPACE_DIR / "web"

PLAYBOOKS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Default Stages & Data Templates
# ==========================================

DEFAULT_STAGES: List[Dict[str, Any]] = [
    {
        "id": "stage_opening",
        "order": 1,
        "name": "Opening",
        "goal": "Set a positive tone and gain engagement.",
        "description": "Create rapport and get permission to have a conversation.",
        "icon": "hand",
    },
    {
        "id": "stage_discovery",
        "order": 2,
        "name": "Discovery",
        "goal": "Understand their motivation and timeline.",
        "description": "Ask questions and uncover their situation and needs.",
        "icon": "search",
    },
    {
        "id": "stage_value_prop",
        "order": 3,
        "name": "Value Proposition",
        "goal": "Build credibility and differentiate yourself.",
        "description": "Position your value and show how you solve their problem.",
        "icon": "diamond",
    },
    {
        "id": "stage_objections",
        "order": 4,
        "name": "Handle Objections",
        "goal": "Overcome doubt and strengthen confidence.",
        "description": "Address concerns and remove hesitation.",
        "icon": "shield",
    },
    {
        "id": "stage_close",
        "order": 5,
        "name": "Close",
        "goal": "Get a commitment to move forward.",
        "description": "Guide them to commit to the next step.",
        "icon": "target",
    },
    {
        "id": "stage_follow_up",
        "order": 6,
        "name": "Follow Up",
        "goal": "Stay top of mind and convert later.",
        "description": "Reinforce value and keep the momentum.",
        "icon": "phone",
    },
]


# ==========================================
# 2. Pydantic Models
# ==========================================

class PlaybookStage(BaseModel):
    id: str = Field(default_factory=lambda: f"stage_{uuid.uuid4().hex[:8]}")
    order: int = 1
    name: str = Field(..., min_length=1, max_length=100)
    goal: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1500)
    icon: str = Field("hand", description="hand | search | diamond | shield | target | phone")


class PlaybookBasicInfo(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    short_description: str = Field("", max_length=120)
    industry: str = Field("", max_length=100)
    lead_types: List[str] = Field(default_factory=list)
    experience_level: str = Field("", max_length=80)
    voice_phrases: str = Field("", max_length=500)
    philosophy: str = Field("", max_length=1000)
    goals: List[str] = Field(default_factory=list)


class PlaybookStyle(BaseModel):
    overall_tone: str = Field("Confident", max_length=50)
    communication_style: str = Field("Consultative", max_length=80)
    energy_level: str = Field("Medium", description="Low | Medium | High")
    sentence_style: str = Field("Short & Conversational", max_length=80)
    formality: str = Field("Professional", max_length=80)
    humor_level: str = Field("Light", description="None | Light | Moderate")
    voice_phrases: Optional[str] = Field(None, max_length=500)
    core_selling_principles: Optional[str] = Field(None, max_length=1000)


class Playbook(BaseModel):
    id: str = Field(default_factory=lambda: f"pb_{uuid.uuid4().hex[:12]}")
    title: str = Field("Untitled Playbook")
    status: str = Field("draft", description="draft | published")
    current_step: int = Field(1, ge=1, le=4)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    basic_info: PlaybookBasicInfo
    style: PlaybookStyle = Field(default_factory=PlaybookStyle)
    stages: List[PlaybookStage] = Field(default_factory=lambda: [PlaybookStage(**s) for s in DEFAULT_STAGES])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlaybookSaveRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = "draft"
    current_step: Optional[int] = 1
    basic_info: PlaybookBasicInfo
    style: Optional[PlaybookStyle] = None
    stages: Optional[List[PlaybookStage]] = None
    metadata: Optional[Dict[str, Any]] = None


# ==========================================
# 3. Persistence Manager
# ==========================================

class PlaybookStore:
    """Manages local JSON file persistence for playbooks."""

    def __init__(self, storage_dir: Path = PLAYBOOKS_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, playbook_id: str) -> Path:
        safe_id = re.sub(r"[^\w\-]", "", playbook_id)
        return self.storage_dir / f"{safe_id}.json"

    def save(self, playbook: Playbook) -> Playbook:
        playbook.updated_at = datetime.now(timezone.utc).isoformat()
        if not playbook.title and playbook.basic_info.name:
            playbook.title = playbook.basic_info.name
        elif playbook.basic_info.name:
            playbook.title = playbook.basic_info.name

        file_path = self._file_path(playbook.id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(playbook.model_dump(), f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved playbook %s to %s", playbook.id, file_path)
        return playbook

    def get(self, playbook_id: str) -> Optional[Playbook]:
        file_path = self._file_path(playbook_id)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Playbook(**data)
        except Exception as e:
            LOGGER.error("Error reading playbook %s: %s", playbook_id, e)
            return None

    def list_all(self) -> List[Dict[str, Any]]:
        playbooks = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                playbooks.append({
                    "id": data.get("id"),
                    "title": data.get("title") or data.get("basic_info", {}).get("name", "Untitled Playbook"),
                    "status": data.get("status", "draft"),
                    "current_step": data.get("current_step", 1),
                    "industry": data.get("basic_info", {}).get("industry", ""),
                    "stages_count": len(data.get("stages", [])),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                })
            except Exception as e:
                LOGGER.warning("Could not read playbook file %s: %s", file, e)
        # Sort by updated_at descending
        playbooks.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return playbooks

    def delete(self, playbook_id: str) -> bool:
        file_path = self._file_path(playbook_id)
        if file_path.exists():
            file_path.unlink()
            LOGGER.info("Deleted playbook %s", playbook_id)
            return True
        return False


# ==========================================
# 4. FastAPI Router
# ==========================================

def get_playbook_router(store: Optional[PlaybookStore] = None) -> APIRouter:
    """Build and return FastAPI APIRouter for Playbook endpoints."""
    if store is None:
        store = PlaybookStore()

    router = APIRouter(prefix="/api/playbook", tags=["Playbook"])

    @router.get("/templates/default")
    async def get_default_template():
        """Return the default starting template for new playbooks."""
        return {
            "stages": DEFAULT_STAGES,
            "sample_voices": [
                "Here's the thing",
                "Bottom line",
                "What I've seen is...",
                "At the end of the day",
            ],
            "default_tones": [
                "Confident",
                "Friendly",
                "Calm",
                "Direct",
                "Empathetic",
                "Authoritative",
            ],
            "default_communication_styles": [
                "Consultative",
                "Challenger",
                "Solution-Oriented",
                "Relationship-Focused",
                "Direct / High-Urgency",
                "Educational / Advisory",
            ],
            "default_sentence_styles": [
                "Short & Conversational",
                "Detailed & Structured",
                "Punchy & Direct",
                "Storytelling & Metaphorical",
            ],
            "default_formality_levels": [
                "Casual & Conversational",
                "Professional",
                "Executive & Formal",
            ],
            "industries": [
                "SaaS / Software",
                "Real Estate",
                "Financial Services & Insurance",
                "Healthcare & Medical",
                "Consulting & Agency",
                "E-commerce & Retail",
                "B2B Enterprise Services",
                "General Sales",
            ],
            "lead_types": [
                "Inbound Qualified Leads",
                "Cold Outreach / Outbound",
                "Referrals & Warm Intros",
                "Enterprise Decision Makers",
                "SMB Owners",
                "High-Ticket B2C",
            ],
            "experience_levels": [
                "Entry Level (0-1 yrs)",
                "Intermediate (2-4 yrs)",
                "Senior Closer (5+ yrs)",
                "Master / Executive Level",
            ],
        }

    @router.get("/list")
    async def list_playbooks():
        """List all saved playbooks."""
        return {"playbooks": store.list_all()}

    @router.get("/{playbook_id}")
    async def get_playbook(playbook_id: str):
        """Retrieve a specific playbook by ID."""
        playbook = store.get(playbook_id)
        if not playbook:
            raise HTTPException(status_code=404, detail="Playbook not found")
        return {"playbook": playbook.model_dump()}

    @router.post("/save")
    async def save_playbook(req: PlaybookSaveRequest):
        """Create or update a playbook."""
        existing = store.get(req.id) if req.id else None
        
        stages_data = req.stages if req.stages is not None else (
            existing.stages if existing else [PlaybookStage(**s) for s in DEFAULT_STAGES]
        )
        
        # Ensure stages have sequential order
        for idx, stage in enumerate(stages_data, start=1):
            stage.order = idx
            
        style_data = req.style if req.style is not None else (
            existing.style if existing else PlaybookStyle()
        )

        playbook = Playbook(
            id=req.id or (existing.id if existing else f"pb_{uuid.uuid4().hex[:12]}"),
            title=req.title or req.basic_info.name,
            status=req.status or (existing.status if existing else "draft"),
            current_step=req.current_step or (existing.current_step if existing else 1),
            created_at=existing.created_at if existing else datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            basic_info=req.basic_info,
            style=style_data,
            stages=stages_data,
            metadata=req.metadata or (existing.metadata if existing else {}),
        )

        saved = store.save(playbook)
        return {
            "status": "success",
            "message": "Playbook saved successfully",
            "playbook": saved.model_dump(),
        }

    @router.delete("/{playbook_id}")
    async def delete_playbook(playbook_id: str):
        """Delete a playbook by ID."""
        deleted = store.delete(playbook_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Playbook not found")
        return {"status": "success", "message": f"Playbook {playbook_id} deleted"}

    return router


# ==========================================
# 5. Standalone App Factory
# ==========================================

def create_standalone_playbook_app() -> FastAPI:
    """Create a standalone FastAPI app for the Custom Playbook Creator."""
    app = FastAPI(title="PitchProX Custom Playbook Studio", version="1.0.0")

    store = PlaybookStore()
    router = get_playbook_router(store)
    app.include_router(router)

    @app.get("/")
    async def root_view():
        html_file = STATIC_DIR / "playbook.html"
        if not html_file.exists():
            raise HTTPException(status_code=404, detail="Playbook UI file not found")
        return FileResponse(html_file)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": "PitchProX Custom Playbook Studio",
            "playbooks_count": len(store.list_all()),
        }

    return app
