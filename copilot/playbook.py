"""Custom Playbook Module for Sales AI.

Provides data models, JSON file persistence, and FastAPI router for
creating and managing custom sales playbooks according to the PitchProX
Playbook Integration Specification (Version 1.0):

- Playbook as a methodology lens over the Core engine.
- Full 16-prospect lead types catalog.
- Controlled Objection Library with 4 canonical categories (Financial, Risk, Logistics, Relationship)
  and 11 standardized response styles (Reframe, Clarify, Validate, Educate, Quantify, Differentiate,
  De-Risk, Challenge, Social Proof, Future Pace, Direct).
- Runtime sensitivity & threshold controls (Prompt Timing, Emotional Sensitivity, Objection Confidence,
  Trust Alert Threshold, Prompt Cooldown, Do/Don't Boundaries, Language Rules).
- 5-step wizard workflow (Basic Info, Your Style, Conversation Flow, Objection Handling, Review & Save).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("copilot.playbook")
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PLAYBOOKS_DIR = WORKSPACE_DIR / "playbooks"
STATIC_DIR = WORKSPACE_DIR / "web"

PLAYBOOKS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Standard Lead Types & Controlled Catalogs
# ==========================================

STANDARDIZED_LEAD_TYPES: List[Dict[str, str]] = [
    {"name": "Expired Listings", "desc": "Listings that recently expired without selling."},
    {"name": "Pre-Expired Listings", "desc": "Listings approaching expiration."},
    {"name": "FSBO (For Sale By Owner)", "desc": "Owners selling without an agent."},
    {"name": "Absentee Owners", "desc": "Owners who do not occupy the property (often investors)."},
    {"name": "Circle Prospecting", "desc": "Homeowners near a listing or recent sale."},
    {"name": "Just Listed", "desc": "Prospecting around a newly listed property."},
    {"name": "Just Sold", "desc": "Prospecting around a recently sold property."},
    {"name": "Probate", "desc": "Properties involved in probate or estate administration."},
    {"name": "Trust & Estate", "desc": "Trust-owned properties and estate planning situations."},
    {"name": "Pre-Foreclosure", "desc": "Owners showing signs of mortgage distress before foreclosure."},
    {"name": "Foreclosure / REO", "desc": "Bank-owned or foreclosure-related opportunities."},
    {"name": "Divorce", "desc": "Homeowners involved in divorce proceedings."},
    {"name": "Tax Delinquent", "desc": "Owners with delinquent property taxes."},
    {"name": "Vacant Properties", "desc": "Homes appearing vacant or abandoned."},
    {"name": "Investor / Cash Buyers", "desc": "Investors purchasing for rental, flip, or portfolio growth."},
    {"name": "Past Clients & Sphere", "desc": "Previous clients, referrals, and sphere of influence follow-up."},
]

CANONICAL_CATEGORIES: List[str] = [
    "Financial",
    "Risk",
    "Logistics",
    "Relationship",
]

STANDARDIZED_RESPONSE_STYLES: List[str] = [
    "Reframe",
    "Clarify",
    "Validate",
    "Educate",
    "Quantify",
    "Differentiate",
    "De-Risk",
    "Challenge",
    "Social Proof",
    "Future Pace",
    "Direct",
]

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

DEFAULT_RESPONSE_SEQUENCE: List[Dict[str, Any]] = [
    {"id": "seq_1", "order": 1, "name": "Acknowledge", "icon": "mic"},
    {"id": "seq_2", "order": 2, "name": "Validate", "icon": "heart"},
    {"id": "seq_3", "order": 3, "name": "Reframe", "icon": "refresh"},
    {"id": "seq_4", "order": 4, "name": "Guide Forward", "icon": "file"},
]

DEFAULT_OBJECTIONS: List[Dict[str, Any]] = [
    {
        "id": "obj_1",
        "objection": "I don't want to pay commission.",
        "category": "Financial",
        "lead_types": ["FSBO (For Sale By Owner)", "Expired Listings", "Pre-Expired Listings"],
        "response_style": "Quantify",
        "ai_suggestion": 'This "Quantify" reframe highlights net take-home equity. Mention that MLS-represented listings in this area closed on average 7.2% higher.',
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_2",
        "objection": "Not a good time to sell.",
        "category": "Logistics",
        "lead_types": ["FSBO (For Sale By Owner)", "Expired Listings", "Absentee Owners"],
        "response_style": "Future Pace",
        "ai_suggestion": "Validate market uncertainty, then project seasonal equity growth to establish timeline clarity.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_3",
        "objection": "I already have an agent.",
        "category": "Relationship",
        "lead_types": ["Expired Listings", "Pre-Expired Listings"],
        "response_style": "Differentiate",
        "ai_suggestion": "Respect loyalty, then ask diagnostic questions about current marketing reach and recent buyer feedback.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_4",
        "objection": "How did you get my number?",
        "category": "Risk",
        "lead_types": ["FSBO (For Sale By Owner)", "Circle Prospecting", "Absentee Owners"],
        "response_style": "Direct",
        "ai_suggestion": "Be transparent about public record property rolls, then immediately explain why active local buyers asked about their property.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_5",
        "objection": "I'm not interested.",
        "category": "Relationship",
        "lead_types": ["Expired Listings", "Past Clients & Sphere", "Circle Prospecting"],
        "response_style": "Validate",
        "ai_suggestion": "Disarm friction with low-pressure validation: 'Understood, just curious if timing changed or if you already resolved the property need?'",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_6",
        "objection": "We want to wait for interest rates to drop.",
        "category": "Financial",
        "lead_types": ["Investor / Cash Buyers", "Pre-Foreclosure", "Absentee Owners"],
        "response_style": "Reframe",
        "ai_suggestion": "Reframe the trade-off between rate drops and inventory surges that drive acquisition competition.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_7",
        "objection": "We need to think about it.",
        "category": "Logistics",
        "lead_types": ["Trust & Estate", "Probate", "Divorce"],
        "response_style": "Clarify",
        "ai_suggestion": "Clarify whether it is the timeline, legal estate steps, or family consensus holding back the next milestone.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_8",
        "objection": "Send me information first.",
        "category": "Risk",
        "lead_types": ["Just Listed", "Just Sold", "Circle Prospecting"],
        "response_style": "De-Risk",
        "ai_suggestion": "Offer a 1-page local market snapshot while locking in a 5-minute calendar check-in next Tuesday.",
        "last_updated": "May 2, 2024",
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


class PlaybookObjection(BaseModel):
    id: str = Field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:8]}")
    objection: str = Field(..., min_length=1, max_length=300)
    category: str = Field("Financial", description="Financial | Risk | Logistics | Relationship")
    lead_types: List[str] = Field(default_factory=lambda: ["Expired Listings", "FSBO (For Sale By Owner)"])
    response_style: str = Field("Reframe", description="Reframe | Clarify | Validate | Educate | Quantify | Differentiate | De-Risk | Challenge | Social Proof | Future Pace | Direct")
    ai_suggestion: str = Field("", max_length=500)
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%b %d, %Y"))


class ResponseSequenceStep(BaseModel):
    id: str = Field(default_factory=lambda: f"seq_{uuid.uuid4().hex[:6]}")
    order: int = 1
    name: str = Field(..., min_length=1, max_length=60)
    icon: str = Field("mic", max_length=30)


class PlaybookRuntimeSettings(BaseModel):
    """Runtime sensitivity and boundary rules from Playbook Specification Section 3."""
    prompt_timing_sensitivity: str = Field("Normal", description="Normal | Aggressive | Relaxed")
    emotional_sensitivity: str = Field("Normal", description="Normal | High | Adaptive")
    objection_confidence_threshold: float = Field(0.75, ge=0.0, le=1.0)
    trust_alert_threshold: float = Field(0.70, ge=0.0, le=1.0)
    prompt_cooldown_seconds: int = Field(4, ge=1, le=30)
    do_dont_boundaries: List[str] = Field(default_factory=lambda: [
        "Do validate emotion before offering facts.",
        "Do keep teleprompts under 25 words.",
        "Do not offer unauthorized price cuts.",
        "Do not disparage competitors or existing agents."
    ])
    language_rules: List[str] = Field(default_factory=lambda: [
        "Use: 'net proceeds', 'timeline clarity', 'direct buyers'",
        "Avoid: 'cheap', 'hurry up', 'standard commission'"
    ])


class PlaybookBasicInfo(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    short_description: str = Field("", max_length=120)
    industry: str = Field("Real Estate", max_length=100)
    lead_types: List[str] = Field(default_factory=lambda: ["Expired Listings", "FSBO (For Sale By Owner)", "Pre-Expired Listings"])
    experience_level: str = Field("Intermediate to Advanced", max_length=80)
    voice_phrases: str = Field("", max_length=500)
    philosophy: str = Field("", max_length=1000)
    goals: List[str] = Field(default_factory=lambda: ["Build stronger trust", "Increase appointments", "Close more deals"])


class PlaybookStyle(BaseModel):
    overall_tone: str = Field("Confident", max_length=50)
    communication_style: str = Field("Consultative", max_length=80)
    energy_level: str = Field("Medium", description="Low | Medium | High")
    sentence_style: str = Field("Short & Conversational", max_length=80)
    formality: str = Field("Professional", max_length=80)
    humor_level: str = Field("Light", description="None | Light | Moderate")
    voice_phrases: Optional[str] = Field(None, max_length=500)
    core_selling_principles: Optional[str] = Field(None, max_length=1000)
    runtime_settings: PlaybookRuntimeSettings = Field(default_factory=PlaybookRuntimeSettings)


class Playbook(BaseModel):
    id: str = Field(default_factory=lambda: f"pb_{uuid.uuid4().hex[:12]}")
    title: str = Field("Untitled Playbook")
    status: str = Field("draft", description="draft | published")
    current_step: int = Field(1, ge=1, le=5)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    basic_info: PlaybookBasicInfo
    style: PlaybookStyle = Field(default_factory=PlaybookStyle)
    stages: List[PlaybookStage] = Field(default_factory=lambda: [PlaybookStage(**s) for s in DEFAULT_STAGES])
    objections: List[PlaybookObjection] = Field(default_factory=lambda: [PlaybookObjection(**o) for o in DEFAULT_OBJECTIONS])
    response_sequence: List[ResponseSequenceStep] = Field(default_factory=lambda: [ResponseSequenceStep(**sq) for sq in DEFAULT_RESPONSE_SEQUENCE])
    quality_score: int = Field(94, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlaybookSaveRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = "draft"
    current_step: Optional[int] = 1
    basic_info: PlaybookBasicInfo
    style: Optional[PlaybookStyle] = None
    stages: Optional[List[PlaybookStage]] = None
    objections: Optional[List[PlaybookObjection]] = None
    response_sequence: Optional[List[ResponseSequenceStep]] = None
    quality_score: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


# ==========================================
# 3. Runtime Methodology Lens Constructor
# ==========================================

def build_playbook_methodology_prompt(playbook: Playbook, live_context: Optional[Dict[str, Any]] = None) -> str:
    """Build the runtime methodology lens prompt that wraps around PitchProX Core intelligence.

    According to the Playbook Integration Spec:
    - Does NOT replace Core facts, safety, or compliance.
    - Guides tone, response sequence, objection framing, and prompt brevity.
    """
    tone = playbook.style.overall_tone
    comm_style = playbook.style.communication_style
    energy = playbook.style.energy_level
    sentence = playbook.style.sentence_style
    formality = playbook.style.formality
    humor = playbook.style.humor_level
    philosophy = playbook.basic_info.philosophy or "Focus on client value and clear next steps."
    voice = playbook.basic_info.voice_phrases or ""

    seq_order = " -> ".join([s.name for s in sorted(playbook.response_sequence, key=lambda x: x.order)])
    do_dont = "\n".join([f"- {rule}" for rule in playbook.style.runtime_settings.do_dont_boundaries])

    prompt = (
        f"=== METHODOLOGY LENS: {playbook.title.upper()} ===\n"
        f"Role: Act as the real-time live teleprompter coach applying the '{playbook.title}' sales methodology.\n"
        f"Philosophy: {philosophy}\n"
        f"Coaching Tone: {tone} | Style: {comm_style} | Energy: {energy} | Sentence Style: {sentence} | Formality: {formality} | Humor: {humor}\n"
        f"Objection Handling Framework Sequence: {seq_order}\n"
        f"Boundaries & Do/Don'ts:\n{do_dont}\n"
    )
    if voice:
        prompt += f"Signature Phrases/Voice Style:\n{voice}\n"

    prompt += (
        "\nCRITICAL OPERATIONAL RULES:\n"
        "1. Output exactly ONE live teleprompter response for the salesperson to deliver right now.\n"
        "2. Keep the prompt punchy, conversational, and under 25 words.\n"
        "3. Core safety, factual truth, and live conversational reality ALWAYS take precedence over static rules.\n"
    )
    return prompt


# ==========================================
# 4. Quality Score Calculation Helper
# ==========================================

def calculate_playbook_quality_score(pb: Playbook) -> int:
    """Compute 0-100 quality score based on completeness."""
    score = 0
    if pb.basic_info.name and len(pb.basic_info.name.strip()) > 3:
        score += 8
    if pb.basic_info.short_description and len(pb.basic_info.short_description.strip()) > 10:
        score += 6
    if pb.basic_info.industry:
        score += 5
    if pb.basic_info.lead_types:
        score += 5
    if pb.basic_info.goals:
        score += 6

    if pb.basic_info.voice_phrases and len(pb.basic_info.voice_phrases.strip()) > 5:
        score += 10
    if pb.basic_info.philosophy and len(pb.basic_info.philosophy.strip()) > 10:
        score += 10

    if pb.stages:
        if len(pb.stages) >= 6:
            score += 25
        elif len(pb.stages) >= 3:
            score += 15
        else:
            score += 8

    if pb.objections:
        if len(pb.objections) >= 5:
            score += 25
        elif len(pb.objections) >= 2:
            score += 15
        else:
            score += 8

    return min(max(score, 20), 100)


# ==========================================
# 5. Persistence Manager
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

        playbook.quality_score = calculate_playbook_quality_score(playbook)

        file_path = self._file_path(playbook.id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(playbook.model_dump(), f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved playbook %s (Quality Score: %s) to %s", playbook.id, playbook.quality_score, file_path)
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
                    "objections_count": len(data.get("objections", [])),
                    "quality_score": data.get("quality_score", 94),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                })
            except Exception as e:
                LOGGER.warning("Could not read playbook file %s: %s", file, e)
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
# 6. FastAPI Router
# ==========================================

def get_playbook_router(store: Optional[PlaybookStore] = None) -> APIRouter:
    """Build and return FastAPI APIRouter for Playbook endpoints."""
    if store is None:
        store = PlaybookStore()

    router = APIRouter(prefix="/api/playbook", tags=["Playbook"])

    @router.get("/templates/default")
    async def get_default_template():
        """Return the default starting template with the full 16 lead types and canonical catalogs."""
        return {
            "stages": DEFAULT_STAGES,
            "objections": DEFAULT_OBJECTIONS,
            "response_sequence": DEFAULT_RESPONSE_SEQUENCE,
            "lead_types": STANDARDIZED_LEAD_TYPES,
            "canonical_categories": CANONICAL_CATEGORIES,
            "response_styles": STANDARDIZED_RESPONSE_STYLES,
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
                "Real Estate",
                "SaaS / Software",
                "Financial Services & Insurance",
                "Healthcare & Medical",
                "Consulting & Agency",
                "E-commerce & Retail",
                "B2B Enterprise Services",
                "General Sales",
            ],
            "experience_levels": [
                "Entry Level (0-1 yrs)",
                "Intermediate (2-4 yrs)",
                "Intermediate to Advanced",
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

    @router.post("/preview-prompt")
    async def preview_ai_prompt(payload: Dict[str, Any]):
        """Generate live coaching prompt preview based on tone, style, and identity."""
        name = payload.get("name", "Daniel")
        tone = payload.get("overall_tone", "Confident")
        style = payload.get("communication_style", "Consultative")
        sentence = payload.get("sentence_style", "Short & Conversational")
        formality = payload.get("formality", "Professional")

        preview_text = (
            f"Hey {name}, I've got you. "
            f"I'll keep my prompts {tone.lower()}, direct, and {style.lower()} — "
            f"{sentence.lower()}, {formality.lower()}, and to the point. "
            "Let's build a playbook that wins more conversations."
        )
        return {"preview_text": preview_text}

    @router.get("/{playbook_id}/runtime-prompt")
    async def get_runtime_methodology_prompt(playbook_id: str):
        """Build runtime system instruction for the LLM based on the active playbook."""
        playbook = store.get(playbook_id)
        if not playbook:
            raise HTTPException(status_code=404, detail="Playbook not found")
        prompt = build_playbook_methodology_prompt(playbook)
        return {"playbook_id": playbook_id, "methodology_prompt": prompt}

    @router.post("/save")
    async def save_playbook(req: PlaybookSaveRequest):
        """Create or update a playbook."""
        existing = store.get(req.id) if req.id else None

        stages_data = req.stages if req.stages is not None else (
            existing.stages if existing else [PlaybookStage(**s) for s in DEFAULT_STAGES]
        )
        for idx, stage in enumerate(stages_data, start=1):
            stage.order = idx

        objections_data = req.objections if req.objections is not None else (
            existing.objections if existing else [PlaybookObjection(**o) for o in DEFAULT_OBJECTIONS]
        )

        response_seq = req.response_sequence if req.response_sequence is not None else (
            existing.response_sequence if existing else [ResponseSequenceStep(**sq) for sq in DEFAULT_RESPONSE_SEQUENCE]
        )
        for idx, sq in enumerate(response_seq, start=1):
            sq.order = idx

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
            objections=objections_data,
            response_sequence=response_seq,
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
# 7. Standalone App Factory
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
