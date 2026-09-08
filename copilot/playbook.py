"""Custom Playbook Module for Sales AI.

Provides data models, JSON file persistence, and FastAPI router for
creating and managing custom sales playbooks according to the PitchProX
Playbook Integration Specification (Version 1.0):

- Playbook as a methodology lens over the Core engine.
- Full 16-prospect lead types catalog.
- Controlled Objection Library with 4 canonical categories (Financial, Risk, Logistics, Relationship)
  supporting multiple categories per objection and 11 standardized response styles.
- Runtime sensitivity & threshold controls (Prompt Timing, Emotional Sensitivity, Objection Confidence,
  Trust Sensitivity Threshold, Prompt Cooldown, Do/Don't Boundaries, Language Rules).
- Spec Section 3 named components: Trust-Building Style, Objection Tone, Discovery Style,
  Closing Style, Follow-Up Style.
- Semantic objection matching with confidence threshold gating.
- Atomic persistence and read-only protection for premium methodology products.
- 5-step wizard workflow (Basic Info, Your Style, Conversation Flow, Objection Handling, Review & Save).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

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
        "tag": "Goal: Engage",
        "icon": "hand",
    },
    {
        "id": "stage_discovery",
        "order": 2,
        "name": "Discovery",
        "goal": "Understand their motivation and timeline.",
        "description": "Ask questions and uncover their situation and needs.",
        "tag": "Goal: Understand",
        "icon": "search",
    },
    {
        "id": "stage_value_prop",
        "order": 3,
        "name": "Value Proposition",
        "goal": "Build credibility and differentiate yourself.",
        "description": "Position your value and show how you solve their problem.",
        "tag": "Goal: Differentiate",
        "icon": "diamond",
    },
    {
        "id": "stage_objections",
        "order": 4,
        "name": "Handle Objections",
        "goal": "Overcome doubt and strengthen confidence.",
        "description": "Address concerns and remove hesitation.",
        "tag": "Goal: Resolve",
        "icon": "shield",
    },
    {
        "id": "stage_close",
        "order": 5,
        "name": "Close",
        "goal": "Get a commitment to move forward.",
        "description": "Guide them to commit to the next step.",
        "tag": "Goal: Commit",
        "icon": "target",
    },
    {
        "id": "stage_follow_up",
        "order": 6,
        "name": "Follow Up",
        "goal": "Stay top of mind and convert later.",
        "description": "Reinforce value and keep the momentum.",
        "tag": "Goal: Nurture",
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
        "category": ["Financial"],
        "lead_types": ["FSBO (For Sale By Owner)", "Expired Listings", "Pre-Expired Listings"],
        "response_style": "Quantify",
        "ai_suggestion": 'This "Quantify" reframe highlights net take-home equity. Mention that MLS-represented listings in this area closed on average 7.2% higher.',
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_2",
        "objection": "Not a good time to sell.",
        "category": ["Logistics"],
        "lead_types": ["FSBO (For Sale By Owner)", "Expired Listings", "Absentee Owners"],
        "response_style": "Future Pace",
        "ai_suggestion": "Validate market uncertainty, then project seasonal equity growth to establish timeline clarity.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_3",
        "objection": "I already have an agent.",
        "category": ["Relationship"],
        "lead_types": ["Expired Listings", "Pre-Expired Listings"],
        "response_style": "Differentiate",
        "ai_suggestion": "Respect loyalty, then ask diagnostic questions about current marketing reach and recent buyer feedback.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_4",
        "objection": "How did you get my number?",
        "category": ["Risk"],
        "lead_types": ["FSBO (For Sale By Owner)", "Circle Prospecting", "Absentee Owners"],
        "response_style": "Direct",
        "ai_suggestion": "Be transparent about public record property rolls, then immediately explain why active local buyers asked about their property.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_5",
        "objection": "I'm not interested.",
        "category": ["Relationship"],
        "lead_types": ["Expired Listings", "Past Clients & Sphere", "Circle Prospecting"],
        "response_style": "Validate",
        "ai_suggestion": "Disarm friction with low-pressure validation: 'Understood, just curious if timing changed or if you already resolved the property need?'",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_6",
        "objection": "We want to wait for interest rates to drop.",
        "category": ["Financial"],
        "lead_types": ["Investor / Cash Buyers", "Pre-Foreclosure", "Absentee Owners"],
        "response_style": "Reframe",
        "ai_suggestion": "Reframe the trade-off between rate drops and inventory surges that drive acquisition competition.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_7",
        "objection": "We need to think about it.",
        "category": ["Logistics", "Risk"],
        "lead_types": ["Trust & Estate", "Probate", "Divorce"],
        "response_style": "Clarify",
        "ai_suggestion": "Clarify whether it is the timeline, legal estate steps, or family consensus holding back the next milestone.",
        "last_updated": "May 2, 2024",
    },
    {
        "id": "obj_8",
        "objection": "Send me information first.",
        "category": ["Risk"],
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
    tag: str = Field("", max_length=80, description="AI-generated concise stage goal tag, e.g., 'Goal: Engage'")
    icon: str = Field("hand", description="hand | search | diamond | shield | target | phone")


class PlaybookObjection(BaseModel):
    id: str = Field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:8]}")
    objection: str = Field(..., min_length=1, max_length=300)
    category: List[str] = Field(
        default_factory=lambda: ["Financial"],
        description="Financial | Risk | Logistics | Relationship (multiple allowed)",
    )
    lead_types: List[str] = Field(default_factory=lambda: ["Expired Listings", "FSBO (For Sale By Owner)"])
    response_style: str = Field(
        "Reframe",
        description="Reframe | Clarify | Validate | Educate | Quantify | Differentiate | De-Risk | Challenge | Social Proof | Future Pace | Direct",
    )
    ai_suggestion: str = Field("", max_length=500)
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%b %d, %Y"))

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, v: Any) -> List[str]:
        """Support both single strings (legacy data) and list of categories."""
        if isinstance(v, str):
            if "," in v:
                return [c.strip() for c in v.split(",") if c.strip()]
            return [v.strip()] if v.strip() else ["Financial"]
        if isinstance(v, (list, tuple, set)):
            res = [str(c).strip() for c in v if str(c).strip()]
            return res if res else ["Financial"]
        return ["Financial"]


class ResponseSequenceStep(BaseModel):
    id: str = Field(default_factory=lambda: f"seq_{uuid.uuid4().hex[:6]}")
    order: int = 1
    name: str = Field(..., min_length=1, max_length=60)
    icon: str = Field("mic", max_length=30)


class PlaybookRuntimeSettings(BaseModel):
    """Runtime sensitivity and boundary rules from Playbook Specification Section 3."""
    prompt_timing_sensitivity: str = Field("Normal", description="Normal | Aggressive | Relaxed")
    emotional_sensitivity: str = Field(
        "Normal",
        description="Normal | High | Adaptive — Adaptive adjusts emphasis to live in-call signals only; does not create persistent behavior changes outside the approved learning pipeline."
    )
    objection_confidence_threshold: float = Field(0.75, ge=0.0, le=1.0)
    trust_sensitivity_threshold: float = Field(0.70, ge=0.0, le=1.0, description="Point below which Core should weight trust deterioration more heavily")
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

    @property
    def trust_alert_threshold(self) -> float:
        return self.trust_sensitivity_threshold

    @trust_alert_threshold.setter
    def trust_alert_threshold(self, val: float) -> None:
        self.trust_sensitivity_threshold = val

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "prompt_timing" in data and "prompt_timing_sensitivity" not in data:
                data["prompt_timing_sensitivity"] = data["prompt_timing"]
            if "trust_alert_threshold" in data and "trust_sensitivity_threshold" not in data:
                data["trust_sensitivity_threshold"] = data["trust_alert_threshold"]
        return data

    @field_validator("do_dont_boundaries", "language_rules", mode="before")
    @classmethod
    def parse_str_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            lines = [line.strip() for line in v.split("\n") if line.strip()]
            return lines if lines else [v.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("prompt_cooldown_seconds", mode="before")
    @classmethod
    def parse_cooldown(cls, v: Any) -> int:
        if v is not None:
            try:
                return int(round(float(v)))
            except (ValueError, TypeError):
                return 4
        return 4


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
    
    # Spec Section 3 named distinct style components
    trust_building_style: str = Field("Validation-First", description="Validation-First | Credibility-First | Rapport-First | Transparency-First")
    objection_tone: str = Field("Empathetic & Resilient", description="Empathetic & Resilient | Assertive & Reframing | Curious & Exploratory | Calm & Analytical")
    discovery_style: str = Field("Diagnostic / Question-Led", description="Diagnostic / Question-Led | Problem-Agitation | Conversational Inquiry | Mutual Assessment")
    closing_style: str = Field("Low-Pressure Next-Step Commitment", description="Low-Pressure Next-Step Commitment | Direct Close | Alternative Choice | Urgency-Framed")
    follow_up_style: str = Field("Value-Added Persistence", description="Value-Added Persistence | Multi-Touch Education | Time-Sensitive Check-in | Relationship Nurture")

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
    
    # Read-only & source ownership control (Spec Section 2 & 7)
    is_readonly: bool = Field(False, description="Read-only protection for premium/purchased methodology products.")
    source: str = Field("custom", description="custom | premium | template")
    
    basic_info: PlaybookBasicInfo
    style: PlaybookStyle = Field(default_factory=PlaybookStyle)
    stages: List[PlaybookStage] = Field(default_factory=lambda: [PlaybookStage(**s) for s in DEFAULT_STAGES])
    objections: List[PlaybookObjection] = Field(default_factory=lambda: [PlaybookObjection(**o) for o in DEFAULT_OBJECTIONS])
    response_sequence: List[ResponseSequenceStep] = Field(default_factory=lambda: [ResponseSequenceStep(**sq) for sq in DEFAULT_RESPONSE_SEQUENCE])
    completeness_score: int = Field(94, ge=0, le=100)
    quality_score: int = Field(94, ge=0, le=100)
    quality_feedback: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlaybookSaveRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = "draft"
    current_step: Optional[int] = 1
    is_readonly: Optional[bool] = False
    source: Optional[str] = "custom"
    basic_info: PlaybookBasicInfo
    style: Optional[PlaybookStyle] = None
    stages: Optional[List[PlaybookStage]] = None
    objections: Optional[List[PlaybookObjection]] = None
    response_sequence: Optional[List[ResponseSequenceStep]] = None
    completeness_score: Optional[int] = None
    quality_score: Optional[int] = None
    quality_feedback: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class RuntimePromptSimulateRequest(BaseModel):
    playbook_id: Optional[str] = None
    playbook: Optional[PlaybookSaveRequest] = None
    trust_score: Optional[float] = None
    detected_objection: Optional[str] = None
    lead_type: Optional[str] = None


# ==========================================
# 3. Semantic Objection Matching & Lens Synthesis
# ==========================================

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "guys", "already", "much", "dont", "didnt", "cant", "wont", "couldnt",
    "wouldnt", "want", "wants", "wanted", "got", "can", "will", "going"
}

_SYNONYM_MAP = {
    "commissions": "commission", "percentage": "commission", "fee": "commission", "fees": "commission",
    "selling": "sell", "sold": "sell", "seller": "sell",
    "buying": "buy", "bought": "buy", "buyer": "buy",
    "pricing": "price", "expensive": "cost", "costly": "cost",
    "rates": "rate", "interest": "rate", "mortgage": "rate",
    "thinking": "think", "thought": "think", "ponder": "think",
    "agent": "realtor", "broker": "realtor", "realtors": "realtor",
    "number": "phone", "phone": "number", "cell": "number", "contact": "number",
    "interested": "interest", "interest": "interest"
}


def compute_token_semantic_similarity(query: str, target: str) -> float:
    """Computes semantic token overlap similarity between query and target string.

    Handles paraphrased conversational objections by normalizing stopwords, stemming/lemmatizing
    domain terms, and computing harmonic Jaccard & target content coverage.
    """
    def tokenize(text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
        tokens = []
        for w in words:
            w_clean = w.replace("'", "")
            mapped = _SYNONYM_MAP.get(w_clean, w_clean)
            tokens.append(mapped)
        return tokens

    q_tokens = tokenize(query)
    t_tokens = tokenize(target)

    if not q_tokens or not t_tokens:
        return 0.0

    q_content = [w for w in q_tokens if w not in _STOP_WORDS]
    t_content = [w for w in t_tokens if w not in _STOP_WORDS]

    if q_content and t_content:
        q_set = set(q_content)
        t_set = set(t_content)
        intersection = q_set.intersection(t_set)
        if intersection:
            target_coverage = len(intersection) / len(t_set)
            jaccard = len(intersection) / len(q_set.union(t_set))
            if target_coverage >= 0.99:
                score = max(0.88, 0.70 * target_coverage + 0.30 * jaccard)
            else:
                score = 0.75 * target_coverage + 0.25 * jaccard
            return min(1.0, score)

    q_str = " ".join(q_tokens)
    t_str = " ".join(t_tokens)
    if t_str in q_str or q_str in t_str:
        return 0.85

    return 0.0


def match_objection_semantically(
    detected_objection: str,
    objections: List[PlaybookObjection],
    threshold: float = 0.75,
) -> Optional[Tuple[PlaybookObjection, float]]:
    """Semantically matches a spoken objection against the playbook objection library.

    Enforces objection_confidence_threshold: returns None if the best similarity score
    does not meet or exceed threshold.
    """
    if not detected_objection or not objections:
        return None

    best_obj: Optional[PlaybookObjection] = None
    best_score: float = 0.0

    for obj in objections:
        score = compute_token_semantic_similarity(detected_objection, obj.objection)
        if score > best_score:
            best_score = score
            best_obj = obj

    if best_obj and best_score >= threshold:
        return best_obj, best_score

    LOGGER.debug(
        "Objection '%s' best match '%s' score %.2f rejected by threshold %.2f",
        detected_objection,
        best_obj.objection if best_obj else "None",
        best_score,
        threshold,
    )
    return None


async def ai_classify_stage_tag(goal: str, stage_name: str = "") -> Optional[str]:
    """Calls Groq or OpenAI to dynamically extract a concise 'Goal: <Verb>' action tag from goal text.
    
    Returns None if no API keys are provided or if calls fail/timeout, enabling graceful fallback.
    """
    if not goal:
        return None

    api_key_groq = os.getenv("GROQ_API_KEY")
    api_key_openai = os.getenv("OPENAI_API_KEY")

    prompt = (
        f"You are a sales methodology AI. A salesperson created a conversation stage called '{stage_name}' "
        f"with the goal: \"{goal}\".\n"
        f"Generate a single, ultra-concise 2-word stage tag in the exact format 'Goal: <Verb>' "
        f"(for example: 'Goal: Engage', 'Goal: Understand', 'Goal: Differentiate', 'Goal: Resolve', 'Goal: Commit', 'Goal: Nurture', 'Goal: Qualify', 'Goal: Reframe').\n"
        f"Output ONLY the tag string, nothing else."
    )

    # 1. Try Groq if configured with non-mock key
    if api_key_groq and not api_key_groq.startswith("mock_"):
        try:
            import groq
            client = groq.Groq(api_key=api_key_groq, timeout=3.0)
            def _call_groq():
                resp = client.chat.completions.create(
                    model=os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=20,
                )
                return resp.choices[0].message.content.strip()

            res = await asyncio.to_thread(_call_groq)
            clean = res.split("\n")[0].strip().strip('"\'')
            if clean.startswith("Goal:"):
                return clean
            words = re.findall(r"\b[A-Za-z]+\b", clean)
            if words:
                return f"Goal: {words[-1].capitalize()}"
        except Exception as exc:
            LOGGER.debug("Groq stage tag classification fallback: %s", exc)

    # 2. Try OpenAI if configured with non-mock key
    if api_key_openai and not api_key_openai.startswith("mock_"):
        try:
            import openai
            client = openai.OpenAI(api_key=api_key_openai, timeout=3.0)
            def _call_openai():
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=20,
                )
                return resp.choices[0].message.content.strip()

            res = await asyncio.to_thread(_call_openai)
            clean = res.split("\n")[0].strip().strip('"\'')
            if clean.startswith("Goal:"):
                return clean
            words = re.findall(r"\b[A-Za-z]+\b", clean)
            if words:
                return f"Goal: {words[-1].capitalize()}"
        except Exception as exc:
            LOGGER.debug("OpenAI stage tag classification fallback: %s", exc)

    return None


def classify_stage_tag_heuristic(goal: str, stage_name: str = "") -> str:
    """Deterministic keyword-based classifier that extracts a concise stage tag from goal text."""
    goal_lower = (goal or "").lower()
    stage_lower = (stage_name or "").lower()

    if "engage" in goal_lower or "rapport" in goal_lower or "tone" in goal_lower or "opening" in stage_lower:
        return "Goal: Engage"
    elif "understand" in goal_lower or "question" in goal_lower or "need" in goal_lower or "timeline" in goal_lower or "motivation" in goal_lower or "discovery" in stage_lower:
        return "Goal: Understand"
    elif "differentiate" in goal_lower or "credibility" in goal_lower or "value" in goal_lower or "position" in goal_lower:
        return "Goal: Differentiate"
    elif "overcome" in goal_lower or "doubt" in goal_lower or "objection" in goal_lower or "hesitation" in goal_lower or "resolve" in goal_lower:
        return "Goal: Resolve"
    elif "commit" in goal_lower or "close" in goal_lower or "forward" in goal_lower or "next step" in goal_lower:
        return "Goal: Commit"
    elif "nurture" in goal_lower or "follow" in goal_lower or "convert" in goal_lower or "top of mind" in goal_lower:
        return "Goal: Nurture"
    else:
        words = [w for w in re.findall(r"\b[A-Za-z]+\b", goal) if len(w) > 3 and w.lower() not in _STOP_WORDS]
        first_keyword = words[0].capitalize() if words else "Focus"
        return f"Goal: {first_keyword}"


def serialize_playbook_prompt_section(
    playbook: Playbook,
    current_lead_type: Optional[str] = None,
    detected_objection: Optional[str] = None,
    current_trust_score: Optional[float] = None,
) -> str:
    """Serializes structured playbook data into a labeled text section.

    Enforces:
    - Semantic objection matching gated by objection_confidence_threshold.
    - Spec Section 3 distinct style components (Trust, Objection Tone, Discovery, Close, Follow-Up).
    - Trust sensitivity guidance gated by trust_sensitivity_threshold.
    """
    tone = playbook.style.overall_tone
    comm_style = playbook.style.communication_style
    energy = playbook.style.energy_level
    sentence = playbook.style.sentence_style
    formality = playbook.style.formality
    humor = playbook.style.humor_level
    philosophy = playbook.basic_info.philosophy or "Deliver value upfront and guide the prospect to clear next steps."
    voice = playbook.basic_info.voice_phrases or ""
    settings = playbook.style.runtime_settings

    seq_order = " -> ".join([s.name for s in sorted(playbook.response_sequence, key=lambda x: x.order)])
    do_dont_lines = "\n".join([f"  * {rule}" for rule in settings.do_dont_boundaries])
    lang_rules = "\n".join([f"  * {rule}" for rule in settings.language_rules])

    # Find relevant objection using semantic matching with threshold gating
    matching_obj_str = ""
    if detected_objection:
        match_result = match_objection_semantically(
            detected_objection=detected_objection,
            objections=playbook.objections,
            threshold=settings.objection_confidence_threshold,
        )
        if match_result:
            obj, confidence = match_result
            cat_display = ", ".join(obj.category) if isinstance(obj.category, list) else str(obj.category)
            matching_obj_str = (
                f"\n  [Candidate Objection Match — Lexical Similarity: {int(confidence * 100)}% "
                f"(reference signal only; final relevance should be determined using full conversational context, "
                f"objection recurrence, and lead type)]:\n"
                f"  - Possible Objection: {obj.objection} ({cat_display})\n"
                f"  - Suggested Response Style (if Core confirms relevance): {obj.response_style}\n"
                f"  - Reference Strategy Tip: {obj.ai_suggestion}\n"
            )

    # Trust sensitivity guidance
    trust_note_str = ""
    if current_trust_score is not None and current_trust_score < settings.trust_sensitivity_threshold:
        trust_note_str = (
            f"\n  [Trust Sensitivity Note (Trust: {int(current_trust_score * 100)}% below sensitivity point {int(settings.trust_sensitivity_threshold * 100)}%)]:\n"
            f"  Evidence suggests declining trust. Increase the weight given to trust-recovery strategies "
            f"when selecting the next move — preferred methodology: '{playbook.style.trust_building_style}', "
            f"if trust recovery is strategically appropriate given full conversational context."
        )

    lines = [
        f"### ACTIVE PLAYBOOK METHODOLOGY LENS: {playbook.title.upper()} ###",
        f"- Philosophy: {philosophy}",
        f"- Coaching Style: Tone={tone} | Approach={comm_style} | Energy={energy} | Sentence={sentence} | Formality={formality} | Humor={humor}",
        f"- Methodology Style Choices: Trust={playbook.style.trust_building_style} | Objection Tone={playbook.style.objection_tone} | Discovery={playbook.style.discovery_style} | Close={playbook.style.closing_style} | Follow-Up={playbook.style.follow_up_style}",
        f"- Objection Response Sequence: {seq_order}",
        f"- Runtime Sensitivity & Thresholds: Timing={settings.prompt_timing_sensitivity} | Emotion={settings.emotional_sensitivity} | Min Confidence={int(settings.objection_confidence_threshold*100)}% | Trust Sensitivity={int(settings.trust_sensitivity_threshold*100)}% | Cooldown={settings.prompt_cooldown_seconds}s",
    ]

    if settings.emotional_sensitivity == "Adaptive":
        lines.append(
            "- Note: Adaptive emotional sensitivity applies to live signals within THIS call only. "
            "It does not create any permanent change to future calls, other users, or stored methodology."
        )

    lines.extend([
        "- Do / Don't Boundaries:\n" + do_dont_lines,
        "- Language Rules:\n" + lang_rules,
    ])

    if voice:
        lines.append(f"- Signature Voice Expressions:\n{voice}")

    if matching_obj_str:
        lines.append(matching_obj_str)

    if trust_note_str:
        lines.append(trust_note_str)

    lines.append(
        "- Runtime Operational Constraint: This methodology lens guides HOW a response is formulated and offers "
        "suggested response styles and strategic preferences. The Core retains full authority to select a "
        "different response style (Clarify, Validate, De-Risk, Reframe, etc.) if current conversational evidence "
        "indicates a better strategic fit. Never invent unverified facts; always output exactly ONE concise "
        "prompt (<25 words) for the salesperson."
    )
    return "\n".join(lines)


def build_playbook_methodology_prompt(playbook: Playbook, live_context: Optional[Dict[str, Any]] = None) -> str:
    """Build the runtime methodology lens prompt that wraps around PitchProX Core intelligence."""
    lead_type = live_context.get("lead_type") if live_context else None
    detected_obj = live_context.get("detected_objection") if live_context else None
    trust_score_val = live_context.get("trust_score") or live_context.get("current_trust_score") if live_context else None
    current_trust = None
    if trust_score_val is not None:
        try:
            current_trust = float(trust_score_val)
        except (ValueError, TypeError):
            current_trust = None

    return serialize_playbook_prompt_section(
        playbook,
        current_lead_type=lead_type,
        detected_objection=detected_obj,
        current_trust_score=current_trust,
    )


# ==========================================
# 4. Quality & Completeness Evaluation
# ==========================================

def calculate_playbook_completeness_score(pb: Playbook) -> int:
    """Compute 0-100 completeness score based on field population."""
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


async def ai_evaluate_playbook_quality(pb: Playbook) -> Optional[Dict[str, Any]]:
    """LLM reads the actual playbook content and judges methodology quality."""
    objections_summary = "\n".join([
        f"- \"{o.objection}\" ({', '.join(o.category) if isinstance(o.category, list) else o.category}) -> {o.response_style}"
        for o in pb.objections
    ])
    stages_summary = "\n".join([
        f"- {s.name}: {s.goal}" for s in pb.stages
    ])

    prompt = f"""You are an expert sales methodology reviewer for PitchProX.
Evaluate this custom sales playbook on real methodology quality — not just field completeness.

Philosophy: {pb.basic_info.philosophy or "Not specified"}
Communication Style: {pb.style.communication_style}, Tone: {pb.style.overall_tone}
Trust-Building Style: {pb.style.trust_building_style}
Objection Tone: {pb.style.objection_tone}

Conversation Stages:
{stages_summary}

Objection Library:
{objections_summary}

Judge on:
1. Does the philosophy make sense and is it actionable?
2. Do the stages progress logically toward a close?
3. Are the objection responses genuinely strategic (not generic)?
4. Is there good category diversity, or gaps in coverage?
5. Does the overall methodology feel coherent as a system, not disconnected pieces?

Return ONLY valid JSON:
{{
  "llm_quality_score": <int 0-100>,
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "coherence_feedback": "A concise positive 1-sentence confirmation of methodology alignment. Avoid negative critique, gaps, or phrases like 'However, ... may need further detail'."
}}
"""

    api_key_groq = os.getenv("GROQ_API_KEY")
    if api_key_groq and not api_key_groq.startswith("mock_"):
        try:
            import groq
            client = groq.Groq(api_key=api_key_groq, timeout=8.0)
            groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
            if "openai/gpt-oss" in groq_model or "llama" in groq_model:
                groq_model = "qwen/qwen3.8-27b"
            def _call_groq():
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content
            raw = await asyncio.to_thread(_call_groq)
            return json.loads(raw)
        except Exception as exc:
            LOGGER.warning("Groq quality evaluation failed, falling back: %s", exc)

    api_key_openai = os.getenv("OPENAI_API_KEY")
    if api_key_openai and not api_key_openai.startswith("mock_"):
        try:
            import openai
            client = openai.OpenAI(api_key=api_key_openai, timeout=8.0)
            def _call_openai():
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content
            raw = await asyncio.to_thread(_call_openai)
            return json.loads(raw)
        except Exception as exc:
            LOGGER.warning("OpenAI quality evaluation failed, falling back: %s", exc)

    return None


async def ai_summarize_playbook_voice(pb: Playbook, full_prompt: str) -> Optional[str]:
    """LLM reads the full compiled methodology prompt and writes a short, warm,
    first-person preview of how the AI coach will sound — for UI display only.
    Does NOT affect the real methodology_prompt used at live-call time.
    """
    rep_name = pb.basic_info.name.split()[0] if pb.basic_info.name else "there"

    summarize_prompt = f"""You are writing a short, warm preview message FROM the AI sales coach TO the salesperson, 
introducing how it will sound during their calls.

Here is the full compiled methodology configuration for this playbook:
{full_prompt}

Write EXACTLY 3-4 short lines, first person, as if the AI coach is speaking directly to {rep_name}.
Style example (match this tone exactly):
"Hey Daniel, I've got you.
I'll keep my prompts confident, direct, and consultative — short, professional, and to the point.
Let's build a playbook that wins more conversations."

Rules:
- Address them by first name in line 1.
- Reflect the ACTUAL tone, style, and approach from the methodology above — don't invent generic filler.
- Mention 1 concrete, specific thing from their setup (e.g. their objection tone, their trust-building style, or their philosophy) so it feels personalized, not templated.
- Keep total output under 50 words.
- Output ONLY the message text, no quotes, no labels, no JSON.
"""

    api_key_groq = os.getenv("GROQ_API_KEY")
    if api_key_groq and not api_key_groq.startswith("mock_"):
        try:
            import groq
            client = groq.Groq(api_key=api_key_groq, timeout=6.0)
            groq_model = os.getenv("GROQ_FAST_MODEL") or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
            if "openai/gpt-oss" in groq_model or "llama" in groq_model:
                groq_model = "qwen/qwen3.8-27b"
            def _call():
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": summarize_prompt}],
                    temperature=0.6,
                    max_tokens=150,
                )
                text = (resp.choices[0].message.content or "").strip()
                if "<think>" in text and "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                return text if text else None
            res = await asyncio.to_thread(_call)
            if res:
                return res
        except Exception as exc:
            LOGGER.warning("AI voice summary failed, will fall back: %s", exc)

    api_key_openai = os.getenv("OPENAI_API_KEY")
    if api_key_openai and not api_key_openai.startswith("mock_"):
        try:
            import openai
            client = openai.OpenAI(api_key=api_key_openai, timeout=6.0)
            def _call():
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": summarize_prompt}],
                    temperature=0.6,
                    max_tokens=150,
                )
                return (resp.choices[0].message.content or "").strip()
            res = await asyncio.to_thread(_call)
            if res:
                return res
        except Exception as exc:
            LOGGER.warning("AI voice summary (OpenAI) failed: %s", exc)

    return None


def evaluate_playbook_structural_quality(pb: Playbook, store: Optional[PlaybookStore] = None) -> Dict[str, Any]:
    completeness = calculate_playbook_completeness_score(pb)
    num_stages = len(pb.stages)
    num_objs = len(pb.objections)
    categories = set()
    for o in pb.objections:
        categories.update(o.category) if isinstance(o.category, list) else categories.add(str(o.category))

    cat_coverage = min(1.0, len(categories) / 4.0)
    stage_balance = min(1.0, num_stages / 6.0)
    obj_coverage = min(1.0, num_objs / 8.0)

    base_score = int((completeness * 0.40) + (cat_coverage * 25) + (stage_balance * 20) + (obj_coverage * 15))
    quality_score = min(max(base_score, 50), 98)

    rating = "Excellent" if quality_score >= 90 else ("Strong" if quality_score >= 80 else "Developing")
    summary = (
        "Great job! Your playbook methodology is complete and ready to use."
        if quality_score >= 90
        else "Good foundation. Broaden category coverage to strengthen versatility."
    )

    return {
        "quality_score": quality_score,
        "structural_score": quality_score,
        "completeness_score": completeness,
        "rating": rating,
        "summary": summary,
        "methodology_assessment": "Structural quality assessment based on methodology completeness and coverage",
        "quality_source": "structural_fallback",
        "coverage_metrics": {
            "stages_count": num_stages,
            "objections_count": num_objs,
            "category_diversity": f"{len(categories)} of 4 categories covered",
            "tone_calibration": pb.style.overall_tone,
        },
    }


def calculate_playbook_quality_score(pb: Playbook) -> int:
    """Synchronous calculation of quality score for offline/test environments."""
    res = evaluate_playbook_structural_quality(pb)
    return res["quality_score"]


async def evaluate_playbook_quality_benchmark(
    pb: Playbook,
    store: Optional[PlaybookStore] = None,
) -> Dict[str, Any]:
    """Evaluates playbook quality by combining real LLM methodology judgment (60%)
    with structural completeness and coverage analysis (40%).

    Gracefully falls back to structural scoring if LLM is unavailable or offline.
    """
    structural_res = evaluate_playbook_structural_quality(pb, store=store)
    structural_score = structural_res["quality_score"]

    llm_result = await ai_evaluate_playbook_quality(pb)

    if llm_result and "llm_quality_score" in llm_result:
        try:
            llm_score = int(llm_result["llm_quality_score"])
            final_score = int(round((structural_score * 0.4) + (llm_score * 0.6)))
            quality_source = "ai_llm"
        except (ValueError, TypeError):
            final_score = structural_score
            llm_result = None
            quality_source = "structural_fallback"
    else:
        final_score = structural_score
        llm_result = None
        quality_source = "structural_fallback"

    res = dict(structural_res)
    res["quality_score"] = final_score
    res["structural_score"] = structural_score
    res["llm_evaluation"] = llm_result
    res["quality_source"] = quality_source

    if llm_result:
        if llm_result.get("coherence_feedback"):
            res["methodology_assessment"] = llm_result.get("coherence_feedback")
        if llm_result.get("strengths"):
            res["strengths"] = llm_result.get("strengths")
        if llm_result.get("gaps"):
            res["gaps"] = llm_result.get("gaps")

    res["rating"] = "Excellent" if final_score >= 90 else ("Strong" if final_score >= 80 else "Developing")
    res["summary"] = (
        "Great job! Your playbook methodology is complete and ready to use."
        if final_score >= 90
        else "Good foundation. Broaden category coverage to strengthen versatility."
    )

    return res


# ==========================================
# 5. Persistence Manager with Atomic Concurrency
# ==========================================

class AwaitablePlaybook(Playbook):
    """Playbook wrapper that functions as an immediate Playbook and also supports awaiting for async post-save tasks."""
    _store: Optional[Any] = PrivateAttr(default=None)

    def __await__(self):
        async def _coro():
            if self._store:
                try:
                    eval_res = await evaluate_playbook_quality_benchmark(self, store=self._store)
                    self.completeness_score = eval_res.get("completeness_score", self.completeness_score)
                    self.quality_score = eval_res.get("quality_score", self.quality_score)
                    self.quality_feedback = eval_res
                    file_path = self._store._file_path(self.id)
                    tmp_file = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
                    try:
                        with open(tmp_file, "w", encoding="utf-8") as f:
                            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
                        os.replace(tmp_file, file_path)
                    finally:
                        if tmp_file.exists():
                            try:
                                tmp_file.unlink()
                            except OSError:
                                pass
                except Exception as e:
                    LOGGER.warning("Async evaluation during save skipped: %s", e)
            return self
        return _coro().__await__()


class PlaybookStore:
    """Manages local JSON file persistence for playbooks with atomic replace and read-only protection."""

    def __init__(self, storage_dir: Path = PLAYBOOKS_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, playbook_id: str) -> Path:
        safe_id = re.sub(r"[^\w\-]", "", playbook_id)
        return self.storage_dir / f"{safe_id}.json"

    def save(self, playbook: Playbook) -> AwaitablePlaybook:
        file_path = self._file_path(playbook.id)

        # Enforce read-only protection on existing premium/internal playbooks
        if file_path.exists():
            existing = self.get(playbook.id)
            if existing and existing.is_readonly:
                raise HTTPException(
                    status_code=403,
                    detail=f"Playbook '{existing.title}' is a read-only methodology package and cannot be edited.",
                )

        playbook.updated_at = datetime.now(timezone.utc).isoformat()
        if not playbook.title and playbook.basic_info.name:
            playbook.title = playbook.basic_info.name
        elif playbook.basic_info.name:
            playbook.title = playbook.basic_info.name

        eval_res = evaluate_playbook_structural_quality(playbook, store=self)
        playbook.completeness_score = eval_res["completeness_score"]
        playbook.quality_score = eval_res["quality_score"]
        playbook.quality_feedback = eval_res

        # Atomic write pattern to avoid concurrency corruption
        tmp_file = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(playbook.model_dump(), f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, file_path)
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

        LOGGER.info("Saved playbook %s (Quality: %s, Complete: %s) to %s", playbook.id, playbook.quality_score, playbook.completeness_score, file_path)
        
        awaitable_pb = AwaitablePlaybook(**playbook.model_dump())
        awaitable_pb._store = self
        return awaitable_pb

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
                    "is_readonly": data.get("is_readonly", False),
                    "source": data.get("source", "custom"),
                    "current_step": data.get("current_step", 1),
                    "industry": data.get("basic_info", {}).get("industry", ""),
                    "stages_count": len(data.get("stages", [])),
                    "objections_count": len(data.get("objections", [])),
                    "quality_score": data.get("quality_score", 94),
                    "completeness_score": data.get("completeness_score", 94),
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
            existing = self.get(playbook_id)
            if existing and existing.is_readonly:
                raise HTTPException(
                    status_code=403,
                    detail=f"Playbook '{existing.title}' is a read-only premium methodology package and cannot be deleted.",
                )
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
            "default_trust_styles": [
                "Validation-First",
                "Credibility-First",
                "Rapport-First",
                "Transparency-First",
            ],
            "default_objection_tones": [
                "Empathetic & Resilient",
                "Assertive & Reframing",
                "Curious & Exploratory",
                "Calm & Analytical",
            ],
            "default_discovery_styles": [
                "Diagnostic / Question-Led",
                "Problem-Agitation",
                "Conversational Inquiry",
                "Mutual Assessment",
            ],
            "default_closing_styles": [
                "Low-Pressure Next-Step Commitment",
                "Direct Close",
                "Alternative Choice",
                "Urgency-Framed",
            ],
            "default_follow_up_styles": [
                "Value-Added Persistence",
                "Multi-Touch Education",
                "Time-Sensitive Check-in",
                "Relationship Nurture",
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

    @router.post("/generate-stage-tag")
    async def generate_stage_tag(payload: Dict[str, Any]):
        """Generates a concise stage tag (e.g., 'Goal: Engage') from the stage's goal and name.

        Classification Strategy:
        1. If LLM credentials (Groq/OpenAI) are configured, invokes an LLM classification model.
        2. Automatically falls back to a deterministic semantic keyword classifier for sub-millisecond,
           offline execution (e.g., test suites or air-gapped environments).
        """
        goal = (payload.get("goal") or "").strip()
        stage_name = (payload.get("stage_name") or "").strip()

        # 1. Attempt LLM classification if configured
        ai_tag = await ai_classify_stage_tag(goal=goal, stage_name=stage_name)
        if ai_tag:
            return {"tag": ai_tag, "goal": goal, "engine": "ai_llm"}

        # 2. Deterministic semantic heuristic fallback
        heuristic_tag = classify_stage_tag_heuristic(goal=goal, stage_name=stage_name)
        return {"tag": heuristic_tag, "goal": goal, "engine": "semantic_heuristic"}

    @router.post("/evaluate-quality")
    async def evaluate_quality_endpoint(req: PlaybookSaveRequest):
        """Generates quality score based on structural methodology completeness and LLM review."""
        stages_data = req.stages if req.stages is not None else [PlaybookStage(**s) for s in DEFAULT_STAGES]
        objections_data = req.objections if req.objections is not None else [PlaybookObjection(**o) for o in DEFAULT_OBJECTIONS]
        response_seq = req.response_sequence if req.response_sequence is not None else [ResponseSequenceStep(**sq) for sq in DEFAULT_RESPONSE_SEQUENCE]
        style_data = req.style if req.style is not None else PlaybookStyle()

        pb = Playbook(
            basic_info=req.basic_info,
            style=style_data,
            stages=stages_data,
            objections=objections_data,
            response_sequence=response_seq,
        )
        return await evaluate_playbook_quality_benchmark(pb, store=store)

    @router.post("/preview-prompt")
    async def preview_ai_prompt(payload: Dict[str, Any]):
        """Generate live coaching prompt preview based on tone, style, and identity."""
        name = payload.get("name", "Daniel")
        tone = payload.get("overall_tone", "Confident")
        style = payload.get("communication_style", "Consultative")
        sentence = payload.get("sentence_style", "Short & Conversational")
        formality = payload.get("formality", "Professional")

        preview_text = (
            f"Hey {name}, I've got you.\n\n"
            f"I'll keep my prompts {tone.lower()}, direct, and {style.lower()} — "
            f"{sentence.lower()}, {formality.lower()}, and to the point.\n\n"
            "Let's build a playbook that wins more conversations."
        )
        return {"preview_text": preview_text}

    @router.get("/{playbook_id}/runtime-prompt")
    async def get_runtime_methodology_prompt(
        playbook_id: str,
        lead_type: Optional[str] = None,
        trust_score: Optional[float] = None,
        detected_objection: Optional[str] = None,
    ):
        """Build runtime system instruction for the LLM based on the active playbook with full inspection metadata."""
        playbook = store.get(playbook_id)
        if not playbook:
            raise HTTPException(status_code=404, detail="Playbook not found")
        prompt = build_playbook_methodology_prompt(
            playbook,
            live_context={"trust_score": trust_score, "detected_objection": detected_objection, "lead_type": lead_type},
        )

        match_info = None
        if detected_objection:
            match_res = match_objection_semantically(
                detected_objection=detected_objection,
                objections=playbook.objections,
                threshold=playbook.style.runtime_settings.objection_confidence_threshold,
            )
            if match_res:
                matched_obj, conf = match_res
                match_info = {
                    "matched": True,
                    "objection": matched_obj.objection,
                    "matched_objection": matched_obj.objection,
                    "category": matched_obj.category,
                    "response_style": matched_obj.response_style,
                    "confidence": round(conf, 3),
                    "threshold": playbook.style.runtime_settings.objection_confidence_threshold,
                    "threshold_met": True,
                    "note": "lexical similarity only, not final relevance",
                }
            else:
                best_obj = None
                best_conf = 0.0
                for obj in playbook.objections:
                    c = compute_token_semantic_similarity(detected_objection, obj.objection)
                    if c > best_conf:
                        best_conf = c
                        best_obj = obj
                match_info = {
                    "matched": False,
                    "closest_objection": best_obj.objection if best_obj else None,
                    "confidence": round(best_conf, 3),
                    "threshold": playbook.style.runtime_settings.objection_confidence_threshold,
                    "threshold_met": False,
                    "note": "lexical similarity only, not final relevance",
                }

        alert_triggered = bool(
            trust_score is not None and trust_score < playbook.style.runtime_settings.trust_sensitivity_threshold
        )

        top_conf = match_info["confidence"] if match_info else 0.0
        top_thresh_met = match_info["threshold_met"] if match_info else False

        return {
            "status": "success",
            "playbook_id": playbook_id,
            "playbook_title": playbook.title,
            "prompt_section": prompt,
            "methodology_prompt": prompt,
            "detected_objection": detected_objection,
            "match_info": match_info,
            "confidence": top_conf,
            "threshold_met": top_thresh_met,
            "trust_score": trust_score,
            "trust_sensitivity_threshold": playbook.style.runtime_settings.trust_sensitivity_threshold,
            "trust_alert_threshold": playbook.style.runtime_settings.trust_sensitivity_threshold,
            "objection_confidence_threshold": playbook.style.runtime_settings.objection_confidence_threshold,
            "trust_sensitivity_triggered": alert_triggered,
            "trust_alert_triggered": alert_triggered,
        }

    @router.post("/simulate-runtime-prompt")
    async def simulate_runtime_prompt(req: RuntimePromptSimulateRequest):
        """Simulate the live runtime methodology prompt from either a saved playbook or an in-memory draft."""
        pb = None
        if req.playbook_id:
            pb = store.get(req.playbook_id)

        if not pb and req.playbook:
            stages_data = req.playbook.stages if req.playbook.stages is not None else [PlaybookStage(**s) for s in DEFAULT_STAGES]
            objections_data = req.playbook.objections if req.playbook.objections is not None else [PlaybookObjection(**o) for o in DEFAULT_OBJECTIONS]
            response_seq = req.playbook.response_sequence if req.playbook.response_sequence is not None else [ResponseSequenceStep(**sq) for sq in DEFAULT_RESPONSE_SEQUENCE]
            style_data = req.playbook.style if req.playbook.style is not None else PlaybookStyle()

            pb = Playbook(
                id=req.playbook.id or f"pb_sim_{uuid.uuid4().hex[:8]}",
                title=req.playbook.title or req.playbook.basic_info.name,
                status=req.playbook.status or "draft",
                current_step=req.playbook.current_step or 1,
                is_readonly=req.playbook.is_readonly or False,
                source=req.playbook.source or "custom",
                basic_info=req.playbook.basic_info,
                style=style_data,
                stages=stages_data,
                objections=objections_data,
                response_sequence=response_seq,
            )
        elif not pb:
            info = PlaybookBasicInfo(name="Default Simulation Playbook")
            pb = Playbook(title="Default Simulation Playbook", basic_info=info)

        prompt = build_playbook_methodology_prompt(
            pb,
            live_context={
                "trust_score": req.trust_score,
                "detected_objection": req.detected_objection,
                "lead_type": req.lead_type,
            },
        )

        match_info = None
        if req.detected_objection:
            match_res = match_objection_semantically(
                detected_objection=req.detected_objection,
                objections=pb.objections,
                threshold=pb.style.runtime_settings.objection_confidence_threshold,
            )
            if match_res:
                matched_obj, conf = match_res
                match_info = {
                    "matched": True,
                    "objection": matched_obj.objection,
                    "matched_objection": matched_obj.objection,
                    "category": matched_obj.category,
                    "response_style": matched_obj.response_style,
                    "confidence": round(conf, 3),
                    "threshold": pb.style.runtime_settings.objection_confidence_threshold,
                    "threshold_met": True,
                    "note": "lexical similarity only, not final relevance",
                }
            else:
                best_obj = None
                best_conf = 0.0
                for obj in pb.objections:
                    c = compute_token_semantic_similarity(req.detected_objection, obj.objection)
                    if c > best_conf:
                        best_conf = c
                        best_obj = obj
                match_info = {
                    "matched": False,
                    "closest_objection": best_obj.objection if best_obj else None,
                    "confidence": round(best_conf, 3),
                    "threshold": pb.style.runtime_settings.objection_confidence_threshold,
                    "threshold_met": False,
                    "note": "lexical similarity only, not final relevance",
                }

        alert_triggered = bool(
            req.trust_score is not None and req.trust_score < pb.style.runtime_settings.trust_sensitivity_threshold
        )

        top_conf = match_info["confidence"] if match_info else 0.0
        top_thresh_met = match_info["threshold_met"] if match_info else False

        display_summary = await ai_summarize_playbook_voice(pb, prompt)
        if not display_summary:
            rep_name = pb.basic_info.name.split()[0] if pb.basic_info.name else "there"
            display_summary = (
                f"Hey {rep_name}, I've got you.\n"
                f"I'll keep my prompts {pb.style.overall_tone.lower()}, {pb.style.communication_style.lower()}, "
                f"and {pb.style.sentence_style.lower()}.\n"
                f"Let's build a playbook that wins more conversations."
            )
            summary_source = "template_fallback"
        else:
            summary_source = "ai_llm"

        return {
            "status": "success",
            "playbook_id": pb.id,
            "playbook_title": pb.title,
            "prompt_section": prompt,
            "methodology_prompt": prompt,
            "display_summary": display_summary,
            "summary_source": summary_source,
            "detected_objection": req.detected_objection,
            "match_info": match_info,
            "confidence": top_conf,
            "threshold_met": top_thresh_met,
            "trust_score": req.trust_score,
            "trust_sensitivity_threshold": pb.style.runtime_settings.trust_sensitivity_threshold,
            "trust_alert_threshold": pb.style.runtime_settings.trust_sensitivity_threshold,
            "objection_confidence_threshold": pb.style.runtime_settings.objection_confidence_threshold,
            "trust_sensitivity_triggered": alert_triggered,
            "trust_alert_triggered": alert_triggered,
        }

    @router.post("/save")
    async def save_playbook(req: PlaybookSaveRequest):
        """Create or update a playbook."""
        existing = store.get(req.id) if req.id else None

        if existing and existing.is_readonly:
            raise HTTPException(
                status_code=403,
                detail=f"Playbook '{existing.title}' is a read-only methodology package and cannot be modified.",
            )

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

        is_ro = req.is_readonly if req.is_readonly is not None else (existing.is_readonly if existing else False)
        src = req.source if req.source is not None else (existing.source if existing else "custom")

        playbook = Playbook(
            id=req.id or (existing.id if existing else f"pb_{uuid.uuid4().hex[:12]}"),
            title=req.title or req.basic_info.name,
            status=req.status or (existing.status if existing else "draft"),
            current_step=req.current_step or (existing.current_step if existing else 1),
            is_readonly=is_ro,
            source=src,
            created_at=existing.created_at if existing else datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            basic_info=req.basic_info,
            style=style_data,
            stages=stages_data,
            objections=objections_data,
            response_sequence=response_seq,
            metadata=req.metadata or (existing.metadata if existing else {}),
        )

        saved = await store.save(playbook)
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
