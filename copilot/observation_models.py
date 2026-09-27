import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class RepPromptBehavior(str, Enum):
    """Canonical 6-state taxonomy for rep teleprompter behavior from Spec 11 §4.5 and Spec 01 §12."""
    EXACT = "exact"
    STRATEGIC_PARAPHRASE = "strategic_paraphrase"
    USEFUL_DEVIATION = "useful_deviation"
    HARMFUL_DEVIATION = "harmful_deviation"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class PromptOutcome(BaseModel):
    """Spec 11 §4.5 Canonical Machine Object: PromptOutcome.

    Measures whether the salesperson preserved the strategic intent of the teleprompter prompt,
    the resulting prospect reaction, and the observed state delta feeding back into ConversationState.
    """
    outcome_id: str = Field(default_factory=lambda: f"out_{uuid.uuid4().hex[:10]}")
    prompt_id: str
    decision_id: str
    rep_behavior: RepPromptBehavior
    adherence_score: float = Field(..., ge=0.0, le=1.0, description="Computed adherence score from Spec 11 §4.5")
    rep_turn_ids: List[str] = Field(default_factory=list, description="Rep turn IDs delivering this prompt")
    prospect_reaction_turn_ids: List[str] = Field(default_factory=list, description="Prospect reaction turn IDs attributed to this prompt")
    observed_state_delta: Dict[str, float] = Field(default_factory=dict, description="State delta: trust, engagement, momentum, etc.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence per Spec 10")
    violation_reason: Optional[str] = None
    lexical_overlap: float = 0.0
    structural_action_executed: bool = False
    topic_continuity: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ObservationConfig(BaseModel):
    """Configuration governing the Observation Engine and reaction attribution windows."""
    max_reaction_window_turns: int = Field(2, description="Max prospect turns to attribute to a single prompt outcome")
    backchannel_max_words: int = Field(2, description="Max word count to classify an initial prospect turn as a backchannel")
    lexical_exact_threshold: float = Field(0.85, description="Lexical overlap ratio threshold for EXACT behavior")
    semantic_similarity_threshold: float = Field(0.55, description="Minimum semantic similarity threshold for guardrail intent match")
    topic_continuity_threshold: float = Field(0.30, description="Minimum topic anchor similarity to confirm topic continuity")


# ==============================================================================
# NOTE ON SPEC COVERAGE (Spec 11 §13.1 Scenario 22 / Spec 05):
# Spec 11 §4 defines machine objects 4.1 to 4.6, but stops before defining an explicit
# schema for LearningCandidate. This schema is an inferred engineering extension
# implementing Scenario 22 ("Novel objection appears once -> Stored as learning candidate,
# not immediate Core doctrine").
# ==============================================================================
class LearningCandidate(BaseModel):
    """Inferred schema for Spec 11 §13.1 Scenario 22 / Spec 05.

    Captures unclassified, novel objections or unexpected phrasing in isolated retention
    storage without polluting authoritative Core doctrine or active objection taxonomies.
    """
    candidate_id: str = Field(default_factory=lambda: f"lrn_{uuid.uuid4().hex[:10]}")
    call_id: str
    turn_id: str
    raw_utterance: str
    candidate_type: Literal["novel_objection", "unexpected_phrasing", "unresolved_edge_case"] = "novel_objection"
    isolated_from_core_doctrine: bool = True
    occurrence_count: int = 1
    detected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, str] = Field(default_factory=dict)
