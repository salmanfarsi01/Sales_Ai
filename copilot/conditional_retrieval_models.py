from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class KnowledgeClass(str, Enum):
    """The 9 canonical knowledge classes from Doc 02 §3."""
    SALESPERSON_PROFILE = "salesperson_profile"
    PERFORMANCE_PAST_SALES = "performance_past_sales"
    VALUE_PROPOSITION = "value_proposition"
    SERVICE_PROCESS = "service_process"
    PRICING_PACKAGING = "pricing_packaging"
    COMPANY_POLICY = "company_policy"
    MARKET_KNOWLEDGE = "market_knowledge"
    CLIENT_TESTIMONIALS = "client_testimonials"
    OBJECTION_RESPONSES = "objection_responses"


class RetrievalTrigger(str, Enum):
    """The 6 canonical retrieval triggers from Doc 02 §6."""
    FACTUAL_QUESTION = "factual_question"
    CREDIBILITY_OBJECTION = "credibility_objection"
    SERVICE_PROCESS_OBJECTION = "service_process_objection"
    PLAYBOOK_EVIDENCE = "playbook_evidence"
    POLICY_WORDING = "policy_wording"
    PRIOR_CLAIM_REFERENCE = "prior_claim_reference"


class RetrievalNeed(BaseModel):
    """Specific retrieval need declared by Core Intelligence or Playbook."""
    trigger: RetrievalTrigger
    need_statement: str
    topic_tags: List[str] = Field(default_factory=list)
    time_sensitive: bool = False


class RetrievalRequest(BaseModel):
    """Input contract from Core Intelligence to Retrieval per Spec Doc 02 §6 & Workflow Step 1.1."""
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    call_id: str
    state_version: int
    user_id: str
    lead_type: Optional[str] = None
    prospect_type: Optional[str] = None
    strategic_action: str
    retrieval_need: Optional[RetrievalNeed] = None
    conversation_evidence: Optional[str] = None
    deadline_ms: Optional[Union[int, float]] = None
    live_state_facts: Optional[List[Dict[str, Any]]] = None


class RetrievedFact(BaseModel):
    """Verified fact result passed to LLM response construction per Workflow Step 1.2."""
    knowledge_id: str
    text: str
    knowledge_class: str
    verification_status: Literal["user_supplied", "self_attested", "company_approved", "system_validated", "verified", "unverified"] = "user_supplied"
    source_title: str
    source_version: str = "1.0"
    confidence: float = 1.0
    is_self_attested: bool = False
    attestation_metadata: Optional[Dict[str, Any]] = None
    playbook_strategy_retained: bool = False
    substituted_for_playbook_id: Optional[str] = None

    @property
    def source_id(self) -> str:
        return self.knowledge_id

    @property
    def topic(self) -> str:
        return self.knowledge_class

    @property
    def authority_level(self) -> int:
        if self.verification_status in ("company_approved", "system_validated", "verified"):
            return 3
        if self.verification_status == "self_attested":
            return 2
        return 1


class WithheldFact(BaseModel):
    """Fact withheld from context during conflict/permission/guardrail checks per Step 1.2."""
    knowledge_id: str
    reason: Literal["disputed", "stale", "unverified", "restricted"]
    details: Optional[str] = None


class StageLatencies(BaseModel):
    """Per-stage latency measurements in milliseconds per Doc 01 §13."""
    gate_ms: float = 0.0
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    ranking_conflict_ms: float = 0.0
    packaging_ms: float = 0.0
    total_ms: float = 0.0


class RetrievalResult(BaseModel):
    """Structured output returned to LLM Response Construction per Workflow Step 1.2."""
    request_id: str
    state_version: int
    status: Literal[
        "retrieved",
        "none_needed",
        "none_found",
        "low_confidence",
        "conflict_unresolved",
        "timed_out",
        "error",
        "stale_discarded",
    ]
    facts: List[RetrievedFact] = Field(default_factory=list)
    withheld: List[WithheldFact] = Field(default_factory=list)
    fallback_hint: Optional[Literal["use_generic_truthful", "ask_clarifying_question", "choose_alternate_move"]] = None
    latency_ms: float = 0.0
    stage_latencies: Optional[StageLatencies] = None


class KnowledgeRecord(BaseModel):
    """Durable user-owned knowledge chunk per Doc 02 §5 and Client Scope C6."""
    knowledge_id: str = Field(default_factory=lambda: f"know_{uuid.uuid4().hex[:8]}")
    owner_scope: Literal["user"] = "user"
    owner_user_id: str
    knowledge_class: str
    source_title: str
    source_version: str = "1.0"
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    lead_type_tags: List[str] = Field(default_factory=list)
    prospect_type_tags: List[str] = Field(default_factory=list)
    topic_tags: List[str] = Field(default_factory=list)
    verification_status: Literal["user_supplied", "self_attested", "company_approved", "system_validated"] = "user_supplied"
    sensitivity: Literal["normal", "confidential", "restricted"] = "normal"
    text: str
    deleted: bool = False
    superseded_by: Optional[str] = None
    attestation_user_id: Optional[str] = None
    attestation_timestamp: Optional[str] = None
    attestation_text_hash: Optional[str] = None

    @model_validator(mode="before")
    def validate_scope(cls, values: Any) -> Any:
        if isinstance(values, dict):
            scope = str(values.get("owner_scope", "user")).lower()
            if scope in ("team", "brokerage", "company"):
                raise ValueError(
                    f"Team, brokerage, and company scopes are not enabled (Constraint C6); owner_scope must be 'user' (got '{scope}')"
                )
            if scope != "user":
                raise ValueError(f"Invalid owner_scope '{scope}', only 'user' is allowed.")
            if not values.get("owner_user_id"):
                raise ValueError("owner_user_id is mandatory for KnowledgeRecord (Constraint C6).")
        return values


class RankingWeights(BaseModel):
    """Configurable scoring weights for Doc 02 §7 ranking factors (tunable, no hardcoded spec assumptions)."""
    semantic_weight: float = Field(default=0.50, description="Cosine / token overlap similarity weight")
    lead_type_boost: float = Field(default=0.15, description="Boost for matching lead_type_tags")
    prospect_type_boost: float = Field(default=0.10, description="Boost for matching prospect_type_tags")
    verification_boost: float = Field(default=0.15, description="Boost for company_approved / system_validated")
    specificity_boost: float = Field(default=0.10, description="Boost for narrow topic tag overlap")
    recency_boost: float = Field(default=0.10, description="Boost for recency when time_sensitive=True")
    conflict_penalty: float = Field(default=0.30, description="Penalty for conflicting or stale chunks")


class RetrievalConfig(BaseModel):
    """Runtime configuration for Phase 3 & Phase 5 ranking thresholds, timeouts, and caps (Doc 02 §7, Doc 01 §13)."""
    relevance_floor: float = Field(default=0.35, description="Minimum raw semantic similarity floor; below this is dropped")
    marginal_threshold: float = Field(default=0.50, description="Threshold between low_confidence and retrieved for raw semantic similarity")
    max_facts: int = Field(default=3, description="Maximum number of facts returned to LLM context (1-3)")
    allow_user_supplied_performance: bool = Field(default=False, description="Whether user_supplied performance claims are allowed or withheld until verified")
    default_timeout_ms: Optional[float] = Field(default=600.0, description="Default latency budget deadline in milliseconds (Doc 01 §13 raised to 600ms per p95)")
    cache_enabled: bool = Field(default=True, description="Whether query-level caching is enabled (Doc 02 §11)")
    cache_ttl_seconds: int = Field(default=300, description="Maximum TTL for cached retrieval results in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum number of cached retrieval entries")
    weights: RankingWeights = Field(default_factory=RankingWeights)


class GateDecisionLog(BaseModel):
    """Audit entry recording the exact Gate evaluation reason for compliance & observability."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str
    call_id: str
    state_version: int
    user_id: str
    strategic_action: str
    trigger: Optional[str] = None
    gate_passed: bool
    status: str
    reason: str
    fallback_hint: Optional[str] = None
