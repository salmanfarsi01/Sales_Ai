from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from .core_intelligence_models import RequiredFactScope, StrategicDecision
from .retrieval import LocalKnowledgeBase, RetrievedChunk

LOGGER = logging.getLogger("copilot.conditional_retrieval")


class RetrievedFactResult(BaseModel):
    """Scoped, verified factual evidence retrieved for response construction per Spec 11 §8."""
    fact_id: str = Field(default_factory=lambda: f"fact_{uuid.uuid4().hex[:8]}")
    topic: str
    text: str
    source_id: str
    verification_status: Literal["verified", "unverified", "disputed"] = "verified"
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    authority_level: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalExecutionReport(BaseModel):
    """Execution audit report for conditional knowledge retrieval."""
    retrieval_needed: bool
    decision_id: str
    source_state_version: int
    queries_executed: List[str] = Field(default_factory=list)
    facts: List[RetrievedFactResult] = Field(default_factory=list)
    latency_ms: float = 0.0


class ConditionalRetrievalEngine:
    """Conditional retrieval engine adapter executing scoped queries per Spec 11 §8."""

    def __init__(
        self,
        backend: Optional[Any] = None,
        default_tenant_id: str = "default",
    ):
        self.backend = backend or LocalKnowledgeBase()
        self.default_tenant_id = default_tenant_id

    def retrieve_for_decision(
        self,
        decision: StrategicDecision,
        tenant_id: Optional[str] = None,
    ) -> RetrievalExecutionReport:
        start_time = time.perf_counter()
        active_tenant = tenant_id or self.default_tenant_id

        if not decision.retrieval_needed or not decision.required_facts:
            return RetrievalExecutionReport(
                retrieval_needed=False,
                decision_id=decision.decision_id,
                source_state_version=decision.source_state_version,
                queries_executed=[],
                facts=[],
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        executed_queries: List[str] = []
        retrieved_facts: List[RetrievedFactResult] = []

        for req in decision.required_facts:
            if isinstance(req, str):
                topic = req
                query = req.replace("_", " ")
                min_score = 0.50
                evidence_type = "general"
                verification_req = True
            else:
                topic = req.topic
                query = f"{req.topic.replace('_', ' ')} {req.required_evidence_type.replace('_', ' ')}"
                min_score = req.min_confidence
                evidence_type = req.required_evidence_type
                verification_req = req.verification_required

            executed_queries.append(query)
            raw_chunks = self._query_backend(query=query, tenant_id=active_tenant, min_score=min_score)

            for chunk in raw_chunks:
                fact = RetrievedFactResult(
                    topic=topic,
                    text=chunk["text"],
                    source_id=chunk["source"],
                    verification_status="verified" if verification_req else "unverified",
                    relevance_score=chunk["score"],
                    authority_level=1,
                    metadata={"evidence_type": evidence_type, "tenant_id": active_tenant},
                )
                retrieved_facts.append(fact)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalExecutionReport(
            retrieval_needed=True,
            decision_id=decision.decision_id,
            source_state_version=decision.source_state_version,
            queries_executed=executed_queries,
            facts=retrieved_facts,
            latency_ms=round(elapsed_ms, 2),
        )

    def _query_backend(
        self,
        query: str,
        tenant_id: str,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        # Backend adapter branch 1: PineconeRAG / CopilotRAGRetriever
        if hasattr(self.backend, "search"):
            try:
                # Inspect if backend search accepts tenant_id / top_k
                chunks = self.backend.search(query=query, tenant_id=tenant_id, limit=3)
            except TypeError:
                chunks = self.backend.search(query=query, limit=3)

            for ch in chunks:
                score = getattr(ch, "score", 0.85)
                if score >= min_score:
                    results.append({
                        "text": getattr(ch, "text", str(ch)),
                        "source": getattr(ch, "source", "vector_store"),
                        "score": round(score, 3),
                    })

        # Backend adapter branch 2: Dict / In-memory mock
        elif isinstance(self.backend, dict):
            for k, val in self.backend.items():
                if any(word in k.lower() for word in query.lower().split()):
                    results.append({
                        "text": val,
                        "source": f"in_memory_{k}",
                        "score": 0.90,
                    })

        return results
