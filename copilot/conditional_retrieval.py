from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
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
    verification_status: Literal["verified", "unverified", "disputed"] = "unverified"
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
    disputed_topics: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class ConditionalRetrievalEngine:
    """Conditional retrieval engine adapter executing scoped queries with Spec 06 authority resolution."""

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
        raw_retrieved_facts: List[RetrievedFactResult] = []

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
                status, auth_level = self._resolve_verification(chunk)

                # If verification is required, unverified low-authority chunks are held back
                if verification_req and status != "verified" and auth_level < 3:
                    LOGGER.debug("Chunk %s held back: verification required but status is %s", chunk.get("source"), status)
                    continue

                fact = RetrievedFactResult(
                    topic=topic,
                    text=chunk["text"],
                    source_id=chunk["source"],
                    verification_status=status,
                    relevance_score=chunk["score"],
                    authority_level=auth_level,
                    metadata={"evidence_type": evidence_type, "tenant_id": active_tenant, **chunk.get("metadata", {})},
                )
                raw_retrieved_facts.append(fact)

        # Conflict resolution pass over retrieved facts (Spec 06 & Spec 11 §8)
        resolved_facts, disputed_topics = self._resolve_conflicts(raw_retrieved_facts)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalExecutionReport(
            retrieval_needed=True,
            decision_id=decision.decision_id,
            source_state_version=decision.source_state_version,
            queries_executed=executed_queries,
            facts=resolved_facts,
            disputed_topics=disputed_topics,
            latency_ms=round(elapsed_ms, 2),
        )

    def _resolve_verification(self, chunk: Dict[str, Any]) -> Tuple[Literal["verified", "unverified", "disputed"], int]:
        """Maps scope and source metadata to authority level and verification tier (Spec 06 / Spec 10)."""
        meta = chunk.get("metadata", {})
        scope = str(meta.get("scope") or chunk.get("scope") or "").lower()
        owner_id = str(meta.get("owner_id") or chunk.get("owner_id") or "").lower()
        explicit_auth = meta.get("authority_level") or chunk.get("authority_level")

        if explicit_auth is not None:
            auth_level = int(explicit_auth)
        elif scope in ("compliance", "legal"):
            auth_level = 4
        elif scope in ("admin", "verified_proof", "verified") or owner_id == "admin":
            auth_level = 3
        elif scope in ("playbook", "training"):
            auth_level = 2
        else:
            auth_level = 1

        explicit_status = meta.get("verification_status") or chunk.get("verification_status")
        if explicit_status in ("verified", "unverified", "disputed"):
            status = explicit_status
        elif auth_level >= 3:
            status = "verified"
        else:
            status = "unverified"

        return status, auth_level

    def _resolve_conflicts(
        self,
        facts: List[RetrievedFactResult],
    ) -> Tuple[List[RetrievedFactResult], List[str]]:
        """Resolves conflicting documents per Spec 06 and Spec 11 §8.

        Where two sources on the same topic conflict:
        - If authority levels differ, the higher authority level wins and lower is discarded.
        - On a genuine authority tie, both are marked disputed and dropped from verified context.
        """
        if len(facts) <= 1:
            return facts, []

        grouped_by_topic: Dict[str, List[RetrievedFactResult]] = {}
        for f in facts:
            grouped_by_topic.setdefault(f.topic, []).append(f)

        resolved_facts: List[RetrievedFactResult] = []
        disputed_topics: List[str] = []

        for topic, topic_facts in grouped_by_topic.items():
            if len(topic_facts) == 1:
                resolved_facts.append(topic_facts[0])
                continue

            has_conflict = self._detect_conflict(topic_facts)
            if not has_conflict:
                resolved_facts.extend(topic_facts)
                continue

            # Sort by authority level descending, then relevance score
            sorted_facts = sorted(topic_facts, key=lambda x: (x.authority_level, x.relevance_score), reverse=True)
            top_authority = sorted_facts[0].authority_level
            top_tier = [f for f in sorted_facts if f.authority_level == top_authority]

            if len(top_tier) == 1:
                # Single authoritative winner
                resolved_facts.append(top_tier[0])
                LOGGER.info("Resolved conflict on '%s' via authority: %s (auth %d) beats competitors", topic, top_tier[0].source_id, top_authority)
            else:
                # Genuine tie between conflicting claims at the same authority level
                # Spec 11 §8: avoid asserting the disputed fact; drop both from verified knowledge
                LOGGER.warning("Authority tie on conflicting topic '%s': dropping %d disputed facts from prompt context", topic, len(top_tier))
                for f in top_tier:
                    f.verification_status = "disputed"
                disputed_topics.append(topic)

        return resolved_facts, disputed_topics

    def _detect_conflict(self, facts: List[RetrievedFactResult]) -> bool:
        if len(facts) < 2:
            return False
        if any(f.verification_status == "disputed" or f.metadata.get("is_conflicting") for f in facts):
            return True

        # Detect conflicting numerical or factual claims (e.g. 4.5% vs 3.9%, 14 days vs 45 days)
        token_sets = [set(re.findall(r"\b\d+(?:\.\d+)?%?", f.text)) for f in facts]
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                if token_sets[i] and token_sets[j] and token_sets[i] != token_sets[j]:
                    return True
        return False

    def _query_backend(
        self,
        query: str,
        tenant_id: str,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if hasattr(self.backend, "search"):
            try:
                chunks = self.backend.search(query=query, tenant_id=tenant_id, limit=5)
            except TypeError:
                chunks = self.backend.search(query=query, limit=5)

            for ch in chunks:
                score = getattr(ch, "score", 0.85)
                if score >= min_score:
                    meta = getattr(ch, "metadata", {})
                    results.append({
                        "text": getattr(ch, "text", str(ch)),
                        "source": getattr(ch, "source", "vector_store"),
                        "score": round(score, 3),
                        "scope": meta.get("scope", getattr(ch, "file_type", "")),
                        "owner_id": meta.get("owner_id", ""),
                        "metadata": meta,
                    })

        elif isinstance(self.backend, dict):
            for k, val in self.backend.items():
                if any(word in k.lower() for word in query.lower().split()):
                    if isinstance(val, dict):
                        text = val.get("text", "")
                        score = val.get("score", 0.90)
                        scope = val.get("scope", "general")
                        owner_id = val.get("owner_id", "")
                        auth = val.get("authority_level")
                        status = val.get("verification_status")
                        meta = {k: v for k, v in val.items() if k not in ("text", "score")}
                    else:
                        text = str(val)
                        score = 0.90
                        scope = "general"
                        owner_id = ""
                        auth = None
                        status = None
                        meta = {}

                    if score >= min_score:
                        item = {
                            "text": text,
                            "source": f"in_memory_{k}",
                            "score": score,
                            "scope": scope,
                            "owner_id": owner_id,
                            "metadata": meta,
                        }
                        if auth is not None:
                            item["authority_level"] = auth
                        if status is not None:
                            item["verification_status"] = status
                        results.append(item)

        return results
