from __future__ import annotations

import logging
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from .core_intelligence_models import RequiredFactScope, StrategicAction, StrategicDecision
from .retrieval import LocalKnowledgeBase, RetrievedChunk
from .conditional_retrieval_models import (
    GateDecisionLog,
    KnowledgeClass,
    KnowledgeRecord,
    RetrievalNeed,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrigger,
    RetrievedFact,
    WithheldFact,
)

LOGGER = logging.getLogger("copilot.conditional_retrieval")

# Pinned Embedding Model per Doc 02 §3.2 and config
PINNED_EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINNED_EMBEDDING_DIMENSION: int = 1536

WORD_REGEX = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_'-]+")


def _tokenize(text: str) -> Set[str]:
    return {w.casefold() for w in WORD_REGEX.findall(text) if len(w) > 2}


# ==============================================================================
# Legacy Machine Objects (Spec 11 §8 backward compatibility)
# ==============================================================================


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
    status: Optional[str] = None
    fallback_hint: Optional[str] = None
    latency_ms: float = 0.0


# ==============================================================================
# Phase 1: Retrieval Gate & Query Builder (Doc 02 §6)
# ==============================================================================


class RetrievalGate:
    """Deterministic gate sitting in front of vector/knowledge search (Doc 02 §6).

    Gate Rules:
    1. IF no trigger flagged                      -> none_needed (STOP)
    2. IF strategic_action IN {Wait, Acknowledge} -> none_needed (STOP)
    3. IF user has zero active, in-scope knowledge -> none_found  (STOP, skip vector search)
    4. ELSE                                       -> proceed to Query Builder
    """

    EXCLUDED_ACTIONS = {
        "wait_silence",
        "wait",
        "hold",
        "acknowledge",
    }

    def __init__(self) -> None:
        self.decision_logs: List[GateDecisionLog] = []

    @classmethod
    def resolve_fallback_hint(cls, strategic_action: str, trigger: Optional[str]) -> Optional[str]:
        """Resolves the truthful fallback hint per Doc 02 §9 and Step 2 & Step 7."""
        action_clean = str(strategic_action).lower().replace(" ", "_")
        trigger_clean = str(trigger).lower() if trigger else ""

        # 1. Social Proof or Playbook Evidence -> Core must choose an alternate move (never fabricate proof)
        if action_clean in ("social_proof", "socialproof") or trigger_clean == "playbook_evidence":
            return "choose_alternate_move"

        # 2. Specific questions or prior claim reference -> Truthfully ask a clarifying question
        if trigger_clean in ("factual_question", "prior_claim_reference"):
            return "ask_clarifying_question"

        # 3. Policy wording, credibility objections, service process objections -> Generic truthful language
        if trigger_clean in ("policy_wording", "credibility_objection", "service_process_objection"):
            return "use_generic_truthful"

        return "use_generic_truthful"

    def evaluate(
        self,
        request: RetrievalRequest,
        knowledge_store: Optional[KnowledgeStore] = None,
    ) -> Tuple[bool, str, Optional[str], str]:
        """Evaluates whether retrieval should run.

        Returns: (gate_passed, status, fallback_hint, reason)
        """
        # Rule 1: No trigger flagged
        if not request.retrieval_need or not request.retrieval_need.trigger:
            reason = "NO_TRIGGER_FLAGGED"
            self._log_decision(request, gate_passed=False, status="none_needed", reason=reason, fallback_hint=None)
            return False, "none_needed", None, reason

        trigger_val = str(getattr(request.retrieval_need.trigger, "value", request.retrieval_need.trigger)).lower()

        # Rule 2: Non-factual strategic action
        action_val = str(request.strategic_action).lower()
        if action_val in self.EXCLUDED_ACTIONS:
            reason = f"ACTION_EXCLUDED_FROM_RETRIEVAL: {action_val}"
            self._log_decision(request, gate_passed=False, status="none_needed", reason=reason, fallback_hint=None)
            return False, "none_needed", None, reason

        # Rule 3: Zero active in-scope knowledge for this individual user
        if knowledge_store is not None:
            has_knowledge = knowledge_store.has_active_knowledge(
                user_id=request.user_id,
                lead_type=request.lead_type,
            )
            if not has_knowledge:
                fallback_hint = self.resolve_fallback_hint(
                    strategic_action=request.strategic_action,
                    trigger=trigger_val,
                )
                reason = "USER_HAS_ZERO_ACTIVE_IN_SCOPE_KNOWLEDGE"
                self._log_decision(
                    request,
                    gate_passed=False,
                    status="none_found",
                    reason=reason,
                    fallback_hint=fallback_hint,
                )
                return False, "none_found", fallback_hint, reason

        # Gate Passed -> proceed to query builder and search
        reason = f"GATE_PASSED: trigger={trigger_val}"
        self._log_decision(request, gate_passed=True, status="retrieved", reason=reason, fallback_hint=None)
        return True, "retrieved", None, reason

    def _log_decision(
        self,
        request: RetrievalRequest,
        gate_passed: bool,
        status: str,
        reason: str,
        fallback_hint: Optional[str],
    ) -> None:
        trigger_str = None
        if request.retrieval_need and request.retrieval_need.trigger:
            trigger_str = str(getattr(request.retrieval_need.trigger, "value", request.retrieval_need.trigger))

        # IMPORTANT: Store only IDs and reasons, NEVER raw confidential chunk text or transcript
        log_entry = GateDecisionLog(
            request_id=request.request_id,
            call_id=request.call_id,
            state_version=request.state_version,
            user_id=request.user_id,
            strategic_action=request.strategic_action,
            trigger=trigger_str,
            gate_passed=gate_passed,
            status=status,
            reason=reason,
            fallback_hint=fallback_hint,
        )
        self.decision_logs.append(log_entry)
        LOGGER.info(
            "RetrievalGate [call=%s v=%d user=%s]: passed=%s status=%s reason='%s'",
            request.call_id,
            request.state_version,
            request.user_id,
            gate_passed,
            status,
            reason,
        )


class QueryBuilder:
    """Builds targeted query and metadata filters per Workflow Step 3."""

    # Trigger to candidate KnowledgeClass mapping per Doc 02 §3 & Step 3.3
    TRIGGER_TO_CLASSES: Dict[str, List[str]] = {
        "credibility_objection": [
            KnowledgeClass.SALESPERSON_PROFILE.value,
            KnowledgeClass.PERFORMANCE_PAST_SALES.value,
            KnowledgeClass.VALUE_PROPOSITION.value,
            KnowledgeClass.CLIENT_TESTIMONIALS.value,
        ],
        "service_process_objection": [
            KnowledgeClass.SERVICE_PROCESS.value,
            KnowledgeClass.PRICING_PACKAGING.value,
            KnowledgeClass.COMPANY_POLICY.value,
        ],
        "playbook_evidence": [
            KnowledgeClass.CLIENT_TESTIMONIALS.value,
            KnowledgeClass.PERFORMANCE_PAST_SALES.value,
            KnowledgeClass.VALUE_PROPOSITION.value,
            KnowledgeClass.MARKET_KNOWLEDGE.value,
        ],
        "policy_wording": [
            KnowledgeClass.COMPANY_POLICY.value,
            KnowledgeClass.PRICING_PACKAGING.value,
        ],
        "prior_claim_reference": [
            KnowledgeClass.SERVICE_PROCESS.value,
            KnowledgeClass.COMPANY_POLICY.value,
            KnowledgeClass.PRICING_PACKAGING.value,
            KnowledgeClass.SALESPERSON_PROFILE.value,
        ],
        "factual_question": [
            KnowledgeClass.SALESPERSON_PROFILE.value,
            KnowledgeClass.COMPANY_POLICY.value,
            KnowledgeClass.SERVICE_PROCESS.value,
            KnowledgeClass.PRICING_PACKAGING.value,
            KnowledgeClass.MARKET_KNOWLEDGE.value,
        ],
    }

    @classmethod
    def build_query(cls, request: RetrievalRequest) -> Tuple[str, List[str]]:
        """Constructs trimmed query string and target knowledge classes.

        Never passes the whole transcript (Constraint C4 & Step 3.1).
        """
        need = request.retrieval_need
        parts: List[str] = []
        if need:
            if need.need_statement:
                parts.append(need.need_statement.strip())
            if need.topic_tags:
                parts.extend([t.replace("_", " ") for t in need.topic_tags])

        # Short excerpt of prospect statement (trimmed to <= 200 chars)
        if request.conversation_evidence:
            trimmed_evidence = request.conversation_evidence.strip()[:200]
            parts.append(trimmed_evidence)

        query_text = " ".join(parts).strip()

        # Derive target knowledge classes
        trigger_val = str(getattr(need.trigger, "value", need.trigger)).lower() if need else ""
        target_classes = cls.TRIGGER_TO_CLASSES.get(trigger_val, [])

        return query_text, target_classes

    @classmethod
    def generate_query_embedding(
        cls,
        query: str,
        embedder: Optional[Callable[[str], List[float]]] = None,
    ) -> Optional[List[float]]:
        """Generates dense vector embedding using pinned model text-embedding-3-small (Step 3.2)."""
        if embedder is not None:
            return embedder(query)
        return None


# ==============================================================================
# Phase 2: Knowledge Store & Hard Pre-Filters (Doc 02 §5, §7, Constraint C6)
# ==============================================================================


class KnowledgeStore:
    """In-memory user-partitioned knowledge repository enforcing Doc 02 hard pre-filters."""

    def __init__(self) -> None:
        self._records: Dict[str, KnowledgeRecord] = {}

    def add_record(self, record: KnowledgeRecord) -> None:
        """Stores a verified user-owned knowledge chunk.

        Strictly enforces Constraint C6: only 'user' scope allowed.
        """
        if record.owner_scope != "user":
            raise ValueError(
                f"Team, brokerage, and company scopes are not enabled (Constraint C6); owner_scope must be 'user' (got '{record.owner_scope}')"
            )
        if not record.owner_user_id:
            raise ValueError("owner_user_id is mandatory for KnowledgeRecord.")
        self._records[record.knowledge_id] = record

    def delete_record(self, knowledge_id: str, soft: bool = True) -> bool:
        """Deletes a knowledge chunk. If soft=True, marks tombstone per Step 10."""
        if knowledge_id not in self._records:
            return False
        if soft:
            self._records[knowledge_id].deleted = True
        else:
            del self._records[knowledge_id]
        return True

    def supersede_record(self, old_knowledge_id: str, new_knowledge_id: str) -> bool:
        """Marks old chunk as superseded by newer version (Step 10)."""
        if old_knowledge_id in self._records:
            self._records[old_knowledge_id].superseded_by = new_knowledge_id
            return True
        return False

    def has_active_knowledge(
        self,
        user_id: str,
        lead_type: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
    ) -> bool:
        """Cheap metadata check to determine if user has any active eligible chunks."""
        now_utc = datetime.now(timezone.utc)
        for rec in self._records.values():
            if rec.owner_user_id != user_id or rec.owner_scope != "user":
                continue
            if rec.deleted or rec.superseded_by is not None:
                continue
            if rec.sensitivity == "restricted":
                continue

            # Validity window check (Null-safe, UTC-consistent)
            if rec.effective_from:
                try:
                    from_dt = datetime.fromisoformat(rec.effective_from.replace("Z", "+00:00"))
                    if from_dt.tzinfo is None:
                        from_dt = from_dt.replace(tzinfo=timezone.utc)
                    if now_utc < from_dt:
                        continue
                except (ValueError, TypeError):
                    pass
            if rec.effective_to:
                try:
                    to_dt = datetime.fromisoformat(rec.effective_to.replace("Z", "+00:00"))
                    if to_dt.tzinfo is None:
                        to_dt = to_dt.replace(tzinfo=timezone.utc)
                    if now_utc > to_dt:
                        continue
                except (ValueError, TypeError):
                    pass

            # Lead-type filter semantics: Untagged chunks ALWAYS pass!
            if rec.lead_type_tags and lead_type and lead_type not in rec.lead_type_tags:
                continue

            if target_classes and rec.knowledge_class not in target_classes:
                continue
            return True
        return False

    def search_candidates(
        self,
        query: str,
        user_id: str,
        target_classes: Optional[List[str]] = None,
        lead_type: Optional[str] = None,
        topic_tags: Optional[List[str]] = None,
        allow_confidential: bool = False,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[KnowledgeRecord, float]]:
        """Executes candidate retrieval with strict hard pre-filters applied inside query (Step 4 & 5)."""
        now_utc = datetime.now(timezone.utc)
        query_tokens = _tokenize(query)

        scored_candidates: List[Tuple[KnowledgeRecord, float]] = []

        for rec in self._records.values():
            # --- Hard Pre-Filters (Step 4 & Constraint C6) ---
            # 1. Ownership: MUST match current user_id exactly; team/company scopes never match
            if rec.owner_scope != "user" or rec.owner_user_id != user_id:
                continue

            # 2. Deletion: exclude soft-deleted chunks or tombstones
            if rec.deleted:
                continue

            # 3. Supersession: exclude chunks superseded by newer versions
            if rec.superseded_by is not None:
                continue

            # 4. Validity window (Null-safe, UTC-consistent)
            if rec.effective_from:
                try:
                    from_dt = datetime.fromisoformat(rec.effective_from.replace("Z", "+00:00"))
                    if from_dt.tzinfo is None:
                        from_dt = from_dt.replace(tzinfo=timezone.utc)
                    if now_utc < from_dt:
                        continue
                except (ValueError, TypeError):
                    pass
            if rec.effective_to:
                try:
                    to_dt = datetime.fromisoformat(rec.effective_to.replace("Z", "+00:00"))
                    if to_dt.tzinfo is None:
                        to_dt = to_dt.replace(tzinfo=timezone.utc)
                    if now_utc > to_dt:
                        continue
                except (ValueError, TypeError):
                    pass

            # 5. Sensitivity: restricted always excluded; confidential requires explicit allow
            if rec.sensitivity == "restricted":
                continue
            if rec.sensitivity == "confidential" and not allow_confidential:
                continue

            # 6. Lead-type constraint (Step 4 & Step 3.3):
            # A chunk with NO lead_type_tags (empty list) MUST ALWAYS PASS!
            # Only chunks that explicitly declare lead_type_tags AND where lead_type
            # is not in those tags are excluded.
            if rec.lead_type_tags and lead_type:
                if lead_type not in rec.lead_type_tags:
                    continue

            # --- Similarity & Match Scoring ---
            doc_text = f"{rec.text} {' '.join(rec.topic_tags)}"
            doc_tokens = _tokenize(doc_text)
            if not doc_tokens:
                continue

            overlap = query_tokens & doc_tokens
            if not overlap and query_tokens:
                # Check topic tag match as fallback
                tag_match = any(t.casefold() in query.casefold() for t in rec.topic_tags) if rec.topic_tags else False
                if not tag_match:
                    continue
                score = 0.35
            else:
                precision = len(overlap) / math.sqrt(max(1, len(doc_tokens)))
                coverage = len(overlap) / max(1, len(query_tokens))
                score = (precision + coverage) / 2.0

            # Boost for matching target knowledge class
            if target_classes and rec.knowledge_class in target_classes:
                score += 0.15

            # Boost for matching topic tags
            if topic_tags and rec.topic_tags:
                tag_overlap = set(t.casefold() for t in topic_tags) & set(t.casefold() for t in rec.topic_tags)
                if tag_overlap:
                    score += 0.10

            score = min(round(score, 3), 1.0)
            if score >= min_score:
                scored_candidates.append((rec, score))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]


# ==============================================================================
# Conditional Retrieval Engine (Adapter for Phase 1, Phase 2, and Spec 11)
# ==============================================================================


class ConditionalRetrievalEngine:
    """Main conditional retrieval engine coordinating Gate, Store, and Conflict Resolution."""

    def __init__(
        self,
        backend: Optional[Any] = None,
        store: Optional[KnowledgeStore] = None,
        gate: Optional[RetrievalGate] = None,
        default_tenant_id: str = "default",
        default_user_id: str = "default_user",
    ):
        self.backend = backend or LocalKnowledgeBase()
        self.store = store or KnowledgeStore()
        self.gate = gate or RetrievalGate()
        self.default_tenant_id = default_tenant_id
        self.default_user_id = default_user_id

    # --------------------------------------------------------------------------
    # Phase 1 & 2 Execution: Core Request -> Retrieval Gate -> Safe Store Search
    # --------------------------------------------------------------------------

    def retrieve_for_request(
        self,
        request: RetrievalRequest,
        allow_confidential: bool = False,
        relevance_floor: float = 0.20,
    ) -> RetrievalResult:
        """Executes retrieval workflow per Doc 02 and Client Implementation Contract."""
        start_time = time.perf_counter()

        # Step 1: Retrieval Gate check (Doc 02 §6)
        gate_passed, status, fallback_hint, reason = self.gate.evaluate(
            request=request,
            knowledge_store=self.store,
        )

        # Fast exit if Gate says no retrieval needed or no knowledge exists
        if not gate_passed:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResult(
                request_id=request.request_id,
                state_version=request.state_version,
                status=status,  # "none_needed" or "none_found"
                facts=[],
                withheld=[],
                fallback_hint=fallback_hint,
                latency_ms=round(elapsed_ms, 2),
            )

        # Step 2: Query Builder (Doc 02 §6 & Step 3)
        query_text, target_classes = QueryBuilder.build_query(request)
        topic_tags = request.retrieval_need.topic_tags if request.retrieval_need else []

        # Step 3: Candidate search with hard pre-filters applied in-query (Step 4 & 5)
        candidates = self.store.search_candidates(
            query=query_text,
            user_id=request.user_id,
            target_classes=target_classes,
            lead_type=request.lead_type,
            topic_tags=topic_tags,
            allow_confidential=allow_confidential,
            top_k=5,
            min_score=relevance_floor,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not candidates:
            trigger_val = str(getattr(request.retrieval_need.trigger, "value", request.retrieval_need.trigger)).lower() if request.retrieval_need else ""
            fallback = RetrievalGate.resolve_fallback_hint(request.strategic_action, trigger_val)
            return RetrievalResult(
                request_id=request.request_id,
                state_version=request.state_version,
                status="none_found",
                facts=[],
                withheld=[],
                fallback_hint=fallback,
                latency_ms=round(elapsed_ms, 2),
            )

        # Step 4: Package facts (cap at top 3 per Doc 02 §7)
        retrieved_facts: List[RetrievedFact] = []
        for rec, score in candidates[:3]:
            fact = RetrievedFact(
                knowledge_id=rec.knowledge_id,
                text=rec.text,
                knowledge_class=rec.knowledge_class,
                verification_status=rec.verification_status,
                source_title=rec.source_title,
                source_version=rec.source_version,
                confidence=score,
            )
            retrieved_facts.append(fact)

        return RetrievalResult(
            request_id=request.request_id,
            state_version=request.state_version,
            status="retrieved",
            facts=retrieved_facts,
            withheld=[],
            fallback_hint=None,
            latency_ms=round(elapsed_ms, 2),
        )

    # --------------------------------------------------------------------------
    # Backward-Compatible Method: StrategicDecision -> Wraps Gate & Pre-Filters
    # --------------------------------------------------------------------------

    def retrieve_for_decision(
        self,
        decision: StrategicDecision,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> RetrievalExecutionReport:
        """Wraps retrieve_for_request to guarantee Gate & ownership enforcement (Constraint C1 & C6)."""
        start_time = time.perf_counter()
        active_tenant = tenant_id or self.default_tenant_id
        active_user = (
            user_id
            or getattr(decision, "user_id", None)
            or (decision.metadata.get("user_id") if hasattr(decision, "metadata") and isinstance(decision.metadata, dict) else None)
        )

        action_str = str(getattr(decision.primary_action, "value", decision.primary_action)).lower()

        # Build RetrievalNeed from decision properties
        if not decision.retrieval_needed or not decision.required_facts:
            retrieval_need = None
        else:
            # Map strategic action to canonical trigger
            if action_str in ("social_proof", "socialproof"):
                trigger = RetrievalTrigger.PLAYBOOK_EVIDENCE
            elif action_str in ("quantify", "educate", "differentiate"):
                trigger = RetrievalTrigger.FACTUAL_QUESTION
            elif any("fee" in str(f).lower() or "price" in str(f).lower() for f in decision.required_facts):
                trigger = RetrievalTrigger.SERVICE_PROCESS_OBJECTION
            else:
                trigger = RetrievalTrigger.CREDIBILITY_OBJECTION

            topic_tags = [f.topic if hasattr(f, "topic") else str(f) for f in decision.required_facts]
            retrieval_need = RetrievalNeed(
                trigger=trigger,
                need_statement=decision.strategic_objective or "Verified business facts",
                topic_tags=topic_tags,
            )

        # Formulate formal RetrievalRequest
        request = RetrievalRequest(
            request_id=f"req_{decision.decision_id}",
            call_id=decision.call_sid or decision.call_id,
            state_version=decision.source_state_version,
            user_id=active_user or "default_user",
            strategic_action=action_str,
            retrieval_need=retrieval_need,
            conversation_evidence=decision.evidence_considered[0] if decision.evidence_considered else None,
        )

        # Checkpoint 1: Execute through RetrievalGate (NEVER bypasses the Gate)
        gate_passed, gate_status, fallback_hint, reason = self.gate.evaluate(
            request=request,
            knowledge_store=self.store if self.store.has_active_knowledge(active_user) else None,
        )

        if not gate_passed:
            return RetrievalExecutionReport(
                retrieval_needed=False,
                decision_id=decision.decision_id,
                source_state_version=decision.source_state_version,
                queries_executed=[],
                facts=[],
                status=gate_status,
                fallback_hint=fallback_hint,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        # If knowledge store has candidates for this user, retrieve them directly
        if self.store.has_active_knowledge(active_user):
            res = self.retrieve_for_request(request)
            legacy_facts = [
                RetrievedFactResult(
                    fact_id=f.knowledge_id,
                    topic=f.knowledge_class,
                    text=f.text,
                    source_id=f.source_title,
                    verification_status="verified" if f.verification_status in ("company_approved", "system_validated", "verified") else "unverified",
                    relevance_score=f.confidence,
                    authority_level=f.authority_level,
                    metadata={"user_id": active_user},
                )
                for f in res.facts
            ]
            queries = [f"{t} proof" for t in request.retrieval_need.topic_tags] if request.retrieval_need else []
            return RetrievalExecutionReport(
                retrieval_needed=True,
                decision_id=decision.decision_id,
                source_state_version=decision.source_state_version,
                queries_executed=queries,
                facts=legacy_facts,
                status=res.status,
                fallback_hint=res.fallback_hint,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        # Fallback to configured backend with server-side / mock ownership filtering
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
            raw_chunks = self._query_backend(
                query=query,
                tenant_id=active_tenant,
                user_id=active_user,
                min_score=min_score,
            )

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
            status="retrieved" if resolved_facts else "none_found",
            fallback_hint="use_generic_truthful" if not resolved_facts else None,
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
                resolved_facts.append(top_tier[0])
                LOGGER.info("Resolved conflict on '%s' via authority: %s (auth %d) beats competitors", topic, top_tier[0].source_id, top_authority)
            else:
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
        user_id: Optional[str] = None,
        min_score: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """Queries backend with in-query server-side ownership pre-filtering (Doc 02 Step 4 & C6)."""
        results: List[Dict[str, Any]] = []

        if hasattr(self.backend, "search"):
            try:
                # Production vector store search passing owner_id and scope in-query
                chunks = self.backend.search(
                    query=query,
                    tenant_id=tenant_id,
                    owner_id=user_id,
                    scope="user",
                    limit=5,
                )
            except TypeError:
                try:
                    chunks = self.backend.search(query=query, tenant_id=tenant_id, limit=5)
                except TypeError:
                    chunks = self.backend.search(query=query, limit=5)

            for ch in chunks:
                score = getattr(ch, "score", 0.85)
                if score >= min_score:
                    meta = getattr(ch, "metadata", {})
                    # Hard pre-filter check: enforce owner isolation
                    chunk_owner = meta.get("owner_id", getattr(ch, "owner_id", ""))
                    if chunk_owner and user_id and chunk_owner != user_id and chunk_owner != "admin":
                        continue

                    results.append({
                        "text": getattr(ch, "text", str(ch)),
                        "source": getattr(ch, "source", "vector_store"),
                        "score": round(score, 3),
                        "scope": meta.get("scope", getattr(ch, "file_type", "")),
                        "owner_id": chunk_owner,
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

                    # In-query hard pre-filter for dictionary backend:
                    # Chunks owned by a different user are excluded immediately
                    if owner_id and user_id and owner_id != user_id and owner_id != "admin":
                        continue

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
