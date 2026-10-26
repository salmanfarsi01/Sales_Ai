from __future__ import annotations

import hashlib
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
    RankingWeights,
    RetrievalConfig,
    RetrievalNeed,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrigger,
    RetrievedFact,
    StageLatencies,
    WithheldFact,
)
from .compliance_rules import (
    evaluate_content_compliance,
    detect_prompt_injection,
    classify_attested_claim,
)

LOGGER = logging.getLogger("copilot.conditional_retrieval")

# Pinned Embedding Model per Doc 02 §3.2 and config
PINNED_EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINNED_EMBEDDING_DIMENSION: int = 1536

WORD_REGEX = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'-]+")


def _tokenize(text: str) -> Set[str]:
    clean = text.replace("_", " ")
    return {w.casefold() for w in WORD_REGEX.findall(clean) if len(w) > 2}


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


class RetrievalCache:
    """Thread-safe user-partitioned retrieval cache with immediate invalidation (Doc 02 §11, Step 10)."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[RetrievalResult, float]] = {}
        self._user_keys: Dict[str, Set[str]] = {}

    def _make_key(
        self,
        request: RetrievalRequest,
        allow_confidential: bool = False,
        relevance_floor: Optional[float] = None,
        config: Optional[RetrievalConfig] = None,
        user_knowledge_version: int = 1,
    ) -> str:
        need_str = ""
        if request.retrieval_need:
            t = getattr(request.retrieval_need.trigger, "value", request.retrieval_need.trigger)
            need_str = f"{t}:{request.retrieval_need.need_statement}:{sorted(request.retrieval_need.topic_tags)}"
        cfg_sig = ""
        if config:
            cfg_sig = f"{config.allow_user_supplied_performance}:{config.marginal_threshold}:{config.relevance_floor}:{hash(str(config.weights.model_dump()))}"
        return f"{request.user_id}:v{user_knowledge_version}:{request.state_version}:{request.strategic_action}:{request.lead_type}:{allow_confidential}:{relevance_floor}:{cfg_sig}:{need_str}"

    def get(
        self,
        request: RetrievalRequest,
        allow_confidential: bool = False,
        relevance_floor: Optional[float] = None,
        config: Optional[RetrievalConfig] = None,
        user_knowledge_version: int = 1,
    ) -> Optional[RetrievalResult]:
        key = self._make_key(
            request,
            allow_confidential=allow_confidential,
            relevance_floor=relevance_floor,
            config=config,
            user_knowledge_version=user_knowledge_version,
        )
        if key not in self._cache:
            return None
        result, created_at = self._cache[key]
        if time.time() - created_at > self.ttl_seconds:
            self.delete(key)
            return None
        return result

    def set(
        self,
        request: RetrievalRequest,
        result: RetrievalResult,
        allow_confidential: bool = False,
        relevance_floor: Optional[float] = None,
        config: Optional[RetrievalConfig] = None,
        user_knowledge_version: int = 1,
    ) -> None:
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            self.delete(oldest_key)
        key = self._make_key(
            request,
            allow_confidential=allow_confidential,
            relevance_floor=relevance_floor,
            config=config,
            user_knowledge_version=user_knowledge_version,
        )
        self._cache[key] = (result, time.time())
        if request.user_id not in self._user_keys:
            self._user_keys[request.user_id] = set()
        self._user_keys[request.user_id].add(key)

    def invalidate_user(self, user_id: str) -> int:
        """Immediately flushes all cached results for a specific user upon edit/delete/permission change."""
        keys = self._user_keys.pop(user_id, set())
        for k in keys:
            self._cache.pop(k, None)
        return len(keys)

    def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
        for user_id, keys in list(self._user_keys.items()):
            keys.discard(key)

    def clear(self) -> None:
        self._cache.clear()
        self._user_keys.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self._cache),
            "users_cached": len(self._user_keys),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


class KnowledgeStore:
    """In-memory user-partitioned knowledge repository enforcing Doc 02 hard pre-filters."""

    def __init__(
        self,
        cache: Optional[RetrievalCache] = None,
        pinecone_rag: Optional[Any] = None,
        sync_to_pinecone: bool = True,
    ) -> None:
        self._records: Dict[str, KnowledgeRecord] = {}
        self._user_knowledge_versions: Dict[str, int] = {}
        self.cache: Optional[RetrievalCache] = cache
        self.pinecone_rag = pinecone_rag
        self.sync_to_pinecone = sync_to_pinecone
        if self.pinecone_rag is None and self.sync_to_pinecone:
            pk = os.getenv("PINECONE_API_KEY")
            ok = os.getenv("OPENAI_API_KEY")
            if pk and ok and not pk.startswith("mock") and not ok.startswith("mock"):
                try:
                    from .rag_pinecone import PineconeRAG
                    self.pinecone_rag = PineconeRAG(pinecone_api_key=pk, openai_api_key=ok)
                except Exception as exc:
                    LOGGER.debug("Could not initialize PineconeRAG for KnowledgeStore: %s", exc)

    def get_user_knowledge_version(self, user_id: str) -> int:
        return self._user_knowledge_versions.get(user_id, 1)

    def _bump_user_knowledge_version(self, user_id: str) -> int:
        new_v = self._user_knowledge_versions.get(user_id, 1) + 1
        self._user_knowledge_versions[user_id] = new_v
        return new_v

    def bump_user_knowledge_version(self, user_id: str) -> int:
        """Public method to increment user knowledge version and invalidate cache."""
        self._notify_invalidation(user_id)
        return self.get_user_knowledge_version(user_id)

    def _notify_invalidation(self, user_id: Optional[str]) -> None:
        if user_id:
            self._bump_user_knowledge_version(user_id)
            if self.cache:
                self.cache.invalidate_user(user_id)

    def get_record(self, knowledge_id: str) -> Optional[KnowledgeRecord]:
        return self._records.get(knowledge_id)

    def add_record(self, record: KnowledgeRecord) -> None:
        """Stores a verified user-owned knowledge chunk.

        Strictly enforces Constraint C6: only 'user' scope allowed.
        Invalidates attestation if text hash mismatches or text changed on re-ingestion.
        Immediately invalidates the user's retrieval cache (Step 10).
        """
        if record.owner_scope != "user":
            raise ValueError(
                f"Team, brokerage, and company scopes are not enabled (Constraint C6); owner_scope must be 'user' (got '{record.owner_scope}')"
            )
        if not record.owner_user_id:
            raise ValueError("owner_user_id is mandatory for KnowledgeRecord.")

        # Invalidate attestation on unverified hash or text tampering
        if record.verification_status == "self_attested":
            expected_hash = hashlib.sha256(record.text.strip().encode("utf-8")).hexdigest()
            if not record.attestation_text_hash or record.attestation_text_hash != expected_hash:
                record.verification_status = "user_supplied"
                record.attestation_user_id = None
                record.attestation_timestamp = None
                record.attestation_text_hash = None

        # Re-ingestion check: if record already exists with different text, clear attestation
        if record.knowledge_id in self._records:
            existing = self._records[record.knowledge_id]
            if existing.text != record.text and record.verification_status == "self_attested":
                record.verification_status = "user_supplied"
                record.attestation_user_id = None
                record.attestation_timestamp = None
                record.attestation_text_hash = None

        self._records[record.knowledge_id] = record
        self._notify_invalidation(record.owner_user_id)

        # Sync to Pinecone index when configured
        if self.pinecone_rag and self.sync_to_pinecone:
            try:
                self.pinecone_rag.upsert_record(record, tenant_id=record.owner_user_id)
            except Exception as e:
                LOGGER.debug("Could not sync record %s to Pinecone: %s", record.knowledge_id, e)

    def delete_record(self, knowledge_id: str, soft: bool = True) -> bool:
        """Deletes a knowledge chunk. If soft=True, marks tombstone per Step 10."""
        if knowledge_id not in self._records:
            return False
        rec = self._records[knowledge_id]
        user_id = rec.owner_user_id
        if soft:
            rec.deleted = True
        else:
            del self._records[knowledge_id]
        self._notify_invalidation(user_id)
        return True

    def supersede_record(self, old_knowledge_id: str, new_knowledge_id: str) -> bool:
        """Marks old chunk as superseded by newer version (Step 10)."""
        if old_knowledge_id in self._records:
            rec = self._records[old_knowledge_id]
            rec.superseded_by = new_knowledge_id
            self._notify_invalidation(rec.owner_user_id)
            return True
        return False

    def replace_record(self, old_knowledge_id: str, new_record: KnowledgeRecord) -> KnowledgeRecord:
        """Replaces an outdated knowledge chunk with a newer version (Step 10).
        
        Marks the old chunk as superseded by new_record.knowledge_id and stores new_record.
        Immediately invalidates the user's retrieval cache.
        """
        if old_knowledge_id not in self._records:
            raise KeyError(f"Knowledge record '{old_knowledge_id}' not found.")
        old_rec = self._records[old_knowledge_id]
        if old_rec.owner_user_id != new_record.owner_user_id:
            raise PermissionError(
                f"Cannot replace record owned by '{old_rec.owner_user_id}' with record for '{new_record.owner_user_id}'."
            )
        self.supersede_record(old_knowledge_id, new_record.knowledge_id)
        self.add_record(new_record)
        return new_record

    def attest_record(self, knowledge_id: str, user_id: str) -> KnowledgeRecord:
        """Explicit attestation action by the rep.

        Verifies that user_id owns the record.
        Enforces one-way gate: attesting NEVER unlocks prohibited high-risk claims.
        Records user ID, UTC timestamp, and SHA-256 hash of the exact text version.
        Sets verification_status to 'self_attested'.
        """
        if knowledge_id not in self._records:
            raise KeyError(f"Knowledge record '{knowledge_id}' not found.")
        rec = self._records[knowledge_id]
        if rec.owner_scope != "user" or rec.owner_user_id != user_id:
            raise PermissionError(f"User '{user_id}' cannot attest knowledge owned by '{rec.owner_user_id}'.")

        # One-way gate: attesting NEVER unlocks prohibited high-risk claims
        is_concrete, reason = classify_attested_claim(rec.text)
        if not is_concrete:
            raise ValueError(f"Cannot self-attest high-risk or unverified claim: {reason}")

        rec.verification_status = "self_attested"
        rec.attestation_user_id = user_id
        rec.attestation_timestamp = datetime.now(timezone.utc).isoformat()
        rec.attestation_text_hash = hashlib.sha256(rec.text.strip().encode("utf-8")).hexdigest()
        self._notify_invalidation(user_id)
        return rec

    def update_record_text(self, knowledge_id: str, new_text: str) -> KnowledgeRecord:
        """Updates record text. Any edit to the text clears previous attestation."""
        if knowledge_id not in self._records:
            raise KeyError(f"Knowledge record '{knowledge_id}' not found.")
        rec = self._records[knowledge_id]
        if rec.text != new_text:
            rec.text = new_text
            if rec.verification_status == "self_attested":
                rec.verification_status = "user_supplied"
            rec.attestation_user_id = None
            rec.attestation_timestamp = None
            rec.attestation_text_hash = None
            self._notify_invalidation(rec.owner_user_id)
        return rec

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
        prospect_type: Optional[str] = None,
        topic_tags: Optional[List[str]] = None,
        allow_confidential: bool = False,
        time_sensitive: bool = False,
        top_k: int = 10,
        min_score: float = 0.0,
        weights: Optional[RankingWeights] = None,
    ) -> List[Tuple[KnowledgeRecord, float]]:
        """Executes candidate retrieval with strict hard pre-filters applied inside query (Step 4 & 5)."""
        now_utc = datetime.now(timezone.utc)
        query_tokens = _tokenize(query)
        w = weights or RankingWeights()

        scored_candidates: List[Tuple[KnowledgeRecord, float]] = []

        # Try Pinecone vector search first if configured and enabled
        if self.pinecone_rag and os.getenv("ENABLE_PINECONE_RETRIEVAL", "0") == "1":
            try:
                rag_matches = self.pinecone_rag.search(
                    query=query,
                    tenant_id=user_id,
                    top_k=top_k * 2,
                    min_score=min_score,
                    owner_id=user_id,
                )
                if rag_matches:
                    for rm in rag_matches:
                        auth_rec = self.get_record(rm.id)
                        if not auth_rec:
                            auth_rec = KnowledgeRecord(
                                knowledge_id=rm.id,
                                owner_scope="user",
                                owner_user_id=user_id,
                                knowledge_class=rm.metadata.get("knowledge_class", "general"),
                                source_title=rm.metadata.get("source_title", rm.source),
                                source_version=rm.metadata.get("source_version", "1.0"),
                                topic_tags=rm.metadata.get("topic_tags", []),
                                text=rm.text,
                                verification_status=rm.metadata.get("verification_status", "company_approved"),
                                sensitivity=rm.metadata.get("sensitivity", "normal"),
                            )

                        if auth_rec.owner_scope != "user" or auth_rec.owner_user_id != user_id:
                            continue
                        if auth_rec.deleted or auth_rec.superseded_by is not None:
                            continue
                        if auth_rec.sensitivity == "restricted":
                            continue
                        if auth_rec.sensitivity == "confidential" and not allow_confidential:
                            continue
                        if auth_rec.lead_type_tags and lead_type and lead_type not in auth_rec.lead_type_tags:
                            continue

                        raw_semantic = min(1.0, max(0.0, getattr(rm, "score", 0.75)))
                        if raw_semantic < min_score:
                            continue

                        score = raw_semantic * w.semantic_weight
                        if auth_rec.lead_type_tags and lead_type and lead_type in auth_rec.lead_type_tags:
                            score += w.lead_type_boost
                        if auth_rec.verification_status in ("company_approved", "system_validated"):
                            score += w.verification_boost
                        elif auth_rec.verification_status == "self_attested":
                            score += round(0.4 * w.verification_boost, 3)

                        scored_candidates.append((auth_rec, score, raw_semantic))

                    if scored_candidates:
                        scored_candidates.sort(key=lambda x: x[1], reverse=True)
                        return scored_candidates[:top_k]
            except Exception as exc:
                LOGGER.warning("Pinecone candidate search encountered error, falling back to local store: %s", exc)

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

            # --- Similarity & Match Scoring with Doc 02 §7 Ranking Factors ---
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
                base_semantic = 0.35
            else:
                precision = len(overlap) / math.sqrt(max(1, len(doc_tokens)))
                coverage = len(overlap) / max(1, len(query_tokens))
                base_semantic = (precision + coverage) / 2.0
            raw_semantic = min(round(base_semantic, 3), 1.0)

            # CRITICAL (Point 1): Relevance floor applies to RAW SEMANTIC SIMILARITY!
            # Otherwise boosts can pull an irrelevant chunk over the floor.
            if raw_semantic < min_score:
                continue

            # 1. Semantic relevance weight (Doc 02 §7)
            score = raw_semantic * w.semantic_weight

            # 2. Lead/prospect type relevance boost (Doc 02 §7)
            if rec.lead_type_tags and lead_type and lead_type in rec.lead_type_tags:
                score += w.lead_type_boost
            if rec.prospect_type_tags and prospect_type and prospect_type in rec.prospect_type_tags:
                score += w.prospect_type_boost

            # 3. Recency / effective date boost: only applied when time_sensitive=True (Doc 02 §7)
            if time_sensitive:
                try:
                    uploaded_dt = datetime.fromisoformat(rec.uploaded_at.replace("Z", "+00:00"))
                    if uploaded_dt.tzinfo is None:
                        uploaded_dt = uploaded_dt.replace(tzinfo=timezone.utc)
                    age_days = (now_utc - uploaded_dt).total_seconds() / 86400.0
                    if age_days <= 90:
                        score += w.recency_boost
                except Exception:
                    pass

            # 4. Verification/approval status boost (Doc 02 §7 & Client Decision)
            # company approved / system validated > self_attested > user supplied
            if rec.verification_status in ("company_approved", "system_validated"):
                score += w.verification_boost
            elif rec.verification_status == "self_attested":
                expected_hash = hashlib.sha256(rec.text.strip().encode("utf-8")).hexdigest()
                if rec.attestation_text_hash == expected_hash:
                    score += round(0.4 * w.verification_boost, 3)
            elif rec.verification_status == "user_supplied":
                # plain user_supplied gets 0 boost (never equals approved or attested)
                pass

            # 5. Specificity to the current prospect/question (topic tags overlap & narrower chunks)
            if topic_tags and rec.topic_tags:
                tag_overlap = set(t.casefold() for t in topic_tags) & set(t.casefold() for t in rec.topic_tags)
                if tag_overlap:
                    score += w.specificity_boost

            # Target knowledge class alignment boost
            if target_classes and rec.knowledge_class in target_classes:
                score += 0.10

            score = min(round(score, 3), 1.0)
            scored_candidates.append((rec, score, raw_semantic))

        # Sort descending by composite score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]


# ==============================================================================
# Latency Metrics Aggregation (Stage Percentiles & Deadline Hit Alerting)
# ==============================================================================


class LatencyMetricsAggregator:
    """Aggregates per-stage latencies, computes p50/p95/p99 quantiles, and alerts on frequent deadline hits."""

    def __init__(
        self,
        window_size: int = 100,
        deadline_hit_alert_threshold: float = 0.15,
        min_samples_for_alert: int = 5,
    ) -> None:
        self.window_size = window_size
        self.deadline_hit_alert_threshold = deadline_hit_alert_threshold
        self.min_samples_for_alert = min_samples_for_alert
        self.history: List[Dict[str, Any]] = []
        self._alert_active: bool = False
        self._alert_message: Optional[str] = None

    def record(self, result: RetrievalResult) -> None:
        """Records a retrieval execution result."""
        lat = result.stage_latencies or StageLatencies(total_ms=result.latency_ms)
        entry = {
            "timestamp": time.time(),
            "status": result.status,
            "timed_out": (result.status == "timed_out"),
            "gate_ms": lat.gate_ms,
            "embedding_ms": lat.embedding_ms,
            "search_ms": lat.search_ms,
            "ranking_conflict_ms": lat.ranking_conflict_ms,
            "packaging_ms": lat.packaging_ms,
            "total_ms": lat.total_ms,
        }
        self.history.append(entry)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Check deadline hit frequency
        recent = self.history[-self.window_size:]
        if len(recent) >= self.min_samples_for_alert:
            timeouts = sum(1 for e in recent if e["timed_out"])
            rate = timeouts / len(recent)
            if rate >= self.deadline_hit_alert_threshold:
                self._alert_active = True
                self._alert_message = (
                    f"CRITICAL LATENCY ALERT: Frequent retrieval deadline hits detected! "
                    f"{timeouts}/{len(recent)} requests ({rate*100:.1f}%) timed out."
                )
                LOGGER.error(self._alert_message)
            else:
                self._alert_active = False
                self._alert_message = None

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(round(p * (len(sorted_vals) - 1)))
        return round(sorted_vals[idx], 2)

    def get_stage_percentiles(self) -> Dict[str, Dict[str, float]]:
        """Returns p50, p95, p99 for each stage."""
        stages = ["gate_ms", "embedding_ms", "search_ms", "ranking_conflict_ms", "packaging_ms", "total_ms"]
        stats: Dict[str, Dict[str, float]] = {}
        for stage in stages:
            vals = [e[stage] for e in self.history if stage in e]
            stats[stage] = {
                "p50": self._percentile(vals, 0.50),
                "p95": self._percentile(vals, 0.95),
                "p99": self._percentile(vals, 0.99),
            }
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Provides full metrics summary including sample count, percentiles, and alert status."""
        timeouts = sum(1 for e in self.history if e["timed_out"])
        return {
            "total_recorded": len(self.history),
            "timed_out_count": timeouts,
            "timeout_rate": round(timeouts / max(1, len(self.history)), 4),
            "alert_active": self._alert_active,
            "alert_message": self._alert_message,
            "stage_percentiles": self.get_stage_percentiles(),
        }


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
        cache: Optional[RetrievalCache] = None,
        config: Optional[RetrievalConfig] = None,
        latency_aggregator: Optional[LatencyMetricsAggregator] = None,
        default_tenant_id: str = "default",
        default_user_id: str = "default_user",
    ):
        self.backend = backend or LocalKnowledgeBase()
        self.cache = cache or RetrievalCache()
        self.store = store or KnowledgeStore(cache=self.cache)
        if not self.store.cache:
            self.store.cache = self.cache
        self.gate = gate or RetrievalGate()
        self.config = config or RetrievalConfig()
        self.latency_aggregator = latency_aggregator or LatencyMetricsAggregator()
        self.default_tenant_id = default_tenant_id
        self.default_user_id = default_user_id

    def get_latency_metrics(self) -> Dict[str, Any]:
        """Returns latency metrics aggregation including p50/p95/p99 and deadline alerts."""
        return self.latency_aggregator.get_summary()

    def _parse_version(self, version_str: str) -> Tuple[int, ...]:
        """Parses version strings like '1.0', '2.1', 'v2.0' into numeric tuples for comparison."""
        cleaned = re.sub(r"^[^\d]*", "", str(version_str))
        try:
            return tuple(int(p) for p in cleaned.split(".") if p.isdigit())
        except Exception:
            return (1, 0)

    def _extract_key_metric(self, text: str) -> Optional[str]:
        """Extracts key quantitative assertion like '5%', '$500', '3 days' from chunk text."""
        # Check percentage
        m = re.search(r"(\d+(?:\.\d+)?\s*%)", text)
        if m:
            return m.group(1).replace(" ", "")
        # Check dollar
        m = re.search(r"(\$\s*\d+(?:,\d+)*(?:\.\d+)?)", text)
        if m:
            return m.group(1).replace(" ", "")
        return None

    def _detect_contradiction(self, rec1: KnowledgeRecord, rec2: KnowledgeRecord) -> bool:
        """Determines if two candidate records contradict each other on the same subject.
        
        Doc 02 §8: When two chunks make divergent or contradictory assertions on the same
        topic, the system must default to withholding when unsure.
        """
        # 1. Must share knowledge class OR be a Playbook vs Company Policy comparison
        is_policy_comparison = (
            (rec1.knowledge_class == KnowledgeClass.COMPANY_POLICY.value or "policy" in rec1.source_title.casefold()) and
            (rec2.knowledge_class != KnowledgeClass.COMPANY_POLICY.value or "playbook" in rec2.source_title.casefold())
        ) or (
            (rec2.knowledge_class == KnowledgeClass.COMPANY_POLICY.value or "policy" in rec2.source_title.casefold()) and
            (rec1.knowledge_class != KnowledgeClass.COMPANY_POLICY.value or "playbook" in rec1.source_title.casefold())
        )

        if rec1.knowledge_class != rec2.knowledge_class and not is_policy_comparison:
            return False

        # 2. Must share at least one topic tag or similar title or high token overlap
        overlap_tags = set(t.casefold() for t in rec1.topic_tags) & set(t.casefold() for t in rec2.topic_tags)
        same_title = (rec1.source_title.casefold() == rec2.source_title.casefold())
        tokens1 = _tokenize(rec1.text)
        tokens2 = _tokenize(rec2.text)
        common_tokens = tokens1 & tokens2
        high_topic_overlap = len(common_tokens) >= 3 and (len(common_tokens) / max(1, min(len(tokens1), len(tokens2)))) >= 0.35

        if not overlap_tags and not same_title and not high_topic_overlap:
            return False

        # 3. Check for differing key quantitative metrics (e.g. 5% vs 6%, $500 vs $600)
        m1 = self._extract_key_metric(rec1.text)
        m2 = self._extract_key_metric(rec2.text)
        if m1 and m2 and m1 != m2:
            return True

        # 4. Check for opposing factual assertions and polarities
        text1 = rec1.text.casefold()
        text2 = rec2.text.casefold()

        # Free / zero-cost vs fee / charge / penalty
        free_markers = ["free", "no fee", "zero fee", "no charge", "without fee", "at no cost"]
        fee_markers = ["fee required", "administrative fee", "charge required", "penalty", "cancellation fee", "termination fee"]
        has_free_1 = any(m in text1 for m in free_markers)
        has_fee_1 = any(m in text1 for m in fee_markers)
        has_free_2 = any(m in text2 for m in free_markers)
        has_fee_2 = any(m in text2 for m in fee_markers)

        if (has_free_1 and has_fee_2) or (has_free_2 and has_fee_1):
            return True

        opposing_pairs = [
            ("free", "charge"),
            ("free", "fee"),
            ("no fee", "fee"),
            ("no upfront", "upfront"),
            ("exclusive", "non-exclusive"),
            ("guarantee", "no guarantee"),
            ("allowed", "prohibited"),
            ("permitted", "forbidden"),
            ("required", "optional"),
            ("mandatory", "voluntary"),
            ("unlimited", "limited"),
            ("direct", "indirect"),
        ]
        for term1, term2 in opposing_pairs:
            pattern1 = rf"\b{re.escape(term1)}\b"
            pattern2 = rf"\b{re.escape(term2)}\b"
            if (re.search(pattern1, text1) and re.search(pattern2, text2)) or (
                re.search(pattern2, text1) and re.search(pattern1, text2)
            ):
                return True

        # 5. Direct polarity negation conflict on the same topic:
        # e.g. "pre-approval required" vs "no pre-approval required"
        negation_terms = {"not", "never", "cannot", "neither", "no", "prohibited", "disallowed", "without"}
        has_neg1 = bool(set(text1.split()) & negation_terms)
        has_neg2 = bool(set(text2.split()) & negation_terms)
        if has_neg1 != has_neg2 and (same_title or len(overlap_tags) >= 2 or high_topic_overlap):
            substantive_common = common_tokens - {"with", "from", "that", "this", "have", "your", "more", "number", "point"}
            if len(substantive_common) >= 2:
                return True

        return False

    def _resolve_candidate_conflicts(
        self,
        candidates: List[Tuple[Any, ...]],
        config: Optional[RetrievalConfig] = None,
        request: Optional[RetrievalRequest] = None,
    ) -> Tuple[List[RetrievedFact], List[WithheldFact], bool]:
        """Executes Doc 02 §8 Conflict Resolution table & Guardrails on ranked candidates."""
        withheld: List[WithheldFact] = []
        eligible: List[Tuple[KnowledgeRecord, float, float]] = []
        allow_user_perf = config.allow_user_supplied_performance if config else False

        # Pass 1: Individual chunk guardrails (injection, compliance, sensitivity, attestation, conversation truth)
        for item in candidates:
            rec = item[0]
            score = item[1]
            raw_semantic = item[2] if len(item) > 2 else score

            # Prompt Injection Defense (bonus regex layer over structural wrapper)
            has_inj, inj_pattern = detect_prompt_injection(rec.text)
            if has_inj:
                withheld.append(
                    WithheldFact(
                        knowledge_id=rec.knowledge_id,
                        reason="restricted",
                        details=f"Prompt injection pattern detected ('{inj_pattern}'). Chunk withheld for safety.",
                    )
                )
                continue

            # Sensitivity check
            if rec.sensitivity == "restricted":
                withheld.append(
                    WithheldFact(
                        knowledge_id=rec.knowledge_id,
                        reason="restricted",
                        details="Restricted content cannot be provided in prompt context. Compliance/safety boundary wins.",
                    )
                )
                continue

            # Unified Compliance Gate Check (Doc 01 Stage 2 & Doc 02 §8)
            is_compliant, comp_cat, comp_details = evaluate_content_compliance(rec.text)
            if not is_compliant:
                withheld.append(
                    WithheldFact(
                        knowledge_id=rec.knowledge_id,
                        reason="restricted",
                        details=f"Compliance/safety violation detected ({comp_cat}): {comp_details} Compliance rules override user uploads.",
                    )
                )
                continue

            # Attestation integrity & Fail-Closed Claim Classification across ALL knowledge classes
            if rec.verification_status == "self_attested":
                # Check SHA-256 text hash integrity
                expected_hash = hashlib.sha256(rec.text.strip().encode("utf-8")).hexdigest()
                if not rec.attestation_text_hash or rec.attestation_text_hash != expected_hash:
                    rec.verification_status = "user_supplied"
                    rec.attestation_user_id = None
                    rec.attestation_timestamp = None
                    rec.attestation_text_hash = None
                    withheld.append(
                        WithheldFact(
                            knowledge_id=rec.knowledge_id,
                            reason="unverified",
                            details="Attestation invalidated: text modified since attestation (hash mismatch).",
                        )
                    )
                    continue

                # Fail-closed classification: checkable concrete fact (tenure, territory, named transactions)
                # Guarantees, rankings/awards, stats, certifications, or ambiguous claims fail closed!
                is_concrete, claim_reason = classify_attested_claim(rec.text)
                if not is_concrete:
                    withheld.append(
                        WithheldFact(
                            knowledge_id=rec.knowledge_id,
                            reason="unverified",
                            details=f"High-risk or ambiguous claim in '{rec.knowledge_class}' cannot be self-attested: {claim_reason} Fails closed.",
                        )
                    )
                    continue

            elif rec.verification_status == "user_supplied":
                # User-supplied performance claims withheld unless allow_user_supplied_performance is True
                is_perf_class = rec.knowledge_class in (
                    KnowledgeClass.PERFORMANCE_PAST_SALES.value,
                    "performance_past_sales",
                )
                if is_perf_class and not allow_user_perf:
                    withheld.append(
                        WithheldFact(
                            knowledge_id=rec.knowledge_id,
                            reason="unverified",
                            details="Performance and past sales claims require explicit self-attestation or company approval.",
                        )
                    )
                    continue

                # Salesperson profile: unapproved high-risk superlatives/rankings require company approval
                if rec.knowledge_class == KnowledgeClass.SALESPERSON_PROFILE.value:
                    is_concrete, claim_reason = classify_attested_claim(rec.text)
                    if not is_concrete and any(
                        k in rec.text.casefold() for k in ["#1", "top ", "producer", "award", "guarantee"]
                    ):
                        withheld.append(
                            WithheldFact(
                                knowledge_id=rec.knowledge_id,
                                reason="unverified",
                                details=f"Unverified high-risk superlative/claim in '{rec.knowledge_class}' requires company approval: {claim_reason}",
                            )
                        )
                        continue

            # Rule 3: Live Conversation Truth over Stored Scripts (Doc 02 §8 - Point 3B)
            if request:
                live_evidence = (request.conversation_evidence or "").casefold()
                live_facts_dict = {}
                if request.live_state_facts:
                    for f_item in request.live_state_facts:
                        if isinstance(f_item, dict):
                            k = str(f_item.get("fact_key", "")).casefold()
                            v = str(f_item.get("fact_value", "")).casefold()
                            if k and v:
                                live_facts_dict[k] = v

                chunk_text_lower = rec.text.casefold()

                # Timing / Scheduling contradiction:
                # Asking for consultation hours with a proposed meeting time is a request, not a factual contradiction.
                # Only a firm negative constraint contradicts stored schedule.
                is_asking_hours = any(w in live_evidence for w in ("consultation hours", "hours", "what are your", "when are consultations"))
                timing_contradiction = False
                if not is_asking_hours:
                    has_afternoon_only = ("afternoon" in live_evidence and "only" in live_evidence) or "afternoons only" in live_evidence or "never morning" in live_evidence or live_facts_dict.get("scheduling_constraint") in ("afternoon_only", "afternoons_only", "never_mornings")
                    has_morning_only = ("morning" in live_evidence and "only" in live_evidence) or "mornings only" in live_evidence or live_facts_dict.get("scheduling_constraint") in ("morning_only", "mornings_only", "never_afternoons")
                    if has_afternoon_only and ("morning" in chunk_text_lower or "10 am" in chunk_text_lower or "9 am" in chunk_text_lower):
                        timing_contradiction = True
                    elif has_morning_only and ("afternoon" in chunk_text_lower or "evening" in chunk_text_lower):
                        timing_contradiction = True

                # Property tenure / Buyer vs Seller contradiction (Doc 02 §8):
                # Stored script claiming first-time buyer conflicts with live owner of 20 years selling their home
                tenure_contradiction = False
                if ("20 years" in live_evidence or "owned for" in live_evidence or "selling" in live_evidence or live_facts_dict.get("ownership_duration") == "20_years") and (
                    "first-time buyer" in chunk_text_lower or "first purchase" in chunk_text_lower or "first home purchase" in chunk_text_lower
                ):
                    tenure_contradiction = True

                # Property type contradiction (Doc 02 §8):
                # Stored script assuming 3-bedroom single-family house conflicts with prospect declaring condo/townhome
                property_type_contradiction = False
                is_condo = any(w in live_evidence for w in ("condo", "condominium", "townhouse", "townhome", "apartment", "unit")) or live_facts_dict.get("property_type") in ("condo", "townhouse", "apartment")
                if is_condo and any(w in chunk_text_lower for w in ("single-family home", "single family", "3-bedroom home", "3 bedroom house", "detached home", "large yard", "sprawling yard")):
                    property_type_contradiction = True

                # Relationship history contradiction (Doc 02 §8):
                # Stored first-call/introduction script conflicts with prospect who already met or worked with the rep
                relationship_history_contradiction = False
                has_prior_relationship = any(w in live_evidence for w in ("already met", "met last week", "worked together before", "sold our last", "spoke yesterday", "already worked with")) or live_facts_dict.get("relationship_status") in ("prior_client", "already_met")
                if has_prior_relationship and any(w in chunk_text_lower for w in ("nice to meet you", "first time speaking", "first-time caller", "first call", "introducing our team", "new prospective client")):
                    relationship_history_contradiction = True

                # Stakeholder / Decision Maker contradiction
                stakeholder_contradiction = False
                if ("wife" in live_evidence or "spouse" in live_evidence or "partner" in live_evidence or "co_owner" in live_facts_dict) and (
                    "single decision maker" in chunk_text_lower or "solo homeowner" in chunk_text_lower or "unilateral decision" in chunk_text_lower
                ):
                    stakeholder_contradiction = True

                # Contact preference contradiction
                channel_contradiction = False
                if ("email only" in live_evidence or "do not call" in live_evidence or "no daily text" in live_evidence or "don't text" in live_evidence or live_facts_dict.get("contact_preference") in ("email_only", "no_sms")) and (
                    "daily phone calls" in chunk_text_lower or "call prospect daily" in chunk_text_lower or "daily text" in chunk_text_lower
                ):
                    channel_contradiction = True

                if (
                    timing_contradiction
                    or tenure_contradiction
                    or property_type_contradiction
                    or relationship_history_contradiction
                    or stakeholder_contradiction
                    or channel_contradiction
                ):
                    withheld.append(
                        WithheldFact(
                            knowledge_id=rec.knowledge_id,
                            reason="disputed",
                            details="Conflicts with live conversation truth. Live conversation state wins.",
                        )
                    )
                    continue

            eligible.append((rec, score, raw_semantic))

        # Pass 2: Supersession & version coexistence (Doc 02 §8)
        # Group by (knowledge_class, source_title)
        stale_ids: Set[str] = set()
        for i, (rec_i, _, _) in enumerate(eligible):
            for j, (rec_j, _, _) in enumerate(eligible):
                if i >= j:
                    continue
                if rec_i.source_title.casefold() == rec_j.source_title.casefold() and rec_i.knowledge_class == rec_j.knowledge_class:
                    v_i = self._parse_version(rec_i.source_version)
                    v_j = self._parse_version(rec_j.source_version)
                    if v_i > v_j:
                        stale_ids.add(rec_j.knowledge_id)
                        withheld.append(
                            WithheldFact(
                                knowledge_id=rec_j.knowledge_id,
                                reason="stale",
                                details=f"Superseded by newer source version {rec_i.source_version} of '{rec_i.source_title}'.",
                            )
                        )
                    elif v_j > v_i:
                        stale_ids.add(rec_i.knowledge_id)
                        withheld.append(
                            WithheldFact(
                                knowledge_id=rec_i.knowledge_id,
                                reason="stale",
                                details=f"Superseded by newer source version {rec_j.source_version} of '{rec_j.source_title}'.",
                            )
                        )

        active_eligible = [c for c in eligible if c[0].knowledge_id not in stale_ids]

        # Pass 3: Contradiction resolution between documents (Doc 02 §8)
        # Includes Playbook method vs Company policy (Doc 02 §8) and Authority overrides
        disputed_ids: Set[str] = set()
        playbook_substitutions: Dict[str, str] = {}
        had_unresolved_conflict = False

        for i, (rec_i, _, _) in enumerate(active_eligible):
            for j, (rec_j, _, _) in enumerate(active_eligible):
                if i >= j:
                    continue
                if rec_i.knowledge_id in disputed_ids or rec_j.knowledge_id in disputed_ids:
                    continue

                if self._detect_contradiction(rec_i, rec_j):
                    # Rule 4: Playbook method vs Company Policy (Doc 02 §8)
                    # Doc 02 §8: Keep the Playbook strategy but substitute the company-approved fact or policy!
                    is_policy_i = rec_i.knowledge_class == KnowledgeClass.COMPANY_POLICY.value or "policy" in rec_i.source_title.lower()
                    is_policy_j = rec_j.knowledge_class == KnowledgeClass.COMPANY_POLICY.value or "policy" in rec_j.source_title.lower()

                    if is_policy_i and not is_policy_j:
                        disputed_ids.add(rec_j.knowledge_id)
                        playbook_substitutions[rec_i.knowledge_id] = rec_j.knowledge_id
                        withheld.append(
                            WithheldFact(
                                knowledge_id=rec_j.knowledge_id,
                                reason="disputed",
                                details=f"Playbook script conflicts with company-approved factual policy '{rec_i.source_title}'. Company policy wins; retaining playbook strategy with company policy substituted.",
                            )
                        )
                        continue
                    elif is_policy_j and not is_policy_i:
                        disputed_ids.add(rec_i.knowledge_id)
                        playbook_substitutions[rec_j.knowledge_id] = rec_i.knowledge_id
                        withheld.append(
                            WithheldFact(
                                knowledge_id=rec_i.knowledge_id,
                                reason="disputed",
                                details=f"Playbook script conflicts with company-approved factual policy '{rec_j.source_title}'. Company policy wins; retaining playbook strategy with company policy substituted.",
                            )
                        )
                        continue

                    # Authority resolution
                    auth_i = 3 if rec_i.verification_status in ("company_approved", "system_validated") else 1
                    auth_j = 3 if rec_j.verification_status in ("company_approved", "system_validated") else 1

                    if auth_i > auth_j:
                        disputed_ids.add(rec_j.knowledge_id)
                        withheld.append(
                            WithheldFact(
                                knowledge_id=rec_j.knowledge_id,
                                reason="disputed",
                                details=f"Contradicts higher authority approved source '{rec_i.source_title}'.",
                            )
                        )
                    elif auth_j > auth_i:
                        disputed_ids.add(rec_i.knowledge_id)
                        withheld.append(
                            WithheldFact(
                                knowledge_id=rec_i.knowledge_id,
                                reason="disputed",
                                details=f"Contradicts higher authority approved source '{rec_j.source_title}'.",
                            )
                        )
                    else:
                        v_i = self._parse_version(rec_i.source_version)
                        v_j = self._parse_version(rec_j.source_version)
                        if v_i > v_j:
                            disputed_ids.add(rec_j.knowledge_id)
                            withheld.append(
                                WithheldFact(
                                    knowledge_id=rec_j.knowledge_id,
                                    reason="stale",
                                    details=f"Superseded by newer source version {rec_i.source_version} ('{rec_i.source_title}').",
                                )
                            )
                        elif v_j > v_i:
                            disputed_ids.add(rec_i.knowledge_id)
                            withheld.append(
                                WithheldFact(
                                    knowledge_id=rec_i.knowledge_id,
                                    reason="stale",
                                    details=f"Superseded by newer source version {rec_j.source_version} ('{rec_j.source_title}').",
                                )
                            )
                        else:
                            # Genuine authority tie: DO NOT ASSERT DISPUTED FACT!
                            disputed_ids.add(rec_i.knowledge_id)
                            disputed_ids.add(rec_j.knowledge_id)
                            had_unresolved_conflict = True
                            withheld.append(
                                WithheldFact(
                                    knowledge_id=rec_i.knowledge_id,
                                    reason="disputed",
                                    details=f"Contradictory facts detected between '{rec_i.source_title}' and '{rec_j.source_title}'. Withheld on unresolved authority tie.",
                                )
                            )
                            withheld.append(
                                WithheldFact(
                                    knowledge_id=rec_j.knowledge_id,
                                    reason="disputed",
                                    details=f"Contradictory facts detected between '{rec_i.source_title}' and '{rec_j.source_title}'. Withheld on unresolved authority tie.",
                                )
                            )

        # Convert remaining undisputed candidates to RetrievedFact
        accepted_facts: List[RetrievedFact] = []
        for rec, score, _ in active_eligible:
            if rec.knowledge_id not in disputed_ids:
                is_attested = (rec.verification_status == "self_attested")
                attestation_meta = None
                if is_attested:
                    attestation_meta = {
                        "attestation_user_id": rec.attestation_user_id,
                        "attestation_timestamp": rec.attestation_timestamp,
                    }
                sub_playbook_id = playbook_substitutions.get(rec.knowledge_id)
                fact = RetrievedFact(
                    knowledge_id=rec.knowledge_id,
                    text=rec.text,
                    knowledge_class=rec.knowledge_class,
                    verification_status=rec.verification_status,
                    source_title=rec.source_title,
                    source_version=rec.source_version,
                    confidence=score,
                    is_self_attested=is_attested,
                    attestation_metadata=attestation_meta,
                    playbook_strategy_retained=bool(sub_playbook_id),
                    substituted_for_playbook_id=sub_playbook_id,
                )
                accepted_facts.append(fact)

        return accepted_facts, withheld, had_unresolved_conflict

    # --------------------------------------------------------------------------
    # Phase 1, 2 & 3 Execution: Core Request -> Gate -> Search -> Quality & Conflict Check
    # --------------------------------------------------------------------------

    def retrieve_for_request(
        self,
        request: RetrievalRequest,
        allow_confidential: bool = False,
        relevance_floor: Optional[float] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalResult:
        """Executes retrieval workflow per Doc 02 and Client Implementation Contract.

        Phase 5 Hardening:
        - Per-stage latency instrumentation (gate_ms, embedding_ms, search_ms, ranking_conflict_ms, packaging_ms, total_ms)
        - Deadline and timeout fallback (deadline_ms / default_timeout_ms)
        - Immediate cache hit check and population
        """
        start_time = time.perf_counter()
        cfg = config or getattr(self, "config", None) or RetrievalConfig()
        floor = relevance_floor if relevance_floor is not None else cfg.relevance_floor
        effective_deadline_ms = request.deadline_ms if request.deadline_ms is not None else cfg.default_timeout_ms
        user_kv = self.store.get_user_knowledge_version(request.user_id) if hasattr(self, "store") and hasattr(self.store, "get_user_knowledge_version") else 1

        gate_ms = 0.0
        embedding_ms = 0.0
        search_ms = 0.0
        ranking_conflict_ms = 0.0
        packaging_ms = 0.0

        def _timeout_result() -> RetrievalResult:
            tot = (time.perf_counter() - start_time) * 1000.0
            lat = StageLatencies(
                gate_ms=round(gate_ms, 2),
                embedding_ms=round(embedding_ms, 2),
                search_ms=round(search_ms, 2),
                ranking_conflict_ms=round(ranking_conflict_ms, 2),
                packaging_ms=round(packaging_ms, 2),
                total_ms=round(tot, 2),
            )
            r = RetrievalResult(
                request_id=request.request_id,
                state_version=request.state_version,
                status="timed_out",
                facts=[],
                withheld=[],
                fallback_hint="use_generic_truthful",
                latency_ms=round(tot, 2),
                stage_latencies=lat,
            )
            if hasattr(self, "latency_aggregator") and self.latency_aggregator:
                self.latency_aggregator.record(r)
            return r

        def _is_timed_out() -> bool:
            if effective_deadline_ms is None or effective_deadline_ms <= 0:
                return False
            return ((time.perf_counter() - start_time) * 1000.0) >= effective_deadline_ms

        def _finish(res_obj: RetrievalResult) -> RetrievalResult:
            if cfg.cache_enabled and getattr(self, "cache", None):
                self.cache.set(
                    request,
                    res_obj,
                    allow_confidential=allow_confidential,
                    relevance_floor=relevance_floor,
                    config=cfg,
                    user_knowledge_version=user_kv,
                )
            if hasattr(self, "latency_aggregator") and self.latency_aggregator:
                self.latency_aggregator.record(res_obj)
            return res_obj

        # Check Cache with multi-instance knowledge versioning (Doc 02 §11)
        if cfg.cache_enabled and getattr(self, "cache", None):
            cached = self.cache.get(
                request,
                allow_confidential=allow_confidential,
                relevance_floor=relevance_floor,
                config=cfg,
                user_knowledge_version=user_kv,
            )
            if cached is not None:
                tot = (time.perf_counter() - start_time) * 1000.0
                lat = StageLatencies(
                    gate_ms=0.0,
                    embedding_ms=0.0,
                    search_ms=0.0,
                    ranking_conflict_ms=0.0,
                    packaging_ms=round(tot, 2),
                    total_ms=round(tot, 2),
                )
                hit_res = cached.model_copy(
                    update={
                        "latency_ms": round(tot, 2),
                        "stage_latencies": lat,
                    }
                )
                if hasattr(self, "latency_aggregator") and self.latency_aggregator:
                    self.latency_aggregator.record(hit_res)
                return hit_res

        # Step 1: Retrieval Gate check (Doc 02 §6)
        t_gate_0 = time.perf_counter()
        gate_passed, status, fallback_hint, reason = self.gate.evaluate(
            request=request,
            knowledge_store=self.store,
        )
        gate_ms = (time.perf_counter() - t_gate_0) * 1000.0

        if not gate_passed:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            lat = StageLatencies(
                gate_ms=round(gate_ms, 2),
                embedding_ms=0.0,
                search_ms=0.0,
                ranking_conflict_ms=0.0,
                packaging_ms=0.0,
                total_ms=round(elapsed_ms, 2),
            )
            res = RetrievalResult(
                request_id=request.request_id,
                state_version=request.state_version,
                status=status,  # "none_needed" or "none_found"
                facts=[],
                withheld=[],
                fallback_hint=fallback_hint,
                latency_ms=round(elapsed_ms, 2),
                stage_latencies=lat,
            )
            return _finish(res)

        # Checkpoint 1: Deadline check after Gate
        if _is_timed_out():
            return _timeout_result()

        # Step 2: Query Builder (Doc 02 §6 & Step 3)
        t_embed_0 = time.perf_counter()
        query_text, target_classes = QueryBuilder.build_query(request)
        topic_tags = request.retrieval_need.topic_tags if request.retrieval_need else []
        time_sensitive = request.retrieval_need.time_sensitive if request.retrieval_need else False
        embedding_ms = (time.perf_counter() - t_embed_0) * 1000.0

        # Checkpoint 2: Deadline check after Query Builder / Embedding
        if _is_timed_out():
            return _timeout_result()

        # Step 3: Candidate search with hard pre-filters applied in-query (Step 4 & 5)
        t_search_0 = time.perf_counter()
        candidates = self.store.search_candidates(
            query=query_text,
            user_id=request.user_id,
            target_classes=target_classes,
            lead_type=request.lead_type,
            prospect_type=request.prospect_type,
            topic_tags=topic_tags,
            allow_confidential=allow_confidential,
            time_sensitive=time_sensitive,
            top_k=10,
            min_score=floor,
            weights=cfg.weights,
        )
        search_ms = (time.perf_counter() - t_search_0) * 1000.0

        # Step 3b: Authoritative store re-verification (Doc 02 §11, Eventual Consistency Defense):
        # Pinecone/vector store deletions and supersessions may experience eventual consistency latency.
        # Re-verify each candidate against the authoritative store: exclude deleted chunks, superseded
        # chunks, or chunks where self-attestation text hash was invalidated.
        verified_candidates: List[Tuple[KnowledgeRecord, float, float]] = []
        withheld_pre: List[WithheldFact] = []
        for cand in candidates:
            cand_rec = cand[0]
            cand_score = cand[1]
            cand_raw = cand[2] if len(cand) > 2 else cand_score

            if hasattr(self.store, "get_record"):
                auth_rec = self.store.get_record(cand_rec.knowledge_id)
                if auth_rec is None or auth_rec.deleted or auth_rec.superseded_by is not None:
                    LOGGER.debug("Authoritative store re-check: candidate '%s' is deleted or superseded; excluded.", cand_rec.knowledge_id)
                    withheld_pre.append(
                        WithheldFact(
                            knowledge_id=cand_rec.knowledge_id,
                            reason="stale",
                            details="Deleted or superseded in authoritative store.",
                        )
                    )
                    continue
                if auth_rec.verification_status == "self_attested":
                    expected_hash = hashlib.sha256(auth_rec.text.strip().encode("utf-8")).hexdigest()
                    if auth_rec.attestation_text_hash != expected_hash:
                        LOGGER.warning("Authoritative store re-check: candidate '%s' attestation hash mismatch; excluded.", cand_rec.knowledge_id)
                        auth_rec.verification_status = "user_supplied"
                        auth_rec.attestation_user_id = None
                        auth_rec.attestation_timestamp = None
                        auth_rec.attestation_text_hash = None
                        withheld_pre.append(
                            WithheldFact(
                                knowledge_id=cand_rec.knowledge_id,
                                reason="unverified",
                                details="Attestation invalidated: text hash mismatch in authoritative store.",
                            )
                        )
                        continue
                verified_candidates.append((auth_rec, cand_score, cand_raw))
            else:
                verified_candidates.append(cand)
        candidates = verified_candidates

        # Checkpoint 3: Deadline check after Search
        if _is_timed_out():
            return _timeout_result()

        trigger_val = str(getattr(request.retrieval_need.trigger, "value", request.retrieval_need.trigger)).lower() if request.retrieval_need else ""

        if not candidates:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            fallback = RetrievalGate.resolve_fallback_hint(request.strategic_action, trigger_val)
            lat = StageLatencies(
                gate_ms=round(gate_ms, 2),
                embedding_ms=round(embedding_ms, 2),
                search_ms=round(search_ms, 2),
                ranking_conflict_ms=0.0,
                packaging_ms=0.0,
                total_ms=round(elapsed_ms, 2),
            )
            res = RetrievalResult(
                request_id=request.request_id,
                state_version=request.state_version,
                status="none_found",
                facts=[],
                withheld=withheld_pre,
                fallback_hint=fallback,
                latency_ms=round(elapsed_ms, 2),
                stage_latencies=lat,
            )
            return _finish(res)

        # Step 4: Conflict Resolution and Guardrail checking (Doc 02 §8 & §9, Step 6)
        t_rc_0 = time.perf_counter()
        accepted_facts, withheld_facts, had_unresolved_conflict = self._resolve_candidate_conflicts(
            candidates,
            config=cfg,
            request=request,
        )
        withheld_facts = withheld_pre + withheld_facts
        ranking_conflict_ms = (time.perf_counter() - t_rc_0) * 1000.0

        # Checkpoint 4: Deadline check after Conflict Resolution
        if _is_timed_out():
            return _timeout_result()

        # Step 5: Packaging & Formatting
        t_pkg_0 = time.perf_counter()
        if not accepted_facts:
            if had_unresolved_conflict:
                status = "conflict_unresolved"
                fallback = "use_generic_truthful"
            else:
                status = "none_found"
                fallback = RetrievalGate.resolve_fallback_hint(request.strategic_action, trigger_val)
            packaging_ms = (time.perf_counter() - t_pkg_0) * 1000.0
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            lat = StageLatencies(
                gate_ms=round(gate_ms, 2),
                embedding_ms=round(embedding_ms, 2),
                search_ms=round(search_ms, 2),
                ranking_conflict_ms=round(ranking_conflict_ms, 2),
                packaging_ms=round(packaging_ms, 2),
                total_ms=round(elapsed_ms, 2),
            )
            res = RetrievalResult(
                request_id=request.request_id,
                state_version=request.state_version,
                status=status,
                facts=[],
                withheld=withheld_facts,
                fallback_hint=fallback,
                latency_ms=round(elapsed_ms, 2),
                stage_latencies=lat,
            )
            return _finish(res)

        # Check marginal threshold on RAW SEMANTIC SIMILARITY & Cap facts (Doc 02 §7 - Point 1)
        accepted_ids = {f.knowledge_id for f in accepted_facts}
        accepted_raw_scores = [item[2] for item in candidates if item[0].knowledge_id in accepted_ids]
        top_raw_semantic = max(accepted_raw_scores) if accepted_raw_scores else 0.0

        trigger_val = str(getattr(request.retrieval_need.trigger, "value", request.retrieval_need.trigger)).lower() if request.retrieval_need else ""
        action_str = str(getattr(request.strategic_action, "value", request.strategic_action)).lower()
        if top_raw_semantic < cfg.marginal_threshold:
            status = "low_confidence"
            if trigger_val in ("factual_question", "prior_claim_reference") or action_str == "question":
                fallback_hint = "ask_clarifying_question"
            else:
                fallback_hint = "use_generic_truthful"
            final_facts = accepted_facts[:cfg.max_facts]
        else:
            status = "retrieved"
            fallback_hint = None
            final_facts = accepted_facts[:cfg.max_facts]
        packaging_ms = (time.perf_counter() - t_pkg_0) * 1000.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        lat = StageLatencies(
            gate_ms=round(gate_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
            ranking_conflict_ms=round(ranking_conflict_ms, 2),
            packaging_ms=round(packaging_ms, 2),
            total_ms=round(elapsed_ms, 2),
        )

        res = RetrievalResult(
            request_id=request.request_id,
            state_version=request.state_version,
            status=status,
            facts=final_facts,
            withheld=withheld_facts,
            fallback_hint=fallback_hint,
            latency_ms=round(elapsed_ms, 2),
            stage_latencies=lat,
        )
        return _finish(res)

    @classmethod
    def validate_or_discard_stale_result(
        cls,
        result: RetrievalResult,
        current_state_version: int,
    ) -> RetrievalResult:
        """Compares state_version on result return per Doc 01 §13.

        If the state has diverged (result.state_version != current_state_version),
        discards late facts and marks status='stale_discarded' with a truthful fallback hint.
        """
        if result.state_version != current_state_version:
            return result.model_copy(
                update={
                    "status": "stale_discarded",
                    "facts": [],
                    "fallback_hint": "use_generic_truthful",
                }
            )
        return result

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
            or self.default_user_id
        )

        # Legacy path fails closed when user_id is missing per Constraint C6
        if not active_user:
            LOGGER.warning("retrieve_for_decision rejected: missing user_id (fails closed per Constraint C6).")
            return RetrievalExecutionReport(
                retrieval_needed=False,
                decision_id=decision.decision_id,
                source_state_version=decision.source_state_version,
                queries_executed=[],
                facts=[],
                status="none_needed",
                fallback_hint="use_generic_truthful",
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        action_str = str(getattr(decision.primary_action, "value", decision.primary_action)).lower()

        # Build RetrievalNeed from decision properties
        if not decision.retrieval_needed:
            retrieval_need = None
        elif getattr(decision, "retrieval_need", None) is not None:
            retrieval_need = decision.retrieval_need
        elif not decision.required_facts:
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
            user_id=active_user,
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
            disputed = [w.knowledge_id for w in res.withheld if w.reason in ("disputed", "stale")]
            return RetrievalExecutionReport(
                retrieval_needed=True,
                decision_id=decision.decision_id,
                source_state_version=decision.source_state_version,
                queries_executed=queries,
                facts=legacy_facts,
                disputed_topics=disputed,
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
                    if owner_id and user_id and user_id != "default_user" and owner_id != user_id and owner_id != "admin":
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
