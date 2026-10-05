"""Prospect Memory Architecture for PitchProX Copilot.

Enforces strict user-scoped memory isolation per Group 3 (Points 15-18):
    user_id -> prospect_id -> prospect memory

Key Architectural Principles:
1. Point 15: Mandatory user_id keying. Memory is never shared across users.
   If User A and User B call the same phone number or prospect, User B has zero access
   to User A's memories. Normalizes user_id and prospect_id (phone formats) to prevent
   partition splitting or collision.
2. Point 16: Standardized schema with all required fields:
   - user_id
   - prospect_id
   - source_call_sid
   - timestamp
   - memory_type
   - source_event_id / source_turn_id (where applicable)
3. Point 17: Five verified scenarios:
   - Same user + same prospect (loads under policy)
   - New call + same user + same prospect (fresh state, memory loaded separately with provenance)
   - Different user + same prospect/phone (zero retrieval, zero exposure)
   - New simulation/test (memory off by default)
   - Any current call (state isolated to call_sid)
4. Point 18: No team or shared memory. Structural guard against unpartitioned retrieval.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

LOGGER = logging.getLogger(__name__)


def normalize_user_id(user_id: Any) -> str:
    """Normalizes user_id to prevent type confusion (e.g. int vs str) or whitespace collisions."""
    if user_id is None:
        raise ValueError("user_id is mandatory and cannot be None (Point 15)")
    norm = str(user_id).strip()
    if not norm:
        raise ValueError("user_id is mandatory and cannot be empty (Point 15)")
    return norm


def normalize_prospect_id(prospect_id: Any) -> str:
    """Canonicalizes phone numbers and prospect identifiers to prevent formatting bypass.

    Normalizes phone variants (+1-555-0100, +15550100, (555) 0100) to standard E.164 digits,
    while trimming non-phone alphanumeric IDs.
    """
    if prospect_id is None:
        raise ValueError("prospect_id is mandatory and cannot be None (Point 15)")
    raw = str(prospect_id).strip()
    if not raw:
        raise ValueError("prospect_id is mandatory and cannot be empty (Point 15)")

    digits_only = "".join(ch for ch in raw if ch.isdigit())
    # If it has standard phone digit length and only phone characters, canonicalize digits
    if len(digits_only) >= 7 and all(ch.isdigit() or ch in "+-(). " for ch in raw):
        # Handle NANP (North American) numbers with or without country code 1
        if len(digits_only) in (7, 10):
            return f"+1{digits_only}"
        if len(digits_only) in (8, 11) and digits_only.startswith("1"):
            return f"+{digits_only}"
        # Other international or standard numbers
        return f"+{digits_only}" if raw.startswith("+") else digits_only
    return raw.lower()


class MemoryProvenance(BaseModel):
    """Provenance tracking for historical memory loaded into a conversation."""
    source_call_sid: str
    source_turn_id: Optional[int] = None
    source_event_id: Optional[str] = None
    original_timestamp: int = Field(..., ge=1)
    originating_user_id: str
    originating_prospect_id: str
    loaded_at_timestamp: int = Field(default_factory=lambda: int(time.time() * 1000), ge=1)


class LoadedProspectMemory(BaseModel):
    """Eligible historical memory record attached to a conversation state.

    Kept separate from current-call facts to prevent silent fact conflation.
    """
    memory_id: str
    user_id: str
    prospect_id: str
    memory_type: str
    key: str
    value: Any
    content: str
    provenance: MemoryProvenance
    confidence: float = 1.0
    data: Dict[str, Any] = Field(default_factory=dict)
    is_historical: bool = True


class ProspectMemoryRecord(BaseModel):
    """Canonical schema for persistent Prospect Memory items (Point 16)."""
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:10]}")
    user_id: str = Field(..., description="Mandatory owner ID (salesperson). Never optional.")
    prospect_id: str = Field(..., description="Target prospect identifier or canonical phone number.")
    source_call_sid: str = Field(..., description="The call_sid where this memory was captured.")
    timestamp: int = Field(..., ge=1, description="Unix timestamp (ms). Positive integer required.")
    memory_type: str = Field(..., description="Category of memory: contact_preference, timeline, fact, objection, constraint.")
    source_turn_id: Optional[int] = Field(None, description="Originating turn ID within the source call.")
    source_event_id: Optional[str] = Field(None, description="Originating event ID if attached to a conversion/compliance event.")
    key: str = Field(..., description="Semantic lookup key.")
    value: Any = Field(..., description="Semantic value or structured payload.")
    content: str = Field(..., description="Human-readable memory summary or quote.")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: Any) -> str:
        return normalize_user_id(v)

    @field_validator("prospect_id")
    @classmethod
    def validate_prospect_id(cls, v: Any) -> str:
        return normalize_prospect_id(v)

    @field_validator("source_call_sid")
    @classmethod
    def validate_source_call_sid(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("source_call_sid is mandatory and cannot be empty (Point 16)")
        return str(v).strip()

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("memory_type is mandatory and cannot be empty (Point 16)")
        return str(v).strip()

    @model_validator(mode="after")
    def validate_in_call_applicability(self) -> ProspectMemoryRecord:
        """Point 16 'where applicable' enforcement:
        Any record derived from in-call interaction (source_call_sid begins with 'call_' or 'CA_' or 'sim_')
        must provide at least one of source_turn_id or source_event_id.
        """
        is_in_call = any(self.source_call_sid.lower().startswith(prefix) for prefix in ("call_", "ca_", "sim_"))
        if is_in_call and self.source_turn_id is None and not self.source_event_id:
            raise ValueError(
                "Point 16 violation: in-call memory records must supply at least one of "
                "source_turn_id or source_event_id ('where applicable')."
            )
        return self

    def to_loaded_memory(self) -> LoadedProspectMemory:
        """Converts persistent record into an immutable loaded snapshot item with provenance."""
        provenance = MemoryProvenance(
            source_call_sid=self.source_call_sid,
            source_turn_id=self.source_turn_id,
            source_event_id=self.source_event_id,
            original_timestamp=self.timestamp,
            originating_user_id=self.user_id,
            originating_prospect_id=self.prospect_id,
        )
        return LoadedProspectMemory(
            memory_id=self.memory_id,
            user_id=self.user_id,
            prospect_id=self.prospect_id,
            memory_type=self.memory_type,
            key=self.key,
            value=self.value,
            content=self.content,
            provenance=provenance,
            confidence=self.confidence,
            data=dict(self.metadata),
            is_historical=True,
        )


class ProspectMemoryStore:
    """Strictly partitioned Prospect Memory Store.

    Data hierarchy:
        user_id -> prospect_id -> List[ProspectMemoryRecord]

    Architectural Invariants:
    1. Physical user partition: self._storage is a pure dict, where lookups are side-effect-free.
       Unknown user queries return [] without creating dictionary keys.
    2. Universal queries across users or queries missing user_id are strictly prohibited (Point 18).
    3. Both user_id and prospect_id are normalized on all read and write paths.
    """

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            base_dir = Path(__file__).resolve().parent.parent / "knowledge"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.store_path = base_dir / "prospect_memory_store.json"
        else:
            self.store_path = store_path

        # Pure dict storage partitioned by user_id -> prospect_id -> List[ProspectMemoryRecord]
        self._storage: Dict[str, Dict[str, List[ProspectMemoryRecord]]] = {}
        self._load()

    def _load(self) -> None:
        """Loads records from disk with strict schema parsing."""
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Format: {user_id: {prospect_id: [record_dicts]}}
            for u_id, p_dict in raw_data.items():
                norm_u = normalize_user_id(u_id)
                if norm_u not in self._storage:
                    self._storage[norm_u] = {}
                if isinstance(p_dict, dict):
                    for p_id, records in p_dict.items():
                        norm_p = normalize_prospect_id(p_id)
                        if norm_p not in self._storage[norm_u]:
                            self._storage[norm_u][norm_p] = []
                        for rec in records:
                            try:
                                record_obj = ProspectMemoryRecord.model_validate(rec)
                                self._storage[norm_u][norm_p].append(record_obj)
                            except Exception as parse_err:
                                LOGGER.warning("Invalid memory record skipped: %s", parse_err)
        except Exception as exc:
            LOGGER.warning("Could not load ProspectMemoryStore from %s: %s", self.store_path, exc)

    def _persist(self) -> None:
        """Persists current partitioned storage to disk."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            serializable: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for u_id, p_map in self._storage.items():
                serializable[u_id] = {}
                for p_id, records in p_map.items():
                    serializable[u_id][p_id] = [r.model_dump() for r in records]

            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as exc:
            LOGGER.warning("Could not persist ProspectMemoryStore to %s: %s", self.store_path, exc)

    def save_memory(self, record: ProspectMemoryRecord) -> ProspectMemoryRecord:
        """Saves a memory record under its mandatory user_id and prospect_id."""
        u_id = normalize_user_id(record.user_id)
        p_id = normalize_prospect_id(record.prospect_id)

        if u_id not in self._storage:
            self._storage[u_id] = {}
        if p_id not in self._storage[u_id]:
            self._storage[u_id][p_id] = []

        existing = self._storage[u_id][p_id]
        for idx, item in enumerate(existing):
            if item.memory_id == record.memory_id:
                existing[idx] = record
                self._persist()
                return record

        existing.append(record)
        self._persist()
        return record

    def save_record(
        self,
        user_id: Any,
        prospect_id: Any,
        source_call_sid: str,
        memory_type: str,
        key: str,
        value: Any,
        content: str,
        timestamp: Optional[int] = None,
        source_turn_id: Optional[int] = None,
        source_event_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProspectMemoryRecord:
        """Convenience method to construct, validate, and store a ProspectMemoryRecord."""
        norm_u = normalize_user_id(user_id)
        norm_p = normalize_prospect_id(prospect_id)
        record = ProspectMemoryRecord(
            user_id=norm_u,
            prospect_id=norm_p,
            source_call_sid=source_call_sid,
            timestamp=timestamp or int(time.time() * 1000),
            memory_type=memory_type,
            source_turn_id=source_turn_id,
            source_event_id=source_event_id,
            key=key,
            value=value,
            content=content,
            confidence=confidence,
            metadata=metadata or {},
        )
        return self.save_memory(record)

    def retrieve_memories(
        self,
        user_id: Any,
        prospect_id: Any,
        memory_type: Optional[str] = None,
    ) -> List[ProspectMemoryRecord]:
        """Retrieves eligible memories strictly for the requesting user_id and prospect_id.

        Point 15 & Point 18 Security Invariants:
        1. Universal or unscoped lookups are strictly prohibited. user_id must be provided.
        2. Side-effect free: querying non-existent user_ids or prospect_ids returns []
           without adding keys or modifying partition counts.
        """
        if user_id is None:
            raise ValueError("user_id is mandatory for prospect memory retrieval; universal lookups prohibited (Point 18)")
        u_id = normalize_user_id(user_id)

        if prospect_id is None:
            return []
        p_id = normalize_prospect_id(prospect_id)

        user_bucket = self._storage.get(u_id)
        if user_bucket is None:
            return []

        records = user_bucket.get(p_id, [])
        if memory_type:
            return [m for m in records if m.memory_type == memory_type]
        return list(records)

    def get_shared_team_memories(self, *args, **kwargs):
        """Explicit guard against team or shared memory (Point 18)."""
        raise NotImplementedError(
            "Team or shared memory is explicitly prohibited per Point 18 instruction: "
            "'Confirm there is no code path that shares memory across users/teams.'"
        )

    def delete_for_user(self, user_id: Any, prospect_id: Optional[Any] = None) -> int:
        """Deletes memories belonging strictly to a single user."""
        u_id = normalize_user_id(user_id)
        if u_id not in self._storage:
            return 0
        if prospect_id is not None:
            p_id = normalize_prospect_id(prospect_id)
            count = len(self._storage[u_id].pop(p_id, []))
            self._persist()
            return count
        else:
            count = sum(len(recs) for recs in self._storage[u_id].values())
            del self._storage[u_id]
            self._persist()
            return count

    def clear(self, test_only_confirmation: bool = False) -> None:
        """Clears all records in memory and on disk (strictly for test fixture setup)."""
        import os
        is_test_env = "PYTEST_CURRENT_TEST" in os.environ or test_only_confirmation
        if not is_test_env:
            raise RuntimeError(
                "ProspectMemoryStore.clear() is restricted to test environments. "
                "Production callers must not invoke unpartitioned clear operations."
            )
        self._storage.clear()
        if self.store_path.exists():
            try:
                self.store_path.unlink()
            except Exception as exc:
                LOGGER.warning("Could not unlink %s: %s", self.store_path, exc)
