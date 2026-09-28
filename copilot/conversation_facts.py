from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .conversation_state_models import (
    PersistentFactCategory,
    PersistentFactRecord,
)

LOGGER = logging.getLogger("copilot.conversation_facts")


class PersistentFactsManager:
    """Manages long-lived conversation facts that never expire until explicitly superseded.

    Guarantees:
    1. Facts stated early remain active across 20+ unrelated turns.
    2. Superseding a fact never deletes history; old fact is marked 'superseded' with bidirectional linking.
    3. Audit trail preserves source turn ID, timestamp, and confidence.
    """

    def __init__(self, initial_facts: Optional[List[PersistentFactRecord]] = None):
        self._facts: List[PersistentFactRecord] = list(initial_facts) if initial_facts else []

    @property
    def facts(self) -> List[PersistentFactRecord]:
        """Returns the internal list of fact records."""
        return self._facts

    @facts.setter
    def facts(self, value: List[PersistentFactRecord]) -> None:
        """Sets the internal list of fact records."""
        self._facts = list(value)

    def record_fact(
        self,
        category: PersistentFactCategory,
        fact_key: str,
        fact_value: str,
        source_turn_id: int,
        timestamp_ms: int,
        confidence: float = 1.0,
        notes: Optional[str] = None,
    ) -> PersistentFactRecord:
        """Records a new fact. If an active fact with the same key exists with a different value,

        it is automatically superseded.
        """
        existing_active = self.get_active_fact(fact_key)
        if existing_active:
            if existing_active.fact_value.strip().lower() == fact_value.strip().lower():
                # Value unchanged: preserve original source turn while updating confidence if higher
                if confidence > existing_active.confidence:
                    existing_active.confidence = confidence
                return existing_active

            # Value has changed: supersede existing
            _, new_fact = self.supersede_fact(
                old_fact_id=existing_active.fact_id,
                new_fact_value=fact_value,
                source_turn_id=source_turn_id,
                timestamp_ms=timestamp_ms,
                confidence=confidence,
                notes=notes,
            )
            return new_fact

        # Brand new fact
        new_fact = PersistentFactRecord(
            category=category,
            fact_key=fact_key,
            fact_value=fact_value,
            status="active",
            confidence=confidence,
            source_turn_id=source_turn_id,
            timestamp_ms=timestamp_ms,
            notes=notes,
        )
        self._facts.append(new_fact)
        return new_fact

    def supersede_fact(
        self,
        old_fact_id: str,
        new_fact_value: str,
        source_turn_id: int,
        timestamp_ms: int,
        confidence: float = 1.0,
        notes: Optional[str] = None,
    ) -> Tuple[PersistentFactRecord, PersistentFactRecord]:
        """Supersedes an existing active fact, linking both records in an immutable audit trail."""
        old_fact = next((f for f in self._facts if f.fact_id == old_fact_id), None)
        if not old_fact:
            raise ValueError(f"Fact with ID '{old_fact_id}' not found in store.")

        new_fact = PersistentFactRecord(
            category=old_fact.category,
            fact_key=old_fact.fact_key,
            fact_value=new_fact_value,
            status="active",
            confidence=confidence,
            source_turn_id=source_turn_id,
            timestamp_ms=timestamp_ms,
            notes=notes,
        )

        old_fact.status = "superseded"
        old_fact.superseded_by_fact_id = new_fact.fact_id
        old_fact.superseded_at_turn_id = source_turn_id

        self._facts.append(new_fact)
        return old_fact, new_fact

    def get_active_fact(self, fact_key: str) -> Optional[PersistentFactRecord]:
        """Returns the current active fact for a given key, if any."""
        for fact in reversed(self._facts):
            if fact.fact_key == fact_key and fact.status == "active":
                return fact
        return None

    def get_active_facts(self) -> List[PersistentFactRecord]:
        """Returns all currently active facts."""
        return [f for f in self._facts if f.status == "active"]

    def get_superseded_facts(self) -> List[PersistentFactRecord]:
        """Returns all superseded historical facts."""
        return [f for f in self._facts if f.status == "superseded"]

    def get_fact_history(self, fact_key: str) -> List[PersistentFactRecord]:
        """Returns the complete chronological history of a fact key (both superseded and active)."""
        return [f for f in self._facts if f.fact_key == fact_key]

    def get_all_facts(self) -> List[PersistentFactRecord]:
        """Returns all facts in chronological order."""
        return list(self._facts)
