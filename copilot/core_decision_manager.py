from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .conversation_state_models import ConversationStateSnapshot
from .core_intelligence_models import (
    StrategicDecision,
    DecisionEvaluationResult,
)
from .core_intelligence_engine import PitchProXCoreIntelligenceEngine
from .core_strategy_modifiers import StrategyModifierPipeline

LOGGER = logging.getLogger("copilot.core_decision_manager")


class StrategicDecisionCache:
    """Explicit cache for StrategicDecision results keyed strictly on (call_sid, source_state_version).
    Never keyed on turn number alone or un-scoped pointers (Client Requirement Point 3).
    """
    _cache: Dict[Tuple[str, int], StrategicDecision] = {}
    _call_turn_index: Dict[Tuple[str, int], StrategicDecision] = {}

    @classmethod
    def put(
        cls,
        call_sid: str,
        decision: StrategicDecision,
        turn_id: Optional[int] = None,
    ) -> None:
        if not call_sid:
            raise ValueError("call_sid is required to cache StrategicDecision; un-scoped caching is prohibited.")
        version_key = (str(call_sid), int(decision.source_state_version))
        frozen = decision.model_copy(deep=True)
        cls._cache[version_key] = frozen
        if turn_id is not None:
            turn_key = (str(call_sid), int(turn_id))
            cls._call_turn_index[turn_key] = frozen

    @classmethod
    def get(cls, call_sid: str, source_state_version: int) -> Optional[StrategicDecision]:
        """Retrieves a cached decision strictly by (call_sid, source_state_version)."""
        if not call_sid:
            raise ValueError("call_sid is required to query cached StrategicDecision.")
        dec = cls._cache.get((str(call_sid), int(source_state_version)))
        return dec.model_copy(deep=True) if dec else None

    @classmethod
    def get_by_call_and_turn(cls, call_sid: str, turn_id: int) -> Optional[StrategicDecision]:
        """Retrieves a cached decision strictly scoped by (call_sid, turn_id).
        Never allows lookup on turn number alone.
        """
        if not call_sid:
            raise ValueError("call_sid is required; turn lookup cannot be performed without call_sid.")
        dec = cls._call_turn_index.get((str(call_sid), int(turn_id)))
        return dec.model_copy(deep=True) if dec else None

    @classmethod
    def clear(cls, call_sid: Optional[str] = None) -> None:
        """Clears cache entries. If call_sid is specified, clears only that call's entries."""
        if call_sid is None:
            cls._cache.clear()
            cls._call_turn_index.clear()
        else:
            sid_str = str(call_sid)
            cls._cache = {k: v for k, v in cls._cache.items() if k[0] != sid_str}
            cls._call_turn_index = {k: v for k, v in cls._call_turn_index.items() if k[0] != sid_str}


class CoreDecisionManager:
    """Coordinates Core Intelligence evaluation, two-tiered stale-decision tracking, and state version lineage."""

    def __init__(
        self,
        call_sid: str,
        lead_type: str = "general",
        engine: Optional[PitchProXCoreIntelligenceEngine] = None,
        modifier_pipeline: Optional[StrategyModifierPipeline] = None,
    ):
        if not call_sid:
            raise ValueError("call_sid must be explicitly specified; anonymous or shared singletons are prohibited.")
        self.call_sid = str(call_sid)
        self.lead_type = lead_type
        self.engine = engine or PitchProXCoreIntelligenceEngine(lead_type=lead_type)
        self.modifier_pipeline = modifier_pipeline or StrategyModifierPipeline()
        self.decision_history: List[StrategicDecision] = []
        self.decisions_by_turn: Dict[int, StrategicDecision] = {}
        self.decisions_by_version: Dict[int, StrategicDecision] = {}
        self.latest_decision: Optional[StrategicDecision] = None
        self.latest_evaluation_result: Optional[DecisionEvaluationResult] = None

    def evaluate_state(
        self,
        snapshot: ConversationStateSnapshot,
        playbook: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        turn_speaker: str = "prospect",
        turn_text: str = "",
        turn_timestamp_ms: int = 0,
        turn_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> DecisionEvaluationResult:
        effective_turn_id = turn_id if turn_id is not None else getattr(snapshot, "last_updated_turn_id", None)

        # Point 3: Process-level cache check keyed strictly on (call_sid, source_state_version)
        if use_cache and snapshot.state_version is not None and not playbook and not calibration:
            cached_decision = StrategicDecisionCache.get(self.call_sid, snapshot.state_version)
            if cached_decision is not None and (effective_turn_id is None or cached_decision.source_turn_id == effective_turn_id):
                context = self.engine._build_context(snapshot)
                eval_result = DecisionEvaluationResult(
                    decision=cached_decision,
                    context=context,
                )
                self.latest_decision = cached_decision
                self.latest_evaluation_result = eval_result
                self.decision_history.append(cached_decision)
                if effective_turn_id is not None:
                    self.decisions_by_turn[int(effective_turn_id)] = cached_decision
                self.decisions_by_version[int(cached_decision.source_state_version)] = cached_decision
                return eval_result

        eval_result = self.engine.evaluate(
            snapshot=snapshot,
            turn_speaker=turn_speaker,
            turn_text=turn_text,
            turn_timestamp_ms=turn_timestamp_ms,
            turn_id=effective_turn_id,
        )

        final_decision = self.modifier_pipeline.apply_modifiers(
            decision=eval_result.decision,
            playbook=playbook,
            calibration=calibration,
        )

        # Ensure deep-copied immutable decision snapshot per turn
        frozen_decision = final_decision.model_copy(deep=True)
        eval_result.decision = frozen_decision
        self.latest_decision = frozen_decision
        self.latest_evaluation_result = eval_result
        self.decision_history.append(frozen_decision)

        effective_turn_id = turn_id if turn_id is not None else getattr(snapshot, "last_updated_turn_id", None)
        if effective_turn_id is not None:
            self.decisions_by_turn[int(effective_turn_id)] = frozen_decision
        self.decisions_by_version[int(frozen_decision.source_state_version)] = frozen_decision

        # Point 3: Populate cache strictly keyed on (call_sid, source_state_version)
        StrategicDecisionCache.put(
            call_sid=self.call_sid,
            decision=frozen_decision,
            turn_id=effective_turn_id,
        )

        return eval_result

    def get_decision_for_turn(self, turn_id: int) -> Optional[StrategicDecision]:
        """Returns the immutable StrategicDecision produced strictly at turn_id for this call_sid."""
        if int(turn_id) in self.decisions_by_turn:
            return self.decisions_by_turn[int(turn_id)]
        return StrategicDecisionCache.get_by_call_and_turn(self.call_sid, int(turn_id))

    def get_decision_for_version(self, state_version: int) -> Optional[StrategicDecision]:
        """Returns the immutable StrategicDecision produced strictly for this call_sid at source_state_version."""
        if int(state_version) in self.decisions_by_version:
            return self.decisions_by_version[int(state_version)]
        return StrategicDecisionCache.get(self.call_sid, int(state_version))

    def get_cached_decision(self, source_state_version: int) -> Optional[StrategicDecision]:
        """Queries the cache by (call_sid, source_state_version)."""
        return StrategicDecisionCache.get(self.call_sid, int(source_state_version))



    def is_decision_stale(
        self,
        decision: StrategicDecision,
        current_snapshot: ConversationStateSnapshot,
    ) -> bool:
        """Tier 1 Checkpoint: Pre-dispatch staleness check before sending decision to LLM Gateway/Retrieval."""
        if current_snapshot.state_version == decision.source_state_version:
            return False

        if current_snapshot.state_version < decision.source_state_version:
            return False

        # Invalidate if compliance status changed
        if current_snapshot.contact_compliance.hard_boundary_active:
            if "HARD_BOUNDARY_ACTIVE" not in decision.reason_codes:
                return True

        # Invalidate if conversion event status changed
        curr_conv = current_snapshot.get_active_conversion_event()
        is_curr_confirmed = curr_conv and curr_conv.status == "confirmed"
        was_confirmed = "CONVERSION_CONFIRMED" in decision.reason_codes
        if is_curr_confirmed != was_confirmed:
            return True

        # Invalidate if active objections set changed
        curr_unresolved = {o.canonical_category for o in current_snapshot.get_unresolved_objections()}
        dec_unresolved = set()
        if self.latest_evaluation_result:
            dec_unresolved = set(self.latest_evaluation_result.context.active_objections)
        if curr_unresolved != dec_unresolved:
            return True

        return True

    def validate_for_teleprompter_display(
        self,
        source_state_version: int,
        current_snapshot: ConversationStateSnapshot,
    ) -> Tuple[bool, Optional[str]]:
        """Tier 2 Checkpoint: Pre-display gate verifying prompt source_state_version is valid at the moment of display."""
        if current_snapshot.state_version == source_state_version:
            return True, None

        if current_snapshot.state_version < source_state_version:
            return True, None

        # Material supersession checks
        if current_snapshot.contact_compliance.hard_boundary_active:
            return False, "HARD_BOUNDARY_SUPERSEDED_PROMPT"

        curr_conv = current_snapshot.get_active_conversion_event()
        if curr_conv and curr_conv.status == "confirmed":
            return False, "CONVERSION_ALREADY_CONFIRMED_STALE_PROMPT"

        # Check if new turn has materially changed state
        return False, f"STALE_STATE_VERSION_SUPERSEDED: prompt_v{source_state_version}_current_v{current_snapshot.state_version}"
