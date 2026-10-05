from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .conversation_state_models import ConversationStateSnapshot
from .core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
    DecisionEvaluationResult,
)
from .core_intelligence_engine import PitchProXCoreIntelligenceEngine
from .core_strategy_modifiers import StrategyModifierPipeline

LOGGER = logging.getLogger("copilot.core_decision_manager")


class StrategicDecisionCache:
    """Explicit cache for StrategicDecision results strictly keyed and validated on all four keys:
    (call_sid, turn_id, source_state_version, source_event_id).
    Matching versions alone are insufficient because multiple turns—and different calls—can share
    the same version number (Client Requirement Point 1 & Point 3).
    """
    _bound_cache: Dict[Tuple[str, int, int, str], StrategicDecision] = {}
    _cache: Dict[Tuple[str, int], StrategicDecision] = {}
    _call_turn_index: Dict[Tuple[str, int], StrategicDecision] = {}

    @classmethod
    def put(
        cls,
        call_sid: str,
        decision: StrategicDecision,
        turn_id: Optional[int] = None,
        source_event_id: Optional[str] = None,
    ) -> None:
        if not call_sid:
            raise ValueError("call_sid is required to cache StrategicDecision; un-scoped caching is prohibited.")
        eff_turn = turn_id if turn_id is not None else getattr(decision, "source_turn_id", None)
        if eff_turn is None:
            eff_turn = getattr(decision, "utterance_turn_id", 0)
        eff_turn = int(eff_turn)
        version_num = int(decision.source_state_version)
        eff_event = str(source_event_id or getattr(decision, "source_event_id", None) or f"ev_turn_{eff_turn}_v{version_num}")

        frozen = decision.model_copy(deep=True)
        frozen.call_sid = str(call_sid)
        frozen.call_id = str(call_sid)
        frozen.source_turn_id = eff_turn
        frozen.source_state_version = version_num
        frozen.source_event_id = eff_event

        # 1. Primary 4-key binding cache: (call_sid, turn_id, source_state_version, source_event_id)
        bound_key = (str(call_sid), eff_turn, version_num, eff_event)
        cls._bound_cache[bound_key] = frozen

        # 2. Auxiliary indices for compatibility
        cls._cache[(str(call_sid), version_num)] = frozen
        cls._call_turn_index[(str(call_sid), eff_turn)] = frozen

    @classmethod
    def get_strictly_bound(
        cls,
        call_sid: str,
        turn_id: int,
        source_state_version: int,
        source_event_id: str,
    ) -> Optional[StrategicDecision]:
        """Validates all four keys together: call_sid + turn_id + source_state_version + source_event_id.
        Rejects lookup if any key mismatches or is missing (Point 1).
        """
        if not call_sid or turn_id is None or source_state_version is None or not source_event_id:
            return None

        key = (str(call_sid), int(turn_id), int(source_state_version), str(source_event_id))
        dec = cls._bound_cache.get(key)
        if dec is not None:
            actual_call = dec.call_sid or dec.call_id
            actual_turn = dec.source_turn_id if dec.source_turn_id is not None else dec.utterance_turn_id
            actual_version = dec.source_state_version
            actual_event = dec.source_event_id
            if (
                actual_call
                and actual_call == str(call_sid)
                and actual_turn is not None
                and actual_turn == int(turn_id)
                and actual_version is not None
                and actual_version == int(source_state_version)
                and actual_event
                and actual_event == str(source_event_id)
            ):
                return dec.model_copy(deep=True)
        return None

    @classmethod
    def validate_binding(
        cls,
        decision: Optional[StrategicDecision],
        call_sid: str,
        turn_id: int,
        source_state_version: int,
        source_event_id: str,
    ) -> Tuple[bool, str]:
        """Validates that a decision strictly matches all 4 keys.
        Fails if any key is missing or null on either side.
        """
        if decision is None:
            return False, "decision is None"
        if not call_sid:
            return False, "Missing call_sid for binding validation"
        if turn_id is None:
            return False, "Missing turn_id for binding validation"
        if source_state_version is None:
            return False, "Missing source_state_version for binding validation"
        if not source_event_id:
            return False, "Missing source_event_id for binding validation"

        actual_call = decision.call_sid or decision.call_id
        if not actual_call:
            return False, "Missing call_sid on StrategicDecision"
        if actual_call != str(call_sid):
            return False, f"call_sid mismatch: expected '{call_sid}', got '{actual_call}'"

        actual_turn = decision.source_turn_id if decision.source_turn_id is not None else decision.utterance_turn_id
        if actual_turn is None:
            return False, "Missing turn_id on StrategicDecision"
        if actual_turn != int(turn_id):
            return False, f"turn_id mismatch: expected Turn {turn_id}, got Turn {actual_turn}"

        if decision.source_state_version is None:
            return False, "Missing source_state_version on StrategicDecision"
        if int(decision.source_state_version) != int(source_state_version):
            return False, f"source_state_version mismatch: expected V{source_state_version}, got V{decision.source_state_version}"

        actual_event = decision.source_event_id
        if not actual_event:
            return False, "Missing source_event_id on StrategicDecision: decision lacks explicit event provenance"

        expected_event = str(source_event_id)
        if actual_event != expected_event and actual_event != f"ev_t{turn_id}":
            return False, f"source_event_id mismatch: expected '{expected_event}', got '{actual_event}'"

        return True, "Valid 4-key binding"

    @classmethod
    def get(
        cls,
        call_sid: str,
        source_state_version: int,
        turn_id: Optional[int] = None,
        source_event_id: Optional[str] = None,
    ) -> Optional[StrategicDecision]:
        """Retrieves a cached decision. When turn_id and source_event_id are provided,
        strictly validates all 4 keys together (call_sid, turn_id, source_state_version, source_event_id).
        """
        if not call_sid:
            raise ValueError("call_sid is required to query cached StrategicDecision.")

        if turn_id is not None and source_event_id is not None:
            return cls.get_strictly_bound(
                call_sid=call_sid,
                turn_id=turn_id,
                source_state_version=source_state_version,
                source_event_id=source_event_id,
            )

        if turn_id is not None:
            dec = cls._call_turn_index.get((str(call_sid), int(turn_id)))
            if dec and dec.source_state_version == int(source_state_version):
                return dec.model_copy(deep=True)
            return None

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
            cls._bound_cache.clear()
            cls._cache.clear()
            cls._call_turn_index.clear()
        else:
            sid_str = str(call_sid)
            cls._bound_cache = {k: v for k, v in cls._bound_cache.items() if k[0] != sid_str}
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
        max_carried_turns: int = 2,
        confidence_threshold: Optional[float] = None,
        unmeasured_trust_cap: Optional[float] = None,
        momentum_adjustments: Optional[Dict[str, float]] = None,
    ):
        if not call_sid:
            raise ValueError("call_sid must be explicitly specified; anonymous or shared singletons are prohibited.")
        self.call_sid = str(call_sid)
        self.lead_type = lead_type
        if engine is not None:
            self.engine = engine
        else:
            self.engine = PitchProXCoreIntelligenceEngine(
                lead_type=lead_type,
                confidence_threshold=confidence_threshold,
                unmeasured_trust_cap=unmeasured_trust_cap,
                momentum_adjustments=momentum_adjustments,
            )
        self.modifier_pipeline = modifier_pipeline or StrategyModifierPipeline()
        self.max_carried_turns = max_carried_turns
        self.decision_history: List[StrategicDecision] = []
        self.decisions_by_turn: Dict[int, StrategicDecision] = {}
        self.decisions_by_version: Dict[int, StrategicDecision] = {}
        self.latest_decision: Optional[StrategicDecision] = None
        self.latest_evaluation_result: Optional[DecisionEvaluationResult] = None
        self.consecutive_carried_count: int = 0

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
        effective_event_id = f"ev_turn_{effective_turn_id}_v{snapshot.state_version}" if (effective_turn_id is not None and snapshot.state_version is not None) else None

        # Point 1 & Point 3: Process-level cache check validating all four keys
        if use_cache and snapshot.state_version is not None and effective_turn_id is not None and effective_event_id and not playbook and not calibration:
            cached_decision = StrategicDecisionCache.get_strictly_bound(
                call_sid=self.call_sid,
                turn_id=int(effective_turn_id),
                source_state_version=int(snapshot.state_version),
                source_event_id=effective_event_id,
            )
            if cached_decision is not None:
                context = self.engine._build_context(snapshot)
                eval_result = DecisionEvaluationResult(
                    decision=cached_decision,
                    context=context,
                )
                self.latest_decision = cached_decision
                self.latest_evaluation_result = eval_result
                self.decision_history.append(cached_decision)
                self.decisions_by_turn[int(effective_turn_id)] = cached_decision
                self.decisions_by_version[int(cached_decision.source_state_version)] = cached_decision
                return eval_result

        # Point 13: Preserve strategy across turns when nothing material changed
        filler_acknowledgments = {"okay", "ok", "yeah", "sure", "uh huh", "uh-huh", "got it", "right", "i see"}
        cleaned_turn = (turn_text or "").strip().lower().rstrip("!.,")
        is_filler = cleaned_turn in filler_acknowledgments

        # Question pending check: if the previous action was an explicit question, closing ask,
        # coordination inquiry, or if the prompt ended with '?',
        # 'okay' / 'yeah' is an affirmative answer to the prompt, NOT an aimless filler backchannel!
        prev_dec = self.latest_decision
        has_pending_question = False
        if prev_dec is not None:
            has_pending_question = (
                prev_dec.primary_action in (StrategicAction.QUESTION, StrategicAction.COMMITMENT_CLOSE)
                or prev_dec.secondary_action in (StrategicAction.QUESTION, StrategicAction.COMMITMENT_CLOSE)
                or prev_dec.strategic_posture == "coordinate"
                or (bool(prev_dec.final_prompt_text) and prev_dec.final_prompt_text.strip().endswith("?"))
                or (bool(prev_dec.gateway_fallback_stub) and prev_dec.gateway_fallback_stub.strip().endswith("?"))
                or bool(re.search(r"\b(ask|inquire|propose|schedule|question|verify)\b", prev_dec.strategic_objective, re.I))
            )

        # Staleness limit: Max consecutive carried-forward turns before requiring fresh strategic evaluation
        staleness_limit_reached = self.consecutive_carried_count >= self.max_carried_turns

        if (
            is_filler
            and prev_dec is not None
            and not has_pending_question
            and not staleness_limit_reached
            and not playbook
            and not calibration
        ):
            if prev_dec.primary_action not in (StrategicAction.WAIT_SILENCE, StrategicAction.HOLD) and "HARD_BOUNDARY_ACTIVE" not in prev_dec.reason_codes:
                self.consecutive_carried_count += 1
                carried_dec = prev_dec.model_copy(deep=True)
                carried_dec.decision_id = f"dec_{uuid.uuid4().hex[:10]}"
                carried_dec.source_turn_id = effective_turn_id
                carried_dec.utterance_turn_id = effective_turn_id
                carried_dec.metrics_source_turn_id = effective_turn_id
                carried_dec.source_state_version = snapshot.state_version
                carried_dec.source_event_id = effective_event_id
                # Chained carry-forward: always point to the original root decision and turn
                carried_dec.carried_forward_from_decision_id = prev_dec.carried_forward_from_decision_id or prev_dec.decision_id
                carried_dec.carried_forward_from_turn_id = prev_dec.carried_forward_from_turn_id or prev_dec.source_turn_id
                carried_dec.should_prompt = False  # Point 8: no new prompt needed for non-material filler
                if "STRATEGY_CARRIED_FORWARD" not in carried_dec.reason_codes:
                    carried_dec.reason_codes.append("STRATEGY_CARRIED_FORWARD")
                carried_dec.strategic_objective = f"Carried forward from Turn {carried_dec.carried_forward_from_turn_id}: {prev_dec.strategic_objective}"
                carried_dec.evidence_considered = [
                    f'Turn utterance ({turn_speaker}): "{turn_text.strip()}"',
                    f"Strategy carried forward from decision {carried_dec.carried_forward_from_decision_id} (Turn {carried_dec.carried_forward_from_turn_id})",
                ]

                context = self.engine._build_context(snapshot)
                eval_result = DecisionEvaluationResult(decision=carried_dec, context=context)
                self.latest_decision = carried_dec
                self.latest_evaluation_result = eval_result
                self.decision_history.append(carried_dec)
                if effective_turn_id is not None:
                    self.decisions_by_turn[int(effective_turn_id)] = carried_dec
                self.decisions_by_version[int(carried_dec.source_state_version)] = carried_dec
                return eval_result
        else:
            self.consecutive_carried_count = 0

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

        # Rep turn continuation check (Point 13 & Point 8):
        # If rep is speaking and the resulting strategy is an unchanged continuation of prev_dec,
        # mark as carried forward and suppress prompt generation to prevent stale/redundant guidance.
        if (
            turn_speaker in ("salesperson", "rep", "agent")
            and prev_dec is not None
            and not playbook
            and not calibration
            and final_decision.primary_action == prev_dec.primary_action
            and final_decision.strategic_posture == prev_dec.strategic_posture
            and final_decision.push_strength == prev_dec.push_strength
            and final_decision.secondary_action == prev_dec.secondary_action
            and set(final_decision.reason_codes) == set(r for r in prev_dec.reason_codes if r != "STRATEGY_CARRIED_FORWARD")
        ):
            carried_dec = prev_dec.model_copy(deep=True)
            carried_dec.decision_id = f"dec_{uuid.uuid4().hex[:10]}"
            carried_dec.source_turn_id = effective_turn_id
            carried_dec.utterance_turn_id = effective_turn_id
            carried_dec.metrics_source_turn_id = effective_turn_id
            carried_dec.source_state_version = snapshot.state_version
            carried_dec.source_event_id = effective_event_id
            carried_dec.carried_forward_from_decision_id = prev_dec.carried_forward_from_decision_id or prev_dec.decision_id
            carried_dec.carried_forward_from_turn_id = prev_dec.carried_forward_from_turn_id or prev_dec.source_turn_id
            carried_dec.should_prompt = False  # Point 8: Rep already spoke; suppress prompt generation
            if "STRATEGY_CARRIED_FORWARD" not in carried_dec.reason_codes:
                carried_dec.reason_codes.append("STRATEGY_CARRIED_FORWARD")
            carried_dec.strategic_objective = f"Continuation from Turn {carried_dec.carried_forward_from_turn_id}: {prev_dec.strategic_objective}"
            if carried_dec.evidence_considered:
                carried_dec.evidence_considered = [f'Turn utterance ({turn_speaker}): "{turn_text.strip()}"'] + [e for e in carried_dec.evidence_considered if not e.startswith("Turn utterance")]
            else:
                carried_dec.evidence_considered = [f'Turn utterance ({turn_speaker}): "{turn_text.strip()}"']
            final_decision = carried_dec

        # Prompt dispatch happens only on prospect utterances (Point 8 invariant)
        # Suppress prompts on all rep turns whether continuation or fresh evaluation
        if turn_speaker in ("salesperson", "rep", "agent"):
            final_decision.should_prompt = False

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

        # Point 1 & Point 3: Populate cache strictly keyed on all four keys
        StrategicDecisionCache.put(
            call_sid=self.call_sid,
            decision=frozen_decision,
            turn_id=effective_turn_id,
            source_event_id=frozen_decision.source_event_id,
        )

        return eval_result

    def get_decision_for_turn(self, turn_id: int) -> Optional[StrategicDecision]:
        """Returns the immutable StrategicDecision produced strictly at turn_id for this call_sid."""
        if int(turn_id) in self.decisions_by_turn:
            dec = self.decisions_by_turn[int(turn_id)]
            if (dec.call_sid or dec.call_id) == self.call_sid and (dec.source_turn_id == int(turn_id) or dec.utterance_turn_id == int(turn_id)):
                return dec
        return StrategicDecisionCache.get_by_call_and_turn(self.call_sid, int(turn_id))

    def get_decision_for_version(self, state_version: int) -> Optional[StrategicDecision]:
        """Returns the immutable StrategicDecision produced strictly for this call_sid at source_state_version."""
        if int(state_version) in self.decisions_by_version:
            return self.decisions_by_version[int(state_version)]
        return StrategicDecisionCache.get(self.call_sid, int(state_version))

    def get_cached_decision(self, source_state_version: int) -> Optional[StrategicDecision]:
        """Queries the cache by (call_sid, source_state_version)."""
        return StrategicDecisionCache.get(self.call_sid, int(source_state_version))

    def get_strictly_bound_decision(
        self,
        call_sid: str,
        turn_id: int,
        source_state_version: int,
        source_event_id: str,
    ) -> Optional[StrategicDecision]:
        """Retrieves and validates a decision matching all four keys together:
        call_sid + turn_id + source_state_version + source_event_id (Point 1).
        """
        if str(call_sid) != self.call_sid:
            return None
        return StrategicDecisionCache.get_strictly_bound(
            call_sid=str(call_sid),
            turn_id=int(turn_id),
            source_state_version=int(source_state_version),
            source_event_id=str(source_event_id),
        )



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
