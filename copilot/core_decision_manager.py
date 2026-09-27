from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .conversation_state_models import ConversationStateSnapshot
from .core_intelligence_models import (
    StrategicDecision,
    DecisionEvaluationResult,
)
from .core_intelligence_engine import PitchProXCoreIntelligenceEngine
from .core_strategy_modifiers import StrategyModifierPipeline

LOGGER = logging.getLogger("copilot.core_decision_manager")


class CoreDecisionManager:
    """Coordinates Core Intelligence evaluation, stale-decision tracking, and state version lineage."""

    def __init__(
        self,
        call_sid: str,
        lead_type: str = "general",
        engine: Optional[PitchProXCoreIntelligenceEngine] = None,
        modifier_pipeline: Optional[StrategyModifierPipeline] = None,
    ):
        self.call_sid = call_sid
        self.lead_type = lead_type
        self.engine = engine or PitchProXCoreIntelligenceEngine(lead_type=lead_type)
        self.modifier_pipeline = modifier_pipeline or StrategyModifierPipeline()
        self.decision_history: List[StrategicDecision] = []
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
    ) -> DecisionEvaluationResult:
        eval_result = self.engine.evaluate(
            snapshot=snapshot,
            turn_speaker=turn_speaker,
            turn_text=turn_text,
            turn_timestamp_ms=turn_timestamp_ms,
        )

        final_decision = self.modifier_pipeline.apply_modifiers(
            decision=eval_result.decision,
            playbook=playbook,
            calibration=calibration,
        )

        eval_result.decision = final_decision
        self.latest_decision = final_decision
        self.latest_evaluation_result = eval_result
        self.decision_history.append(final_decision)

        return eval_result

    def is_decision_stale(
        self,
        decision: StrategicDecision,
        current_snapshot: ConversationStateSnapshot,
    ) -> bool:
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
        dec_unresolved = set(decision.what_to_protect)  # or context active objections
        if self.latest_evaluation_result:
            dec_unresolved = set(self.latest_evaluation_result.context.active_objections)
        if curr_unresolved != dec_unresolved:
            return True

        return True
