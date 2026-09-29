from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .core_intelligence_models import StrategicDecision
from .conversation_state_models import ConversationStateSnapshot
from .conditional_retrieval import ConditionalRetrievalEngine, RetrievalExecutionReport
from .llm_response_gateway import LLMResponseGateway, Prompt
from .core_decision_manager import CoreDecisionManager

LOGGER = logging.getLogger("copilot.prompt_delivery_service")


class PromptDeliveryResult(BaseModel):
    """Result of prompt assembly, generation, and Tier 2 teleprompter display gate."""
    prompt: Prompt
    retrieval_report: RetrievalExecutionReport
    displayed: bool
    cancellation_reason: Optional[str] = None


class PromptDeliveryService:
    """Coordinates conditional retrieval, LLM prompt generation, and two-tier staleness validation."""

    def __init__(
        self,
        decision_manager: Optional[CoreDecisionManager] = None,
        retrieval_engine: Optional[ConditionalRetrievalEngine] = None,
        llm_gateway: Optional[LLMResponseGateway] = None,
        call_sid: Optional[str] = None,
    ):
        if decision_manager is not None:
            self.decision_manager = decision_manager
        elif call_sid:
            self.decision_manager = CoreDecisionManager(call_sid=call_sid)
        else:
            raise ValueError(
                "Either decision_manager or call_sid must be explicitly provided to PromptDeliveryService to prevent cross-call state leakage."
            )
        self.retrieval_engine = retrieval_engine or ConditionalRetrievalEngine()
        self.llm_gateway = llm_gateway or LLMResponseGateway()

    def process_and_deliver(
        self,
        decision: StrategicDecision,
        current_snapshot: Optional[ConversationStateSnapshot] = None,
        dispatch_snapshot: Optional[ConversationStateSnapshot] = None,
        display_snapshot: Optional[ConversationStateSnapshot] = None,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        tenant_id: Optional[str] = None,
    ) -> PromptDeliveryResult:
        # Resolve snapshots for pre-dispatch (Tier 1) and pre-display (Tier 2)
        snap_dispatch = dispatch_snapshot or current_snapshot
        if snap_dispatch is None:
            raise ValueError("Either current_snapshot or dispatch_snapshot must be provided.")
        snap_display = display_snapshot or snap_dispatch

        # Checkpoint 1 (Pre-dispatch Gate): Ensure decision is not already stale
        if self.decision_manager.is_decision_stale(decision, snap_dispatch):
            cancelled_prompt = Prompt(
                decision_id=decision.decision_id,
                source_state_version=decision.source_state_version,
                text="",
                strategic_action=decision.primary_action,
                strategic_objective=decision.strategic_objective,
                status="cancelled",
                cancellation_reason="PRE_DISPATCH_STALENESS_DETECTED",
            )
            return PromptDeliveryResult(
                prompt=cancelled_prompt,
                retrieval_report=RetrievalExecutionReport(
                    retrieval_needed=False,
                    decision_id=decision.decision_id,
                    source_state_version=decision.source_state_version,
                ),
                displayed=False,
                cancellation_reason="PRE_DISPATCH_STALENESS_DETECTED",
            )

        # Conditional Knowledge Retrieval (only when decision.retrieval_needed == True)
        retrieval_report = self.retrieval_engine.retrieve_for_decision(
            decision=decision,
            tenant_id=tenant_id,
        )

        # LLM Generation from structured bounded context contract (Spec 11 §7)
        prompt = self.llm_gateway.generate_prompt(
            decision=decision,
            snapshot=snap_dispatch,
            facts=retrieval_report.facts,
            recent_turns=recent_turns,
        )

        # Checkpoint 2 (Pre-display Tier 2 Gate): Validate immediately before teleprompter display
        is_valid_for_display, cancel_reason = self.decision_manager.validate_for_teleprompter_display(
            source_state_version=prompt.source_state_version,
            current_snapshot=snap_display,
        )

        if not is_valid_for_display:
            prompt.status = "cancelled"
            prompt.cancellation_reason = cancel_reason
            LOGGER.info("Prompt %s cancelled before teleprompter display: %s", prompt.prompt_id, cancel_reason)
            return PromptDeliveryResult(
                prompt=prompt,
                retrieval_report=retrieval_report,
                displayed=False,
                cancellation_reason=cancel_reason,
            )

        prompt.status = "displayed"
        prompt.displayed_at = datetime.now(timezone.utc).isoformat()
        return PromptDeliveryResult(
            prompt=prompt,
            retrieval_report=retrieval_report,
            displayed=True,
            cancellation_reason=None,
        )
