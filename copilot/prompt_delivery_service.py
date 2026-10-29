from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional
from pydantic import BaseModel, Field

from .core_intelligence_models import StrategicDecision
from .conversation_state_models import ConversationStateSnapshot
from .conditional_retrieval import ConditionalRetrievalEngine, RetrievalExecutionReport
from .llm_response_gateway import LLMResponseGateway, Prompt
from .core_decision_manager import CoreDecisionManager
from .core_strategy_modifiers import StrategyModifierPipeline

LOGGER = logging.getLogger("copilot.prompt_delivery_service")


class PromptDeliveryResult(BaseModel):
    """Result of prompt assembly, generation, and Tier 2 teleprompter display gate."""
    prompt: Prompt
    retrieval_report: RetrievalExecutionReport
    displayed: bool
    cancellation_reason: Optional[str] = None
    immediate_directive: Optional[str] = None
    latency_metrics: Optional[Dict[str, Any]] = None


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
        self.modifier_pipeline = StrategyModifierPipeline()

    def process_and_deliver(
        self,
        decision: StrategicDecision,
        current_snapshot: Optional[ConversationStateSnapshot] = None,
        dispatch_snapshot: Optional[ConversationStateSnapshot] = None,
        display_snapshot: Optional[ConversationStateSnapshot] = None,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        tenant_id: Optional[str] = None,
        playbook: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
    ) -> PromptDeliveryResult:
        # Resolve snapshots for pre-dispatch (Tier 1) and pre-display (Tier 2)
        snap_dispatch = dispatch_snapshot or current_snapshot
        if snap_dispatch is None:
            raise ValueError("Either current_snapshot or dispatch_snapshot must be provided.")
        snap_display = display_snapshot or snap_dispatch

        working_decision = decision
        if playbook or calibration:
            working_decision = self.modifier_pipeline.apply_modifiers(
                decision=decision,
                playbook=playbook,
                calibration=calibration,
            )

        # Checkpoint 1 (Pre-dispatch Gate): Ensure decision is not already stale
        if self.decision_manager.is_decision_stale(working_decision, snap_dispatch):
            cancelled_prompt = Prompt(
                decision_id=working_decision.decision_id,
                source_state_version=working_decision.source_state_version,
                text="",
                strategic_action=working_decision.primary_action,
                strategic_objective=working_decision.strategic_objective,
                status="cancelled",
                cancellation_reason="PRE_DISPATCH_STALENESS_DETECTED",
            )
            return PromptDeliveryResult(
                prompt=cancelled_prompt,
                retrieval_report=RetrievalExecutionReport(
                    retrieval_needed=False,
                    decision_id=working_decision.decision_id,
                    source_state_version=working_decision.source_state_version,
                ),
                displayed=False,
                cancellation_reason="PRE_DISPATCH_STALENESS_DETECTED",
            )

        # Conditional Knowledge Retrieval (only when decision.retrieval_needed == True)
        retrieval_report = self.retrieval_engine.retrieve_for_decision(
            decision=working_decision,
            tenant_id=tenant_id,
        )

        # LLM Generation from structured bounded context contract (Spec 11 §7)
        prompt = self.llm_gateway.generate_prompt(
            decision=working_decision,
            snapshot=snap_dispatch,
            facts=retrieval_report.facts,
            recent_turns=recent_turns,
            playbook=playbook,
            calibration=calibration,
            use_llm=use_llm,
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
                immediate_directive=prompt.immediate_directive,
                latency_metrics=prompt.latency_metrics,
            )

        prompt.status = "displayed"
        prompt.displayed_at = datetime.now(timezone.utc).isoformat()
        return PromptDeliveryResult(
            prompt=prompt,
            retrieval_report=retrieval_report,
            displayed=True,
            cancellation_reason=None,
            immediate_directive=prompt.immediate_directive,
            latency_metrics=prompt.latency_metrics,
        )

    def process_and_deliver_stream_sync(
        self,
        decision: StrategicDecision,
        current_snapshot: Optional[ConversationStateSnapshot] = None,
        dispatch_snapshot: Optional[ConversationStateSnapshot] = None,
        display_snapshot: Optional[ConversationStateSnapshot] = None,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        tenant_id: Optional[str] = None,
        playbook: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        state_version_checker: Optional[Callable[[], int]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Synchronously coordinates Two-Track teleprompter delivery with in-flight cancellation (clarification.md §6)."""
        snap_dispatch = dispatch_snapshot or current_snapshot
        if snap_dispatch is None:
            raise ValueError("Either current_snapshot or dispatch_snapshot must be provided.")
        snap_display = display_snapshot or snap_dispatch

        working_decision = decision
        if playbook or calibration:
            working_decision = self.modifier_pipeline.apply_modifiers(
                decision=decision,
                playbook=playbook,
                calibration=calibration,
            )

        # Checkpoint 1: Pre-dispatch Gate
        if self.decision_manager.is_decision_stale(working_decision, snap_dispatch):
            cancelled_prompt = Prompt(
                decision_id=working_decision.decision_id,
                source_state_version=working_decision.source_state_version,
                text="",
                strategic_action=working_decision.primary_action,
                strategic_objective=working_decision.strategic_objective,
                status="cancelled",
                cancellation_reason="PRE_DISPATCH_STALENESS_DETECTED",
            )
            yield {
                "type": "cancelled",
                "prompt": cancelled_prompt,
                "cancellation_reason": "PRE_DISPATCH_STALENESS_DETECTED",
                "displayed": False,
            }
            return

        # Conditional Retrieval
        retrieval_report = self.retrieval_engine.retrieve_for_decision(
            decision=working_decision,
            tenant_id=tenant_id,
        )

        # Stream generation through LLM Gateway (sync stream)
        for event in self.llm_gateway.generate_prompt_stream_sync(
            decision=working_decision,
            snapshot=snap_dispatch,
            facts=retrieval_report.facts,
            recent_turns=recent_turns,
            retrieval_result=retrieval_report.result if hasattr(retrieval_report, "result") else None,
            playbook=playbook,
            calibration=calibration,
            use_llm=use_llm,
            is_cancelled_callback=is_cancelled_callback,
            state_version_checker=state_version_checker,
        ):
            ev_type = event.get("type")
            if ev_type in ("track1_directive", "track2_token", "prompt_suppressed"):
                yield event
            elif ev_type == "cancelled":
                yield event
                return
            elif ev_type in ("prompt_complete", "guardrail_fallback"):
                prompt_obj: Prompt = event["prompt"]
                # Checkpoint 2: Pre-display Tier 2 validation
                is_valid, cancel_reason = self.decision_manager.validate_for_teleprompter_display(
                    source_state_version=prompt_obj.source_state_version,
                    current_snapshot=snap_display,
                )
                if not is_valid:
                    prompt_obj.status = "cancelled"
                    prompt_obj.cancellation_reason = cancel_reason
                    yield {
                        "type": "cancelled",
                        "prompt": prompt_obj,
                        "retrieval_report": retrieval_report,
                        "cancellation_reason": cancel_reason,
                        "displayed": False,
                    }
                else:
                    prompt_obj.status = "displayed"
                    prompt_obj.displayed_at = datetime.now(timezone.utc).isoformat()
                    yield {
                        "type": "displayed",
                        "prompt": prompt_obj,
                        "retrieval_report": retrieval_report,
                        "displayed": True,
                        "immediate_directive": prompt_obj.immediate_directive,
                        "latency_metrics": prompt_obj.latency_metrics,
                    }

    async def process_and_deliver_stream(
        self,
        decision: StrategicDecision,
        current_snapshot: Optional[ConversationStateSnapshot] = None,
        dispatch_snapshot: Optional[ConversationStateSnapshot] = None,
        display_snapshot: Optional[ConversationStateSnapshot] = None,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        tenant_id: Optional[str] = None,
        playbook: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        state_version_checker: Optional[Callable[[], int]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously coordinates Two-Track teleprompter delivery with in-flight cancellation (clarification.md §6)."""
        for event in self.process_and_deliver_stream_sync(
            decision=decision,
            current_snapshot=current_snapshot,
            dispatch_snapshot=dispatch_snapshot,
            display_snapshot=display_snapshot,
            recent_turns=recent_turns,
            tenant_id=tenant_id,
            playbook=playbook,
            calibration=calibration,
            use_llm=use_llm,
            is_cancelled_callback=is_cancelled_callback,
            state_version_checker=state_version_checker,
        ):
            yield event
