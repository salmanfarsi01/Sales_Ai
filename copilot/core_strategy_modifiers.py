from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .core_intelligence_models import StrategicAction, StrategicDecision, RequiredFactScope

LOGGER = logging.getLogger("copilot.core_strategy_modifiers")


class StrategyModifierPipeline:
    """Applies Playbook methodology weights, Calibration delivery constraints, and conditional retrieval."""

    def apply_modifiers(
        self,
        decision: StrategicDecision,
        playbook: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
    ) -> StrategicDecision:
        modified = decision.model_copy(deep=True)

        if playbook:
            self._apply_playbook(modified, playbook)

        if calibration:
            self._apply_calibration(modified, calibration)

        self._evaluate_retrieval_need(modified)
        return modified

    def _apply_playbook(self, decision: StrategicDecision, playbook: Dict[str, Any]) -> None:
        methodology = playbook.get("methodology", "").lower()
        decision.playbook_influence = {
            "playbook_id": playbook.get("id", "default"),
            "methodology": methodology,
        }

        # Playbook methodology influence on secondary action or styling without overriding Core primary action
        if "challenger" in methodology and decision.primary_action in (StrategicAction.REFRAME, StrategicAction.EDUCATE):
            if decision.secondary_action is None:
                decision.secondary_action = StrategicAction.CHALLENGE
                decision.reason_codes.append("PLAYBOOK_CHALLENGER_EMPHASIS")

        preferred_topics = playbook.get("required_topics", [])
        if preferred_topics:
            decision.what_to_protect.extend(preferred_topics)

    def _apply_calibration(self, decision: StrategicDecision, calibration: Dict[str, Any]) -> None:
        target_wpm = calibration.get("target_wpm", 130)
        brevity_mode = calibration.get("brevity_mode", "standard")
        max_words = decision.max_prompt_words

        if brevity_mode == "concise":
            max_words = min(max_words, 16)
        elif brevity_mode == "detailed":
            max_words = min(max_words + 8, 36)

        decision.max_prompt_words = max_words
        decision.calibration_influence = {
            "brevity_mode": brevity_mode,
            "target_wpm": target_wpm,
            "max_prompt_words": max_words,
        }

    def _evaluate_retrieval_need(self, decision: StrategicDecision) -> None:
        # Knowledge retrieval is triggered conditionally only if specific factual evidence is needed (Spec 11 §8)
        if decision.primary_action == StrategicAction.QUANTIFY:
            decision.retrieval_needed = True
            decision.required_facts = [
                RequiredFactScope(
                    topic="market_comps",
                    required_evidence_type="market_comps",
                    verification_required=True,
                    min_confidence=0.85,
                ),
                RequiredFactScope(
                    topic="commission_roi_calculation",
                    required_evidence_type="roi_calculator",
                    verification_required=True,
                    min_confidence=0.80,
                ),
            ]
        elif decision.primary_action == StrategicAction.SOCIAL_PROOF:
            decision.retrieval_needed = True
            decision.required_facts = [
                RequiredFactScope(
                    topic="verified_client_case_study",
                    required_evidence_type="case_study",
                    verification_required=True,
                    min_confidence=0.90,
                ),
            ]
        elif decision.primary_action == StrategicAction.EDUCATE and "market" in decision.strategic_objective.lower():
            decision.retrieval_needed = True
            decision.required_facts = [
                RequiredFactScope(
                    topic="local_market_inventory_and_pricing_trends",
                    required_evidence_type="market_trends",
                    verification_required=True,
                    min_confidence=0.85,
                ),
            ]
        else:
            decision.retrieval_needed = False
            decision.required_facts = []
