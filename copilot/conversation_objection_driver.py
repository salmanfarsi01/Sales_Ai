from __future__ import annotations

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional

from .conversation_state_models import ObjectionDriverLayer

LOGGER = logging.getLogger("copilot.conversation_objection_driver")

# -----------------------------------------------------------------------------
# Closed Driver Taxonomy (Client Feedback No. 5)
# Explicit, bounded set of underlying drivers and default strategic targets
# defined per canonical objection category.
# -----------------------------------------------------------------------------
CLOSED_DRIVER_TAXONOMY: Dict[str, Dict[str, Dict[str, str]]] = {
    "commission_fee": {
        "price_resistance": {
            "description": "Pure numerical or fee sensitivity; prospect finds the absolute fee or percentage excessive.",
            "strategic_target": "justify_fee_structure_and_net_proceeds_roi",
        },
        "perceived_value_deficit": {
            "description": "Prospect doubts whether marketing effort, agent services, or execution justifies the cost.",
            "strategic_target": "demonstrate_marketing_execution_and_direct_value_delivery",
        },
        "previous_agent_outcome": {
            "description": "Dissatisfaction anchored to a prior agent experience where full fee was paid with disappointing results.",
            "strategic_target": "diagnose_prior_service_failures_and_differentiate_operational_process",
        },
        "market_comparison": {
            "description": "Comparing the commission rate against discount brokerages, flat-fee options, or DIY/FSBO.",
            "strategic_target": "contrast_full_service_risk_mitigation_with_discount_broker_limitations",
        },
    },
    "market_timing": {
        "interest_rate_anxiety": {
            "description": "Hesitation driven by mortgage rates or macroeconomic rate environment.",
            "strategic_target": "frame_rate_fluctuations_and_buyer_pool_demand_dynamics",
        },
        "seasonal_expectation": {
            "description": "Belief that waiting for spring/summer will yield a higher price or faster sale.",
            "strategic_target": "analyze_inventory_scarcity_advantages_vs_peak_competition",
        },
        "personal_life_transition": {
            "description": "Timing dependent on life transitions (job relocation, retirement, school year).",
            "strategic_target": "coordinate_contingent_timelines_and_flexible_possession",
        },
        "fear_of_market_drop": {
            "description": "Anxiety that home values are falling or a market crash is imminent.",
            "strategic_target": "provide_hyperlocal_sold_comps_and_absorption_data",
        },
    },
    "broker_representation": {
        "existing_exclusive_contract": {
            "description": "Active contractual obligation with another broker or agent.",
            "strategic_target": "verify_listing_agreement_expiration_and_maintain_compliance",
        },
        "friend_family_loyalty": {
            "description": "Social pressure or loyalty to a friend/relative in the real estate business.",
            "strategic_target": "separate_personal_relationship_from_commercial_transaction_fiduciary",
        },
        "prior_bad_experience": {
            "description": "General skepticism or negative impression of real estate agents from past transactions.",
            "strategic_target": "establish_accountability_metrics_and_written_performance_commitments",
        },
    },
    "pricing_value": {
        "financial_net_requirement": {
            "description": "Need to net a specific dollar amount to pay off debt or buy a replacement property.",
            "strategic_target": "reverse_engineer_target_net_proceeds_and_pricing_corridor",
        },
        "emotional_attachment": {
            "description": "Overvaluing property due to sentimental history or unrecouped custom improvements.",
            "strategic_target": "acknowledge_property_care_while_anchoring_to_appraisal_realities",
        },
        "skepticism_of_cma": {
            "description": "Suspecting the agent is underpricing for a quick commission at seller expense.",
            "strategic_target": "present_transparent_bracketed_absorption_and_buyer_demand_analytics",
        },
    },
    "general_hesitation": {
        "process_overwhelm": {
            "description": "Feeling daunted by the logistical burden of prep, decluttering, showings, or moving.",
            "strategic_target": "decompose_process_into_low_friction_bite_sized_milestones",
        },
        "information_deficit": {
            "description": "Uncertainty about process steps, options, or local market trends.",
            "strategic_target": "provide_educational_clarity_without_transactional_pressure",
        },
        "unexpressed_underlying_concern": {
            "description": "Polite hesitation concealing an unstated core constraint.",
            "strategic_target": "deploy_gentle_diagnostic_inquiry_to_surface_root_constraint",
        },
    },
}


class ObjectionDriverClassifier:
    """Classifies the second-layer underlying driver of an objection using LLM or deterministic heuristics."""

    def __init__(
        self,
        groq_client: Optional[Any] = None,
        timeout_seconds: float = 2.0,
        model: str = "qwen/qwen3.8-27b",
    ):
        self.groq_client = groq_client
        self.timeout_seconds = timeout_seconds
        self.model = os.getenv("GROQ_FAST_MODEL", model)

    def classify_driver(
        self,
        canonical_category: str,
        utterance_text: str,
        context_history: Optional[List[str]] = None,
    ) -> ObjectionDriverLayer:
        """Determines the underlying driver layer for a given objection instance."""
        category = canonical_category.lower().strip()
        if category not in CLOSED_DRIVER_TAXONOMY:
            category = "general_hesitation"

        api_key = os.getenv("GROQ_API_KEY")
        if api_key and not api_key.startswith("mock_"):
            try:
                return self._classify_via_llm(category, utterance_text, context_history, api_key)
            except Exception as exc:
                LOGGER.warning("LLM objection driver classification failed (%s), falling back to heuristic", exc)

        return self._classify_via_heuristic(category, utterance_text, context_history)

    def _classify_via_llm(
        self,
        category: str,
        utterance_text: str,
        context_history: Optional[List[str]],
        api_key: str,
    ) -> ObjectionDriverLayer:
        """Evaluates underlying driver using LLM classification constrained to the closed taxonomy."""
        available_drivers = CLOSED_DRIVER_TAXONOMY.get(category, {})
        driver_descriptions = "\n".join(
            f"- {driver_name}: {info['description']} (Target: {info['strategic_target']})"
            for driver_name, info in available_drivers.items()
        )

        context_str = ""
        if context_history:
            context_str = "Recent context:\n" + "\n".join(f"- {c}" for c in context_history[-3:]) + "\n\n"

        prompt = (
            f"You are an expert sales conversational psychologist analyzing objection root causes.\n\n"
            f"Surface Objection Category: '{category}'\n\n"
            f"{context_str}"
            f"Prospect Statement:\n"
            f"\"{utterance_text}\"\n\n"
            f"Task: Classify the underlying driver from this EXACT closed set of options for '{category}':\n"
            f"{driver_descriptions}\n\n"
            f"Respond with raw JSON containing:\n"
            f"- underlying_driver: string (MUST be one of the exact driver names listed above)\n"
            f"- origin_context: string or null (e.g. 'referenced prior agent experience', 'fee calculation shock', etc.)\n"
            f"- supporting_evidence: array of strings (quoted phrases from utterance justifying this classification)\n"
            f"- strategic_target: string (the strategic objective the salesperson's response must address)\n"
            f"- confidence: float between 0.0 and 1.0\n\n"
            f"Output ONLY raw JSON."
        )

        if self.groq_client is not None:
            client = self.groq_client
        else:
            import groq
            client = groq.Groq(api_key=api_key, timeout=self.timeout_seconds, max_retries=0)

        groq_model = self.model
        if "openai/gpt-oss" in groq_model or "llama" in groq_model:
            groq_model = "qwen/qwen3.8-27b"

        completion = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(completion.choices[0].message.content.strip())

        driver = parsed.get("underlying_driver", "")
        if driver not in available_drivers:
            driver = list(available_drivers.keys())[0]

        default_target = available_drivers[driver]["strategic_target"]
        target = parsed.get("strategic_target") or default_target
        origin = parsed.get("origin_context")
        evidence = parsed.get("supporting_evidence", [utterance_text.strip()])
        conf = float(parsed.get("confidence", 0.90))

        return ObjectionDriverLayer(
            surface_objection=category,
            underlying_driver=driver,
            origin_context=origin,
            supporting_evidence=evidence if isinstance(evidence, list) else [str(evidence)],
            confidence=conf,
            strategic_target=target,
            classification_source="llm",
        )

    def _classify_via_heuristic(
        self,
        category: str,
        utterance_text: str,
        context_history: Optional[List[str]] = None,
    ) -> ObjectionDriverLayer:
        """Deterministic semantic fallback providing instant classification and high offline test fidelity."""
        text_lower = utterance_text.lower().strip()
        available_drivers = CLOSED_DRIVER_TAXONOMY.get(category, CLOSED_DRIVER_TAXONOMY["general_hesitation"])

        # ---------------------------------------------------------------------
        # 1. Commission Fee Driver Discrimination
        # ---------------------------------------------------------------------
        if category == "commission_fee":
            # Check for Previous Agent Outcome
            prev_agent_patterns = [
                r"\b(?:last|prior|previous|other)\s+(?:agent|realtor|broker|time|experience|listing)\b",
                r"\b(?:what\s+we\s+got|what\s+i\s+got|got\s+for\s+it)\b",
                r"\bpaid\s+(?:an?\s+agent|thousands|full\s+commission)\s+(?:before|last|and\s+nothing)\b",
                r"\bdid\s+nothing\b",
                r"\bjust\s+sat\s+there\b",
            ]
            if any(re.search(p, text_lower) for p in prev_agent_patterns):
                return ObjectionDriverLayer(
                    surface_objection="commission_fee",
                    underlying_driver="previous_agent_outcome",
                    origin_context="referenced prior agent experience",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["previous_agent_outcome"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )

            # Check for Perceived Value Deficit
            value_deficit_patterns = [
                r"\b(?:for\s+what\s+you\s+do|putting\s+a\s+sign|open\s+a\s+door|putting\s+it\s+on\s+the\s+mls)\b",
                r"\b(?:what\s+do\s+you\s+actually\s+do|why\s+does\s+it\s+cost|worth\s+(?:that\s+much|the\s+money))\b",
                r"\b(?:doesn't|don't)\s+see\s+the\s+value\b",
                r"\bdeserves?\s+(?:that|so)\s+much\b",
            ]
            if any(re.search(p, text_lower) for p in value_deficit_patterns):
                return ObjectionDriverLayer(
                    surface_objection="commission_fee",
                    underlying_driver="perceived_value_deficit",
                    origin_context="skepticism of service delivery vs cost",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["perceived_value_deficit"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )

            # Check for Market / Competitor Comparison
            comparison_patterns = [
                r"\b(?:other|another)\s+(?:broker|agent|company|realtor)\s+(?:charges?|said|offered|will\s+do)\b",
                r"\b(?:redfin|zillow|discount|flat\s+fee|1\s*%|2\s*%)\b",
                r"\bcheaper\s+somewhere\s+else\b",
            ]
            if any(re.search(p, text_lower) for p in comparison_patterns):
                return ObjectionDriverLayer(
                    surface_objection="commission_fee",
                    underlying_driver="market_comparison",
                    origin_context="comparison against discount / competing broker models",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["market_comparison"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )

            # Default: Pure Price Resistance (Fallback baseline)
            return ObjectionDriverLayer(
                surface_objection="commission_fee",
                underlying_driver="price_resistance",
                origin_context="direct fee resistance",
                supporting_evidence=[utterance_text.strip()],
                confidence=0.50,
                strategic_target=available_drivers["price_resistance"]["strategic_target"],
                classification_source="heuristic_default",
            )

        # ---------------------------------------------------------------------
        # 2. Market Timing Driver Discrimination
        # ---------------------------------------------------------------------
        if category == "market_timing":
            # Interest Rate Anxiety
            if any(re.search(p, text_lower) for p in [r"\b(?:interest\s+rates?|rates|mortgage|fed)\b"]):
                return ObjectionDriverLayer(
                    surface_objection="market_timing",
                    underlying_driver="interest_rate_anxiety",
                    origin_context="macroeconomic interest rate anxiety",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["interest_rate_anxiety"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )

            # Seasonal Expectation
            if any(re.search(p, text_lower) for p in [r"\b(?:spring|summer|fall|winter|holidays)\b"]):
                return ObjectionDriverLayer(
                    surface_objection="market_timing",
                    underlying_driver="seasonal_expectation",
                    origin_context="seasonal selling window expectation",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["seasonal_expectation"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )

            # Fear of Market Drop
            if any(re.search(p, text_lower) for p in [r"\b(?:crash|drop|dropping|falling|plunge|bubble)\b"]):
                return ObjectionDriverLayer(
                    surface_objection="market_timing",
                    underlying_driver="fear_of_market_drop",
                    origin_context="anxiety over market devaluation",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["fear_of_market_drop"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )

            # Default: Personal life transition (Fallback baseline)
            return ObjectionDriverLayer(
                surface_objection="market_timing",
                underlying_driver="personal_life_transition",
                origin_context="personal milestone or life timing",
                supporting_evidence=[utterance_text.strip()],
                confidence=0.50,
                strategic_target=available_drivers["personal_life_transition"]["strategic_target"],
                classification_source="heuristic_default",
            )

        # ---------------------------------------------------------------------
        # 3. Broker Representation Driver Discrimination
        # ---------------------------------------------------------------------
        if category == "broker_representation":
            if any(re.search(p, text_lower) for p in [r"\b(?:friend|cousin|relative|brother|sister|uncle|nephew|family)\b"]):
                return ObjectionDriverLayer(
                    surface_objection="broker_representation",
                    underlying_driver="friend_family_loyalty",
                    origin_context="social or familial relationship loyalty",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["friend_family_loyalty"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )
            if any(re.search(p, text_lower) for p in [r"\b(?:signed|agreement|contract|exclusive)\b"]):
                return ObjectionDriverLayer(
                    surface_objection="broker_representation",
                    underlying_driver="existing_exclusive_contract",
                    origin_context="existing legal representation agreement",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["existing_exclusive_contract"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )
            # Default: Prior bad experience (Fallback baseline)
            return ObjectionDriverLayer(
                surface_objection="broker_representation",
                underlying_driver="prior_bad_experience",
                origin_context="general representation skepticism",
                supporting_evidence=[utterance_text.strip()],
                confidence=0.50,
                strategic_target=available_drivers["prior_bad_experience"]["strategic_target"],
                classification_source="heuristic_default",
            )

        # ---------------------------------------------------------------------
        # 4. Pricing / Value Driver Discrimination
        # ---------------------------------------------------------------------
        if category == "pricing_value":
            if any(re.search(p, text_lower) for p in [r"\b(?:net|need\s+to\s+walk\s+away|mortgage\s+payoff|buy\s+our\s+next)\b"]):
                return ObjectionDriverLayer(
                    surface_objection="pricing_value",
                    underlying_driver="financial_net_requirement",
                    origin_context="strict financial net threshold",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["financial_net_requirement"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )
            if any(re.search(p, text_lower) for p in [r"\b(?:put\s+so\s+much\s+into|renovated|sweat\s+equity|special|love\s+this)\b"]):
                return ObjectionDriverLayer(
                    surface_objection="pricing_value",
                    underlying_driver="emotional_attachment",
                    origin_context="emotional and custom upgrade attachment",
                    supporting_evidence=[utterance_text.strip()],
                    confidence=0.70,
                    strategic_target=available_drivers["emotional_attachment"]["strategic_target"],
                    classification_source="heuristic_pattern",
                )
            # Default: Skepticism of CMA (Fallback baseline)
            return ObjectionDriverLayer(
                surface_objection="pricing_value",
                underlying_driver="skepticism_of_cma",
                origin_context="skepticism of valuation methodology",
                supporting_evidence=[utterance_text.strip()],
                confidence=0.50,
                strategic_target=available_drivers["skepticism_of_cma"]["strategic_target"],
                classification_source="heuristic_default",
            )

        # ---------------------------------------------------------------------
        # 5. General Hesitation Driver Discrimination (Default fallback)
        # ---------------------------------------------------------------------
        if any(re.search(p, text_lower) for p in [r"\b(?:overwhelm|too\s+much|stress|clutter|clean|pack|pack\s+up)\b"]):
            driver = "process_overwhelm"
            context = "logistical burden of moving prep"
            conf = 0.70
            src = "heuristic_pattern"
        elif any(re.search(p, text_lower) for p in [r"\b(?:don't\s+know|not\s+sure\s+how|unclear|questions?)\b"]):
            driver = "information_deficit"
            context = "process and market uncertainty"
            conf = 0.70
            src = "heuristic_pattern"
        else:
            driver = "unexpressed_underlying_concern"
            context = "general unexpressed hesitation"
            conf = 0.50
            src = "heuristic_default"

        return ObjectionDriverLayer(
            surface_objection=category,
            underlying_driver=driver,
            origin_context=context,
            supporting_evidence=[utterance_text.strip()],
            confidence=conf,
            strategic_target=available_drivers.get(driver, {}).get("strategic_target", "deploy_gentle_diagnostic_inquiry_to_surface_root_constraint"),
            classification_source=src,
        )
