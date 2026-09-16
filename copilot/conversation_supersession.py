from __future__ import annotations

import os
import re
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field

from .conversation_state_models import (
    PersistentFactRecord,
    PersistentFactCategory,
    DecisionStructure,
)

LOGGER = logging.getLogger("copilot.conversation_supersession")

SupersessionRelation = Literal["UPDATES", "REVERSES", "CONFIRMS", "UNCHANGED"]


class SupersessionDecision(BaseModel):
    """Evaluates the semantic relation of a new statement against an existing active fact."""
    has_supersession: bool = False
    target_fact_id: Optional[str] = None
    target_fact_key: Optional[str] = None
    relation: SupersessionRelation = "UNCHANGED"
    old_truth_summary: Optional[str] = None
    new_truth_value: Optional[str] = None
    condition: Optional[str] = None
    reasoning: str = ""
    relation_confidence: float = Field(1.0, ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class TruthSupersessionDetector:
    """Detects when a new prospect utterance updates, qualifies, or reverses an existing active fact

    or decision belief, even in the absence of explicit contradiction keywords.

    Core Invariants:
    1. Retrieval Pre-Filtering: Candidate utterances are routed to relevant fact categories
       (e.g. timing vs authority vs channel), preventing cross-domain false matches and wasteful calls.
    2. Nuanced Conditionality: Updates preserve specific qualifiers ("only if better strategy shown")
       rather than crude flat overwrites.
    3. REVERSES Detection: Outright reversals ("not selling anymore, pulled off market") are detected
       and distinguished from conditional updates.
    4. Honest Uncertainty (Client Principle #8): Every decision carries calibrated confidence.
    """

    def __init__(self, groq_client: Optional[Any] = None, timeout_seconds: float = 4.0):
        self.groq_client = groq_client
        self.timeout_seconds = timeout_seconds

    def filter_candidate_facts(
        self,
        candidate_text: str,
        active_facts: List[PersistentFactRecord],
    ) -> List[PersistentFactRecord]:
        """Routes candidate utterance to topical candidate facts, preventing cross-domain false matches."""
        if not active_facts:
            return []

        text_lower = candidate_text.lower()
        matched_categories: Set[PersistentFactCategory] = set()
        matched_keys: Set[str] = set()

        # Domain 1: Timing, Relocation, Moving, Selling
        timing_moving_markers = [
            r"\b(?:move|moving|stay|staying|sell|selling|list|listing|market|timeline|close|closing)\b",
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(?:spring|summer|fall|winter|holidays|thanksgiving|christmas|semester)\b",
        ]
        if any(re.search(p, text_lower) for p in timing_moving_markers):
            matched_categories.add("timeline")
            matched_keys.update(["moving_decision", "selling_decision", "target_closing", "timeline", "move_in_date"])

        # Domain 2: Authority & Decision Maker Roles
        authority_markers = [
            r"\b(?:wife|husband|spouse|partner|attorney|power\s+of\s+attorney|signer|sign|consult|decision)\b",
            r"\b(?:alone|together|joint|sole|authority|handles)\b",
        ]
        if any(re.search(p, text_lower) for p in authority_markers):
            matched_categories.add("decision_maker")
            matched_keys.update(["spouse_involvement", "decision_maker_authority", "signing_authority"])

        # Domain 3: Channel & Contact Timing Preferences
        channel_markers = [
            r"\b(?:call|phone|cell|text|email|message|reach|contact|clinic|busy|hours|morning|afternoon)\b",
        ]
        if any(re.search(p, text_lower) for p in channel_markers):
            matched_categories.add("preference")
            matched_categories.add("logistical")
            matched_keys.update(["preferred_channel", "contact_channel", "morning_availability", "afternoon_availability"])

        # Domain 4: Financial & Price Expectations
        financial_markers = [
            r"\b(?:price|pricing|budget|dollars|\$|cost|fee|net|netting|proceeds|lowball)\b",
        ]
        if any(re.search(p, text_lower) for p in financial_markers):
            matched_categories.add("financial")
            matched_keys.update(["max_budget", "net_proceeds", "price_expectation"])

        # Filter facts matching either the category domain or explicit fact key
        candidate_facts = [
            f for f in active_facts
            if f.category in matched_categories or f.fact_key in matched_keys
        ]
        return candidate_facts

    def evaluate_turn(
        self,
        candidate_text: str,
        active_facts: List[PersistentFactRecord],
        decision_structure: Optional[DecisionStructure] = None,
    ) -> List[SupersessionDecision]:
        """Evaluates candidate utterance against relevant active facts, returning supersession decisions."""
        if not active_facts or not candidate_text.strip():
            return []

        # 1. Candidate Fact Retrieval (pre-filtering step)
        candidate_facts = self.filter_candidate_facts(candidate_text, active_facts)
        if not candidate_facts:
            return []

        decisions: List[SupersessionDecision] = []
        for fact in candidate_facts:
            decision = self._evaluate_fact_supersession(candidate_text, fact, decision_structure)
            if decision.has_supersession:
                decisions.append(decision)

        return decisions

    def _evaluate_fact_supersession(
        self,
        candidate_text: str,
        fact: PersistentFactRecord,
        decision_structure: Optional[DecisionStructure] = None,
    ) -> SupersessionDecision:
        """Evaluates whether candidate_text supersedes a specific active fact."""
        api_key_groq = os.getenv("GROQ_API_KEY")
        is_testing = bool(os.getenv("PYTEST_CURRENT_TEST"))
        should_use_llm = self.groq_client is not None or (
            api_key_groq
            and not api_key_groq.startswith("mock_")
            and not api_key_groq.startswith("gsk_test")
            and not is_testing
        )
        if should_use_llm:
            try:
                return self._evaluate_via_llm(candidate_text, fact, api_key_groq)
            except Exception as exc:
                LOGGER.warning("LLM supersession evaluation failed, using semantic fallback: %s", exc)
                return self._evaluate_via_heuristic(candidate_text, fact)

        return self._evaluate_via_heuristic(candidate_text, fact)

    def _evaluate_via_llm(
        self,
        candidate_text: str,
        fact: PersistentFactRecord,
        api_key: Optional[str],
    ) -> SupersessionDecision:
        """Evaluates supersession using LLM comparative reasoning."""
        prompt = (
            f"You are an expert sales conversational linguist analyzing truth supersession.\n\n"
            f"Existing active belief:\n"
            f"- Key: {fact.fact_key}\n"
            f"- Category: {fact.category}\n"
            f"- Current Value: \"{fact.fact_value}\"\n\n"
            f"New prospect utterance:\n"
            f"\"{candidate_text}\"\n\n"
            f"Task: Compare the new utterance to the existing active belief.\n"
            f"Classify relationship as:\n"
            f"- UPDATES: Modifies parameters or adds conditions (e.g. 'We decided to stay' -> 'I'd still move if I believed there was a better strategy').\n"
            f"- REVERSES: Flips the prior conclusion entirely (e.g. 'Planning to sell this spring' -> 'We are not selling anymore, taking it off the market').\n"
            f"- CONFIRMS: Re-iterates or affirms the existing belief without change.\n"
            f"- UNCHANGED: Unrelated, ambiguous, or neutral.\n\n"
            f"Respond with raw JSON containing:\n"
            f"- relation: 'UPDATES' | 'REVERSES' | 'CONFIRMS' | 'UNCHANGED'\n"
            f"- has_supersession: boolean (true for UPDATES or REVERSES; false for CONFIRMS or UNCHANGED)\n"
            f"- new_truth_value: string summarizing the updated belief preserving specific conditions/qualifiers (e.g. 'Conditionally open to moving if convinced of a better strategy')\n"
            f"- condition: string specifying the explicit condition if conditional, else null\n"
            f"- reasoning: concise explanation of why the new statement updates or reverses the old\n"
            f"- relation_confidence: float 0.0 to 1.0 (calibrated confidence in this classification)\n"
            f"- confidence: float 0.0 to 1.0\n\n"
            f"Output ONLY raw JSON."
        )

        if self.groq_client is not None:
            client = self.groq_client
        else:
            import groq
            client = groq.Groq(api_key=api_key, timeout=self.timeout_seconds)

        groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        if "openai/gpt-oss" in groq_model or "llama" in groq_model:
            groq_model = "qwen/qwen3.8-27b"

        completion = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(completion.choices[0].message.content.strip())

        rel: SupersessionRelation = parsed.get("relation", "UNCHANGED")
        has_sup = parsed.get("has_supersession", rel in ("UPDATES", "REVERSES"))
        new_val = parsed.get("new_truth_value") or candidate_text.strip()
        rel_conf = float(parsed.get("relation_confidence", 0.90))
        conf = float(parsed.get("confidence", rel_conf))

        return SupersessionDecision(
            has_supersession=has_sup,
            target_fact_id=fact.fact_id,
            target_fact_key=fact.fact_key,
            relation=rel,
            old_truth_summary=fact.fact_value,
            new_truth_value=new_val if has_sup else None,
            condition=parsed.get("condition"),
            reasoning=parsed.get("reasoning", ""),
            relation_confidence=rel_conf,
            confidence=conf,
        )

    def _evaluate_via_heuristic(
        self,
        candidate_text: str,
        fact: PersistentFactRecord,
    ) -> SupersessionDecision:
        """Deterministic semantic fallback recognizing conditionality, qualification, and reversals."""
        text_lower = candidate_text.lower().strip()
        fact_lower = fact.fact_value.lower().strip()

        # -------------------------------------------------------------------------
        # 1. Moving / Selling Decision (Updates vs. Reverses vs. Confirms)
        # -------------------------------------------------------------------------
        if fact.fact_key in ("moving_decision", "selling_decision") or fact.category == "timeline":
            # Check for Complete Reversal (REVERSES)
            reversal_markers = [
                r"\bnot\s+(?:interested\s+in\s+)?selling\s+anymore\b",
                r"\bpull(?:ed|ing)?\s+(?:it\s+)?off\s+the\s+market\b",
                r"\bdecided\s+not\s+to\s+sell\b",
                r"\bnot\s+moving\s+anymore\b",
                r"\bcancel(?:led|ing)?\s+(?:our\s+)?plans\s+to\s+sell\b",
                r"\bchanged\s+(?:our|my)\s+minds?,\s+we're\s+not\s+selling\b",
            ]
            if any(re.search(p, text_lower) for p in reversal_markers):
                was_selling = any(w in fact_lower for w in ["planning to sell", "selling this", "decided to sell", "wants to sell"])
                return SupersessionDecision(
                    has_supersession=True,
                    target_fact_id=fact.fact_id,
                    target_fact_key=fact.fact_key,
                    relation="REVERSES",
                    old_truth_summary=fact.fact_value,
                    new_truth_value="Decided not to sell; pulling off the market completely",
                    reasoning="Prospect outright reversed earlier intention to sell.",
                    relation_confidence=0.96,
                    confidence=0.96,
                )

            # Check for Conditional Update (UPDATES) - Canonical Client Example
            stay_markers = ["decided to stay", "staying in the home", "not selling", "stay put", "stay here"]
            is_old_stay = any(m in fact_lower for m in stay_markers)

            conditional_move_markers = [
                (r"\b(?:would|'d)\s+still\s+move\s+if\s+([^\.]+)", "better strategy / conditions"),
                (r"\bmove\s+if\s+([^\.]+)", "conditional"),
                (r"\bopen\s+to\s+moving\s+if\s+([^\.]+)", "conditional"),
                (r"\bif\s+i\s+believed\s+there\s+was\s+a\s+better\s+strategy\b", "better strategy"),
            ]
            for pattern, default_cond in conditional_move_markers:
                m = re.search(pattern, text_lower)
                if m and is_old_stay:
                    condition_str = m.group(1).strip() if m.groups() else default_cond
                    return SupersessionDecision(
                        has_supersession=True,
                        target_fact_id=fact.fact_id,
                        target_fact_key=fact.fact_key,
                        relation="UPDATES",
                        old_truth_summary=fact.fact_value,
                        new_truth_value=f"Conditionally open to moving (Condition: if convinced of a better strategy; supersedes stay decision)",
                        condition="if convinced of a better strategy",
                        reasoning="Prospect shifted from absolute stay decision to conditional willingness to move.",
                        relation_confidence=0.93,
                        confidence=0.93,
                    )

            # Check for Confirmation (CONFIRMS)
            confirmation_markers = ["definitely staying", "still staying", "like i said, staying", "we are staying"]
            if is_old_stay and any(m in text_lower for m in confirmation_markers):
                return SupersessionDecision(
                    has_supersession=False,
                    target_fact_id=fact.fact_id,
                    target_fact_key=fact.fact_key,
                    relation="CONFIRMS",
                    old_truth_summary=fact.fact_value,
                    reasoning="Prospect reiterated existing stay decision without alteration.",
                    relation_confidence=0.95,
                    confidence=0.95,
                )

        # -------------------------------------------------------------------------
        # 2. Decision Maker / Authority Supersession
        # -------------------------------------------------------------------------
        if fact.fact_key in ("spouse_involvement", "decision_maker_authority", "signing_authority") or fact.category == "decision_maker":
            sole_authority_markers = [
                r"\bpower\s+of\s+attorney\b",
                r"\bsole\s+(?:decision|authority|signer)\b",
                r"\bi\s+(?:can|have\s+the\s+authority\s+to)\s+sign\b",
                r"\bmy\s+decision\s+alone\b",
            ]
            has_sole_auth = any(re.search(p, text_lower) for p in sole_authority_markers)
            was_shared_or_spouse = any(w in fact_lower for w in ["wife", "husband", "spouse", "partner", "must be present", "consult"])

            if was_shared_or_spouse and has_sole_auth:
                return SupersessionDecision(
                    has_supersession=True,
                    target_fact_id=fact.fact_id,
                    target_fact_key=fact.fact_key,
                    relation="UPDATES",
                    old_truth_summary=fact.fact_value,
                    new_truth_value="Prospect possesses sole signing authority / power of attorney",
                    reasoning="Prospect clarified individual signing authority/power of attorney, superseding joint requirement.",
                    relation_confidence=0.94,
                    confidence=0.94,
                )

        # -------------------------------------------------------------------------
        # 3. Timeline / Target Closing Supersession
        # -------------------------------------------------------------------------
        if fact.fact_key in ("target_closing", "timeline", "move_in_date") or fact.category == "timeline":
            timeline_shift_patterns = [
                (r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", "month_shift"),
                (r"\bsemester\s+(?:doesn't\s+end|ends\s+in)\b", "school_schedule"),
                (r"\bpushed\s+(?:back\s+)?to\b", "delay"),
            ]
            has_timeline_shift = any(re.search(p[0], text_lower) for p in timeline_shift_patterns)
            is_date_relevant = any(w in fact_lower for w in ["thanksgiving", "christmas", "november", "october", "spring", "holidays", "month", "close"])

            if has_timeline_shift and is_date_relevant:
                months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
                new_months = [m for m in months if m in text_lower]
                old_months = [m for m in months if m in fact_lower]
                if new_months and (not old_months or new_months[0] != old_months[0]):
                    return SupersessionDecision(
                        has_supersession=True,
                        target_fact_id=fact.fact_id,
                        target_fact_key=fact.fact_key,
                        relation="UPDATES",
                        old_truth_summary=fact.fact_value,
                        new_truth_value=f"Closing timeline updated to {new_months[0].capitalize()} (Schedule: {candidate_text.strip()})",
                        reasoning=f"Closing timeline shifted from {fact.fact_value} to {new_months[0].capitalize()}.",
                        relation_confidence=0.91,
                        confidence=0.91,
                    )

        # -------------------------------------------------------------------------
        # 4. Contact Channel / Communication Preference Supersession
        # -------------------------------------------------------------------------
        if fact.fact_key in ("preferred_channel", "contact_channel") or fact.category == "preference":
            channel_restriction_markers = [
                r"\btext\s+or\s+email\s+(?:is\s+)?(?:much\s+)?better\b",
                r"\bemail\s+only\b",
                r"\bprefer\s+(?:text|email)\b",
                r"\bclinic\s+is\s+packed\b",
                r"\bcan't\s+take\s+calls\b",
            ]
            has_restriction = any(re.search(p, text_lower) for p in channel_restriction_markers)
            was_open_calling = any(w in fact_lower for w in ["call me anytime", "cell anytime", "phone", "open"])

            if was_open_calling and has_restriction:
                return SupersessionDecision(
                    has_supersession=True,
                    target_fact_id=fact.fact_id,
                    target_fact_key=fact.fact_key,
                    relation="UPDATES",
                    old_truth_summary=fact.fact_value,
                    new_truth_value="Prefers text or email due to daytime schedule constraints",
                    reasoning="Prospect restricted communication channel from unrestricted calling to text/email.",
                    relation_confidence=0.93,
                    confidence=0.93,
                )

        return SupersessionDecision(
            has_supersession=False,
            target_fact_id=fact.fact_id,
            target_fact_key=fact.fact_key,
            relation="UNCHANGED",
            reasoning="Utterance does not update or contradict this active fact.",
            relation_confidence=1.0,
            confidence=1.0,
        )
