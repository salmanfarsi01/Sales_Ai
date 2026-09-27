from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from .core_intelligence_models import StrategicAction, StrategicDecision
from .conversation_state_models import ConversationStateSnapshot
from .conditional_retrieval import RetrievedFactResult

LOGGER = logging.getLogger("copilot.llm_response_gateway")


class Prompt(BaseModel):
    """Canonical Prompt machine object per Spec 11 §4.4."""
    prompt_id: str = Field(default_factory=lambda: f"prm_{uuid.uuid4().hex[:10]}")
    decision_id: str
    source_state_version: int
    text: str
    strategic_action: StrategicAction
    strategic_objective: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    displayed_at: Optional[str] = None
    status: Literal["generated", "displayed", "consumed", "skipped", "cancelled", "stale"] = "generated"
    cancellation_reason: Optional[str] = None
    expires_at: Optional[str] = None
    max_prompt_words: int = 24
    confidence: float = 1.0


class LLMResponseGateway:
    """Assembles structured context per Spec 11 §7 and generates exact teleprompter language."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        generator_func: Optional[Callable[[str, int], str]] = None,
        model_name: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.generator_func = generator_func
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def assemble_context(
        self,
        decision: StrategicDecision,
        snapshot: ConversationStateSnapshot,
        facts: List[RetrievedFactResult],
        recent_turns: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Assembles the 6-block bounded context contract required by Spec 11 §7."""
        # Block 1: Strategic Decision Contract
        block1 = (
            "### BLOCK 1: STRATEGIC DECISION CONTRACT\n"
            f"- Primary Strategic Action: {decision.primary_action.value.upper()}\n"
            f"- Secondary Action: {decision.secondary_action.value.upper() if decision.secondary_action else 'NONE'}\n"
            f"- Strategic Objective: {decision.strategic_objective}\n"
            f"- Push Strength: {decision.push_strength}\n"
            f"- Reason Codes: {', '.join(decision.reason_codes)}\n"
            f"- Prohibited Actions (DO NOT DO): {', '.join(decision.do_not_do) if decision.do_not_do else 'None'}\n"
            f"- What To Protect: {', '.join(decision.what_to_protect) if decision.what_to_protect else 'Rapport'}\n"
            f"- Question Allowed: {'YES' if decision.question_allowed else 'NO'}\n"
        )

        # Block 2: Conversation Truth & Recent Turns
        active_objs = [f"{o.canonical_category} ({o.lifecycle_state.value})" for o in snapshot.get_unresolved_objections()]
        turns_summary = ""
        if recent_turns:
            turns_lines = [f"{t.get('speaker', 'prospect').upper()}: {t.get('text', '')}" for t in recent_turns[-3:]]
            turns_summary = "\n".join(turns_lines)

        block2 = (
            "### BLOCK 2: CURRENT CONVERSATION TRUTH\n"
            f"- Stage: {snapshot.conversation_stage.value.upper()}\n"
            f"- Active Objections: {', '.join(active_objs) if active_objs else 'None'}\n"
            f"- Decision Maker Present: {'YES' if snapshot.decision_structure.decision_maker_present else 'NO'}\n"
            f"- State Version: {snapshot.state_version}\n"
            f"- Recent Turns:\n{turns_summary if turns_summary else 'Initial call interaction.'}\n"
        )

        # Block 3: Verified Knowledge Facts (Spec 11 §8)
        if facts:
            facts_lines = [f"- [{f.source_id}] {f.topic}: {f.text} (Status: {f.verification_status})" for f in facts]
            facts_text = "\n".join(facts_lines)
        else:
            facts_text = "- No external facts required for this conversational moment."

        block3 = (
            "### BLOCK 3: VERIFIED KNOWLEDGE\n"
            f"{facts_text}\n"
        )

        # Block 4: Playbook Methodology Constraints
        playbook_text = "Standard PitchProX sales methodology."
        if decision.playbook_influence:
            playbook_text = f"Methodology: {decision.playbook_influence.get('methodology', 'standard')} (ID: {decision.playbook_influence.get('playbook_id')})"

        block4 = (
            "### BLOCK 4: PLAYBOOK CONSTRAINTS\n"
            f"- {playbook_text}\n"
        )

        # Block 5: Calibration Delivery Constraints
        block5 = (
            "### BLOCK 5: CALIBRATION DELIVERY CONSTRAINTS\n"
            f"- Maximum Prompt Words: {decision.max_prompt_words}\n"
            f"- Urgency: {decision.urgency.upper()}\n"
            f"- Mode: Concise, conversational, spoken delivery\n"
        )

        # Block 6: Generation Directives
        block6 = (
            "### BLOCK 6: MANDATORY GENERATION DIRECTIVES\n"
            "Generate the exact word-for-word line the salesperson should say right now.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the spoken response. Do not wrap in quotes.\n"
            "2. Do NOT include coaching labels like 'Acknowledge:' or 'Tip:'.\n"
            "3. Do NOT invent unverified facts, credentials, or numbers.\n"
            f"4. Maximum length is {decision.max_prompt_words} words.\n"
        )

        return f"{block1}\n{block2}\n{block3}\n{block4}\n{block5}\n{block6}"

    def generate_prompt(
        self,
        decision: StrategicDecision,
        snapshot: ConversationStateSnapshot,
        facts: List[RetrievedFactResult],
        recent_turns: Optional[List[Dict[str, str]]] = None,
    ) -> Prompt:
        """Executes LLM language construction adhering to the strategic decision contract."""
        context = self.assemble_context(
            decision=decision,
            snapshot=snapshot,
            facts=facts,
            recent_turns=recent_turns,
        )

        if self.generator_func:
            raw_text = self.generator_func(context, decision.max_prompt_words)
        else:
            raw_text = self._call_llm(context, decision)

        cleaned_text = self._clean_and_truncate(raw_text, decision.max_prompt_words)

        return Prompt(
            decision_id=decision.decision_id,
            source_state_version=decision.source_state_version,
            text=cleaned_text,
            strategic_action=decision.primary_action,
            strategic_objective=decision.strategic_objective,
            max_prompt_words=decision.max_prompt_words,
            confidence=decision.confidence,
        )

    def _call_llm(self, context: str, decision: StrategicDecision) -> str:
        # If client is configured and not mock, execute real LLM call
        if self.llm_client and hasattr(self.llm_client, "chat"):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are PitchProX Teleprompter Engine. Output only the exact sentence to speak."},
                        {"role": "user", "content": context},
                    ],
                    temperature=0.2,
                    max_tokens=60,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                LOGGER.warning("LLM call failed, falling back to deterministic formulation: %s", e)

        # Fallback deterministic formulation matching the strategic action
        return self._deterministic_fallback(decision)

    def _deterministic_fallback(self, decision: StrategicDecision) -> str:
        action = decision.primary_action
        if action == StrategicAction.ACKNOWLEDGE:
            if "HARD_BOUNDARY_ACTIVE" in decision.reason_codes:
                return "I completely respect that. Thank you for your time, and have a great rest of your day."
            if "CONVERSION_CONFIRMED" in decision.reason_codes:
                return "Perfect, I have Thursday at three confirmed on my calendar. I will see you both then."
            return "I completely understand where you're coming from."
        elif action == StrategicAction.VALIDATE:
            return "That makes complete sense, and a lot of sellers have that exact same concern."
        elif action == StrategicAction.REFRAME:
            return "What matters most isn't just the fee percentage, but what you walk away with in your pocket."
        elif action == StrategicAction.DE_RISK:
            return "There is zero obligation—if our approach doesn't make total sense, you can walk away anytime."
        elif action == StrategicAction.COMMITMENT_CLOSE:
            if decision.secondary_action == StrategicAction.QUESTION:
                return "Would Thursday at two or Friday morning work better for a brief walkthrough?"
            return "Let's schedule a twenty minute walkthrough so we can review the exact numbers in person."
        elif action == StrategicAction.QUESTION:
            return "What would be the most important outcome for you if you were to make a move?"
        elif action == StrategicAction.EDUCATE:
            return "Current market inventory is moving faster than last quarter, which directly impacts your pricing window."
        else:
            return "I appreciate you sharing that, let's explore what makes the most sense for your timeline."

    def _clean_and_truncate(self, text: str, max_words: int) -> str:
        cleaned = text.strip().strip('"').strip("'")
        cleaned = re.sub(r"^(Teleprompter|Rep|Agent|Response|Action):\s*", "", cleaned, flags=re.IGNORECASE)
        words = cleaned.split()
        if len(words) > max_words:
            truncated = " ".join(words[:max_words])
            if not truncated.endswith((".", "!", "?")):
                truncated += "."
            return truncated
        return cleaned
