import re
from typing import Any, Dict, List, Optional, Set, Tuple
from copilot.core_intelligence_models import StrategicAction, StrategicDecision
from copilot.conversation_state_models import ConversationStateSnapshot
from copilot.llm_response_gateway import Prompt
from copilot.observation_models import (
    LearningCandidate,
    ObservationConfig,
    PromptOutcome,
    RepPromptBehavior,
)
from copilot.playbook import compute_token_semantic_similarity

# Domain-specific generic stopwords that do NOT indicate substantive topic continuity on their own
_DOMAIN_GENERIC_STOPWORDS: Set[str] = {
    "house", "home", "property", "price", "sale", "sell", "buy", "think", "talk",
    "call", "agent", "realtor", "market", "know", "want", "need", "make", "time",
    "good", "well", "said", "just", "like", "look", "really", "going", "come",
    "hello", "hi", "hey", "yes", "no", "yeah", "okay", "sure", "fine", "thing",
}

# Referential pronouns and stems indicating anaphoric continuity (e.g. "talk to her", "both of us")
_REFERENTIAL_STEMS: Set[str] = {
    "her", "him", "she", "he", "they", "them", "both", "together", "we", "us",
    "that", "this", "those", "issue", "concern", "decision", "situation",
}

# Canonical semantic violation exemplars for high-stakes guardrail checks
_GUARDRAIL_EXEMPLARS: Dict[str, List[str]] = {
    "premature_close": [
        "sign the contract today",
        "list with me right now",
        "ready to put your signature on the agreement",
        "commit to working with me today",
        "sign the paperwork right now",
    ],
    "oversell": [
        "let me give you another reason why we are the best",
        "did i mention our award winning marketing",
        "we are number one in the county",
        "let me tell you more about our accomplishments",
    ],
    "press_absent_spouse": [
        "you can decide without your wife",
        "dont wait for your husband",
        "why do we need both of you",
        "you can make the call yourself",
        "sign now and tell her later",
    ],
    "reopen_resolved_objections": [
        "are you sure you are okay with the commission",
        "going back to your concern about our fee",
        "do you still have doubts about our contract",
    ],
    "boundary_violation": [
        "i know you said stop calling but",
        "just give me two more minutes please",
        "ignore what you said about not contacting you",
    ],
}


class PromptObservationEngine:
    """Spec 11 §4.5 and Spec 01 §12 Observation Engine.

    Measures rep teleprompter adherence, distinguishes useful from harmful deviations,
    and unconditionally enforces do_not_do and what_to_protect guardrails against ASR noise.
    """

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()

    def observe_turn(
        self,
        prompt: Prompt,
        decision: StrategicDecision,
        rep_utterance: str,
        rep_turn_id: str,
        initial_snapshot: ConversationStateSnapshot,
        prospect_reactions: Optional[List[Dict[str, Any]]] = None,
        final_snapshot: Optional[ConversationStateSnapshot] = None,
        was_interrupted: bool = False,
        verified_facts_present: bool = False,
    ) -> PromptOutcome:
        """Evaluates rep teleprompter execution, computes adherence, and captures prospect reaction."""
        clean_rep = rep_utterance.strip().lower()
        clean_prompt = prompt.text.strip().lower()

        # 1. Compute Independent Signal: Lexical Overlap
        lexical_overlap = self._compute_lexical_overlap(clean_prompt, clean_rep)

        # 2. Compute Independent Signal: Structural Action Execution (all 15 canonical actions)
        structural_action_executed = self._check_structural_action(clean_rep, prompt.strategic_action, initial_snapshot)

        # 3. Compute Independent Signal: Topic Continuity (substantive anchor matching)
        topic_continuity = self._compute_topic_continuity(clean_rep, decision, initial_snapshot)

        # 4. Attribute Prospect Reaction Window and State Deltas
        reaction_turn_ids, observed_delta = self._attribute_reaction_window(
            prospect_reactions=prospect_reactions or [],
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
        )

        delta_trust = observed_delta.get("trust", 0.0)
        delta_momentum = observed_delta.get("momentum", 0.0)

        # 5. Priority 1: Unconditional Guardrail Override (do_not_do & what_to_protect)
        has_prohibition_breach, breach_reason = self._detect_prohibited_action(
            rep_text=clean_rep,
            do_not_do=decision.do_not_do,
            verified_facts_present=verified_facts_present,
            strategic_action=prompt.strategic_action,
        )
        has_protection_breach, protection_reason = self._detect_protection_breach(
            rep_text=clean_rep,
            what_to_protect=decision.what_to_protect,
        )

        if has_prohibition_breach or has_protection_breach:
            violation = breach_reason or protection_reason
            return PromptOutcome(
                prompt_id=prompt.prompt_id,
                decision_id=decision.decision_id,
                rep_behavior=RepPromptBehavior.HARMFUL_DEVIATION,
                adherence_score=0.05,
                rep_turn_ids=[rep_turn_id],
                prospect_reaction_turn_ids=reaction_turn_ids,
                observed_state_delta=observed_delta,
                confidence=0.95,
                violation_reason=violation,
                lexical_overlap=round(lexical_overlap, 3),
                structural_action_executed=structural_action_executed,
                topic_continuity=topic_continuity,
            )

        # 6. Priority 2: Interrupted Delivery
        if was_interrupted:
            score = round(min(0.50, max(0.10, lexical_overlap * 0.50)), 3)
            return PromptOutcome(
                prompt_id=prompt.prompt_id,
                decision_id=decision.decision_id,
                rep_behavior=RepPromptBehavior.INTERRUPTED,
                adherence_score=score,
                rep_turn_ids=[rep_turn_id],
                prospect_reaction_turn_ids=reaction_turn_ids,
                observed_state_delta=observed_delta,
                confidence=0.90,
                lexical_overlap=round(lexical_overlap, 3),
                structural_action_executed=structural_action_executed,
                topic_continuity=topic_continuity,
            )

        # 7. Priority 3: Exact Delivery
        if lexical_overlap >= self.config.lexical_exact_threshold:
            score = round(0.90 + 0.10 * lexical_overlap, 3)
            return PromptOutcome(
                prompt_id=prompt.prompt_id,
                decision_id=decision.decision_id,
                rep_behavior=RepPromptBehavior.EXACT,
                adherence_score=min(1.0, score),
                rep_turn_ids=[rep_turn_id],
                prospect_reaction_turn_ids=reaction_turn_ids,
                observed_state_delta=observed_delta,
                confidence=0.92,
                lexical_overlap=round(lexical_overlap, 3),
                structural_action_executed=structural_action_executed,
                topic_continuity=topic_continuity,
            )

        # 8. Priority 4 & 5: Structural Action Executed -> Strategic Paraphrase vs. Skipped
        if structural_action_executed:
            if topic_continuity:
                score = round(0.70 + 0.20 * lexical_overlap, 3)
                return PromptOutcome(
                    prompt_id=prompt.prompt_id,
                    decision_id=decision.decision_id,
                    rep_behavior=RepPromptBehavior.STRATEGIC_PARAPHRASE,
                    adherence_score=min(0.90, score),
                    rep_turn_ids=[rep_turn_id],
                    prospect_reaction_turn_ids=reaction_turn_ids,
                    observed_state_delta=observed_delta,
                    confidence=0.88,
                    lexical_overlap=round(lexical_overlap, 3),
                    structural_action_executed=True,
                    topic_continuity=True,
                )
            else:
                return PromptOutcome(
                    prompt_id=prompt.prompt_id,
                    decision_id=decision.decision_id,
                    rep_behavior=RepPromptBehavior.SKIPPED,
                    adherence_score=0.0,
                    rep_turn_ids=[rep_turn_id],
                    prospect_reaction_turn_ids=reaction_turn_ids,
                    observed_state_delta=observed_delta,
                    confidence=0.85,
                    lexical_overlap=round(lexical_overlap, 3),
                    structural_action_executed=True,
                    topic_continuity=False,
                )

        # 9. Priority 6, 7 & 8: Structural Action Not Executed -> Useful / Harmful Deviation vs. Skipped
        if topic_continuity:
            # Deterministically partitions (delta_trust, delta_momentum):
            # Useful: Net positive or neutral (including delta_trust=0, delta_momentum>0)
            if delta_trust >= 0.0 and delta_momentum >= 0.0:
                bonus = min(0.20, max(0.0, (delta_trust + delta_momentum) / 50.0))
                score = round(0.40 + bonus, 3)
                return PromptOutcome(
                    prompt_id=prompt.prompt_id,
                    decision_id=decision.decision_id,
                    rep_behavior=RepPromptBehavior.USEFUL_DEVIATION,
                    adherence_score=min(0.60, score),
                    rep_turn_ids=[rep_turn_id],
                    prospect_reaction_turn_ids=reaction_turn_ids,
                    observed_state_delta=observed_delta,
                    confidence=0.82,
                    lexical_overlap=round(lexical_overlap, 3),
                    structural_action_executed=False,
                    topic_continuity=True,
                )
            else:
                # Regressed trust or momentum -> Harmful deviation
                penalty = min(0.15, max(0.0, abs(min(delta_trust, delta_momentum)) / 50.0))
                score = round(max(0.05, 0.20 - penalty), 3)
                return PromptOutcome(
                    prompt_id=prompt.prompt_id,
                    decision_id=decision.decision_id,
                    rep_behavior=RepPromptBehavior.HARMFUL_DEVIATION,
                    adherence_score=score,
                    rep_turn_ids=[rep_turn_id],
                    prospect_reaction_turn_ids=reaction_turn_ids,
                    observed_state_delta=observed_delta,
                    confidence=0.85,
                    lexical_overlap=round(lexical_overlap, 3),
                    structural_action_executed=False,
                    topic_continuity=True,
                )

        # 10. Abandoned topic and action -> Skipped
        return PromptOutcome(
            prompt_id=prompt.prompt_id,
            decision_id=decision.decision_id,
            rep_behavior=RepPromptBehavior.SKIPPED,
            adherence_score=0.0,
            rep_turn_ids=[rep_turn_id],
            prospect_reaction_turn_ids=reaction_turn_ids,
            observed_state_delta=observed_delta,
            confidence=0.90,
            lexical_overlap=round(lexical_overlap, 3),
            structural_action_executed=False,
            topic_continuity=False,
        )

    def _compute_lexical_overlap(self, prompt_text: str, rep_text: str) -> float:
        """Computes Jaccard word-level overlap ratio between prompt and rep utterance."""
        p_words = set(re.findall(r"\b[a-z0-9']+\b", prompt_text))
        r_words = set(re.findall(r"\b[a-z0-9']+\b", rep_text))
        if not p_words or not r_words:
            return 0.0
        intersection = p_words.intersection(r_words)
        union = p_words.union(r_words)
        return len(intersection) / len(union)

    def _check_structural_action(
        self,
        rep_text: str,
        action: StrategicAction,
        snapshot: ConversationStateSnapshot,
    ) -> bool:
        """Validates execution of the required strategic action across all 15 Spec 01 §9 categories."""
        if action == StrategicAction.WAIT_SILENCE:
            return len(rep_text.split()) == 0

        elif action == StrategicAction.ACKNOWLEDGE:
            stems = ["understand", "hear you", "got it", "appreciate", "fair point", "absolutely", "makes sense"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.CLARIFY:
            stems = ["when you say", "specifically", "help me understand", "tell me more", "mean by", "what sort of"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.VALIDATE:
            stems = ["makes complete sense", "makes total sense", "understandable", "natural to feel", "valid concern", "lot of folks feel", "normal to wonder"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.MIRROR:
            # Check for repetition of prospect's key words from previous turn if available
            return any(stem in rep_text for stem in ["you mentioned", "as you said", "like you said"]) or len(rep_text.split()) <= 4

        elif action == StrategicAction.REFRAME:
            stems = ["what really matters", "look at it from", "instead of just", "the real question", "bottom line", "walk away with", "net return"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.EDUCATE:
            stems = ["typically in this market", "the way contracts work", "how the timeline unfolds", "inventory data shows", "industry standard", "average days on market"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.QUANTIFY:
            stems = ["numbers", "net sheet", "break down the figures", "walk through the math", "dollar", "calculate", "side by side"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.DIFFERENTIATE:
            stems = ["unlike other agents", "what sets us apart", "our specific approach", "the difference between", "how we handle marketing", "unique strategy"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.DE_RISK:
            stems = ["zero obligation", "no risk", "walk away anytime", "no pressure", "cancel anytime", "no commitment today", "feel free to say no"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.SOCIAL_PROOF:
            stems = ["other homeowners", "neighbors on", "recent sellers", "past clients", "similar situation in your neighborhood"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.CHALLENGE:
            stems = ["are you certain", "what happens if", "is that really", "have you factored in", "what is the cost of waiting", "does that guarantee"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.FUTURE_PACE:
            stems = ["fast forward", "imagine once", "after closing", "down the road", "when you're settled", "months from now"]
            return any(s in rep_text for s in stems)

        elif action == StrategicAction.QUESTION:
            # ASR-safe question check: syntactic inversions and interrogative pronouns (zero dependency on '?')
            interrogatives = ["what", "how", "why", "when", "where", "who", "which"]
            inversions = [
                "would you", "could we", "can you", "will you", "should we", "may i",
                "do you", "did you", "have you", "are you", "is it", "does that", "has anyone",
            ]
            tags = ["right", "correct", "does that sound fair", "fair enough", "make sense", "work for you"]

            words = re.findall(r"\b[a-z']+\b", rep_text)
            if not words:
                return False

            first_three = " ".join(words[:3])
            has_interrogative = any(rep_text.startswith(q) or f" {q} " in rep_text for q in interrogatives)
            has_inversion = any(inv in rep_text for inv in inversions)
            has_tag = any(rep_text.endswith(t) or f"{t} " in rep_text for t in tags)

            return has_interrogative or has_inversion or has_tag or rep_text.endswith("?")

        elif action == StrategicAction.COMMITMENT_CLOSE:
            stems = ["walkthrough", "meet", "stop by", "calendar", "thursday", "friday", "twenty minutes", "schedule a time", "come by"]
            return any(s in rep_text for s in stems)

        return False

    def _compute_topic_continuity(
        self,
        rep_text: str,
        decision: StrategicDecision,
        snapshot: ConversationStateSnapshot,
    ) -> bool:
        """Determines topic continuity against substantive objection/stage anchors and referential pronouns."""
        rep_words = set(re.findall(r"\b[a-z0-9']+\b", rep_text))
        substantive_rep = rep_words - _DOMAIN_GENERIC_STOPWORDS

        if not substantive_rep:
            return False

        # Extract substantive topic anchors from active objections
        target_anchors: Set[str] = set()
        for obj in snapshot.objections:
            if obj.lifecycle_state not in ("resolved", "superseded", "dormant"):
                cat = obj.canonical_category.lower()
                if "commission" in cat or "financial" in cat:
                    target_anchors.update(["commission", "fee", "percentage", "six", "discount", "compensation", "net", "cut", "cost"])
                elif "spouse" in cat or "decision" in cat:
                    target_anchors.update(["wife", "husband", "spouse", "partner", "together", "both", "discuss", "consult"])
                elif "timing" in cat or "timeline" in cat:
                    target_anchors.update(["spring", "rush", "wait", "holiday", "rates", "interest", "holding", "delay"])
                elif "condition" in cat:
                    target_anchors.update(["repairs", "as-is", "inspection", "fix", "contractor", "roof", "hvac", "work"])

        # Extract substantive tokens from decision objective
        obj_words = set(re.findall(r"\b[a-z0-9']+\b", decision.strategic_objective.lower()))
        target_anchors.update(obj_words - _DOMAIN_GENERIC_STOPWORDS)

        if not target_anchors:
            return True

        # Check for substantive anchor intersection
        if substantive_rep.intersection(target_anchors):
            return True

        # Check for referential/anaphoric continuity (e.g. "talk to her", "both of us") when dealing with spouse/decision structure
        has_referential = bool(substantive_rep.intersection(_REFERENTIAL_STEMS))
        if has_referential and any("spouse" in a or "wife" in a or "partner" in a for a in target_anchors):
            return True

        # Semantic similarity fallback across anchors
        anchor_text = " ".join(target_anchors)
        sim = compute_token_semantic_similarity(rep_text, anchor_text)
        return sim >= self.config.topic_continuity_threshold

    def _detect_prohibited_action(
        self,
        rep_text: str,
        do_not_do: List[str],
        verified_facts_present: bool,
        strategic_action: StrategicAction,
    ) -> Tuple[bool, Optional[str]]:
        """Unconditionally detects do_not_do violations and unverified social proof improvisation."""
        # 1. Social Proof Fabrication Guard: Rep cites client results when evidence is unverified
        if strategic_action == StrategicAction.SOCIAL_PROOF or any(kw in do_not_do for kw in ("fabricate_client_results", "claim_prior_customer_outcomes")):
            has_client_entity = any(w in rep_text for w in ["homeowner", "homeowners", "client", "clients", "seller", "sellers", "neighbor", "neighbors"])
            has_claim_verb = any(w in rep_text for w in ["helped", "sold", "achieve", "achieved", "saved", "track record", "past results", "similar situation"])
            if has_client_entity and has_claim_verb and not verified_facts_present:
                return True, "FABRICATED_UNVERIFIED_CLAIM_BREACH"

        # 2. Semantic Intent matching against do_not_do constraint exemplars
        for prohibited in do_not_do:
            exemplars = _GUARDRAIL_EXEMPLARS.get(prohibited, [])
            for ex in exemplars:
                sim = compute_token_semantic_similarity(rep_text, ex)
                if sim >= self.config.semantic_similarity_threshold:
                    return True, f"PROHIBITED_DIRECTIVE_BREACH_{prohibited.upper()}"

        return False, None

    def _detect_protection_breach(
        self,
        rep_text: str,
        what_to_protect: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """Detects damage to protected conversational entities (e.g. confirmed appointments, hard boundaries)."""
        if "confirmed_appointment" in what_to_protect or "protect_confirmed_appointment" in what_to_protect:
            # Re-selling or reopening after Thursday confirmed
            reopen_stems = ["why not meet earlier", "are you sure thursday works", "let me tell you why we are better"]
            if any(s in rep_text for s in reopen_stems):
                return True, "CONFIRMED_APPOINTMENT_COMPROMISED"

        if "hard_boundary" in what_to_protect:
            # Continuing persuasion when prospect set a hard boundary
            boundary_stems = ["let me just say", "give me one minute", "before you hang up"]
            if any(s in rep_text for s in boundary_stems):
                return True, "CONTACT_BOUNDARY_COMPROMISED"

        return False, None

    def _attribute_reaction_window(
        self,
        prospect_reactions: List[Dict[str, Any]],
        initial_snapshot: ConversationStateSnapshot,
        final_snapshot: Optional[ConversationStateSnapshot],
    ) -> Tuple[List[str], Dict[str, float]]:
        """Attributes prospect response window (N=1 with N<=2 backchannel extension) and computes delta."""
        if not prospect_reactions:
            return [], {}

        attributed_turn_ids: List[str] = []
        for i, turn in enumerate(prospect_reactions[:self.config.max_reaction_window_turns]):
            text = str(turn.get("text", "")).strip()
            t_id = str(turn.get("turn_id", f"turn_rx_{i}"))
            attributed_turn_ids.append(t_id)

            # If first turn is substantial (> backchannel_max_words), do not extend window
            word_count = len(text.split())
            if i == 0 and word_count > self.config.backchannel_max_words:
                break

        # Compute observed state delta between initial snapshot and final snapshot
        delta: Dict[str, float] = {}
        if final_snapshot and initial_snapshot:
            # Trust delta
            delta["trust"] = round(final_snapshot.dimensions.trust - initial_snapshot.dimensions.trust, 2)
            # Engagement delta
            delta["engagement"] = round(final_snapshot.dimensions.engagement - initial_snapshot.dimensions.engagement, 2)
            # Momentum delta
            init_m = initial_snapshot.momentum.momentum_score if initial_snapshot.momentum else 50.0
            fin_m = final_snapshot.momentum.momentum_score if final_snapshot.momentum else 50.0
            delta["momentum"] = round(fin_m - init_m, 1)

        return attributed_turn_ids, delta

    def record_learning_candidate(
        self,
        call_id: str,
        turn_id: str,
        raw_utterance: str,
        candidate_type: str = "novel_objection",
        metadata: Optional[Dict[str, str]] = None,
    ) -> LearningCandidate:
        """Spec 11 §13.1 Scenario 22 / Spec 05.

        Persists novel, unclassified objections in isolated candidate retention storage
        to prevent immediate pollution of authoritative Core doctrine.
        """
        return LearningCandidate(
            call_id=call_id,
            turn_id=turn_id,
            raw_utterance=raw_utterance,
            candidate_type=candidate_type, # type: ignore
            isolated_from_core_doctrine=True,
            occurrence_count=1,
            metadata=metadata or {},
        )
