from __future__ import annotations

import os
import re
import json
import asyncio
import logging
from typing import List, Optional, Literal, Tuple, Any, Dict
from pydantic import BaseModel, Field

from .behavioral_normalization import NormalizedUtterance, CallMetadata

LOGGER = logging.getLogger("copilot.behavioral_semantic")

CONFIDENCE_BY_FEATURE: Dict[str, Dict[str, float]] = {
    "llm": {
        "boundary": 0.95,
        "specificity": 0.95,
        "future_language": 0.95,
        "recurrence": 0.95,
        "question_type": 0.95,
        "agreement": 0.95,
    },
    "heuristic": {
        "boundary": 0.95,
        "specificity": 0.85,
        "future_language": 0.85,
        "recurrence": 0.80,
        "question_type": 0.70,
        "agreement": 0.65,
    },
    "heuristic_bypass": {
        "boundary": 0.95,        # Evaluated via deterministic compliance regex
        "recurrence": 0.80,      # Evaluated via deterministic recurrence engine
        "agreement": 0.65,       # Evaluated via polite agreement regex
        "question_type": 0.70,   # Evaluated via syntax/question mark absence
        "specificity": 0.50,     # Bypassed/unmeasured: neutral prior, not evaluated by LLM
        "future_language": 0.50, # Bypassed/unmeasured: neutral prior, not evaluated by LLM
    },
}


class SemanticFeatureSnapshot(BaseModel):
    utterance_id: str
    call_sid: str
    speaker_id: Literal["salesperson", "client"]
    extraction_mode: Literal["llm", "heuristic_timeout", "heuristic_error", "heuristic_offline", "mixed"] = "llm"
    recurrence_id: Optional[str] = None
    recurrence_type: Literal[
        "same_objection_repeated",
        "concern_after_failed_reframe",
        "positive_echo",
        "boundary_repeated",
        "scheduling_detail_repeated",
        "scheduling_repeated",
        "none"
    ] = "none"
    recurrence_count: int = 0
    question_type: Literal["evaluation", "transactional", "clarifying", "hostile", "rhetorical", "none"] = "none"
    specificity_score: float = Field(0.0, ge=0.0, le=1.0)
    future_language_score: float = Field(0.0, ge=0.0, le=1.0)
    boundary_score: float = Field(0.0, ge=0.0, le=1.0)
    contact_preference: Literal["none", "reduced_frequency", "channel_restriction", "timing_restriction"] = "none"
    contact_preference_confidence: float = Field(0.0, ge=0.0, le=1.0)
    contact_preference_details: Optional[str] = None
    agreement_score: float = Field(0.0, ge=0.0, le=1.0)
    filler_score: Optional[float] = None
    is_filler_available: bool = False
    boundary_confidence: float = Field(0.95, ge=0.0, le=1.0)
    recurrence_confidence: float = Field(0.95, ge=0.0, le=1.0)
    question_type_confidence: float = Field(0.95, ge=0.0, le=1.0)
    specificity_confidence: float = Field(0.95, ge=0.0, le=1.0)
    future_language_confidence: float = Field(0.95, ge=0.0, le=1.0)
    agreement_confidence: float = Field(0.95, ge=0.0, le=1.0)
    semantic_confidence: float = Field(1.0, ge=0.0, le=1.0)


STOP_CONTACT_PATTERNS = [
    r"\bdo\s+not\s+call\b",
    r"\bdon['’]?t\s+call\b",
    r"\bstop\s+calling\b",
    r"\btake\s+me\s+off\b",
    r"\b(take|remove)\s+(my\s+)?(name|number|info|information)\s+off\b",
    r"\bremove\s+(my\s+)?(number|name)\b",
    r"\blose\s+my\s+number\b",
    r"\bdo\s+not\s+(ever\s+)?contact\b",
    r"\bdon['’]?t\s+(ever\s+)?contact\b",
    r"\bstop\s+contacting\b",
    r"\b(do\s+not|don['’]?t|rather\s+you\s+didn['’]?t|prefer\s+you\s+didn['’]?t|stop)\s+reach(ing)?\s+out\b",
    r"\bnot\s+reach\s+out\b",
    r"\b(prefer|rather)\s+not\s+to\s+be\s+contacted\b",
    r"\bstop\s+harassing\b",
    r"\bleave\s+me\s+alone\b",
    r"\bnot\s+interested\b.*?\b(do\s+not|don['’]?t)\s+(call|contact|reach|message|bother|email|text)\b",
    r"\balready\s+(have\s+an?\s+agent|listed|under\s+contract|represented|signed)\b",
    r"\bunder\s+contract\b",
    r"\b(working|signed)\s+with\s+(another|someone|an?\s+agent|a\s+realtor|a\s+broker)\b",
    r"\b(have|got)\s+an?\s+(agent|realtor|broker)\b",
    r"\b(another|an)\s+(agent|realtor|broker)\b",
    r"\brepresented\s+by\b",
    r"\bhave\s+an\s+exclusive\b",
    r"\bcall\s+my\s+(attorney|lawyer)\b",
    r"\b(put\s+me\s+on\s+the|on\s+the|to\s+the)\s+do\s+not\s+call\b",
    r"\bdnc\s+list\b",
]

# Separate pattern dictionary for soft communication cadence and channel preferences.
# Distinct from STOP_CONTACT_PATTERNS to guarantee soft preferences never trigger hard compliance overrides.
SOFT_CONTACT_PREFERENCE_PATTERNS: Dict[str, List[str]] = {
    "reduced_frequency": [
        r"\bdon['’]?t\s+(text|call|message|reach\s+out)\s+(every\s+day|all\s+the\s+time|so\s+(often|much)|multiple\s+times|constantly|daily)\b",
        r"\b(please\s+)?don['’]?t\s+call\s+so\s+often\b",
        r"\bstop\s+calling\s+(so\s+much|so\s+often|every\s+day|multiple\s+times)\b",
        r"\b(call|reach\s+out|contact)\s+(me\s+)?less\s+often\b",
        r"\btoo\s+many\s+(calls|texts|messages)\b",
        r"\bnot\s+every\s+day\b",
        r"\bonce\s+a\s+(week|month)\s+is\s+(enough|fine|plenty)\b",
    ],
    "channel_restriction": [
        r"\bemail\s+(instead\s+of|rather\s+than)\s+(calling|call|texting|text)\b",
        r"\b(please\s+)?email\s+me\s+instead\b",
        r"\b(just|only)\s+email\s+me\b",
        r"\bdon['’]?t\s+call\s*(just|only|,)?\s*(send\s+an?\s+)?email\b",
        r"\b(text|texting)\s+(only|instead)\b",
        r"\bprefer\s+(an?\s+)?email\b",
        r"\bprefer\s+(to\s+be\s+contacted\s+by|via)\s+email\b",
        r"\bsend\s+me\s+an?\s+email\s+instead\b",
        r"\breach\s+out\s+(by|via)\s+email\s+instead\b",
        r"\bcommunicate\s+(by|via)\s+email\s+only\b",
        r"\btext\s+me\s+instead\s+of\s+calling\b",
    ],
    "timing_restriction": [
        r"\b(only\s+)?(reach\s+out|call|contact)\s+(during|in)\s+business\s+hours\b",
        r"\bonly\s+during\s+business\s+hours\b",
        r"\b(call|reach\s+out)\s+after\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b",
        r"\bdon['’]?t\s+call\s+before\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b",
        r"\bonly\s+(call|reach\s+out)\s+on\s+weekends\b",
        r"\bcall\s+me\s+in\s+the\s+evening\b",
        r"\bdon['’]?t\s+call\s+(during|while\s+i['’]?m\s+at)\s+work\b",
        r"\bonly\s+call\s+in\s+the\s+afternoon\b",
        r"\bcall\s+between\s+\d{1,2}\s+and\s+\d{1,2}\b",
    ],
}

SPECIFICITY_PATTERNS = [
    r"\$\s*\d+[\d,]*(\.\d+)?\s*(k|m|million|thousand)?\b",
    r"\b\d+[\d,]*\s*(percent|%)\b",
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(day|days|week|weeks|month|months|year|years)(\s+old)?\b",
    r"\b(last|next|past|this)\s+(month|week|year|weekend|quarter|summer|spring|winter|fall|january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(one|two|three|four|five|six|\d+)\s+(bed|bedroom|bedrooms|bath|baths|bathroom|bathrooms|sqft|square\s+feet)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
    r"\b(yesterday|tomorrow|today)\b",
    r"\b(attorney|probate|tenant|landlord|contractor|escrow|appraiser|inspector)\b",
]

FUTURE_LANGUAGE_PATTERNS = [
    r"\bwhen\s+(we|i)\s+(sell|list|move|close|relocate)\b",
    r"\b(we|i)\s+(plan|intend|want)\s+to\s+(sell|list|move|close)\b",
    r"\b(by|in|around)\s+(october|november|december|january|spring|summer|fall|next\s+month|the\s+end\s+of)\b",
    r"\bonce\s+(the\s+tenant|we\s+finish|probate\s+clears|school\s+starts)\b",
    r"\bour\s+timeline\s+is\b",
]

SUBSTANTIVE_AGREEMENT_PATTERNS = [
    r"\b(yes|yeah|sure)\b.*?\b(tuesday|wednesday|thursday|friday|monday|tomorrow|\d+\s*(pm|am))\b",
    r"\b(that\s+works|sounds\s+good|let['’]?s\s+do\s+that|i\s+agree|let['’]?s\s+meet)\b",
    r"\bi\s+can\s+(do|meet|show|send)\b",
]

POLITE_AGREEMENT_PATTERNS = [
    r"^(yeah|yes|okay|ok|sure|right|uh-huh|yep|gotcha)[\.\,\!]*$",
]

HOSTILE_QUESTION_PATTERNS = [
    r"\bhow\s+did\s+you\s+(get|find)\s+my\b",
    r"\bwho\s+gave\s+you\s+(this|my)\b",
    r"\bwhy\s+are\s+you\s+(calling|bothering|contacting)\b",
    r"\bwhy\s+(do|are)\s+you\s+(call|calling)\b",
    r"\bwho\s+(is\s+this|are\s+you)\b",
    r"\bwhat\s+do\s+you\s+want\b",
]

TRANSACTIONAL_QUESTION_PATTERNS = [
    r"\bwhat\s+(is|are)\s+(your\s+)?([a-z]+\s+)?(commission|fee|fees|rate|cost|percentage)\b",
    r"\bhow\s+much\s+(do\s+you\s+charge|is\s+it|does\s+it\s+cost)\b",
    r"\bhow\s+long\s+is\s+the\s+listing\s+(agreement|contract)\b",
    r"\b(commission|listing\s+fee|brokerage\s+fee|closing\s+costs)\b",
]

EVALUATION_QUESTION_PATTERNS = [
    r"\bhow\s+do\s+you\s+(market|sell|find\s+buyers|differentiate)\b",
    r"\bwhat\s+do\s+you\s+do\s+(differently|different)\b",
    r"\bwhat\s+is\s+your\s+(strategy|track\s+record|experience)\b",
]

OBJECTION_RESISTANCE_PATTERNS = [
    r"\b(too\s+(much|high|expensive|steep|costly|pricey|far))\b",
    r"\b(feels?|seems?|think|thought)\s+(pretty\s+|really\s+|just\s+)?(like\s+a\s+lot|steep|high)\b",
    r"\b(a\s+lot\s+(for|of\s+money))\b",
    r"\b(paying|pay|charged?)\s+too\s+much\b",
    r"\bdon['’]?t\s+want\s+to\s+pay\b",
    r"\b(can['’]?t|cannot)\s+afford\b",
    r"\b(out\s+of|over)\s+(our|my)\s+budget\b",
    r"\bnot\s+worth\s+(it|the\s+money|the\s+cost|the\s+fee)\b",
    r"\b(hesitant|concerned|worried|skeptical|doubtful)\b",
    r"\b(worries|concerns|bothers)\s+me\b",
    r"\b(not|doesn['’]?t|does\s+not)\s+(?:[a-z]+\s+){0,3}(?:right\s+(?:with\s+me|for\s+us)|sit\s+right)\b",
    r"\b(unclear|confusing|hard\s+to\s+understand)\b",
    r"\b(not\s+sure|not\s+ready|rethink|second\s+thought)\b",
    r"\b(lower|cut|reduce|discount)\s+(your\s+)?(commission|fee|rate|price)\b",
]

OBJECTION_TOPIC_PATTERNS = {
    "commission_pricing": [
        r"\b(commission|fee|fees|cost|costs|price|pricing|paying|charge|rate)\b",
    ],
    "timing_delay": [
        r"\b(rush|hurry|wait|later|not\s+ready|bad\s+time|delay)\b",
    ],
    "spousal_third_party": [
        r"\b(husband|wife|spouse|partner|board|attorney|lawyer)\b",
    ],
}

STOPWORDS = {
    "a", "an", "the", "that", "this", "these", "those", "our", "your", "my", "we", "you",
    "i", "us", "it", "to", "for", "in", "on", "at", "with", "still", "way", "too", "is",
    "are", "was", "were", "and", "but", "or", "of", "as", "be", "so", "do", "does", "did",
    "just", "really", "honestly", "hear", "heard", "see", "seen", "look", "what", "where",
    "when", "why", "how", "here", "there", "right", "well", "like", "feel", "feels",
    "about", "out", "then", "now", "not", "no", "yes", "yeah", "ok", "okay", "sure",
    "get", "got", "can", "could", "would", "should", "have", "has", "had", "been",
    "much", "more", "most", "some", "any", "all", "very", "even", "actually", "mean",
    "think", "know", "something", "someone", "everything", "anything", "thing", "things",
}


class SemanticFeatureEngine:
    def __init__(
        self,
        groq_client: Optional[Any] = None,
        model: str = "qwen/qwen3.6-27b",
        timeout_seconds: float = 1.0,
    ):
        self.groq_client = groq_client
        self.model = os.getenv("GROQ_FAST_MODEL", model)
        self.timeout_seconds = float(os.getenv("BEHAVIORAL_SEMANTIC_TIMEOUT_SEC", str(timeout_seconds)))

    def classify_boundary_deterministic(self, text: str) -> float:
        cleaned = text.strip().lower()
        # Soft contact preferences must never accidentally satisfy the hard-boundary check
        pref, _, _ = self.detect_contact_preference(cleaned)
        if pref != "none":
            hard_explicit_patterns = [
                r"\b(do\s+not|don['’]?t)\s+call\s+(me\s+)?(any\s*more|again|ever)\b",
                r"\bnever\s+call\b",
                r"\bstop\s+calling\s+me\b",
                r"\b(do\s+not|don['’]?t)\s+(ever\s+)?contact\b",
                r"\bstop\s+contacting\b",
                r"\bnot\s+call\s+(me\s+)?(any\s*more|again)\b",
                r"\btake\s+me\s+off\b",
                r"\b(take|remove)\s+(my\s+)?(name|number|info|information)\s+off\b",
                r"\bremove\s+(my\s+)?(number|name)\b",
                r"\blose\s+my\s+number\b",
                r"\b(put\s+me\s+on\s+the|on\s+the|to\s+the)\s+do\s+not\s+call\b",
                r"\bdnc\s+list\b",
                r"\bcall\s+my\s+(attorney|lawyer)\b",
                r"\balready\s+(have\s+an?\s+agent|listed|under\s+contract|represented|signed)\b",
                r"\bunder\s+contract\b",
                r"\bstop\s+harassing\b",
                r"\bleave\s+me\s+alone\b",
            ]
            if any(re.search(p, cleaned, re.IGNORECASE) for p in hard_explicit_patterns):
                return 1.0
            return 0.0

        for pat in STOP_CONTACT_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return 1.0
        return 0.0

    def detect_contact_preference(
        self, text: str
    ) -> Tuple[Literal["none", "reduced_frequency", "channel_restriction", "timing_restriction"], float, Optional[str]]:
        cleaned = text.strip().lower()
        for pref_type, patterns in SOFT_CONTACT_PREFERENCE_PATTERNS.items():
            for pat in patterns:
                match = re.search(pat, cleaned, re.IGNORECASE)
                if match:
                    return pref_type, 0.90, match.group(0)
        return "none", 0.0, None

    def _is_objection_utterance(self, text: str, playbook_objections: Optional[List[Any]] = None) -> Tuple[bool, Optional[str]]:
        cleaned = text.strip().lower()
        has_resistance = any(re.search(pat, cleaned) for pat in OBJECTION_RESISTANCE_PATTERNS)
        matched_cat = None
        for cat, pats in OBJECTION_TOPIC_PATTERNS.items():
            if any(re.search(pat, cleaned) for pat in pats):
                matched_cat = cat
                break
        if has_resistance:
            return True, (matched_cat or "general_objection")
        if playbook_objections:
            for obj in playbook_objections:
                obj_text = getattr(obj, "objection", "") if hasattr(obj, "objection") else str(obj)
                if obj_text and obj_text.lower() in cleaned:
                    return True, "playbook_objection"
        return False, None

    def detect_semantic_recurrence(
        self,
        current_text: str,
        history: List[NormalizedUtterance],
        playbook_objections: Optional[List[Any]] = None,
    ) -> Tuple[Optional[str], Literal["same_objection_repeated", "concern_after_failed_reframe", "positive_echo", "boundary_repeated", "scheduling_detail_repeated", "scheduling_repeated", "none"], int]:
        cleaned = current_text.strip().lower()
        if len(cleaned.split()) < 3:
            return None, "none", 0

        # Check boundary repeated
        if self.classify_boundary_deterministic(cleaned) >= 0.8:
            prior_boundaries = [
                u for u in history
                if u.speaker_id == "client" and self.classify_boundary_deterministic(u.text) >= 0.8
            ]
            if prior_boundaries:
                return "BOUND_STOP_01", "boundary_repeated", len(prior_boundaries) + 1

        # Check scheduling / value detail repeated
        rep_recent = [u for u in history if u.speaker_id == "salesperson"]
        if rep_recent:
            last_rep_text = rep_recent[-1].text.lower()
            rep_spec_matches = {
                m.group(0).lower() for pat in (SPECIFICITY_PATTERNS + SUBSTANTIVE_AGREEMENT_PATTERNS)
                for m in re.finditer(pat, last_rep_text)
            }
            if rep_spec_matches:
                client_spec_matches = {
                    m.group(0).lower() for pat in (SPECIFICITY_PATTERNS + SUBSTANTIVE_AGREEMENT_PATTERNS)
                    for m in re.finditer(pat, cleaned)
                }
                if rep_spec_matches.intersection(client_spec_matches):
                    return "SCHED_DETAIL_01", "scheduling_detail_repeated", 1

        prior_sched = [
            u for u in history
            if u.speaker_id == "client" and any(re.search(pat, u.text.lower()) for pat in SUBSTANTIVE_AGREEMENT_PATTERNS)
        ]
        if prior_sched and any(re.search(pat, cleaned) for pat in SUBSTANTIVE_AGREEMENT_PATTERNS):
            return "SCHED_DETAIL_01", "scheduling_detail_repeated", len(prior_sched) + 1

        # Check positive echo of rep wording
        rep_recent = [u for u in history if u.speaker_id == "salesperson"]
        if rep_recent:
            last_rep_words = set(re.findall(r"\b[a-z]{4,}\b", rep_recent[-1].text.lower()))
            current_words = set(re.findall(r"\b[a-z]{4,}\b", cleaned))
            overlap = last_rep_words.intersection(current_words)
            if len(overlap) >= 3 and any(w in cleaned for w in ["yes", "exactly", "that's right", "makes sense", "agree"]):
                return "ECHO_ALIGN_01", "positive_echo", 1

        # Check objection recurrence against previous client utterances
        curr_is_obj, curr_cat = self._is_objection_utterance(cleaned, playbook_objections)

        raw_tokens = set(re.findall(r"\b[a-z]{3,}\b", cleaned))
        tokens = {t for t in raw_tokens if t not in STOPWORDS}

        for prev in reversed([u for u in history if u.speaker_id == "client"]):
            prev_raw = set(re.findall(r"\b[a-z]{3,}\b", prev.text.lower()))
            prev_tokens = {t for t in prev_raw if t not in STOPWORDS}
            if not tokens or not prev_tokens:
                continue

            prev_is_obj, prev_cat = self._is_objection_utterance(prev.text, playbook_objections)

            common = tokens.intersection(prev_tokens)
            overlap_min = len(common) / min(len(tokens), len(prev_tokens)) if common else 0.0
            overlap_max = len(common) / max(len(tokens), len(prev_tokens)) if common else 0.0

            # Condition 1: Both utterances are actual objections within the same specific objection category
            same_category_objection = bool(
                curr_is_obj
                and prev_is_obj
                and curr_cat
                and curr_cat != "general_objection"
                and curr_cat == prev_cat
            )

            # Condition 2: Both utterances are objections and share at least 2 distinct content words with high overlap
            lexical_objection_repeat = bool(curr_is_obj and prev_is_obj and len(common) >= 2 and (overlap_min >= 0.5 or overlap_max >= 0.35))

            # Condition 3: Substantial content recurrence (at least 3 content words overlap, and both are objections)
            substantial_overlap = bool(curr_is_obj and prev_is_obj and len(common) >= 3 and (overlap_min >= 0.5 or overlap_max >= 0.35))

            if same_category_objection or lexical_objection_repeat or substantial_overlap:
                intervening_rep = [
                    u for u in history
                    if u.speaker_id == "salesperson" and prev.end_ms <= u.start_ms
                ]
                rec_type = "concern_after_failed_reframe" if intervening_rep else "same_objection_repeated"
                rec_id = f"OBJ_REC_{abs(hash(prev.text)) % 10000:04d}"
                return rec_id, rec_type, 2

        return None, "none", 0

    def analyze_deterministic_heuristic(
        self,
        utterance: NormalizedUtterance,
        context_history: List[NormalizedUtterance],
        mode: Literal["heuristic_timeout", "heuristic_error", "heuristic_offline", "mixed"] = "heuristic_offline",
        is_bypass: bool = False,
    ) -> SemanticFeatureSnapshot:
        text = utterance.text.strip()
        lower_text = text.lower()

        boundary_score = self.classify_boundary_deterministic(text)
        rec_id, rec_type, rec_count = self.detect_semantic_recurrence(text, context_history)

        # Question Type
        question_type: Literal["evaluation", "transactional", "clarifying", "hostile", "rhetorical", "none"] = "none"
        is_interrogative = (
            text.endswith("?")
            or any(lower_text.startswith(w) for w in ["what", "how", "why", "who", "when", "where", "which", "can", "could", "would", "should", "is", "are", "do", "does", "did", "haven't", "hasn't", "isn't", "aren't"])
        )
        if is_interrogative and boundary_score < 0.8:
            if any(re.search(pat, lower_text) for pat in HOSTILE_QUESTION_PATTERNS):
                question_type = "hostile"
            elif any(re.search(pat, lower_text) for pat in TRANSACTIONAL_QUESTION_PATTERNS):
                question_type = "transactional"
            elif any(re.search(pat, lower_text) for pat in EVALUATION_QUESTION_PATTERNS):
                question_type = "evaluation"
            elif any(w in lower_text for w in ["mean", "clarify", "repeat", "which", "say"]):
                question_type = "clarifying"
            else:
                question_type = "evaluation"

        # Specificity
        spec_matches = sum(1 for pat in SPECIFICITY_PATTERNS if re.search(pat, lower_text))
        specificity_score = min(1.0, spec_matches * 0.35)

        # Future Language
        future_matches = sum(1 for pat in FUTURE_LANGUAGE_PATTERNS if re.search(pat, lower_text))
        future_language_score = min(1.0, future_matches * 0.5)

        # Agreement
        agreement_score = 0.0
        if any(re.search(pat, lower_text) for pat in SUBSTANTIVE_AGREEMENT_PATTERNS):
            agreement_score = 0.85
        elif any(re.search(pat, lower_text) for pat in POLITE_AGREEMENT_PATTERNS):
            agreement_score = 0.30

        # Contact preference (soft preference distinct from hard boundary)
        pref_type, pref_conf, pref_details = self.detect_contact_preference(utterance.text)
        if boundary_score >= 0.8:
            pref_type = "none"
            pref_conf = 0.0
            pref_details = None

        conf_profile = "heuristic_bypass" if is_bypass else "heuristic"
        conf = CONFIDENCE_BY_FEATURE[conf_profile]
        sem_conf = round(min(min(conf.values()), float(utterance.asr_confidence)), 2)

        return SemanticFeatureSnapshot(
            utterance_id=utterance.utterance_id,
            call_sid=utterance.call_sid,
            speaker_id=utterance.speaker_id,
            extraction_mode=mode,
            recurrence_id=rec_id,
            recurrence_type=rec_type,
            recurrence_count=rec_count,
            question_type=question_type,
            specificity_score=round(specificity_score, 2),
            future_language_score=round(future_language_score, 2),
            boundary_score=round(boundary_score, 2),
            contact_preference=pref_type,
            contact_preference_confidence=pref_conf,
            contact_preference_details=pref_details,
            agreement_score=round(agreement_score, 2),
            filler_score=None,
            is_filler_available=False,
            boundary_confidence=conf["boundary"],
            recurrence_confidence=conf["recurrence"],
            question_type_confidence=conf["question_type"],
            specificity_confidence=conf["specificity"],
            future_language_confidence=conf["future_language"],
            agreement_confidence=conf["agreement"],
            semantic_confidence=sem_conf,
        )

    async def analyze_turn_semantic(
        self,
        utterance: NormalizedUtterance,
        context_history: List[NormalizedUtterance],
        playbook_objections: Optional[List[Any]] = None,
        call_metadata: Optional[CallMetadata] = None,
    ) -> SemanticFeatureSnapshot:
        # 1. Deterministic boundary filter runs first (zero false negatives)
        boundary_score = self.classify_boundary_deterministic(utterance.text)
        pref_type, pref_conf, pref_details = self.detect_contact_preference(utterance.text)
        if boundary_score >= 0.8:
            pref_type = "none"
            pref_conf = 0.0
            pref_details = None
        rec_id, rec_type, rec_count = self.detect_semantic_recurrence(
            utterance.text, context_history, playbook_objections
        )

        timeout = (
            call_metadata.semantic_timeout_sec
            if call_metadata and call_metadata.semantic_timeout_sec is not None
            else self.timeout_seconds
        )

        # Short fragment fast-path:
        # Avoid firing an expensive/slow LLM call on 1-2 word breathing fragments or simple nods (e.g., "case.", "So", "yeah")
        # unless they contain explicit boundary language, soft contact preferences, or terminal question marks.
        words = utterance.text.strip().split()
        is_short_fragment = len(words) < 3 and not utterance.text.strip().endswith("?")
        if is_short_fragment and boundary_score == 0.0 and rec_type == "none" and pref_type == "none":
            return self.analyze_deterministic_heuristic(
                utterance, context_history, mode="heuristic_offline", is_bypass=True
            )

        api_key_groq = os.getenv("GROQ_API_KEY")
        if not api_key_groq or api_key_groq.startswith("mock_"):
            return self.analyze_deterministic_heuristic(
                utterance, context_history, mode="heuristic_offline"
            )

        prompt = (
            f"You are an expert sales conversational linguist. Analyze this speaker turn.\n\n"
            f"Context (last 2 turns):\n"
            + "\n".join(f"- {u.speaker_id}: {u.text}" for u in context_history[-2:])
            + f"\n\nCurrent turn ({utterance.speaker_id}): \"{utterance.text}\"\n\n"
            f"Classify into valid JSON with these exact fields:\n"
            f"- question_type: 'evaluation' | 'transactional' | 'clarifying' | 'hostile' | 'rhetorical' | 'none' (STRICT: MUST be 'none' unless this turn explicitly asks a question. Declarative statements, demands, objections, and stop-contact requests like 'please don't contact me' are NOT questions and MUST be 'none')\n"
            f"- specificity_score: float 0.0 to 1.0 (dates, dollar amounts, named entities, hard numbers)\n"
            f"- future_language_score: float 0.0 to 1.0 (operational future commitment vs vague hypotheticals)\n"
            f"- agreement_score: float 0.0 to 1.0 (substantive meeting/pricing commitment ~0.8-1.0; polite nod like 'yeah' ~0.2-0.3)\n"
            f"- boundary_score: float 0.0 or 1.0 (STRICT: 1.0 ONLY for explicit stop-contact, DNC, existing broker representation, or legal threats. Exploring options, general hesitation, or soft cadence/channel preferences like 'don't call so often', 'email instead' MUST be 0.0)\n\n"
            f"Output ONLY raw JSON."
        )

        try:
            if self.groq_client is not None:
                client = self.groq_client
            else:
                import groq
                client = groq.Groq(api_key=api_key_groq, timeout=timeout)

            def _call_groq():
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 150,
                }
                if "qwen" in self.model.lower():
                    kwargs["reasoning_effort"] = "none"
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()

            raw_resp = await asyncio.wait_for(
                asyncio.to_thread(_call_groq),
                timeout=timeout,
            )
            if "<think>" in raw_resp and "</think>" in raw_resp:
                raw_resp = raw_resp.split("</think>")[-1].strip()
            raw_resp = raw_resp.strip("`").replace("json\n", "").strip()

            parsed = json.loads(raw_resp)

            # Boundary detection is high-stakes compliance (zero false negatives from regex,
            # zero false positives from LLM hallucinations).
            # A deterministic regex match always produces 1.0.
            # An LLM output is only accepted as a boundary if >= 0.80, and only when not a soft preference.
            llm_boundary = float(parsed.get("boundary_score", 0.0))
            if pref_type != "none":
                effective_boundary = boundary_score
            else:
                effective_boundary = 1.0 if (boundary_score >= 0.8 or llm_boundary >= 0.8) else 0.0
            conf = CONFIDENCE_BY_FEATURE["llm"]

            raw_q = parsed.get("question_type", "none")
            is_q = (
                utterance.text.strip().endswith("?")
                or any(utterance.text.strip().lower().startswith(w) for w in ["what", "how", "why", "who", "when", "where", "which", "can", "could", "would", "should", "is", "are", "do", "does", "did", "haven't", "hasn't", "isn't", "aren't"])
            )
            validated_q_type = raw_q if (is_q and effective_boundary < 0.8) else "none"

            return SemanticFeatureSnapshot(
                utterance_id=utterance.utterance_id,
                call_sid=utterance.call_sid,
                speaker_id=utterance.speaker_id,
                extraction_mode="llm",
                recurrence_id=rec_id,
                recurrence_type=rec_type,
                recurrence_count=rec_count,
                question_type=validated_q_type,
                specificity_score=min(1.0, max(0.0, float(parsed.get("specificity_score", 0.0)))),
                future_language_score=min(1.0, max(0.0, float(parsed.get("future_language_score", 0.0)))),
                boundary_score=min(1.0, max(0.0, effective_boundary)),
                contact_preference=pref_type,
                contact_preference_confidence=pref_conf,
                contact_preference_details=pref_details,
                agreement_score=min(1.0, max(0.0, float(parsed.get("agreement_score", 0.0)))),
                filler_score=None,
                is_filler_available=False,
                boundary_confidence=conf["boundary"],
                recurrence_confidence=conf["recurrence"],
                question_type_confidence=conf["question_type"],
                specificity_confidence=conf["specificity"],
                future_language_confidence=conf["future_language"],
                agreement_confidence=conf["agreement"],
                semantic_confidence=round(min(0.95, float(utterance.asr_confidence)), 2),
            )
        except asyncio.TimeoutError:
            LOGGER.warning("Semantic LLM extraction timed out after %.2fs; falling back to heuristic", timeout)
            return self.analyze_deterministic_heuristic(
                utterance, context_history, mode="heuristic_timeout"
            )
        except Exception as exc:
            LOGGER.warning("Semantic LLM extraction failed (%s); falling back to heuristic", exc)
            return self.analyze_deterministic_heuristic(
                utterance, context_history, mode="heuristic_error"
            )
