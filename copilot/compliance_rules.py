"""Canonical Compliance Gate Rules & Heuristic Filters (Doc 01 Stage 2 & Doc 02 §8).

DISCLAIMER & GOVERNANCE:
-------------------------
The keyword and regex patterns in this module represent a conservative heuristic
filter to catch obvious compliance risks early at the ingestion and retrieval gates.
This module DOES NOT constitute formal legal analysis or comprehensive statutory review.
The client's compliance officer and legal counsel must review, approve, and own
the authoritative list of prohibited terms, Fair Housing keywords, and RESPA guidelines.
Antigravity and the engineering team provide this as an operational technical guardrail,
not as a substitute for professional legal advice.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ==============================================================================
# Conservative Compliance Rule Definitions (Doc 01 Stage 2)
# ==============================================================================

COMPLIANCE_PATTERNS: List[Tuple[str, str, str]] = [
    # (Regex pattern, Violation Category, Specific Regulatory Reason)
    (
        r"\b(?:guarantee(?:s|d)?|promise(?:s|d)?)\b.*?\b(?:profit|return|roi|investment|income|gain|money\s+back)\b|\b(?:100%\s+profit|guarantee(?:s|d)?\s+100%|guaranteed\s+(?:sale|buyer|closing|profit|return|roi))\b",
        "prohibited_guarantee",
        "Guaranteed investment return promises or performance guarantees violate real estate advertising and licensing regulations.",
    ),
    (
        r"\b(?:exclusive\s+(?:for\s+adults|neighborhood\s+without\s+children)|whites?\s+only|no\s+families|no\s+children|christian\s+only|safe\s+neighborhood\s+for\s+whites)\b",
        "fair_housing_steering",
        "Discriminatory steering based on familial status, race, or religion violates the Fair Housing Act (FHA).",
    ),
    (
        r"\b(?:cash\s+kickback|rebate\s+under\s+the\s+table|undisclosed\s+fee|unlicensed\s+referral\s+fee)\b",
        "respa_violation",
        "Undisclosed kickbacks and unlawful referral fee agreements violate the Real Estate Settlement Procedures Act (RESPA).",
    ),
    (
        r"\b(?:guaranteed\s+tax\s+shelter|fiduciary\s+guarantee|promise\s+no\s+tax\s+liability)\b",
        "unauthorized_legal_tax_advice",
        "Unlicensed legal or tax outcome guarantees violate professional practice boundaries.",
    ),
]

# ==============================================================================
# Prompt Injection Heuristic Patterns (Bonus Defense layer over Structural Wrapper)
# ==============================================================================

PROMPT_INJECTION_PATTERNS: List[str] = [
    r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+(?:rules|instructions|directives|prompts)\b",
    r"\bdisregard\s+(?:all\s+)?(?:previous|prior|system)\s+(?:rules|instructions|directives)\b",
    r"\bsystem\s+override\b",
    r"\bjailbreak\b",
    r"\byou\s+are\s+now\s+in\s+developer\s+mode\b",
    r"\bDAN\s+mode\b",
    r"\bassistant\s*:\s*(?:you\s+must|say|output)\b",
    r"\[UNTRUSTED_REFERENCE_DATA_END\]",
    r"\[UNTRUSTED_REFERENCE_DATA_START\]",
]


def evaluate_content_compliance(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Evaluates text against conservative real estate compliance rules (Doc 01 Stage 2).

    Returns:
        (is_compliant, violation_category, details)
        - is_compliant: True if text passed all conservative checks, False if a violation was flagged.
    """
    if not text:
        return (True, None, None)

    for pattern, category, details in COMPLIANCE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return (False, category, details)

    return (True, None, None)


def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """Evaluates text for overt prompt injection attempts or delimiter breakout payloads.

    Returns:
        (has_injection, matched_pattern)
    """
    if not text:
        return (False, None)

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return (True, pattern)

    return (False, None)


def classify_attested_claim(text: str) -> Tuple[bool, str]:
    """Classifies whether an attested claim is an allowable concrete fact or prohibited high-risk.

    Fails closed: if the claim cannot be definitively verified as a concrete checkable fact
    (years of experience, geographic territory, or specific named past transactions),
    it is categorized as ineligible for self-attestation across all knowledge classes.
    """
    if not text or not text.strip():
        return (False, "Empty text cannot be attested.")

    # High-Risk / Prohibited markers (Guarantees, comparative stats, awards, certifications, legal status)
    prohibited_markers = [
        (r"\b(?:guarantee(?:s|d)?|promise(?:s|d)?|risk[- ]free|100%\s+guaranteed)\b", "Guarantees and outcome promises require company approval."),
        (r"\b(?:\d+%\s+(?:list[- ]to[- ]sale|conversion|close\s+rate|success\s+rate)|sell\s+\d+x\s+faster|outperform(?:s|ed)?\s+(?:the\s+)?average|\d+%\s+faster|\d+%\s+more\s+money)\b", "Conversion rates and comparative statistical claims require company approval or system validation."),
        (r"\b(?:award|top\s+(?:\d+%|producer)|#1\s+agent|number\s+one\s+agent|agent\s+of\s+the\s+year|million\s+dollar\s+club|realtor\s+of\s+the\s+year|best\s+agent|premier\s+agent)\b", "Awards, rankings, and superlatives cannot be self-attested."),
        (r"\b(?:certified|certification|CRS|ABR|GRI|SRES|SRS|broker[- ]associate\s+certified|luxury\s+specialist)\b", "Professional certifications cannot be self-attested."),
        (r"\b(?:fiduciary\s+guarantee|licensed\s+attorney|escrow\s+holder|power\s+of\s+attorney|title\s+guaranty)\b", "Legal and compliance status claims cannot be self-attested."),
    ]
    for pattern, reason in prohibited_markers:
        if re.search(pattern, text, re.IGNORECASE):
            return (False, reason)

    # Allowed Concrete Checkable Markers
    concrete_patterns = [
        # 1. Years of experience / tenure
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty)\s+years?\s+(?:of\s+experience|serving|in\s+real\s+estate|in\s+the\s+business|licensed)\b|\ba\s+decade\s+(?:of\s+experience|in\s+real\s+estate)\b",
        # 2. Geographic territory / areas served
        r"\b(?:serving|territory|areas?\s+served|based\s+in|specializing\s+in\s+the|focusing\s+on)\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\b",
        # 3. Specific named past transactions / volume without uncheckable superlatives
        r"\b(?:closed|sold|represented|completed|handled)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+(?:homes|properties|transactions|listings|sales|deals|residential)\b",
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+(?:residential\s+)?(?:homes|transactions|sales|properties)\s+(?:closed|sold|completed)\b",
    ]

    for pat in concrete_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return (True, "Concrete checkable personal fact.")

    # FAIL CLOSED: If we can't tell, withhold!
    return (False, "Claim does not meet concrete checkable fact criteria (tenure, territory, or specific past transactions). Fails closed.")
