import sys
sys.path.insert(0, ".")
import re
from copilot.behavioral_semantic import STOP_CONTACT_PATTERNS

PROPOSED_PATTERNS = [
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
    r"\bnot\s+interested\b.*?\bdo\s+not\b",
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

boundary_phrases = [
    "I'd rather you didn't reach out for a while.",
    "Please don't contact me about this again.",
    "We're actually working with someone already.",
    "I prefer not to be contacted anymore.",
    "Please leave me alone.",
    "We already signed with a realtor last week.",
    "Do not reach out to this number.",
    "I'd appreciate it if you don't call this number again.",
    "take my number off your list",
    "We have an agent helping us.",
    "I'm working with an agent right now.",
    "Stop reaching out please.",
    "Put me on the do not call list.",
]

benign_phrases = [
    "So we have been looking at a few different options for a while now, honestly.",
    "We are not in rush exactly, but our lease is up in four months.",
    "I'm not sure if now is the right time for us.",
    "We're looking for something with three bedrooms and a nice yard.",
    "The budget is around 450k, maybe up to 500k if it's really turnkey.",
    "What kind of commission structure do you typically work with?",
    "Yeah that sounds reasonable, let's talk next Tuesday.",
    "We were burned before by another deal that fell through on financing.",
]

print("=== TESTING BOUNDARY PHRASES (MUST ALL BE TRUE) ===")
all_boundaries_matched = True
for p in boundary_phrases:
    matches = [pat for pat in PROPOSED_PATTERNS if re.search(pat, p, re.I)]
    matched = bool(matches)
    if not matched:
        all_boundaries_matched = False
    print(f"[{'PASS' if matched else 'FAIL'}] '{p}' -> {matches}")

print("\n=== TESTING BENIGN PHRASES (MUST ALL BE FALSE) ===")
all_benign_passed = True
for p in benign_phrases:
    matches = [pat for pat in PROPOSED_PATTERNS if re.search(pat, p, re.I)]
    matched = bool(matches)
    if matched:
        all_benign_passed = False
    print(f"[{'FAIL' if matched else 'PASS'}] '{p}' -> {matches}")

print(f"\nSummary: All Boundaries Matched: {all_boundaries_matched}, All Benign Clean: {all_benign_passed}")

