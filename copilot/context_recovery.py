from __future__ import annotations

import re
from collections.abc import Iterable

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
CONTINUATION = re.compile(
    r"^(?:and|also|but|because|so|then|plus|or|especially|specifically|"
    r"in detail|more detail|for example|about that|regarding that|what about|how about)\b",
    re.IGNORECASE,
)
REFERENCE = re.compile(
    r"\b(?:it|its|that|this|these|those|they|them|their|he|she|there|such)\b",
    re.IGNORECASE,
)
GENERIC = {
    "a", "an", "and", "are", "about", "can", "could", "detail", "do", "does",
    "explain", "for", "how", "i", "in", "is", "it", "me", "more", "of", "please",
    "tell", "that", "the", "then", "this", "to", "what", "when", "where", "which",
    "who", "why", "will", "work", "works", "working", "would", "you", "your",
}


def _words(text: str) -> list[str]:
    return [word.casefold() for word in WORD.findall(text)]


def _has_subject(text: str) -> bool:
    return any(word not in GENERIC and len(word) > 2 for word in _words(text))


def needs_context(text: str) -> bool:
    """Return whether a transcript fragment is unsafe as a standalone query."""
    stripped = text.strip()
    words = _words(stripped)
    if not words:
        return True
    if CONTINUATION.search(stripped):
        return True
    if stripped[:1].islower() and len(words) <= 8:
        return True
    if REFERENCE.search(stripped) and not _has_subject(stripped):
        return True
    return len(words) <= 3 and not _has_subject(stripped)


def recover_query(
    latest: str,
    prior_client_turns: Iterable[str],
    *,
    max_turns: int = 4,
    max_chars: int = 1000,
) -> str:
    """Prepend only enough recent client speech to resolve the latest fragment."""
    query = latest.strip()
    if not needs_context(query):
        return query
    prior = [turn.strip() for turn in prior_client_turns if turn and turn.strip()]
    for used, turn in enumerate(reversed(prior), start=1):
        if used > max_turns:
            break
        candidate = f"{turn} {query}".strip()
        if len(candidate) > max_chars:
            break
        query = candidate
        if _has_subject(query) and not CONTINUATION.search(query):
            break
    return query
