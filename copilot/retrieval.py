from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

WORD = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_'-]+")


def tokens(text: str) -> set[str]:
    return {word.casefold() for word in WORD.findall(text) if len(word) > 2}


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


class LocalKnowledgeBase:
    """Small, dependency-free retrieval layer for local MVP knowledge files."""

    def __init__(self, directory: str | Path = "knowledge", chunk_chars: int = 1200):
        self.directory = Path(directory)
        self.chunk_chars = chunk_chars
        self._chunks: list[tuple[str, str, set[str]]] = []
        self.reload()

    def reload(self) -> None:
        self._chunks.clear()
        if not self.directory.exists():
            return
        for path in sorted(self.directory.rglob("*")):
            if path.suffix.casefold() not in {".txt", ".md"} or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            current = ""
            for paragraph in paragraphs:
                if current and len(current) + len(paragraph) > self.chunk_chars:
                    self._add(path, current)
                    current = ""
                current = f"{current}\n\n{paragraph}".strip()
            if current:
                self._add(path, current)

    def _add(self, path: Path, text: str) -> None:
        self._chunks.append((str(path.as_posix()), text, tokens(text)))

    def search(self, query: str, limit: int = 3, tenant_id: Optional[str] = None) -> list[RetrievedChunk]:
        query_tokens = tokens(query)
        if not query_tokens:
            return []
        results: list[RetrievedChunk] = []
        for source, text, document_tokens in self._chunks:
            if tenant_id:
                path_parts = Path(source).parts
                # For Windows paths, source might be e.g. "knowledge/tenant_id/..."
                # Let's check if the tenant_id folder matches
                if len(path_parts) > 1 and path_parts[1] != tenant_id:
                    continue
            overlap = query_tokens & document_tokens
            if not overlap:
                continue
            precision = len(overlap) / math.sqrt(max(1, len(document_tokens)))
            coverage = len(overlap) / len(query_tokens)
            results.append(RetrievedChunk(source, text, precision + coverage))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]
