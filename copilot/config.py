from __future__ import annotations

import os
from dataclasses import dataclass


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    deepgram_api_key: str
    groq_api_key: str
    host: str = "127.0.0.1"
    port: int = 5000
    llm_model: str = "llama-3.1-8b-instant"
    suggestion_interval_ms: int = 250
    audio_queue_size: int = 32
    transcript_window: int = 12

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            deepgram_api_key=_required("DEEPGRAM_API_KEY"),
            groq_api_key=_required("GROQ_API_KEY"),
            host=os.getenv("COPILOT_HOST", "127.0.0.1"),
            port=int(os.getenv("COPILOT_PORT", "5000")),
            llm_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            suggestion_interval_ms=int(os.getenv("SUGGESTION_INTERVAL_MS", "250")),
            audio_queue_size=int(os.getenv("AUDIO_QUEUE_SIZE", "32")),
            transcript_window=int(os.getenv("TRANSCRIPT_WINDOW", "12")),
        )
