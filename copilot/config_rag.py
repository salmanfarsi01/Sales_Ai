"""Updated configuration for RAG and multi-tenant support."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True, slots=True)
class RAGSettings:
    """RAG and vector database configuration."""
    pinecone_api_key: str
    openai_api_key: str
    rag_enabled: bool = True
    admin_mode: bool = False
    chunk_size: int = 1500
    chunk_overlap: int = 150
    search_top_k: int = 5
    min_score_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class Settings:
    """Complete application settings."""
    # Transcription
    deepgram_api_key: str
    
    # LLM
    groq_api_key: str
    
    # Server
    host: str = "127.0.0.1"
    port: int = 5000
    
    # LLM Model
    llm_model: str = "llama-3.1-8b-instant"
    
    # Real-time settings
    suggestion_interval_ms: int = 250
    audio_queue_size: int = 32
    transcript_window: int = 12
    
    # RAG Settings
    rag: RAGSettings = None
    
    # Multi-tenant
    default_tenant_id: str = "default"

    @classmethod
    def from_env(cls) -> "Settings":
        # Basic settings (required)
        rag_settings = RAGSettings(
            pinecone_api_key=_required("PINECONE_API_KEY"),
            openai_api_key=_required("OPENAI_API_KEY"),
            rag_enabled=_optional("RAG_ENABLED", "true").lower() == "true",
            admin_mode=_optional("ADMIN_MODE", "false").lower() == "true",
            chunk_size=int(_optional("RAG_CHUNK_SIZE", "1500")),
            chunk_overlap=int(_optional("RAG_CHUNK_OVERLAP", "150")),
            search_top_k=int(_optional("RAG_SEARCH_TOP_K", "5")),
            min_score_threshold=float(_optional("RAG_MIN_SCORE", "0.5")),
        )
        
        return cls(
            deepgram_api_key=_required("DEEPGRAM_API_KEY"),
            groq_api_key=_required("GROQ_API_KEY"),
            host=_optional("COPILOT_HOST", "127.0.0.1"),
            port=int(_optional("COPILOT_PORT", "5000")),
            llm_model=_optional("GROQ_MODEL", "llama-3.1-8b-instant"),
            suggestion_interval_ms=int(_optional("SUGGESTION_INTERVAL_MS", "250")),
            audio_queue_size=int(_optional("AUDIO_QUEUE_SIZE", "32")),
            transcript_window=int(_optional("TRANSCRIPT_WINDOW", "12")),
            rag=rag_settings,
            default_tenant_id=_optional("DEFAULT_TENANT_ID", "default"),
        )
