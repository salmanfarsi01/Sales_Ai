"""AI Voice Calibration Service.

Provides an 8-round sales voice calibration system:
1. Dynamically generates 8-round business scenarios, prospect questions, and teleprompt scripts via Groq LLM.
2. Synthesizes voice audio on-demand per round via ElevenLabs (with fallback to OpenAI TTS).
3. Transcribes user voice responses via Deepgram / Groq Whisper.
4. Evaluates user repetition and speech delivery on 6 core dimensions:
   - Word choice
   - Pacing
   - Sentiment
   - Tone/Emphasis
   - Pause/Filters
   - Energy/Inflection
5. Generates comprehensive calibration reports matching the UI results dashboard.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import time
import uuid
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("copilot.calibration")
STATIC_DIR = Path(__file__).resolve().parent.parent / "web"
AUDIO_CACHE_DIR = Path(__file__).resolve().parent.parent / "calibration_audio"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_audio_duration_seconds(
    audio_bytes: Optional[bytes],
    mime_type: str = "audio/webm",
    fallback_seconds: float = 0.0,
) -> float:
    """Extract audio duration server-side when WAV or complete EBML header is present;
    otherwise validates client-supplied timer against physical byte-rate bounds (800-45,000 B/s) as an outlier filter.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return max(round(fallback_seconds, 2), 0.0) if fallback_seconds > 0 else 0.0

    # 1. RIFF / WAV exact frame header parsing
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / float(rate), 2)
        except Exception:
            pass

    # 2. WebM / Matroska EBML Duration element parser (0x4489)
    try:
        idx = audio_bytes.find(b"\x44\x89")
        if idx != -1 and idx + 7 <= len(audio_bytes):
            import struct
            len_byte = audio_bytes[idx + 2]
            if len_byte == 0x84 and idx + 7 <= len(audio_bytes):
                val = struct.unpack(">f", audio_bytes[idx + 3 : idx + 7])[0]
                if 0.1 <= val <= 3600:
                    return round(val, 2)
                elif val > 3600:
                    return round(val / 1000.0, 2)
            elif len_byte == 0x88 and idx + 11 <= len(audio_bytes):
                val = struct.unpack(">d", audio_bytes[idx + 3 : idx + 11])[0]
                if 0.1 <= val <= 3600:
                    return round(val, 2)
                elif val > 3600:
                    return round(val / 1000.0, 2)
    except Exception:
        pass

    # 3. If client provided high-resolution timer duration, validate against byte sanity (1,000 - 45,000 B/s)
    bytes_len = len(audio_bytes)
    if fallback_seconds >= 0.5:
        effective_bps = bytes_len / fallback_seconds
        if 800 <= effective_bps <= 45000:
            return round(fallback_seconds, 2)

    # 4. WebM/Opus stream estimation (~48 kbps = 6,000 bytes/sec)
    estimated_sec = bytes_len / 6000.0
    if 0.5 <= estimated_sec <= 120.0:
        return round(estimated_sec, 2)

    return max(round(fallback_seconds, 2), 0.0) if fallback_seconds > 0 else 5.0

# Curated ElevenLabs Voice Pool & Sales Personas
VOICE_PERSONAS = [
    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "title": "VP of Engineering", "style": "Calm & Professional"},
    {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi", "title": "Head of Procurement", "style": "Energetic & Direct"},
    {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "title": "Director of Customer Success", "style": "Warm & Consultative"},
    {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "title": "VP of Product", "style": "Clear & Confident"},
    {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli", "title": "Enterprise Operations Lead", "style": "Pleasant & Detail-Oriented"},
    {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh", "title": "Chief Information Officer", "style": "Authoritative Decision Maker"},
    {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold", "title": "Chief Financial Officer", "style": "Analytical & ROI-Focused"},
    {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "title": "Chief Technology Officer", "style": "Crisp & Visionary"},
]

SALES_STAGES = [
    {"round": 1, "stage": "Discovery & Onboarding", "focus": "Time to value, setup complexity, and dedicated kickoff support."},
    {"round": 2, "stage": "Security & Compliance", "focus": "SOC 2 Type II, data isolation, encryption, and GDPR compliance."},
    {"round": 3, "stage": "Competitive Differentiator", "focus": "Switching rationale versus established legacy vendors and unique AI capabilities."},
    {"round": 4, "stage": "Pricing Pushback & ROI", "focus": "Justifying premium pricing, payback period, and 4x measurable ROI."},
    {"round": 5, "stage": "Adoption & Change Management", "focus": "Rep adoption friction, ease of use, and workflow integration."},
    {"round": 6, "stage": "Executive / CFO Justification", "focus": "High-level metrics for board signoff and sales velocity impact."},
    {"round": 7, "stage": "Contract Terms & Flexibility", "focus": "Structured pilot milestones, SLA guarantees, and flexible quarterly terms."},
    {"round": 8, "stage": "Closing & Next Steps", "focus": "Clear kickoff timeline, immediate workspace provisioning, and launch schedule."},
]

INDUSTRIES = [
    "B2B Enterprise AI & Cloud Software",
    "FinTech & Automated Payments Platform",
    "Healthcare & Life Sciences Operations",
    "Cybersecurity & Threat Intelligence",
    "Supply Chain & Logistics Automation",
    "Modern CRM & Sales Revenue Intelligence",
]


def _clean_json_text(text: str) -> str:
    """Extract valid JSON from potential markdown code fences."""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else cleaned


# ==========================================
# Data Models & Style Profiling Definitions
# ==========================================

CALIBRATION_FORMULA_VERSION = "v2.0-style-profiling"

DEFAULT_PROFILING_WEIGHTS: dict[str, float] = {
    "speaking_pace": 0.25,        # Words per minute tempo
    "response_timing": 0.15,      # Prompt-to-speech readiness
    "sentence_handling": 0.15,    # Sentence conciseness vs. narrative
    "vocabulary_complexity": 0.15,# Lexical diversity & density
    "pause_pattern": 0.15,        # Pause cadence & filler density
    "energy_dynamics": 0.15,      # Vocal energy dynamic range
}

PROMPTING_LEVELS: dict[str, dict[str, Any]] = {
    "direct": {
        "level_id": "direct",
        "name": "Prompting Level: Direct",
        "badge": "Direct",
        "min_score": 75,
        "description": "Fast tempo, concise responses. Teleprompter produces short, punchy prompts (<15 words) with minimal pause buffering.",
        "directive": "Shorter prompts, punchier phrasing, rapid lead time, action-oriented assertions.",
    },
    "balanced": {
        "level_id": "balanced",
        "name": "Prompting Level: Balanced",
        "badge": "Balanced",
        "min_score": 50,
        "description": "Conversational tempo with natural phrasing. Teleprompter produces balanced prompts (15-22 words) with natural conversational transitions.",
        "directive": "Moderate sentence length, natural conversational transitions, standard lead time.",
    },
    "layered": {
        "level_id": "layered",
        "name": "Prompting Level: Layered",
        "badge": "Layered",
        "min_score": 0,
        "description": "Deliberate, thoughtful speaking style. Teleprompter produces digestible, chunked multi-step cues with extended lead time.",
        "directive": "Digestible phrase chunks, longer lead time, pause-friendly structure, step-by-step guidance.",
    },
}


def resolve_prompting_level(score: int) -> dict[str, Any]:
    """Map composite style score (0-100) to prompting level profile."""
    if score >= 75:
        return PROMPTING_LEVELS["direct"]
    elif score >= 50:
        return PROMPTING_LEVELS["balanced"]
    else:
        return PROMPTING_LEVELS["layered"]


class RoundScoreBreakdown(BaseModel):
    # Core Style Dimensions (v2.0)
    speaking_pace: int = Field(default=80, ge=0, le=100, description="Speaking pace tempo index (WPM normalized)")
    response_timing: int = Field(default=75, ge=0, le=100, description="Response readiness & timing index")
    sentence_handling: int = Field(default=75, ge=0, le=100, description="Sentence brevity & structure index")
    vocabulary_complexity: int = Field(default=80, ge=0, le=100, description="Lexical diversity & complexity")
    pause_pattern: int = Field(default=85, ge=0, le=100, description="Pause cadence and rhythm index")
    energy_dynamics: int = Field(default=75, ge=0, le=100, description="Vocal energy dynamic range index")

    # Backwards-compatible aliases for legacy reporting & UI
    word_choice: int = Field(default=80, ge=0, le=100, description="Legacy word choice alias")
    pacing: int = Field(default=80, ge=0, le=100, description="Legacy pacing alias")
    sentiment: int = Field(default=75, ge=0, le=100, description="Legacy sentiment alias")
    tone_emphasis: int = Field(default=75, ge=0, le=100, description="Legacy tone alias")
    pause_filters: int = Field(default=85, ge=0, le=100, description="Legacy pause filters alias")
    energy_inflection: int = Field(default=75, ge=0, le=100, description="Legacy energy alias")


class RoundEvaluation(BaseModel):
    round_number: int
    stage: str
    overall_score: int
    score_breakdown: RoundScoreBreakdown
    transcribed_text: str
    target_teleprompt: str
    feedback: str
    strengths: list[str] = []
    improvements: list[str] = []
    detected_fillers: list[str] = []
    wpm: Optional[int] = None
    confidence: float = 1.0
    is_excluded: bool = False
    style_traits: list[str] = []
    prompting_adaptation: str = ""


class RoundData(BaseModel):
    round_number: int
    stage: str
    scenario_type: str
    persona_name: str
    persona_title: str
    voice_id: str
    voice_name: str
    question_text: str
    teleprompt_text: str
    audio_id: Optional[str] = None
    audio_url: Optional[str] = None
    evaluation: Optional[RoundEvaluation] = None
    completed: bool = False
    duration_seconds: float = 0.0


class CalibrationSummaryReport(BaseModel):
    session_id: str
    overall_score: int
    prompting_level: dict[str, Any] = Field(default_factory=dict)
    formula_version: str = CALIBRATION_FORMULA_VERSION
    formula_weights: dict[str, float] = Field(default_factory=dict)
    rounds_completed: int
    rounds_profiled: int = 8
    total_rounds: int = 8
    total_time_seconds: float
    total_time_formatted: str
    completed_on: str
    status_message: str
    round_performance: list[dict[str, Any]]
    score_breakdown: RoundScoreBreakdown
    what_this_means: list[str]
    what_is_next: list[str]


# ==========================================
# Audio & ElevenLabs Integration
# ==========================================

class ElevenLabsTTSClient:
    """Handles on-demand ElevenLabs TTS synthesis with hash-based caching and fallback to OpenAI TTS."""

    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=15.0) if self.openai_api_key else None

    async def generate_speech(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> tuple[bytes, str]:
        """Synthesize speech audio for the dynamic question text with hash-based local caching.

        Returns (audio_bytes, format_mime).
        """
        clean_text = text.strip()

        # 1. Check local hash-based cache in AUDIO_CACHE_DIR (including full voice parameter footprint)
        model_id = "eleven_multilingual_v2"
        stability = 0.55
        similarity_boost = 0.8
        style = 0.15
        use_speaker_boost = True

        cache_key = hashlib.sha256(
            f"{model_id}:{voice_id}:{stability}:{similarity_boost}:{style}:{use_speaker_boost}:{clean_text}".encode("utf-8")
        ).hexdigest()
        cache_path = AUDIO_CACHE_DIR / f"tts_{cache_key}.mp3"

        if cache_path.exists() and cache_path.stat().st_size > 200:
            try:
                LOGGER.info("TTS Cache hit: %s (voice: %s)", cache_path.name, voice_id)
                audio_bytes = await asyncio.to_thread(cache_path.read_bytes)
                return audio_bytes, "audio/mpeg"
            except Exception as exc:
                LOGGER.warning("TTS Cache read error: %s", exc)

        # 2. Try ElevenLabs first if API key is provided
        if self.elevenlabs_api_key:
            try:
                LOGGER.info("Generating ElevenLabs speech on-demand for: %s...", clean_text[:40])
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {
                    "xi-api-key": self.elevenlabs_api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                }
                payload = {
                    "text": clean_text,
                    "model_id": model_id,
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                        "style": style,
                        "use_speaker_boost": use_speaker_boost,
                    },
                }
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200 and len(response.content) > 200:
                        audio_bytes = response.content
                        await asyncio.to_thread(cache_path.write_bytes, audio_bytes)
                        return audio_bytes, "audio/mpeg"
                    else:
                        LOGGER.warning(
                            "ElevenLabs API returned %d: %s. Falling back to alternative TTS.",
                            response.status_code,
                            response.text,
                        )
            except Exception as exc:
                LOGGER.warning("ElevenLabs synthesis error: %s", exc)

        # 3. Fallback to OpenAI TTS if available
        if self.openai_client:
            try:
                LOGGER.info("Generating OpenAI TTS speech fallback...")
                voice_map = {
                    "21m00Tcm4TlvDq8ikWAM": "alloy",
                    "AZnzlk1XvdvUeBnXmlld": "nova",
                    "EXAVITQu4vr4xnSDxMaL": "shimmer",
                    "ErXwobaYiN019PkySvjV": "echo",
                    "MF3mGyEYCl7XYWbV9V6O": "fable",
                    "TxGEqnHWrfWFTfGW9XjX": "onyx",
                    "VR6AewLTigWG4xSOukaG": "echo",
                    "pNInz6obpgDQGcFmaJgB": "alloy",
                }
                selected_voice = voice_map.get(voice_id, "alloy")
                openai_cache_key = hashlib.sha256(f"openai:tts-1:{selected_voice}:{clean_text}".encode("utf-8")).hexdigest()
                openai_cache_path = AUDIO_CACHE_DIR / f"tts_{openai_cache_key}.mp3"

                if openai_cache_path.exists() and openai_cache_path.stat().st_size > 200:
                    audio_bytes = await asyncio.to_thread(openai_cache_path.read_bytes)
                    return audio_bytes, "audio/mpeg"

                def _synth():
                    res = self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice=selected_voice,
                        input=clean_text,
                        timeout=15.0,
                    )
                    return res.content

                audio_bytes = await asyncio.to_thread(_synth)
                if len(audio_bytes) > 200:
                    await asyncio.to_thread(openai_cache_path.write_bytes, audio_bytes)
                    return audio_bytes, "audio/mpeg"
            except Exception as exc:
                LOGGER.warning("OpenAI TTS error: %s", exc)

        # 4. Lightweight synthetic tone placeholder if no external audio key
        return self._generate_synthetic_placeholder(clean_text), "audio/wav"

    def _generate_synthetic_placeholder(self, text: str) -> bytes:
        """Create a valid RIFF WAV audio placeholder."""
        import struct

        sample_rate = 8000
        duration_sec = min(max(len(text.split()) * 0.35, 1.5), 5.0)
        num_samples = int(sample_rate * duration_sec)
        byte_rate = sample_rate * 2
        block_align = 2
        bits_per_sample = 16
        data_size = num_samples * 2
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            chunk_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            1,  # Mono
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )

        samples = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            val = int(math.sin(2 * math.pi * 440 * t) * 1500 + math.sin(2 * math.pi * 880 * t) * 800)
            samples.extend(struct.pack("<h", max(-32767, min(32767, val))))

        return bytes(header + samples)


# ==========================================
# Speech-To-Text Engine
# ==========================================

class SpeechToTextEngine:
    """Transcribes user speech using Deepgram or Groq Whisper (with non-blocking in-memory audio)."""

    def __init__(
        self,
        deepgram_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.deepgram_api_key = deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key, timeout=15.0) if self.groq_api_key else None

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """Transcribe user audio to text."""
        # 1. Deepgram STT
        if self.deepgram_api_key:
            try:
                LOGGER.info("Transcribing audio via Deepgram...")
                url = "https://api.deepgram.com/v1/listen?punctuate=true&model=nova-2&language=en"
                headers = {
                    "Authorization": f"Token {self.deepgram_api_key}",
                    "Content-Type": mime_type,
                }
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(url, headers=headers, content=audio_bytes)
                    if resp.status_code == 200:
                        data = resp.json()
                        transcript = (
                            data.get("results", {})
                            .get("channels", [{}])[0]
                            .get("alternatives", [{}])[0]
                            .get("transcript", "")
                        )
                        if transcript.strip():
                            return transcript.strip()
            except Exception as exc:
                LOGGER.warning("Deepgram STT failed: %s. Trying Groq Whisper.", exc)

        # 2. Groq Whisper fallback (In-Memory, No Blocking Disk I/O on Event Loop)
        if self.groq_client:
            try:
                LOGGER.info("Transcribing audio via Groq Whisper (in-memory)...")
                ext = "webm" if "webm" in mime_type else "wav"
                filename = f"audio.{ext}"

                def _whisper():
                    transcription = self.groq_client.audio.transcriptions.create(
                        file=(filename, io.BytesIO(audio_bytes).read()),
                        model="whisper-large-v3-turbo",
                        language="en",
                        timeout=15.0,
                    )
                    return transcription.text

                res = await asyncio.to_thread(_whisper)
                if res and res.strip():
                    return res.strip()
            except Exception as exc:
                LOGGER.warning("Groq Whisper failed: %s", exc)

        return ""


# ==========================================
# Calibration Dynamic Session Service
# ==========================================

class CalibrationService:
    """Manages dynamic 8-round calibration sessions, Groq LLM text generation, ElevenLabs TTS, and evaluation."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        deepgram_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.deepgram_api_key = deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
        self.llm_model = llm_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        self.groq_client = Groq(api_key=self.groq_api_key, timeout=15.0) if self.groq_api_key else None
        self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=15.0) if self.openai_api_key else None
        self.tts = ElevenLabsTTSClient(self.elevenlabs_api_key, self.openai_api_key)
        self.stt = SpeechToTextEngine(self.deepgram_api_key, self.groq_api_key)

        self.sessions: dict[str, dict[str, Any]] = {}
        self.audio_store: dict[str, tuple[bytes, str]] = {}
        self.profiling_weights: dict[str, float] = dict(DEFAULT_PROFILING_WEIGHTS)

    def rate_sample_quality(
        self,
        transcript: str,
        audio_bytes: Optional[bytes] = None,
        duration_seconds: float = 0.0,
    ) -> float:
        """Calculate confidence score (0.0 to 1.0). Bad audio / silence is 0.0 (excluded from profile calculation)."""
        clean = transcript.strip() if transcript else ""
        words = [w for w in clean.split() if w]

        # 1. Total silence or empty transcript
        if not clean or len(words) == 0:
            return 0.0

        # 2. Too brief (< 1.2s or fewer than 2 words)
        if duration_seconds > 0 and duration_seconds < 1.2:
            return 0.0
        if len(words) < 2:
            return 0.0

        # 3. Degraded sample (< 4 words or < 2.0s) -> down-weight
        if len(words) < 4 or (duration_seconds > 0 and duration_seconds < 2.0):
            return 0.4

        return 1.0

    def compute_speaking_pace(self, word_count: int, duration_seconds: float) -> tuple[int, int]:
        """Compute raw WPM and normalized pace score (0-100) toward prompting level."""
        wpm = int(round((word_count / max(duration_seconds, 0.5)) * 60)) if duration_seconds > 0 else 140
        # Map 90 WPM -> ~50 (Layered), 135 WPM -> ~75 (Balanced), 160+ WPM -> ~88+ (Direct)
        pace_score = min(100, max(25, int(round(wpm / 1.8))))
        return wpm, pace_score

    def compute_sentence_handling(self, transcript: str, word_count: int) -> tuple[float, int]:
        """Evaluate sentence length and structure (brevity vs narrative)."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", transcript) if s.strip()]
        avg_len = word_count / max(len(sentences), 1)
        if avg_len <= 10:
            score = 88  # Crisp, punchy clauses
        elif avg_len <= 16:
            score = 76  # Balanced conversational
        elif avg_len <= 22:
            score = 62  # Elaborative
        else:
            score = 48  # Layered narrative
        return round(avg_len, 1), score

    def compute_vocabulary_complexity(self, transcript: str, word_count: int) -> tuple[float, int]:
        """Evaluate lexical diversity (Type-Token Ratio) and word length."""
        words = [re.sub(r"[^\w]", "", w.lower()) for w in transcript.split() if w]
        if not words:
            return 0.0, 70
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        avg_chars = sum(len(w) for w in words) / len(words)
        score = min(98, max(40, int(round((ttr * 55) + (avg_chars * 7)))))
        return round(ttr, 2), score

    def compute_pause_cadence(
        self,
        transcript: str,
        duration_seconds: float,
        word_count: int,
        detected_fillers: list[str],
    ) -> int:
        """Evaluate pause frequency and delivery continuity."""
        filler_count = len(detected_fillers)
        # Moderate deduction for hesitations without failing the user
        score = max(40, min(95, 90 - (filler_count * 8)))
        return score

    def compute_energy_dynamics(self, audio_bytes: Optional[bytes], transcript: str) -> int:
        """Evaluate vocal energy dynamic range or phrasing intensity."""
        has_emphasis = any(ch in transcript for ch in ["!", "?"])
        return 82 if has_emphasis else 74

    def compute_response_timing(self, latency_ms: Optional[float] = None) -> int:
        """Evaluate response prompt-to-speech timing readiness."""
        if latency_ms is None:
            return 78
        if latency_ms <= 600:
            return 90
        elif latency_ms <= 1500:
            return 78
        else:
            return 55

    async def _call_llm_json(
        self,
        system: str,
        prompt: str,
        temperature: float = 0.7,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Unified internal helper: try Groq -> try OpenAI with timeout, to_thread, and clean JSON extraction."""
        if not self.groq_client and not self.openai_client:
            return {}

        # 1. Try Groq
        if self.groq_client:
            try:
                def _call_groq():
                    res = self.groq_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        timeout=timeout,
                    )
                    return res.choices[0].message.content or "{}"

                raw = await asyncio.to_thread(_call_groq)
                return json.loads(_clean_json_text(raw))
            except Exception as exc:
                LOGGER.warning("Groq LLM JSON call failed (%s). Attempting OpenAI fallback.", exc)

        # 2. Fallback to OpenAI
        if self.openai_client:
            try:
                def _call_openai():
                    res = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        timeout=timeout,
                    )
                    return res.choices[0].message.content or "{}"

                raw = await asyncio.to_thread(_call_openai)
                return json.loads(_clean_json_text(raw))
            except Exception as exc:
                LOGGER.warning("OpenAI LLM JSON call failed: %s", exc)

        return {}

    async def create_dynamic_session(self, user_id: str = "sales_rep_1", industry: Optional[str] = None) -> dict[str, Any]:
        """Generate a completely dynamic 8-round calibration session plan."""
        session_id = f"calib_{uuid.uuid4().hex[:12]}"
        selected_industry = industry or random.choice(INDUSTRIES)

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "industry": selected_industry,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "start_time_monotonic": time.monotonic(),
            "end_time_monotonic": None,
            "current_round": 1,
            "total_rounds": 8,
            "status": "in_progress",
            "rounds": {},  # Populated dynamically on-demand per round
            "summary_report": None,
        }
        self.sessions[session_id] = session_data
        LOGGER.info("Created dynamic 8-round calibration session: %s (Industry: %s)", session_id, selected_industry)
        return session_data

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Fetch session data or raise 404."""
        session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Calibration session not found")
        return session

    async def generate_round_voice_on_demand(self, session_id: str, round_num: int) -> dict[str, Any]:
        """Generate a dynamic question & teleprompt using Groq LLM, then synthesize ElevenLabs voice on-demand."""
        session = self.get_session(session_id)
        if round_num < 1 or round_num > 8:
            raise HTTPException(status_code=400, detail="Round number must be between 1 and 8")

        # If already generated for this round in this session, return existing
        if round_num in session["rounds"] and session["rounds"][round_num].get("audio_id"):
            return session["rounds"][round_num]

        stage_info = SALES_STAGES[round_num - 1]
        stage = stage_info["stage"]
        focus = stage_info["focus"]
        persona = VOICE_PERSONAS[(round_num - 1) % len(VOICE_PERSONAS)]
        industry = session.get("industry", "B2B Enterprise SaaS & AI")

        # Generate unique scenario dynamically using Groq LLM
        dynamic_prompt = f"""You are an elite B2B sales simulation coach generating Round {round_num} of 8.
Target Industry: {industry}
Sales Stage: {stage}
Key Focus: {focus}
Buyer Persona: {persona['name']} ({persona['title']}, {persona['style']})

Generate:
1. "question_text": A realistic, natural question or objection from {persona['name']} ({persona['title']}). Must be conversational, specific to {industry}, and 1-2 sentences.
2. "teleprompt_text": An ideal, high-converting salesperson response script to repeat (20-35 words, confident, structured, addressing {persona['name']}'s concern directly).

Respond ONLY with valid JSON with keys "question_text" and "teleprompt_text":
{{
  "question_text": "...",
  "teleprompt_text": "..."
}}
"""

        question_text = f"In terms of {stage.lower()}, how does your solution address {focus.lower()}?"
        teleprompt_text = f"We provide a proven enterprise solution tailored for {industry} that delivers fast measurable results with dedicated engineering support."

        parsed = await self._call_llm_json(
            system="You are an enterprise sales simulator returning valid JSON.",
            prompt=dynamic_prompt,
            temperature=0.8,
            timeout=15.0,
        )
        if parsed.get("question_text") and parsed.get("teleprompt_text"):
            question_text = parsed["question_text"].strip()
            teleprompt_text = parsed["teleprompt_text"].strip()

        # Synthesize ElevenLabs voice on-demand for this specific dynamic question
        voice_id = persona["voice_id"]
        voice_name = persona["name"]
        audio_id = f"audio_{session_id}_{round_num}_{uuid.uuid4().hex[:6]}"
        audio_bytes, mime_type = await self.tts.generate_speech(question_text, voice_id=voice_id)
        self.audio_store[audio_id] = (audio_bytes, mime_type)

        round_data = {
            "round_number": round_num,
            "stage": stage,
            "scenario_type": f"{industry} • {persona['title']}",
            "persona_name": persona["name"],
            "persona_title": persona["title"],
            "voice_id": voice_id,
            "voice_name": voice_name,
            "question_text": question_text,
            "teleprompt_text": teleprompt_text,
            "audio_id": audio_id,
            "audio_url": f"/api/calibration/audio/{audio_id}",
            "evaluation": None,
            "completed": False,
            "duration_seconds": 0.0,
        }

        session["rounds"][round_num] = round_data
        session["current_round"] = round_num
        return round_data

    async def evaluate_round(
        self,
        session_id: str,
        round_num: int,
        audio_bytes: Optional[bytes] = None,
        mime_type: str = "audio/webm",
        transcript_override: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Transcribe user audio, evaluate against teleprompt, compute 6 exact metrics."""
        session = self.get_session(session_id)
        if round_num not in session["rounds"]:
            await self.generate_round_voice_on_demand(session_id, round_num)

        round_data = session["rounds"][round_num]
        teleprompt = round_data["teleprompt_text"]

        # 1. Derive duration server-side from received audio_bytes rather than trusting client form
        server_duration = get_audio_duration_seconds(audio_bytes, mime_type, fallback_seconds=duration_seconds)
        final_duration = server_duration if server_duration > 0 else duration_seconds

        # 2. Transcribe speech if audio is provided
        if transcript_override is not None:
            user_transcript = transcript_override.strip()
        elif audio_bytes and len(audio_bytes) > 100:
            user_transcript = (await self.stt.transcribe(audio_bytes, mime_type)).strip()
        else:
            user_transcript = ""

        # 3. Compute Communication Style Profile (Deterministic Math + Optional Signal Engine)
        evaluation = await self._run_llm_evaluation(
            round_num=round_num,
            stage=round_data["stage"],
            target_teleprompt=teleprompt,
            user_transcript=user_transcript,
            duration_seconds=final_duration,
            audio_bytes=audio_bytes,
        )

        round_data["evaluation"] = evaluation
        round_data["completed"] = True
        round_data["duration_seconds"] = final_duration

        # Advance session round
        if round_num < 8:
            session["current_round"] = round_num + 1
        else:
            session["status"] = "completed"
            session["end_time_monotonic"] = time.monotonic()
            session["summary_report"] = self._compute_summary_report(session)

        return round_data

    async def _run_llm_evaluation(
        self,
        round_num: int,
        stage: str,
        target_teleprompt: str,
        user_transcript: str,
        duration_seconds: float,
        audio_bytes: Optional[bytes] = None,
    ) -> dict[str, Any]:
        """Perform communication style profiling on delivery characteristics without script accuracy comparison."""
        user_transcript_clean = user_transcript.strip() if user_transcript else ""

        # Detect filler words
        filler_patterns = [
            r"\b(um+)\b", r"\b(uh+)\b", r"\b(er+)\b", r"\b(ah+)\b",
            r"\b(like)\b", r"\b(you know)\b", r"\b(sort of)\b",
            r"\b(kind of)\b", r"\b(actually)\b", r"\b(basically)\b",
            r"\b(literally)\b", r"\b(so yeah)\b", r"\b(i mean)\b"
        ]
        detected_fillers = []
        lower_transcript = user_transcript_clean.lower()
        for pat in filler_patterns:
            matches = re.findall(pat, lower_transcript)
            if matches:
                detected_fillers.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])

        words = [w for w in re.sub(r"[^\w\s]", "", lower_transcript).split() if w]
        word_count = len(words)

        # Check sample quality & confidence: silence / bad audio is excluded, never scored 0% failing
        confidence = self.rate_sample_quality(user_transcript_clean, audio_bytes, duration_seconds)
        is_excluded = (confidence == 0.0)

        if is_excluded:
            neutral_breakdown = RoundScoreBreakdown(
                speaking_pace=60,
                response_timing=60,
                sentence_handling=60,
                vocabulary_complexity=60,
                pause_pattern=60,
                energy_dynamics=60,
                word_choice=60,
                pacing=60,
                sentiment=60,
                tone_emphasis=60,
                pause_filters=60,
                energy_inflection=60,
            )
            return RoundEvaluation(
                round_number=round_num,
                stage=stage,
                overall_score=60,
                score_breakdown=neutral_breakdown,
                transcribed_text=user_transcript_clean or "[No speech detected]",
                target_teleprompt=target_teleprompt,
                feedback="No speech detected. This round is flagged as low confidence and excluded from your communication profile calculation.",
                strengths=["Sample excluded from profile calculation"],
                improvements=["Speak clearly into your microphone during recording."],
                detected_fillers=[],
                wpm=0,
                confidence=0.0,
                is_excluded=True,
                style_traits=["Excluded - Insufficient audio signal"],
                prompting_adaptation="Round excluded from teleprompter profile weighting.",
            ).model_dump()

        # Deterministic Style Measurements
        wpm, pace_score = self.compute_speaking_pace(word_count, duration_seconds)
        timing_score = self.compute_response_timing()
        avg_sentence_len, sentence_score = self.compute_sentence_handling(user_transcript_clean, word_count)
        ttr, vocab_score = self.compute_vocabulary_complexity(user_transcript_clean, word_count)
        pause_score = self.compute_pause_cadence(user_transcript_clean, duration_seconds, word_count, detected_fillers)
        energy_score = self.compute_energy_dynamics(audio_bytes, user_transcript_clean)

        # Apply configurable, versioned weight table
        w = self.profiling_weights
        composite_score = int(round(
            pace_score * w.get("speaking_pace", 0.25) +
            timing_score * w.get("response_timing", 0.15) +
            sentence_score * w.get("sentence_handling", 0.15) +
            vocab_score * w.get("vocabulary_complexity", 0.15) +
            pause_score * w.get("pause_pattern", 0.15) +
            energy_score * w.get("energy_dynamics", 0.15)
        ))
        composite_score = max(0, min(100, composite_score))
        prompting_lvl = resolve_prompting_level(composite_score)

        style_traits = []
        if wpm >= 150:
            style_traits.append(f"Direct, brisk tempo ({wpm} WPM)")
        elif wpm >= 120:
            style_traits.append(f"Conversational tempo ({wpm} WPM)")
        else:
            style_traits.append(f"Deliberate, measured tempo ({wpm} WPM)")

        if avg_sentence_len <= 12:
            style_traits.append("Direct, punchy sentence structure")
        else:
            style_traits.append("Elaborative, multi-clause structure")

        if len(detected_fillers) == 0:
            style_traits.append("Fluid cadence with minimal fillers")
        else:
            style_traits.append(f"Conversational pauses ({len(detected_fillers)} filler(s) detected)")

        breakdown = RoundScoreBreakdown(
            speaking_pace=pace_score,
            response_timing=timing_score,
            sentence_handling=sentence_score,
            vocabulary_complexity=vocab_score,
            pause_pattern=pause_score,
            energy_dynamics=energy_score,
            word_choice=vocab_score,
            pacing=pace_score,
            sentiment=timing_score,
            tone_emphasis=sentence_score,
            pause_filters=pause_score,
            energy_inflection=energy_score,
        )

        # Optional qualitative style observation via LLM if client is available
        llm_feedback = None
        if self.groq_client or self.openai_client:
            try:
                style_prompt = f"""You are an expert executive communication profiler analyzing a salesperson's natural speech style.
Spoken Transcript: "{user_transcript_clean}"
Tempo: {wpm} WPM | Avg Sentence Length: {avg_sentence_len} words | Detected Fillers: {detected_fillers}
Assigned Prompting Level: {prompting_lvl['name']} ({prompting_lvl['description']})

Provide 1 sentence describing their natural communication style traits and how it fits {prompting_lvl['name']}. Do NOT judge correctness, do NOT grade pass/fail, and do NOT compare against any script.
Return valid JSON:
{{"style_observation": "..."}}
"""
                llm_data = await self._call_llm_json(
                    system="You are an executive communication style profiler returning valid JSON.",
                    prompt=style_prompt,
                    temperature=0.3,
                    timeout=5.0,
                )
                if llm_data and "style_observation" in llm_data:
                    llm_feedback = str(llm_data["style_observation"]).strip()
                elif llm_data and "feedback" in llm_data:
                    llm_feedback = str(llm_data["feedback"]).strip()
            except Exception:
                llm_feedback = None

        final_feedback = llm_feedback or f"Natural delivery profiled at {wpm} WPM with {style_traits[1].lower()}. Aligned with {prompting_lvl['name']}."

        return RoundEvaluation(
            round_number=round_num,
            stage=stage,
            overall_score=composite_score,
            score_breakdown=breakdown,
            transcribed_text=user_transcript_clean,
            target_teleprompt=target_teleprompt,
            feedback=final_feedback,
            strengths=style_traits,
            improvements=[f"Teleprompter alignment: {prompting_lvl['directive']}"],
            detected_fillers=detected_fillers,
            wpm=wpm,
            confidence=confidence,
            is_excluded=False,
            style_traits=style_traits,
            prompting_adaptation=prompting_lvl["description"],
        ).model_dump()

    def _compute_summary_report(self, session: dict[str, Any]) -> dict[str, Any]:
        """Compute final 8-round composite communication style profile report."""
        rounds = session["rounds"]
        completed_rounds = [r for r in rounds.values() if r.get("completed") and r.get("evaluation")]
        num_completed = len(completed_rounds)

        if not completed_rounds:
            return {}

        # Exclude low-confidence/silent rounds from profile averaging
        valid_rounds = [
            r for r in completed_rounds
            if not r["evaluation"].get("is_excluded", False) and r["evaluation"].get("confidence", 1.0) > 0.1
        ]
        if not valid_rounds:
            valid_rounds = completed_rounds

        total_conf = sum(r["evaluation"].get("confidence", 1.0) for r in valid_rounds) or 1.0
        overall_score = int(round(
            sum(r["evaluation"]["overall_score"] * r["evaluation"].get("confidence", 1.0) for r in valid_rounds) / total_conf
        ))

        def _w_avg(key: str, fallback_key: str = "word_choice") -> int:
            vals = []
            for r in valid_rounds:
                b = r["evaluation"].get("score_breakdown", {})
                v = b.get(key, b.get(fallback_key, 75))
                vals.append(v * r["evaluation"].get("confidence", 1.0))
            return int(round(sum(vals) / total_conf))

        avg_pace = _w_avg("speaking_pace", "pacing")
        avg_timing = _w_avg("response_timing", "sentiment")
        avg_sentence = _w_avg("sentence_handling", "tone_emphasis")
        avg_vocab = _w_avg("vocabulary_complexity", "word_choice")
        avg_pause = _w_avg("pause_pattern", "pause_filters")
        avg_energy = _w_avg("energy_dynamics", "energy_inflection")

        valid_wpms = [r["evaluation"]["wpm"] for r in valid_rounds if r["evaluation"].get("wpm", 0) > 0]
        avg_wpm = int(round(sum(valid_wpms) / len(valid_wpms))) if valid_wpms else 140

        prompting_level = resolve_prompting_level(overall_score)

        start_mono = session.get("start_time_monotonic", time.monotonic())
        end_mono = session.get("end_time_monotonic") or time.monotonic()
        total_seconds = max(end_mono - start_mono, 1.0)
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        time_formatted = f"{minutes}m {seconds:02d}s" if minutes > 0 else f"{seconds}s"
        completed_on_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

        round_performance_list = []
        for i in range(1, 9):
            r_data = rounds.get(i)
            if r_data and r_data.get("evaluation"):
                ev = r_data["evaluation"]
                round_performance_list.append({
                    "round_number": i,
                    "stage": r_data["stage"],
                    "score": ev["overall_score"],
                    "wpm": ev.get("wpm", 0),
                    "confidence": ev.get("confidence", 1.0),
                    "is_excluded": ev.get("is_excluded", False),
                    "completed": True,
                })
            else:
                stage_name = SALES_STAGES[i - 1]["stage"]
                round_performance_list.append({
                    "round_number": i,
                    "stage": stage_name,
                    "score": None,
                    "wpm": None,
                    "completed": False,
                })

        rounds_detailed = []
        for i in range(1, 9):
            r_data = rounds.get(i)
            if r_data:
                ev = r_data.get("evaluation") or {}
                rounds_detailed.append({
                    "round_number": i,
                    "stage": r_data.get("stage", f"Round {i}"),
                    "persona_name": r_data.get("persona_name", "AI Prospect"),
                    "persona_title": r_data.get("persona_title", "Executive"),
                    "voice_name": r_data.get("voice_name", "Rachel"),
                    "question_text": r_data.get("question_text", ""),
                    "teleprompt_text": r_data.get("teleprompt_text", ""),
                    "user_transcript": ev.get("transcribed_text", ""),
                    "duration_seconds": r_data.get("duration_seconds", 0.0),
                    "wpm": ev.get("wpm", 0),
                    "overall_score": ev.get("overall_score"),
                    "score_breakdown": ev.get("score_breakdown", {}),
                    "confidence": ev.get("confidence", 1.0),
                    "is_excluded": ev.get("is_excluded", False),
                    "style_traits": ev.get("style_traits", []),
                    "feedback": ev.get("feedback", ""),
                    "strengths": ev.get("strengths", []),
                    "improvements": ev.get("improvements", []),
                    "detected_fillers": ev.get("detected_fillers", []),
                })

        report_dict = {
            "session_id": session["session_id"],
            "user_id": session.get("user_id", "sales_rep_1"),
            "industry": session.get("industry", "Enterprise B2B SaaS"),
            "created_at": session.get("created_at"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "overall_score": overall_score,
            "wpm": avg_wpm,
            "prompting_level": prompting_level,
            "formula_version": CALIBRATION_FORMULA_VERSION,
            "formula_weights": self.profiling_weights,
            "rounds_completed": num_completed,
            "rounds_profiled": len(valid_rounds),
            "total_rounds": 8,
            "total_time_seconds": round(total_seconds, 1),
            "total_time_formatted": time_formatted,
            "completed_on": completed_on_str,
            "status_message": f"Great job! Calibration complete. Your natural speaking style is mapped to {prompting_level['name']}.",
            "round_performance": round_performance_list,
            "rounds_detail": rounds_detailed,
            "score_breakdown": {
                "speaking_pace": avg_pace,
                "response_timing": avg_timing,
                "sentence_handling": avg_sentence,
                "vocabulary_complexity": avg_vocab,
                "pause_pattern": avg_pause,
                "energy_dynamics": avg_energy,
                # Backwards compatible aliases
                "word_choice": avg_vocab,
                "pacing": avg_pace,
                "sentiment": avg_timing,
                "tone_emphasis": avg_sentence,
                "pause_filters": avg_pause,
                "energy_inflection": avg_energy,
            },
            "what_this_means": [
                f"Your speaking cadence averages {avg_wpm} WPM with {prompting_level['badge'].lower()} sentence flow.",
                f"Teleprompter lines will automatically use {prompting_level['directive']}",
                "Prompting levels route guidance without judging speech correctness or accuracy.",
                "Recalibrate anytime if your speaking environment or preferences change.",
            ],
            "what_is_next": [
                f"Live call teleprompter configured with {prompting_level['name']}.",
                "Real-time guidance will adapt to your tempo automatically.",
                "Recalibrate anytime to fine-tune your voice profile.",
            ],
        }

        # Save report to local disk in reports/ directory
        try:
            reports_dir = Path(__file__).resolve().parent.parent / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Unique session report
            report_filename = f"calibration_{session['session_id']}.json"
            report_path = reports_dir / report_filename
            report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            
            # 2. Latest calibration pointer
            latest_path = reports_dir / "latest_calibration_report.json"
            latest_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            
            report_dict["local_file_path"] = str(report_path.resolve())
            report_dict["local_filename"] = report_filename
            LOGGER.info("Successfully stored calibration JSON report to local disk: %s", report_path)
        except Exception as exc:
            LOGGER.warning("Could not write report to local disk: %s", exc)

        return report_dict


# ==========================================
# FastAPI Router & App Factory
# ==========================================

def get_calibration_router(service: Optional[CalibrationService] = None) -> APIRouter:
    """Create dedicated FastAPI APIRouter for calibration endpoints."""
    if service is None:
        service = CalibrationService()
    router = APIRouter(prefix="/api/calibration", tags=["Calibration"])

    @router.post("/start")
    async def start_session(user_id: str = Form("sales_rep_1"), industry: Optional[str] = Form(None)):
        session = await service.create_dynamic_session(user_id=user_id, industry=industry)
        round_1 = await service.generate_round_voice_on_demand(session["session_id"], 1)
        return {
            "status": "success",
            "session": session,
            "current_round_data": round_1,
        }

    @router.get("/session/{session_id}")
    async def get_session_status(session_id: str):
        session = service.get_session(session_id)
        report = session.get("summary_report")
        if not report and session.get("status") == "completed":
            report = service._compute_summary_report(session)
        return {
            "session_id": session_id,
            "status": session["status"],
            "industry": session.get("industry"),
            "current_round": session["current_round"],
            "total_rounds": session["total_rounds"],
            "rounds": session["rounds"],
            "summary_report": report,
        }

    @router.get("/latest")
    async def get_latest_calibration():
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
        latest_file = reports_dir / "latest_calibration_report.json"
        if not latest_file.exists():
            return {
                "status": "default",
                "wpm": 140,
                "prompting_level": PROMPTING_LEVELS["balanced"],
                "tone": "Natural & Confident",
                "pacing": "Balanced conversational 140 WPM pace",
                "clarity": 85,
            }
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "status": "success",
                "report": data,
                "wpm": data.get("wpm", 140),
                "prompting_level": data.get("prompting_level", PROMPTING_LEVELS["balanced"]),
            }
        except Exception as exc:
            return {
                "status": "default",
                "wpm": 140,
                "prompting_level": PROMPTING_LEVELS["balanced"],
                "tone": "Natural & Confident",
                "error": str(exc),
            }

    @router.post("/round/generate")
    async def generate_round_endpoint(session_id: str = Form(...), round_number: int = Form(...)):
        round_data = await service.generate_round_voice_on_demand(session_id, round_number)
        return {"status": "success", "round_data": round_data}

    @router.post("/round/evaluate")
    async def evaluate_round_endpoint(
        session_id: str = Form(...),
        round_number: int = Form(...),
        duration_seconds: float = Form(0.0),
        transcript_override: Optional[str] = Form(None),
        audio_file: Optional[UploadFile] = File(None),
    ):
        audio_bytes = None
        mime_type = "audio/webm"
        if audio_file:
            audio_bytes = await audio_file.read()
            mime_type = audio_file.content_type or "audio/webm"

        round_data = await service.evaluate_round(
            session_id=session_id,
            round_num=round_number,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            transcript_override=transcript_override,
            duration_seconds=duration_seconds,
        )

        session = service.get_session(session_id)
        is_completed = session.get("status") == "completed"

        return {
            "status": "success",
            "round_data": round_data,
            "is_session_completed": is_completed,
            "next_round": session["current_round"] if not is_completed else None,
            "summary_report": session.get("summary_report") if is_completed else None,
        }

    @router.get("/audio/{audio_id}")
    async def get_audio_stream(audio_id: str):
        if audio_id not in service.audio_store:
            raise HTTPException(status_code=404, detail="Audio file not found")
        audio_bytes, mime_type = service.audio_store[audio_id]
        return Response(content=audio_bytes, media_type=mime_type)

    return router


def create_standalone_calibration_app() -> FastAPI:
    """Create standalone FastAPI application for the Voice Calibration system."""
    app = FastAPI(title="AI Voice Calibration Studio", version="1.0.0")

    service = CalibrationService()
    router = get_calibration_router(service)
    app.include_router(router)

    # Serve static UI
    @app.get("/")
    async def root_view():
        html_file = STATIC_DIR / "calibration.html"
        if not html_file.exists():
            raise HTTPException(status_code=404, detail="Calibration UI file not found")
        return FileResponse(html_file)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": "AI Voice Calibration",
            "elevenlabs_configured": bool(service.elevenlabs_api_key),
            "groq_configured": bool(service.groq_api_key),
            "deepgram_configured": bool(service.deepgram_api_key),
        }

    return app
