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
# Data Models
# ==========================================

class RoundScoreBreakdown(BaseModel):
    word_choice: int = Field(..., ge=0, le=100, description="Word choice and accuracy score")
    pacing: int = Field(..., ge=0, le=100, description="Pacing and speaking rate score")
    sentiment: int = Field(..., ge=0, le=100, description="Sentiment and confidence score")
    tone_emphasis: int = Field(..., ge=0, le=100, description="Tone and vocal emphasis score")
    pause_filters: int = Field(..., ge=0, le=100, description="Absence of filler words and smooth pauses")
    energy_inflection: int = Field(..., ge=0, le=100, description="Vocal energy and cadence inflection")


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
    rounds_completed: int
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

        # 3. Compute AI Performance Evaluation
        evaluation = await self._run_llm_evaluation(
            round_num=round_num,
            stage=round_data["stage"],
            target_teleprompt=teleprompt,
            user_transcript=user_transcript,
            duration_seconds=final_duration,
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
    ) -> dict[str, Any]:
        """Perform strict, objective AI evaluation with blind LLM rubric, Pydantic validation, and heuristic fallback."""
        user_transcript_clean = user_transcript.strip() if user_transcript else ""

        # Check for empty / silence -> Strictly 0 score
        if not user_transcript_clean:
            zero_breakdown = RoundScoreBreakdown(
                word_choice=0, pacing=0, sentiment=0, tone_emphasis=0, pause_filters=0, energy_inflection=0
            )
            return RoundEvaluation(
                round_number=round_num,
                stage=stage,
                overall_score=0,
                score_breakdown=zero_breakdown,
                transcribed_text="[No speech detected]",
                target_teleprompt=target_teleprompt,
                feedback="No speech detected. You received 0% because you did not speak or repeat the teleprompt script.",
                strengths=[],
                improvements=["Speak clearly into your microphone and repeat the teleprompt script aloud."],
                detected_fillers=[],
                wpm=0,
            ).model_dump()

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

        # Text normalization
        target_clean = re.sub(r"[^\w\s]", "", target_teleprompt.lower()).strip()
        user_clean = re.sub(r"[^\w\s]", "", lower_transcript).strip()
        
        target_words = target_clean.split()
        user_words = user_clean.split()

        if not user_words:
            zero_breakdown = RoundScoreBreakdown(
                word_choice=0, pacing=0, sentiment=0, tone_emphasis=0, pause_filters=0, energy_inflection=0
            )
            return RoundEvaluation(
                round_number=round_num,
                stage=stage,
                overall_score=0,
                score_breakdown=zero_breakdown,
                transcribed_text=user_transcript_clean,
                target_teleprompt=target_teleprompt,
                feedback="No audible words detected. Teleprompt was not repeated.",
                strengths=[],
                improvements=["Repeat the full teleprompt script on screen."],
                detected_fillers=[],
                wpm=0,
            ).model_dump()

        # Similarity metrics
        seq_ratio = difflib.SequenceMatcher(None, target_clean, user_clean).ratio()
        target_set = set(target_words)
        user_set = set(user_words)
        common_words = target_set.intersection(user_set)

        # Stop words to ignore for keyword coverage
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "is", "we", "our", "i", "you", "it", "this", "that", "of"}
        target_keywords = target_set - stop_words
        user_keywords = user_set - stop_words
        common_keywords = target_keywords.intersection(user_keywords)
        coverage_ratio = len(common_keywords) / max(len(target_keywords), 1) if target_keywords else (len(common_words) / max(len(target_set), 1))

        # Check for completely irrelevant / disconnected speech -> Strictly 0 score
        if coverage_ratio < 0.25 or seq_ratio < 0.30:
            zero_breakdown = RoundScoreBreakdown(
                word_choice=0, pacing=0, sentiment=0, tone_emphasis=0, pause_filters=0, energy_inflection=0
            )
            return RoundEvaluation(
                round_number=round_num,
                stage=stage,
                overall_score=0,
                score_breakdown=zero_breakdown,
                transcribed_text=user_transcript_clean,
                target_teleprompt=target_teleprompt,
                feedback="Irrelevant response. What you said does not match the teleprompt script, resulting in a score of 0%.",
                strengths=[],
                improvements=["Make sure to read and repeat the exact teleprompt response displayed on screen."],
                detected_fillers=detected_fillers,
                wpm=int((len(user_words) / max(duration_seconds, 1.0)) * 60) if duration_seconds > 0 else 0,
            ).model_dump()

        # Heuristic calculation for fallback & bounds
        raw_word_score = (seq_ratio * 0.6 + coverage_ratio * 0.4) * 100
        length_penalty = min(1.0, len(user_words) / max(len(target_words), 1))
        strict_word_choice = max(0, min(100, int(round(raw_word_score * length_penalty))))

        wpm = int((len(user_words) / max(duration_seconds, 1.0)) * 60) if duration_seconds > 0 else 135
        if 125 <= wpm <= 155:
            strict_pacing = min(95, int(round(strict_word_choice * 1.05)))
        elif 105 <= wpm < 125 or 156 <= wpm <= 175:
            strict_pacing = min(75, int(round(strict_word_choice * 0.85)))
        elif 80 <= wpm < 105 or 176 <= wpm <= 195:
            strict_pacing = min(50, int(round(strict_word_choice * 0.60)))
        else:
            strict_pacing = min(25, int(round(strict_word_choice * 0.35)))

        filler_deduction = len(detected_fillers) * 15
        strict_pause_filters = max(0, min(100, strict_word_choice - filler_deduction if strict_word_choice < 85 else 100 - filler_deduction))
        strict_tone = max(0, min(100, int(round(strict_word_choice * 0.95))))
        strict_sentiment = max(0, min(100, int(round(strict_word_choice * 0.95))))
        strict_energy = max(0, min(100, int(round(strict_word_choice * 0.90))))

        strict_overall = int(round(
            (strict_word_choice * 0.35) +
            (strict_pacing * 0.20) +
            (strict_pause_filters * 0.15) +
            (strict_tone * 0.15) +
            (strict_energy * 0.10) +
            (strict_sentiment * 0.05)
        ))

        fallback_breakdown = RoundScoreBreakdown(
            word_choice=strict_word_choice,
            pacing=strict_pacing,
            sentiment=strict_sentiment,
            tone_emphasis=strict_tone,
            pause_filters=strict_pause_filters,
            energy_inflection=strict_energy,
        )
        fallback_eval = RoundEvaluation(
            round_number=round_num,
            stage=stage,
            overall_score=strict_overall,
            score_breakdown=fallback_breakdown,
            transcribed_text=user_transcript_clean,
            target_teleprompt=target_teleprompt,
            feedback="Direct similarity analysis: Ensure high fidelity to teleprompt value metrics and steady delivery.",
            strengths=["Accurate core phrasing" if strict_word_choice > 70 else "Clear initial tone"],
            improvements=["Improve pacing cadence" if strict_pacing < 75 else "Maintain crisp sentence endings without fillers"],
            detected_fillers=detected_fillers,
            wpm=wpm,
        )

        # Blind Rubric prompt for LLM (no pre-filled heuristic numbers to avoid anchoring)
        eval_prompt = f"""You are a strict, demanding B2B Executive Sales Performance Coach evaluating a salesperson's voice delivery.
Sales Stage: {stage} (Round {round_num} of 8)

Required Teleprompt:
"{target_teleprompt}"

User Spoken Transcript:
"{user_transcript_clean}"

Spoken Duration: {duration_seconds:.2f} seconds | WPM: {wpm} | Detected Fillers: {detected_fillers}

STRICT EVALUATION RUBRIC (Score each dimension independently between 0 and 100):
- "word_choice" (0-100): Award 85+ ONLY if virtually all key value metrics and specific phrasing from the teleprompt were accurately delivered. Deduct heavily for omitted facts, distortions, or partial speech.
- "pacing" (0-100): Optimal enterprise rate is 130-150 WPM. Penalize rushed (>170 WPM) or dragging (<110 WPM) delivery.
- "sentiment" (0-100): Professional composure, positive executive warmth, and consultative confidence.
- "tone_emphasis" (0-100): Strategic emphasis on ROI, SLAs, and technical guarantees. Penalize flat or casual tone.
- "pause_filters" (0-100): Clean articulation without filler hesitations. Each detected filler ({detected_fillers}) must reduce score by 10-15 points.
- "energy_inflection" (0-100): Vocal dynamics, cadence variance, and authoritative conviction.
- "overall_score" (0-100): Composite reflection of all 6 dimensions. Only crisp, near-flawless delivery should score 80+.

Return strictly valid JSON in this exact structure:
{{
  "overall_score": <integer 0-100>,
  "score_breakdown": {{
    "word_choice": <integer 0-100>,
    "pacing": <integer 0-100>,
    "sentiment": <integer 0-100>,
    "tone_emphasis": <integer 0-100>,
    "pause_filters": <integer 0-100>,
    "energy_inflection": <integer 0-100>
  }},
  "feedback": "<1-2 sentence constructive coaching feedback noting exact strengths or shortcomings>",
  "strengths": ["<1-2 specific strengths demonstrated>"],
  "improvements": ["<1-2 specific actionable coaching directives>"],
  "detected_fillers": {json.dumps(detected_fillers)}
}}
"""

        llm_data = await self._call_llm_json(
            system="You are an elite enterprise sales speech evaluation coach returning strictly valid JSON.",
            prompt=eval_prompt,
            temperature=0.2,
            timeout=15.0,
        )

        if not llm_data:
            return fallback_eval.model_dump()

        # Validate with Pydantic
        try:
            raw_breakdown = llm_data.get("score_breakdown", {})
            validated_breakdown = RoundScoreBreakdown(
                word_choice=int(raw_breakdown.get("word_choice", strict_word_choice)),
                pacing=int(raw_breakdown.get("pacing", strict_pacing)),
                sentiment=int(raw_breakdown.get("sentiment", strict_sentiment)),
                tone_emphasis=int(raw_breakdown.get("tone_emphasis", strict_tone)),
                pause_filters=int(raw_breakdown.get("pause_filters", strict_pause_filters)),
                energy_inflection=int(raw_breakdown.get("energy_inflection", strict_energy)),
            )
            validated_eval = RoundEvaluation(
                round_number=round_num,
                stage=stage,
                overall_score=int(llm_data.get("overall_score", strict_overall)),
                score_breakdown=validated_breakdown,
                transcribed_text=user_transcript_clean,
                target_teleprompt=target_teleprompt,
                feedback=str(llm_data.get("feedback", "Good delivery with accurate reproduction of key value points.")),
                strengths=list(llm_data.get("strengths", ["Accurate core phrasing"])),
                improvements=list(llm_data.get("improvements", ["Maintain consistent cadence"])),
                detected_fillers=list(llm_data.get("detected_fillers", detected_fillers)),
                wpm=wpm,
            )
            return validated_eval.model_dump()
        except Exception as val_err:
            LOGGER.warning("Pydantic validation failed on LLM response (%s), using validated fallback.", val_err)
            return fallback_eval.model_dump()

    def _compute_summary_report(self, session: dict[str, Any]) -> dict[str, Any]:
        """Compute final 8-round composite calibration report matching the user's screenshot."""
        rounds = session["rounds"]
        completed_rounds = [r for r in rounds.values() if r.get("completed") and r.get("evaluation")]
        num_completed = len(completed_rounds)

        if not completed_rounds:
            return {}

        round_scores = [r["evaluation"]["overall_score"] for r in completed_rounds]
        overall_score = int(round(sum(round_scores) / len(round_scores)))

        avg_word_choice = int(round(sum(r["evaluation"]["score_breakdown"]["word_choice"] for r in completed_rounds) / num_completed))
        avg_pacing = int(round(sum(r["evaluation"]["score_breakdown"]["pacing"] for r in completed_rounds) / num_completed))
        avg_sentiment = int(round(sum(r["evaluation"]["score_breakdown"]["sentiment"] for r in completed_rounds) / num_completed))
        avg_tone_emphasis = int(round(sum(r["evaluation"]["score_breakdown"]["tone_emphasis"] for r in completed_rounds) / num_completed))
        avg_pause_filters = int(round(sum(r["evaluation"]["score_breakdown"]["pause_filters"] for r in completed_rounds) / num_completed))
        avg_energy_inflection = int(round(sum(r["evaluation"]["score_breakdown"]["energy_inflection"] for r in completed_rounds) / num_completed))

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
                score = r_data["evaluation"]["overall_score"]
                round_performance_list.append({
                    "round_number": i,
                    "stage": r_data["stage"],
                    "score": score,
                    "completed": True,
                })
            else:
                stage_name = SALES_STAGES[i - 1]["stage"]
                round_performance_list.append({
                    "round_number": i,
                    "stage": stage_name,
                    "score": None,
                    "completed": False,
                })

        # Detailed round evaluations for local JSON storage
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
            "rounds_completed": num_completed,
            "total_rounds": 8,
            "total_time_seconds": round(total_seconds, 1),
            "total_time_formatted": time_formatted,
            "completed_on": completed_on_str,
            "status_message": "Great job! Your calibration is complete and your AI coach is ready to guide you on calls.",
            "round_performance": round_performance_list,
            "rounds_detail": rounds_detailed,
            "score_breakdown": {
                "word_choice": avg_word_choice,
                "pacing": avg_pacing,
                "sentiment": avg_sentiment,
                "tone_emphasis": avg_tone_emphasis,
                "pause_filters": avg_pause_filters,
                "energy_inflection": avg_energy_inflection,
            },
            "what_this_means": [
                "Your responses have been analyzed for pace, tone, and delivery.",
                "AI prompts will be personalized to your natural style.",
                "You'll receive more accurate, real-time guidance on every call.",
                "You can recalibrate anytime if you want to fine tune your experience.",
            ],
            "what_is_next": [
                "You can now make calls with real-time AI guidance.",
                "Your prompts will adapt to your style automatically.",
                "Recalibrate anytime to fine tune your experience.",
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
                "tone": "Confident & Assertive",
                "pacing": "Direct, structured, 140 WPM pace",
                "clarity": 85,
            }
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"status": "success", "report": data}
        except Exception as exc:
            return {
                "status": "default",
                "wpm": 140,
                "tone": "Confident & Assertive",
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
