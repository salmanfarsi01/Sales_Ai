#!/usr/bin/env python3
"""Standalone runner for the Behavioral Signal Engine test console.

Runs only the behavioral signal pipeline (Deterministic Timing, Semantic Features,
Baseline & Change-Points, Multi-Window Aggregation, Downstream Inference)
without running the full Twilio or Call Copilot infrastructure.
"""

from __future__ import annotations

import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Load environment variables (.env)
load_dotenv(override=True)

# Behavioral Signal Engine imports
from copilot.calibration import SpeechToTextEngine
from copilot.behavioral_normalization import (
    NormalizedUtterance,
    NormalizedWord,
    segment_audio_transcript_turns,
    merge_dual_speaker_tracks,
)
from copilot.behavioral_timing import DeterministicTimingEngine, TimingFeatureSnapshot
from copilot.behavioral_semantic import SemanticFeatureEngine, SemanticFeatureSnapshot
from copilot.behavioral_baseline import (
    BaselineAndChangePointEngine,
    ProspectBaselineStore,
    ChangePointEvent,
    DEFAULT_INTRA_CALL_WINDOW_MS,
)
from copilot.behavioral_evidence import (
    MultiWindowAggregator,
    SQLiteEvidenceLogStore,
    MultiWindowEvidenceFrame,
    BehavioralEvidenceSnapshot,
)
from copilot.behavioral_inference import (
    DownstreamInferenceEngine,
    DownstreamInferenceState,
)
from copilot.behavioral_pipeline_service import execute_behavioral_turn_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("behavioral_signal_server")

STATIC_DIR = Path(__file__).resolve().parent / "web"
HTML_FILE = STATIC_DIR / "behavioral_test.html"

# Initialize FastAPI app
app = FastAPI(
    title="PitchProX Behavioral Signal Engine Test Server",
    version="1.0.0",
    description="Dedicated server for live testing behavioral signal inference from prospect audio.",
)

# Enable CORS for local testing from any browser origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Stores and Engines
baseline_store = ProspectBaselineStore()
evidence_store = SQLiteEvidenceLogStore()
inference_engine = DownstreamInferenceEngine()

# In-memory sessions per prospect to support multi-turn calibration & change points
prospect_sessions: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(prospect_id: str, call_sid: str) -> Dict[str, Any]:
    session_key = f"{prospect_id}_{call_sid}"
    if session_key not in prospect_sessions:
        timing_engine = DeterministicTimingEngine()
        semantic_engine = SemanticFeatureEngine()
        cal_window_ms = int(os.getenv("BEHAVIORAL_CALIBRATION_WINDOW_MS", str(DEFAULT_INTRA_CALL_WINDOW_MS)))
        baseline_engine = BaselineAndChangePointEngine(
            call_sid=call_sid,
            prospect_id=prospect_id,
            store=baseline_store,
            intra_call_window_ms=cal_window_ms,
        )
        aggregator = MultiWindowAggregator(
            call_sid=call_sid,
            timing_engine=timing_engine,
            baseline_engine=baseline_engine,
            store=evidence_store,
        )
        prospect_sessions[session_key] = {
            "timing_engine": timing_engine,
            "semantic_engine": semantic_engine,
            "baseline_engine": baseline_engine,
            "aggregator": aggregator,
            "context_history": [],
            "turn_index": 0,
        }
    return prospect_sessions[session_key]


@app.get("/")
@app.get("/behavioral-test")
async def serve_test_console():
    """Serves the Behavioral Signal Test Console HTML."""
    if not HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="behavioral_test.html not found in web/ directory")
    return FileResponse(HTML_FILE)


@app.get("/api/test/health")
async def health():
    return {
        "status": "online",
        "service": "Behavioral Signal Engine",
        "deepgram_available": bool(os.getenv("DEEPGRAM_API_KEY")),
        "groq_available": bool(os.getenv("GROQ_API_KEY")),
    }


async def execute_turn_pipeline(
    segmented_utts: List[NormalizedUtterance],
    active_call_sid: str,
    prospect_id: str,
    transcript: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    agent_id = getattr(session.get("baseline_engine"), "agent_id", None) or "test_agent_001"
    reports_dir = Path(__file__).resolve().parent / "reports"
    res = await execute_behavioral_turn_pipeline(
        segmented_utts=segmented_utts,
        call_sid=active_call_sid,
        prospect_id=prospect_id,
        agent_id=agent_id,
        transcript=transcript,
        timing_engine=session["timing_engine"],
        semantic_engine=session["semantic_engine"],
        baseline_engine=session["baseline_engine"],
        aggregator=session["aggregator"],
        inference_engine=inference_engine,
        context_history=session["context_history"],
        turn_index_start=session["turn_index"] + 1,
        reports_dir=reports_dir,
    )
    session["turn_index"] += len(segmented_utts)
    return res


@app.post("/api/test/analyze-recording")
async def analyze_recording(
    audio: UploadFile = File(...),
    prospect_id: str = Form("test_prospect_001"),
    call_sid: Optional[str] = Form(None),
):
    """Processes uploaded prospect audio through the full behavioral pipeline:
    1. STT (Deepgram with diarization) with word-level timestamps.
    2. Deterministic timing analysis (turn WPM, pause duration, response latency).
    3. Semantic feature extraction (intent, recurrence, boundary).
    4. Baseline & change-point detection (z-score departures per speaker).
    5. Multi-window evidence frame construction.
    6. Downstream inference scoring (Trust, Pacing, Engagement, Momentum, Readiness, Emotion).
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    active_call_sid = call_sid or f"call_{uuid.uuid4().hex[:8]}"
    session = get_or_create_session(prospect_id, active_call_sid)

    # 1. Transcribe audio with word-level timestamps and diarization
    content_type = audio.content_type or "audio/webm"
    stt = SpeechToTextEngine(
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    LOGGER.info("Transcribing %d bytes (%s) for prospect '%s'...", len(audio_bytes), content_type, prospect_id)
    transcript, words_raw = await stt.transcribe_with_timestamps(
        audio_bytes, mime_type=content_type, utt_split=0.5
    )
    raw_utterances = getattr(stt, "latest_raw_utterances", None) or []
    transcript = transcript.strip() if transcript else ""
    if not transcript:
        transcript = "I see, thanks for letting me know."
        LOGGER.info("No audible speech detected; using fallback placeholder transcript.")

    # 2. Segment audio into distinct utterance turns with diarization
    segmented_utts = segment_audio_transcript_turns(
        words_raw=words_raw,
        transcript=transcript,
        call_sid=active_call_sid,
        speaker_id="client",
        raw_utterances=raw_utterances,
        min_pause_split_ms=500,
    )
    LOGGER.info("Segmented audio into %d sequential turn(s)", len(segmented_utts))

    return await execute_turn_pipeline(
        segmented_utts=segmented_utts,
        active_call_sid=active_call_sid,
        prospect_id=prospect_id,
        transcript=transcript,
        session=session,
    )


@app.post("/api/test/analyze-two-tracks")
async def analyze_two_tracks(
    agent_audio: UploadFile = File(...),
    prospect_audio: UploadFile = File(...),
    agent_start_epoch_ms: int = Form(0),
    prospect_start_epoch_ms: int = Form(0),
    prospect_offset_ms: int = Form(0),
    prospect_id: str = Form("test_prospect_001"),
    agent_id: str = Form("test_agent_001"),
    call_sid: Optional[str] = Form(None),
):
    """Processes two independently recorded single-speaker tracks (Agent & Prospect),
    transcribes them separately with diarization disabled, merges them onto a synchronized
    chronological timeline, and runs the full behavioral pipeline with isolated per-speaker baselines.
    """
    try:
        agent_bytes = await agent_audio.read()
        prospect_bytes = await prospect_audio.read()

        if not agent_bytes and not prospect_bytes:
            raise HTTPException(status_code=400, detail="Both audio tracks are empty")

        active_call_sid = call_sid or f"call_{uuid.uuid4().hex[:8]}"
        # Ensure clean session state for fresh dual-track analysis so previous runs don't leak turns
        session_key = f"{prospect_id}_{active_call_sid}"
        if session_key in prospect_sessions:
            LOGGER.info("Resetting existing session '%s' for fresh dual-track analysis", session_key)
            prospect_sessions.pop(session_key, None)
        session = get_or_create_session(prospect_id, active_call_sid)
        session["baseline_engine"].agent_id = agent_id

        stt = SpeechToTextEngine(
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        agent_segmented: List[NormalizedUtterance] = []
        if agent_bytes:
            agent_mime = agent_audio.content_type or "audio/webm"
            LOGGER.info("Transcribing Agent track (single-speaker): %d bytes (%s)...", len(agent_bytes), agent_mime)
            agent_transcript, agent_words = await stt.transcribe_with_timestamps(
                agent_bytes, mime_type=agent_mime, utt_split=0.5, diarize=False
            )
            agent_raw_utts = getattr(stt, "latest_raw_utterances", None) or []
            agent_segmented = segment_audio_transcript_turns(
                words_raw=agent_words,
                transcript=agent_transcript or "",
                call_sid=active_call_sid,
                speaker_id="salesperson",
                raw_utterances=agent_raw_utts,
                min_pause_split_ms=500,
                enforce_single_speaker=True,
            )

        prospect_segmented: List[NormalizedUtterance] = []
        if prospect_bytes:
            prospect_mime = prospect_audio.content_type or "audio/webm"
            LOGGER.info("Transcribing Prospect track (single-speaker): %d bytes (%s)...", len(prospect_bytes), prospect_mime)
            prospect_transcript, prospect_words = await stt.transcribe_with_timestamps(
                prospect_bytes, mime_type=prospect_mime, utt_split=0.5, diarize=False
            )
            prospect_raw_utts = getattr(stt, "latest_raw_utterances", None) or []
            prospect_segmented = segment_audio_transcript_turns(
                words_raw=prospect_words,
                transcript=prospect_transcript or "",
                call_sid=active_call_sid,
                speaker_id="client",
                raw_utterances=prospect_raw_utts,
                min_pause_split_ms=500,
                enforce_single_speaker=True,
            )

        merged_utts = merge_dual_speaker_tracks(
            agent_utterances=agent_segmented,
            prospect_utterances=prospect_segmented,
            agent_start_epoch_ms=agent_start_epoch_ms,
            prospect_start_epoch_ms=prospect_start_epoch_ms,
            prospect_offset_ms=prospect_offset_ms,
        )
        LOGGER.info(
            "Merged dual tracks: %d agent turns + %d prospect turns -> %d total chronological turns (offset=%d ms)",
            len(agent_segmented),
            len(prospect_segmented),
            len(merged_utts),
            prospect_offset_ms,
        )

        combined_transcript = " ".join(
            f"[{'Agent' if u.speaker_id == 'salesperson' else 'Prospect'}]: {u.text}"
            for u in merged_utts
        )

        return await execute_turn_pipeline(
            segmented_utts=merged_utts,
            active_call_sid=active_call_sid,
            prospect_id=prospect_id,
            transcript=combined_transcript,
            session=session,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Dual-track analysis failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"Dual-track analysis failed: {str(exc)}"},
        )


@app.post("/api/test/reset-session")
async def reset_session(prospect_id: str = Form("test_prospect_001")):
    """Resets the in-memory engine state for a prospect."""
    keys_to_remove = [k for k in prospect_sessions if k.startswith(f"{prospect_id}_")]
    for k in keys_to_remove:
        prospect_sessions.pop(k, None)
    return {"status": "success", "message": f"Session reset for prospect '{prospect_id}'"}


if __name__ == "__main__":
    port = int(os.getenv("BEHAVIORAL_PORT", "5002"))
    host = os.getenv("BEHAVIORAL_HOST", "127.0.0.1")

    print("\n" + "=" * 65)
    print("  [*] PITCHPROX BEHAVIORAL SIGNAL ENGINE TEST SERVER")
    print(f"  --> Web UI Console:   http://{host}:{port}/behavioral-test")
    print(f"  --> Live Endpoint:    http://{host}:{port}/api/test/analyze-recording")
    print(f"  --> Dual Track:       http://{host}:{port}/api/test/analyze-two-tracks")
    print("=" * 65 + "\n")

    uvicorn.run("run_behavioral_signal:app", host=host, port=port, reload=True, log_level="info")
