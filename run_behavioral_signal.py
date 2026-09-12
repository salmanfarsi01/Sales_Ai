#!/usr/bin/env python3
"""Standalone runner for the Behavioral Signal Engine test console.

Runs only the behavioral signal pipeline (Deterministic Timing, Semantic Features,
Baseline & Change-Points, Multi-Window Aggregation, Downstream Inference)
without running the full Twilio or Call Copilot infrastructure.
"""

from __future__ import annotations

import os
import uuid
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


@app.post("/api/test/analyze-recording")
async def analyze_recording(
    audio: UploadFile = File(...),
    prospect_id: str = Form("test_prospect_001"),
    call_sid: Optional[str] = Form(None),
):
    """Processes uploaded prospect audio through the full behavioral pipeline:
    1. STT (Deepgram or Groq Whisper) with word-level timestamps.
    2. Deterministic timing analysis (WPM, pause duration, response latency).
    3. Semantic feature extraction (intent, recurrence, boundary).
    4. Baseline & change-point detection (z-score departures).
    5. Multi-window evidence frame construction.
    6. Downstream inference scoring (Trust, Pacing, Engagement, Momentum, Readiness, Emotion).
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    active_call_sid = call_sid or f"call_{uuid.uuid4().hex[:8]}"
    session = get_or_create_session(prospect_id, active_call_sid)
    session["turn_index"] += 1

    timing_engine: DeterministicTimingEngine = session["timing_engine"]
    semantic_engine: SemanticFeatureEngine = session["semantic_engine"]
    baseline_engine: BaselineAndChangePointEngine = session["baseline_engine"]
    aggregator: MultiWindowAggregator = session["aggregator"]

    # 1. Transcribe audio with word-level timestamps and pre-recorded utterances
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

    # 2. Segment audio into distinct utterance turns (Deepgram utterances or pause gaps >= 500ms)
    segmented_utts = segment_audio_transcript_turns(
        words_raw=words_raw,
        transcript=transcript,
        call_sid=active_call_sid,
        speaker_id="client",
        raw_utterances=raw_utterances,
        min_pause_split_ms=500,
    )
    LOGGER.info("Segmented audio into %d sequential turn(s)", len(segmented_utts))

    last_timing_snap = None
    last_sem_snap = None
    last_evidence_frame = None
    last_inference_state = None

    # 3. Process each turn sequentially through the behavioral pipeline
    for norm_utt in segmented_utts:
        session["turn_index"] += 1

        # Timing
        timing_snap = timing_engine.process_utterance(norm_utt)
        last_timing_snap = timing_snap

        # Semantic
        sem_snap = None
        try:
            sem_snap = await semantic_engine.analyze_turn_semantic(
                utterance=norm_utt,
                context_history=session["context_history"],
            )
        except Exception as exc:
            LOGGER.warning("Semantic feature extraction error: %s", exc)
        last_sem_snap = sem_snap
        session["context_history"].append(norm_utt)

        # Baseline profile comparison & change points
        new_cps = baseline_engine.update_with_utterance(norm_utt, timing_snap, sem_snap)

        # Multi-window evidence frame
        evidence_frame = aggregator.process_turn(
            norm_utt, timing_snap, sem_snap, new_change_points=new_cps
        )
        last_evidence_frame = evidence_frame

        # Downstream inference scoring
        inference_state = inference_engine.compute_inference(
            call_sid=active_call_sid,
            current_frame=evidence_frame,
        )
        last_inference_state = inference_state

        LOGGER.info(
            "Turn %d analyzed: transcript='%s' | WPM=%.1f | Pacing=%.2f | Trust=%.2f | Readiness=%.2f",
            session["turn_index"],
            norm_utt.text[:40],
            timing_snap.speech_rate_wpm,
            inference_state.pacing.score,
            inference_state.trust.score,
            inference_state.readiness.score,
        )

    elapsed_ms = (
        (last_timing_snap.timestamp_ms - baseline_engine.earliest_sample_ms)
        if (last_timing_snap and baseline_engine.earliest_sample_ms is not None)
        else 0
    )
    calib_turns = len(baseline_engine.prospect_samples["turn_length_words"])
    calib_words = int(sum(baseline_engine.prospect_samples["turn_length_words"]))
    elapsed_sec = round(elapsed_ms / 1000.0, 1)
    target_sec = round(baseline_engine.intra_call_window_ms / 1000.0, 1)

    return {
        "status": "success",
        "call_sid": active_call_sid,
        "prospect_id": prospect_id,
        "turns_processed": len(segmented_utts),
        "turn_index": session["turn_index"],
        "is_baseline_locked": baseline_engine.is_intra_call_locked,
        "calibration_progress": {
            "is_locked": baseline_engine.is_intra_call_locked,
            "elapsed_sec": elapsed_sec,
            "target_sec": target_sec,
            "turn_count": calib_turns,
            "target_turns": baseline_engine.min_calibration_turns,
            "total_words": calib_words,
            "target_words": baseline_engine.min_cumulative_words,
        },
        "transcript": transcript,
        "utterances": [
            {
                "turn_index": session["turn_index"] - len(segmented_utts) + idx + 1,
                "text": u.text,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
                "words_count": len(u.words),
            }
            for idx, u in enumerate(segmented_utts)
        ],
        "words_count": sum(len(u.words) for u in segmented_utts),
        "latest_timing": last_timing_snap.model_dump() if last_timing_snap else {},
        "latest_inference": last_inference_state.model_dump() if last_inference_state else {},
        "latest_evidence_frame": last_evidence_frame.model_dump() if last_evidence_frame else {},
    }


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
    print("=" * 65 + "\n")

    uvicorn.run("run_behavioral_signal:app", host=host, port=port, reload=False, log_level="info")
