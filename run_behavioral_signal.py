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
    timing_engine: DeterministicTimingEngine = session["timing_engine"]
    semantic_engine: SemanticFeatureEngine = session["semantic_engine"]
    baseline_engine: BaselineAndChangePointEngine = session["baseline_engine"]
    aggregator: MultiWindowAggregator = session["aggregator"]

    last_timing_snap = None
    last_sem_snap = None
    last_evidence_frame = None
    last_inference_state = None
    play_by_play_turns = []

    for norm_utt in segmented_utts:
        session["turn_index"] += 1
        speaker_role = norm_utt.speaker_id  # "salesperson" or "client"
        speaker_label = "Agent" if speaker_role == "salesperson" else "Prospect"

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

        # Baseline profile comparison & change points (isolated per speaker)
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

        active_prof = baseline_engine.get_active_profile(speaker_role)
        base_wpm = (
            round(active_prof.speech_rate_wpm.mean, 1)
            if (active_prof and active_prof.speech_rate_wpm.sample_count > 0)
            else None
        )
        curr_wpm = timing_snap.turn_speech_rate_wpm or round(timing_snap.speech_rate_wpm, 1)
        pace_delta = (
            round(((curr_wpm - base_wpm) / base_wpm) * 100.0, 1)
            if (base_wpm and base_wpm > 0)
            else None
        )
        base_pause_ms = (
            round(active_prof.avg_pause_duration_ms.mean, 1)
            if (active_prof and active_prof.avg_pause_duration_ms.sample_count > 0)
            else None
        )
        pause_delta = (
            round(((timing_snap.avg_pause_duration_ms - base_pause_ms) / base_pause_ms) * 100.0, 1)
            if (base_pause_ms and base_pause_ms > 0 and timing_snap.pause_measured)
            else None
        )

        play_by_play_turns.append({
            "turn_index": session["turn_index"],
            "speaker_id": speaker_role,
            "speaker_label": speaker_label,
            "start_ms": norm_utt.start_ms,
            "end_ms": norm_utt.end_ms,
            "start_sec": round(norm_utt.start_ms / 1000.0, 1),
            "end_sec": round(norm_utt.end_ms / 1000.0, 1),
            "text": norm_utt.text,
            "words_count": len(norm_utt.words) if norm_utt.words else len(norm_utt.text.split()),
            "turn_duration_ms": timing_snap.turn_duration_ms,
            "speech_rate_wpm": curr_wpm,
            "turn_wpm": curr_wpm,
            "rolling_wpm_60s": round(timing_snap.speech_rate_wpm, 1),
            "baseline_wpm": base_wpm,
            "pace_delta_pct": pace_delta,
            "avg_pause_duration_ms": round(timing_snap.avg_pause_duration_ms, 1),
            "baseline_pause_ms": base_pause_ms,
            "pause_delta_pct": pause_delta,
            "response_latency_ms": timing_snap.response_latency_ms,
            "question_type": sem_snap.question_type if sem_snap else "none",
            "recurrence_type": sem_snap.recurrence_type if sem_snap else "none",
            "recurrence_count": sem_snap.recurrence_count if sem_snap else 0,
            "boundary_score": sem_snap.boundary_score if sem_snap else 0.0,
            "specificity_score": sem_snap.specificity_score if sem_snap else 0.0,
            "future_language_score": sem_snap.future_language_score if sem_snap else 0.0,
            "agreement_score": sem_snap.agreement_score if sem_snap else 0.0,
            "semantic_confidence": sem_snap.semantic_confidence if sem_snap else 0.50,
            "pacing_score": inference_state.pacing.score,
            "trust_score": inference_state.trust.score,
            "engagement_score": inference_state.engagement.score,
            "momentum_score": inference_state.momentum.score,
            "readiness_score": inference_state.readiness.score,
            "expressed_valence": inference_state.emotion.expressed_valence if inference_state.emotion else 0.0,
            "tension_level": inference_state.emotion.tension_level if inference_state.emotion else 0.20,
            "is_baseline_locked": baseline_engine.is_locked(speaker_role),
        })

        LOGGER.info(
            "Turn %d [%s] analyzed: transcript='%s' | TurnWPM=%.1f (Base=%s) | Latency=%s | Pacing=%.2f | Trust=%.2f",
            session["turn_index"],
            speaker_label,
            norm_utt.text[:40],
            curr_wpm,
            str(base_wpm),
            str(timing_snap.response_latency_ms),
            inference_state.pacing.score,
            inference_state.trust.score,
        )

    elapsed_ms = (
        (last_timing_snap.timestamp_ms - baseline_engine.earliest_sample_ms)
        if (last_timing_snap and baseline_engine.earliest_sample_ms is not None)
        else 0
    )
    prospect_turns = len(baseline_engine.speaker_samples["client"]["turn_length_words"])
    prospect_words = int(sum(baseline_engine.speaker_samples["client"]["turn_length_words"]))
    agent_turns = len(baseline_engine.speaker_samples["salesperson"]["turn_length_words"])
    agent_words = int(sum(baseline_engine.speaker_samples["salesperson"]["turn_length_words"]))
    elapsed_sec = round(elapsed_ms / 1000.0, 1)
    target_sec = round(baseline_engine.intra_call_window_ms / 1000.0, 1)

    result_payload = {
        "status": "success",
        "call_sid": active_call_sid,
        "prospect_id": prospect_id,
        "turns_processed": len(segmented_utts),
        "turn_index": session["turn_index"],
        "is_baseline_locked": baseline_engine.is_locked("client"),
        "agent_baseline_locked": baseline_engine.is_locked("salesperson"),
        "prospect_baseline_locked": baseline_engine.is_locked("client"),
        "calibration_progress": {
            "is_locked": baseline_engine.is_locked("client"),
            "agent_is_locked": baseline_engine.is_locked("salesperson"),
            "prospect_is_locked": baseline_engine.is_locked("client"),
            "elapsed_sec": elapsed_sec,
            "target_sec": target_sec,
            "turn_count": prospect_turns,
            "target_turns": baseline_engine.min_calibration_turns,
            "total_words": prospect_words,
            "target_words": baseline_engine.min_cumulative_words,
            "agent_turn_count": agent_turns,
            "agent_total_words": agent_words,
        },
        "transcript": transcript,
        "utterances": [
            {
                "turn_index": session["turn_index"] - len(segmented_utts) + idx + 1,
                "speaker_id": u.speaker_id,
                "speaker_label": "Agent" if u.speaker_id == "salesperson" else "Prospect",
                "text": u.text,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
                "words_count": len(u.words),
            }
            for idx, u in enumerate(segmented_utts)
        ],
        "play_by_play_turns": play_by_play_turns,
        "words_count": sum(len(u.words) for u in segmented_utts),
        "latest_timing": last_timing_snap.model_dump() if last_timing_snap else {},
        "latest_inference": last_inference_state.model_dump() if last_inference_state else {},
        "latest_evidence_frame": last_evidence_frame.model_dump() if last_evidence_frame else {},
    }

    try:
        reports_dir = Path(__file__).resolve().parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_filename = f"behavioral_signal_{active_call_sid}.json"
        report_path = reports_dir / report_filename
        result_payload["report_file"] = f"reports/{report_filename}"
        report_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        LOGGER.info("Successfully stored behavioral signal report to local disk: %s", report_path)
    except Exception as exc:
        LOGGER.warning("Could not save behavioral signal report to disk: %s", exc)

    return result_payload


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
    prospect_id: str = Form("test_prospect_001"),
    agent_id: str = Form("test_agent_001"),
    call_sid: Optional[str] = Form(None),
):
    """Processes two independently recorded single-speaker tracks (Agent & Prospect),
    transcribes them separately, merges them onto a synchronized chronological timeline,
    and runs the full behavioral pipeline with isolated per-speaker baselines.
    """
    agent_bytes = await agent_audio.read()
    prospect_bytes = await prospect_audio.read()

    if not agent_bytes and not prospect_bytes:
        raise HTTPException(status_code=400, detail="Both audio tracks are empty")

    active_call_sid = call_sid or f"call_{uuid.uuid4().hex[:8]}"
    session = get_or_create_session(prospect_id, active_call_sid)
    session["baseline_engine"].agent_id = agent_id

    stt = SpeechToTextEngine(
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    agent_segmented: List[NormalizedUtterance] = []
    if agent_bytes:
        agent_mime = agent_audio.content_type or "audio/webm"
        LOGGER.info("Transcribing Agent track: %d bytes (%s)...", len(agent_bytes), agent_mime)
        agent_transcript, agent_words = await stt.transcribe_with_timestamps(
            agent_bytes, mime_type=agent_mime, utt_split=0.5
        )
        agent_raw_utts = getattr(stt, "latest_raw_utterances", None) or []
        agent_segmented = segment_audio_transcript_turns(
            words_raw=agent_words,
            transcript=agent_transcript or "",
            call_sid=active_call_sid,
            speaker_id="salesperson",
            raw_utterances=agent_raw_utts,
            min_pause_split_ms=500,
        )
        for u in agent_segmented:
            u.speaker_id = "salesperson"

    prospect_segmented: List[NormalizedUtterance] = []
    if prospect_bytes:
        prospect_mime = prospect_audio.content_type or "audio/webm"
        LOGGER.info("Transcribing Prospect track: %d bytes (%s)...", len(prospect_bytes), prospect_mime)
        prospect_transcript, prospect_words = await stt.transcribe_with_timestamps(
            prospect_bytes, mime_type=prospect_mime, utt_split=0.5
        )
        prospect_raw_utts = getattr(stt, "latest_raw_utterances", None) or []
        prospect_segmented = segment_audio_transcript_turns(
            words_raw=prospect_words,
            transcript=prospect_transcript or "",
            call_sid=active_call_sid,
            speaker_id="client",
            raw_utterances=prospect_raw_utts,
            min_pause_split_ms=500,
        )
        for u in prospect_segmented:
            u.speaker_id = "client"

    merged_utts = merge_dual_speaker_tracks(
        agent_utterances=agent_segmented,
        prospect_utterances=prospect_segmented,
        agent_start_epoch_ms=agent_start_epoch_ms,
        prospect_start_epoch_ms=prospect_start_epoch_ms,
    )
    LOGGER.info(
        "Merged dual tracks: %d agent turns + %d prospect turns -> %d total chronological turns",
        len(agent_segmented),
        len(prospect_segmented),
        len(merged_utts),
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
