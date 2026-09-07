import asyncio
import io
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from copilot.calibration import (
    CalibrationService,
    ElevenLabsTTSClient,
    SpeechToTextEngine,
    get_audio_duration_seconds,
    analyze_audio_signal,
    RoundScoreBreakdown,
    RoundEvaluation,
    AUDIO_CACHE_DIR,
)


def _make_dummy_wav(duration_sec: float = 2.0, sample_rate: int = 8000) -> bytes:
    """Generate in-memory PCM WAV bytes for testing duration parser."""
    num_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _make_dummy_webm_ebml(duration_sec: float = 3.5) -> bytes:
    """Generate in-memory WebM bytes containing an EBML 0x4489 duration element."""
    # EBML Header + Segment Info with Duration element (0x4489, 0x84, float32)
    ebml_header = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01"
    duration_element = b"\x44\x89\x84" + struct.pack(">f", duration_sec)
    filler = b"\x00" * 5000
    return ebml_header + duration_element + filler


@pytest.mark.asyncio
async def test_communication_style_profiling_and_paraphrasing():
    """Verify speech is profiled on delivery style dimensions regardless of script match."""
    service = CalibrationService()
    target_script = "We guarantee 99.99% uptime with enterprise SLAs and dedicated support."
    # Rep paraphrases completely differently
    spoken_paraphrase = "Our infrastructure runs with extreme high availability, backed by immediate round the clock engineering assistance."

    eval_result = await service._run_llm_evaluation(
        round_num=1,
        stage="Discovery",
        target_teleprompt=target_script,
        user_transcript=spoken_paraphrase,
        duration_seconds=5.0,
    )

    # Asserts valid style profile is generated without teleprompt accuracy penalty
    assert eval_result["overall_score"] >= 50
    assert eval_result["confidence"] == 1.0
    assert eval_result["is_excluded"] is False
    assert "score_breakdown" in eval_result

    b = eval_result["score_breakdown"]
    assert "speaking_pace" in b
    assert "vocabulary_complexity" in b
    assert "sentence_handling" in b
    assert "pause_pattern" in b
    assert eval_result["wpm"] > 100
    assert len(eval_result["style_traits"]) >= 2
    assert "teleprompter alignment" in eval_result["improvements"][0].lower()


@pytest.mark.asyncio
async def test_sample_quality_and_silence_exclusion():
    """Verify silence / bad audio is flagged as low confidence and excluded, NEVER given 0% failing grade."""
    service = CalibrationService()
    target = "Our platform is SOC 2 Type II certified and GDPR compliant."

    # 1. Total Silence / Empty speech
    silence_res = await service._run_llm_evaluation(
        round_num=3,
        stage="Security",
        target_teleprompt=target,
        user_transcript="",
        duration_seconds=2.0,
    )
    # Must NOT receive 0% failing grade
    assert silence_res["overall_score"] > 0
    assert silence_res["confidence"] == 0.0
    assert silence_res["is_excluded"] is True
    assert "excluded from your communication profile" in silence_res["feedback"].lower()
    assert "excluded" in silence_res["style_traits"][0].lower()

    # 2. Too brief / inaudible noise
    short_res = await service._run_llm_evaluation(
        round_num=4,
        stage="Pricing",
        target_teleprompt=target,
        user_transcript="uh",
        duration_seconds=0.8,
    )
    assert short_res["confidence"] == 0.0
    assert short_res["is_excluded"] is True


def test_prompting_level_resolution_and_formula():
    """Verify score bands resolve to Direct, Balanced, and Layered prompting levels."""
    from copilot.calibration import CALIBRATION_FORMULA_VERSION, resolve_prompting_level

    assert CALIBRATION_FORMULA_VERSION == "v2.0-style-profiling"

    direct_profile = resolve_prompting_level(82)
    assert direct_profile["level_id"] == "direct"
    assert "Direct" in direct_profile["name"]

    balanced_profile = resolve_prompting_level(65)
    assert balanced_profile["level_id"] == "balanced"
    assert "Balanced" in balanced_profile["name"]

    layered_profile = resolve_prompting_level(45)
    assert layered_profile["level_id"] == "layered"
    assert "Layered" in layered_profile["name"]


def _make_audio_with_dynamics(duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    import numpy as np
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, False)
    # Speech-like modulated envelope with pauses between syllables
    envelope = np.maximum(0.0, np.sin(2 * np.pi * 2.5 * t)) * (0.4 + 0.6 * np.sin(2 * np.pi * 0.7 * t))
    carrier = np.sin(2 * np.pi * 320 * t) + 0.5 * np.sin(2 * np.pi * 640 * t)
    pcm = (carrier * envelope * 24000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _make_flat_monotone_audio(duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    import numpy as np
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, False)
    pcm = (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_pause_cadence_with_word_timestamps():
    """Verify pause cadence measures actual inter-word timing gaps, hesitations, and filler count."""
    service = CalibrationService()
    transcript = "We deliver enterprise security with complete data isolation and instant deployment."

    # 1. Fluid cadence: 5 words with natural 150-250ms gaps between words
    fluid_timings = [
        {"word": "we", "start": 0.2, "end": 0.5},
        {"word": "deliver", "start": 0.7, "end": 1.1},
        {"word": "enterprise", "start": 1.3, "end": 1.9},
        {"word": "security", "start": 2.1, "end": 2.7},
        {"word": "today", "start": 2.9, "end": 3.3},
    ]
    fluid_score = service.compute_pause_cadence(
        transcript=transcript,
        duration_seconds=3.5,
        word_count=5,
        detected_fillers=[],
        word_timings=fluid_timings,
    )
    assert fluid_score >= 90, f"Expected high fluid score, got {fluid_score}"

    # 2. Hesitant cadence: long gaps (1.4s) and filler words
    hesitant_timings = [
        {"word": "we", "start": 0.2, "end": 0.5},
        {"word": "um", "start": 1.9, "end": 2.2},  # 1.4s gap
        {"word": "deliver", "start": 3.5, "end": 3.9},  # 1.3s gap
    ]
    hesitant_score = service.compute_pause_cadence(
        transcript="we um deliver",
        duration_seconds=4.5,
        word_count=3,
        detected_fillers=["um"],
        word_timings=hesitant_timings,
    )
    assert hesitant_score < fluid_score
    assert hesitant_score <= 75, f"Expected lower hesitant score, got {hesitant_score}"


def test_response_timing_measurement():
    """Verify response timing uses actual continuous latency rather than a hardcoded 78 constant."""
    service = CalibrationService()

    # Fast / prompt response (400ms)
    fast_score = service.compute_response_timing(latency_ms=400.0)
    assert fast_score >= 88, f"Expected >= 88 for fast response, got {fast_score}"

    # Conversational response (1200ms)
    med_score = service.compute_response_timing(latency_ms=1200.0)
    assert 74 <= med_score <= 84, f"Expected ~78 for 1200ms, got {med_score}"

    # Delayed response (3500ms)
    slow_score = service.compute_response_timing(latency_ms=3500.0)
    assert slow_score <= 60, f"Expected <= 60 for delayed response, got {slow_score}"

    # Auto-start near-zero client latency with natural speech onset (15ms client + 350ms speech onset = 365ms)
    auto_score = service.compute_response_timing(latency_ms=15.0, speech_onset_sec=0.35)
    assert auto_score >= 88, f"Expected >= 88 for auto-start with quick onset, got {auto_score}"

    # Verify scores are strictly differentiated
    assert fast_score > med_score > slow_score


def test_energy_dynamics_with_audio_signals():
    """Verify vocal energy dynamics derives real dynamic range and modulation from audio signal."""
    service = CalibrationService()

    dynamic_wav = _make_audio_with_dynamics(duration_sec=2.5)
    flat_wav = _make_flat_monotone_audio(duration_sec=2.5)

    dyn_score = service.compute_energy_dynamics(audio_bytes=dynamic_wav, transcript="We guarantee 99.9% uptime!")
    flat_score = service.compute_energy_dynamics(audio_bytes=flat_wav, transcript="we guarantee 99.9 uptime")

    # Dynamic speech audio must score distinctly higher than flat monotone audio
    assert dyn_score > flat_score
    assert dyn_score >= 75, f"Expected dynamic speech to score >= 75, got {dyn_score}"
    assert flat_score <= 65, f"Expected flat monotone to score <= 65, got {flat_score}"


@pytest.mark.asyncio
async def test_evaluate_round_with_latency_and_timing_telemetry():
    """Verify evaluate_round accepts and propagates response_latency_ms into score_breakdown."""
    service = CalibrationService()
    session = await service.create_dynamic_session(user_id="test_rep")
    session_id = session["session_id"]

    dynamic_wav = _make_audio_with_dynamics(duration_sec=3.0)

    # Prompt response round
    round_result = await service.evaluate_round(
        session_id=session_id,
        round_num=1,
        audio_bytes=dynamic_wav,
        mime_type="audio/wav",
        transcript_override="We provide automated compliance with real time alerts across your cloud infrastructure.",
        duration_seconds=3.0,
        response_latency_ms=450.0,
    )

    ev = round_result["evaluation"]
    breakdown = ev["score_breakdown"]
    assert breakdown["response_timing"] >= 85
    assert breakdown["energy_dynamics"] >= 72
    assert ev["overall_score"] > 0
    assert ev["confidence"] == 1.0


def test_energy_dynamics_ffmpeg_failsafe():
    """Verify compute_energy_dynamics never throws on corrupt or unsupported audio bytes and returns safe fallback."""
    service = CalibrationService()
    garbage_bytes = b"RIFF" + b"\xff" * 2000  # corrupted header & invalid PCM

    # Must fail safely without raising 500 error or Exception
    safe_score = service.compute_energy_dynamics(audio_bytes=garbage_bytes, transcript="Safe fallback test.")
    assert safe_score == 74 or safe_score == 80
    assert isinstance(safe_score, int)


def test_energy_dynamics_low_frame_guard():
    """Verify sparse/low active speech frame audio falls back safely rather than computing noisy percentiles."""
    service = CalibrationService()
    # 0.3s audio (only 6 frames, below the 12-frame threshold)
    short_wav = _make_flat_monotone_audio(duration_sec=0.3)

    score = service.compute_energy_dynamics(audio_bytes=short_wav, transcript="Brief test.")
    assert score == 74 or score == 80


def test_response_timing_sanity_bounds():
    """Verify out-of-bounds client latency (<50ms or >15000ms) is sanitized and discarded."""
    service = CalibrationService()

    # Negative skew (client clock drift or tamper) -> sanitized to fallback
    neg_score = service.compute_response_timing(latency_ms=-500.0)
    assert neg_score == 78

    # Background tab sleep (> 15 seconds) -> sanitized to fallback
    tab_sleep_score = service.compute_response_timing(latency_ms=30000.0)
    assert tab_sleep_score == 78


def test_static_content_bank_structure():
    """Verify both tracks contain exactly 7 rounds, 4 variants (A-D), and final statements."""
    from copilot.calibration_content import CALIBRATION_TRACKS
    
    assert "seller" in CALIBRATION_TRACKS
    assert "internal_rep" in CALIBRATION_TRACKS

    for track_key, track_data in CALIBRATION_TRACKS.items():
        assert len(track_data["rounds"]) == 7, f"Track {track_key} must have 7 rounds"
        assert track_data["final_statement"].strip(), f"Track {track_key} missing final statement"
        assert track_data["voice_id"], f"Track {track_key} missing voice_id"

        for idx, r in enumerate(track_data["rounds"], 1):
            assert r["round_number"] == idx
            assert r["title"].strip()
            assert r["seller_line"].strip()
            variants = r["response_variants"]
            assert set(variants.keys()) == {"A", "B", "C", "D"}
            for v_key, v_text in variants.items():
                assert len(v_text.strip()) > 10, f"Round {idx} variant {v_key} too short"


@pytest.mark.asyncio
async def test_variant_secrecy_and_round_generation():
    """Verify generate_round_voice_on_demand returns only the chosen variant without leaking the others."""
    from copilot.calibration_content import CALIBRATION_TRACKS

    service = CalibrationService()
    session = await service.create_calibration_session(track="seller")
    s_id = session["session_id"]

    round_info = await service.generate_round_voice_on_demand(s_id, 1)

    # Must contain only teleprompt_text, NOT the whole response_variants map
    assert "response_variants" not in round_info
    assert round_info["selected_variant"] in ["A", "B", "C", "D"]
    
    # Selected text must match the bank's entry for the chosen variant
    expected_text = CALIBRATION_TRACKS["seller"]["rounds"][0]["response_variants"][round_info["selected_variant"]]
    assert round_info["teleprompt_text"] == expected_text


@pytest.mark.asyncio
async def test_round_7_completion_triggers_payoff():
    """Verify completing round 7 transitions session to completed, synthesizes final statement payoff, and does not exceed 7."""
    service = CalibrationService()
    session = await service.create_calibration_session(track="internal_rep")
    s_id = session["session_id"]

    # Play rounds 1 through 7
    for r in range(1, 8):
        await service.generate_round_voice_on_demand(s_id, r)
        res = await service.evaluate_round(
            session_id=s_id,
            round_num=r,
            transcript_override=session["rounds"][r]["teleprompt_text"],
            duration_seconds=3.0,
        )
        if r < 7:
            assert res["completed"] is True
            assert session["status"] == "in_progress"
            assert session["current_round"] == r + 1
        else:
            assert res["completed"] is True
            assert session["status"] == "completed"
            assert session["current_round"] == 7
            assert session["final_audio_url"] is not None
            assert session["summary_report"] is not None
            assert session["summary_report"]["rounds_completed"] == 7
            assert session["summary_report"]["track"] == "internal_rep"

