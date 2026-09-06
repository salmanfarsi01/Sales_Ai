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
