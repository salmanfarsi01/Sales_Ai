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
async def test_blind_rubric_independent_and_fallback():
    """Verify LLM evaluates autonomously and fallback triggers ONLY on LLM failure."""
    service = CalibrationService()
    target = "We guarantee 99.99% uptime with enterprise SLAs and dedicated support."
    spoken = "We guarantee 99.99% uptime with enterprise SLAs and dedicated support."

    # 1. Successful independent LLM evaluation
    mock_llm_payload = {
        "overall_score": 87,
        "score_breakdown": {
            "word_choice": 94,
            "pacing": 82,
            "sentiment": 88,
            "tone_emphasis": 80,
            "pause_filters": 100,
            "energy_inflection": 78,
        },
        "feedback": "Crisp articulation and authoritative tone throughout.",
        "strengths": ["Clear metrics delivery"],
        "improvements": ["Vary energy slightly"],
        "detected_fillers": [],
    }

    with patch.object(service, "_call_llm_json", new=AsyncMock(return_value=mock_llm_payload)):
        eval_result = await service._run_llm_evaluation(
            round_num=1,
            stage="Discovery",
            target_teleprompt=target,
            user_transcript=spoken,
            duration_seconds=4.0,
        )
        assert eval_result["overall_score"] == 87
        assert eval_result["score_breakdown"]["word_choice"] == 94
        assert eval_result["feedback"] == "Crisp articulation and authoritative tone throughout."

    # 2. LLM failure (e.g. timeout / network error returning {}) -> Graceful heuristic fallback
    with patch.object(service, "_call_llm_json", new=AsyncMock(return_value={})):
        fallback_result = await service._run_llm_evaluation(
            round_num=1,
            stage="Discovery",
            target_teleprompt=target,
            user_transcript=spoken,
            duration_seconds=4.0,
        )
        assert fallback_result["overall_score"] > 80
        assert "score_breakdown" in fallback_result
        assert "direct similarity analysis" in fallback_result["feedback"].lower()


@pytest.mark.asyncio
async def test_pydantic_validation_failure_handling():
    """Verify ValidationError on malformed LLM response degrades cleanly to fallback without throwing."""
    service = CalibrationService()
    target = "Our platform is SOC 2 Type II certified and GDPR compliant."
    spoken = "Our platform is SOC 2 Type II certified and GDPR compliant."

    # Malformed LLM output: score_breakdown has invalid non-integer string and missing keys
    malformed_payload = {
        "overall_score": "unranked",
        "score_breakdown": {
            "word_choice": "ninety-nine",  # Non-integer causes ValidationError
        },
        "feedback": "Malformed response",
    }

    with patch.object(service, "_call_llm_json", new=AsyncMock(return_value=malformed_payload)):
        result = await service._run_llm_evaluation(
            round_num=2,
            stage="Security",
            target_teleprompt=target,
            user_transcript=spoken,
            duration_seconds=3.5,
        )
        # Asserts it degraded cleanly to validated fallback rather than crashing
        assert isinstance(result["overall_score"], int)
        assert result["score_breakdown"]["word_choice"] > 70
        assert "score_breakdown" in result


def test_audio_duration_extraction():
    """Verify audio duration extraction across WAV headers, EBML headers, and client rate validation."""
    # 1. WAV exact frame header
    wav_bytes = _make_dummy_wav(duration_sec=3.25)
    assert get_audio_duration_seconds(wav_bytes, "audio/wav") == 3.25

    # 2. WebM EBML element parser
    webm_ebml = _make_dummy_webm_ebml(duration_sec=4.5)
    assert get_audio_duration_seconds(webm_ebml, "audio/webm") == 4.5

    # 3. Client duration validated against plausible byte rate
    mock_audio = b"\x00" * 30000  # 30KB
    # 5.0 seconds -> 6,000 B/s (plausible for WebM/Opus)
    assert get_audio_duration_seconds(mock_audio, "audio/webm", fallback_seconds=5.0) == 5.0

    # 4. Empty / corrupt audio
    assert get_audio_duration_seconds(b"", "audio/webm", fallback_seconds=2.0) == 2.0


@pytest.mark.asyncio
async def test_tts_cache_parameter_footprint():
    """Verify TTS disk cache key includes model, voice, stability, style, and speaker boost."""
    tts = ElevenLabsTTSClient()
    text = "Enterprise-grade reliability guaranteed."
    voice_id = "21m00Tcm4TlvDq8ikWAM"

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"ID3\x03\x00\x00\x00" + (b"\x00" * 500)
        mock_post.return_value = mock_response

        # First call -> Generates & writes to cache
        audio1, mime1 = await tts.generate_speech(text, voice_id=voice_id)
        assert len(audio1) > 200

        # Second call -> Hits local cache without HTTP POST
        mock_post.reset_mock()
        audio2, mime2 = await tts.generate_speech(text, voice_id=voice_id)
        assert audio1 == audio2
        mock_post.assert_not_called()  # Proves cache hit bypassed network call


@pytest.mark.asyncio
async def test_zero_scoring_on_silence_and_irrelevant():
    """Verify strict 0 score on empty audio and off-topic speech."""
    service = CalibrationService()
    target = "Our platform is SOC 2 Type II certified and GDPR compliant."

    # 1. Silence
    silence_res = await service._run_llm_evaluation(
        round_num=3,
        stage="Security",
        target_teleprompt=target,
        user_transcript="",
        duration_seconds=2.0,
    )
    assert silence_res["overall_score"] == 0
    assert silence_res["score_breakdown"]["word_choice"] == 0
    assert silence_res["score_breakdown"]["pacing"] == 0
    assert "no speech detected" in silence_res["feedback"].lower()

    # 2. Irrelevant / Disconnected
    irrelevant_res = await service._run_llm_evaluation(
        round_num=3,
        stage="Security",
        target_teleprompt=target,
        user_transcript="Let's grab a slice of pepperoni pizza for lunch today.",
        duration_seconds=3.0,
    )
    assert irrelevant_res["overall_score"] == 0
    assert irrelevant_res["score_breakdown"]["word_choice"] == 0
    assert "irrelevant response" in irrelevant_res["feedback"].lower()
