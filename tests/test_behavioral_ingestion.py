from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from copilot.behavioral_normalization import (
    CallMetadata,
    NormalizedWord,
    NormalizedUtterance,
    NormalizedTurnEvent,
    normalize_deepgram_result,
    normalize_generic_transcript,
    detect_turn_events,
)
from copilot.fastapi_app import app


def test_normalize_deepgram_result_with_words():
    raw_payload = {
        "type": "Results",
        "channel_index": [0, 1],
        "duration": 2.45,
        "start": 1.12,
        "is_final": True,
        "speech_final": True,
        "channel": {
            "alternatives": [
                {
                    "transcript": "Hello, how are you today?",
                    "confidence": 0.985,
                    "words": [
                        {"word": "hello", "start": 1.12, "end": 1.45, "confidence": 0.99, "punctuated_word": "Hello,"},
                        {"word": "how", "start": 1.50, "end": 1.68, "confidence": 0.98, "punctuated_word": "how"},
                        {"word": "are", "start": 1.70, "end": 1.82, "confidence": 0.97, "punctuated_word": "are"},
                        {"word": "you", "start": 1.84, "end": 1.99, "confidence": 0.99, "punctuated_word": "you"},
                        {"word": "today", "start": 2.05, "end": 2.45, "confidence": 0.98, "punctuated_word": "today?"},
                    ],
                }
            ]
        },
    }

    result = normalize_deepgram_result(
        raw_result=raw_payload,
        call_sid="call_test_123",
        speaker_id="client",
        stream_offset_ms=500,
    )

    assert result is not None
    assert result.call_sid == "call_test_123"
    assert result.speaker_id == "client"
    assert result.text == "Hello, how are you today?"
    assert result.is_final is True
    assert result.speech_final is True
    assert result.start_ms == 1620
    assert result.end_ms == 2950
    assert result.is_estimated_timing is False
    assert len(result.words) == 5

    first_word = result.words[0]
    assert first_word.word == "hello"
    assert first_word.punctuated_word == "Hello,"
    assert first_word.start_ms == 1620
    assert first_word.end_ms == 1950
    assert first_word.confidence == 0.99
    assert first_word.is_estimated is False

    last_word = result.words[-1]
    assert last_word.word == "today"
    assert last_word.punctuated_word == "today?"
    assert last_word.start_ms == 2550
    assert last_word.end_ms == 2950
    assert last_word.is_estimated is False


def test_normalize_deepgram_result_synthetic_words_fallback():
    raw_payload = {
        "type": "Results",
        "duration": 1.50,
        "start": 0.50,
        "is_final": True,
        "speech_final": False,
        "channel": {
            "alternatives": [
                {
                    "transcript": "Quick update on the listing.",
                    "confidence": 0.92,
                    "words": [],
                }
            ]
        },
    }

    result = normalize_deepgram_result(
        raw_result=raw_payload,
        call_sid="call_test_456",
        speaker_id="salesperson",
        stream_offset_ms=0,
    )

    assert result is not None
    assert result.is_estimated_timing is True
    assert result.start_ms == 500
    assert result.end_ms == 2000
    assert len(result.words) == 5
    assert [w.word for w in result.words] == ["Quick", "update", "on", "the", "listing."]
    assert result.words[0].start_ms == 500
    assert result.words[0].is_estimated is True
    assert result.words[-1].end_ms == 2000
    assert result.words[-1].is_estimated is True


def test_normalize_deepgram_result_empty_or_malformed():
    assert normalize_deepgram_result({}, "call_1", "client") is None
    assert normalize_deepgram_result({"channel": {}}, "call_1", "client") is None
    assert normalize_deepgram_result({"channel": {"alternatives": []}}, "call_1", "client") is None

    empty_text_payload = {
        "channel": {
            "alternatives": [
                {"transcript": "   ", "confidence": 0.0}
            ]
        }
    }
    assert normalize_deepgram_result(empty_text_payload, "call_1", "client") is None


def test_normalize_generic_transcript():
    result = normalize_generic_transcript(
        text="I want to sell my property.",
        speaker_id="client",
        start_ms=1000,
        end_ms=2500,
        call_sid="call_test_789",
        confidence=0.95,
    )

    assert result.call_sid == "call_test_789"
    assert result.speaker_id == "client"
    assert result.text == "I want to sell my property."
    assert result.start_ms == 1000
    assert result.end_ms == 2500
    assert len(result.words) == 6
    assert result.words[0].start_ms == 1000
    assert result.words[-1].end_ms == 2500


def test_detect_turn_events_silence_and_overlap():
    utt1 = normalize_generic_transcript(
        text="Are you looking to list this month?",
        speaker_id="salesperson",
        start_ms=1000,
        end_ms=3000,
        call_sid="call_turn_test",
    )

    utt2_silence = normalize_generic_transcript(
        text="Yes, we need to sell before October.",
        speaker_id="client",
        start_ms=3800,
        end_ms=5500,
        call_sid="call_turn_test",
    )

    events_silence = detect_turn_events(utt2_silence, utt1)
    event_types = [e.event_type for e in events_silence]
    assert "turn_start" in event_types
    assert "speaker_switch" in event_types
    assert "silence" in event_types
    assert "turn_end" in event_types

    silence_evt = next(e for e in events_silence if e.event_type == "silence")
    assert silence_evt.duration_ms == 800
    assert silence_evt.timestamp_ms == 3000

    utt3_overlap = normalize_generic_transcript(
        text="Wait, let me explain.",
        speaker_id="salesperson",
        start_ms=5200,
        end_ms=6400,
        call_sid="call_turn_test",
    )

    events_overlap = detect_turn_events(utt3_overlap, utt2_silence)
    overlap_types = [e.event_type for e in events_overlap]
    assert "overlap" in overlap_types

    overlap_evt = next(e for e in events_overlap if e.event_type == "overlap")
    assert overlap_evt.duration_ms == 300
    assert overlap_evt.timestamp_ms == 5200


def test_api_behavioral_endpoint():
    client = TestClient(app)

    response = client.get("/api/call/behavioral/non_existent_call")
    assert response.status_code == 200
    data = response.json()
    assert data["call_sid"] == "non_existent_call"
    assert data["count"] == 0
    assert data["utterances"] == []


def test_call_metadata_contract():
    meta = CallMetadata(
        call_id="call_meta_001",
        lead_type="Pre-Foreclosure",
        user_id="sales_rep_42",
        prospect_id="prospect_999",
        stage="Qualification",
        active_playbook="pb_enterprise_standard",
        calibration_profile="calib_direct_tempo",
    )

    dump = meta.model_dump()
    assert dump["call_id"] == "call_meta_001"
    assert dump["lead_type"] == "Pre-Foreclosure"
    assert dump["user_id"] == "sales_rep_42"
    assert dump["prospect_id"] == "prospect_999"
    assert dump["stage"] == "Qualification"
    assert dump["active_playbook"] == "pb_enterprise_standard"
    assert dump["calibration_profile"] == "calib_direct_tempo"


def test_cross_track_stream_synchronization():
    raw_sp_payload = {
        "type": "Results",
        "duration": 1.0,
        "start": 0.2,
        "is_final": True,
        "speech_final": True,
        "channel": {
            "alternatives": [
                {
                    "transcript": "Hello there.",
                    "confidence": 0.98,
                    "words": [
                        {"word": "Hello", "start": 0.2, "end": 0.6, "confidence": 0.98},
                        {"word": "there.", "start": 0.7, "end": 1.1, "confidence": 0.98},
                    ],
                }
            ]
        },
    }

    raw_client_payload = {
        "type": "Results",
        "duration": 0.8,
        "start": 0.1,
        "is_final": True,
        "speech_final": True,
        "channel": {
            "alternatives": [
                {
                    "transcript": "Hi, who is this?",
                    "confidence": 0.95,
                    "words": [
                        {"word": "Hi,", "start": 0.1, "end": 0.4, "confidence": 0.95},
                        {"word": "who", "start": 0.45, "end": 0.6, "confidence": 0.95},
                        {"word": "is", "start": 0.62, "end": 0.75, "confidence": 0.95},
                        {"word": "this?", "start": 0.78, "end": 0.9, "confidence": 0.95},
                    ],
                }
            ]
        },
    }

    sp_offset_ms = 100
    client_offset_ms = 1500

    sp_utt = normalize_deepgram_result(raw_sp_payload, "call_sync", "salesperson", stream_offset_ms=sp_offset_ms)
    client_utt = normalize_deepgram_result(raw_client_payload, "call_sync", "client", stream_offset_ms=client_offset_ms)

    assert sp_utt is not None
    assert client_utt is not None

    assert sp_utt.start_ms == 300
    assert sp_utt.end_ms == 1200

    assert client_utt.start_ms == 1600
    assert client_utt.end_ms == 2400

    events = detect_turn_events(client_utt, sp_utt)
    silence_evt = next(e for e in events if e.event_type == "silence")
    assert silence_evt.duration_ms == 400
    assert silence_evt.timestamp_ms == 1200

