import pytest
from typing import List

from copilot.behavioral_normalization import (
    NormalizedUtterance,
    NormalizedWord,
    merge_dual_speaker_tracks,
    segment_audio_transcript_turns,
)
from copilot.behavioral_timing import DeterministicTimingEngine
from copilot.behavioral_baseline import (
    BaselineAndChangePointEngine,
    ProspectBaselineStore,
)


def _make_words(text: str, start_ms: int, end_ms: int) -> List[NormalizedWord]:
    words = text.split()
    if not words:
        return []
    step = (end_ms - start_ms) // len(words)
    res = []
    for i, w in enumerate(words):
        w_start = start_ms + i * step
        w_end = start_ms + (i + 1) * step if i < len(words) - 1 else end_ms
        res.append(NormalizedWord(
            word=w,
            start_ms=w_start,
            end_ms=max(w_start + 50, w_end),
            confidence=0.95,
            punctuated_word=w,
        ))
    return res


def test_merge_dual_speaker_tracks_chronological_ordering():
    """Validates that independent tracks with start epochs align and interleave chronologically."""
    # Agent started at epoch 1000ms
    # Prospect started at epoch 2000ms (1000ms later)
    agent_utts = [
        NormalizedUtterance(
            call_sid="call_sync_01",
            speaker_id="salesperson",
            text="Hello thank you for taking my call today",
            start_ms=0,
            end_ms=2000,
            words=_make_words("Hello thank you for taking my call today", 0, 2000),
        ),
        NormalizedUtterance(
            call_sid="call_sync_01",
            speaker_id="salesperson",
            text="I wanted to discuss our enterprise package",
            start_ms=4000,
            end_ms=6500,
            words=_make_words("I wanted to discuss our enterprise package", 4000, 6500),
        ),
    ]

    prospect_utts = [
        NormalizedUtterance(
            call_sid="call_sync_01",
            speaker_id="client",
            text="Hi sure I have about ten minutes right now",
            start_ms=1500,  # relative to prospect start (2000), this is 3500ms on combined timeline
            end_ms=3500,   # 5500ms on combined timeline
            words=_make_words("Hi sure I have about ten minutes right now", 1500, 3500),
        ),
    ]

    merged = merge_dual_speaker_tracks(
        agent_utterances=agent_utts,
        prospect_utterances=prospect_utts,
        agent_start_epoch_ms=1000,
        prospect_start_epoch_ms=2000,
    )

    assert len(merged) == 3

    # Turn 1: Agent (0 + 0 = 0 to 2000ms)
    assert merged[0].speaker_id == "salesperson"
    assert merged[0].start_ms == 0
    assert merged[0].end_ms == 2000
    assert merged[0].words[0].start_ms == 0

    # Turn 2: Prospect (1500 + 1000 = 2500 to 4500ms)
    assert merged[1].speaker_id == "client"
    assert merged[1].start_ms == 2500
    assert merged[1].end_ms == 4500
    assert merged[1].words[0].start_ms == 2500

    # Turn 3: Agent (4000 + 0 = 4000 to 6500ms)
    assert merged[2].speaker_id == "salesperson"
    assert merged[2].start_ms == 4000
    assert merged[2].end_ms == 6500


def test_cross_speaker_response_latency_calculation():
    """Validates that DeterministicTimingEngine computes response latency strictly across speaker transitions."""
    timing = DeterministicTimingEngine()

    # Turn 1: Agent speaks 0 to 2000ms
    u1 = NormalizedUtterance(
        call_sid="call_lat_01",
        speaker_id="salesperson",
        text="Can you hear me alright today?",
        start_ms=0,
        end_ms=2000,
        words=_make_words("Can you hear me alright today?", 0, 2000),
    )
    s1 = timing.process_utterance(u1)
    # First turn in call has no prior speaker -> latency is None
    assert s1.response_latency_ms is None

    # Turn 2: Prospect answers starting at 2800ms (latency = 800ms)
    u2 = NormalizedUtterance(
        call_sid="call_lat_01",
        speaker_id="client",
        text="Yes I hear you loud and clear.",
        start_ms=2800,
        end_ms=4500,
        words=_make_words("Yes I hear you loud and clear.", 2800, 4500),
    )
    s2 = timing.process_utterance(u2)
    assert s2.response_latency_ms == 800

    # Turn 3: Prospect speaks again without an Agent turn (same speaker continuation)
    u3 = NormalizedUtterance(
        call_sid="call_lat_01",
        speaker_id="client",
        text="What did you want to talk about?",
        start_ms=5000,
        end_ms=7000,
        words=_make_words("What did you want to talk about?", 5000, 7000),
    )
    s3 = timing.process_utterance(u3)
    # Since previous speaker was also "client", cross-speaker response latency is None
    # or measured from last agent turn. Timing engine measures across speaker switches.
    assert s3.response_latency_ms == 3000  # 5000 - 2000 (last salesperson utterance end)

    # Turn 4: Agent responds at 7400ms (latency from Prospect's end at 7000ms is 400ms)
    u4 = NormalizedUtterance(
        call_sid="call_lat_01",
        speaker_id="salesperson",
        text="I am reaching out regarding the commercial proposal.",
        start_ms=7400,
        end_ms=10500,
        words=_make_words("I am reaching out regarding the commercial proposal.", 7400, 10500),
    )
    s4 = timing.process_utterance(u4)
    assert s4.response_latency_ms == 400


def test_instantaneous_turn_wpm_differs_on_consecutive_turns():
    """Validates that consecutive turns with different speaking rates show distinct turn_speech_rate_wpm."""
    timing = DeterministicTimingEngine()

    # Turn 1: 10 words in 2 seconds = 300 WPM
    words_fast = "one two three four five six seven eight nine ten"
    u1 = NormalizedUtterance(
        call_sid="call_wpm_01",
        speaker_id="client",
        text=words_fast,
        start_ms=0,
        end_ms=2000,
        words=_make_words(words_fast, 0, 2000),
    )
    s1 = timing.process_utterance(u1)
    assert s1.turn_speech_rate_wpm == 300.0

    # Turn 2: 6 words in 4 seconds = 90 WPM
    words_slow = "one two three four five six"
    u2 = NormalizedUtterance(
        call_sid="call_wpm_01",
        speaker_id="client",
        text=words_slow,
        start_ms=3000,
        end_ms=7000,
        words=_make_words(words_slow, 3000, 7000),
    )
    s2 = timing.process_utterance(u2)
    assert s2.turn_speech_rate_wpm == 90.0

    # Verify that the two consecutive turns show radically different instantaneous rates
    assert s1.turn_speech_rate_wpm != s2.turn_speech_rate_wpm
    assert s1.turn_speech_rate_wpm == 300.0
    assert s2.turn_speech_rate_wpm == 90.0


def test_isolated_per_speaker_baselines():
    """Validates that Agent and Prospect baselines are completely separate and never mix."""
    store = ProspectBaselineStore()
    engine = BaselineAndChangePointEngine(
        call_sid="call_iso_01",
        prospect_id="prospect_99",
        agent_id="agent_01",
        store=store,
        intra_call_window_ms=60000,
    )
    timing = DeterministicTimingEngine()

    # Interleaved conversation: Agent speaks, then Prospect responds
    for i in range(4):
        # Agent speaks fast turn (20 words in 6s = 200 WPM)
        agent_start = i * 20000
        agent_end = agent_start + 6000
        text_agent = "word " * 20
        u_agent = NormalizedUtterance(
            call_sid="call_iso_01",
            speaker_id="salesperson",
            text=text_agent.strip(),
            start_ms=agent_start,
            end_ms=agent_end,
            words=_make_words(text_agent.strip(), agent_start, agent_end),
        )
        snap_agent = timing.process_utterance(u_agent)
        engine.update_with_utterance(u_agent, snap_agent)

        # Prospect responds slow turn (10 words in 6s = 100 WPM) after 800ms latency
        prospect_start = agent_end + 800
        prospect_end = prospect_start + 6000
        text_prospect = "word " * 10
        u_prospect = NormalizedUtterance(
            call_sid="call_iso_01",
            speaker_id="client",
            text=text_prospect.strip(),
            start_ms=prospect_start,
            end_ms=prospect_end,
            words=_make_words(text_prospect.strip(), prospect_start, prospect_end),
        )
        snap_prospect = timing.process_utterance(u_prospect)
        engine.update_with_utterance(u_prospect, snap_prospect)

    # Force lock evaluation
    engine.finalize_and_persist("salesperson")
    engine.finalize_and_persist("client")

    agent_prof = engine.get_active_profile("salesperson")
    prospect_prof = engine.get_active_profile("client")

    assert agent_prof is not None
    assert prospect_prof is not None

    # Agent mean turn length words should be 20
    assert agent_prof.turn_length_words.mean == pytest.approx(20.0, rel=0.05)
    # Prospect mean turn length words should be 10
    assert prospect_prof.turn_length_words.mean == pytest.approx(10.0, rel=0.05)

    # They must remain strictly different and isolated
    assert agent_prof.turn_length_words.mean > prospect_prof.turn_length_words.mean * 1.5


def test_merge_dual_speaker_tracks_relative_offset_positive_and_negative():
    """Validates that relative offsets (+2500ms and -1500ms) correctly shift the timeline."""
    agent_utts = [
        NormalizedUtterance(
            call_sid="call_rel_01",
            speaker_id="salesperson",
            text="Agent speaking first",
            start_ms=0,
            end_ms=2000,
            words=_make_words("Agent speaking first", 0, 2000),
        )
    ]
    prospect_utts = [
        NormalizedUtterance(
            call_sid="call_rel_01",
            speaker_id="client",
            text="Prospect speaking next",
            start_ms=500,
            end_ms=2500,
            words=_make_words("Prospect speaking next", 500, 2500),
        )
    ]

    # 1. Positive relative offset: Prospect started 2500ms after Agent
    merged_pos = merge_dual_speaker_tracks(
        agent_utterances=agent_utts,
        prospect_utterances=prospect_utts,
        prospect_offset_ms=2500,
    )
    assert merged_pos[0].speaker_id == "salesperson"
    assert merged_pos[0].start_ms == 0
    assert merged_pos[0].end_ms == 2000

    assert merged_pos[1].speaker_id == "client"
    assert merged_pos[1].start_ms == 3000  # 500 + 2500
    assert merged_pos[1].end_ms == 5000    # 2500 + 2500

    # 2. Negative relative offset: Prospect started 1500ms before Agent
    merged_neg = merge_dual_speaker_tracks(
        agent_utterances=agent_utts,
        prospect_utterances=prospect_utts,
        prospect_offset_ms=-1500,
    )
    # Prospect offset is 0, Agent offset is 1500
    # Prospect turn: 500 to 2500ms
    # Agent turn: 0 + 1500 = 1500 to 3500ms
    assert merged_neg[0].speaker_id == "client"
    assert merged_neg[0].start_ms == 500
    assert merged_neg[0].end_ms == 2500

    assert merged_neg[1].speaker_id == "salesperson"
    assert merged_neg[1].start_ms == 1500
    assert merged_neg[1].end_ms == 3500


def test_merge_dual_speaker_tracks_mismatched_file_lengths():
    """Validates that different track durations (e.g. Agent 3m vs Prospect 2m45s) merge cleanly."""
    # Agent track runs to 180,000ms (3 min)
    agent_utts = [
        NormalizedUtterance(
            call_sid="call_len_01",
            speaker_id="salesperson",
            text="Agent early turn",
            start_ms=10000,
            end_ms=20000,
            words=_make_words("Agent early turn", 10000, 20000),
        ),
        NormalizedUtterance(
            call_sid="call_len_01",
            speaker_id="salesperson",
            text="Agent late closing turn at minute three",
            start_ms=168000,
            end_ms=178000,
            words=_make_words("Agent late closing turn at minute three", 168000, 178000),
        ),
    ]

    # Prospect track stops at 165,000ms (2 min 45s)
    prospect_utts = [
        NormalizedUtterance(
            call_sid="call_len_01",
            speaker_id="client",
            text="Prospect final goodbye at 2 minutes 45 seconds",
            start_ms=155000,
            end_ms=165000,
            words=_make_words("Prospect final goodbye at 2 minutes 45 seconds", 155000, 165000),
        )
    ]

    merged = merge_dual_speaker_tracks(
        agent_utterances=agent_utts,
        prospect_utterances=prospect_utts,
        prospect_offset_ms=0,
    )

    assert len(merged) == 3
    assert merged[0].speaker_id == "salesperson"
    assert merged[0].start_ms == 10000

    assert merged[1].speaker_id == "client"
    assert merged[1].start_ms == 155000

    assert merged[2].speaker_id == "salesperson"
    assert merged[2].start_ms == 168000
    assert merged[2].end_ms == 178000

    # Ensure timing engine handles the sequence without error
    timing = DeterministicTimingEngine()
    snaps = [timing.process_utterance(u) for u in merged]
    assert len(snaps) == 3
    # Turn 3 latency is measured from Prospect's end at 165,000ms: 168,000 - 165,000 = 3000ms
    assert snaps[2].response_latency_ms == 3000


def test_segment_audio_transcript_turns_enforce_single_speaker():
    """Validates that enforce_single_speaker=True prevents Deepgram diarization tags from hijacking speaker identity."""
    raw_deepgram_utterances = [
        {
            "transcript": "I need to check our budget with procurement.",
            "speaker": 0,  # Diarizer guessed 0 (which normally maps to salesperson)
            "words": [
                {"word": "I", "start": 0.0, "end": 0.2},
                {"word": "need", "start": 0.3, "end": 0.5},
                {"word": "to", "start": 0.6, "end": 0.7},
                {"word": "check", "start": 0.8, "end": 1.1},
                {"word": "our", "start": 1.2, "end": 1.3},
                {"word": "budget", "start": 1.4, "end": 1.8},
                {"word": "with", "start": 1.9, "end": 2.1},
                {"word": "procurement.", "start": 2.2, "end": 2.8},
            ],
            "confidence": 0.95,
        }
    ]

    # With enforce_single_speaker=True, speaker_id="client" must remain "client"
    utts = segment_audio_transcript_turns(
        words_raw=raw_deepgram_utterances[0]["words"],
        transcript="I need to check our budget with procurement.",
        call_sid="call_spk_01",
        speaker_id="client",
        raw_utterances=raw_deepgram_utterances,
        enforce_single_speaker=True,
    )

    assert len(utts) == 1
    assert utts[0].speaker_id == "client"


def test_fastapi_app_analyze_two_tracks_and_upload_track(monkeypatch):
    """Validates /api/test/analyze-two-tracks and /api/test/upload-track endpoints on FastAPI app."""
    from fastapi.testclient import TestClient
    from copilot.fastapi_app import app
    from copilot.calibration import SpeechToTextEngine

    async def mock_transcribe(self, audio_bytes, mime_type="audio/webm", utt_split=0.5, diarize=True):
        if b"AGENT" in audio_bytes:
            return (
                "Hello thanks for taking my call today regarding enterprise pricing.",
                [
                    {"word": "Hello", "start": 0.0, "end": 0.4},
                    {"word": "thanks", "start": 0.5, "end": 0.9},
                    {"word": "for", "start": 1.0, "end": 1.2},
                    {"word": "taking", "start": 1.3, "end": 1.6},
                    {"word": "my", "start": 1.7, "end": 1.8},
                    {"word": "call", "start": 1.9, "end": 2.2},
                    {"word": "today", "start": 2.3, "end": 2.6},
                    {"word": "regarding", "start": 2.7, "end": 3.1},
                    {"word": "enterprise", "start": 3.2, "end": 3.7},
                    {"word": "pricing.", "start": 3.8, "end": 4.3},
                ],
            )
        else:
            return (
                "Hi sure I have about five minutes to discuss the contract terms.",
                [
                    {"word": "Hi", "start": 0.0, "end": 0.3},
                    {"word": "sure", "start": 0.4, "end": 0.7},
                    {"word": "I", "start": 0.8, "end": 0.9},
                    {"word": "have", "start": 1.0, "end": 1.2},
                    {"word": "about", "start": 1.3, "end": 1.6},
                    {"word": "five", "start": 1.7, "end": 2.0},
                    {"word": "minutes", "start": 2.1, "end": 2.5},
                    {"word": "to", "start": 2.6, "end": 2.8},
                    {"word": "discuss", "start": 2.9, "end": 3.3},
                    {"word": "the", "start": 3.4, "end": 3.6},
                    {"word": "contract", "start": 3.7, "end": 4.1},
                    {"word": "terms.", "start": 4.2, "end": 4.7},
                ],
            )

    monkeypatch.setattr(SpeechToTextEngine, "transcribe_with_timestamps", mock_transcribe)

    client = TestClient(app)

    # 1. Test POST /api/test/analyze-two-tracks
    files = {
        "agent_audio": ("agent.wav", b"RIFF_AGENT_AUDIO_TRACK", "audio/wav"),
        "prospect_audio": ("prospect.wav", b"RIFF_PROSPECT_AUDIO_TRACK", "audio/wav"),
    }
    data = {
        "agent_id": "test_agent_100",
        "prospect_id": "test_prospect_200",
        "prospect_offset_ms": 2000,
    }
    res = client.post("/api/test/analyze-two-tracks", files=files, data=data)
    assert res.status_code == 200
    res_json = res.json()

    assert res_json["status"] == "success"
    assert res_json["turns_processed"] == 2
    assert res_json["agent_id"] == "test_agent_100"
    assert res_json["prospect_id"] == "test_prospect_200"

    turns = res_json["play_by_play_turns"]
    assert turns[0]["speaker_label"] == "Agent"
    assert turns[0]["start_ms"] == 0
    assert turns[1]["speaker_label"] == "Prospect"
    assert turns[1]["start_ms"] == 2000  # Offset by 2000ms!

    # 2. Test POST /api/test/upload-track (individual track ingestion)
    track_call_sid = "test_upload_split_call"
    # Upload agent track first
    agent_file = {"audio": ("agent.wav", b"RIFF_AGENT_AUDIO_TRACK", "audio/wav")}
    agent_data = {"call_sid": track_call_sid, "speaker_role": "salesperson", "start_epoch_ms": 0}
    res_track1 = client.post("/api/test/upload-track", files=agent_file, data=agent_data)
    assert res_track1.status_code == 200
    assert res_track1.json()["status"] == "waiting_for_partner_track"

    # Upload prospect track second
    prospect_file = {"audio": ("prospect.wav", b"RIFF_PROSPECT_AUDIO_TRACK", "audio/wav")}
    prospect_data = {"call_sid": track_call_sid, "speaker_role": "client", "start_epoch_ms": 2500}
    res_track2 = client.post("/api/test/upload-track", files=prospect_file, data=prospect_data)
    assert res_track2.status_code == 200
    assert res_track2.json()["status"] == "merged_success"
    assert res_track2.json()["turns_processed"] == 2


def test_run_behavioral_signal_analyze_two_tracks(monkeypatch):
    """Validates /api/test/analyze-two-tracks on run_behavioral_signal.app using shared pipeline."""
    from fastapi.testclient import TestClient
    from run_behavioral_signal import app as runner_app
    from copilot.calibration import SpeechToTextEngine

    async def mock_transcribe(self, audio_bytes, mime_type="audio/webm", utt_split=0.5, diarize=True):
        if b"AGENT" in audio_bytes:
            return (
                "Agent track statement regarding enterprise roadmap.",
                [
                    {"word": "Agent", "start": 0.0, "end": 0.5},
                    {"word": "track", "start": 0.6, "end": 1.0},
                    {"word": "statement", "start": 1.1, "end": 1.7},
                    {"word": "regarding", "start": 1.8, "end": 2.2},
                    {"word": "enterprise", "start": 2.3, "end": 2.8},
                    {"word": "roadmap.", "start": 2.9, "end": 3.4},
                ],
            )
        else:
            return (
                "Prospect track confirmation.",
                [
                    {"word": "Prospect", "start": 0.0, "end": 0.4},
                    {"word": "track", "start": 0.5, "end": 0.9},
                    {"word": "confirmation.", "start": 1.0, "end": 1.6},
                ],
            )

    monkeypatch.setattr(SpeechToTextEngine, "transcribe_with_timestamps", mock_transcribe)

    client = TestClient(runner_app)
    files = {
        "agent_audio": ("agent.wav", b"RIFF_AGENT_AUDIO_TRACK", "audio/wav"),
        "prospect_audio": ("prospect.wav", b"RIFF_PROSPECT_AUDIO_TRACK", "audio/wav"),
    }
    data = {
        "agent_id": "test_agent_runner",
        "prospect_id": "test_prospect_runner",
        "prospect_offset_ms": 1500,
    }
    res = client.post("/api/test/analyze-two-tracks", files=files, data=data)
    assert res.status_code == 200
    res_json = res.json()

    assert res_json["status"] == "success"
    assert res_json["turns_processed"] == 2
    assert "agent_baseline_wpm" in res_json
    assert "prospect_baseline_wpm" in res_json
    turns = res_json["play_by_play_turns"]
    assert turns[0]["speaker_label"] == "Agent"
    assert turns[1]["speaker_label"] == "Prospect"
    assert turns[1]["start_ms"] == 1500


