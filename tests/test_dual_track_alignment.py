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
