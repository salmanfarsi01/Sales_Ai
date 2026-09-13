import pytest
from pathlib import Path
from copilot.behavioral_normalization import NormalizedUtterance, NormalizedWord
from copilot.behavioral_timing import DeterministicTimingEngine
from copilot.behavioral_evidence import MultiWindowAggregator, SQLiteEvidenceLogStore
from copilot.behavioral_baseline import BaselineAndChangePointEngine
from copilot.behavioral_inference import DownstreamInferenceEngine
from copilot.behavioral_pipeline_service import execute_behavioral_turn_pipeline


def make_test_utterance(
    utterance_id: str,
    speaker_id: str,
    text: str,
    start_ms: int,
    end_ms: int,
    asr_confidence: float = 0.98,
    is_dual_track: bool = True,
) -> NormalizedUtterance:
    words = text.split()
    step = (end_ms - start_ms) // max(1, len(words))
    norm_words = [
        NormalizedWord(
            word=w,
            start_ms=start_ms + i * step,
            end_ms=start_ms + (i + 1) * step if i < len(words) - 1 else end_ms,
            confidence=asr_confidence,
            is_estimated=False,
        )
        for i, w in enumerate(words)
    ]
    return NormalizedUtterance(
        utterance_id=utterance_id,
        call_sid="call_sprint3_test",
        speaker_id=speaker_id,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        asr_confidence=asr_confidence,
        is_final=True,
        speech_final=True,
        is_estimated_timing=False,
        words=norm_words,
        metadata={"is_dual_track": is_dual_track, "speaker_separation_confidence": 1.0 if is_dual_track else 0.80},
    )


def test_cross_speaker_latency_computation_with_3s_natural_pause():
    """Sprint 3 DoD: Verify cross-speaker turnaround time lookup.
    Specifically tests that a natural pause (the scripted ~3-second silence)
    produces a correctly measured latency of 3000ms in that exact row.
    """
    timing_engine = DeterministicTimingEngine()

    # Turn 1: Agent speaks 0s - 2.0s (0ms - 2000ms)
    u1 = make_test_utterance("u1", "salesperson", "Hello this is Alex calling from Acme Realty.", 0, 2000)
    snap1 = timing_engine.process_utterance(u1)
    # First turn: no prior other-speaker utterance
    assert snap1.response_latency_ms is None

    # Turn 2: Prospect responds 2.8s - 4.5s (2800ms - 4500ms)
    # Gap from Agent end (2000ms) to Prospect start (2800ms) = 800ms
    u2 = make_test_utterance("u2", "client", "Hi Alex, thanks for following up with me today.", 2800, 4500)
    snap2 = timing_engine.process_utterance(u2)
    assert snap2.response_latency_ms == 800

    # Turn 3: SCRIPTED NATURAL PAUSE (~3-second silence)
    # Prospect ended at 4500ms. Agent pauses naturally, then starts at 7500ms.
    # Latency: 7500 - 4500 = 3000ms!
    u3 = make_test_utterance("u3", "salesperson", "I wanted to check on your timeline for listing.", 7500, 10000)
    snap3 = timing_engine.process_utterance(u3)
    assert snap3.response_latency_ms == 3000

    # Turn 4: Cross-speaker interruption / overlap
    # Agent finished at 10000ms. Prospect interrupts at 9700ms (starts before Agent finished)
    # Latency: 9700 - 10000 = -300ms (negative latency = overlap)
    u4 = make_test_utterance("u4", "client", "Wait, before that, what about commission?", 9700, 12000)
    snap4 = timing_engine.process_utterance(u4)
    assert snap4.response_latency_ms == -300

    # Turn 5: Client continues another turn
    # Other speaker (Agent) last ended at 10000ms. Client starts at 12500ms.
    # Latency to other speaker: 12500 - 10000 = 2500ms
    u5 = make_test_utterance("u5", "client", "Because another agent offered four percent.", 12500, 15000)
    snap5 = timing_engine.process_utterance(u5)
    assert snap5.response_latency_ms == 2500


@pytest.mark.asyncio
async def test_pipeline_service_surfaces_latency_in_play_by_play(tmp_path: Path):
    """Verify that process_turns populates response_latency_ms in play_by_play_turns."""
    db_path = tmp_path / "test_latency.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_lat_pipe")
    aggregator = MultiWindowAggregator("call_lat_pipe", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    u1 = make_test_utterance("u1", "salesperson", "Hello Alex here.", 0, 2000)
    u2 = make_test_utterance("u2", "client", "Hi there how are you?", 2500, 4000)
    # 3-second natural silence: Prospect ended at 4000ms, Agent starts at 7000ms
    u3 = make_test_utterance("u3", "salesperson", "Doing very well thank you.", 7000, 9500)

    result = await execute_behavioral_turn_pipeline(
        segmented_utts=[u1, u2, u3],
        call_sid="call_lat_pipe",
        prospect_id="client",
        agent_id="salesperson",
        transcript="Hello Alex here. Hi there how are you? Doing very well thank you.",
        timing_engine=timing_engine,
        semantic_engine=None,
        baseline_engine=baseline_engine,
        aggregator=aggregator,
        inference_engine=inference_engine,
    )

    turns = result["play_by_play_turns"]
    assert len(turns) == 3
    assert turns[0]["response_latency_ms"] is None
    assert turns[1]["response_latency_ms"] == 500  # 2500 - 2000
    assert turns[2]["response_latency_ms"] == 3000  # 7000 - 4000 (3-second silence)

    store.close()


def test_interruption_mid_sentence_single_and_split_turns():
    """Specifically exercises the negative latency / overlap scenario:
    1. Clean single-turn mid-sentence interruption:
       Speaker A is speaking [10000ms - 15000ms].
       Speaker B starts speaking at 13500ms (1500ms before Speaker A ends).
       Expected latency: 13500 - 15000 = -1500ms (negative value represents 1500ms overlap).

    2. Mid-sentence interruption when Speaker A's speech was split into fragments:
       Turn A1: [20000ms - 22000ms] ("We offer three tiers of service:")
       Turn A2: [22300ms - 26000ms] ("basic, professional, and enterprise.")
       Speaker B interrupts at [24000ms - 25500ms] ("Wait, what about the pricing?")
       The lookup must resolve to the *actively interrupted fragment* Turn A2 (ends 26000ms),
       NOT the earlier finished fragment Turn A1.
       Expected latency: 24000 - 26000 = -2000ms (negative value represents 2000ms overlap).

    3. Multi-fragment interruption where Speaker B's interruption is itself split:
       Turn A is [30000ms - 35000ms].
       Turn B1: [33000ms - 34000ms] ("Wait, hold on...")
       Turn B2: [34200ms - 36000ms] ("...what is the total fee?")
       When Turn B1 starts at 33000ms, Turn A has 2000ms left -> latency = -2000ms.
       When Turn B2 starts at 34200ms, Turn A is STILL speaking with 800ms left -> latency = -800ms.
       Both fragments correctly measure their active overlap against Speaker A.
    """
    timing_engine = DeterministicTimingEngine()

    # Scenario 1: Clean mid-sentence interruption
    u_a1 = make_test_utterance("ua1", "salesperson", "We have been providing this service since 2012.", 10000, 15000)
    timing_engine.process_utterance(u_a1)

    u_b1 = make_test_utterance("ub1", "client", "Wait, is that nationwide or just local?", 13500, 16000)
    snap_b1 = timing_engine.process_utterance(u_b1)
    # Speaker B started at 13500ms, Speaker A ended at 15000ms
    assert snap_b1.response_latency_ms == -1500

    # Scenario 2: Interrupted speaker's turn was split into sequential fragments
    u_a2a = make_test_utterance("ua2a", "salesperson", "We offer three tiers of service:", 20000, 22000)
    timing_engine.process_utterance(u_a2a)

    u_a2b = make_test_utterance("ua2b", "salesperson", "basic, professional, and enterprise packages.", 22300, 26000)
    timing_engine.process_utterance(u_a2b)

    # Client interrupts during the second fragment (ua2b)
    u_b2 = make_test_utterance("ub2", "client", "Wait, what about the pricing for enterprise?", 24000, 25500)
    snap_b2 = timing_engine.process_utterance(u_b2)
    # Client started at 24000ms. Active fragment is ua2b (ends 26000ms), NOT ua2a (ended 22000ms).
    assert snap_b2.response_latency_ms == -2000

    # Scenario 3: Interrupting speaker is itself split into multiple turns while other speaker is speaking
    u_a3 = make_test_utterance("ua3", "salesperson", "Our standard onboarding takes approximately six to eight weeks.", 30000, 35000)
    timing_engine.process_utterance(u_a3)

    # First fragment of client interruption
    u_b3a = make_test_utterance("ub3a", "client", "Wait, hold on...", 33000, 34000)
    snap_b3a = timing_engine.process_utterance(u_b3a)
    assert snap_b3a.response_latency_ms == -2000  # 33000 - 35000

    # Second fragment of client interruption while salesperson is STILL speaking
    u_b3b = make_test_utterance("ub3b", "client", "...did you say eight weeks?", 34200, 36500)
    snap_b3b = timing_engine.process_utterance(u_b3b)
    assert snap_b3b.response_latency_ms == -800  # 34200 - 35000
