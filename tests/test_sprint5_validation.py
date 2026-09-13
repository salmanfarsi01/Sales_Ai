import pytest
from pathlib import Path
from typing import List
from copilot.behavioral_normalization import (
    NormalizedUtterance,
    NormalizedWord,
    normalize_generic_transcript,
)
from copilot.behavioral_semantic import (
    SemanticFeatureEngine,
    STOP_CONTACT_PATTERNS,
    SPECIFICITY_PATTERNS,
)
from copilot.behavioral_timing import DeterministicTimingEngine
from copilot.behavioral_baseline import BaselineAndChangePointEngine
from copilot.behavioral_evidence import MultiWindowAggregator, SQLiteEvidenceLogStore
from copilot.behavioral_inference import DownstreamInferenceEngine
from copilot.behavioral_pipeline_service import execute_behavioral_turn_pipeline


def make_test_utterance(
    utterance_id: str,
    speaker_id: str,
    text: str,
    start_ms: int,
    end_ms: int,
    call_sid: str = "call_sprint5",
) -> NormalizedUtterance:
    words = text.split()
    step = (end_ms - start_ms) // max(1, len(words))
    norm_words = [
        NormalizedWord(
            word=w,
            start_ms=start_ms + i * step,
            end_ms=start_ms + (i + 1) * step if i < len(words) - 1 else end_ms,
            confidence=0.98,
            is_estimated=False,
        )
        for i, w in enumerate(words)
    ]
    return NormalizedUtterance(
        utterance_id=utterance_id,
        call_sid=call_sid,
        speaker_id=speaker_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        words=norm_words,
        asr_confidence=0.98,
        source_channel=0 if speaker_id == "salesperson" else 1,
        metadata={"is_dual_track": True, "speaker_separation_confidence": 1.0},
    )


def test_sprint5_recurrence_on_separated_speaker_turns():
    """
    Client Point #6 / #4 — Recurrence:
    Re-run a two-speaker script with the same objection worded two different ways;
    confirm same_objection_repeated fires correctly on genuinely separated speaker turns.
    """
    engine = SemanticFeatureEngine()

    # Scenario 1: Genuinely separated prospect turns (consecutive turns from prospect without intervening rep)
    t1 = normalize_generic_transcript(
        text="Honestly, a five percent commission feels pretty steep for what's involved here.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4500,
        call_sid="call_sprint5_rec",
    )
    t2_text = "I hear you, but that commission rate still just does not sit right with me."

    rec_id, rec_type, rec_count = engine.detect_semantic_recurrence(t2_text, [t1])

    assert rec_type == "same_objection_repeated", f"Expected same_objection_repeated, got {rec_type}"
    assert rec_count == 2
    assert rec_id is not None

    # Scenario 2: Two-speaker separated conversation with intervening agent turn
    turn_agent1 = normalize_generic_transcript(
        text="Hello, thank you for taking the time to speak with me today.",
        speaker_id="salesperson",
        start_ms=0,
        end_ms=3000,
        call_sid="call_sprint5_dialogue",
    )
    turn_prospect1 = normalize_generic_transcript(
        text="I really don't want to pay high fees for listing this property.",
        speaker_id="client",
        start_ms=3500,
        end_ms=7000,
        call_sid="call_sprint5_dialogue",
    )
    turn_agent2 = normalize_generic_transcript(
        text="We include full staging and professional media to maximize your net proceeds.",
        speaker_id="salesperson",
        start_ms=7500,
        end_ms=11500,
        call_sid="call_sprint5_dialogue",
    )
    turn_prospect2_text = "Still, your commission fee is just way too expensive for our budget."

    rec_id2, rec_type2, rec_count2 = engine.detect_semantic_recurrence(
        turn_prospect2_text,
        [turn_agent1, turn_prospect1, turn_agent2],
    )
    # When intervening agent speaks, it fires concern_after_failed_reframe (the conversational re-objection subtype)
    assert rec_type2 in ("same_objection_repeated", "concern_after_failed_reframe")
    assert rec_count2 == 2
    assert rec_id2 is not None


@pytest.mark.asyncio
async def test_sprint5_boundary_exact_client_phrase(tmp_path: Path):
    """
    Client Point #7 / #5 — Boundary:
    Run the client's exact suggested phrase: "Please take me off your list and don't call me again"
    Verify boundary score = 1.0, readiness overrides to 0.0, trust penalized.
    """
    engine = SemanticFeatureEngine()
    client_phrase = "Please take me off your list and don't call me again"

    # 1. Test deterministic regex classification directly
    boundary_score = engine.classify_boundary_deterministic(client_phrase)
    assert boundary_score == 1.0, f"Expected boundary score 1.0, got {boundary_score}"

    # 2. Test full pipeline processing through execute_behavioral_turn_pipeline
    db_path = tmp_path / "test_boundary.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_boundary_sprint5")
    aggregator = MultiWindowAggregator("call_boundary_sprint5", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    u1 = make_test_utterance(
        "u1",
        "salesperson",
        "Hello, I am following up on your property inquiry from earlier this week.",
        0,
        3000,
        "call_boundary_sprint5",
    )
    u2 = make_test_utterance(
        "u2",
        "client",
        client_phrase,
        3500,
        7000,
        "call_boundary_sprint5",
    )

    result = await execute_behavioral_turn_pipeline(
        segmented_utts=[u1, u2],
        call_sid="call_boundary_sprint5",
        prospect_id="client",
        agent_id="salesperson",
        transcript="Hello, I am following up on your property inquiry. " + client_phrase,
        timing_engine=timing_engine,
        semantic_engine=engine,
        baseline_engine=baseline_engine,
        aggregator=aggregator,
        inference_engine=inference_engine,
    )

    play_turns = result["play_by_play_turns"]
    assert len(play_turns) == 2

    prospect_turn = play_turns[1]
    assert prospect_turn["boundary_score"] == 1.0
    assert prospect_turn["readiness_score"] == 0.0, (
        f"Readiness should immediately override to 0.0 on hard boundary, got {prospect_turn['readiness_score']}"
    )
    # Trust score should reflect boundary penalty
    assert prospect_turn["trust_score"] <= 0.30
    assert prospect_turn["tension_level"] >= 0.40

    store.close()


@pytest.mark.asyncio
async def test_sprint5_specificity_score_visible_jump(tmp_path: Path):
    """
    Client Point #8 — Specificity:
    Script one turn as vague ("maybe sometime"), a later turn as specific ("Thursday at 4pm"),
    confirm the score visibly jumps.
    """
    engine = SemanticFeatureEngine()

    vague_text = "maybe sometime"
    specific_text = "Thursday at 4pm"

    # Evaluate vague turn directly
    snap_vague = engine.analyze_deterministic_heuristic(
        normalize_generic_transcript(vague_text, "client", 1000, 2500, "call_spec"),
        [],
    )
    # Evaluate specific turn directly
    snap_specific = engine.analyze_deterministic_heuristic(
        normalize_generic_transcript(specific_text, "client", 3500, 5000, "call_spec"),
        [],
    )

    score_vague = snap_vague.specificity_score
    score_specific = snap_specific.specificity_score

    assert score_vague == 0.0, f"Vague phrase '{vague_text}' should have specificity 0.0, got {score_vague}"
    assert score_specific >= 0.70, f"Specific phrase '{specific_text}' should have specificity >= 0.70, got {score_specific}"
    jump = score_specific - score_vague
    assert jump >= 0.70, f"Expected specificity jump of at least 0.70, observed jump was {jump:.2f}"

    # Also test through full execute_behavioral_turn_pipeline
    db_path = tmp_path / "test_spec.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_spec_test")
    aggregator = MultiWindowAggregator("call_spec_test", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    u1 = make_test_utterance("u1", "client", vague_text, 1000, 2500, "call_spec_test")
    u2 = make_test_utterance("u2", "client", specific_text, 3500, 5000, "call_spec_test")

    result = await execute_behavioral_turn_pipeline(
        segmented_utts=[u1, u2],
        call_sid="call_spec_test",
        prospect_id="client",
        agent_id="salesperson",
        transcript=f"{vague_text} {specific_text}",
        timing_engine=timing_engine,
        semantic_engine=engine,
        baseline_engine=baseline_engine,
        aggregator=aggregator,
        inference_engine=inference_engine,
    )

    turns = result["play_by_play_turns"]
    assert len(turns) == 2
    assert turns[0]["specificity_score"] == 0.0
    assert turns[1]["specificity_score"] >= 0.70

    store.close()


@pytest.mark.asyncio
async def test_sprint5_per_utterance_latency_exposed(tmp_path: Path):
    """
    Client Point #10 / #8 — Per-utterance latency:
    Expose the existing per-turn inference_latency_ms (under 10ms per Phase 8)
    as its own visible column in the diagnostics table, instead of reporting
    the whole batch's 15-20 second transcription time.
    """
    db_path = tmp_path / "test_lat.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_latency_sprint5")
    aggregator = MultiWindowAggregator("call_latency_sprint5", timing_engine, baseline_engine, store)
    inference_engine = DownstreamInferenceEngine()

    u1 = make_test_utterance(
        "u1",
        "salesperson",
        "Good afternoon, this is Alex calling from Beacon Realty.",
        0,
        2500,
        "call_latency_sprint5",
    )
    u2 = make_test_utterance(
        "u2",
        "client",
        "Hi Alex, we are thinking about putting our home on the market.",
        3200,
        5500,
        "call_latency_sprint5",
    )

    result = await execute_behavioral_turn_pipeline(
        segmented_utts=[u1, u2],
        call_sid="call_latency_sprint5",
        prospect_id="client",
        agent_id="salesperson",
        transcript="Good afternoon, this is Alex calling. Hi Alex, we are thinking about putting our home on the market.",
        timing_engine=timing_engine,
        semantic_engine=SemanticFeatureEngine(),
        baseline_engine=baseline_engine,
        aggregator=aggregator,
        inference_engine=inference_engine,
    )

    play_turns = result["play_by_play_turns"]
    assert len(play_turns) == 2

    for t in play_turns:
        assert "inference_latency_ms" in t, f"Missing inference_latency_ms in turn #{t['turn_index']}"
        inf_lat = t["inference_latency_ms"]
        assert isinstance(inf_lat, (int, float))
        # Must be sub-15ms (typical is 0.5 - 3ms)
        assert 0.0 <= inf_lat < 15.0, f"Inference latency {inf_lat}ms exceeds 15ms threshold"

    store.close()
