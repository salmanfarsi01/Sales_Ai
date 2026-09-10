from __future__ import annotations

import pytest

from copilot.behavioral_normalization import (
    normalize_generic_transcript,
    CallMetadata,
)
from copilot.behavioral_semantic import (
    SemanticFeatureEngine,
    SemanticFeatureSnapshot,
)


def test_boundary_detection_zero_false_negatives():
    engine = SemanticFeatureEngine()

    boundary_phrases = [
        "Do not call me ever again.",
        "Please take me off your list.",
        "Stop calling this number.",
        "I already have an agent representing me.",
        "We are under contract with another realtor.",
        "Lose my number and stop contacting me.",
        "Do not contact my family, call my attorney.",
    ]

    for phrase in boundary_phrases:
        score = engine.classify_boundary_deterministic(phrase)
        assert score >= 0.8, f"Failed to detect boundary in: '{phrase}'"

    benign_phrases = [
        "What is your commission rate?",
        "We are thinking about selling in the spring.",
        "Can you send me some market comps?",
        "I need to discuss this with my spouse.",
    ]

    for phrase in benign_phrases:
        score = engine.classify_boundary_deterministic(phrase)
        assert score == 0.0, f"False positive boundary on: '{phrase}'"


def test_semantic_recurrence_types():
    engine = SemanticFeatureEngine()

    # 1. Objection repeated after rep response -> concern_after_failed_reframe
    turn1 = normalize_generic_transcript(
        text="Your commission is way too high.",
        speaker_id="client",
        start_ms=1000,
        end_ms=3000,
        call_sid="call_rec_1",
    )
    turn2 = normalize_generic_transcript(
        text="I understand, our focus is on maximum net proceeds.",
        speaker_id="salesperson",
        start_ms=3200,
        end_ms=6000,
        call_sid="call_rec_1",
    )
    turn3 = normalize_generic_transcript(
        text="Still, that commission fee is too high for us.",
        speaker_id="client",
        start_ms=6500,
        end_ms=9000,
        call_sid="call_rec_1",
    )

    rec_id, rec_type, rec_count = engine.detect_semantic_recurrence(turn3.text, [turn1, turn2])
    assert rec_id is not None
    assert rec_type == "concern_after_failed_reframe"
    assert rec_count >= 2

    # 2. Boundary repeated
    bound1 = normalize_generic_transcript(
        text="Do not call this number.",
        speaker_id="client",
        start_ms=10000,
        end_ms=12000,
        call_sid="call_rec_1",
    )
    bound2_text = "I told you, stop calling me!"
    b_id, b_type, b_count = engine.detect_semantic_recurrence(bound2_text, [turn1, turn2, turn3, bound1])
    assert b_type == "boundary_repeated"
    assert b_count == 2

    # 3. Positive echo of rep wording
    rep_turn = normalize_generic_transcript(
        text="We focus on timeline clarity, direct buyers, and net proceeds.",
        speaker_id="salesperson",
        start_ms=13000,
        end_ms=16000,
        call_sid="call_rec_1",
    )
    echo_text = "Yes, exactly, timeline clarity and net proceeds makes sense."
    e_id, e_type, e_count = engine.detect_semantic_recurrence(echo_text, [rep_turn])
    assert e_type == "positive_echo"

    # 4. Scheduling detail repeated
    rep_sched = normalize_generic_transcript(
        text="Could we meet at the property this Tuesday at 2pm?",
        speaker_id="salesperson",
        start_ms=17000,
        end_ms=20000,
        call_sid="call_rec_1",
    )
    sched_repeat_text = "Yes, Tuesday at 2pm works for me."
    s_id, s_type, s_count = engine.detect_semantic_recurrence(sched_repeat_text, [rep_sched])
    assert s_type == "scheduling_detail_repeated"
    assert s_count == 1


@pytest.mark.asyncio
async def test_question_type_and_agreement_distinction():
    engine = SemanticFeatureEngine()

    # Evaluation Question
    q_eval = normalize_generic_transcript(
        text="How do you market homes differently than other agents?",
        speaker_id="client",
        start_ms=1000,
        end_ms=3000,
        call_sid="call_q_1",
    )
    snap_eval = await engine.analyze_turn_semantic(q_eval, [])
    assert snap_eval.question_type == "evaluation"

    # Transactional Question
    q_trans = normalize_generic_transcript(
        text="What is your commission percentage?",
        speaker_id="client",
        start_ms=3500,
        end_ms=5000,
        call_sid="call_q_1",
    )
    snap_trans = await engine.analyze_turn_semantic(q_trans, [q_eval])
    assert snap_trans.question_type == "transactional"

    # Hostile Question
    q_hostile = normalize_generic_transcript(
        text="Why are you calling my private cell phone?",
        speaker_id="client",
        start_ms=5500,
        end_ms=7500,
        call_sid="call_q_1",
    )
    snap_hostile = await engine.analyze_turn_semantic(q_hostile, [q_eval, q_trans])
    assert snap_hostile.question_type == "hostile"

    # Polite nod agreement vs Substantive commitment
    polite_nod = normalize_generic_transcript(
        text="Yeah, okay.",
        speaker_id="client",
        start_ms=8000,
        end_ms=9000,
        call_sid="call_q_1",
    )
    snap_polite = await engine.analyze_turn_semantic(polite_nod, [q_eval, q_trans, q_hostile])
    assert snap_polite.agreement_score <= 0.40

    substantive_commit = normalize_generic_transcript(
        text="Yes, that works, let's meet this Tuesday at 2pm.",
        speaker_id="client",
        start_ms=9500,
        end_ms=12000,
        call_sid="call_q_1",
    )
    snap_substantive = await engine.analyze_turn_semantic(substantive_commit, [polite_nod])
    assert snap_substantive.agreement_score >= 0.80


@pytest.mark.asyncio
async def test_specificity_and_future_language_scoring():
    engine = SemanticFeatureEngine()

    rich_turn = normalize_generic_transcript(
        text="We want to sell our 4 bedroom home in October once probate clears for $750k.",
        speaker_id="client",
        start_ms=1000,
        end_ms=5000,
        call_sid="call_spec_1",
    )

    snap = await engine.analyze_turn_semantic(rich_turn, [])
    assert snap.specificity_score >= 0.70
    assert snap.future_language_score >= 0.50
    assert snap.is_filler_available is False
    assert snap.filler_score is None


@pytest.mark.asyncio
async def test_groq_timeout_fallback_is_immediate_and_non_blocking(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_for_timeout")

    class SlowChatCompletions:
        def create(self, *args, **kwargs):
            import time
            time.sleep(1.5)
            return None

    class SlowGroqClient:
        chat = type("Chat", (), {"completions": SlowChatCompletions()})()

    slow_client = SlowGroqClient()
    engine = SemanticFeatureEngine(groq_client=slow_client, timeout_seconds=0.2)

    turn = normalize_generic_transcript(
        text="What is your listing fee and commission percentage?",
        speaker_id="client",
        start_ms=1000,
        end_ms=2500,
        call_sid="call_timeout_test",
    )

    import time
    start = time.monotonic()
    snapshot = await engine.analyze_turn_semantic(turn, [])
    elapsed = time.monotonic() - start

    assert elapsed < 0.6
    assert isinstance(snapshot, SemanticFeatureSnapshot)
    assert snapshot.extraction_mode == "heuristic_timeout"
    assert snapshot.question_type == "transactional"
    assert snapshot.boundary_confidence == 0.95
    assert snapshot.agreement_confidence == 0.65
    assert snapshot.question_type_confidence == 0.70
    assert snapshot.semantic_confidence == 0.80


@pytest.mark.asyncio
async def test_per_call_semantic_timeout_via_call_metadata(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_for_timeout")

    class SlowChatCompletions:
        def create(self, *args, **kwargs):
            import time
            time.sleep(1.0)
            return None

    class SlowGroqClient:
        chat = type("Chat", (), {"completions": SlowChatCompletions()})()

    engine = SemanticFeatureEngine(groq_client=SlowGroqClient(), timeout_seconds=2.0)

    # CallMetadata overrides engine's 2.0s timeout with 0.15s
    meta = CallMetadata(call_id="call_meta_override", semantic_timeout_sec=0.15)

    turn = normalize_generic_transcript(
        text="What are your fees?",
        speaker_id="client",
        start_ms=1000,
        end_ms=2000,
        call_sid="call_meta_override",
    )

    import time
    start = time.monotonic()
    snapshot = await engine.analyze_turn_semantic(turn, [], call_metadata=meta)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    assert snapshot.extraction_mode == "heuristic_timeout"
    assert snapshot.boundary_confidence == 0.95
    assert snapshot.agreement_confidence == 0.65
    assert snapshot.question_type_confidence == 0.70
