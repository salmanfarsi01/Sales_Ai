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
        "I'd rather you didn't reach out for a while.",
        "Please don't contact me about this again.",
        "We're actually working with someone already.",
        "I prefer not to be contacted anymore.",
        "Please leave me alone.",
        "We already signed with a realtor last week.",
        "Do not reach out to this number.",
        "I'd appreciate it if you don't call this number again.",
        "take my number off your list",
        "Stop reaching out please.",
        "Put me on the do not call list.",
    ]

    for phrase in boundary_phrases:
        score = engine.classify_boundary_deterministic(phrase)
        assert score >= 0.8, f"Failed to detect boundary in: '{phrase}'"

    benign_phrases = [
        "What is your commission rate?",
        "We are thinking about selling in the spring.",
        "Can you send me some market comps?",
        "I need to discuss this with my spouse.",
        "So we have been looking at a few different options for a while now, honestly.",
        "We are not in rush exactly, but our lease is up in four months.",
        "I'm not sure if now is the right time for us.",
        "We're looking for something with three bedrooms and a nice yard.",
        "The budget is around 450k, maybe up to 500k if it's really turnkey.",
        "What kind of commission structure do you typically work with?",
        "Yeah that sounds reasonable, let's talk next Tuesday.",
        "We were burned before by another deal that fell through on financing.",
        "I'm not interested in the townhouse, but do not want to rule out the other listing either.",
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
    # Overall semantic_confidence reflects the weakest link of the heuristic profile (0.65)
    assert snapshot.semantic_confidence == 0.65


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


def test_relative_temporal_specificity_heuristic():
    """Verifies that relative temporal anchors and spelled-out numerals
    ("four months", "two year old", "last month") produce high specificity in heuristic mode.
    """
    engine = SemanticFeatureEngine()
    turn = normalize_generic_transcript(
        text="Well, it was listed for four months with another agent last month, and my two year old kid needs space so we want to sell.",
        speaker_id="client",
        start_ms=0,
        end_ms=4000,
        call_sid="call_rel_spec",
    )

    snap = engine.analyze_deterministic_heuristic(turn, [], mode="heuristic_offline")
    assert snap.specificity_score >= 0.60
    assert snap.extraction_mode == "heuristic_offline"


@pytest.mark.asyncio
async def test_short_fragment_bypass_per_feature_confidence():
    """
    Verifies that a 1-word fragment ("case."):
    1. Bypasses LLM network calls directly via heuristic_offline.
    2. Reports honest unmeasured confidences (0.50) for skipped dimensions (specificity, future language).
    3. Retains evaluated confidences for deterministic regex (boundary: 0.95, agreement: 0.65).
    """
    engine = SemanticFeatureEngine()
    fragment_turn = normalize_generic_transcript(
        text="case.",
        speaker_id="client",
        start_ms=5000,
        end_ms=6000,
        call_sid="call_fragment_test",
    )

    snap = await engine.analyze_turn_semantic(fragment_turn, [])
    assert snap.extraction_mode == "heuristic_offline"
    assert snap.specificity_score == 0.0
    assert snap.future_language_score == 0.0
    assert snap.boundary_score == 0.0

    # Unmeasured features honestly report 0.50 (neutral prior), NEVER a fake 0.85/0.90!
    assert snap.specificity_confidence == 0.50
    assert snap.future_language_confidence == 0.50

    # Evaluated features reflect deterministic certainty
    assert snap.boundary_confidence == 0.95
    assert snap.recurrence_confidence == 0.80
    assert snap.agreement_confidence == 0.65
    assert snap.question_type_confidence == 0.70

    # Overall semantic_confidence reflects the weakest link of the bypass profile (0.50)
    assert snap.semantic_confidence == 0.50


@pytest.mark.asyncio
async def test_benign_explorative_turn_rejects_soft_llm_boundary_hallucination(monkeypatch):
    """
    Verifies that when an LLM returns a soft hallucinated boundary (e.g. 0.60) on benign
    explorative speech ('looking at options'), the engine filters it to 0.0 because
    neither the deterministic regex nor high-confidence LLM threshold (>= 0.80) was met.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_for_boundary")

    class MockChatCompletions:
        def create(self, *args, **kwargs):
            class Msg:
                content = (
                    '{"question_type": "none", "specificity_score": 0.1, '
                    '"future_language_score": 0.0, "agreement_score": 0.0, '
                    '"boundary_score": 0.6}'
                )
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class MockGroqClient:
        chat = type("Chat", (), {"completions": MockChatCompletions()})()

    engine = SemanticFeatureEngine(groq_client=MockGroqClient())
    turn = normalize_generic_transcript(
        text="So we have been looking at a few different options for a while now, honestly.",
        speaker_id="client",
        start_ms=1000,
        end_ms=5000,
        call_sid="call_boundary_test",
    )

    snap = await engine.analyze_turn_semantic(turn, [])
    assert snap.extraction_mode == "llm"
    # Soft LLM hallucination (0.60) MUST be filtered to 0.0
    assert snap.boundary_score == 0.0


@pytest.mark.asyncio
async def test_soft_but_genuine_boundary_statement_still_triggers_override(monkeypatch, tmp_path):
    """
    Adversarial test verifying zero false negatives for genuine, softly-worded boundary phrasing:
    1. Deterministic regex catches soft boundaries ('I'd rather you didn't reach out',
       'working with someone already', 'leave me alone') with boundary_score=1.0.
    2. Even under an uncalibrated / conservative LLM returning sub-0.80 scores (e.g. 0.65),
       effective_boundary remains 1.0 because regex carries zero-false-negative responsibility.
    3. The resulting boundary_score (1.0) feeds into MultiWindowAggregator and DownstreamInferenceEngine,
       successfully triggering the hard zero override on Readiness (0.0), suppressing Trust,
       and raising tension in Emotion.
    """
    from copilot.behavioral_evidence import MultiWindowAggregator, SQLiteEvidenceLogStore
    from copilot.behavioral_inference import DownstreamInferenceEngine
    from copilot.behavioral_timing import DeterministicTimingEngine

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_for_boundary")

    # Mock LLM returning a conservative / borderline score (0.65)
    class MockChatCompletions:
        def create(self, *args, **kwargs):
            class Msg:
                content = (
                    '{"question_type": "none", "specificity_score": 0.0, '
                    '"future_language_score": 0.0, "agreement_score": 0.0, '
                    '"boundary_score": 0.65}'
                )
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class MockGroqClient:
        chat = type("Chat", (), {"completions": MockChatCompletions()})()

    engine = SemanticFeatureEngine(groq_client=MockGroqClient())

    soft_boundary_turns = [
        "I'd rather you didn't reach out for a while.",
        "Please don't contact me about this again.",
        "We're actually working with someone already.",
        "I prefer not to be contacted anymore.",
        "Please leave me alone.",
        "We already signed with a realtor last week.",
    ]

    for text in soft_boundary_turns:
        turn = normalize_generic_transcript(
            text=text,
            speaker_id="client",
            start_ms=1000,
            end_ms=4000,
            call_sid="call_soft_boundary",
        )

        snap = await engine.analyze_turn_semantic(turn, [])
        # Deterministic regex + effective_boundary ensures 1.0 despite LLM scoring 0.65
        assert snap.boundary_score == 1.0, f"Failed zero-false-negative contract on: '{text}'"

        # Now verify full downstream inference override
        db_path = tmp_path / f"test_{abs(hash(text))}.db"
        store = SQLiteEvidenceLogStore(db_path=db_path)
        timing_engine = DeterministicTimingEngine()
        aggregator = MultiWindowAggregator(
            call_sid="call_soft_boundary",
            timing_engine=timing_engine,
            store=store,
        )
        inference_engine = DownstreamInferenceEngine()

        t_snap = timing_engine.process_utterance(turn)
        frame = aggregator.process_turn(turn, t_snap, semantic_snapshot=snap)
        inference = inference_engine.compute_inference(
            call_sid="call_soft_boundary",
            current_frame=frame,
        )

        # 1. Readiness is hard-overridden to 0.0
        assert inference.readiness.score == 0.0
        assert any("hard boundary" in d.lower() for d in inference.readiness.drivers)
        assert inference.readiness.confidence == 0.95

        # 2. Trust is heavily suppressed
        assert inference.trust.score <= 0.20
        assert any("boundary" in d.lower() for d in inference.trust.drivers)

        # 3. Emotion records hard boundary signal
        assert any("boundary" in s.lower() for s in inference.emotion.observable_signals)
        store.close()


def test_same_speaker_consecutive_objection_recurrence_with_conversational_framing():
    """
    Verifies that same-speaker consecutive turns repeating an objection are detected as
    'same_objection_repeated' even when surrounded by conversational framing / filler words
    and slight phonetic ASR variations (e.g. 'five person' instead of 'five percent').
    """
    engine = SemanticFeatureEngine()

    t13 = normalize_generic_transcript(
        text="Honestly, a five person commission feels pretty steep for what's involved here.",
        speaker_id="client",
        start_ms=44855,
        end_ms=50555,
        call_sid="call_monologue_test",
    )
    t14_text = "I hear you, but five person still just does not see it right with me."

    rec_id, rec_type, rec_count = engine.detect_semantic_recurrence(t14_text, [t13])
    assert rec_type == "same_objection_repeated"
    assert rec_count == 2
    assert rec_id is not None


def test_recurrence_adversarial_rejects_incidental_word_overlap():
    """
    Adversarial test verifying that two genuinely different turns sharing 1 or 2 incidental
    content words do NOT produce false positive objection recurrence:
    1. 'the closing timeline worries me' vs 'the closing paperwork is confusing' (only shares 'closing')
       -> must be 'none'.
    2. 'We looked at that house on the market with the large yard' vs
       'Is another house entering the market in that neighborhood soon?' (shares 'house' and 'market')
       -> must be 'none'.
    3. 'The closing timeline worries me' vs 'The closing timeline is really unclear to us'
       (genuine repeat of closing timeline concern sharing 'closing' and 'timeline' with high overlap)
       -> must be 'same_objection_repeated'.
    """
    engine = SemanticFeatureEngine()

    # 1. Shared single incidental word ('closing') across distinct concerns (timeline vs paperwork)
    t1 = normalize_generic_transcript("The closing timeline worries me.", "client", 1000, 3000, "call_adv")
    t2_text = "The closing paperwork is confusing."
    r_id, r_type, r_count = engine.detect_semantic_recurrence(t2_text, [t1])
    assert r_type == "none"

    # 2. Shared incidental topic nouns ('house', 'market') across distinct non-objection utterances
    t3 = normalize_generic_transcript(
        "We looked at that house on the market with the large yard.",
        "client",
        4000,
        7000,
        "call_adv",
    )
    t4_text = "Is another house entering the market in that neighborhood soon?"
    r_id2, r_type2, r_count2 = engine.detect_semantic_recurrence(t4_text, [t3])
    assert r_type2 == "none"

    # 3. Genuine recurrence of closing timeline concern
    t5_text = "The closing timeline is really unclear to us."
    r_id3, r_type3, r_count3 = engine.detect_semantic_recurrence(t5_text, [t1])
    assert r_type3 == "same_objection_repeated"
    assert r_count3 == 2
    assert r_id3 is not None





