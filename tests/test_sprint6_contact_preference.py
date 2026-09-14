import pytest
from pathlib import Path
import tempfile
import asyncio

from copilot.behavioral_normalization import NormalizedUtterance, NormalizedWord
from copilot.behavioral_timing import DeterministicTimingEngine
from copilot.behavioral_semantic import SemanticFeatureEngine, SemanticFeatureSnapshot
from copilot.behavioral_baseline import (
    BaselineAndChangePointEngine,
    ContactPreferenceStore,
    ContactPreferenceRecord,
)
from copilot.behavioral_evidence import SQLiteEvidenceLogStore, MultiWindowAggregator
from copilot.behavioral_inference import DownstreamInferenceEngine
from copilot.behavioral_pipeline_service import execute_behavioral_turn_pipeline


@pytest.fixture
def semantic_engine():
    return SemanticFeatureEngine(groq_client=None)


def test_soft_contact_preference_patterns(semantic_engine):
    # Test reduced_frequency
    pref, conf, details = semantic_engine.detect_contact_preference("Please don't call so often, once a week is fine.")
    assert pref == "reduced_frequency"
    assert conf >= 0.85
    assert details is not None

    # Test channel_restriction
    pref, conf, details = semantic_engine.detect_contact_preference("Can you email instead of calling?")
    assert pref == "channel_restriction"
    assert conf >= 0.85

    pref2, conf2, details2 = semantic_engine.detect_contact_preference("Just email me the details please.")
    assert pref2 == "channel_restriction"
    assert conf2 >= 0.85

    # Test timing_restriction
    pref, conf, details = semantic_engine.detect_contact_preference("Only reach out during business hours.")
    assert pref == "timing_restriction"
    assert conf >= 0.85

    # Test non-preference
    pref, conf, details = semantic_engine.detect_contact_preference("Sounds good, let's talk about listing the house.")
    assert pref == "none"
    assert conf == 0.0


def test_soft_preference_does_not_trigger_boundary_override(semantic_engine):
    soft_utterances = [
        "Please don't call so often, maybe once a week is plenty.",
        "Email instead of calling please.",
        "Just email me whenever you have updates.",
        "Only reach out during business hours.",
        "Don't call before 10am.",
    ]

    for utt_text in soft_utterances:
        boundary_score = semantic_engine.classify_boundary_deterministic(utt_text)
        assert boundary_score == 0.0, f"False positive boundary override for: {utt_text}"

        pref_type, pref_conf, _ = semantic_engine.detect_contact_preference(utt_text)
        assert pref_type != "none"


def test_adversarial_mixed_boundary_and_soft_preference(semantic_engine):
    """Proves that when an utterance contains both soft preference phrasing and explicit hard cease-contact
    language (e.g. 'please don't call me anymore, just text instead'), the hard boundary compliance
    override STRICTLY takes precedence over the soft preference."""
    mixed_adversarial = [
        "Please don't call me anymore, just text instead.",
        "Stop calling me, email me instead.",
        "Don't call me again, send an email.",
        "Take me off your list and email me the info.",
        "Lose my number, text only.",
        "Do not contact me anymore, just send an email.",
    ]

    for utt_text in mixed_adversarial:
        boundary_score = semantic_engine.classify_boundary_deterministic(utt_text)
        assert boundary_score == 1.0, f"Expected hard boundary override (1.0) for: '{utt_text}', got {boundary_score}"


@pytest.mark.asyncio
async def test_adversarial_mixed_turn_semantic_overrides_soft_preference(semantic_engine):
    """Proves that analyze_turn_semantic sets boundary_score=1.0 and suppresses soft contact preference."""
    utt = NormalizedUtterance(
        utterance_id="u_adv_1",
        call_sid="call_adv_mixed",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        text="Please don't call me anymore, just text instead.",
        words=[NormalizedWord(word="Please", start_ms=1000, end_ms=1300)],
        asr_confidence=0.96,
    )
    snap = await semantic_engine.analyze_turn_semantic(utt, context_history=[])
    assert snap.boundary_score == 1.0
    assert snap.contact_preference == "none"


@pytest.mark.asyncio
async def test_contact_preference_semantic_snapshot(semantic_engine):
    utt = NormalizedUtterance(
        utterance_id="u_pref_1",
        call_sid="call_pref_123",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        text="Please don't call so often, email instead of calling.",
        words=[NormalizedWord(word="Please", start_ms=1000, end_ms=1300)],
        asr_confidence=0.96,
    )
    snap = await semantic_engine.analyze_turn_semantic(utt, context_history=[])
    assert snap.boundary_score == 0.0
    assert snap.contact_preference in ("reduced_frequency", "channel_restriction")
    assert snap.contact_preference_confidence >= 0.85


def test_contact_preference_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "test_prefs.json"
        store = ContactPreferenceStore(store_path=store_path)

        assert store.get_preference("prospect_999") is None

        record = ContactPreferenceRecord(
            preference="channel_restriction",
            confidence=0.90,
            details="email instead of calling",
            first_observed_turn=3,
            call_sid="call_abc",
            timestamp_ms=12500,
        )
        store.save_preference("prospect_999", record)

        saved = store.get_preference("prospect_999")
        assert saved is not None
        assert saved.preference == "channel_restriction"
        assert saved.confidence == 0.90
        assert saved.details == "email instead of calling"
        assert saved.first_observed_turn == 3


@pytest.mark.asyncio
async def test_pipeline_soft_preference_readiness_and_latency(tmp_path):
    pref_store_path = tmp_path / "prefs.json"
    pref_store = ContactPreferenceStore(store_path=pref_store_path)

    db_path = tmp_path / "evidence.db"
    evidence_store = SQLiteEvidenceLogStore(db_path=db_path)

    timing_engine = DeterministicTimingEngine()
    semantic_engine = SemanticFeatureEngine(groq_client=None)
    baseline_engine = BaselineAndChangePointEngine(call_sid="call_test_pipeline")
    aggregator = MultiWindowAggregator("call_test_pipeline", timing_engine, baseline_engine, evidence_store)
    inference_engine = DownstreamInferenceEngine()

    utts = [
        NormalizedUtterance(
            utterance_id="u1",
            call_sid="call_test_pipeline",
            speaker_id="salesperson",
            start_ms=0,
            end_ms=2500,
            text="Hi John, following up on our listing appointment for Friday.",
            words=[
                NormalizedWord(word="Hi", start_ms=0, end_ms=300),
                NormalizedWord(word="John", start_ms=350, end_ms=700),
            ],
            asr_confidence=0.98,
        ),
        NormalizedUtterance(
            utterance_id="u2",
            call_sid="call_test_pipeline",
            speaker_id="client",
            start_ms=3200,
            end_ms=6500,
            text="Friday works for me at 2pm, but please email instead of calling so much.",
            words=[
                NormalizedWord(word="Friday", start_ms=3200, end_ms=3600),
                NormalizedWord(word="works", start_ms=3650, end_ms=4000),
                NormalizedWord(word="for", start_ms=4050, end_ms=4200),
                NormalizedWord(word="me", start_ms=4250, end_ms=4400),
            ],
            asr_confidence=0.97,
        ),
    ]

    result = await execute_behavioral_turn_pipeline(
        segmented_utts=utts,
        call_sid="call_test_pipeline",
        prospect_id="prospect_john_1",
        agent_id="agent_sarah_1",
        transcript="...",
        timing_engine=timing_engine,
        semantic_engine=semantic_engine,
        baseline_engine=baseline_engine,
        aggregator=aggregator,
        inference_engine=inference_engine,
        preference_store=pref_store,
    )

    # Check inference latency is non-zero (greater than 0.0)
    assert result["latest_inference"]["inference_latency_ms"] >= 0.01
    for turn in result["play_by_play_turns"]:
        assert turn["inference_latency_ms"] >= 0.01

    # Check turn 2 had soft contact preference detected
    turn2 = result["play_by_play_turns"][1]
    assert turn2["contact_preference"] in ("channel_restriction", "reduced_frequency")
    assert turn2["boundary_score"] == 0.0

    # Check readiness is NOT suppressed to zero
    assert result["latest_inference"]["readiness"]["score"] > 0.30
    assert result["latest_inference"]["trust"]["score"] > 0.40

    # Check preference persisted to store
    stored_pref = pref_store.get_preference("prospect_john_1")
    assert stored_pref is not None
    assert stored_pref.preference in ("channel_restriction", "reduced_frequency")
    assert result["prospect_contact_preference"] is not None
