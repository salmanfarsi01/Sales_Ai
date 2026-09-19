from __future__ import annotations

import pytest
from pathlib import Path

from copilot.behavioral_normalization import normalize_generic_transcript, NormalizedUtterance
from copilot.behavioral_timing import DeterministicTimingEngine, TimingFeatureSnapshot
from copilot.behavioral_semantic import (
    SemanticFeatureEngine,
    SemanticFeatureSnapshot,
    HARD_BOUNDARY_PATTERNS,
    SOFT_PREFERENCE_PATTERNS,
    extract_structured_contact_preference,
)
from copilot.behavioral_evidence import (
    SQLiteEvidenceLogStore,
    MultiWindowAggregator,
    BehavioralEvidenceSnapshot,
    MultiWindowEvidenceFrame,
)
from copilot.behavioral_inference import (
    DownstreamInferenceEngine,
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.conversation_state_models import (
    ContactPreference,
    ContactComplianceState,
    DecisionStakeholder,
)
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager


def _create_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.55,
    emotion_valence: float = 0.0,
    emotion_tension: float = 0.20,
    readiness: float = 0.55,
    engagement: float = 0.60,
    pacing: float = 0.60,
    momentum: float = 0.55,
    boundary: float = 0.0,
    recurrence_id: str = None,
    recurrence_type: str = "none",
    salesperson_strategy_tag: str = None,
    agreement: float = 0.50,
    specificity: float = 0.50,
    future_lang: float = 0.50,
    contact_preference: str = "none",
    contact_preference_details: str = None,
    contact_preference_confidence: float = 0.0,
    call_sid: str = "CA_decoupling_test",
):
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_60_90s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=0.85),
        pacing=DimensionScore(score=pacing, confidence=0.80, primary_horizon="last_5_10s"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_60_90s"),
        momentum=DimensionScore(score=momentum, confidence=0.85, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.85, primary_horizon="full_call"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        recurrence_id=recurrence_id,
        recurrence_type=recurrence_type,
        agreement_score=agreement,
        specificity_score=specificity,
        future_language_score=future_lang,
        contact_preference=contact_preference,
        contact_preference_details=contact_preference_details,
        contact_preference_confidence=contact_preference_confidence,
    )

    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
        salesperson_strategy_tag=salesperson_strategy_tag,
    )


# =============================================================================
# Issue #7 Tests: Contact Compliance (Soft Preference vs Hard Boundary)
# =============================================================================

def test_issue7_contact_preference_data_model_and_extraction():
    """Verifies ContactPreference structured extraction and schema separation."""
    # 1. Soft preference extraction
    soft_utt = "Please don't start texting me every day before Thursday."
    pref = extract_structured_contact_preference(soft_utt, source_turn_id=1, confidence=0.92)
    assert pref is not None
    assert pref.channel == "sms"
    assert pref.allowed is True
    assert pref.cadence == "reduced"
    assert pref.boundary_strength == "preference"
    assert pref.source_turn_id == 1
    assert pref.confidence == 0.92

    # 2. Channel preference: email instead of call
    channel_utt = "Can you email instead of calling?"
    pref_chan = extract_structured_contact_preference(channel_utt, source_turn_id=2, confidence=0.90)
    assert pref_chan is not None
    assert pref_chan.channel in ("call", "email")
    assert pref_chan.boundary_strength == "preference"

    # 3. Hard boundary text should NOT extract as soft preference
    hard_utt = "Do not call me again, take me off your list."
    pref_hard = extract_structured_contact_preference(hard_utt, source_turn_id=3)
    assert pref_hard is None


def test_issue7_compliance_state_separation_and_gate_interaction():
    """Proves soft preference does NOT trigger hard_boundary_active and does NOT block Gate Condition 7 or 6.
    Proves hard boundary activates hard_boundary_active and blocks Gate Condition 7 without destroying soft preferences.
    """
    manager = ConversationStateManager(call_sid="call_compliance_test", conversion_target="appointment")

    # Turn 1: Decision-maker establishes authority & neutral baseline
    t1 = _create_bundle(
        turn_id=1,
        speaker_id="client",
        text="I own the property and I make the listing decisions.",
        trust=0.55,
        readiness=0.55,
        agreement=0.60,
    )
    manager.process_turn_bundle(
        t1,
        decision_updates={
            "primary_decision_maker": "Self",
            "decision_maker_present": True,
            "stakeholders": [DecisionStakeholder(name="Self", role="owner", presence="on_call")],
        },
        fact_updates=[
            {"category": "property", "fact_key": "property_address", "fact_value": "123 Main St"},
            {"category": "timeline", "fact_key": "target_closing", "fact_value": "Summer"},
        ],
    )

    # Turn 2: Prospect expresses soft preference ("please don't text daily")
    t2 = _create_bundle(
        turn_id=2,
        speaker_id="client",
        text="That sounds fine, but please don't start texting me every day before Thursday.",
        trust=0.55,
        readiness=0.55,
        contact_preference="reduced_frequency",
        contact_preference_details="texting every day",
        contact_preference_confidence=0.90,
    )
    manager.process_turn_bundle(t2)

    comp = manager.current_state.contact_compliance
    # Assert soft preference is cleanly captured
    assert comp.hard_boundary_active is False
    assert comp.hard_boundary_reason is None
    assert len(comp.contact_preferences) == 1
    assert comp.contact_preferences[0].channel == "sms"
    assert comp.contact_preferences[0].cadence == "reduced"
    assert comp.contact_preferences[0].boundary_strength == "preference"

    # Assert Gate conditions: Condition 7 (no_active_boundary) MUST pass
    gate = manager.current_state.conversion_gate
    c7 = next(c for c in gate.conditions if c.condition_name == "no_active_boundary")
    assert c7.met is True
    # Condition 6 (plausible_logistics) MUST pass
    c6 = next(c for c in gate.conditions if c.condition_name == "plausible_logistics")
    assert c6.met is True

    # Turn 3: Prospect sets a hard boundary ("do not call me again")
    t3 = _create_bundle(
        turn_id=3,
        speaker_id="client",
        text="Actually stop calling me, do not call me again.",
        boundary=0.95,
        recurrence_type="boundary_repeated",
    )
    manager.process_turn_bundle(t3)

    comp3 = manager.current_state.contact_compliance
    # Assert hard boundary is active
    assert comp3.hard_boundary_active is True
    assert "call" in comp3.hard_boundary_channels
    # Assert previous soft preference was preserved non-destructively
    assert len(comp3.contact_preferences) >= 1
    assert any(p.channel == "sms" for p in comp3.contact_preferences)

    # Assert Gate Condition 7 fails and gate is closed
    gate3 = manager.current_state.conversion_gate
    assert gate3.is_open is False
    c7_t3 = next(c for c in gate3.conditions if c.condition_name == "no_active_boundary")
    assert c7_t3.met is False
    assert "Hard compliance boundary active" in c7_t3.reason

    # Turn 4: Subsequent turn (non-destructive persistence)
    t4 = _create_bundle(
        turn_id=4,
        speaker_id="client",
        text="I am hanging up now.",
        boundary=0.0,
    )
    manager.process_turn_bundle(t4)
    comp4 = manager.current_state.contact_compliance
    assert comp4.hard_boundary_active is True
    gate4 = manager.current_state.conversion_gate
    assert gate4.is_open is False


# =============================================================================
# Issue #8 Tests: Dimension Decoupling (Trust, Engagement, Momentum, Readiness)
# =============================================================================

def test_issue8_script_a_meeting_confirmation_decouples_trust():
    """Script A Validation:
    Prospect confirms meeting slot: 'Thursday at 3 works to see the numbers.'
    Commitment and Readiness jump / gate opens, but Trust remains flat (delta <= 0.10).
    """
    manager = ConversationStateManager(call_sid="call_script_a_decoupling", conversion_target="appointment")

    # Turn 1: Setup clean authority & property context with neutral trust
    t1 = _create_bundle(
        turn_id=1,
        speaker_id="client",
        text="I own the property and make the listing decisions.",
        trust=0.52,
        readiness=0.55,
        momentum=0.55,
        agreement=0.60,
    )
    manager.process_turn_bundle(
        t1,
        decision_updates={
            "primary_decision_maker": "Self",
            "decision_maker_present": True,
            "stakeholders": [DecisionStakeholder(name="Self", role="owner", presence="on_call")],
        },
        fact_updates=[
            {"category": "property", "fact_key": "property_address", "fact_value": "456 Oak Avenue"},
            {"category": "timeline", "fact_key": "target_closing", "fact_value": "Next month"},
        ],
    )
    initial_trust = manager.current_state.dimensions.trust
    assert 0.50 <= initial_trust <= 0.55

    # Turn 2: Prospect confirms meeting slot with substantive agreement
    t2 = _create_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 3 works to see the numbers.",
        trust=0.54,  # Downstream inference does not inject artificial agreement spike into trust
        readiness=0.80,
        momentum=0.75,
        agreement=0.85,
        specificity=0.80,
    )
    manager.process_turn_bundle(t2)

    gate = manager.current_state.conversion_gate
    assert gate.explicit_commitment_detected is True
    assert gate.is_open is True

    # Assert Trust did NOT jump in lockstep with readiness/commitment (delta <= 0.10)
    current_trust = manager.current_state.dimensions.trust
    trust_delta = abs(current_trust - initial_trust)
    assert trust_delta <= 0.10, f"Trust moved too much on logistical agreement: delta={trust_delta:.3f}"
    assert current_trust <= 0.60, f"Trust should remain moderate without deep disclosure: {current_trust:.2f}"


def test_issue8_sensitive_disclosure_increases_trust(tmp_path: Path):
    """Proves Trust increases specifically when sensitive constraints are disclosed,
    not on superficial logistical agreements.
    """
    db_path = tmp_path / "test_trust_disc.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator("call_trust_disc", timing_engine, store=store)
    engine = DownstreamInferenceEngine()

    # Turn 1: Logistical confirmation without sensitive disclosure
    utt1 = normalize_generic_transcript(
        text="Thursday at 3 works for me to see the numbers.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        call_sid="call_trust_disc",
    )
    t_snap1 = timing_engine.process_utterance(utt1)
    sem_snap1 = SemanticFeatureSnapshot(
        utterance_id=utt1.utterance_id,
        call_sid="call_trust_disc",
        speaker_id="client",
        agreement_score=0.90,
        specificity_score=0.75,
    )
    frame1 = aggregator.process_turn(utt1, t_snap1, sem_snap1)
    inf1 = engine.compute_inference("call_trust_disc", current_frame=frame1)
    # Baseline trust without disclosure
    assert inf1.trust.score <= 0.55

    # Turn 2: Vulnerable sensitive constraint disclosure (mortgage payoff, spouse, financial vulnerability)
    utt2 = normalize_generic_transcript(
        text="My wife and I must clear enough from the mortgage payoff to afford our next home before we can proceed.",
        speaker_id="client",
        start_ms=10000,
        end_ms=18000,
        call_sid="call_trust_disc",
    )
    t_snap2 = timing_engine.process_utterance(utt2)
    sem_snap2 = SemanticFeatureSnapshot(
        utterance_id=utt2.utterance_id,
        call_sid="call_trust_disc",
        speaker_id="client",
        agreement_score=0.40,
        specificity_score=0.85,
    )
    frame2 = aggregator.process_turn(utt2, t_snap2, sem_snap2)
    inf2 = engine.compute_inference("call_trust_disc", current_frame=frame2, recent_frames=[frame1])

    # Assert Trust jumps because of sensitive disclosure + multi-turn depth
    assert inf2.trust.score >= 0.70, f"Trust should reflect sensitive disclosure: got {inf2.trust.score:.2f}"
    assert any("sensitive constraint" in d.lower() for d in inf2.trust.drivers)
    store.close()


def test_issue8_script_b_commission_debate_decouples_engagement(tmp_path: Path):
    """Script B Validation:
    Prospect pushes back hard on commission ('Why should I pay 5%? What did you do to earn that?').
    Engagement remains high (>= 0.70) due to vocal participation, question velocity, and turn length,
    even while Momentum and Readiness dip from friction.
    """
    db_path = tmp_path / "test_debate.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator("call_script_b_debate", timing_engine, store=store)
    engine = DownstreamInferenceEngine()

    # Turn 1: Salesperson introduces commission
    utt1 = normalize_generic_transcript(
        text="Our standard commission fee is 5% which includes staging and premium marketing.",
        speaker_id="salesperson",
        start_ms=1000,
        end_ms=5000,
        call_sid="call_script_b_debate",
    )
    t_snap1 = timing_engine.process_utterance(utt1)
    frame1 = aggregator.process_turn(utt1, t_snap1)

    # Turn 2: Client challenges commission vigorously with questions and active speech
    utt2 = normalize_generic_transcript(
        text="Why should I pay 5% commission when the last agent did nothing? What do you actually do differently?",
        speaker_id="client",
        start_ms=7000,
        end_ms=15000,
        call_sid="call_script_b_debate",
    )
    t_snap2 = timing_engine.process_utterance(utt2)
    # Ensure high question velocity and client vocal share
    t_snap2.question_rate_per_min = 2.0
    sem_snap2 = SemanticFeatureSnapshot(
        utterance_id=utt2.utterance_id,
        call_sid="call_script_b_debate",
        speaker_id="client",
        agreement_score=0.10,
        specificity_score=0.70,
        question_type="evaluation",
        recurrence_id="objection_commission_fee",
        recurrence_type="same_objection_repeated",
    )
    frame2 = aggregator.process_turn(utt2, t_snap2, sem_snap2)
    # Ensure talk_ratio_client is in active/balanced range (0.55) and tension is elevated across horizons
    for snap in frame2.windows.values():
        snap.talk_ratio_client = 0.55
        snap.timing_features.question_rate_per_min = 2.0
        snap.semantic_features = sem_snap2

    inference = engine.compute_inference("call_script_b_debate", current_frame=frame2, recent_frames=[frame1])

    # 1. Engagement remains HIGH: Prospect is actively involved, debating, asking questions
    assert inference.engagement.score >= 0.70, (
        f"Engagement should remain high during vigorous objection debate: got {inference.engagement.score:.2f}"
    )
    assert any("inquiry" in d.lower() or "dialogue" in d.lower() or "active" in d.lower() for d in inference.engagement.drivers)

    # 2. Momentum DIPS: Dampened by conversational friction/recurrent objection
    assert inference.momentum.score <= 0.55, (
        f"Momentum should be dampened by friction/objection: got {inference.momentum.score:.2f}"
    )

    # 3. Readiness DIPS: Dampened by active unresolved objection
    assert inference.readiness.score <= 0.50, (
        f"Readiness should dip under active objection: got {inference.readiness.score:.2f}"
    )

    # 4. Confirm Decoupling: Engagement and Momentum diverge by at least 0.15
    decoupling_gap = inference.engagement.score - inference.momentum.score
    assert decoupling_gap >= 0.15, f"Expected engagement to diverge from momentum under debate: gap={decoupling_gap:.2f}"
    store.close()


def test_hard_boundary_is_sticky_against_subsequent_softer_preference():
    """Client Question Validation:
    Proves that a hard boundary logged first ('Do not call me again') is PERMANENTLY STICKY
    and is NOT downgraded, diluted, or lifted when a softer statement is uttered later
    ('Actually, texting occasionally is fine').
    """
    manager = ConversationStateManager(call_sid="call_sticky_boundary", conversion_target="appointment")

    # Turn 1: Decision maker baseline
    t1 = _create_bundle(
        turn_id=1,
        speaker_id="client",
        text="I own the property and handle the sale.",
        trust=0.55,
        readiness=0.55,
    )
    manager.process_turn_bundle(
        t1,
        decision_updates={
            "primary_decision_maker": "Self",
            "decision_maker_present": True,
            "stakeholders": [DecisionStakeholder(name="Self", role="owner", presence="on_call")],
        },
        fact_updates=[
            {"category": "property", "fact_key": "property_address", "fact_value": "789 Pine Road"},
            {"category": "timeline", "fact_key": "target_closing", "fact_value": "Immediate"},
        ],
    )

    # Turn 2: Prospect sets a hard boundary
    t2 = _create_bundle(
        turn_id=2,
        speaker_id="client",
        text="Do not call me again, remove my number from your database.",
        boundary=1.0,
        recurrence_type="boundary_repeated",
    )
    manager.process_turn_bundle(t2)

    comp2 = manager.current_state.contact_compliance
    assert comp2.hard_boundary_active is True
    assert "call" in comp2.hard_boundary_channels
    assert manager.current_state.conversion_gate.is_open is False
    c7_turn2 = next(c for c in manager.current_state.conversion_gate.conditions if c.condition_name == "no_active_boundary")
    assert c7_turn2.met is False

    # Turn 3: Prospect utters a softer statement ('actually, texting occasionally is fine')
    t3 = _create_bundle(
        turn_id=3,
        speaker_id="client",
        text="Actually, texting occasionally is fine once a week.",
        boundary=0.0,
        contact_preference="reduced_frequency",
        contact_preference_details="texting once a week",
        contact_preference_confidence=0.90,
    )
    manager.process_turn_bundle(t3)

    comp3 = manager.current_state.contact_compliance
    # CRITICAL COMPLIANCE INVARIANT:
    # 1. hard_boundary_active remains strictly True - NOT downgraded by later soft language
    assert comp3.hard_boundary_active is True
    # 2. Prohibited channel remains logged
    assert "call" in comp3.hard_boundary_channels
    # 3. The soft preference was recorded non-destructively
    assert any(p.channel == "sms" and p.cadence == "reduced" for p in comp3.contact_preferences)
    # 4. Gate Condition 7 remains BLOCKED (gate remains closed)
    gate3 = manager.current_state.conversion_gate
    assert gate3.is_open is False
    c7_turn3 = next(c for c in gate3.conditions if c.condition_name == "no_active_boundary")
    assert c7_turn3.met is False
    assert "Hard compliance boundary active" in c7_turn3.reason


def test_short_blunt_sensitive_disclosure_increases_trust_without_word_count_gate(tmp_path: Path):
    """Client Question Validation:
    Proves that a short, blunt vulnerable disclosure under 12 words
    ('My wife has to sign off too, always has' - 9 words)
    successfully triggers the sensitive disclosure trust boost (+0.15)
    WITHOUT being blocked or gated by the turn_length_words >= 12 threshold.
    """
    db_path = tmp_path / "test_short_disc.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator("call_short_disc", timing_engine, store=store)
    engine = DownstreamInferenceEngine()

    # Turn 1: Neutral opening
    utt1 = normalize_generic_transcript(
        text="Hello, we are considering selling later this year.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        call_sid="call_short_disc",
    )
    t_snap1 = timing_engine.process_utterance(utt1)
    frame1 = aggregator.process_turn(utt1, t_snap1)
    inf1 = engine.compute_inference("call_short_disc", current_frame=frame1)
    assert inf1.trust.score <= 0.55

    # Turn 2: Short blunt sensitive disclosure (9 words < 12 words)
    utt2 = normalize_generic_transcript(
        text="My wife has to sign off too, always has.",
        speaker_id="client",
        start_ms=6000,
        end_ms=9000,
        call_sid="call_short_disc",
    )
    t_snap2 = timing_engine.process_utterance(utt2)
    assert t_snap2.turn_length_words < 12, f"Expected < 12 words, got {t_snap2.turn_length_words}"

    sem_snap2 = SemanticFeatureSnapshot(
        utterance_id=utt2.utterance_id,
        call_sid="call_short_disc",
        speaker_id="client",
        agreement_score=0.40,
        specificity_score=0.60,
    )
    frame2 = aggregator.process_turn(utt2, t_snap2, sem_snap2)
    inf2 = engine.compute_inference("call_short_disc", current_frame=frame2, recent_frames=[frame1])

    # Assert Trust increased specifically due to sensitive constraint disclosure (+0.15)
    # Even though t_len < 12 precluded the sustained disclosure depth boost (+0.10)
    assert inf2.trust.score >= 0.65, f"Short sensitive disclosure should boost trust: got {inf2.trust.score:.2f}"
    assert any("sensitive constraint" in d.lower() for d in inf2.trust.drivers)
    # Sustained disclosure depth driver should NOT be present since turn was short
    assert not any("sustained multi-turn disclosure depth" in d.lower() for d in inf2.trust.drivers)
    store.close()


def test_casual_family_mention_does_not_trigger_sensitive_trust_boost(tmp_path: Path):
    """Client Question Validation:
    Proves that casual positive family mentions ('My family and I love this neighborhood')
    do NOT trigger the sensitive disclosure trust boost because they lack decision authority
    or financial constraint markers.
    """
    db_path = tmp_path / "test_casual_family.db"
    store = SQLiteEvidenceLogStore(db_path=db_path)
    timing_engine = DeterministicTimingEngine()
    aggregator = MultiWindowAggregator("call_casual_family", timing_engine, store=store)
    engine = DownstreamInferenceEngine()

    # Turn 1: Neutral opening
    utt1 = normalize_generic_transcript(
        text="Hello, we were thinking about property values in the area.",
        speaker_id="client",
        start_ms=1000,
        end_ms=4000,
        call_sid="call_casual_family",
    )
    t_snap1 = timing_engine.process_utterance(utt1)
    frame1 = aggregator.process_turn(utt1, t_snap1)
    inf1 = engine.compute_inference("call_casual_family", current_frame=frame1)

    # Turn 2: Casual family mention without decision-making or financial constraint (7 words < 12)
    utt2 = normalize_generic_transcript(
        text="My family and I love this neighborhood.",
        speaker_id="client",
        start_ms=6000,
        end_ms=9000,
        call_sid="call_casual_family",
    )
    t_snap2 = timing_engine.process_utterance(utt2)
    sem_snap2 = SemanticFeatureSnapshot(
        utterance_id=utt2.utterance_id,
        call_sid="call_casual_family",
        speaker_id="client",
        agreement_score=0.40,
        specificity_score=0.40,
    )
    frame2 = aggregator.process_turn(utt2, t_snap2, sem_snap2)
    inf2 = engine.compute_inference("call_casual_family", current_frame=frame2, recent_frames=[frame1])

    # Assert Trust does NOT trigger sensitive constraint boost
    assert inf2.trust.score <= 0.55, f"Casual family mention should not inflate trust: got {inf2.trust.score:.2f}"
    assert not any("sensitive constraint" in d.lower() for d in inf2.trust.drivers)
    store.close()


