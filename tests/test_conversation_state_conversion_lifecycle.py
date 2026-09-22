from __future__ import annotations

import pytest
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_models import (
    ConversionEventObject,
    ConversionEventStatus,
    DecisionStakeholder,
    PersistentFactRecord,
)
from copilot.conversation_conversion import MeetingConversionGateEngine
from copilot.conversation_replay import ConversationReplayEngine
from copilot.conversation_supersession import TruthSupersessionDetector


def _create_turn_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.75,
    emotion_valence: float = 0.1,
    emotion_tension: float = 0.15,
    readiness: float = 0.65,
    engagement: float = 0.70,
    pacing: float = 0.60,
    boundary: float = 0.0,
    agreement: float = 0.60,
    specificity: float = 0.70,
    future_lang: float = 0.60,
    contact_preference: str = "none",
    salesperson_strategy_tag: str = None,
    call_sid: str = "CA_conv_lifecycle_test",
):
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=0.8),
        pacing=DimensionScore(score=pacing, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.65, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        agreement_score=agreement,
        specificity_score=specificity,
        future_language_score=future_lang,
        contact_preference=contact_preference,
    )

    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
        salesperson_strategy_tag=salesperson_strategy_tag,
    )


def _setup_baseline_manager(call_sid: str) -> ConversationStateManager:
    manager = ConversationStateManager(call_sid=call_sid, conversion_target="appointment")
    t1 = _create_turn_bundle(
        turn_id=1,
        speaker_id="client",
        text="I own the house and make decisions on listing it.",
    )
    manager.process_turn_bundle(
        t1,
        decision_updates={
            "primary_decision_maker": "Self (Owner)",
            "decision_maker_present": True,
            "stakeholders": [DecisionStakeholder(name="Self", role="owner", presence="on_call")],
        },
        fact_updates=[
            {"category": "property", "fact_key": "property_address", "fact_value": "123 Main St"},
        ],
    )
    return manager


def test_conversion_event_status_enum_and_lineage_fields():
    """Verify ConversionEventStatus enum matches client requirements and ConversionEventObject has lineage fields."""
    assert ConversionEventStatus.PROPOSED == "proposed"
    assert ConversionEventStatus.TENTATIVE == "tentative"
    assert ConversionEventStatus.CONFIRMED == "confirmed"
    assert ConversionEventStatus.CANCELLED == "cancelled"
    assert ConversionEventStatus.RESCHEDULED == "rescheduled"
    assert ConversionEventStatus.COMPLETED == "completed"

    obj = ConversionEventObject(
        status=ConversionEventStatus.CONFIRMED,
        conversion_type="in_person_meeting",
        start_at="Thursday At 4",
        source_turn_ids=[2],
        superseded_by_event_id="conv_next",
        supersedes_event_id="conv_prev",
        superseded_at_turn_id=3,
        reversal_reason="rescheduled",
    )
    assert obj.status == ConversionEventStatus.CONFIRMED
    assert obj.superseded_by_event_id == "conv_next"
    assert obj.supersedes_event_id == "conv_prev"
    assert obj.superseded_at_turn_id == 3
    assert obj.reversal_reason == "rescheduled"


def test_trigger_a_hard_boundary_automatically_cancels_confirmed_event():
    """Trigger A: Hard boundary fires while active conversion event is confirmed.
    The old confirmed record is NOT overwritten; a new CANCELLED record is linked.
    """
    manager = _setup_baseline_manager("CA_trigA_boundary")

    # Turn 2: Prospect confirms appointment
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me, let's meet then.",
        agreement=0.80,
    )
    s2 = manager.process_turn_bundle(t2)
    ev_confirmed = s2.conversion_event
    assert ev_confirmed is not None
    assert ev_confirmed.status == ConversionEventStatus.CONFIRMED
    assert ev_confirmed.superseded_by_event_id is None

    # Turn 3: Hard compliance boundary is triggered
    t3 = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="Stop calling me! Take me off your list and do not call again.",
        boundary=0.95,
        trust=0.30,
        agreement=0.0,
    )
    manager.current_state.contact_compliance.hard_boundary_active = True
    s3 = manager.process_turn_bundle(t3)

    ev_cancelled = s3.conversion_event
    assert ev_cancelled is not None
    assert ev_cancelled.status == ConversionEventStatus.CANCELLED
    assert ev_cancelled.reversal_reason == "hard_boundary"
    assert ev_cancelled.supersedes_event_id == ev_confirmed.event_id

    # Check that historical confirmed event was superseded forward, NOT erased
    history = manager.get_conversion_event_history()
    assert len(history) == 3
    assert history[0].status == ConversionEventStatus.ELIGIBLE
    assert history[1].event_id == ev_confirmed.event_id
    assert history[1].status == ConversionEventStatus.CONFIRMED
    assert history[1].superseded_by_event_id == ev_cancelled.event_id
    assert history[1].superseded_at_turn_id == 3

    assert history[2].event_id == ev_cancelled.event_id
    assert history[2].status == ConversionEventStatus.CANCELLED
    assert history[2].supersedes_event_id == ev_confirmed.event_id


def test_trigger_b_explicit_reversal_cancels_without_boundary():
    """Trigger B: Explicit cancellation language without any compliance boundary.
    'Actually, something came up, let's cancel Thursday.'
    """
    manager = _setup_baseline_manager("CA_trigB_reversal")

    # Turn 2: Confirmed meeting
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me.",
        agreement=0.80,
    )
    s2 = manager.process_turn_bundle(t2)
    ev_confirmed = s2.conversion_event
    assert ev_confirmed.status == ConversionEventStatus.CONFIRMED

    # Turn 3: Explicit cancellation (boundary is 0.0)
    t3 = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="Actually, something came up, let's cancel Thursday.",
        boundary=0.0,
        agreement=0.40,
    )
    s3 = manager.process_turn_bundle(t3)
    ev_cancelled = s3.conversion_event

    assert ev_cancelled is not None
    assert ev_cancelled.status == ConversionEventStatus.CANCELLED
    assert "cancel" in ev_cancelled.reversal_reason.lower()
    assert ev_cancelled.supersedes_event_id == ev_confirmed.event_id

    # All records exist in history
    history = manager.get_conversion_event_history()
    assert len(history) == 3
    assert history[0].status == ConversionEventStatus.ELIGIBLE
    assert history[1].status == ConversionEventStatus.CONFIRMED
    assert history[1].superseded_by_event_id == ev_cancelled.event_id
    assert history[2].status == ConversionEventStatus.CANCELLED
    assert history[2].supersedes_event_id == ev_confirmed.event_id


def test_trigger_b_adversarial_negations_and_hypotheticals_do_not_cancel():
    """Adversarial negative testing on detect_explicit_reversal:
    Negated cancellations and hypothetical phrasing must NOT trigger cancellation.
    """
    engine = MeetingConversionGateEngine()
    manager = _setup_baseline_manager("CA_trigB_adversarial")

    # Negation 1: "I do not want to cancel"
    b_neg = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="I do not want to cancel Thursday, I still want to meet.",
    )
    is_rev, reason = engine.detect_explicit_reversal(b_neg, manager.current_state)
    assert is_rev is False
    assert reason is None

    # Negation 2: "Don't cancel"
    b_dont = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="Don't cancel our walkthrough, we are still good for then.",
    )
    is_rev, reason = engine.detect_explicit_reversal(b_dont, manager.current_state)
    assert is_rev is False

    # Hypothetical 1: "What if hypothetically we had to cancel"
    b_hyp = _create_turn_bundle(
        turn_id=4,
        speaker_id="client",
        text="What if hypothetically we had to cancel, could we reschedule later?",
    )
    is_rev, reason = engine.detect_explicit_reversal(b_hyp, manager.current_state)
    assert is_rev is False
    assert reason is None


def test_full_lifecycle_chain_proposed_confirmed_cancelled():
    """Full chain: PROPOSED -> CONFIRMED -> CANCELLED.
    All 3 events are preserved in chronological order with bidirectional links.
    """
    manager = _setup_baseline_manager("CA_full_chain")

    # Turn 2 (Salesperson proposes slot)
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="salesperson",
        text="Would Thursday at 4:00 PM work for you to see the property?",
        salesperson_strategy_tag="specific_slot_proposal",
    )
    s2 = manager.process_turn_bundle(t2)
    ev_prop = s2.conversion_event
    assert ev_prop is not None
    assert ev_prop.status == ConversionEventStatus.PROPOSED
    assert "Thursday At 4:00" in ev_prop.start_at

    # Turn 3 (Client confirms)
    t3 = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="Yes, that works.",
        agreement=0.85,
    )
    s3 = manager.process_turn_bundle(t3)
    ev_conf = s3.conversion_event
    assert ev_conf is not None
    assert ev_conf.status == ConversionEventStatus.CONFIRMED
    assert ev_conf.supersedes_event_id == ev_prop.event_id

    # Turn 4 (Client cancels)
    t4 = _create_turn_bundle(
        turn_id=4,
        speaker_id="client",
        text="Never mind, forget Thursday, I changed my mind.",
        agreement=0.30,
    )
    s4 = manager.process_turn_bundle(t4)
    ev_canc = s4.conversion_event
    assert ev_canc is not None
    assert ev_canc.status == ConversionEventStatus.CANCELLED
    assert ev_canc.supersedes_event_id == ev_conf.event_id

    # Verify complete chain in history (ELIGIBLE -> PROPOSED -> CONFIRMED -> CANCELLED)
    history = manager.get_conversion_event_history()
    assert len(history) == 4

    assert history[0].status == ConversionEventStatus.ELIGIBLE

    assert history[1].status == ConversionEventStatus.PROPOSED
    assert history[1].superseded_by_event_id == ev_conf.event_id
    assert history[1].superseded_at_turn_id == 3

    assert history[2].status == ConversionEventStatus.CONFIRMED
    assert history[2].supersedes_event_id == ev_prop.event_id
    assert history[2].superseded_by_event_id == ev_canc.event_id
    assert history[2].superseded_at_turn_id == 4

    assert history[3].status == ConversionEventStatus.CANCELLED
    assert history[3].supersedes_event_id == ev_conf.event_id
    assert history[3].superseded_by_event_id is None


def test_reschedule_supersession_links_prior_confirmed_event():
    """Client reschedules from Thursday to Friday.
    The Thursday event is superseded by the Friday event, preserving both.
    """
    manager = _setup_baseline_manager("CA_reschedule")

    # Turn 2: Thursday confirmed
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me.",
        agreement=0.80,
    )
    s2 = manager.process_turn_bundle(t2)
    ev_thur = s2.conversion_event
    assert ev_thur.status == ConversionEventStatus.CONFIRMED
    assert "Thursday At 4" in ev_thur.start_at

    # Turn 3: Reschedule to Friday at 2
    t3 = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="Actually, can we do Friday at 2 instead? Friday at 2 works much better.",
        agreement=0.80,
    )
    s3 = manager.process_turn_bundle(t3)
    ev_fri = s3.conversion_event

    assert ev_fri is not None
    assert ev_fri.status == ConversionEventStatus.CONFIRMED
    assert "Friday At 2" in ev_fri.start_at
    assert ev_fri.supersedes_event_id == ev_thur.event_id
    assert ev_fri.reversal_reason == "rescheduled"

    history = manager.get_conversion_event_history()
    assert len(history) == 3
    assert history[0].status == ConversionEventStatus.ELIGIBLE
    assert history[1].event_id == ev_thur.event_id
    assert history[1].superseded_by_event_id == ev_fri.event_id
    assert history[2].event_id == ev_fri.event_id
    assert history[2].supersedes_event_id == ev_thur.event_id


def test_replay_service_get_conversion_event_history(tmp_path):
    """Verify ConversationReplayEngine.get_conversion_event_history returns full chain."""
    engine = ConversationReplayEngine(reports_dir=tmp_path)
    call_sid = "sim_lifecycle_replay"

    raw_turns = [
        {"turn_id": 1, "speaker_id": "client", "text": "I own 456 Oak Avenue and make all decisions."},
        {"turn_id": 2, "speaker_id": "salesperson", "text": "Would Thursday at 4:00 PM work for you?", "strategy_tag": "specific_slot_proposal"},
        {"turn_id": 3, "speaker_id": "client", "text": "Yes, Thursday at 4 works.", "agreement": 0.85},
        {"turn_id": 4, "speaker_id": "client", "text": "Actually, cancel Thursday.", "agreement": 0.30},
    ]

    report = engine.replay_dialogue_turns(call_sid=call_sid, raw_turns=raw_turns, save_report=True)
    assert report is not None

    events_history = engine.get_conversion_event_history(call_sid)
    assert len(events_history) >= 2
    statuses = [e["status"] for e in events_history]
    assert "confirmed" in statuses
    assert "cancelled" in statuses


def test_truth_supersession_detector_event_and_meeting_fact_relation_classification():
    """Verify TruthSupersessionDetector evaluates conversion events and confirmed_meeting_time facts
    using the unified relation classification logic (REVERSES, UPDATES, UNCHANGED).
    """
    detector = TruthSupersessionDetector()

    # 1. Evaluate event supersession (REVERSES)
    event_thur = ConversionEventObject(
        event_id="conv_test_thur",
        conversion_type="in_person_meeting",
        status=ConversionEventStatus.CONFIRMED,
        start_at="Thursday At 4:00 PM",
    )
    dec_rev = detector.evaluate_event_supersession("Actually, let's cancel Thursday.", event_thur)
    assert dec_rev.has_supersession is True
    assert dec_rev.relation == "REVERSES"
    assert dec_rev.new_truth_value == "cancelled"

    # 2. Evaluate event supersession (UPDATES / Reschedule)
    dec_upd = detector.evaluate_event_supersession("Can we do Friday at 2:00 PM instead?", event_thur)
    assert dec_upd.has_supersession is True
    assert dec_upd.relation == "UPDATES"
    assert "Friday At 2:00 Pm" in dec_upd.new_truth_value

    # 3. Evaluate event supersession (UNCHANGED / Adversarial Negation)
    dec_neg = detector.evaluate_event_supersession("I do not want to cancel, we are still good for Thursday.", event_thur)
    assert dec_neg.has_supersession is False
    assert dec_neg.relation == "UNCHANGED"

    # 4. Evaluate confirmed_meeting_time fact via evaluate_turn
    fact_thur = PersistentFactRecord(
        fact_id="fact_meet_001",
        category="timeline",
        fact_key="confirmed_meeting_time",
        fact_value="Thursday At 4:00 PM",
        source_turn_id=2,
        timestamp_ms=1000,
    )
    dec_fact_rev = detector.evaluate_turn("Never mind, forget Thursday.", [fact_thur])
    assert len(dec_fact_rev) == 1
    assert dec_fact_rev[0].relation == "REVERSES"
    assert dec_fact_rev[0].new_truth_value == "Cancelled"


def test_fact_supersession_is_automatic_deterministic_cascade_from_event_status():
    """Verify that when a conversion event is cancelled or rescheduled, the corresponding
    confirmed_meeting_time fact is automatically and deterministically superseded,
    guaranteeing zero internal contradiction between the event store and the fact store.
    """
    manager = _setup_baseline_manager("CA_fact_cascade")

    # Turn 2: Confirm Thursday at 4
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me.",
        agreement=0.85,
    )
    s2 = manager.process_turn_bundle(t2)
    ev2 = s2.conversion_event
    assert ev2.status == ConversionEventStatus.CONFIRMED

    # Active fact must show Thursday at 4
    active_facts = [f for f in s2.facts if f.fact_key == "confirmed_meeting_time" and f.status == "active"]
    assert len(active_facts) == 1
    assert "Thursday At 4" in active_facts[0].fact_value

    # Turn 3: Cancellation via hard compliance boundary
    t3 = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="Do not contact me anymore! Take me off your list!",
        boundary=0.95,
    )
    manager.current_state.contact_compliance.hard_boundary_active = True
    s3 = manager.process_turn_bundle(t3)

    # Event is CANCELLED
    assert s3.conversion_event.status == ConversionEventStatus.CANCELLED

    # Deterministic cascade: confirmed_meeting_time fact MUST now be superseded to "Cancelled"
    active_facts_post = [f for f in s3.facts if f.fact_key == "confirmed_meeting_time" and f.status == "active"]
    assert len(active_facts_post) == 1
    assert active_facts_post[0].fact_value == "Cancelled"

    # Prior fact is preserved in superseded status with forward pointer
    superseded_facts = [f for f in s3.facts if f.fact_key == "confirmed_meeting_time" and f.status == "superseded"]
    assert len(superseded_facts) == 1
    assert "Thursday At 4" in superseded_facts[0].fact_value
    assert superseded_facts[0].superseded_by_fact_id == active_facts_post[0].fact_id


def test_hedged_proposal_acceptance_and_gate_regression_muc5lyux():
    """Verify sim_muc5lyux exact behavior:
    1. Turn 5: Salesperson proposes walkthrough on Thursday -> Event PROPOSED ('Thursday')
    2. Turn 6: Client hedged acceptance ('Sure, Thursday could work.') ->
       - Gate explicit_commitment_detected is False, commitment_slot is None
       - Event status is TENTATIVE with concrete start_at='Thursday' (no generic placeholder)
       - Fact tentative_meeting_time is recorded, confirmed_meeting_time is NOT created
    3. Turn 7: Client adds absent decision maker -> Gate regresses to closed, event transitions to BLOCKED ('Thursday').
    """
    manager = _setup_baseline_manager("CA_muc5lyux_hedged")

    # Turn 5: Salesperson proposal
    t5 = _create_turn_bundle(
        turn_id=5,
        speaker_id="salesperson",
        text="Great - would Thursday work for a walkthrough?",
        salesperson_strategy_tag="specific_slot_proposal",
    )
    s5 = manager.process_turn_bundle(t5)
    ev5 = s5.conversion_event
    assert ev5 is not None
    assert ev5.status == ConversionEventStatus.PROPOSED
    assert ev5.start_at == "Thursday"
    assert ev5.conversion_type == "property_walkthrough"

    # Turn 6: Client hedged response
    t6 = _create_turn_bundle(
        turn_id=6,
        speaker_id="client",
        text="Sure, Thursday could work.",
        agreement=0.60,
        trust=0.75,
    )
    s6 = manager.process_turn_bundle(t6)
    gate6 = s6.conversion_gate
    ev6 = s6.conversion_event

    # Gate must NOT register explicit commitment for hedged language
    assert gate6.explicit_commitment_detected is False
    assert gate6.commitment_slot is None

    # Event must be TENTATIVE, not CONFIRMED, and must extract "Thursday"
    assert ev6 is not None
    assert ev6.status == ConversionEventStatus.TENTATIVE
    assert ev6.start_at == "Thursday"
    assert ev6.supersedes_event_id == ev5.event_id

    # Facts: tentative recorded, confirmed not recorded
    active_confirmed = [f for f in s6.facts if f.fact_key == "confirmed_meeting_time" and f.status == "active"]
    assert len(active_confirmed) == 0
    active_tentative = [f for f in s6.facts if f.fact_key == "tentative_meeting_time" and f.status == "active"]
    assert len(active_tentative) == 1
    assert "Thursday" in active_tentative[0].fact_value

    # Turn 7: Absent decision maker regresses gate to closed
    t7 = _create_turn_bundle(
        turn_id=7,
        speaker_id="client",
        text="Actually wait, my wife would need to be part of this conversation before we go any further.",
    )
    s7 = manager.process_turn_bundle(t7)
    gate7 = s7.conversion_gate
    ev7 = s7.conversion_event

    assert gate7.is_open is False
    assert "decision_maker_aligned" in gate7.failed_conditions
    assert ev7 is not None
    assert ev7.status == ConversionEventStatus.BLOCKED
    assert ev7.start_at == "Thursday"
    assert ev7.event_id == ev6.event_id
    assert ev7.supersedes_event_id == ev5.event_id

