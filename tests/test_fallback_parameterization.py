import pytest
from unittest.mock import MagicMock
from copilot.core_intelligence_models import (
    StrategicDecision,
    StrategicAction,
)
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    MeetingConversionGate,
    ConversionEventObject,
    ConversionEventStatus,
    PersistentFactRecord,
)
from copilot.llm_response_gateway import LLMResponseGateway


def test_deterministic_fallback_parameterizes_slot_from_decision():
    """Confirms fallback prompt dynamically uses decision.commitment_slot instead of hardcoded Thursday."""
    gateway = LLMResponseGateway()

    # Test 1: Friday at 11am
    dec_friday = StrategicDecision(
        call_id="call_test_1",
        source_state_version=10,
        primary_action=StrategicAction.ACKNOWLEDGE,
        reason_codes=["CONVERSION_CONFIRMED"],
        strategic_objective="Protect confirmed appointment",
        commitment_slot="Friday at 11am",
    )
    prompt_friday = gateway._deterministic_fallback(dec_friday)
    assert "Friday at 11am" in prompt_friday
    assert "Thursday" not in prompt_friday
    assert prompt_friday == "Perfect, I've noted Friday at 11am for us. I will see you both then."

    # Test 2: Monday at 2pm
    dec_monday = StrategicDecision(
        call_id="call_test_2",
        source_state_version=10,
        primary_action=StrategicAction.ACKNOWLEDGE,
        reason_codes=["CONVERSION_CONFIRMED"],
        strategic_objective="Protect confirmed appointment",
        commitment_slot="Monday at 2pm",
    )
    prompt_monday = gateway._deterministic_fallback(dec_monday)
    assert "Monday at 2pm" in prompt_monday
    assert "Thursday" not in prompt_monday
    assert prompt_monday == "Perfect, I've noted Monday at 2pm for us. I will see you both then."


def test_deterministic_fallback_parameterizes_slot_from_snapshot_sources():
    """Confirms fallback pulls from gate, conversion event, or fact if decision.commitment_slot is unset."""
    gateway = LLMResponseGateway()

    # Source A: Gate commitment slot
    dec = StrategicDecision(
        call_id="call_gate",
        source_state_version=10,
        primary_action=StrategicAction.ACKNOWLEDGE,
        reason_codes=["CONVERSION_CONFIRMED"],
        strategic_objective="Protect confirmed appointment",
    )
    snap_gate = ConversationStateSnapshot(
        call_sid="call_gate",
        state_version=10,
        conversion_gate=MeetingConversionGate(
            is_open=True,
            status="open",
            commitment_slot="Wednesday at 10am",
        ),
    )
    prompt_gate = gateway._deterministic_fallback(dec, snapshot=snap_gate)
    assert prompt_gate == "Perfect, I've noted Wednesday at 10am for us. I will see you both then."

    # Source B: Active conversion event start_at
    snap_event = ConversationStateSnapshot(
        call_sid="call_event",
        state_version=10,
        conversion_events=[
            ConversionEventObject(
                conversion_type="property_walkthrough",
                status=ConversionEventStatus.CONFIRMED,
                start_at="Tuesday at 4pm",
            )
        ],
    )
    prompt_event = gateway._deterministic_fallback(dec, snapshot=snap_event)
    assert prompt_event == "Perfect, I've noted Tuesday at 4pm for us. I will see you both then."

    # Source C: Confirmed meeting time fact
    snap_fact = ConversationStateSnapshot(
        call_sid="call_fact",
        state_version=10,
        facts=[
            PersistentFactRecord(
                fact_key="confirmed_meeting_time",
                fact_value="Saturday morning",
                category="logistical",
                timestamp_ms=1000,
                source_turn_id=5,
            )
        ],
    )
    prompt_fact = gateway._deterministic_fallback(dec, snapshot=snap_fact)
    assert prompt_fact == "Perfect, I've noted Saturday morning for us. I will see you both then."

    # Source D: No slot at all -> graceful time-neutral fallback (NEVER hardcodes Thursday)
    snap_empty = ConversationStateSnapshot(call_sid="call_empty", state_version=1)
    prompt_empty = gateway._deterministic_fallback(dec, snapshot=snap_empty)
    assert "Thursday" not in prompt_empty
    assert prompt_empty == "Perfect, I have that noted down for us. I will see you both then."


def test_commitment_close_two_window_choice_is_not_hardcoded():
    """Confirms COMMITMENT_CLOSE with two_window_choice does not hardcode Thursday at two or Friday morning."""
    gateway = LLMResponseGateway()

    dec = StrategicDecision(
        call_id="call_close",
        source_state_version=5,
        primary_action=StrategicAction.COMMITMENT_CLOSE,
        secondary_action=StrategicAction.QUESTION,
        push_strength="two_window_choice",
        strategic_objective="Offer two-window choice",
    )
    prompt = gateway._deterministic_fallback(dec)
    assert "Thursday" not in prompt
    assert "Friday" not in prompt
    assert prompt == "Would mornings or afternoons generally work better for a brief walkthrough?"


def test_production_error_and_timeout_circuit_breaker():
    """Confirms production exceptions (timeout, rate limit, 500 error) gracefully trigger fallback with slot."""
    mock_llm_client = MagicMock()
    mock_llm_client.chat.completions.create.side_effect = TimeoutError("Upstream LLM timed out after 3000ms")

    gateway = LLMResponseGateway(llm_client=mock_llm_client)

    dec = StrategicDecision(
        call_id="call_prod_err",
        source_state_version=15,
        primary_action=StrategicAction.ACKNOWLEDGE,
        reason_codes=["CONVERSION_CONFIRMED"],
        strategic_objective="Protect confirmed appointment",
        commitment_slot="Friday at 11am",
    )
    snapshot = ConversationStateSnapshot(call_sid="call_prod_err", state_version=15)

    prompt = gateway.generate_prompt(decision=dec, snapshot=snapshot, facts=[])
    # Verifies graceful catch and dynamic slot injection even during LLM timeout
    assert prompt.text == "Perfect, I've noted Friday at 11am for us. I will see you both then."
    assert "Thursday" not in prompt.text

    # Rate limit test (HTTP 429)
    mock_llm_client.chat.completions.create.side_effect = RuntimeError("HTTP 429 Too Many Requests: Rate limit exceeded")
    prompt_ratelimit = gateway.generate_prompt(decision=dec, snapshot=snapshot, facts=[])
    assert prompt_ratelimit.text == "Perfect, I've noted Friday at 11am for us. I will see you both then."


def test_early_turn_outage_does_not_fire_conversion_confirmed_fallback():
    """Validates that a production API failure on an early, unconfirmed turn (e.g. Turn 4)
    never triggers ACKNOWLEDGE / CONVERSION_CONFIRMED fallback copy or stale appointment claims.
    """
    mock_llm_client = MagicMock()
    # Simulate a network/API failure on Turn 4
    mock_llm_client.chat.completions.create.side_effect = TimeoutError("Upstream LLM 504 Gateway Timeout")

    gateway = LLMResponseGateway(llm_client=mock_llm_client)

    # Turn 4 state: Objection raised, gate CLOSED, commitment_slot NONE
    turn_4_snapshot = ConversationStateSnapshot(
        call_sid="call_benchmark_early_turn",
        state_version=4,
        conversion_gate=MeetingConversionGate(
            is_open=False,
            status="closed",
            commitment_slot=None,
            explicit_commitment_detected=False,
        ),
    )

    # Core Intelligence decision at Turn 4: VALIDATE objection, zero conversion reason codes
    dec_turn_4 = StrategicDecision(
        call_id="call_benchmark_early_turn",
        call_sid="call_benchmark_early_turn",
        source_state_version=4,
        source_turn_id=4,
        utterance_turn_id=4,
        metrics_source_turn_id=4,
        primary_action=StrategicAction.VALIDATE,
        secondary_action=StrategicAction.CLARIFY,
        push_strength="resolve_then_ask",
        reason_codes=["OBJECTION_VALIDATE_PRIORITY"],
        strategic_objective="Validate prospect concern without premature close",
        meeting_gate_open=False,
        conversion_confirmed=False,
        commitment_slot=None,
    )

    prompt = gateway.generate_prompt(decision=dec_turn_4, snapshot=turn_4_snapshot, facts=[])

    # 1. Assert zero conversion leakage
    assert "confirmed on my calendar" not in prompt.text.lower()
    assert "see you both then" not in prompt.text.lower()
    assert "thursday" not in prompt.text.lower()
    assert "confirmed" not in prompt.text.lower()
    assert "appointment" not in prompt.text.lower()

    # 2. Assert the prompt strictly delivers the objection handling validation fallback
    assert prompt.text == "That makes complete sense—let's focus directly on what matters most for your specific situation."
    assert prompt.strategic_action == StrategicAction.VALIDATE


def test_deterministic_fallback_stage_and_fact_branching():
    """Confirms deterministic fallback branches correctly based on stage, reason codes, and facts without turn hardcoding."""
    gateway = LLMResponseGateway()

    # 1. Discovery without timeline fact
    dec_disc = StrategicDecision(
        call_id="call_test",
        source_state_version=2,
        primary_action=StrategicAction.QUESTION,
        reason_codes=["STAGE_DISCOVERY", "EXPLORE_PROSPECT_NEEDS"],
        strategic_objective="Discovery",
    )
    snap_no_tl = ConversationStateSnapshot(call_sid="call_test", state_version=2, facts=[])
    res1 = gateway._deterministic_fallback(dec_disc, snapshot=snap_no_tl)
    assert res1 == "What would be the most important priority for you when evaluating your options?"

    # 2. Discovery with timeline fact
    snap_with_tl = ConversationStateSnapshot(
        call_sid="call_test",
        state_version=4,
        facts=[
            PersistentFactRecord(
                fact_key="timeline_horizon",
                fact_value="sometime next year",
                category="logistical",
                timestamp_ms=1000,
                source_turn_id=4,
            )
        ],
    )
    res2 = gateway._deterministic_fallback(dec_disc, snapshot=snap_with_tl)
    assert res2 == "Understood, what is driving your timeline for making a move?"

    # 3. Walkthrough transition
    dec_trans = StrategicDecision(
        call_id="call_test",
        source_state_version=7,
        primary_action=StrategicAction.QUESTION,
        reason_codes=["STAGE_DEFAULT_ENGAGEMENT"],
        strategic_objective="Walkthrough transition",
    )
    res3 = gateway._deterministic_fallback(dec_trans, snapshot=snap_no_tl)
    assert res3 == "Would sometime next week work for a quick 15-minute walkthrough?"

    # 4. Scheduling without constraints
    dec_sched = StrategicDecision(
        call_id="call_test",
        source_state_version=9,
        primary_action=StrategicAction.QUESTION,
        reason_codes=["STAGE_SCHEDULING"],
        strategic_objective="Scheduling",
    )
    res4 = gateway._deterministic_fallback(dec_sched, snapshot=snap_no_tl)
    assert res4 == "Would Tuesday or Thursday afternoon work better for your schedule?"

    # 5. Scheduling with morning constraint
    snap_with_constraint = ConversationStateSnapshot(
        call_sid="call_test",
        state_version=16,
        facts=[
            PersistentFactRecord(
                fact_key="scheduling_constraint",
                fact_value="mornings don't work",
                category="logistical",
                timestamp_ms=1000,
                source_turn_id=16,
            )
        ],
    )
    res5 = gateway._deterministic_fallback(dec_sched, snapshot=snap_with_constraint)
    assert res5 == "What days or times usually work best for your schedule when reviewing options?"

    # 6. Contact preference commitment
    dec_contact = StrategicDecision(
        call_id="call_test",
        source_state_version=15,
        primary_action=StrategicAction.ACKNOWLEDGE,
        reason_codes=["CONTACT_PREFERENCE_DECLARED"],
        strategic_objective="Protect preference",
    )
    res6 = gateway._deterministic_fallback(dec_contact, snapshot=snap_no_tl)
    assert res6 == "I completely understand, no problem at all. We will respect your preferences."


