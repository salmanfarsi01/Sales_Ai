"""Tests for Issue #6: Conversion Header & Disposition Presentation.

Guarantees:
1. Turn 18 Header Regression:
   On Turn 18 of the reference trace, deal_milestone_status evaluates to 'appointment_confirmed',
   and renders 'APPOINTMENT CONFIRMED — Thursday At 3', strictly superseding 'OPEN (Ready to Close)'.
2. Open Concerns Dossier Line:
   Captures active and partially_resolved concerns (e.g. general_hesitation),
   strictly excluding dormant and resolved objections.
3. Priority Ordering:
   conversion_event.status == 'confirmed' always takes precedence over conversion_gate.is_open,
   preserving 'appointment_confirmed' even if the gate is closed post-booking.
"""

import json
import pytest

from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversionEventObject,
    ConversionEventStatus,
    MeetingConversionGate,
    ObjectionLifecycleState,
    ObjectionRecord,
)
from copilot.conversation_presentation import (
    compute_deal_milestone_status,
    format_milestone_label,
    compute_open_concerns,
    format_open_concerns_summary,
    build_conversion_presentation,
)
from copilot.conversation_replay import ConversationReplayEngine


def test_turn_18_header_shows_appointment_confirmed_not_ready_to_close():
    """Client's exact named regression:
    Turn 18 of reference trace must render 'APPOINTMENT CONFIRMED — Thursday At 3',
    NOT 'OPEN (Ready to Close)'.
    """
    with open("reports/synthetic/conversation_state_sim_mucj0p5s.json", encoding="utf-8") as f:
        data = json.load(f)

    mgr = ConversationStateManager(call_sid="sim_mucj0p5s_header_test")
    for t in data["timeline"]:
        eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
        mgr.process_turn_bundle(eb)

    final_state = mgr.current_state
    pres = build_conversion_presentation(final_state)

    # 1. Milestone status must be appointment_confirmed
    assert pres["deal_milestone_status"] == "appointment_confirmed"

    # 2. Header label must show concrete appointment time
    assert pres["milestone_label"] == "APPOINTMENT CONFIRMED — Thursday At 3"
    assert "Ready to Close" not in pres["milestone_label"]

    # 3. Open concerns dossier must capture the partially resolved financial hesitation
    assert len(pres["open_concerns"]) == 1
    c = pres["open_concerns"][0]
    assert c["category"] == "general_hesitation"
    assert c["lifecycle_state"] == "partially_resolved"

    # 4. Open concerns summary line must format as specified
    assert pres["open_concerns_summary"] == "Open Concerns: 1 (financial net proceeds / general hesitation — partially resolved)"


def test_open_concerns_includes_partially_resolved_excludes_dormant():
    """Verify compute_open_concerns includes partially_resolved and active,
    while strictly excluding dormant and resolved objections.
    """
    snap = ConversationStateSnapshot(
        call_sid="CA_concerns_filter_test",
        objections=[
            ObjectionRecord(
                objection_id="obj_dormant",
                canonical_category="commission_fee",
                initial_statement="6 percent is too high",
                latest_statement="6 percent is too high",
                first_turn_id=5,
                last_updated_turn_id=5,
                lifecycle_state=ObjectionLifecycleState.DORMANT,
            ),
            ObjectionRecord(
                objection_id="obj_resolved",
                canonical_category="timing",
                initial_statement="I need time to think",
                latest_statement="I need time to think",
                first_turn_id=7,
                last_updated_turn_id=7,
                lifecycle_state=ObjectionLifecycleState.RESOLVED,
            ),
            ObjectionRecord(
                objection_id="obj_partial",
                canonical_category="general_hesitation",
                initial_statement="Worried about net proceeds",
                latest_statement="Worried about net proceeds",
                first_turn_id=12,
                last_updated_turn_id=14,
                lifecycle_state=ObjectionLifecycleState.PARTIALLY_RESOLVED,
            ),
        ],
    )

    concerns = compute_open_concerns(snap)
    assert len(concerns) == 1
    assert concerns[0]["category"] == "general_hesitation"
    assert concerns[0]["lifecycle_state"] == "partially_resolved"

    summary = format_open_concerns_summary(concerns)
    assert summary == "Open Concerns: 1 (financial net proceeds / general hesitation — partially resolved)"


def test_milestone_status_priority_conversion_event_over_gate_status():
    """Priority order test:
    When conversion_event is CONFIRMED, deal_milestone_status MUST evaluate to
    'appointment_confirmed', even if conversion_gate.is_open is False.
    """
    snap = ConversationStateSnapshot(
        call_sid="CA_priority_test",
        conversion_event=ConversionEventObject(
            event_id="conv_confirmed_test",
            conversion_type="in_person_meeting",
            status=ConversionEventStatus.CONFIRMED,
            start_at="Friday At 2",
        ),
        conversion_gate=MeetingConversionGate(
            is_open=False,
            status="closed",
            failed_conditions=["boundary_active"],
            blocking_reasons=["Active boundary detected"],
        ),
    )

    milestone = compute_deal_milestone_status(snap)
    assert milestone == "appointment_confirmed"

    label = format_milestone_label(milestone, snap.conversion_event)
    assert label == "APPOINTMENT CONFIRMED — Friday At 2"


def test_milestone_status_tentative_and_gate_tiers():
    """Verify intermediate milestone status tiers:
    - Tentative appointment -> 'appointment_tentative'
    - Gate open without meeting -> 'ready_to_close'
    - Gate closed without meeting -> 'blocked'
    """
    # 1. Tentative
    snap_tentative = ConversationStateSnapshot(
        call_sid="CA_tentative_test",
        conversion_event=ConversionEventObject(
            event_id="conv_tentative_test",
            conversion_type="in_person_meeting",
            status=ConversionEventStatus.TENTATIVE,
            start_at="Next Week",
        ),
        conversion_gate=MeetingConversionGate(is_open=True, status="open"),
    )
    assert compute_deal_milestone_status(snap_tentative) == "appointment_tentative"
    assert format_milestone_label("appointment_tentative", snap_tentative.conversion_event) == "TENTATIVE — Next Week"

    # 2. Ready to Close
    snap_gate_open = ConversationStateSnapshot(
        call_sid="CA_gate_open_test",
        conversion_gate=MeetingConversionGate(is_open=True, status="open"),
    )
    assert compute_deal_milestone_status(snap_gate_open) == "ready_to_close"
    assert format_milestone_label("ready_to_close") == "OPEN (Ready to Close)"

    # 3. Blocked
    snap_blocked = ConversationStateSnapshot(
        call_sid="CA_blocked_test",
        conversion_gate=MeetingConversionGate(is_open=False, status="closed"),
    )
    assert compute_deal_milestone_status(snap_blocked) == "blocked"
    assert format_milestone_label("blocked") == "BLOCKED"


def test_conversion_replay_report_serializes_presentation_fields():
    """Verify that ConversationReplayEngine generates reports containing the enriched presentation fields."""
    engine = ConversationReplayEngine()
    report = engine.replay_dialogue_turns(
        call_sid="sim_header_unit_test",
        raw_turns=[
            {"turn_id": 1, "speaker_id": "salesperson", "text": "Hello"},
            {"turn_id": 2, "speaker_id": "client", "text": "I'm the only one deciding"},
        ],
        save_report=False,
    )

    conv = report.conversion_summary
    assert "deal_milestone_status" in conv
    assert "milestone_label" in conv
    assert "open_concerns" in conv
    assert "open_concerns_summary" in conv
