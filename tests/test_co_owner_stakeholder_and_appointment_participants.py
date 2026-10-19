import pytest
from copilot.conversation_replay import ConversationReplayEngine
from copilot.conversation_state_models import ConversionEventStatus
from copilot.core_decision_manager import StrategicDecisionCache


def test_co_owner_capture_turn2_and_participant_turn13():
    """Validates that:
    1. Turn 2: Client identifies brother Daniel as co-owner ('Yes, I own it with my brother Daniel. He lives abroad, so decisions are a bit complicated.').
       - Daniel is captured in decision_structure.stakeholders with role='co_owner', name='Daniel', presence='absent', is_decision_maker=True.
       - decision_structure.decision_maker_present is False and co_decision_required is True.
    2. Turn 8: Appointment booked ('Friday at 2').
       - conversion_event.participants is ['Client', 'Agent'].
    3. Turn 13: Client confirms Daniel will attend ('Daniel will join by video, he said he's fine to talk about it then.').
       - Daniel's presence in decision_structure.stakeholders flips to 'confirmed_attending'.
       - decision_structure.decision_maker_present flips to True.
       - conversion_event.participants updates to include 'Daniel' (['Client', 'Agent', 'Daniel']).
       - Downstream strategic decision passes 4-key binding and evidence consistency.
    """
    turns = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi Maria, thanks for taking my call. I'm following up about your house on Cedar Lane."},
        {"turn_id": 2, "speaker_id": "client", "text": "Yes, I own it with my brother Daniel. He lives abroad, so decisions are a bit complicated."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Understood. What's making you consider selling?"},
        {"turn_id": 4, "speaker_id": "client", "text": "We inherited it last year and honestly the property taxes are a burden."},
        {"turn_id": 5, "speaker_id": "client", "text": "But I'm worried your fee will eat most of what we make."},
        {"turn_id": 6, "speaker_id": "salesperson", "text": "Our fee is 4% and includes staging and photography. Let's look at your net proceeds."},
        {"turn_id": 7, "speaker_id": "client", "text": "That helps a little."},
        {"turn_id": 8, "speaker_id": "client", "text": "Fine, I can meet Friday at 2 to see the numbers."},
        {"turn_id": 9, "speaker_id": "salesperson", "text": "Great, Friday at 2 works. I'll bring recent comparable sales."},
        {"turn_id": 10, "speaker_id": "client", "text": "okay"},
        {"turn_id": 11, "speaker_id": "client", "text": "Actually, can we move it to Saturday at 11 instead?"},
        {"turn_id": 12, "speaker_id": "client", "text": "Message me on WhatsApp only, and no calls before noon."},
        {"turn_id": 13, "speaker_id": "client", "text": "Daniel will join by video, he said he's fine to talk about it then."},
        {"turn_id": 14, "speaker_id": "client", "text": "Actually, please stop contacting me altogether, I've changed my mind."},
    ]

    engine = ConversationReplayEngine()
    report = engine.replay_dialogue_turns(call_sid="sim_test_co_owner_daniel", raw_turns=turns, save_report=False)

    step2 = next(s for s in report.timeline if s.turn_id == 2)
    step8 = next(s for s in report.timeline if s.turn_id == 8)
    step11 = next(s for s in report.timeline if s.turn_id == 11)
    step13 = next(s for s in report.timeline if s.turn_id == 13)
    step14 = next(s for s in report.timeline if s.turn_id == 14)

    # Turn 2 Checks: Daniel captured as absent co-owner
    ds2 = step2.state_after.decision_structure
    assert ds2.decision_maker_present is False
    assert ds2.co_decision_required is True
    assert len(ds2.stakeholders) >= 1
    daniel2 = next((s for s in ds2.stakeholders if s.name == "Daniel"), None)
    assert daniel2 is not None, "Daniel must be captured in decision_structure.stakeholders at Turn 2"
    assert daniel2.role == "co_owner"
    assert daniel2.presence == "absent"
    assert daniel2.is_decision_maker is True

    # Turn 8 Checks: Confirmed meeting with Client and Agent
    ce8 = step8.state_after.conversion_event
    assert ce8 is not None
    assert ce8.status == ConversionEventStatus.CONFIRMED
    assert ce8.start_at == "Friday at 2"
    assert "Client" in ce8.participants
    assert "Agent" in ce8.participants

    # Turn 11 Checks: Rescheduled to Saturday at 11
    ce11 = step11.state_after.conversion_event
    assert ce11 is not None
    assert ce11.status == ConversionEventStatus.CONFIRMED
    assert ce11.start_at == "Saturday at 11"

    # Turn 13 Checks: Daniel confirms attendance; participants include Daniel
    ds13 = step13.state_after.decision_structure
    assert ds13.decision_maker_present is True
    assert ds13.co_decision_required is True
    daniel13 = next((s for s in ds13.stakeholders if s.name == "Daniel"), None)
    assert daniel13 is not None
    assert daniel13.presence == "confirmed_attending"

    ce13 = step13.state_after.conversion_event
    assert ce13 is not None
    assert ce13.status == ConversionEventStatus.CONFIRMED
    assert "Daniel" in ce13.participants
    assert "Client" in ce13.participants
    assert "Agent" in ce13.participants
    assert ce13.participants == ["Client", "Agent", "Daniel"]

    # Check state change record exists on conversion_event for Turn 13
    conv_changes = [c for c in step13.state_changes if c.field_path == "conversion_event"]
    assert len(conv_changes) >= 1, "State change record must be recorded when conversion_event participants change"
    assert conv_changes[0].new_value["participants"] == ["Client", "Agent", "Daniel"]

    # Turn 14 Checks: Hard boundary preserves confirmed appointment with all participants
    ce14 = step14.state_after.conversion_event
    assert ce14 is not None
    assert ce14.status == ConversionEventStatus.CONFIRMED
    assert "Daniel" in ce14.participants

    # Validate binding and evidence consistency for all steps
    for step in report.timeline:
        if step.strategic_decision:
            ok, reason = StrategicDecisionCache.validate_evidence_consistency(step.strategic_decision, step.state_after, require_active=True)
            assert ok, f"Evidence consistency failure at Turn {step.turn_id}: {reason}"
