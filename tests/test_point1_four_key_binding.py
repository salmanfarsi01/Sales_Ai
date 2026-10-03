from __future__ import annotations

import json
from pathlib import Path
import pytest

from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStructure,
)
from copilot.core_intelligence_models import (
    StrategicDecision,
    StrategicAction,
)
from copilot.core_decision_manager import (
    CoreDecisionManager,
    StrategicDecisionCache,
)
from copilot.conversation_replay import (
    ConversationReplayEngine,
    TurnReplayStep,
)
from copilot.conversation_state_contract import BehavioralSignalInputBundle, DimensionScore, EmotionState
from copilot.conversation_materiality import MaterialityClassification


def test_point1_same_version_different_turns_within_same_call():
    """Client Requirement Point 1:
    Two different turns in the same call share the same source_state_version (e.g. V4).
    Every lookup and display must validate all four keys together:
    (call_sid, turn_id, source_state_version, source_event_id).
    Matching version alone is prohibited.
    """
    call_sid = "call_sim_point1_same_version_diff_turns"
    StrategicDecisionCache.clear(call_sid)

    # Turn 4 Decision: V4, Turn 4
    dec_turn4 = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=4,
        source_turn_id=4,
        source_event_id="ev_turn_4_v4",
        utterance_turn_id=4,
        metrics_source_turn_id=4,
        should_prompt=True,
        strategic_objective="Clarify move timing",
        primary_action=StrategicAction.CLARIFY,
        confidence=0.85,
    )

    # Turn 5 Decision: V4 (no state mutation occurred on turn 5), Turn 5
    dec_turn5 = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=4,
        source_turn_id=5,
        source_event_id="ev_turn_5_v4",
        utterance_turn_id=5,
        metrics_source_turn_id=5,
        should_prompt=True,
        strategic_objective="Reframe timing hesitation",
        primary_action=StrategicAction.REFRAME,
        confidence=0.85,
    )

    # Cache both decisions
    StrategicDecisionCache.put(call_sid=call_sid, decision=dec_turn4, turn_id=4, source_event_id="ev_turn_4_v4")
    StrategicDecisionCache.put(call_sid=call_sid, decision=dec_turn5, turn_id=5, source_event_id="ev_turn_5_v4")

    # 1. Strictly bound lookup for Turn 4 retrieves Turn 4's decision (CLARIFY), NEVER Turn 5's decision (REFRAME)
    bound_t4 = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=4,
        source_state_version=4,
        source_event_id="ev_turn_4_v4",
    )
    assert bound_t4 is not None
    assert bound_t4.source_turn_id == 4
    assert bound_t4.primary_action == StrategicAction.CLARIFY
    assert bound_t4.strategic_objective == "Clarify move timing"

    # 2. Strictly bound lookup for Turn 5 retrieves Turn 5's decision (REFRAME), NEVER Turn 4's decision (CLARIFY)
    bound_t5 = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=5,
        source_state_version=4,
        source_event_id="ev_turn_5_v4",
    )
    assert bound_t5 is not None
    assert bound_t5.source_turn_id == 5
    assert bound_t5.primary_action == StrategicAction.REFRAME
    assert bound_t5.strategic_objective == "Reframe timing hesitation"

    # 3. Lookup with mixed keys MUST return None (version matches, but turn_id and source_event_id mismatch)
    # Query Turn 4 with Turn 5's event ID
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=4,
        source_state_version=4,
        source_event_id="ev_turn_5_v4",
    ) is None

    # Query Turn 5 with Turn 4's event ID
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=5,
        source_state_version=4,
        source_event_id="ev_turn_4_v4",
    ) is None

    # 4. CoreDecisionManager strictly bound retrieval
    cdm = CoreDecisionManager(call_sid=call_sid)
    cdm_bound_4 = cdm.get_strictly_bound_decision(
        call_sid=call_sid,
        turn_id=4,
        source_state_version=4,
        source_event_id="ev_turn_4_v4",
    )
    assert cdm_bound_4 is not None
    assert cdm_bound_4.primary_action == StrategicAction.CLARIFY

    cdm_bound_5 = cdm.get_strictly_bound_decision(
        call_sid=call_sid,
        turn_id=5,
        source_state_version=4,
        source_event_id="ev_turn_5_v4",
    )
    assert cdm_bound_5 is not None
    assert cdm_bound_5.primary_action == StrategicAction.REFRAME

    # Cross-turn validation check
    valid_4_on_4, reason_4_4 = StrategicDecisionCache.validate_binding(
        dec_turn4, call_sid=call_sid, turn_id=4, source_state_version=4, source_event_id="ev_turn_4_v4"
    )
    assert valid_4_on_4 is True

    valid_4_on_5, reason_4_5 = StrategicDecisionCache.validate_binding(
        dec_turn4, call_sid=call_sid, turn_id=5, source_state_version=4, source_event_id="ev_turn_5_v4"
    )
    assert valid_4_on_5 is False
    assert "turn_id mismatch" in reason_4_5


def test_point1_same_version_different_calls():
    """Client Requirement Point 1:
    Two different calls share the same source_state_version (e.g. V2) and same turn_id (Turn 1).
    Validation of call_sid + turn_id + source_state_version + source_event_id prevents cross-call leakage.
    """
    call_a = "call_point1_alpha"
    call_b = "call_point1_beta"
    StrategicDecisionCache.clear(call_a)
    StrategicDecisionCache.clear(call_b)

    dec_a = StrategicDecision(
        call_id=call_a,
        call_sid=call_a,
        source_state_version=2,
        source_turn_id=1,
        source_event_id="ev_turn_1_v2",
        utterance_turn_id=1,
        metrics_source_turn_id=1,
        should_prompt=True,
        strategic_objective="Discovery probe",
        primary_action=StrategicAction.QUESTION,
        confidence=0.88,
    )

    dec_b = StrategicDecision(
        call_id=call_b,
        call_sid=call_b,
        source_state_version=2,
        source_turn_id=1,
        source_event_id="ev_turn_1_v2",
        utterance_turn_id=1,
        metrics_source_turn_id=1,
        should_prompt=True,
        strategic_objective="Acknowledge severe hesitation",
        primary_action=StrategicAction.DE_RISK,
        confidence=0.72,
    )

    StrategicDecisionCache.put(call_sid=call_a, decision=dec_a, turn_id=1, source_event_id="ev_turn_1_v2")
    StrategicDecisionCache.put(call_sid=call_b, decision=dec_b, turn_id=1, source_event_id="ev_turn_1_v2")

    # Call Alpha lookup retrieves dec_a
    bound_a = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_a,
        turn_id=1,
        source_state_version=2,
        source_event_id="ev_turn_1_v2",
    )
    assert bound_a is not None
    assert bound_a.call_sid == call_a
    assert bound_a.primary_action == StrategicAction.QUESTION

    # Call Beta lookup retrieves dec_b
    bound_b = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_b,
        turn_id=1,
        source_state_version=2,
        source_event_id="ev_turn_1_v2",
    )
    assert bound_b is not None
    assert bound_b.call_sid == call_b
    assert bound_b.primary_action == StrategicAction.DE_RISK

    # Cross-call lookup is strictly rejected
    valid_cross, reason_cross = StrategicDecisionCache.validate_binding(
        dec_a,
        call_sid=call_b,
        turn_id=1,
        source_state_version=2,
        source_event_id="ev_turn_1_v2",
    )
    assert valid_cross is False
    assert "call_sid mismatch" in reason_cross


def test_point1_same_call_turn_version_different_event_id():
    """Client Requirement Point 1:
    Even with identical call_sid, turn_id, and source_state_version, a mismatched source_event_id
    fails binding.
    """
    call_sid = "call_point1_event_mismatch"
    StrategicDecisionCache.clear(call_sid)

    dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=3,
        source_turn_id=2,
        source_event_id="ev_turn_2_v3_attempt1",
        utterance_turn_id=2,
        metrics_source_turn_id=2,
        should_prompt=True,
        strategic_objective="Educate on pricing",
        primary_action=StrategicAction.EDUCATE,
        confidence=0.80,
    )
    StrategicDecisionCache.put(call_sid=call_sid, decision=dec, turn_id=2, source_event_id="ev_turn_2_v3_attempt1")

    # Querying with attempt2 fails
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=2,
        source_state_version=3,
        source_event_id="ev_turn_2_v3_attempt2",
    ) is None

    # Validating against attempt2 fails
    valid, reason = StrategicDecisionCache.validate_binding(
        dec,
        call_sid=call_sid,
        turn_id=2,
        source_state_version=3,
        source_event_id="ev_turn_2_v3_attempt2",
    )
    assert valid is False
    assert "source_event_id mismatch" in reason


def test_point1_html_binding_validation_logic():
    """Verify that web/conversation_state_replay.html strictly validates all 4 keys."""
    html_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Confirm 4-key validation function exists in frontend
    assert "function validateDecisionBinding(d, st, sa, repCallSid)" in content
    assert "call_sid mismatch" in content
    assert "turn_id mismatch" in content
    assert "source_state_version mismatch" in content
    assert "source_event_id mismatch" in content
    assert "hasValidDecision = bindingCheck.valid" in content
    assert "4-Key Bound:" in content
    assert "Binding rejected:" in content


def test_point1_end_to_end_replay_four_key_bound_on_all_steps():
    """Point 1 E2E: Replay dialogue turns and assert every TurnReplayStep carries
    matching 4 keys on both step and strategic_decision.
    """
    turns = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, thanks for taking the call."},
        {"turn_id": 2, "speaker_id": "client", "text": "Sure, I have a few minutes."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Great, are you looking to buy or sell?"},
        {"turn_id": 4, "speaker_id": "client", "text": "We are looking to sell our house in Austin."},
        {"turn_id": 5, "speaker_id": "salesperson", "text": "Excellent. When are you looking to list?"},
        {"turn_id": 6, "speaker_id": "client", "text": "Next month would be ideal."},
    ]

    engine = ConversationReplayEngine()
    call_sid = "sim_test_point1_e2e_all_steps"
    rep = engine.replay_dialogue_turns(call_sid, turns, save_report=False)

    assert len(rep.timeline) == 6

    for step in rep.timeline:
        sd = step.strategic_decision
        assert sd is not None, f"Turn {step.turn_id} missing StrategicDecision"
        sa = step.state_after

        # 1. call_sid match
        assert (sd.call_sid or sd.call_id) == call_sid
        assert sa.call_sid == call_sid

        # 2. turn_id match
        assert sd.source_turn_id == step.turn_id

        # 3. source_state_version match
        assert sd.source_state_version == sa.state_version

        # 4. source_event_id match
        expected_event_id = f"ev_turn_{step.turn_id}_v{sa.state_version}"
        assert step.source_event_id == expected_event_id
        assert sd.source_event_id == expected_event_id

        # Cache lookup with all 4 keys must succeed
        cached = StrategicDecisionCache.get_strictly_bound(
            call_sid=call_sid,
            turn_id=step.turn_id,
            source_state_version=sa.state_version,
            source_event_id=expected_event_id,
        )
        assert cached is not None
        assert cached.decision_id == sd.decision_id

        # Validation function passes
        is_valid, reason = StrategicDecisionCache.validate_binding(
            sd,
            call_sid=call_sid,
            turn_id=step.turn_id,
            source_state_version=sa.state_version,
            source_event_id=expected_event_id,
        )
        assert is_valid is True, f"Turn {step.turn_id} failed binding: {reason}"


def test_point1_missing_or_null_source_event_id_strictly_fails():
    """Client Point 1 Gap Verification:
    Verify that when source_event_id is None, empty, or missing on either side,
    binding validation strictly fails and NEVER treats None === None as a match.
    """
    call_sid = "call_point1_null_event_test"

    # Case A: StrategicDecision has source_event_id = None
    dec_null_event = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=2,
        source_turn_id=1,
        source_event_id=None,
        utterance_turn_id=1,
        metrics_source_turn_id=1,
        should_prompt=True,
        strategic_objective="Probe",
        primary_action=StrategicAction.QUESTION,
        confidence=0.8,
    )

    # 1. Validating against a valid expected event fails because decision lacks source_event_id
    valid, reason = StrategicDecisionCache.validate_binding(
        dec_null_event,
        call_sid=call_sid,
        turn_id=1,
        source_state_version=2,
        source_event_id="ev_turn_1_v2",
    )
    assert valid is False
    assert "Missing source_event_id" in reason

    # 2. Validating when expected event is ALSO None/empty fails (None == None is strictly prohibited)
    valid_both_null, reason_both_null = StrategicDecisionCache.validate_binding(
        dec_null_event,
        call_sid=call_sid,
        turn_id=1,
        source_state_version=2,
        source_event_id="",
    )
    assert valid_both_null is False
    assert "Missing source_event_id" in reason_both_null

    # 3. get_strictly_bound returns None when source_event_id is None or empty
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=1,
        source_state_version=2,
        source_event_id="",
    ) is None


def test_point1_missing_or_null_other_keys_strictly_fails():
    """Verify that null/missing call_sid, turn_id, or source_state_version strictly fails."""
    call_sid = "call_point1_missing_keys"

    dec_valid = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=2,
        source_turn_id=1,
        source_event_id="ev_turn_1_v2",
        should_prompt=True,
        strategic_objective="Probe",
        primary_action=StrategicAction.QUESTION,
        confidence=0.8,
    )

    # Missing call_sid
    v_call, r_call = StrategicDecisionCache.validate_binding(
        dec_valid, call_sid="", turn_id=1, source_state_version=2, source_event_id="ev_turn_1_v2"
    )
    assert v_call is False
    assert "call_sid" in r_call

    # Missing turn_id
    v_turn, r_turn = StrategicDecisionCache.validate_binding(
        dec_valid, call_sid=call_sid, turn_id=None, source_state_version=2, source_event_id="ev_turn_1_v2"
    )
    assert v_turn is False
    assert "turn_id" in r_turn

    # Missing source_state_version
    v_ver, r_ver = StrategicDecisionCache.validate_binding(
        dec_valid, call_sid=call_sid, turn_id=1, source_state_version=None, source_event_id="ev_turn_1_v2"
    )
    assert v_ver is False
    assert "source_state_version" in r_ver


def test_point1_html_rejects_null_event_id_and_renders_pending():
    """Verify that the frontend JS logic in web/conversation_state_replay.html
    strictly rejects decisions where source_event_id is null on either side,
    preventing older un-bound reports from falsely passing.
    """
    html_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"
    content = html_path.read_text(encoding="utf-8")

    # Confirm mandatory non-null checks in validateDecisionBinding
    assert "Missing source_event_id on decision" in content
    assert "Missing source_event_id on replay step" in content
    assert "Missing call_sid on decision" in content
    assert "Missing turn_id on decision" in content
    assert "Missing source_state_version on decision" in content

