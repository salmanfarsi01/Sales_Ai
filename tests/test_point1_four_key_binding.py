from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import pytest

from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStructure,
    MeetingConversionGate,
    ReadinessBreakdown,
    ObjectionRecord,
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
        state_call_sid=call_sid,
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
        state_call_sid=call_sid,
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
        state_call_sid=call_sid,
    ) is None

    # Query Turn 5 with Turn 4's event ID
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=5,
        source_state_version=4,
        source_event_id="ev_turn_4_v4",
        state_call_sid=call_sid,
    ) is None

    # 4. CoreDecisionManager strictly bound retrieval
    cdm = CoreDecisionManager(call_sid=call_sid)
    cdm_bound_4 = cdm.get_strictly_bound_decision(
        call_sid=call_sid,
        turn_id=4,
        source_state_version=4,
        source_event_id="ev_turn_4_v4",
        state_call_sid=call_sid,
    )
    assert cdm_bound_4 is not None
    assert cdm_bound_4.primary_action == StrategicAction.CLARIFY

    cdm_bound_5 = cdm.get_strictly_bound_decision(
        call_sid=call_sid,
        turn_id=5,
        source_state_version=4,
        source_event_id="ev_turn_5_v4",
        state_call_sid=call_sid,
    )
    assert cdm_bound_5 is not None
    assert cdm_bound_5.primary_action == StrategicAction.REFRAME

    # Cross-turn validation check
    valid_4_on_4, reason_4_4 = StrategicDecisionCache.validate_binding(
        dec_turn4,
        call_sid=call_sid,
        turn_id=4,
        source_state_version=4,
        source_event_id="ev_turn_4_v4",
        state_call_sid=call_sid,
    )
    assert valid_4_on_4 is True

    valid_4_on_5, reason_4_5 = StrategicDecisionCache.validate_binding(
        dec_turn4,
        call_sid=call_sid,
        turn_id=5,
        source_state_version=4,
        source_event_id="ev_turn_5_v4",
        state_call_sid=call_sid,
    )
    assert valid_4_on_5 is False
    assert "turn_id mismatch" in reason_4_5


def test_point1_same_version_different_calls():
    """Client Requirement Point 1:
    Even with identical source_state_version, turn_id, and event ID, different call_sid values
    are completely isolated.
    """
    call_a = "call_sim_point1_call_alpha"
    call_b = "call_sim_point1_call_beta"
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
        strategic_objective="Assess needs for call A",
        primary_action=StrategicAction.QUESTION,
        confidence=0.80,
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
        strategic_objective="De-risk for call B",
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
        state_call_sid=call_a,
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
        state_call_sid=call_b,
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
        state_call_sid=call_b,
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
        state_call_sid=call_sid,
    ) is None

    # Validating against attempt2 fails
    valid, reason = StrategicDecisionCache.validate_binding(
        dec,
        call_sid=call_sid,
        turn_id=2,
        source_state_version=3,
        source_event_id="ev_turn_2_v3_attempt2",
        state_call_sid=call_sid,
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


def test_point1_frontend_cross_call_source_state_rejection_node():
    """Gap 1: Test that executes the exact JavaScript validateDecisionBinding function
    from web/conversation_state_replay.html via Node.js.
    Tests the exact case described by the client:
    - Report call_sid ('call_sim_123') and decision call_sid ('call_sim_123') match.
    - Source state (state_after) belongs to another call ('call_other_foreign_999').
    Asserts rejection and visible error message rendering.
    """
    html_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"
    content = html_path.read_text(encoding="utf-8")

    # Extract the exact validateDecisionBinding JS function definition
    match = re.search(r"(function validateDecisionBinding\(d, st, sa, repCallSid\)\s*\{[\s\S]*?\n      \})", content)
    assert match is not None, "Could not find validateDecisionBinding in HTML file"
    func_js = match.group(1)

    node_script = f"""
{func_js}

const reportCallSid = 'call_sim_123';
const decision = {{
    call_sid: 'call_sim_123',
    call_id: 'call_sim_123',
    source_turn_id: 4,
    source_state_version: 10,
    source_event_id: 'ev_turn_4_v10'
}};
const step = {{
    turn_id: 4,
    source_event_id: 'ev_turn_4_v10'
}};

// Case 1: Client exact case - source state belongs to another call
const foreignState = {{
    call_sid: 'call_other_foreign_999',
    state_version: 10
}};
const resCross = validateDecisionBinding(decision, step, foreignState, reportCallSid);

// Case 2: Matching state (valid)
const matchingState = {{
    call_sid: 'call_sim_123',
    state_version: 10
}};
const resValid = validateDecisionBinding(decision, step, matchingState, reportCallSid);

// Case 3: Version mismatch (V8 vs V10)
const staleDecision = {{
    ...decision,
    source_state_version: 8
}};
const resVersionMismatch = validateDecisionBinding(staleDecision, step, matchingState, reportCallSid);

// Case 4: Simulate UI card rendering on rejection
function renderCard(bindingCheck, step, sAfter, sd) {{
    const hasValidDecision = bindingCheck.valid;
    if (!hasValidDecision) {{
        return `<div class="card-unit warning">No valid Strategic Decision bound for this turn (Turn ${{step.turn_id}}, V${{sAfter ? (sAfter.state_version || 0) : 0}}). Binding rejected: ${{bindingCheck.reason || ''}}</div>`;
    }}
    return `<div class="card-unit success">Bound successfully</div>`;
}}
const uiCardHtml = renderCard(resCross, step, foreignState, decision);

// Case 5: Missing or undefined source state (sa is null or sa.call_sid undefined)
const resMissingState = validateDecisionBinding(decision, step, null, reportCallSid);
const undefinedCallSidState = {{ state_version: 10 }};
const resUndefinedCallSid = validateDecisionBinding(decision, step, undefinedCallSidState, reportCallSid);

console.log(JSON.stringify({{
    cross: resCross,
    valid: resValid,
    versionMismatch: resVersionMismatch,
    uiCard: uiCardHtml,
    missingState: resMissingState,
    undefinedCallSid: resUndefinedCallSid
}}));
"""

    proc = subprocess.run(
        ["node"],
        input=node_script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    out = json.loads(proc.stdout)

    # 1. Cross-call source state must be rejected with explicit error
    assert out["cross"]["valid"] is False
    assert "Cross-call source state ownership mismatch" in out["cross"]["reason"]
    assert "call_other_foreign_999" in out["cross"]["reason"]

    # 2. UI card must display visible rejection message
    assert "Binding rejected: Cross-call source state ownership mismatch" in out["uiCard"]
    assert "call_other_foreign_999" in out["uiCard"]

    # 3. Matching source state must be valid
    assert out["valid"]["valid"] is True

    # 4. Version mismatch must be rejected
    assert out["versionMismatch"]["valid"] is False
    assert "source_state_version mismatch" in out["versionMismatch"]["reason"]

    # 5. Missing source state or undefined call_sid must be rejected (Case 5)
    assert out["missingState"]["valid"] is False
    assert "Missing call_sid on source state" in out["missingState"]["reason"]
    assert out["undefinedCallSid"]["valid"] is False
    assert "Missing call_sid on source state" in out["undefinedCallSid"]["reason"]


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
            state_call_sid=call_sid,
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
            state_call_sid=call_sid,
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
        state_call_sid=call_sid,
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
        state_call_sid=call_sid,
    )
    assert valid_both_null is False
    assert "Missing source_event_id" in reason_both_null

    # 3. get_strictly_bound returns None when source_event_id is None or empty
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=1,
        source_state_version=2,
        source_event_id="",
        state_call_sid=call_sid,
    ) is None


def test_point1_missing_or_null_other_keys_strictly_fails():
    """Verify that null/missing call_sid, turn_id, source_state_version, or state_call_sid strictly fails."""
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
        dec_valid, call_sid="", turn_id=1, source_state_version=2, source_event_id="ev_turn_1_v2", state_call_sid=call_sid
    )
    assert v_call is False
    assert "call_sid" in r_call

    # Missing turn_id
    v_turn, r_turn = StrategicDecisionCache.validate_binding(
        dec_valid, call_sid=call_sid, turn_id=None, source_state_version=2, source_event_id="ev_turn_1_v2", state_call_sid=call_sid
    )
    assert v_turn is False
    assert "turn_id" in r_turn

    # Missing source_state_version
    v_ver, r_ver = StrategicDecisionCache.validate_binding(
        dec_valid, call_sid=call_sid, turn_id=1, source_state_version=None, source_event_id="ev_turn_1_v2", state_call_sid=call_sid
    )
    assert v_ver is False
    assert "source_state_version" in r_ver

    # Missing state_call_sid (Gap 2: fails closed, never fails open)
    v_state, r_state = StrategicDecisionCache.validate_binding(
        dec_valid, call_sid=call_sid, turn_id=1, source_state_version=2, source_event_id="ev_turn_1_v2", state_call_sid=""
    )
    assert v_state is False
    assert "Missing state_call_sid" in r_state


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
    assert "Cross-call source state ownership mismatch" in content


def test_point1_cross_call_source_state_ownership_rejection_in_cache():
    """Client Requirement Point 1:
    Validate call ownership across report, source state, and decision.
    Reject when source state belongs to another call even if report and decision match.
    """
    call_sid = "call_sim_point1_ownership_test"
    other_call = "call_sim_point1_different_call"
    StrategicDecisionCache.clear(call_sid)

    dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=10,
        source_turn_id=4,
        source_event_id="ev_turn_4_v10",
        utterance_turn_id=4,
        metrics_source_turn_id=4,
        should_prompt=False,
        strategic_objective="Continuation from Turn 3",
        primary_action=StrategicAction.ACKNOWLEDGE,
        confidence=0.8,
    )
    StrategicDecisionCache.put(call_sid=call_sid, decision=dec, turn_id=4, source_event_id="ev_turn_4_v10")

    # 1. Matching state_call_sid passes
    valid_same, reason_same = StrategicDecisionCache.validate_binding(
        dec,
        call_sid=call_sid,
        turn_id=4,
        source_state_version=10,
        source_event_id="ev_turn_4_v10",
        state_call_sid=call_sid,
    )
    assert valid_same is True
    assert "Valid 4-key binding" in reason_same

    # 2. Cross-call source state fails
    valid_diff, reason_diff = StrategicDecisionCache.validate_binding(
        dec,
        call_sid=call_sid,
        turn_id=4,
        source_state_version=10,
        source_event_id="ev_turn_4_v10",
        state_call_sid=other_call,
    )
    assert valid_diff is False
    assert "Cross-call source state ownership mismatch" in reason_diff
    assert other_call in reason_diff

    # 3. get_strictly_bound with cross-call state_call_sid returns None
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=4,
        source_state_version=10,
        source_event_id="ev_turn_4_v10",
        state_call_sid=other_call,
    ) is None


def test_point1_snapshot_consistency_turn4_carry_forward():
    """Client Requirement Point 1 & Gap 3 (sim_muy04bh0 Turn 4 verification):
    At Turn 4, the rep turn is a continuation of Turn 3's strategy.
    The decision bound to Turn 4's snapshot V10 MUST:
    - Have primary_action == ACKNOWLEDGE, strategic_posture == PROTECT, push_strength == NONE
    - Have dynamically recomputed confidence matching the new snapshot (0.845, NOT Turn 3's 0.925)
    - Report reason_codes: ['CONVERSION_CONFIRMED', 'CONFIRM_AND_PROTECT_ACTIVE', 'STRATEGY_CARRIED_FORWARD']
    - Report meeting_gate_open == False (matching V10 conversion_gate closed state)
    - In evidence_considered: report Gate: CLOSED (NEVER Turn 3's Gate: OPEN)
    - Preserve historical strategy provenance separately: carried_forward_from_turn_id == 3,
      STRATEGY_CARRIED_FORWARD in reason_codes, and 'Strategy provenance: carried forward from Turn 3'.
    """
    rep_file = Path(__file__).resolve().parent.parent / "reports" / "synthetic" / "conversation_state_sim_muy04bh0.json"
    assert rep_file.exists(), f"Missing synthetic replay file: {rep_file}"
    rep_data = json.loads(rep_file.read_text(encoding="utf-8"))

    step3 = [s for s in rep_data["timeline"] if s["turn_id"] == 3][0]
    step4 = [s for s in rep_data["timeline"] if s["turn_id"] == 4][0]

    snap3 = ConversationStateSnapshot.model_validate(step3["state_after"])
    snap4 = ConversationStateSnapshot.model_validate(step4["state_after"])

    cdm = CoreDecisionManager(call_sid="sim_muy04bh0")
    res3 = cdm.evaluate_state(snapshot=snap3, turn_speaker="client", turn_text=step3["text"], turn_id=3)
    dec3 = res3.decision
    assert dec3.meeting_gate_open is True
    assert dec3.confidence == 0.925

    # Evaluate Turn 4 (Rep turn continuation)
    res4 = cdm.evaluate_state(snapshot=snap4, turn_speaker="salesperson", turn_text=step4["text"], turn_id=4)
    dec4 = res4.decision

    # 1. Action, Posture, Push, Confidence, Reason Codes (Gap 3)
    assert dec4.primary_action == StrategicAction.ACKNOWLEDGE
    assert dec4.strategic_posture == "protect"
    assert dec4.push_strength == "none"
    # Carried confidence is recomputed from the new snapshot, NOT the old Turn 3 value
    assert dec4.confidence == 0.845
    assert dec4.confidence != dec3.confidence
    assert "CONVERSION_CONFIRMED" in dec4.reason_codes
    assert "CONFIRM_AND_PROTECT_ACTIVE" in dec4.reason_codes
    assert "STRATEGY_CARRIED_FORWARD" in dec4.reason_codes

    # 2. Snapshot consistency: Decision bound to V10 MUST reflect V10's closed gate and readiness
    assert dec4.source_state_version == snap4.state_version  # V10
    assert dec4.meeting_gate_open == snap4.conversion_gate.is_open, "Turn 4 decision must match V10 snapshot gate"
    assert dec4.meeting_gate_open is False, "Turn 4 decision must report closed gate matching V10 snapshot"
    assert dec4.conversion_confirmed is True
    assert dec4.strategic_interpretation["meeting_gate_open"] is False

    # 3. Evidence Considered must reflect current immutable source snapshot V10
    gate_ev = [e for e in dec4.evidence_considered if e.startswith("Gate:")]
    assert len(gate_ev) == 1
    assert "Gate: CLOSED" in gate_ev[0], f"Gate in evidence_considered must be CLOSED, got {gate_ev[0]}"
    assert "Gate: OPEN" not in gate_ev[0]

    # 4. Preserve historical strategy provenance separately
    assert dec4.carried_forward_from_turn_id == 3
    assert dec4.carried_forward_from_decision_id == dec3.decision_id
    assert any("Strategy provenance: carried forward from Turn 3" in e for e in dec4.evidence_considered)
    assert dec4.should_prompt is False  # Suppressed on rep turn


def test_point1_carried_strategy_invalidated_when_gate_closes_on_closing_ask():
    """Gap 3: Add validity check on carry-forward.
    When previous strategy was a closing ask (COMMITMENT_CLOSE) or close-style push,
    and on the subsequent turn the gate is CLOSED, carry-forward is strictly INVALID.
    It must abort carry-forward and trigger fresh evaluation to protect the closed-gate invariant.
    """
    call_sid = "sim_test_carry_invalidation"
    cdm = CoreDecisionManager(call_sid=call_sid)

    # Create previous decision: Open gate closing ask
    prev_dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=5,
        source_turn_id=3,
        source_event_id="ev_turn_3_v5",
        utterance_turn_id=3,
        should_prompt=True,
        strategic_objective="Direct appointment close",
        primary_action=StrategicAction.COMMITMENT_CLOSE,
        strategic_posture="advance",
        push_strength="direct_ask",
        confidence=0.88,
        meeting_gate_open=True,
        conversion_confirmed=False,
    )
    cdm.latest_decision = prev_dec

    # Current snapshot where gate has closed (e.g. objection raised or criteria lapsed)
    closed_snap = ConversationStateSnapshot(
        state_id="state_closed_test",
        call_sid=call_sid,
        state_version=6,
        last_updated_turn_id=4,
        conversation_stage="discovery",
        conversion_gate=MeetingConversionGate(
            is_open=False,
            status="closed",
            conversion_target="appointment",
            conditions=[],
        ),
        conversion_confirmed=False,
        readiness=ReadinessBreakdown(readiness_score=45.0, confidence=0.7),
        decision_structure=DecisionStructure(decision_maker_present=True, confidence=0.8),
    )

    # 1. Direct validity check helper rejects the carried closing ask
    context = cdm.engine._build_context(closed_snap)
    is_valid, reason = cdm._is_carried_strategy_valid_for_snapshot(prev_dec, closed_snap, context)
    assert is_valid is False
    assert "Cannot carry forward COMMITMENT_CLOSE when meeting gate is closed" in reason

    # 2. In evaluate_state on filler utterance ("okay"):
    # Carry-forward is aborted and fresh evaluation runs (producing a non-closing action compliant with closed gate)
    eval_res = cdm.evaluate_state(
        snapshot=closed_snap,
        turn_speaker="prospect",
        turn_text="okay",
        turn_id=4,
    )
    dec = eval_res.decision
    assert dec.primary_action != StrategicAction.COMMITMENT_CLOSE
    assert "STRATEGY_CARRIED_FORWARD" not in dec.reason_codes
    assert dec.meeting_gate_open is False


def test_point1_live_state_mutation_does_not_affect_frozen_decision():
    """Smaller items: 'Immutable' verification.
    Assert that mutating the live ConversationStateManager / snapshot after the decision
    is made leaves the recorded decision's gate, readiness, and evidence identical
    to the frozen snapshot at the source version.
    """
    call_sid = "sim_test_immutability"
    cdm = CoreDecisionManager(call_sid=call_sid)

    snap = ConversationStateSnapshot(
        state_id="state_immutable_test",
        call_sid=call_sid,
        state_version=3,
        last_updated_turn_id=2,
        conversation_stage="discovery",
        conversion_gate=MeetingConversionGate(
            is_open=True,
            status="open",
            conversion_target="appointment",
            conditions=[],
        ),
        conversion_confirmed=False,
        readiness=ReadinessBreakdown(readiness_score=72.5, confidence=0.8),
        decision_structure=DecisionStructure(decision_maker_present=True, confidence=0.8),
    )

    res = cdm.evaluate_state(
        snapshot=snap,
        turn_speaker="prospect",
        turn_text="We might be interested next month.",
        turn_id=2,
    )
    decision = res.decision

    # Capture initial frozen values
    initial_gate_open = decision.meeting_gate_open
    initial_readiness = decision.strategic_interpretation.get("readiness_score")
    initial_evidence = list(decision.evidence_considered)

    # Mutate the live snapshot object in-place
    snap.conversion_gate.is_open = False
    snap.conversion_gate.status = "closed"
    snap.readiness.readiness_score = 12.0
    snap.state_version = 99

    # Retrieve decision from cache
    cached_dec = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=2,
        source_state_version=3,
        source_event_id=decision.source_event_id,
        state_call_sid=call_sid,
    )

    # The cached decision and original decision instance remain completely unaffected
    assert cached_dec is not None
    assert cached_dec.meeting_gate_open == initial_gate_open
    assert cached_dec.meeting_gate_open is True
    assert cached_dec.strategic_interpretation.get("readiness_score") == initial_readiness
    assert cached_dec.evidence_considered == initial_evidence
    assert cached_dec.source_state_version == 3


def test_point1_snapshot_missing_call_sid_fails_closed_in_cache_check():
    """Smaller items: Fail-closed verification when snapshot has no call_sid.
    Asserts that a snapshot with missing or None call_sid strictly refuses cache lookup
    and NEVER falls back to self.call_sid to create a false cache hit.
    """
    call_sid = "sim_test_missing_snap_call_sid"
    cdm = CoreDecisionManager(call_sid=call_sid)

    # 1. Direct cache check with empty state_call_sid returns None
    assert StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=1,
        source_state_version=1,
        source_event_id="ev_turn_1_v1",
        state_call_sid="",
    ) is None

    # 2. Populate cache with a valid decision for turn 1
    valid_dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=1,
        source_turn_id=1,
        source_event_id="ev_turn_1_v1",
        strategic_objective="Inquire regarding move timeline",
        primary_action=StrategicAction.QUESTION,
        confidence=0.8,
    )
    StrategicDecisionCache.put(call_sid=call_sid, decision=valid_dec, turn_id=1, source_event_id="ev_turn_1_v1")

    # 3. Snapshot with call_sid = "" must NOT fall back to self.call_sid
    snap_no_sid = ConversationStateSnapshot(
        state_id="state_no_call_sid",
        call_sid="",
        state_version=1,
        last_updated_turn_id=1,
        conversation_stage="discovery",
    )
    # Cache lookup should not return valid_dec because snap_call_sid is empty
    eval_res = cdm.evaluate_state(
        snapshot=snap_no_sid,
        turn_speaker="client",
        turn_text="hello",
        turn_id=1,
        use_cache=True,
    )
    # The result decision is freshly evaluated and NOT the cached decision instance
    assert eval_res.decision.decision_id != valid_dec.decision_id


def test_point1_carried_strategy_rejected_when_active_objection_recurs():
    """Smaller items & Client Item 3:
    When an objection is active and recurring (e.g. commission fee raised repeatedly at Turns 6 and 7),
    carrying forward a confirmed conversion or protect strategy is strictly INVALID.
    It must abort carry-forward so the objection is addressed directly.
    """
    call_sid = "sim_test_recurring_obj_rejection"
    cdm = CoreDecisionManager(call_sid=call_sid)

    # Previous decision was a confirmed conversion strategy
    prev_dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=8,
        source_turn_id=3,
        source_event_id="ev_turn_3_v8",
        strategic_objective="Protect confirmed commitment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        strategic_posture="protect",
        push_strength="none",
        reason_codes=["CONVERSION_CONFIRMED", "CONFIRM_AND_PROTECT_ACTIVE"],
        confidence=0.90,
        meeting_gate_open=True,
        conversion_confirmed=True,
    )
    cdm.latest_decision = prev_dec

    # Current snapshot with recurring fee objection (recurrence_count = 2)
    recurring_obj = ObjectionRecord(
        objection_id="obj_fee_rec",
        canonical_category="commission_fee",
        initial_statement="Your commission is too expensive",
        latest_statement="But honestly the commission still bothers me",
        first_turn_id=3,
        last_updated_turn_id=6,
        recurrence_count=2,
        lifecycle_state="active",
        confidence=0.85,
    )
    snap = ConversationStateSnapshot(
        state_id="state_rec_obj",
        call_sid=call_sid,
        state_version=14,
        last_updated_turn_id=6,
        conversation_stage="commitment_confirmed",
        conversion_gate=MeetingConversionGate(
            is_open=True,
            status="open",
            conversion_target="appointment",
            conditions=[],
        ),
        objections=[recurring_obj],
        conversion_confirmed=True,
    )

    context = cdm.engine._build_context(snap)
    is_valid, reason = cdm._is_carried_strategy_valid_for_snapshot(prev_dec, snap, context)
    assert is_valid is False
    assert "Cannot carry forward confirmation or close strategy while active objection is recurring" in reason


@pytest.mark.parametrize("run_semantic", [False, True])
def test_point1_second_fixture_canonical_live_replay(run_semantic: bool):
    """Smaller items: Second fixture live replay verification.
    Replays the canonical 18-turn script LIVE with both semantic layer OFF and ON.
    Proves that every single step passes strict 4-key binding, source state ownership,
    and bound cache retrieval.
    """
    script_path = Path(__file__).resolve().parent.parent / "data" / "script_canonical.json"
    assert script_path.exists(), f"Canonical script file not found: {script_path}"
    raw_turns = json.loads(script_path.read_text(encoding="utf-8"))

    call_sid = f"sim_canonical_live_sem_{run_semantic}"
    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns(
        call_sid=call_sid,
        raw_turns=raw_turns,
        save_report=False,
        run_semantic_analysis=run_semantic,
    )

    assert len(rep.timeline) >= 10, "Expected at least 10 turns in canonical replay"

    for step in rep.timeline:
        sd = step.strategic_decision
        if not sd:
            continue
        sa = step.state_after

        # Validate 4 keys + source state ownership
        is_valid, reason = StrategicDecisionCache.validate_binding(
            decision=sd,
            call_sid=call_sid,
            turn_id=step.turn_id,
            source_state_version=sa.state_version,
            source_event_id=sd.source_event_id,
            state_call_sid=sa.call_sid,
        )
        assert is_valid is True, f"Turn {step.turn_id} failed binding: {reason}"

        # Bound lookup succeeds
        bound = StrategicDecisionCache.get_strictly_bound(
            call_sid=call_sid,
            turn_id=step.turn_id,
            source_state_version=sa.state_version,
            source_event_id=sd.source_event_id,
            state_call_sid=sa.call_sid,
        )
        assert bound is not None
        assert bound.source_turn_id == step.turn_id
        assert bound.source_state_version == sa.state_version

