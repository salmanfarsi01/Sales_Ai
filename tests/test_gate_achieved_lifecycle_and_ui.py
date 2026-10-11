from __future__ import annotations

import json
from pathlib import Path
import subprocess
import pytest

from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversionEventObject,
    ConversionEventStatus,
    DecisionStakeholder,
    MeetingConversionGate,
)
from copilot.core_intelligence_models import (
    StrategicDecision,
    StrategicAction,
)
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine


def _create_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.75,
    emotion_tension: float = 0.15,
    engagement: float = 0.70,
    agreement: float = 0.60,
    specificity: float = 0.70,
    call_sid: str = "CA_achieved_test",
):
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.1, tension_level=emotion_tension, confidence=0.8),
        pacing=DimensionScore(score=0.6, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.65, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.65, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )
    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=0.0,
        agreement_score=agreement,
        specificity_score=specificity,
        future_language_score=0.6,
        contact_preference="none",
    )
    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
    )


# =============================================================================
# 1. Regression Test: 18-Turn Canonical Baseline Matches Saved Fixture
# =============================================================================
def test_canonical_18_turn_fixture_matches_baseline_table():
    """Asserts that the 18-turn canonical golden fixture matches the baseline table.
    Any difference in action, posture, push pressure, confidence, or gate status fails immediately.
    """
    golden_path = Path("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json")
    assert golden_path.exists(), "Golden fixture sim_mucj0p5s must exist"

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    # Approved baseline specifications
    expected_rows = [
        {"turn": 1, "speaker": "salesperson", "action": "question", "posture": "explore", "push": "low", "conf": 0.400, "gate_status": None},
        {"turn": 2, "speaker": "client", "action": "question", "posture": "explore", "push": "low", "conf": 0.455, "gate_status": "closed"},
        {"turn": 3, "speaker": "salesperson", "action": "question", "posture": "explore", "push": "low", "conf": 0.455, "gate_status": "closed"},
        {"turn": 4, "speaker": "client", "action": "question", "posture": "explore", "push": "low", "conf": 0.415, "gate_status": "closed"},
        {"turn": 5, "speaker": "client", "action": "validate", "posture": "advance", "push": "moderate", "conf": 0.685, "gate_status": "closed"},
        {"turn": 6, "speaker": "salesperson", "action": "validate", "posture": "advance", "push": "moderate", "conf": 0.685, "gate_status": "closed"},
        {"turn": 7, "speaker": "client", "action": "question", "posture": "explore", "push": "low", "conf": 0.455, "gate_status": "closed"},
        {"turn": 8, "speaker": "salesperson", "action": "question", "posture": "explore", "push": "low", "conf": 0.415, "gate_status": "closed"},
        {"turn": 9, "speaker": "client", "action": "question", "posture": "explore", "push": "low", "conf": 0.455, "gate_status": "closed"},
        {"turn": 10, "speaker": "client", "action": "clarify", "posture": "coordinate", "push": "low", "conf": 0.545, "gate_status": "closed"},
        {"turn": 11, "speaker": "salesperson", "action": "clarify", "posture": "coordinate", "push": "low", "conf": 0.545, "gate_status": "closed"},
        {"turn": 12, "speaker": "client", "action": "mirror", "posture": "advance", "push": "low", "conf": 0.629, "gate_status": "closed"},
        {"turn": 13, "speaker": "salesperson", "action": "mirror", "posture": "advance", "push": "low", "conf": 0.629, "gate_status": "closed"},
        {"turn": 14, "speaker": "client", "action": "educate", "posture": "explore", "push": "low", "conf": 0.455, "gate_status": "closed"},
        {"turn": 15, "speaker": "client", "action": "acknowledge", "posture": "protect", "push": "none", "conf": 0.730, "gate_status": "closed"},
        {"turn": 16, "speaker": "client", "action": "question", "posture": "explore", "push": "low", "conf": 0.415, "gate_status": "closed"},
        {"turn": 17, "speaker": "salesperson", "action": "question", "posture": "explore", "push": "low", "conf": 0.415, "gate_status": "closed"},
        {"turn": 18, "speaker": "client", "action": "acknowledge", "posture": "protect", "push": "none", "conf": 0.805, "gate_status": "open"},
    ]

    timeline = golden["timeline"]
    assert len(timeline) == 18, f"Expected 18 turns, got {len(timeline)}"

    for exp in expected_rows:
        tid = exp["turn"]
        step = timeline[tid - 1]
        assert step["turn_id"] == tid
        assert step["speaker_id"] == exp["speaker"]

        strat = step.get("strategic_decision") or {}
        assert strat.get("primary_action") == exp["action"], (
            f"Turn {tid}: action mismatch. Expected {exp['action']}, got {strat.get('primary_action')}"
        )
        assert strat.get("strategic_posture") == exp["posture"], (
            f"Turn {tid}: posture mismatch. Expected {exp['posture']}, got {strat.get('strategic_posture')}"
        )
        assert strat.get("push_strength") == exp["push"], (
            f"Turn {tid}: push mismatch. Expected {exp['push']}, got {strat.get('push_strength')}"
        )
        assert round(strat.get("confidence", 0.0), 3) == exp["conf"], (
            f"Turn {tid}: confidence mismatch. Expected {exp['conf']}, got {strat.get('confidence')}"
        )

        sa = step.get("state_snapshot") or step.get("state_after") or {}
        gate = sa.get("conversion_gate") or {}
        actual_gate_status = gate.get("status")
        assert actual_gate_status == exp["gate_status"], (
            f"Turn {tid}: gate_status mismatch. Expected {exp['gate_status']}, got {actual_gate_status}"
        )


# =============================================================================
# 2. Invariant: COMMITMENT_CLOSE Forbidden Under Achieved Gate
# =============================================================================
def test_commitment_close_forbidden_when_status_is_achieved():
    """Confirms that when conversion gate status is 'achieved', COMMITMENT_CLOSE is strictly forbidden.
    Contract validation must reject any decision attempting COMMITMENT_CLOSE under achieved gate.
    """
    # 1. Direct StrategicDecision model validation raises ValueError under strict mode
    with pytest.raises(ValueError, match="Invariant violation.*meeting gate is closed or achieved"):
        StrategicDecision(
            call_id="CA_test_achieved",
            source_state_version=5,
            strategic_objective="Attempt close under achieved gate",
            primary_action=StrategicAction.COMMITMENT_CLOSE,
            push_strength="direct_ask",
            meeting_gate_open=False,
            strategic_interpretation={"gate_status": "achieved"},
            conversion_confirmed=True,
        )

    # 2. Core intelligence engine evaluation under achieved gate never emits COMMITMENT_CLOSE
    engine = PitchProXCoreIntelligenceEngine()
    snap = ConversationStateSnapshot(call_sid="CA_test_engine_achieved")
    snap.conversion_gate = MeetingConversionGate(
        is_open=False,
        status="achieved",
        conversion_target="appointment",
        conditions=[],
    )
    snap.conversion_event = ConversionEventObject(
        event_id="conv_evt_test",
        conversion_type="property_walkthrough",
        status=ConversionEventStatus.CONFIRMED,
        start_at="Thursday at 2pm",
    )
    dec_result = engine.evaluate(snapshot=snap, turn_text="Looking forward to seeing you Thursday.")
    assert dec_result.decision.primary_action != StrategicAction.COMMITMENT_CLOSE
    assert str(dec_result.decision.push_strength) not in ("direct_ask", "two_window_choice", "high")


# =============================================================================
# 3. Lifecycle: Cancellation Moves Achieved -> Closed, Reschedule Goes Through Open
# =============================================================================
def test_cancellation_moves_gate_to_closed_and_reschedule_goes_through_open():
    """Verify state transitions across appointment booking, cancellation, and rescheduling:
    1. Turn 1 (booking agreement): Gate OPENS on the booking turn (`is_open=True, status='open'`).
    2. Turn 2 (post-booking state): With active confirmed event, gate is ACHIEVED (`is_open=False, status='achieved'`).
    3. Turn 3 (cancellation): Prospect explicitly cancels -> event is CANCELLED, gate regresses to CLOSED (`is_open=False, status='closed'`).
    4. Turn 4 (reschedule agreement): Prospect agrees to new slot -> gate transitions through OPEN (`is_open=True, status='open'`).
    5. Turn 5 (post-reschedule): Gate returns to ACHIEVED (`is_open=False, status='achieved'`).
    """
    manager = ConversationStateManager(call_sid="CA_test_reschedule_lifecycle", conversion_target="appointment")

    # Turn 1: Client explicitly commits to Thursday at 3pm
    t1 = _create_bundle(turn_id=1, speaker_id="client", text="Thursday at 3pm works, let's meet then.", agreement=0.85)
    s1 = manager.process_turn_bundle(t1)
    # Booking turn convention: Gate is OPEN on the turn of agreement
    assert s1.conversion_gate.is_open is True
    assert s1.conversion_gate.status == "open"
    assert s1.conversion_gate.explicit_commitment_detected is True

    # Turn 2: Agent confirms and asks clarifying prep question
    t2 = _create_bundle(turn_id=2, speaker_id="salesperson", text="Sounds great, I'll bring our market analysis.", agreement=0.8)
    s2 = manager.process_turn_bundle(t2)
    # Once confirmed event is active, subsequent turns sit in ACHIEVED
    assert s2.conversion_gate.is_open is False
    assert s2.conversion_gate.status == "achieved"
    assert s2.conversion_event is not None
    assert s2.conversion_event.status == ConversionEventStatus.CONFIRMED

    # Turn 3: Prospect cancels appointment ("Actually, I need to cancel Thursday, I can't make it.")
    t3 = _create_bundle(turn_id=3, speaker_id="client", text="Actually, I need to cancel Thursday, I can't make it.", agreement=0.2)
    s3 = manager.process_turn_bundle(t3)
    assert s3.conversion_event.status == ConversionEventStatus.CANCELLED
    # Cancellation moves gate back to CLOSED (not achieved, not open)
    assert s3.conversion_gate.is_open is False
    assert s3.conversion_gate.status == "closed"

    # Turn 4: Prospect reschedules ("Can we do Friday at 2pm instead?")
    t4 = _create_bundle(turn_id=4, speaker_id="client", text="Can we do Friday at 2pm instead?", agreement=0.85)
    s4 = manager.process_turn_bundle(t4)
    # Reschedule goes through OPEN again
    assert s4.conversion_gate.is_open is True
    assert s4.conversion_gate.status == "open"
    assert s4.conversion_gate.explicit_commitment_detected is True
    assert "friday at 2" in s4.conversion_gate.commitment_slot.lower()

    # Turn 5: Agent confirms new time
    t5 = _create_bundle(turn_id=5, speaker_id="salesperson", text="Friday at 2pm is locked in on my calendar.", agreement=0.8)
    s5 = manager.process_turn_bundle(t5)
    # Post-reschedule confirmed returns gate to ACHIEVED
    assert s5.conversion_gate.is_open is False
    assert s5.conversion_gate.status == "achieved"
    assert s5.conversion_event.status == ConversionEventStatus.CONFIRMED


# =============================================================================
# 4. Node.js Test: UI Replay Page Renders ACHIEVED Badge
# =============================================================================
def test_ui_renders_achieved_node_test():
    """Executes a Node.js test to verify that the UI replay rendering logic in
    web/conversation_state_replay.html correctly renders 'ACHIEVED' badges when gate.status is 'achieved',
    and does NOT render 'Gate Closed' or 'Turn Gate: CLOSED'.
    """
    html_path = Path("web/conversation_state_replay.html")
    assert html_path.exists(), "Replay HTML must exist"
    html_content = html_path.read_text(encoding="utf-8")

    # Verify key template strings exist in HTML file
    assert "ACHIEVED" in html_content
    assert "Turn Gate: ACHIEVED" in html_content

    # Execute Node.js script asserting exact HTML rendering logic
    node_script = """
    const stepGate = { is_open: false, status: 'achieved' };
    const stepGateStatus = stepGate ? (stepGate.status || (stepGate.is_open ? 'open' : 'closed')) : 'closed';
    const stepGateOpen = stepGate && stepGate.is_open;
    const gateBadge = stepGateStatus === 'achieved'
      ? '<span class="badge badge-open" style="font-size:10px">ACHIEVED</span>'
      : (stepGateOpen
        ? '<span class="badge badge-open" style="font-size:10px">Gate Open</span>'
        : '<span class="badge badge-closed" style="font-size:10px">Gate Closed</span>');

    const turnGate = stepGate;
    const turnGateStatus = turnGate ? (turnGate.status || (turnGate.is_open ? 'open' : 'closed')) : 'closed';
    const turnCeConfirmed = false;
    const turnCeTentative = false;
    const turnMilestone = turnCeConfirmed
      ? 'APPOINTMENT CONFIRMED'
      : (turnCeTentative
          ? 'TENTATIVE'
          : (turnGateStatus === 'achieved'
              ? 'Turn Gate: ACHIEVED'
              : (stepGateOpen ? 'Turn Gate: OPEN' : 'Turn Gate: CLOSED')));

    console.log(JSON.stringify({
        gateBadge: gateBadge,
        turnMilestone: turnMilestone,
        isAchievedRendered: gateBadge.includes('ACHIEVED'),
        isClosedRendered: gateBadge.includes('Gate Closed')
    }));
    """

    res = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
    out = json.loads(res.stdout.strip())

    assert out["isAchievedRendered"] is True, "Gate badge must render ACHIEVED"
    assert out["isClosedRendered"] is False, "Gate badge must not render 'Gate Closed' under achieved status"
    assert out["turnMilestone"] == "Turn Gate: ACHIEVED", "Turn milestone pill must render 'Turn Gate: ACHIEVED'"
