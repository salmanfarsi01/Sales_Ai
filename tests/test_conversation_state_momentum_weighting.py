"""Tests for Issue #5: Dynamic & Responsive Momentum Family Scoring and Decision Clarity.

Guarantees:
1. Swallowed-ELIF Regression Guard:
   Sole decision maker declarations ('I am the one making this decision') preceded by
   a salesperson turn are recorded as primary_decision_maker='sole_decision_maker'
   and decision_maker_present=True.
2. Responsive Decision Structure Clarity:
   - Turn 1 (unestablished): 40.0
   - Turn 2 (sole authority declared): 95.0
   - Turn 10 (absent spouse declared): 55.0 (regresses)
   - Turn 18 (spouse confirmed attending): 95.0 (recovers)
3. Responsive Future / Operational Behavior:
   - Base floor in discovery: 15.0
   - Scheduling coordination / tentative meeting: 45.0+
   - Confirmed meeting / appointment: 80.0+
4. Responsive Commitment Behavior:
   - Reflects dims.commitment / confirmed meeting status (81.5+) rather than flat 54.0.
5. Preservation of Issue #4 Guarantee:
   - Turn 18 objection_movement stays strictly at 70.0 (financial concern stays partially_resolved).
6. Forward Momentum Trajectory:
   - Turn 18 momentum rises to ~71.8% with an 'advancing' trend (delta >= +4.0).
"""

import pytest
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import (
    BehavioralSignalInputBundle,
    DimensionScore,
    EmotionState,
)
from copilot.conversation_state_models import (
    ConversationStage,
    ConversionEventStatus,
    DecisionStakeholder,
    PersistentFactRecord,
)
from copilot.conversation_scoring import ConversationScoringEngine


def _make_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    agreement: float = 0.5,
    pacing: float = 0.6,
    trust: float = 0.5,
    engagement: float = 0.5,
    future_lang: float = 0.0,
    specificity: float = 0.5,
) -> BehavioralSignalInputBundle:
    return BehavioralSignalInputBundle(
        call_sid="CA_issue5_test",
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        timestamp_ms=turn_id * 3000,
        trust=DimensionScore(score=trust, confidence=0.8, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.1, confidence=0.85),
        pacing=DimensionScore(score=pacing, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.8, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        agreement_score=agreement,
        specificity_score=specificity,
        future_language_score=future_lang,
        inference_confidence=0.85,
        semantic_confidence=0.90,
    )


def test_turn_2_sole_decision_maker_declaration_recorded_after_salesperson_turn():
    """Guards against swallowed-elif bug: Turn 2 sole decision maker declaration must set primary_decision_maker."""
    mgr = ConversationStateManager(call_sid="CA_turn2_guard")

    # Turn 1: Salesperson opening (no third-party inquiry)
    t1 = _make_bundle(1, "salesperson", "Hi, thanks for making time today — tell me a bit about what's going on.")
    snap1 = mgr.process_turn_bundle(t1)
    assert snap1.decision_structure.primary_decision_maker is None
    assert snap1.momentum.family_scores["decision_structure_clarity"] == 40.0

    # Turn 2: Prospect sole authority declaration
    t2 = _make_bundle(2, "client", "I'm the one making this decision, no one else needs to sign off.")
    snap2 = mgr.process_turn_bundle(t2)

    # Must be recorded in decision_structure
    assert snap2.decision_structure.primary_decision_maker == "sole_decision_maker"
    assert snap2.decision_structure.decision_maker_present is True
    assert len(snap2.decision_structure.stakeholders) == 0

    # Decision clarity family score must leap to 95.0
    assert snap2.momentum.family_scores["decision_structure_clarity"] == 95.0


def test_turn_10_decision_regression_drops_decision_clarity():
    """Verify declaring an absent spouse on Turn 10 drops decision_structure_clarity from 95.0 to 55.0."""
    mgr = ConversationStateManager(call_sid="CA_turn10_regression")

    t1 = _make_bundle(1, "salesperson", "Tell me about the house.")
    mgr.process_turn_bundle(t1)
    t2 = _make_bundle(2, "client", "I'm the one making this decision, no one else needs to sign off.")
    snap2 = mgr.process_turn_bundle(t2)
    assert snap2.momentum.family_scores["decision_structure_clarity"] == 95.0

    # Turn 10: Spouse absent disclosure
    t10 = _make_bundle(10, "client", "Actually, my wife would really need to be part of this conversation before we go any further.")
    snap10 = mgr.process_turn_bundle(t10)

    assert snap10.decision_structure.decision_maker_present is False
    assert any(s.role == "wife" and s.presence == "absent" for s in snap10.decision_structure.stakeholders)
    # Clarity drops because a key stakeholder is absent
    assert snap10.momentum.family_scores["decision_structure_clarity"] == 55.0


def test_future_operational_behavior_escalation_across_stages():
    """Verify future_operational_behavior scales from discovery floor (15.0) to scheduling (45.0) to confirmed meeting (80.0+)."""
    engine = ConversationScoringEngine()

    mgr = ConversationStateManager(call_sid="CA_future_op_test")
    t1 = _make_bundle(1, "client", "We are just getting started.")
    snap1 = mgr.process_turn_bundle(t1)
    # Discovery floor without closing targets
    assert snap1.momentum.family_scores["future_operational_behavior"] == 15.0

    # Stage scheduling with tentative meeting fact
    snap1.conversation_stage = ConversationStage.SCHEDULING
    mom_sched = engine.compute_momentum(_make_bundle(2, "client", "Next week works"), snap1)
    assert mom_sched.family_scores["future_operational_behavior"] >= 45.0

    # Confirmed meeting
    snap1.facts.append(
        PersistentFactRecord(
            fact_id="f_confirmed_test",
            category="timeline",
            fact_key="confirmed_meeting_time",
            fact_value="Thursday At 3",
            status="active",
            confidence=0.9,
            source_turn_id=3,
            timestamp_ms=3000,
        )
    )
    mom_confirmed = engine.compute_momentum(_make_bundle(3, "client", "Thursday at 3 works"), snap1)
    assert mom_confirmed.family_scores["future_operational_behavior"] >= 80.0


def test_turn_18_momentum_reflects_forward_movement_and_preserves_objection_movement():
    """Verify Turn 18 confirmed appointment advances momentum into 'advancing' (>65%) while strictly keeping objection_movement at 70.0."""
    import json
    with open("reports/synthetic/conversation_state_sim_mucj0p5s.json") as f:
        data = json.load(f)

    mgr = ConversationStateManager(call_sid="sim_mucj0p5s_full_test")
    snaps = []
    for t in data["timeline"]:
        eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
        snap = mgr.process_turn_bundle(eb)
        snaps.append(snap.model_copy(deep=True))

    # Turn 17 check
    snap17 = snaps[16]
    assert snap17.momentum.family_scores["objection_movement"] == 70.0
    assert snap17.momentum.family_scores["decision_structure_clarity"] == 55.0

    # Turn 18 checks
    snap18 = snaps[17]
    fs18 = snap18.momentum.family_scores

    # 1. Issue #4 guarantee preserved: financial objection stays partially_resolved, not 100.0 or 30.0
    assert fs18["objection_movement"] == 70.0

    # 2. Issue #5 fixes:
    # Decision structure clarity recovered to 95.0 (wife confirmed attending, caller present)
    assert fs18["decision_structure_clarity"] == 95.0

    # Future operational behavior reflects confirmed appointment (80.0+)
    assert fs18["future_operational_behavior"] >= 80.0

    # Commitment behavior reflects confirmed conversion next step (80.0+)
    assert fs18["commitment_behavior"] >= 80.0

    # Composite momentum advances above 70%
    assert snap18.momentum.momentum_score >= 70.0
    assert snap18.momentum.trend == "advancing"
    assert snap18.momentum.trend_delta >= 4.0


def test_decision_structure_clarity_transitional_tier_scores_75():
    """Verify that when primary authority is declared/identified, but decision_maker_present
    is False and no stakeholder is absent (transitional tier before physical presence confirmation),
    decision_structure_clarity evaluates strictly to 75.0."""
    engine = ConversationScoringEngine()
    mgr = ConversationStateManager(call_sid="CA_transitional_clarity_test")

    t1 = _make_bundle(1, "salesperson", "Hello")
    snap = mgr.process_turn_bundle(t1)

    # State combination: has_primary=True, decision_maker_present=False, has_absent_stakeholder=False
    snap.decision_structure.primary_decision_maker = "board_chair"
    snap.decision_structure.decision_maker_present = False
    snap.decision_structure.stakeholders = []

    t2 = _make_bundle(2, "client", "Understood.")
    mom = engine.compute_momentum(t2, snap)
    assert mom.family_scores["decision_structure_clarity"] == 75.0


def test_future_operational_behavior_immediate_turn_update_on_conversion_upgrade_and_downgrade():
    """Verify that future_operational_behavior and commitment_behavior immediately reflect
    conversion_event status changes on the EXACT SAME TURN (both upgrade to confirmed and downgrade to tentative),
    eliminating the 1-turn lag where Turn 7 stayed at 15.0 and Turn 9 stayed at 80.0."""
    import json
    with open("reports/synthetic/conversation_state_sim_mucj0p5s.json") as f:
        data = json.load(f)

    mgr = ConversationStateManager(call_sid="sim_lag_verification_test")
    snaps = []
    for t in data["timeline"]:
        eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
        snap = mgr.process_turn_bundle(eb)
        snaps.append(snap.model_copy(deep=True))

    # Turn 6: Tentative pre-confirmation floor
    snap6 = snaps[5]
    assert snap6.momentum.family_scores["future_operational_behavior"] == 15.0

    # Turn 7: UPGRADE case - timing objection resolved, conversion_event transitions to CONFIRMED.
    # MUST immediately reflect confirmed tier (>= 80.0) on Turn 7 itself, NOT lag until Turn 8!
    snap7 = snaps[6]
    assert snap7.conversion_event is not None
    assert str(snap7.conversion_event.status).lower().endswith("confirmed")
    assert snap7.momentum.family_scores["future_operational_behavior"] >= 80.0
    assert snap7.momentum.family_scores["commitment_behavior"] >= 80.0

    # Turn 9: DOWNGRADE case - prospect expresses hesitation ("Maybe next week... let me think"),
    # conversion_event downgrades to TENTATIVE.
    # MUST immediately drop to tentative tier (45.0) on Turn 9 itself, NOT lag until Turn 10!
    snap9 = snaps[8]
    assert snap9.conversion_event is not None
    assert str(snap9.conversion_event.status).lower().endswith("tentative")
    assert snap9.momentum.family_scores["future_operational_behavior"] == 45.0
    assert snap9.momentum.family_scores["commitment_behavior"] <= 65.0

    # Turn 18: CONFIRMATION case - final appointment confirmed.
    snap18 = snaps[17]
    assert snap18.conversion_event is not None
    assert str(snap18.conversion_event.status).lower().endswith("confirmed")
    assert snap18.momentum.family_scores["future_operational_behavior"] >= 80.0
    assert snap18.momentum.family_scores["commitment_behavior"] >= 80.0

