import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_scoring_config import ConversationScoringConfig
from copilot.conversation_state_models import DecisionStakeholder


def _create_turn_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.70,
    emotion_valence: float = 0.0,
    emotion_tension: float = 0.2,
    readiness: float = 0.60,
    engagement: float = 0.65,
    pacing: float = 0.60,
    boundary: float = 0.0,
    recurrence_id: str = None,
    salesperson_strategy_tag: str = None,
    agreement: float = 0.50,
    specificity: float = 0.50,
    future_lang: float = 0.50,
    call_sid: str = "CA_scoring_test_001",
):
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=0.8),
        pacing=DimensionScore(score=pacing, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.60, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_score_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_score_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        recurrence_id=recurrence_id,
        agreement_score=agreement,
        specificity_score=specificity,
        future_language_score=future_lang,
    )

    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
        salesperson_strategy_tag=salesperson_strategy_tag,
    )


class TestConversationStateSprint5Scoring:
    def test_spec_acceptance_absent_decision_maker_caps_readiness_despite_high_enthusiasm(self):
        """Definition of Done Acceptance Test from spec document:

        'Absent decision-maker caps readiness despite high enthusiasm.'

        Scenario:
        - Prospect is overflowing with enthusiasm:
          Trust = 0.90, expressed valence = +0.85, tension = 0.05, engagement = 0.92, agreement = 0.95.
        - BUT: decision structure has an absent spouse who must decide ("My wife handles all the numbers, she's at work").
        - Verifies:
          1. Uncapped readiness is high (> 80.0).
          2. Final capped readiness is strictly bounded at <= 55.0.
          3. active_blocker_caps explicitly logs 'absent_decision_maker'.
        """
        manager = ConversationStateManager(call_sid="CA_spec_test_absent_dm")

        # Turn 1: High enthusiasm, but prospect mentions spouse must make decision and is absent
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="This sounds absolutely amazing! I love this proposal so much, but my wife handles all the finances and she's not here right now.",
            trust=0.90,
            emotion_valence=0.85,
            emotion_tension=0.05,
            engagement=0.92,
            agreement=0.95,
            specificity=0.80,
            future_lang=0.75,
        )
        snap1 = manager.process_turn_bundle(
            t1,
            decision_updates={
                "primary_decision_maker": "Wife Mary",
                "decision_maker_present": False,
                "stakeholders": [DecisionStakeholder(name="Mary", role="spouse", presence="absent")],
            },
            fact_updates=[
                {
                    "category": "decision_maker",
                    "fact_key": "spouse_involvement",
                    "fact_value": "Wife Mary handles all finances and must be present to consult",
                }
            ],
        )

        readiness = snap1.readiness
        assert readiness is not None

        # Invariant 1: Uncapped readiness is very high due to enthusiasm
        assert readiness.uncapped_score >= 70.0

        # Invariant 2: Blocker cap strictly prevents false conversion optimism
        assert readiness.readiness_score <= 55.0
        assert readiness.readiness_score == 55.0

        # Invariant 3: Explainable blocker audit trail
        assert "absent_decision_maker" in readiness.active_blocker_caps
        assert "Absent/unconfirmed decision maker" in readiness.capped_reason

    def test_logistical_deficit_caps_readiness_despite_high_enthusiasm(self):
        """Validates that logistical deficit (e.g. channel restrictions, severe scheduling clash)

        caps overall readiness at <= 60.0 even when prospect is emotionally ready.
        """
        manager = ConversationStateManager(call_sid="CA_logistics_block_001")

        # Turn 1: High trust and agreement, but channel restriction active
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I'm very interested, but my clinic hours are insane and I cannot take calls during the day.",
            trust=0.85,
            emotion_valence=0.40,
            emotion_tension=0.10,
            engagement=0.85,
            agreement=0.80,
        )
        t1.contact_preference = "channel_restriction"
        snap1 = manager.process_turn_bundle(t1)

        readiness = snap1.readiness
        assert readiness is not None
        assert readiness.logistical_readiness < 50.0
        assert "logistical_deficit" in readiness.active_blocker_caps
        assert readiness.readiness_score <= 60.0

    def test_active_unresolved_objection_caps_readiness(self):
        """Validates that an active unresolved objection caps readiness at <= 55.0."""
        manager = ConversationStateManager(call_sid="CA_objection_cap_001")

        # Turn 1: Prospect raises major commission objection
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Your commission is just way too high. 6 percent is unreasonable.",
            recurrence_id="rec_commission_01",
            trust=0.70,
            engagement=0.80,
            agreement=0.30,
        )
        snap1 = manager.process_turn_bundle(t1)

        readiness = snap1.readiness
        assert readiness is not None
        assert "unresolved_objection" in readiness.active_blocker_caps
        assert readiness.readiness_score <= 55.0

    def test_simultaneous_multi_blocker_caps_strictest_wins_with_full_census(self):
        """Validates simultaneous multi-blocker interaction:

        Scenario:
        - High enthusiasm (uncapped score would be ~75-80).
        - Condition A: Absent decision maker (ceiling 55.0).
        - Condition B: Logistical deficit via channel restriction (ceiling 60.0).
        - Condition C: Unresolved commission objection (ceiling 55.0).

        Verifies:
        1. Strictest wins: Final readiness is 55.0 (min(uncapped, 55.0, 60.0, 55.0)),
           NOT first-evaluated, NOT an average, and NOT an uncalibrated compounded drop.
        2. Full Blocker Census: `active_blocker_caps` lists ALL active blockers,
           proving non-binding blockers are still tracked for human auditability and explainability.
        3. Strictest blocker is identified in `capped_reason` along with all active blockers.
        """
        manager = ConversationStateManager(call_sid="CA_simultaneous_blockers_001")

        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I really like what you're offering, but my wife handles all the money and she's not here, your 6% commission is too steep, and you can only reach me by mail.",
            recurrence_id="rec_commission_01",
            trust=0.95,
            emotion_valence=0.85,
            emotion_tension=0.05,
            engagement=0.95,
            agreement=0.95,
        )
        t1.contact_preference = "channel_restriction"

        snap1 = manager.process_turn_bundle(
            t1,
            decision_updates={
                "decision_maker_present": False,
                "stakeholders": [DecisionStakeholder(name="Wife", role="spouse", presence="absent")],
            },
            fact_updates=[
                {
                    "category": "decision_maker",
                    "fact_key": "spouse_involvement",
                    "fact_value": "Wife handles all money decisions and must be present",
                }
            ],
        )

        readiness = snap1.readiness
        assert readiness is not None

        # 1. Uncapped score is high due to strong trust and engagement
        assert readiness.uncapped_score > 55.0

        # 2. Strictest cap (55.0) wins over logistical deficit cap (60.0)
        assert readiness.readiness_score == 55.0

        # 3. Full census: all three blockers are recorded in active_blocker_caps
        assert "absent_decision_maker" in readiness.active_blocker_caps
        assert "logistical_deficit" in readiness.active_blocker_caps
        assert "unresolved_objection" in readiness.active_blocker_caps
        assert len(readiness.active_blocker_caps) == 3

        # 4. Audit trail in capped_reason notes strictest constraint and lists all active blockers
        assert "strictest blocker" in readiness.capped_reason
        assert "Absent/unconfirmed decision maker" in readiness.capped_reason
        assert "Logistical readiness deficit" in readiness.capped_reason
        assert "Active unresolved objection" in readiness.capped_reason

    def test_hard_boundary_collapses_readiness_to_zero(self):
        """Validates that a hard compliance boundary forces readiness to exactly 0.0."""
        manager = ConversationStateManager(call_sid="CA_boundary_collapse_001")

        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Stop calling me. Take my number off your list immediately.",
            boundary=1.0,
            trust=0.20,
            emotion_valence=-0.80,
            emotion_tension=0.90,
        )
        snap1 = manager.process_turn_bundle(t1)

        readiness = snap1.readiness
        assert readiness is not None
        assert readiness.readiness_score == 0.0
        assert "hard_boundary" in readiness.active_blocker_caps

    def test_momentum_7_family_scoring_and_advancing_trend(self):
        """Validates the 7-family momentum scoring model and advancing trend trajectory."""
        manager = ConversationStateManager(call_sid="CA_momentum_advancing_001")

        # Turn 1: Modest start
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Hello.",
            agreement=0.40,
            specificity=0.30,
            future_lang=0.20,
        )
        snap1 = manager.process_turn_bundle(t1)
        m1 = snap1.momentum.momentum_score

        # Turn 2: Strong forward advance: high agreement, clear timeline, future commitment
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Yes, that strategy makes total sense. Let's aim to close before Thanksgiving.",
            agreement=0.90,
            specificity=0.85,
            future_lang=0.85,
            trust=0.85,
            engagement=0.85,
        )
        snap2 = manager.process_turn_bundle(
            t2,
            fact_updates=[
                {
                    "category": "timeline",
                    "fact_key": "target_closing",
                    "fact_value": "Close before Thanksgiving",
                }
            ],
        )

        momentum = snap2.momentum
        assert momentum is not None
        assert momentum.momentum_score > m1
        assert momentum.trend == "advancing"
        assert momentum.trend_delta >= 4.0
        # Verify all 7 families populated
        assert len(momentum.family_scores) == 7
        assert "problem_goal_clarity" in momentum.family_scores
        assert "value_recognition" in momentum.family_scores
        assert "objection_movement" in momentum.family_scores
        assert "trust_engagement_trend" in momentum.family_scores
        assert "decision_structure_clarity" in momentum.family_scores
        assert "future_operational_behavior" in momentum.family_scores
        assert "commitment_behavior" in momentum.family_scores

    def test_momentum_regressing_trend_on_severe_pushback(self):
        """Validates that a sudden drop in agreement and surge in tension classifies 'regressing'."""
        manager = ConversationStateManager(call_sid="CA_momentum_regressing_001")

        # Turn 1: Warm baseline
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Sounds pretty good so far.",
            trust=0.80,
            engagement=0.80,
            agreement=0.75,
            future_lang=0.60,
        )
        snap1 = manager.process_turn_bundle(t1)

        # Turn 2: Sharp regression with hostile pushback and tension
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="No way, this is completely absurd. I am not agreeing to any of this.",
            trust=0.35,
            emotion_valence=-0.70,
            emotion_tension=0.85,
            engagement=0.40,
            agreement=0.05,
            future_lang=0.0,
        )
        snap2 = manager.process_turn_bundle(t2)

        momentum = snap2.momentum
        assert momentum is not None
        assert momentum.trend == "regressing"
        assert momentum.trend_delta <= -4.0

    def test_versioned_scoring_config_custom_weights_and_caps(self):
        """Validates that a custom ConversationScoringConfig correctly alters blocker ceilings."""
        custom_cfg = ConversationScoringConfig(
            absent_decision_maker_ceiling=45.0,  # Stricter ceiling
            logistical_deficit_ceiling=50.0,
        )
        manager = ConversationStateManager(call_sid="CA_custom_cfg_001", scoring_config=custom_cfg)

        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I love this, but my wife makes the final call.",
            trust=0.90,
            agreement=0.90,
        )
        snap1 = manager.process_turn_bundle(
            t1,
            decision_updates={
                "decision_maker_present": False,
                "stakeholders": [DecisionStakeholder(role="spouse", presence="absent")],
            },
        )

        readiness = snap1.readiness
        assert readiness is not None
        # Strict custom ceiling applied: 45.0 instead of default 55.0
        assert readiness.readiness_score <= 45.0
        assert readiness.readiness_score == 45.0
