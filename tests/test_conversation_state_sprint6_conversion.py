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
from copilot.conversation_state_models import DecisionStakeholder, ObjectionRecord


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
    contact_preference: str = "none",
    call_sid: str = "CA_conv_test_001",
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
        contributing_evidence_ids=[f"ev_conv_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_conv_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        recurrence_id=recurrence_id,
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


class TestConversationStateSprint6NearMissAndGate:
    """Rigorous near-miss testing suite verifying that 6 of 7 conditions met

    strictly keeps the Meeting/Conversion Gate CLOSED for each of the 7 conditions.
    """

    def _setup_baseline_passing_manager(self, call_sid: str, target: str = "appointment") -> ConversationStateManager:
        """Sets up a state manager where all 7 gate conditions would pass."""
        manager = ConversationStateManager(call_sid=call_sid, conversion_target=target)
        # Turn 1: Establish baseline facts and decision structure
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I own the property on 123 Maple Street and I make the decisions. Let's talk.",
            trust=0.75,
            emotion_valence=0.2,
            emotion_tension=0.15,
            engagement=0.75,
            agreement=0.70,
            specificity=0.70,
            future_lang=0.65,
        )
        manager.process_turn_bundle(
            t1,
            decision_updates={
                "primary_decision_maker": "Self (Owner)",
                "decision_maker_present": True,
                "stakeholders": [DecisionStakeholder(name="Self", role="owner", presence="on_call")],
            },
            fact_updates=[
                {"category": "property", "fact_key": "property_address", "fact_value": "123 Maple Street"},
                {"category": "timeline", "fact_key": "move_in_date", "fact_value": "November closing desired"},
            ],
        )
        return manager

    def test_near_miss_condition_1_trust_collapsing_keeps_gate_closed(self):
        """Near-Miss 1: Conditions 2-7 pass, but Trust is collapsing (trust=0.35, tension=0.75).

        Asserts gate remains CLOSED with 'trust_not_collapsing' as the failing condition.
        """
        manager = self._setup_baseline_passing_manager("CA_near_miss_1")
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="I am really starting to feel uncomfortable with this conversation.",
            trust=0.35,
            emotion_valence=-0.6,
            emotion_tension=0.75,
            engagement=0.75,  # Condition 2 ok
            agreement=0.60,   # Condition 4 ok
            specificity=0.70,
        )
        snap = manager.process_turn_bundle(t2)
        gate = snap.conversion_gate

        assert gate is not None
        assert gate.is_open is False
        assert gate.status == "closed"
        assert "trust_not_collapsing" in gate.failed_conditions
        assert len(gate.failed_conditions) == 1
        assert "Trust score (0.35) is below minimum threshold" in gate.blocking_reasons[0]
        assert snap.push_strength.state == "protect_and_shorten"

    def test_near_miss_condition_2_engagement_detached_keeps_gate_closed(self):
        """Near-Miss 2: Conditions 1, 3-7 pass, but Engagement is detached (engagement=0.30).

        Asserts gate remains CLOSED with 'engagement_on_topic' as the failing condition.
        """
        manager = self._setup_baseline_passing_manager("CA_near_miss_2")
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Uh huh.",
            trust=0.75,       # Condition 1 ok
            emotion_tension=0.15,
            engagement=0.30,  # Condition 2 FAILS (< 0.50)
            agreement=0.60,   # Condition 4 ok
            specificity=0.60,
        )
        snap = manager.process_turn_bundle(t2)
        gate = snap.conversion_gate

        assert gate is not None
        assert gate.is_open is False
        assert gate.status == "closed"
        assert "engagement_on_topic" in gate.failed_conditions
        assert len(gate.failed_conditions) == 1
        assert "Engagement score (0.30) is below on-topic threshold" in gate.blocking_reasons[0]

    def test_near_miss_condition_3_unresolved_objection_keeps_gate_closed(self):
        """Near-Miss 3: Conditions 1-2, 4-7 pass, but an active unresolved objection exists.

        Asserts gate remains CLOSED with 'objections_resolved_or_partial' as the failing condition.
        """
        manager = self._setup_baseline_passing_manager("CA_near_miss_3", target="signed_listing_agreement")
        # Add an active unresolved objection
        manager.current_state.objections.append(
            ObjectionRecord(
                canonical_category="commission_fee",
                initial_statement="Six percent commission is way too high.",
                latest_statement="Still think that fee is unreasonable.",
                lifecycle_state="unresolved",
                first_turn_id=1,
                last_updated_turn_id=2,
            )
        )
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="I like the marketing plan, but your fee structure still bothers me.",
            trust=0.70,
            engagement=0.75,
            agreement=0.60,
        )
        snap = manager.process_turn_bundle(t2)
        gate = snap.conversion_gate

        assert gate is not None
        assert gate.is_open is False
        assert gate.status == "closed"
        assert "objections_resolved_or_partial" in gate.failed_conditions
        assert len(gate.failed_conditions) == 1
        assert "commission_fee" in gate.blocking_reasons[0]
        assert snap.push_strength.state == "resolve_then_ask"

    def test_near_miss_condition_4_unrecognized_value_keeps_gate_closed(self):
        """Near-Miss 4: Conditions 1-3, 5-7 pass, but prospect sees no value justification.

        Asserts gate remains CLOSED with 'clear_value_reason' as the failing condition.
        """
        manager = self._setup_baseline_passing_manager("CA_near_miss_4")
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="I don't see why I would need any of this services honestly.",
            trust=0.70,
            engagement=0.70,
            agreement=0.10,  # Value failure
            specificity=0.40,
        )
        snap = manager.process_turn_bundle(t2)
        # Force momentum value_recognition family score to be low
        snap.momentum.family_scores["value_recognition"] = 25.0
        # Re-evaluate
        gate = manager.conversion_engine.evaluate_gate(t2, snap)

        assert gate.is_open is False
        assert "clear_value_reason" in gate.failed_conditions
        assert len(gate.failed_conditions) == 1

    def test_near_miss_condition_5_absent_decision_maker_keeps_gate_closed(self):
        """Near-Miss 5: Conditions 1-4, 6-7 pass, but spouse decision-maker is absent.

        Asserts gate remains CLOSED with 'decision_maker_aligned' as the failing condition.
        """
        manager = self._setup_baseline_passing_manager("CA_near_miss_5")
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="I love this, but my husband makes all the final listing calls and he is out of town.",
            trust=0.80,
            engagement=0.80,
            agreement=0.85,
        )
        snap = manager.process_turn_bundle(
            t2,
            decision_updates={
                "decision_maker_present": False,
                "stakeholders": [DecisionStakeholder(name="Husband", role="spouse", presence="absent")],
            },
            fact_updates=[
                {"category": "decision_maker", "fact_key": "spouse_involvement", "fact_value": "Husband handles all listing calls"}
            ],
        )
        gate = snap.conversion_gate

        assert gate is not None
        assert gate.is_open is False
        assert "decision_maker_aligned" in gate.failed_conditions
        assert len(gate.failed_conditions) == 1
        assert "Decision-maker absent" in gate.blocking_reasons[0]

    def test_near_miss_condition_6_logistical_deficit_keeps_gate_closed(self):
        """Near-Miss 6: Conditions 1-5, 7 pass, but logistical readiness is deficient (< 40.0).

        Asserts gate remains CLOSED with 'plausible_logistics' as the failing condition.
        """
        manager = self._setup_baseline_passing_manager("CA_near_miss_6")
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="I am interested, but I cannot take calls or meetings on weekdays at all.",
            trust=0.75,
            engagement=0.70,
            agreement=0.70,
            contact_preference="channel_restriction",
        )
        snap = manager.process_turn_bundle(t2)
        gate = snap.conversion_gate

        assert gate is not None
        assert gate.is_open is False
        assert "plausible_logistics" in gate.failed_conditions
        assert len(gate.failed_conditions) == 1

    def test_near_miss_condition_7_hard_boundary_keeps_gate_closed(self):
        """Near-Miss 7: Conditions 1-6 pass, but prospect utters a hard compliance boundary.

        Asserts gate remains CLOSED with 'no_active_boundary' as the failing condition.
        """
        manager = self._setup_baseline_passing_manager("CA_near_miss_7")
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Please do not call this number again. Take me off your list.",
            trust=0.70,
            engagement=0.70,
            boundary=0.95,  # Hard compliance boundary
        )
        snap = manager.process_turn_bundle(t2)
        gate = snap.conversion_gate

        assert gate is not None
        assert gate.is_open is False
        assert "no_active_boundary" in gate.failed_conditions
        assert snap.push_strength.state == "respect_record_exit"

    def test_all_seven_conditions_met_gate_opens_cleanly(self):
        """All 7 Conditions Met: Trust healthy, on-topic, zero unresolved objections,

        value recognized, decision-maker present, viable logistics, zero boundary.
        Asserts gate opens cleanly with push strength 'direct_ask'.
        """
        manager = self._setup_baseline_passing_manager("CA_all_7_open")
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="That marketing strategy sounds like a great fit for my timeline. Let's coordinate.",
            trust=0.85,
            emotion_valence=0.4,
            emotion_tension=0.10,
            engagement=0.85,
            agreement=0.90,
            specificity=0.80,
            future_lang=0.75,
        )
        snap = manager.process_turn_bundle(t2)
        gate = snap.conversion_gate

        assert gate is not None
        assert gate.is_open is True
        assert gate.status == "open"
        assert len(gate.failed_conditions) == 0
        assert len(gate.blocking_reasons) == 0
        assert snap.push_strength.state == "direct_ask"
        assert "Meeting gate is open" in snap.push_strength.rationale


class TestConversationStateSprint6PushStrengthAndAcceptance:
    """Validates the 6 push strength states and the 5 specification acceptance tests."""

    def test_push_strength_state_machine_states(self):
        """Verifies that all 6 push strength states trigger accurately under their defined conditions."""
        manager = ConversationStateManager(call_sid="CA_push_strength_suite")

        # 1. respect_record_exit
        t_bound = _create_turn_bundle(turn_id=1, speaker_id="client", text="Stop calling me right now.", boundary=0.95)
        snap1 = manager.process_turn_bundle(t_bound)
        assert snap1.push_strength.state == "respect_record_exit"

        # 2. protect_and_shorten (low trust / elevated tension)
        manager_low_trust = ConversationStateManager(call_sid="CA_low_trust")
        t_tense = _create_turn_bundle(turn_id=1, speaker_id="client", text="I don't trust sales reps.", trust=0.30, emotion_tension=0.75)
        snap2 = manager_low_trust.process_turn_bundle(t_tense)
        assert snap2.push_strength.state == "protect_and_shorten"

        # 3. resolve_then_ask (moderate trust + active objection blocking signed_listing_agreement)
        manager_obj = ConversationStateManager(call_sid="CA_resolve_obj", conversion_target="signed_listing_agreement")
        manager_obj.current_state.objections.append(
            ObjectionRecord(
                canonical_category="commission_fee",
                initial_statement="Fee is high",
                latest_statement="Fee is high",
                lifecycle_state="unresolved",
                first_turn_id=1,
                last_updated_turn_id=1,
            )
        )
        t_obj = _create_turn_bundle(turn_id=1, speaker_id="client", text="I'm listening, but what about the fee?", trust=0.60)
        snap3 = manager_obj.process_turn_bundle(t_obj)
        assert snap3.push_strength.state == "resolve_then_ask"

    def test_spec_acceptance_1_friendly_but_vague_prospect(self):
        """Spec Acceptance Test 1: Friendly-but-vague prospect != high readiness; gate stays closed.

        Prospect is polite and agreeable ("Yeah sounds nice, sure"), but non-committal on details.
        Asserts:
        - Readiness is moderate, not high (< 70).
        - Meeting gate is closed.
        - Push strength recommends 'two_window_choice'.
        """
        manager = ConversationStateManager(call_sid="CA_spec_friendly_vague")
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Yeah, that sounds nice. Sure, maybe sometime.",
            trust=0.65,
            emotion_valence=0.3,
            emotion_tension=0.1,
            engagement=0.65,
            agreement=0.75,    # Agreeable
            specificity=0.20,  # Vague
            future_lang=0.15,
        )
        snap = manager.process_turn_bundle(t1)

        assert snap.readiness.readiness_score < 70.0
        assert snap.conversion_gate.is_open is False
        assert snap.push_strength.state == "resolve_then_ask"

    def test_spec_acceptance_2_confirmed_walkthrough_success_even_if_short(self):
        """Spec Acceptance Test 2: Confirmed walkthrough = success even if call was short.

        Prospect explicitly confirms walkthrough on turn 2:
        'Yes, come by Thursday at 4 PM to walk through the property.'
        Asserts:
        - conversion_event.status == 'confirmed'
        - conversion_event.conversion_type == 'property_walkthrough'
        - conversion_event.followup_is_conversion == True
        """
        manager = ConversationStateManager(call_sid="CA_spec_short_walkthrough")

        # Turn 1: Agent introduces
        t1 = _create_turn_bundle(turn_id=1, speaker_id="salesperson", text="Would Thursday afternoon work for a 15-minute walkthrough?")
        manager.process_turn_bundle(t1)

        # Turn 2: Prospect confirms walkthrough explicitly
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Yes, come by Thursday at 4 PM to walk through the property.",
            trust=0.80,
            engagement=0.85,
            agreement=0.90,
            specificity=0.85,
            future_lang=0.85,
        )
        snap = manager.process_turn_bundle(
            t2,
            decision_updates={"primary_decision_maker": "Owner", "decision_maker_present": True},
        )

        conv = snap.conversion_event
        assert conv is not None
        assert conv.status == "confirmed"
        assert conv.conversion_type == "property_walkthrough"
        assert conv.followup_is_conversion is True
        assert "Thursday" in conv.start_at
        assert conv.confirmation_confidence >= 0.85

    def test_spec_acceptance_3_send_me_something_alone_not_conversion(self):
        """Spec Acceptance Test 3: 'Send me something' alone != conversion.

        Prospect says: 'Just send me an email with your pricing information.'
        Asserts:
        - conversion_event.followup_is_conversion == False
        - conversion_event.status == 'blocked' (not 'confirmed')
        - conversion_event.conversion_type == 'information_send'
        - conversion_gate remains closed
        """
        manager = ConversationStateManager(call_sid="CA_spec_send_something")
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Just send me an email with your pricing information and I will review it.",
            trust=0.55,
            engagement=0.55,
            agreement=0.40,
            specificity=0.30,
        )
        snap = manager.process_turn_bundle(t1)

        conv = snap.conversion_event
        assert conv is not None
        assert conv.followup_is_conversion is False
        assert conv.status != "confirmed"
        assert conv.conversion_type == "information_send"
        assert "brush_off_send_only_not_conversion" in conv.blocking_items
        assert snap.conversion_gate.is_open is False

    def test_spec_acceptance_4_repeated_unresolved_objection_blocks_despite_polite_okay(self):
        """Spec Acceptance Test 4: Repeated unresolved objection blocks despite polite 'okay'.

        Prospect raised commission fee objection. Later says 'Okay sure' to an explanation,
        but the objection remains unresolved.
        Asserts:
        - Meeting gate is closed (objections_resolved_or_partial fails).
        - Push strength is 'resolve_then_ask'.
        """
        manager = ConversationStateManager(call_sid="CA_spec_repeat_objection", conversion_target="signed_listing_agreement")

        # Turn 1: Prospect raises commission objection
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Your commission fee is too high. I'm not paying 6%.",
            trust=0.55,
            engagement=0.70,
        )
        manager.process_turn_bundle(t1)

        # Turn 2: Agent pitches marketing value
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="salesperson",
            text="We provide premium video production and MLS syndicated marketing.",
            salesperson_strategy_tag="marketing_value_justification",
        )
        manager.process_turn_bundle(t2)

        # Turn 3: Prospect politely says "Okay sure", but commission objection is still unresolved
        t3 = _create_turn_bundle(
            turn_id=3,
            speaker_id="client",
            text="Okay, sure, that sounds good, but the commission is still a big deal to me.",
            trust=0.60,
            engagement=0.70,
            agreement=0.65,
            recurrence_id="obj_comm_rec_01",
        )
        snap = manager.process_turn_bundle(t3)

        assert snap.conversion_gate.is_open is False
        assert "objections_resolved_or_partial" in snap.conversion_gate.failed_conditions
        assert snap.push_strength.state == "resolve_then_ask"

    def test_spec_acceptance_5_absent_decision_maker_caps_readiness_and_blocks_gate(self):
        """Spec Acceptance Test 5: Absent decision-maker caps readiness and blocks gate despite high enthusiasm.

        Prospect has 95% enthusiasm, but wife Mary is the decision-maker and absent.
        Asserts:
        - Uncapped readiness > 75, capped readiness <= 55.0.
        - Meeting gate is closed ('decision_maker_aligned' fails).
        """
        manager = ConversationStateManager(call_sid="CA_spec_absent_dm_gate")
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I am so thrilled with everything you've shown me! But Mary makes all financial decisions and she's not here.",
            trust=0.90,
            emotion_valence=0.8,
            emotion_tension=0.05,
            engagement=0.90,
            agreement=0.95,
            specificity=0.80,
            future_lang=0.75,
        )
        snap = manager.process_turn_bundle(
            t1,
            decision_updates={
                "primary_decision_maker": "Wife Mary",
                "decision_maker_present": False,
                "stakeholders": [DecisionStakeholder(name="Mary", role="spouse", presence="absent")],
            },
            fact_updates=[
                {"category": "decision_maker", "fact_key": "spouse_involvement", "fact_value": "Mary handles finances and must be present"}
            ],
        )

        assert snap.readiness.readiness_score <= 55.0
        assert snap.conversion_gate.is_open is False
        assert "decision_maker_aligned" in snap.conversion_gate.failed_conditions
