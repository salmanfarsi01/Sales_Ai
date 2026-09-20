"""Phase 10 Acceptance Test Suite: Complete Verification Against Both Specs.

Proves the ConversationState build against all 13 explicit acceptance criteria
from ConversationState_Implementation_Plan.md:

Client Principles:
1. Current truth supersedes old truth, old preserved in history (Principle #1)
2. Evidence separate from interpretation (Principle #2)
3. Interpreted changes traceable to evidence IDs (Principle #4)
4. Not every field updates on every turn / filler suppressed (Principle #3 / Phase 5)
5. Objection reactivated in new words retains history (Principle #5 / Phase 3)
6. Persistent facts survive unrelated turns (Principle #6 / Phase 2)
7. Dimensions move independently (Principle #7)
8. Uncertainty preserved, not forced to certainty (Principle #8)
9. Stale async result cannot overwrite newer state (Principle #9)

Specification Document Acceptance Tests:
10. Friendly-but-passive prospect -> high engagement, moderate readiness naturally (< 70.0) without blocker cap (Test A)
11. Confirmed walkthrough = success even if call was short (Test B)
12. 'Send me something' alone != conversion (Test C)
13. Repeated commission objection: stays unresolved on polite filler, but resolves on genuine commitment (Test D)
14. Absent decision-maker caps readiness despite high enthusiasm (Test E)
"""

import pytest

from copilot.conversation_state_manager import (
    ConversationStateManager,
    StaleStateUpdateError,
)
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    StateChangeRecord,
    PersistentFactRecord,
)
from copilot.conversation_state_contract import (
    BehavioralSignalInputBundle,
    extract_behavioral_bundle,
)
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot


def _create_turn_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    call_sid: str = "CA_phase10_test",
    timestamp_ms: int = None,
    trust: float = 0.70,
    emotion_tension: float = 0.20,
    emotion_valence: float = 0.10,
    pacing: float = 0.60,
    engagement: float = 0.70,
    momentum: float = 0.60,
    readiness: float = 0.50,
    confidence: float = 0.85,
    boundary: float = 0.0,
    recurrence_type: str = "none",
    recurrence_id: str = None,
    recurrence_count: int = 0,
    question_type: str = "none",
    specificity: float = 0.60,
    agreement: float = 0.50,
    future_lang: float = 0.0,
    contact_pref: str = "none",
    salesperson_strategy_tag: str = None,
    evidence_ids: list = None,
) -> BehavioralSignalInputBundle:
    """Helper to synthesize canonical BehavioralSignalInputBundle objects."""
    ts_ms = timestamp_ms if timestamp_ms is not None else turn_id * 3000
    ev_ids = evidence_ids or [f"ev_p10_{turn_id:03d}"]

    inf = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=ts_ms,
        trust=DimensionScore(score=trust, confidence=confidence, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=confidence),
        pacing=DimensionScore(score=pacing, confidence=confidence, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=confidence, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=momentum, confidence=confidence, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=confidence, primary_horizon="last_60_90s"),
        overall_confidence=confidence,
        contributing_evidence_ids=ev_ids,
    )

    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_p10_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        recurrence_type=recurrence_type,
        recurrence_id=recurrence_id,
        recurrence_count=recurrence_count,
        question_type=question_type,
        specificity_score=specificity,
        future_language_score=future_lang,
        agreement_score=agreement,
        contact_preference=contact_pref,
        semantic_confidence=confidence,
    )

    bundle = extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=sem,
        salesperson_strategy_tag=salesperson_strategy_tag,
    )
    bundle.timestamp_ms = ts_ms
    return bundle


class TestConversationStatePhase10Acceptance:

    # =========================================================================
    # 1. Client Principle #1: Truth Supersession & History Preservation
    # =========================================================================
    def test_client_principle_1_truth_supersession_history_preservation(self):
        """Principle #1: Current truth supersedes old truth, while preserving
        superseded truth in history with timestamps and source turn IDs.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_1")

        # Turn 2: Prospect asserts decision to stay
        t2 = _create_turn_bundle(turn_id=2, speaker_id="client", text="We've decided to stay in the home.")
        fact_payload = [
            {
                "category": "timeline",
                "fact_key": "moving_decision",
                "fact_value": "Decided to stay in the home",
                "confidence": 0.90,
            }
        ]
        s2 = manager.process_turn_bundle(t2, fact_updates=fact_payload)
        orig_fact = s2.get_active_fact("moving_decision")
        assert orig_fact is not None
        assert orig_fact.fact_value == "Decided to stay in the home"
        assert orig_fact.status == "active"
        orig_id = orig_fact.fact_id

        # Turn 8: Prospect conditional supersession
        t8 = _create_turn_bundle(
            turn_id=8,
            speaker_id="client",
            text="I'd still move if I believed there was a better strategy.",
        )
        s8 = manager.process_turn_bundle(t8)

        # Invariant A: Active fact is updated with new truth
        active_f = s8.get_active_fact("moving_decision")
        assert active_f is not None
        assert active_f.fact_id != orig_id
        assert active_f.status == "active"
        assert active_f.source_turn_id == 8
        assert "better strategy" in active_f.fact_value

        # Invariant B: Superseded fact is NOT destroyed; preserved in history
        all_facts = manager.facts_manager.get_all_facts()
        superseded_list = [f for f in all_facts if f.fact_key == "moving_decision" and f.status == "superseded"]
        assert len(superseded_list) == 1
        assert superseded_list[0].fact_id == orig_id
        assert superseded_list[0].superseded_by_fact_id == active_f.fact_id
        assert superseded_list[0].superseded_at_turn_id == 8

    # =========================================================================
    # 2. Client Principle #2: Evidence Separated from Interpretation
    # =========================================================================
    def test_client_principle_2_evidence_separated_from_interpretation(self):
        """Principle #2: Evidence is what Behavioral Signal reported; interpretation
        is what ConversationState concluded. Evidence and interpretation are stored
        as genuinely separate data models and fields.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_2")

        evidence_keys = ["ev_acoustic_speech_rate_drop", "ev_semantic_commission_mention"]
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I cannot pay a 6 percent commission on this sale.",
            trust=0.55,
            emotion_tension=0.45,
            recurrence_id="rec_fee_99",
            evidence_ids=evidence_keys,
        )

        state = manager.process_turn_bundle(t1)

        # Invariant A: Raw evidence bundle carries the upstream signal evidence unchanged
        assert hasattr(t1, "contributing_evidence_ids")
        assert all(k in t1.contributing_evidence_ids for k in evidence_keys)
        # Invariant B: State domain models are structurally isolated from the raw evidence object
        assert isinstance(state.dimensions.trust, float)
        assert state.dimensions.trust == 0.55
        assert state.dimensions.emotion_tension == 0.45
        assert len(state.objections) == 1
        assert state.objections[0].canonical_category == "commission_fee"
        # Invariant C: Processing state did NOT mutate or delete the raw bundle's evidence
        assert set(evidence_keys).issubset(set(t1.contributing_evidence_ids))

    # =========================================================================
    # 3. Client Principle #4: Interpreted Changes Traceable to Evidence IDs
    # =========================================================================
    def test_client_principle_4_interpreted_changes_traceable_to_evidence_ids(self):
        """Principle #4: Every single interpreted mutation can be queried backward
        to the exact contributing evidence IDs that justified it.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_4")

        evidence_keys = ["ev_pitch_variance_drop", "ev_semantic_objection_fee"]
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I cannot pay a 6 percent commission on this sale.",
            trust=0.55,
            emotion_tension=0.45,
            recurrence_id="rec_fee_99",
            evidence_ids=evidence_keys,
        )

        state = manager.process_turn_bundle(t1)

        # Invariant A: Every state change record contains originating evidence IDs
        assert len(state.change_history) > 0
        for change in state.change_history:
            assert len(change.evidence_ids) > 0, f"Change {change.field_path} has no evidence IDs"
            assert change.state_version_after > change.state_version_before

        # Invariant B: Specific interpreted domain changes trace backward to input evidence keys
        dim_change = next((c for c in state.change_history if c.field_path == "dimensions"), None)
        assert dim_change is not None
        assert any(ev in dim_change.evidence_ids for ev in evidence_keys)

        obj_change = next((c for c in state.change_history if "objections" in c.field_path), None)
        assert obj_change is not None
        assert any(ev in obj_change.evidence_ids for ev in evidence_keys)

    # =========================================================================
    # 3. Client Principle #3 (Phase 5): Not Every Field Updates on Every Turn
    # =========================================================================
    def test_client_principle_3_materiality_filter_suppresses_filler_churn(self):
        """Principle #3: Conversational filler, acknowledgments, and greetings
        must not mutate state or trigger version churn.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_3")

        # Turn 1: Substantive material turn
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="We own a single family property on Oak Street and want to sell.",
            specificity=0.85,
            trust=0.70,
        )
        s1 = manager.process_turn_bundle(t1)
        v1 = s1.state_version
        dims1 = s1.dimensions.model_dump()

        # Turn 2: Conversational filler ("Yeah.") with minimal shift
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Yeah.",
            specificity=0.10,
            agreement=0.50,
            trust=0.71,  # minor delta below 0.15 threshold
            emotion_tension=0.21,
        )
        s2 = manager.process_turn_bundle(t2)

        # Invariant: State version and dimension stability are preserved
        assert s2.state_version == v1
        assert s2.dimensions.model_dump() == dims1
        assert s2.last_updated_turn_id == 2

    # =========================================================================
    # 4. Client Principle #5 (Phase 3): Objection Reactivated in New Words
    # =========================================================================
    def test_client_principle_5_objection_reactivation_retains_recurrence_history(self):
        """Principle #5: When an objection is re-voiced in new words, it retains
        its full prior occurrence history and reactivates to 'unresolved'.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_5")

        # Turn 2: Objection raised
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Your commission is too steep.",
            recurrence_id="rec_comm_01",
            emotion_tension=0.35,
        )
        s2 = manager.process_turn_bundle(t2)
        assert len(s2.objections) == 1
        obj = s2.objections[0]
        assert obj.canonical_category == "commission_fee"
        assert obj.lifecycle_state == "unresolved"
        assert obj.recurrence_count == 1

        # Turn 3: Agent reframe
        t3 = _create_turn_bundle(
            turn_id=3,
            speaker_id="salesperson",
            text="If we net you more money, would that make sense?",
            salesperson_strategy_tag="financial_net_proceeds_reframe",
        )
        s3 = manager.process_turn_bundle(t3)
        assert "financial_net_proceeds_reframe" in s3.objections[0].attempted_strategies

        # Turn 4: Prospect resolves objection with forward advance
        t4 = _create_turn_bundle(
            turn_id=4,
            speaker_id="client",
            text="Alright, that makes sense. Show me how the net proceeds work on Thursday.",
            agreement=0.85,
            future_lang=0.75,
            readiness=0.82,
        )
        s4 = manager.process_turn_bundle(t4)
        assert s4.objections[0].lifecycle_state == "resolved"

        # Turn 5: Prospect repeats objection in new words -> REACTIVATED
        t5 = _create_turn_bundle(
            turn_id=5,
            speaker_id="client",
            text="Wait, before that, I still feel like paying thousands of dollars is too much money.",
            recurrence_id="rec_comm_01",
            recurrence_type="same_objection_repeated",
            emotion_tension=0.40,
        )
        s5 = manager.process_turn_bundle(t5)

        # Invariant: Single canonical objection record, reactivated with history
        assert len(s5.objections) == 1
        reactivated = s5.objections[0]
        assert reactivated.canonical_category == "commission_fee"
        assert reactivated.lifecycle_state == "reactivated"
        assert reactivated.recurrence_count == 2
        assert reactivated.first_turn_id == 2
        assert reactivated.last_updated_turn_id == 5
        assert "financial_net_proceeds_reframe" in reactivated.attempted_strategies

    # =========================================================================
    # 5. Client Principle #6 (Phase 2): Persistent Facts Survive Unrelated Turns
    # =========================================================================
    def test_client_principle_6_persistent_facts_survive_unrelated_turns(self):
        """Principle #6: Persistent facts established early in a call must
        survive unrelated turns (small talk, objections, logistics) intact.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_6")

        # Turn 1: Fact established
        t1 = _create_turn_bundle(turn_id=1, speaker_id="client", text="We have a 4-bedroom house in Dallas.")
        fact_payload = [
            {"category": "property", "fact_key": "property_type", "fact_value": "4-bedroom house in Dallas"},
            {"category": "financial", "fact_key": "mortgage_status", "fact_value": "Free and clear"},
        ]
        s1 = manager.process_turn_bundle(t1, fact_updates=fact_payload)
        assert len(s1.get_active_facts()) == 2

        # 6 Unrelated intervening turns
        unrelated_dialogue = [
            (2, "salesperson", "How has the weather been over there?"),
            (3, "client", "It has been quite rainy this week."),
            (4, "salesperson", "I understand. I have been serving this market for 12 years."),
            (5, "client", "That is good to know."),
            (6, "salesperson", "Would you like me to walk through the property?"),
            (7, "client", "Maybe sometime, send me your brochure first."),
        ]
        state = s1
        for tid, spk, txt in unrelated_dialogue:
            b = _create_turn_bundle(turn_id=tid, speaker_id=spk, text=txt)
            state = manager.process_turn_bundle(b)

        # Invariant: Both facts survived 6 unrelated turns intact
        active_facts = state.get_active_facts()
        assert len(active_facts) == 2
        f_prop = state.get_active_fact("property_type")
        f_mort = state.get_active_fact("mortgage_status")
        assert f_prop is not None and f_prop.fact_value == "4-bedroom house in Dallas"
        assert f_mort is not None and f_mort.fact_value == "Free and clear"

    # =========================================================================
    # 6. Client Principle #7: Dimensions Move Independently
    # =========================================================================
    def test_client_principle_7_dimensions_move_independently(self):
        """Principle #7: Behavioral dimensions must vary independently based
        on their own horizons and indicators, without artificial coupling.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_7")

        # Scenario A: High Engagement (0.85), Low Readiness (0.25)
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Tell me everything about your marketing! What about flyers and social media?",
            engagement=0.85,
            readiness=0.25,
            trust=0.70,
            specificity=0.30,
        )
        s1 = manager.process_turn_bundle(t1)
        assert s1.dimensions.engagement == 0.85
        assert s1.dimensions.readiness == 0.25
        assert s1.dimensions.engagement != s1.dimensions.readiness

        # Scenario B: High Trust (0.80), High Tension (0.75)
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="I trust your firm completely, but our family is in an urgent estate battle.",
            trust=0.80,
            emotion_tension=0.75,
            engagement=0.75,
            readiness=0.40,
        )
        s2 = manager.process_turn_bundle(t2)
        assert s2.dimensions.trust == 0.80
        assert s2.dimensions.emotion_tension == 0.75

    # =========================================================================
    # 7. Client Principle #8: Uncertainty Preserved, Not Forced to Certainty
    # =========================================================================
    def test_client_principle_8_weakest_link_uncertainty_preserved(self):
        """Principle #8: When upstream evidence has low confidence, ConversationState
        must preserve that uncertainty rather than forcing scores to default certainty.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_8")

        # Low upstream confidence (0.35)
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I suppose we could consider selling.",
            trust=0.60,
            confidence=0.35,  # Uncalibrated / noisy audio
        )
        s1 = manager.process_turn_bundle(t1)

        # Invariant: Confidence remains <= 0.35
        assert s1.dimensions.trust_confidence <= 0.35
        assert s1.dimensions.readiness_confidence <= 0.35
        assert s1.readiness.confidence <= 0.50

    # =========================================================================
    # 8. Client Principle #9: Stale Async Result Cannot Overwrite Newer State
    # =========================================================================
    def test_client_principle_9_stale_async_updates_strictly_rejected(self):
        """Principle #9: Out-of-order or earlier-timestamped packets must be
        rejected with StaleStateUpdateError, preserving state monotonicity.
        """
        manager = ConversationStateManager(call_sid="CA_p10_principle_9")

        t1 = _create_turn_bundle(turn_id=1, speaker_id="client", text="Turn 1", timestamp_ms=1000)
        manager.process_turn_bundle(t1)

        t2 = _create_turn_bundle(turn_id=2, speaker_id="client", text="Turn 2", timestamp_ms=3000)
        s2 = manager.process_turn_bundle(t2)
        assert s2.last_updated_turn_id == 2
        v2 = s2.state_version

        # Stale packet arriving late with earlier turn ID
        stale_turn_bundle = _create_turn_bundle(turn_id=1, speaker_id="client", text="Late Turn 1", timestamp_ms=800)
        with pytest.raises(StaleStateUpdateError) as exc_info:
            manager.process_turn_bundle(stale_turn_bundle)
        assert "Stale state rejection" in str(exc_info.value)
        assert manager.current_state.state_version == v2

    # =========================================================================
    # 9. Spec Doc Acceptance Test A: Friendly-But-Passive Prospect
    # =========================================================================
    def test_spec_acceptance_friendly_passive_prospect_readiness_capped(self):
        """Spec Test A: A friendly, agreeable prospect with vague optimism
        yields high engagement but moderate readiness naturally through weighted scoring (< 70.0),
        without firing an explicit blocker cap. Conversion gate remains closed due to lack of concrete value.
        """
        manager = ConversationStateManager(call_sid="CA_p10_spec_test_a")

        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Yeah, that sounds nice. Sure, maybe sometime. We love your agency!",
            trust=0.65,
            engagement=0.80,
            emotion_valence=0.30,
            emotion_tension=0.10,
            agreement=0.75,
            specificity=0.20,  # Vague, non-committal
            future_lang=0.15,
        )
        s1 = manager.process_turn_bundle(t1)

        # Invariant A: Engagement is high, but readiness is moderate purely from natural weighted scoring
        assert s1.dimensions.engagement >= 0.75
        assert s1.readiness.readiness_score < 70.0

        # Invariant B: ZERO blocker caps active (Proves this is NOT a blocker cap, unlike Spec Test E)
        assert len(s1.readiness.active_blocker_caps) == 0
        assert s1.readiness.readiness_score == s1.readiness.uncapped_score
        assert s1.readiness.capped_reason is None

        # Invariant C: Gate is closed specifically due to vague filler guard on value recognition, NOT a decision blocker
        assert s1.conversion_gate.is_open is False
        assert "clear_value_reason" in s1.conversion_gate.failed_conditions
        assert "decision_maker_aligned" not in s1.conversion_gate.failed_conditions

        # Invariant D: Push strength recommends alternative close for agreeable-but-vague prospect
        assert s1.push_strength.state == "two_window_choice"
        assert "binary choice" in s1.push_strength.recommended_action.lower()

    # =========================================================================
    # 10. Spec Doc Acceptance Test B: Confirmed Walkthrough on Short Call
    # =========================================================================
    def test_spec_acceptance_confirmed_walkthrough_short_call_conversion_success(self):
        """Spec Test B: A concrete walkthrough confirmed on a short call is a
        complete success. Gate opens and push strength selects direct_ask.
        """
        manager = ConversationStateManager(call_sid="CA_p10_spec_test_b")

        # Turn 1: Agent intro
        t1 = _create_turn_bundle(turn_id=1, speaker_id="salesperson", text="Hi, are you open to selling?")
        manager.process_turn_bundle(t1)

        # Turn 2: Prospect immediately commits to walkthrough
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Yes, come by this Thursday at 4 PM to inspect the home.",
            trust=0.85,
            engagement=0.85,
            readiness=0.80,
            specificity=0.90,
            agreement=0.95,
        )
        s2 = manager.process_turn_bundle(t2)

        # Invariants
        assert s2.conversion_event is not None
        assert s2.conversion_event.status == "confirmed"
        assert s2.conversion_event.conversion_type == "property_walkthrough"
        assert s2.conversion_gate.is_open is True
        assert s2.push_strength.state in ("confirm_and_protect", "direct_ask")

    # =========================================================================
    # 11. Spec Doc Acceptance Test C: 'Send Me Something' Alone != Conversion
    # =========================================================================
    def test_spec_acceptance_send_me_something_deflection_not_converted(self):
        """Spec Test C: A brush-off asking for email info does not open the
        conversion gate or select direct_ask.
        """
        manager = ConversationStateManager(call_sid="CA_p10_spec_test_c")

        t1 = _create_turn_bundle(turn_id=1, speaker_id="salesperson", text="Can we meet this week?")
        manager.process_turn_bundle(t1)

        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Just send me an email with your fees and pricing, thanks.",
            contact_pref="channel_restriction",
            specificity=0.25,
            agreement=0.40,
            readiness=0.35,
        )
        s2 = manager.process_turn_bundle(t2)

        # Invariants
        assert s2.conversion_gate.is_open is False
        assert s2.push_strength.state != "direct_ask"
        if s2.conversion_event:
            assert s2.conversion_event.status != "confirmed"

    # =========================================================================
    # 12. Spec Doc Acceptance Test D: Repeated Commission Objection Unresolved on 'Okay'
    # =========================================================================
    def test_spec_acceptance_repeated_commission_stays_unresolved_despite_polite_filler(self):
        """Spec Test D: Repeated commission objection stays unresolved despite
        a polite 'okay'. Conversion gate must stay closed.
        """
        manager = ConversationStateManager(call_sid="CA_p10_spec_test_d", conversion_target="signed_listing_agreement")

        # Turn 1: Objection raised
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="6 percent is too high.",
            recurrence_id="rec_fee_d",
        )
        manager.process_turn_bundle(t1)

        # Turn 2: Objection repeated
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Still think your commission is too expensive.",
            recurrence_id="rec_fee_d",
        )
        manager.process_turn_bundle(t2)

        # Turn 3: Agent reframe attempt
        t3 = _create_turn_bundle(
            turn_id=3,
            speaker_id="salesperson",
            text="We provide premier staging and marketing to maximize price.",
            salesperson_strategy_tag="financial_net_proceeds_reframe",
        )
        manager.process_turn_bundle(t3)

        # Turn 4: Polite conversational filler ("Okay, sure, but commission is still too high.")
        t4 = _create_turn_bundle(
            turn_id=4,
            speaker_id="client",
            text="Okay, sure, that sounds good, but the commission is still a big deal to me.",
            specificity=0.25,
            agreement=0.60,
            recurrence_id="rec_fee_d",
        )
        s4 = manager.process_turn_bundle(t4)

        # Negative Boundary Invariant (Turn 4): Objection remains unresolved, blocking conversion gate
        comm_obj = next((o for o in s4.objections if o.canonical_category == "commission_fee"), None)
        assert comm_obj is not None
        assert comm_obj.lifecycle_state in ("unresolved", "reactivated")
        assert s4.conversion_gate.is_open is False
        assert "objections_resolved_or_partial" in s4.conversion_gate.failed_conditions
        assert s4.push_strength.state == "resolve_then_ask"

        # Turn 5: Agent presents concrete proof comps
        t5 = _create_turn_bundle(
            turn_id=5,
            speaker_id="salesperson",
            text="Fair enough John. Let me bring the actual closed HUD comps on your street to verify the net proceeds.",
            salesperson_strategy_tag="hyperlocal_comps",
        )
        s5 = manager.process_turn_bundle(t5)

        # Turn 6: Prospect shows genuine forward advance and concrete resolution commitment
        t6 = _create_turn_bundle(
            turn_id=6,
            speaker_id="client",
            text="Alright, that sounds fair. Show me those closed comps on Thursday at three.",
            agreement=0.85,
            future_lang=0.75,
            readiness=0.82,
            specificity=0.85,
        )
        s6 = manager.process_turn_bundle(t6)

        # Positive Boundary Invariant (Turn 6): Legitimate forward commitment DOES resolve objection
        comm_obj_resolved = next((o for o in s6.objections if o.canonical_category == "commission_fee"), None)
        assert comm_obj_resolved is not None
        assert comm_obj_resolved.lifecycle_state == "resolved"
        assert comm_obj_resolved.resolution_evidence is not None
        assert "objections_resolved_or_partial" not in s6.conversion_gate.failed_conditions
        assert s6.conversion_gate.is_open is True
        assert s6.push_strength.state in ("confirm_and_protect", "direct_ask")

    # =========================================================================
    # 13. Spec Doc Acceptance Test E: Absent Decision-Maker Caps Readiness
    # =========================================================================
    def test_spec_acceptance_absent_decision_maker_caps_readiness_despite_enthusiasm(self):
        """Spec Test E: Enthusiastic prospect whose co-decision-maker is absent
        has readiness hard-capped at 55.0. Conversion gate remains closed.
        """
        manager = ConversationStateManager(call_sid="CA_p10_spec_test_e")

        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="I love your approach! But my husband is not here and we decide everything together.",
            trust=0.90,
            engagement=0.90,
            momentum=0.85,
            readiness=0.85,  # raw readiness high, but must be capped by blocker
        )
        # Supply decision structure update indicating missing decision maker
        decision_payload = {
            "all_decision_makers_identified": False,
            "decision_maker_present": False,
            "required_approvers": ["husband", "wife"],
            "decision_maker_role": "joint",
        }
        s1 = manager.process_turn_bundle(t1, decision_updates=decision_payload)

        # Invariant A: Deterministic blocker cap actively fires (unlike Spec Test A)
        assert "absent_decision_maker" in s1.readiness.active_blocker_caps
        assert "Absent/unconfirmed decision maker" in s1.readiness.capped_reason

        # Invariant B: Uncapped readiness was high (> 60.0), but is strictly bounded to <= 55.0
        assert s1.readiness.uncapped_score > 60.0
        assert s1.readiness.readiness_score <= 55.0
        assert s1.readiness.readiness_score < s1.readiness.uncapped_score

        # Invariant C: Gate is closed specifically due to missing decision authority, NOT lack of value
        assert s1.conversion_gate.is_open is False
        assert "decision_maker_aligned" in s1.conversion_gate.failed_conditions
        assert "clear_value_reason" not in s1.conversion_gate.failed_conditions
        assert s1.push_strength.state in ("two_window_choice", "protect_and_shorten")
