from __future__ import annotations

import pytest
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStakeholder,
    ObjectionRecord,
)
from copilot.conversation_conversion import MeetingConversionGateEngine
from copilot.conversation_conversion_config import ConversionBlockingConfig


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
    call_sid: str = "CA_target_blocking_test",
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
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
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


def _setup_baseline_passing_manager(call_sid: str, target: str = "appointment") -> ConversationStateManager:
    """Sets up a state manager with clean authority and property facts."""
    manager = ConversationStateManager(call_sid=call_sid, conversion_target=target)
    t1 = _create_turn_bundle(
        turn_id=1,
        speaker_id="client",
        text="I own the property on 742 Evergreen Terrace and I make the listing decisions.",
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
            {"category": "property", "fact_key": "property_address", "fact_value": "742 Evergreen Terrace"},
            {"category": "timeline", "fact_key": "target_closing", "fact_value": "Spring market target"},
        ],
    )
    return manager


# =============================================================================
# Addition 1 & 2: Target-Aware Objection Blocking
# =============================================================================

def test_unresolved_commission_does_not_block_appointment_gate():
    """Client Requirement: An unresolved commission objection should NOT block scheduling an appointment,
    because the meeting is where pricing is resolved.
    """
    manager = _setup_baseline_passing_manager("CA_block_appt", target="appointment")

    # Add active unresolved commission objection
    manager.current_state.objections.append(
        ObjectionRecord(
            canonical_category="commission_fee",
            initial_statement="Your 6% commission is too high.",
            latest_statement="Still not happy with that fee.",
            lifecycle_state="unresolved",
            first_turn_id=1,
            last_updated_turn_id=2,
        )
    )

    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="I still think that fee is high, but let's meet Thursday to talk through it.",
        trust=0.75,
        emotion_tension=0.15,
        engagement=0.75,
        agreement=0.70,
        specificity=0.70,
        future_lang=0.70,
    )
    snap = manager.process_turn_bundle(t2)

    cond3 = next(c for c in snap.conversion_gate.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3.met is True, f"Condition 3 unexpectedly failed: {cond3.reason}"
    assert "No blocking objections for 'appointment'" in cond3.reason
    assert "commission_fee" in cond3.reason

    # The gate is fully OPEN for appointment!
    assert snap.conversion_gate.is_open is True
    assert snap.conversion_gate.status == "open"
    assert snap.conversion_gate.failed_conditions == []
    # Push strength enters confirm_and_protect or direct_ask because commitment was made
    assert snap.push_strength.state in ("confirm_and_protect", "direct_ask")


def test_unresolved_commission_blocks_signed_listing_agreement_gate():
    """Client Requirement: That SAME unresolved commission objection MUST block signing a listing agreement."""
    manager = _setup_baseline_passing_manager("CA_block_listing", target="signed_listing_agreement")

    manager.current_state.objections.append(
        ObjectionRecord(
            canonical_category="commission_fee",
            initial_statement="Your 6% commission is too high.",
            latest_statement="Still not happy with that fee.",
            lifecycle_state="unresolved",
            first_turn_id=1,
            last_updated_turn_id=2,
        )
    )

    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="I like your track record, but that commission is still a real sticking point.",
        trust=0.70,
        emotion_tension=0.20,
        engagement=0.75,
        agreement=0.55,
        specificity=0.50,
    )
    snap = manager.process_turn_bundle(t2)

    cond3 = next(c for c in snap.conversion_gate.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3.met is False, "Condition 3 should fail for signed_listing_agreement"
    assert "Active unresolved objection(s) blocking 'signed_listing_agreement': commission_fee" in cond3.reason
    assert snap.conversion_gate.is_open is False
    assert "objections_resolved_or_partial" in snap.conversion_gate.failed_conditions
    assert snap.push_strength.state == "resolve_then_ask"


def test_dynamic_mid_call_conversion_target_switch():
    """Verify that conversion_target can be switched dynamically mid-call via process_turn_bundle."""
    manager = _setup_baseline_passing_manager("CA_dynamic_switch", target="appointment")

    manager.current_state.objections.append(
        ObjectionRecord(
            canonical_category="pricing_value",
            initial_statement="I'm not sure your valuation is realistic.",
            latest_statement="Valuation still seems questionable.",
            lifecycle_state="unresolved",
            first_turn_id=1,
            last_updated_turn_id=2,
        )
    )

    # Turn 2: Targeting appointment -> pricing_value does NOT block appointment
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Sure, come see the place on Thursday.",
        trust=0.75,
        engagement=0.75,
        agreement=0.75,
        specificity=0.65,
    )
    s2 = manager.process_turn_bundle(t2, conversion_target="appointment")
    assert s2.conversion_gate.is_open is True
    assert s2.conversion_gate.conversion_target == "appointment"

    # Turn 3: Agent attempts to switch target to signed_listing_agreement -> pricing_value DOES block!
    t3 = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="I still have doubts about your pricing numbers though.",
        trust=0.70,
        engagement=0.75,
        agreement=0.45,
    )
    s3 = manager.process_turn_bundle(t3, conversion_target="signed_listing_agreement")
    assert s3.conversion_gate.is_open is False
    assert s3.conversion_gate.conversion_target == "signed_listing_agreement"
    assert "objections_resolved_or_partial" in s3.conversion_gate.failed_conditions


def test_boundary_objection_blocks_all_targets():
    """A hard boundary objection blocks appointment, signed_listing_agreement, and permission_to_follow_up."""
    for target in ["appointment", "signed_listing_agreement", "permission_to_follow_up"]:
        engine = MeetingConversionGateEngine()
        snap = ConversationStateSnapshot(
            call_sid=f"CA_bnd_{target}",
            objections=[
                ObjectionRecord(
                    canonical_category="boundary",
                    initial_statement="Do not call this number again.",
                    latest_statement="Do not call this number again.",
                    lifecycle_state="boundary",
                    first_turn_id=1,
                    last_updated_turn_id=1,
                )
            ],
        )
        bundle = _create_turn_bundle(1, "client", "Do not contact me.")
        gate = engine.evaluate_gate(bundle, snap, conversion_target=target)
        assert gate.is_open is False
        assert "objections_resolved_or_partial" in gate.failed_conditions
        assert "boundary" in gate.failed_conditions or "no_active_boundary" in gate.failed_conditions


# =============================================================================
# Addition 3: Explicit Commitment Override
# =============================================================================

def test_explicit_commitment_overrides_low_inferred_readiness_and_value():
    """Client Requirement: 'Thursday at 4 works' is unambiguous direct evidence.
    It outranks computed/inferred numbers (e.g. low logical_readiness, low value recognition)
    and forces the relevant conditions to pass deterministically.
    """
    manager = _setup_baseline_passing_manager("CA_explicit_commit_01", target="appointment")

    # Turn 2: Prospect gives direct, explicit slot commitment
    # But upstream inferred readiness and agreement numbers were low or artificially depressed
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me, let's do it.",
        trust=0.75,
        emotion_tension=0.15,
        readiness=0.30,      # Inferred score is low
        engagement=0.70,
        agreement=0.55,
        specificity=0.85,
        future_lang=0.80,
    )

    # Force momentum value recognition to be below threshold (e.g. 30.0 < 50.0)
    manager.current_state.momentum = None

    snap = manager.process_turn_bundle(t2)
    gate = snap.conversion_gate

    assert gate.explicit_commitment_detected is True
    assert gate.commitment_slot.lower() == "thursday at 4"

    # Condition 4 (clear_value_reason) should be overridden
    cond4 = next(c for c in gate.conditions if c.condition_name == "clear_value_reason")
    assert cond4.met is True
    assert cond4.is_overridden is True
    assert "[OVERRIDE: Explicit commitment detected" in cond4.reason
    assert "thursday at 4" in cond4.reason.lower()

    # Condition 6 (plausible_logistics) should be satisfied
    cond6 = next(c for c in gate.conditions if c.condition_name == "plausible_logistics")
    assert cond6.met is True

    # Entire gate is OPEN
    assert gate.is_open is True
    assert gate.status == "open"
    # Push strength recognizes concrete commitment (confirm_and_protect)
    assert snap.push_strength.state in ("confirm_and_protect", "direct_ask")
    assert "thursday at 4" in (snap.push_strength.rationale or "").lower() or "thursday at 4" in (snap.push_strength.recommended_action or "").lower()


def test_salesperson_proposal_accepted_by_prospect_triggers_override():
    """Salesperson proposes 'Would Thursday at 4:00 PM work?' and prospect confirms 'Yes, that works'."""
    manager = _setup_baseline_passing_manager("CA_proposal_accept", target="appointment")

    # Turn 2: Salesperson proposes slot
    t2_agent = _create_turn_bundle(
        turn_id=2,
        speaker_id="salesperson",
        text="Would Thursday at 4:00 PM work for you to walk the property?",
        salesperson_strategy_tag="specific_slot_proposal",
    )
    manager.process_turn_bundle(t2_agent)

    # Turn 3: Prospect affirms
    t3_client = _create_turn_bundle(
        turn_id=3,
        speaker_id="client",
        text="Yes, that works.",
        trust=0.72,
        emotion_tension=0.15,
        engagement=0.70,
        agreement=0.85,
        readiness=0.35,  # Low inferred readiness
        specificity=0.40,
    )
    snap = manager.process_turn_bundle(t3_client)
    gate = snap.conversion_gate

    assert gate.explicit_commitment_detected is True
    assert "thursday at 4:00" in gate.commitment_slot.lower()

    cond4 = next(c for c in gate.conditions if c.condition_name == "clear_value_reason")
    assert cond4.met is True
    assert cond4.is_overridden is True
    assert gate.is_open is True


def test_vague_filler_does_not_trigger_override():
    """Polite filler ('Yeah sure') without concrete slot or proposal acceptance does NOT trigger override."""
    manager = _setup_baseline_passing_manager("CA_vague_filler", target="appointment")
    # Clear timeline facts so vague filler guard activates
    manager.current_state.facts = [f for f in manager.current_state.facts if f.category != "timeline"]
    manager.facts_manager.facts = manager.current_state.facts

    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Yeah, okay, sure.",
        trust=0.65,
        engagement=0.60,
        agreement=0.40,  # Below threshold
        specificity=0.20,  # Low specificity
        future_lang=0.10,
        readiness=0.35,
    )
    snap = manager.process_turn_bundle(t2)
    gate = snap.conversion_gate

    assert gate.explicit_commitment_detected is False
    assert gate.commitment_slot is None
    # Gate does NOT open (Condition 4 fails)
    assert gate.is_open is False
    assert "clear_value_reason" in gate.failed_conditions


def test_explicit_commitment_does_not_override_hard_boundary():
    """Deterministic Safety Override: Explicit commitment language cannot override a hard compliance boundary."""
    manager = _setup_baseline_passing_manager("CA_boundary_safety", target="appointment")

    # Hard boundary active
    manager.current_state.contact_compliance.hard_boundary_active = True

    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works, but do not ever call my number again after this.",
        boundary=0.90,
        trust=0.40,
        agreement=0.60,
    )
    snap = manager.process_turn_bundle(t2)
    gate = snap.conversion_gate

    # Condition 7 fails
    cond7 = next(c for c in gate.conditions if c.condition_name == "no_active_boundary")
    assert cond7.met is False
    assert gate.is_open is False
    assert snap.push_strength.state == "respect_record_exit"


# =============================================================================
# Direct Verification: Exact Client Scenario & Adversarial Rigor
# =============================================================================

def test_client_exact_scenario_commission_objection_with_thursday_at_4_pm():
    """Direct verification of client's exact scenario by name:
    'Your commission is still way too expensive, but I could do Thursday at 4 PM to see the numbers.'
    Under conversion_target='appointment': Active unresolved commission_fee does NOT block the gate. Gate is OPEN.
    Under conversion_target='signed_listing_agreement': The exact same objection DOES block the gate. Gate is CLOSED.
    """
    # -------------------------------------------------------------------------
    # Scenario A: Target = 'appointment'
    # -------------------------------------------------------------------------
    manager_appt = _setup_baseline_passing_manager("CA_exact_client_appt", target="appointment")
    t_appt = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Your commission is still way too expensive, but I could do Thursday at 4 PM to see the numbers.",
        trust=0.72,
        emotion_tension=0.20,
        engagement=0.75,
        agreement=0.65,
        specificity=0.80,
        future_lang=0.75,
    )
    s_appt = manager_appt.process_turn_bundle(t_appt)
    gate_appt = s_appt.conversion_gate

    # 1. Commission objection was accurately registered and remains active/unresolved
    comm_obj_appt = next((o for o in s_appt.objections if o.canonical_category == "commission_fee"), None)
    assert comm_obj_appt is not None, "Commission objection must be registered from utterance"
    assert comm_obj_appt.lifecycle_state in ("unresolved", "reactivated")

    # 2. Gate Condition 3 passes because commission_fee does NOT block appointment
    cond3_appt = next(c for c in gate_appt.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3_appt.met is True
    assert "No blocking objections for 'appointment'" in cond3_appt.reason
    assert "commission_fee" in cond3_appt.reason

    # 3. Explicit commitment override detected the slot
    assert gate_appt.explicit_commitment_detected is True
    assert "thursday at 4" in gate_appt.commitment_slot.lower()

    # 4. Entire gate is OPEN and push strength is confirm_and_protect or direct_ask
    assert gate_appt.is_open is True
    assert gate_appt.status == "open"
    assert s_appt.push_strength.state in ("confirm_and_protect", "direct_ask")

    # -------------------------------------------------------------------------
    # Scenario B: Target = 'signed_listing_agreement'
    # -------------------------------------------------------------------------
    manager_listing = _setup_baseline_passing_manager("CA_exact_client_listing", target="signed_listing_agreement")
    t_listing = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Your commission is still way too expensive, but I could do Thursday at 4 PM to see the numbers.",
        trust=0.72,
        emotion_tension=0.20,
        engagement=0.75,
        agreement=0.65,
        specificity=0.80,
        future_lang=0.75,
    )
    s_listing = manager_listing.process_turn_bundle(t_listing)
    gate_listing = s_listing.conversion_gate

    # 1. Commission objection is registered and active
    comm_obj_list = next((o for o in s_listing.objections if o.canonical_category == "commission_fee"), None)
    assert comm_obj_list is not None
    assert comm_obj_list.lifecycle_state in ("unresolved", "reactivated")

    # 2. Gate Condition 3 strictly FAILS for signed_listing_agreement
    cond3_listing = next(c for c in gate_listing.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3_listing.met is False
    assert "Active unresolved objection(s) blocking 'signed_listing_agreement': commission_fee" in cond3_listing.reason

    # 3. Entire gate remains CLOSED and push strength is resolve_then_ask
    assert gate_listing.is_open is False
    assert gate_listing.status == "closed"
    assert "objections_resolved_or_partial" in gate_listing.failed_conditions
    assert s_listing.push_strength.state == "resolve_then_ask"


def test_adversarial_hypothetical_temporal_does_not_trigger_override():
    """Adversarial negative test:
    'Thursday at 4 usually doesn't work for me, but let's say hypothetically it did'
    Must NOT trigger explicit commitment override.
    """
    manager = _setup_baseline_passing_manager("CA_adversarial_neg", target="appointment")
    # Clear goal facts so gate cannot pass via passive fallback
    manager.current_state.facts = [f for f in manager.current_state.facts if f.category != "timeline"]
    manager.facts_manager.facts = manager.current_state.facts

    engine = MeetingConversionGateEngine()
    bundle = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 usually doesn't work for me, but let's say hypothetically it did",
        trust=0.60,
        emotion_tension=0.25,
        agreement=0.45,
        specificity=0.70,
        future_lang=0.60,
    )

    # 1. Engine detection method directly returns False
    has_commit, slot = engine.detect_explicit_commitment(bundle, manager.current_state)
    assert has_commit is False, "Hypothetical negation must NOT be detected as explicit commitment"
    assert slot is None

    # 2. Gate evaluation does not trigger override
    snap = manager.process_turn_bundle(bundle)
    gate = snap.conversion_gate

    assert gate.explicit_commitment_detected is False
    assert gate.commitment_slot is None
    assert gate.is_open is False


def test_override_is_scoped_per_condition_absent_decision_maker_blocks():
    """Per-condition isolation test:
    Prospect explicitly commits to 'Thursday at 4 works', but decision maker (spouse) is absent.
    Condition 4 clear_value_reason passes via override, but Condition 5 decision_maker_aligned FAILS.
    Gate remains strictly CLOSED.
    """
    manager = _setup_baseline_passing_manager("CA_scoped_absent_dm", target="appointment")

    # Set spouse absent in decision structure
    manager.current_state.decision_structure.decision_maker_present = False
    manager.current_state.decision_structure.stakeholders = [
        DecisionStakeholder(name="Wife Mary", role="spouse", presence="absent")
    ]

    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me, let's meet then.",
        trust=0.75,
        engagement=0.75,
        agreement=0.70,
        specificity=0.80,
        future_lang=0.75,
    )
    snap = manager.process_turn_bundle(t2)
    gate = snap.conversion_gate

    # Explicit commitment detected
    assert gate.explicit_commitment_detected is True

    # Condition 4 (clear_value_reason) passed via override
    cond4 = next(c for c in gate.conditions if c.condition_name == "clear_value_reason")
    assert cond4.met is True
    assert cond4.is_overridden is True

    # Condition 5 (decision_maker_aligned) strictly FAILED (NOT swept along by scheduling commitment)
    cond5 = next(c for c in gate.conditions if c.condition_name == "decision_maker_aligned")
    assert cond5.met is False
    assert "Decision-maker absent" in cond5.reason

    # Overall Gate remains CLOSED
    assert gate.is_open is False
    assert "decision_maker_aligned" in gate.failed_conditions


def test_override_is_scoped_per_condition_access_constraints_blocks():
    """Per-condition isolation test:
    Prospect explicitly commits to a slot, but property has unresolved access constraints
    (e.g., tenant occupied requires 48hr notice). Condition 6 plausible_logistics strictly FAILS.
    Gate remains CLOSED.
    """
    manager = _setup_baseline_passing_manager("CA_scoped_logistics", target="appointment")

    # Property has active access constraints
    manager.current_state.decision_structure.access_constraints = [
        "tenant_occupied_requires_48hr_written_notice"
    ]

    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Thursday at 4 works for me, let's meet then.",
        trust=0.75,
        engagement=0.75,
        agreement=0.70,
        specificity=0.80,
        future_lang=0.75,
    )
    snap = manager.process_turn_bundle(t2)
    gate = snap.conversion_gate

    assert gate.explicit_commitment_detected is True

    # Condition 6 (plausible_logistics) strictly FAILED due to access constraints
    cond6 = next(c for c in gate.conditions if c.condition_name == "plausible_logistics")
    assert cond6.met is False
    assert cond6.is_overridden is False
    assert "Access constraints" in cond6.reason

    # Overall Gate remains CLOSED
    assert gate.is_open is False
    assert "plausible_logistics" in gate.failed_conditions


def test_point2_advance_via_blocker_clearance_and_regress_via_new_evidence():
    """Client Point 2 End-to-End Stress Test:
    Demonstrates both directions of dynamic gating:
    1. ADVANCEMENT: Gate starts CLOSED due to an active target-blocking objection ('commission_fee' for 'signed_listing_agreement').
       Upon salesperson reframe and prospect concession, the objection resolves and the gate ADVANCES from CLOSED to OPEN.
    2. REGRESSION: Later in the conversation, new evidence surfaces (absent spouse on deed), causing decision_maker_aligned
       to fail and the gate REGRESSES from OPEN to CLOSED.
    Both transitions are verified with full audit history in change_history.
    """
    from copilot.conversation_state_models import DecisionStakeholder, ConversionEventStatus, ObjectionLifecycleState

    manager = ConversationStateManager(call_sid="CA_point2_stress_test", conversion_target="signed_listing_agreement")

    # Turn 1: Salesperson Opening
    t1 = _create_turn_bundle(
        turn_id=1,
        speaker_id="salesperson",
        text="Hi, thanks for meeting today — are you ready to finalize the listing agreement?",
    )
    s1 = manager.process_turn_bundle(t1)

    # Turn 2: Client raises blocking commission objection
    t2 = _create_turn_bundle(
        turn_id=2,
        speaker_id="client",
        text="Your commission rate is 6%, which is way too high. I'm not signing any agreement at that fee.",
        trust=0.70,
        emotion_tension=0.20,
        engagement=0.75,
        agreement=0.40,
    )
    s2 = manager.process_turn_bundle(
        t2,
        decision_updates={
            "primary_decision_maker": "Self (Owner)",
            "decision_maker_present": True,
            "stakeholders": [DecisionStakeholder(name="Self", role="owner", presence="on_call")],
        },
    )
    gate2 = s2.conversion_gate

    # 1. Commission fee objection is active and blocks signed_listing_agreement
    assert gate2.is_open is False
    assert "objections_resolved_or_partial" in gate2.failed_conditions
    cond3_t2 = next(c for c in gate2.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3_t2.met is False
    assert "commission_fee" in cond3_t2.reason

    # Turn 3: Salesperson reframe
    t3 = _create_turn_bundle(
        turn_id=3,
        speaker_id="salesperson",
        text="Let's walk through your net proceeds after all marketing and staging costs, so you can see the real return you'll walk away with.",
        salesperson_strategy_tag="financial_net_proceeds_reframe",
    )
    s3 = manager.process_turn_bundle(t3)

    # Turn 4: Prospect concession clears the blocking objection
    t4 = _create_turn_bundle(
        turn_id=4,
        speaker_id="client",
        text="Okay, that breakdown makes sense, commission isn't really the issue then. That's fair.",
        trust=0.75,
        emotion_tension=0.15,
        engagement=0.80,
        agreement=0.70,
    )
    s4 = manager.process_turn_bundle(t4)
    gate4 = s4.conversion_gate

    # 2. Objection is resolved and Gate ADVANCES from CLOSED to OPEN
    comm_obj = next((o for o in s4.objections if o.canonical_category == "commission_fee"), None)
    assert comm_obj is not None
    assert comm_obj.lifecycle_state in ("resolved", ObjectionLifecycleState.RESOLVED)

    cond3_t4 = next(c for c in gate4.conditions if c.condition_name == "objections_resolved_or_partial")
    assert cond3_t4.met is True
    assert gate4.is_open is True
    assert len(gate4.failed_conditions) == 0

    # Verify audit change history recorded the advance
    gate_open_change = next((c for c in s4.change_history if c.field_path == "conversion_gate" and c.triggering_turn_id == 4), None)
    assert gate_open_change is not None
    assert "transitioned to OPEN" in gate_open_change.reason

    # Turn 5: Salesperson proposes signing appointment
    t5 = _create_turn_bundle(
        turn_id=5,
        speaker_id="salesperson",
        text="Great — would Thursday work for you to sign the listing agreement?",
        salesperson_strategy_tag="specific_slot_proposal",
    )
    s5 = manager.process_turn_bundle(t5)

    # Turn 6: Client confirms
    t6 = _create_turn_bundle(
        turn_id=6,
        speaker_id="client",
        text="Yes, Thursday at 3 works, let's do it.",
        trust=0.80,
        engagement=0.85,
        agreement=0.80,
    )
    s6 = manager.process_turn_bundle(t6)
    gate6 = s6.conversion_gate
    assert gate6.is_open is True
    assert gate6.explicit_commitment_detected is True

    # Turn 7: Client discloses absent spouse on deed -> Gate REGRESSES from OPEN to CLOSED
    t7 = _create_turn_bundle(
        turn_id=7,
        speaker_id="client",
        text="Actually wait, my wife is on the deed and she would need to be part of this conversation before we sign anything.",
        trust=0.75,
        engagement=0.80,
    )
    s7 = manager.process_turn_bundle(t7)
    gate7 = s7.conversion_gate

    assert gate7.is_open is False
    assert "decision_maker_aligned" in gate7.failed_conditions
    cond5_t7 = next(c for c in gate7.conditions if c.condition_name == "decision_maker_aligned")
    assert cond5_t7.met is False

    # Verify audit change history recorded the regression
    gate_close_change = next((c for c in s7.change_history if c.field_path == "conversion_gate" and c.triggering_turn_id == 7), None)
    assert gate_close_change is not None
    assert "transitioned to CLOSED" in gate_close_change.reason
    assert "decision_maker_aligned" in gate_close_change.reason

