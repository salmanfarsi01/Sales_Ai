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
    ObjectionLifecycleState,
    DealDispositionType,
    DealDispositionRecord,
)


def _make_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    agreement: float = 0.0,
    readiness: float = 0.5,
    boundary: float = 0.0,
    question_type: str = "none",
    recurrence_type: str = "none",
    recurrence_id: str = None,
    future_language: float = 0.0,
    strategy_tag: str = None,
):
    inference = DownstreamInferenceState(
        call_sid="CA_lifecycle_disposition_test",
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=0.80, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.2, confidence=0.8),
        pacing=DimensionScore(score=0.60, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.65, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_turn_{turn_id}"],
    )
    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid="CA_lifecycle_disposition_test",
        speaker_id=speaker_id,
        question_type=question_type,
        agreement_score=agreement,
        boundary_score=boundary,
        future_language_score=future_language,
        recurrence_type=recurrence_type,
        recurrence_id=recurrence_id,
    )
    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
        salesperson_strategy_tag=strategy_tag,
    )


def test_independent_concurrent_concerns_no_collision_or_false_supersession():
    """Client Item No. 3 & No. 4:

    Verify market timing / hesitation ('not sure this is the right time')
    and spousal authority ('my wife needs to be part of this') remain tracked independently.
    Spouse statement routes to DecisionStructure and facts, NOT an objection.
    """
    manager = ConversationStateManager(call_sid="CA_concurrent_concerns")

    # Turn 1: Timing/hesitation concern
    b1 = _make_bundle(1, "client", "Honestly, I'm just not sure this is the right time for us to sell.")
    s1 = manager.process_turn_bundle(b1)

    assert len(s1.objections) == 1
    hesitation_obj = s1.objections[0]
    assert hesitation_obj.canonical_category == "general_hesitation"
    assert hesitation_obj.lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 2: Agent responds respectfully
    b2 = _make_bundle(2, "salesperson", "Understood, timing is everything. What's making you feel uncertain?")
    s2 = manager.process_turn_bundle(b2)

    # Turn 3: Spousal involvement disclosure
    b3 = _make_bundle(3, "client", "Well, my wife really needs to be part of this conversation before we make any decisions.")
    s3 = manager.process_turn_bundle(b3)

    # 1. DecisionStructure populated
    assert s3.decision_structure.decision_maker_present is False
    assert len(s3.decision_structure.stakeholders) == 1
    assert s3.decision_structure.stakeholders[0].role == "wife"
    assert s3.decision_structure.stakeholders[0].presence == "absent"

    # 2. Fact recorded
    spouse_facts = [f for f in s3.facts if f.fact_key == "spouse_involvement" and f.status == "active"]
    assert len(spouse_facts) == 1

    # 3. NO spousal objection created!
    assert not any(o.canonical_category == "spouse_authority" for o in s3.objections)

    # 4. Turn 1 hesitation is NOT superseded! It remains active.
    assert len(s3.objections) == 1
    assert s3.objections[0].canonical_category == "general_hesitation"
    assert s3.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert s3.objections[0].superseded_by_objection_id is None


def test_dormancy_aging_after_three_unaddressed_turns_and_reactivation():
    """Client Item No. 3:

    Concerns that are not addressed or repeated for >= 3 turns age into DORMANT,
    preserving full psychological history. When the prospect repeats the concern,
    it reactivates to ACTIVE with an incremented recurrence count.
    """
    manager = ConversationStateManager(call_sid="CA_dormancy_lifecycle")

    # Turn 1: Commission fee objection
    b1 = _make_bundle(1, "client", "Your commission fee is 6 percent, which feels way too expensive.")
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    fee_obj = s1.objections[0]
    assert fee_obj.lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert fee_obj.recurrence_count == 1

    # Turns 2, 3, 4: Dialogue moves to other topics without addressing or repeating fee
    b2 = _make_bundle(2, "salesperson", "Let's talk about the timeline you had in mind.")
    s2 = manager.process_turn_bundle(b2)
    assert s2.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE  # 1 turn elapsed

    b3 = _make_bundle(3, "client", "We were hoping to move by November.", future_language=0.55)
    s3 = manager.process_turn_bundle(b3)
    assert s3.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE  # 2 turns elapsed

    b4 = _make_bundle(4, "salesperson", "November gives us about two months to prep the home.")
    s4 = manager.process_turn_bundle(b4)

    # At Turn 4 (3 turns elapsed since Turn 1: turns 2, 3, 4), objection ages to DORMANT!
    assert s4.objections[0].lifecycle_state == ObjectionLifecycleState.DORMANT
    assert len(s4.get_dormant_objections()) == 1
    assert len(s4.get_active_objections()) == 0

    # Turn 5: Discussion continues on prep
    b5 = _make_bundle(5, "salesperson", "We could start with minor paint touchups.")
    s5 = manager.process_turn_bundle(b5)
    assert s5.objections[0].lifecycle_state == ObjectionLifecycleState.DORMANT

    # Turn 6: Prospect brings the fee back up -> REACTIVATES to ACTIVE!
    b6 = _make_bundle(6, "client", "Look, paint is fine, but that 6 percent fee is still a big problem for us.")
    s6 = manager.process_turn_bundle(b6)

    reactivated_obj = s6.objections[0]
    assert reactivated_obj.lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert reactivated_obj.recurrence_count == 2
    assert len(s6.get_active_objections()) == 1
    assert len(s6.get_dormant_objections()) == 0


def test_pure_decision_authority_statements_never_spawn_objections():
    """Client Item No. 4:

    Pure decision-authority disclosures (spouse, partner, co-owner, attorney)
    must route exclusively to DecisionStructure and facts, and NEVER create an ObjectionRecord.
    """
    test_cases = [
        ("I need to speak with my husband before committing.", "husband"),
        ("My business partner makes all financial decisions.", "partner"),
        ("Our attorney must review everything first.", "attorney"),
        ("My wife and I have to decide together.", "wife"),
    ]

    for idx, (utterance, expected_role) in enumerate(test_cases, start=1):
        manager = ConversationStateManager(call_sid=f"CA_auth_test_{idx}")
        bundle = _make_bundle(1, "client", utterance)
        snapshot = manager.process_turn_bundle(bundle)

        # 1. No objections created
        assert len(snapshot.objections) == 0, f"Expected 0 objections for '{utterance}', got {snapshot.objections}"

        # 2. Decision structure has absent decision maker
        assert snapshot.decision_structure.decision_maker_present is False
        assert len(snapshot.decision_structure.stakeholders) == 1
        assert snapshot.decision_structure.stakeholders[0].role == expected_role
        assert snapshot.decision_structure.stakeholders[0].presence == "absent"

        # 3. Persistent fact recorded
        assert any(f.fact_key == "spouse_involvement" for f in snapshot.facts)


def test_deal_disposition_lifecycle_and_non_destructive_lineage():
    """Client Item No. 4 Part 2:

    Deal disposition is a first-class tracked concept on ConversationStateSnapshot:
    - Default is ACTIVELY_SELLING.
    - 'Decided to stay / not selling' transitions to DECLINED.
    - Causes Gate Condition 4 (clear_value_reason) to fail and push strength to respect_record_exit.
    - Subsequent conditional reopening ('would still move if...') transitions to RECONSIDERING.
    - All transitions maintain non-destructive lineage via superseded_by_id and audit trail.
    """
    manager = ConversationStateManager(call_sid="CA_deal_disp_test")

    # Initial state
    assert manager.deal_disposition.disposition == DealDispositionType.ACTIVELY_SELLING
    assert len(manager.get_deal_disposition_history()) == 1

    # Turn 1: Normal dialogue
    b1 = _make_bundle(1, "client", "We're thinking of selling our home this fall.")
    s1 = manager.process_turn_bundle(b1)
    assert s1.deal_disposition.disposition == DealDispositionType.ACTIVELY_SELLING

    # Turn 2: Prospect decides to stay
    b2 = _make_bundle(2, "client", "Actually, we talked it over and we've decided to stay in the house. We're not selling anymore.")
    s2 = manager.process_turn_bundle(b2)

    # Verify DealDispositionRecord is DECLINED
    assert s2.deal_disposition.disposition == DealDispositionType.DECLINED
    assert "decided to stay" in s2.deal_disposition.rationale.lower()
    disp_history = manager.get_deal_disposition_history()
    assert len(disp_history) == 2
    # Prior record was superseded
    assert disp_history[0].superseded_by_id == s2.deal_disposition.disposition_id

    # Gate Condition 4 must fail
    assert s2.conversion_gate.is_open is False
    cond4 = next(c for c in s2.conversion_gate.conditions if c.condition_name == "clear_value_reason")
    assert cond4.met is False
    assert "decided to stay and not sell" in cond4.reason

    # Push strength must be respect_record_exit
    assert s2.push_strength.state == "respect_record_exit"

    # Turn 3: Prospect conditionally reconsiders
    b3 = _make_bundle(3, "client", "Well, we would still move if we got an offer over 650k, but otherwise staying.")
    s3 = manager.process_turn_bundle(b3)

    assert s3.deal_disposition.disposition == DealDispositionType.RECONSIDERING
    assert len(manager.get_deal_disposition_history()) == 3
    # Declined record was superseded by reconsidering record
    assert disp_history[1].superseded_by_id == s3.deal_disposition.disposition_id


def test_evidence_gated_objection_supersession_requires_semantic_evidence():
    """Client Item No. 3:

    SUPERSEDED is strictly gated behind relation classification evidence:
    Unrelated statements never supersede active concerns.
    Explicit withdrawal/reversal statements cleanly supersede active concerns.
    """
    manager = ConversationStateManager(call_sid="CA_gated_supersession")

    # Turn 1: Commission fee objection
    b1 = _make_bundle(1, "client", "Your commission fee is too high.")
    s1 = manager.process_turn_bundle(b1)
    assert s1.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 2: Unrelated client question
    b2 = _make_bundle(2, "client", "What's the best email to reach you at?")
    s2 = manager.process_turn_bundle(b2)
    # Objection MUST remain active (not superseded)
    assert s2.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 3: Explicit reversal of fee concern
    b3 = _make_bundle(3, "client", "You know what, we are not worried about the fee anymore, the rate is fine.")
    s3 = manager.process_turn_bundle(b3)

    # Now objection is superseded with clear evidence
    assert s3.objections[0].lifecycle_state == ObjectionLifecycleState.SUPERSEDED
    assert "Commission fee concern withdrawn" in s3.objections[0].resolution_evidence


def test_topically_close_boundary_concerns_merge_as_recurrence_not_duplicate_or_supersession():
    """Adversarial Boundary Test:
    
    Verifies how the system handles two topically close but non-identical concerns:
    Turn 1: "The 6% fee is steep."
    Turn 2: "I don't like how much of the sale price goes to fees."
    
    Expected behavior:
    - Both map to the canonical concern category 'commission_fee'.
    - Turn 2 recognizes the existing active objection rather than spawning a duplicate record.
    - Recurrence count increments (1 -> 2), latest statement updates.
    - Neither concern is falsely superseded because neither reverses or replaces the other.
    - Total active objections remains exactly 1.
    - A subsequent genuinely distinct concern (e.g. market timing) tracks separately as a 2nd objection.
    """
    manager = ConversationStateManager(call_sid="CA_topically_close_boundary")

    # Turn 1: 6% fee concern
    b1 = _make_bundle(1, "client", "The 6% fee is steep.")
    s1 = manager.process_turn_bundle(b1)

    assert len(s1.objections) == 1
    obj1 = s1.objections[0]
    assert obj1.canonical_category == "commission_fee"
    assert obj1.lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert obj1.recurrence_count == 1
    assert obj1.superseded_by_objection_id is None
    original_id = obj1.objection_id

    # Turn 2: Topically close rephrased concern ("sale price goes to fees")
    b2 = _make_bundle(2, "client", "I don't like how much of the sale price goes to fees.")
    s2 = manager.process_turn_bundle(b2)

    # Must NOT spawn a duplicate objection record
    assert len(s2.objections) == 1
    obj2 = s2.objections[0]
    # Retains stable identity and canonical taxonomy
    assert obj2.objection_id == original_id
    assert obj2.canonical_category == "commission_fee"
    assert obj2.lifecycle_state == ObjectionLifecycleState.ACTIVE
    # Recurrence incremented
    assert obj2.recurrence_count == 2
    assert "sale price goes to fees" in obj2.latest_statement
    # Crucial: Not superseded!
    assert obj2.superseded_by_objection_id is None

    # Turn 3: Genuinely distinct concern (market timing)
    b3 = _make_bundle(3, "client", "I don't think the market timing is right.")
    s3 = manager.process_turn_bundle(b3)

    # Now there should be exactly 2 distinct tracked objections, both ACTIVE
    assert len(s3.objections) == 2
    cat_to_obj = {o.canonical_category: o for o in s3.objections}
    assert "commission_fee" in cat_to_obj
    assert "market_timing" in cat_to_obj
    assert cat_to_obj["commission_fee"].lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert cat_to_obj["market_timing"].lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert cat_to_obj["commission_fee"].superseded_by_objection_id is None
    assert cat_to_obj["market_timing"].superseded_by_objection_id is None


def test_reconsidering_deal_disposition_has_distinct_operational_push_strength():
    """Verifies that RECONSIDERING deal disposition triggers a distinct operational
    push strength ('explore_conditional_terms') rather than pushing forward normally (direct_ask)
    or shutting down completely (respect_record_exit).
    """
    manager = ConversationStateManager(call_sid="CA_reconsidering_push")

    # Turn 1: Client declines / decides to stay
    b1 = _make_bundle(1, "client", "We have decided to stay put and take the house off the market.")
    s1 = manager.process_turn_bundle(b1)

    assert s1.deal_disposition.disposition == DealDispositionType.DECLINED
    assert s1.conversion_gate.is_open is False
    assert s1.push_strength.state == "respect_record_exit"

    # Turn 2: Client conditionally reconsiders (Client's exact reported phrase)
    b2 = _make_bundle(2, "client", "I'd still move if I believed there was a better strategy.")
    s2 = manager.process_turn_bundle(b2)

    assert s2.deal_disposition.disposition == DealDispositionType.RECONSIDERING
    # Distinct operational push strength state:
    assert s2.push_strength.state == "explore_conditional_terms"
    assert "reconsidering" in s2.push_strength.rationale.lower()
    assert "criteria" in s2.push_strength.recommended_action.lower() or "strategy" in s2.push_strength.recommended_action.lower()

    # Condition 4 does NOT fail on 'decision to stay' because decline is superseded
    cond4 = next(c for c in s2.conversion_gate.conditions if c.condition_name == "clear_value_reason")
    assert "decided to stay and not sell" not in cond4.reason


def test_dormancy_threshold_configurable_and_provisional():
    """Verifies that the dormancy aging window is fully configurable via ConversationScoringConfig
    and documented as a provisional v1 heuristic.
    """
    from copilot.conversation_scoring_config import ConversationScoringConfig

    # Configure custom 2-turn dormancy threshold
    cfg = ConversationScoringConfig(dormancy_turn_threshold=2)
    manager = ConversationStateManager(call_sid="CA_custom_dormancy", scoring_config=cfg)

    # Turn 1: Commission fee objection
    b1 = _make_bundle(1, "client", "Your commission fee is too high.")
    s1 = manager.process_turn_bundle(b1)
    assert s1.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 2: 1 turn elapsed
    b2 = _make_bundle(2, "client", "Can you send me your bio?", future_language=0.55)
    s2 = manager.process_turn_bundle(b2)
    assert s2.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 3: 2 turns elapsed -> Transitions to DORMANT with threshold=2!
    b3 = _make_bundle(3, "client", "What's the main office number?")
    s3 = manager.process_turn_bundle(b3)
    assert s3.objections[0].lifecycle_state == ObjectionLifecycleState.DORMANT


def test_objection_underlying_driver_layer_discrimination():
    """Client Review Feedback No. 5:
    Verify that objections sharing the canonical category 'commission_fee'
    are correctly discriminated at the underlying driver layer:
    1. 'I refuse to pay 5%' -> pure numerical price resistance
    2. 'The commission last time felt way too high for what we got' -> prior agent outcome disappointment
    3. 'I don't see why an agent deserves that much for putting a sign in the yard' -> perceived value deficit
    All retain stable canonical taxonomy ('commission_fee') but gain actionable strategic targets.
    """
    # 1. Pure Price Resistance
    mgr1 = ConversationStateManager(call_sid="CA_driver_test_1")
    b1 = _make_bundle(1, "client", "I refuse to pay 5%.")
    s1 = mgr1.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    obj1 = s1.objections[0]
    assert obj1.canonical_category == "commission_fee"
    assert obj1.driver_layer is not None
    assert obj1.driver_layer.surface_objection == "commission_fee"
    assert obj1.driver_layer.underlying_driver == "price_resistance"
    assert "justify_fee_structure" in obj1.driver_layer.strategic_target
    # Heuristic confidence discounted for fallback baseline
    assert obj1.driver_layer.confidence == 0.50
    assert obj1.driver_layer.classification_source == "heuristic_default"

    # 2. Previous Agent Disappointment
    mgr2 = ConversationStateManager(call_sid="CA_driver_test_2")
    b2 = _make_bundle(1, "client", "The commission last time felt way too high for what we got.")
    s2 = mgr2.process_turn_bundle(b2)
    assert len(s2.objections) == 1
    obj2 = s2.objections[0]
    assert obj2.canonical_category == "commission_fee"
    assert obj2.driver_layer is not None
    assert obj2.driver_layer.surface_objection == "commission_fee"
    assert obj2.driver_layer.underlying_driver == "previous_agent_outcome"
    assert obj2.driver_layer.origin_context == "referenced prior agent experience"
    assert "diagnose_prior_service_failures" in obj2.driver_layer.strategic_target
    # Heuristic pattern confidence
    assert obj2.driver_layer.confidence == 0.70
    assert obj2.driver_layer.classification_source == "heuristic_pattern"

    # 3. Perceived Value Deficit
    mgr3 = ConversationStateManager(call_sid="CA_driver_test_3")
    b3 = _make_bundle(1, "client", "I don't see why an agent deserves that much for putting a sign in the yard.")
    s3 = mgr3.process_turn_bundle(b3)
    assert len(s3.objections) == 1
    obj3 = s3.objections[0]
    assert obj3.canonical_category == "commission_fee"
    assert obj3.driver_layer is not None
    assert obj3.driver_layer.surface_objection == "commission_fee"
    assert obj3.driver_layer.underlying_driver == "perceived_value_deficit"
    assert "demonstrate_marketing_execution" in obj3.driver_layer.strategic_target
    assert obj3.driver_layer.confidence == 0.70
    assert obj3.driver_layer.classification_source == "heuristic_pattern"


def test_strategy_effectiveness_feedback_loop_and_adaptation():
    """Client Review Feedback No. 6:
    Verify the closed feedback loop between attempted salesperson strategies
    and subsequent prospect reactions:
    - Failed strategies (rejected due to repeated objection) are tagged as failed
      and surfaced via get_failed_strategies() / has_strategy_failed(tag).
    - Tactical family aliasing prevents minor tag variations from evading the guard.
    - Partially accepted strategies are recorded as partial.
    - Effective strategies resolving the concern are recorded as effective.
    - Prevents repeating strategies that failed on this objection instance.
    """
    manager = ConversationStateManager(call_sid="CA_strategy_feedback_loop")

    # Turn 1: Client raises commission fee objection
    b1 = _make_bundle(1, "client", "I'm not willing to pay a 6 percent commission on this house.")
    s1 = manager.process_turn_bundle(b1)
    assert len(s1.objections) == 1
    obj = s1.objections[0]
    assert obj.lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert len(obj.strategy_outcomes) == 0

    # Turn 2: Salesperson attempts Strategy A: financial_net_proceeds_reframe
    b2 = _make_bundle(
        2,
        "salesperson",
        "I understand. If we look at the financial net proceeds sheet, our pricing strategy nets sellers 4% more.",
        strategy_tag="financial_net_proceeds_reframe",
    )
    s2 = manager.process_turn_bundle(b2)
    assert "financial_net_proceeds_reframe" in s2.objections[0].attempted_strategies
    # Note: State should not prematurely be resolved or partially resolved
    assert s2.objections[0].lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 3: Client rejects Strategy A and repeats fee objection
    b3 = _make_bundle(
        3,
        "client",
        "I've looked at those sheets before, I'm still not paying 6 percent.",
        agreement=0.10,
    )
    s3 = manager.process_turn_bundle(b3)
    obj3 = s3.objections[0]
    assert len(obj3.strategy_outcomes) == 1
    outcome1 = obj3.strategy_outcomes[0]
    assert outcome1.strategy_tag == "financial_net_proceeds_reframe"
    assert outcome1.effectiveness == "rejected"
    assert obj3.has_strategy_failed("financial_net_proceeds_reframe") is True
    # Tactical aliasing: near-duplicate tag also detected as failed
    assert obj3.has_strategy_failed("net_proceeds_comparison") is True
    assert obj3.has_strategy_failed("net_sheet_breakdown") is True
    # Strict exact-match mode bypasses alias if specifically requested
    assert obj3.has_strategy_failed("net_proceeds_comparison", match_family=False) is False
    assert "financial_net_proceeds_reframe" in obj3.get_failed_strategies()
    assert obj3.lifecycle_state == ObjectionLifecycleState.ACTIVE

    # Turn 4: Salesperson adapts away from failed strategy, tries Strategy B: hyperlocal_marketing_differentiation
    b4 = _make_bundle(
        4,
        "salesperson",
        "Fair enough. What if we focus on our hyperlocal marketing differentiation so you see the direct buyer pipeline?",
        strategy_tag="hyperlocal_marketing_differentiation",
    )
    s4 = manager.process_turn_bundle(b4)
    assert "hyperlocal_marketing_differentiation" in s4.objections[0].attempted_strategies

    # Turn 5: Client partially accepts Strategy B
    b5 = _make_bundle(
        5,
        "client",
        "Well, I do appreciate marketing that brings real buyers, but the rate still bothers me.",
        agreement=0.55,
    )
    s5 = manager.process_turn_bundle(b5)
    obj5 = s5.objections[0]
    assert len(obj5.strategy_outcomes) == 2
    outcome2 = obj5.strategy_outcomes[1]
    assert outcome2.strategy_tag == "hyperlocal_marketing_differentiation"
    assert outcome2.effectiveness == "partial"
    assert "hyperlocal_marketing_differentiation" in obj5.get_partial_strategies()
    assert obj5.lifecycle_state == ObjectionLifecycleState.PARTIALLY_ADDRESSED

    # Turn 6: Salesperson deploys Strategy C: fee_performance_guarantee
    b6 = _make_bundle(
        6,
        "salesperson",
        "How about our fee performance guarantee: if we don't hit the target price in 21 days, our rate steps down?",
        strategy_tag="fee_performance_guarantee",
    )
    s6 = manager.process_turn_bundle(b6)
    assert "fee_performance_guarantee" in s6.objections[0].attempted_strategies

    # Turn 7: Client resolves objection with behavioral commitment
    b7 = _make_bundle(
        7,
        "client",
        "Alright, that actually protects my downside. Let's do that.",
        agreement=0.85,
        future_language=0.70,
    )
    s7 = manager.process_turn_bundle(b7)
    obj7 = s7.objections[0]
    assert len(obj7.strategy_outcomes) == 3
    outcome3 = obj7.strategy_outcomes[2]
    assert outcome3.strategy_tag == "fee_performance_guarantee"
    assert outcome3.effectiveness == "effective"
    assert obj7.lifecycle_state == ObjectionLifecycleState.RESOLVED

    # Final registry query verification
    assert obj7.get_failed_strategies() == ["financial_net_proceeds_reframe"]
    assert obj7.get_partial_strategies() == ["hyperlocal_marketing_differentiation"]
    assert obj7.get_effective_strategies() == ["fee_performance_guarantee"]
    assert obj7.has_strategy_failed("financial_net_proceeds_reframe") is True
    assert obj7.has_strategy_failed("net_proceeds_comparison") is True
    assert obj7.has_strategy_failed("fee_performance_guarantee") is False


