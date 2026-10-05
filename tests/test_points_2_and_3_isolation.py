import pytest
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ObjectionRecord,
    PersistentFactRecord,
    ObjectionLifecycleState,
)
from copilot.conversation_state_manager import ConversationStateManager
from copilot.core_decision_manager import CoreDecisionManager, StrategicDecisionCache
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from copilot.behavioral_semantic import SemanticFeatureSnapshot


def _make_test_bundle(call_sid: str, turn_id: int, speaker: str, text: str):
    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_{call_sid}_{turn_id}",
        call_sid=call_sid,
        speaker_id=speaker,
        boundary_score=0.0,
    )
    inf = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=turn_id * 1000,
        trust=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.2, confidence=0.85),
        pacing=DimensionScore(score=0.6, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.5, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
    )
    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker,
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=sem,
    )


def test_point2_new_call_initialization_zero_shared_state():
    """Point 2: Ensure a new call_sid always constructs a brand-new ConversationStateManager /
    state object with no inherited state and zero shared mutable references across calls.
    """
    mgr_a = ConversationStateManager(call_sid="call_isolated_A")
    mgr_b = ConversationStateManager(call_sid="call_isolated_B")

    # 1. Assert distinct object references (not just equal-by-value, actually distinct IDs)
    assert id(mgr_a) != id(mgr_b)
    assert id(mgr_a.current_state) != id(mgr_b.current_state)
    assert id(mgr_a.current_state.objections) != id(mgr_b.current_state.objections)
    assert id(mgr_a.current_state.facts) != id(mgr_b.current_state.facts)
    assert id(mgr_a.current_state.dimensions) != id(mgr_b.current_state.dimensions)
    assert id(mgr_a.current_state.contact_compliance) != id(mgr_b.current_state.contact_compliance)
    assert id(mgr_a.current_state.stage_history) != id(mgr_b.current_state.stage_history)
    assert id(mgr_a.facts_manager) != id(mgr_b.facts_manager)
    assert id(mgr_a.objections_engine) != id(mgr_b.objections_engine)
    assert id(mgr_a.conversion_engine) != id(mgr_b.conversion_engine)
    assert id(mgr_a._bundle_history) != id(mgr_b._bundle_history)
    assert id(mgr_a._conversion_events) != id(mgr_b._conversion_events)
    assert id(mgr_a._deal_dispositions) != id(mgr_b._deal_dispositions)

    # 2. Mutate Call A with an objection and a fact
    t1_a = _make_test_bundle("call_isolated_A", 1, "client", "Your commission is way too high.")
    mgr_a.process_turn_bundle(
        t1_a,
        fact_updates=[{"category": "timeline", "fact_key": "property_address", "fact_value": "123 Elm St"}],
    )

    # 3. Verify Call A received the mutation
    assert len(mgr_a.current_state.facts) == 1
    assert mgr_a.current_state.facts[0].fact_value == "123 Elm St"

    # 4. Verify Call B remains 100% clean and unpolluted
    assert len(mgr_b.current_state.objections) == 0
    assert len(mgr_b.current_state.facts) == 0
    assert mgr_b.current_state.conversion_gate is None
    assert len(mgr_b._bundle_history) == 0


def test_point2_prospect_memory_explicit_loading_isolation():
    """Point 2: Prospect memory is ONLY loaded when load_prospect_memory=True is explicitly passed with authorized user_id."""
    # When load_prospect_memory is False (default): clean slate
    mgr_fresh = ConversationStateManager(
        call_sid="call_fresh_prospect",
        user_id="legacy_migrated_user",
        prospect_id="test_prospect_001",
        load_prospect_memory=False,
    )
    assert len(mgr_fresh.current_state.contact_compliance.contact_preferences) == 0
    assert mgr_fresh.current_state.contact_compliance.contact_preference == "none"

    # When load_prospect_memory is True without user_id: fails closed (Point 18 security invariant)
    mgr_no_user = ConversationStateManager(
        call_sid="call_no_user",
        user_id=None,
        prospect_id="test_prospect_001",
        load_prospect_memory=True,
    )
    assert len(mgr_no_user.current_state.contact_compliance.contact_preferences) == 0
    assert len(mgr_no_user.current_state.loaded_prospect_memory) == 0

    # When load_prospect_memory is True with authorized user_id: intentionally loads saved prospect boundary
    mgr_returning = ConversationStateManager(
        call_sid="call_returning_prospect",
        user_id="legacy_migrated_user",
        prospect_id="test_prospect_001",
        load_prospect_memory=True,
    )
    assert mgr_returning.current_state.contact_compliance.contact_preference == "reduced_frequency"
    assert len(mgr_returning.current_state.contact_compliance.contact_preferences) >= 1
    assert len(mgr_returning.current_state.loaded_prospect_memory) >= 1


def test_point3_cache_isolation_keying():
    """Point 3: Cache for Core Intelligence / StrategicDecision must key on
    (call_sid, source_state_version), never on turn number alone or nothing.
    Requests for the same turn number across two different call_sids must yield different decisions.
    """
    StrategicDecisionCache.clear()

    cdm_a = CoreDecisionManager(call_sid="call_alpha")
    cdm_b = CoreDecisionManager(call_sid="call_beta")

    # Both calls evaluate Turn 1, but with completely different utterances and states
    snap_a = ConversationStateSnapshot(call_sid="call_alpha", state_version=2, last_updated_turn_id=1)
    eval_a = cdm_a.evaluate_state(
        snapshot=snap_a,
        turn_speaker="client",
        turn_text="What would be your commission rate?",
        turn_id=1,
    )

    snap_b = ConversationStateSnapshot(call_sid="call_beta", state_version=5, last_updated_turn_id=1)
    eval_b = cdm_b.evaluate_state(
        snapshot=snap_b,
        turn_speaker="client",
        turn_text="Do not ever call this number again!",
        turn_id=1,
    )

    # 1. Query decisions for the exact same turn number (Turn 1) for both calls
    dec_a = cdm_a.get_decision_for_turn(1)
    dec_b = cdm_b.get_decision_for_turn(1)

    assert dec_a is not None
    assert dec_b is not None
    assert dec_a.decision_id != dec_b.decision_id

    # 2. Assert different decisions and different evidence come back
    assert dec_a.evidence_considered != dec_b.evidence_considered
    assert 'Turn utterance (client): "What would be your commission rate?"' in dec_a.evidence_considered[0]
    assert 'Turn utterance (client): "Do not ever call this number again!"' in dec_b.evidence_considered[0]

    # 3. Assert StrategicDecisionCache lookup strictly keyed by (call_sid, source_state_version)
    cached_a = StrategicDecisionCache.get("call_alpha", source_state_version=2)
    cached_b = StrategicDecisionCache.get("call_beta", source_state_version=5)

    assert cached_a is not None
    assert cached_b is not None
    assert cached_a.decision_id == dec_a.decision_id
    assert cached_b.decision_id == dec_b.decision_id
    assert cached_a.decision_id != cached_b.decision_id

    # Cross-lookup must return None (Call Beta source_state_version 2 does not exist)
    assert StrategicDecisionCache.get("call_beta", source_state_version=2) is None

    # 4. Assert un-keyed lookups or missing call_sids are prohibited
    with pytest.raises(ValueError, match="call_sid is required"):
        StrategicDecisionCache.get(call_sid="", source_state_version=2)

    with pytest.raises(ValueError, match="call_sid is required"):
        StrategicDecisionCache.get_by_call_and_turn(call_sid="", turn_id=1)

    # 5. Assert evaluate_state hits process-level cache on identical (call_sid, state_version)
    re_eval_a = cdm_a.evaluate_state(snapshot=snap_a, turn_id=1)
    assert re_eval_a.decision.decision_id == dec_a.decision_id
    assert re_eval_a.context.state_version == 2

    # 6. Assert separate CoreDecisionManager instance for call_alpha resolves from cache
    cdm_a_fresh_instance = CoreDecisionManager(call_sid="call_alpha")
    assert len(cdm_a_fresh_instance.decisions_by_turn) == 0  # fresh instance
    # Pulls from StrategicDecisionCache
    cached_via_fresh = cdm_a_fresh_instance.get_decision_for_turn(1)
    assert cached_via_fresh is not None
    assert cached_via_fresh.decision_id == dec_a.decision_id



def test_prompt_delivery_service_cross_call_isolation():
    """Verify PromptDeliveryService rejects default_call singleton fallback and maintains strict call_sid isolation."""
    from copilot.prompt_delivery_service import PromptDeliveryService

    # 1. Prohibit un-keyed / default initialization
    with pytest.raises(ValueError, match="Either decision_manager or call_sid must be explicitly provided"):
        PromptDeliveryService()

    # 2. Distinct instances with call_sid construct independent managers
    pds_a = PromptDeliveryService(call_sid="call_pds_A")
    pds_b = PromptDeliveryService(call_sid="call_pds_B")

    assert pds_a.decision_manager.call_sid == "call_pds_A"
    assert pds_b.decision_manager.call_sid == "call_pds_B"
    assert id(pds_a.decision_manager) != id(pds_b.decision_manager)


def test_point8_two_back_to_back_test_calls_zero_cross_call_leakage():
    """Client Point 8: Run two different test calls back-to-back through the real ConversationReplayEngine
    pipeline and confirm zero cross-call leakage into Test B:
    - Zero objections from Test A
    - Zero facts from Test A ('Thursday At 3', spouse, address)
    - Zero contact preferences from Test A
    - Gate remains CLOSED (0.00 commitment)
    - Evidence Considered panel contains zero traces of Test A's 'Thursday at 3' utterance.
    """
    from copilot.conversation_replay import ConversationReplayEngine

    turns_A = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, thanks for making time today — tell me a bit about what's going on with the house."},
        {"turn_id": 2, "speaker_id": "client", "text": "I'm the one making this decision, no one else needs to sign off."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "Got it. What's driving the timing for you?"},
        {"turn_id": 4, "speaker_id": "client", "text": "We might look at moving sometime next year, nothing urgent yet."},
        {"turn_id": 5, "speaker_id": "client", "text": "Honestly, we're not sure this is the right time anymore."},
        {"turn_id": 6, "speaker_id": "salesperson", "text": "That's fair — a lot of people feel that way before they see the actual numbers."},
        {"turn_id": 7, "speaker_id": "client", "text": "Okay, that makes sense, I guess timing isn't the biggest issue."},
        {"turn_id": 8, "speaker_id": "salesperson", "text": "Great — would sometime next week work for a walkthrough?"},
        {"turn_id": 9, "speaker_id": "client", "text": "Maybe next week could work, let me think about it."},
        {"turn_id": 10, "speaker_id": "client", "text": "Actually, my wife would really need to be part of this conversation before we go any further."},
        {"turn_id": 11, "speaker_id": "salesperson", "text": "Of course, happy to loop her in whenever works."},
        {"turn_id": 12, "speaker_id": "client", "text": "I guess I'm just worried this isn't really the right move for us financially with everything going on."},
        {"turn_id": 13, "speaker_id": "salesperson", "text": "Totally understand — let's look at your net proceeds after all costs."},
        {"turn_id": 14, "speaker_id": "client", "text": "That's actually really helpful, tell me more."},
        {"turn_id": 15, "speaker_id": "client", "text": "Please don't start texting me every day before we meet, by the way."},
        {"turn_id": 16, "speaker_id": "client", "text": "Mornings don't really work for us either, just so you know."},
        {"turn_id": 17, "speaker_id": "salesperson", "text": "Noted on all of that. What day works best?"},
        {"turn_id": 18, "speaker_id": "client", "text": "Thursday at 3 works, and my wife will be there."}
    ]

    turns_B = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, this is Alex with Premier Realty. How are you doing today?"},
        {"turn_id": 2, "speaker_id": "client", "text": "I'm doing well, but I'm just curious about current home values in our neighborhood."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "I'd be glad to share recent neighborhood sales. Have you considered selling anytime soon?"},
        {"turn_id": 4, "speaker_id": "client", "text": "Maybe in a couple of years, no immediate plans."}
    ]

    engine = ConversationReplayEngine()

    # Run Test A (18 turns)
    rep_a = engine.replay_dialogue_turns("sim_suite_test_A", turns_A, save_report=False)
    final_a = rep_a.final_state
    assert final_a.conversion_gate.is_open is True
    assert final_a.dimensions.commitment == 1.00
    assert any("Thursday At 3" in f.fact_value for f in final_a.facts)

    # Run Test B immediately afterwards through the same engine instance
    rep_b = engine.replay_dialogue_turns("sim_suite_test_B", turns_B, save_report=False)
    final_b = rep_b.final_state

    # 1. Assert zero facts from Test A leaked into Test B
    assert len(final_b.facts) == 0
    for f in final_b.facts:
        assert "Thursday" not in f.fact_value
        assert "wife" not in f.fact_value.lower()
        assert "texting" not in f.fact_value.lower()

    # 2. Assert zero objections leaked from Test A
    test_a_obj_statements = {o.initial_statement for o in final_a.objections}
    for o in final_b.objections:
        assert o.initial_statement not in test_a_obj_statements

    # 3. Assert zero contact preferences leaked from Test A
    assert len(final_b.contact_compliance.contact_preferences) == 0
    assert final_b.contact_compliance.contact_preference == "none"

    # 4. Assert gate is closed and commitment is 0.00
    assert final_b.conversion_gate.status == "closed"
    assert final_b.conversion_gate.is_open is False
    assert final_b.dimensions.commitment == 0.00

    # 5. Assert zero traces of Test A in Evidence Considered panel across all turns of Test B
    for step in rep_b.timeline:
        sd = step.strategic_decision
        if sd:
            for ev in sd.evidence_considered:
                assert "Thursday at 3" not in ev
                assert "wife" not in ev.lower()
                assert "texting" not in ev.lower()


