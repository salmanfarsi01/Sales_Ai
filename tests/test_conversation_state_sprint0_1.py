import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.behavioral_baseline import ContactPreferenceRecord
from copilot.conversation_state_contract import (
    BehavioralSignalInputBundle,
    extract_behavioral_bundle,
)
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStructure,
    DecisionStakeholder,
    ObjectionRecord,
    PersistentFactRecord,
)
from copilot.conversation_facts import PersistentFactsManager
from copilot.conversation_state_manager import (
    ConversationStateManager,
    StaleStateUpdateError,
)


def _create_mock_upstream_data(turn_id: int = 1, call_sid: str = "CA_test_call_001", timestamp_ms: int = None):
    ts = timestamp_ms if timestamp_ms is not None else 1000 * turn_id
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=ts,
        trust=DimensionScore(score=0.82, confidence=0.9, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.4, tension_level=0.2, confidence=0.85),
        pacing=DimensionScore(score=0.65, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.88, confidence=0.9, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.78, confidence=0.85, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.72, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.88,
        contributing_evidence_ids=["ev_inf_001", "ev_inf_002"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id="client",
        contact_preference="reduced_frequency",
        contact_preference_confidence=0.95,
        contact_preference_details="Please don't call every day",
        specificity_score=0.8,
        future_language_score=0.75,
        boundary_score=0.0,
    )

    pref_record = ContactPreferenceRecord(
        preference="reduced_frequency",
        confidence=0.95,
        details="Please don't call every day",
        first_observed_turn=turn_id,
        call_sid=call_sid,
        timestamp_ms=ts,
    )

    return inference, semantic, pref_record


class TestConversationStateSprint0InputContract:
    def test_extract_behavioral_bundle(self):
        inference, semantic, pref_record = _create_mock_upstream_data(turn_id=2)

        bundle = extract_behavioral_bundle(
            turn_id=2,
            speaker_id="client",
            utterance_text="Thursday afternoon works best for me.",
            inference_state=inference,
            semantic_snapshot=semantic,
            contact_preference_record=pref_record,
        )

        assert isinstance(bundle, BehavioralSignalInputBundle)
        assert bundle.turn_id == 2
        assert bundle.call_sid == "CA_test_call_001"
        assert bundle.speaker_id == "client"
        assert bundle.utterance_text == "Thursday afternoon works best for me."
        assert bundle.trust.score == 0.82
        assert bundle.readiness.score == 0.72
        assert bundle.contact_preference == "reduced_frequency"
        assert bundle.boundary_score == 0.0
        assert "utt_002" in bundle.contributing_evidence_ids
        assert "ev_inf_001" in bundle.contributing_evidence_ids

    def test_strategy_tagging_in_input_contract(self):
        inference, semantic, pref_record = _create_mock_upstream_data(turn_id=3)

        bundle = extract_behavioral_bundle(
            turn_id=3,
            speaker_id="salesperson",
            utterance_text="If you look at the cost of waiting another quarter, taxes alone exceed the difference.",
            inference_state=inference,
            semantic_snapshot=semantic,
            contact_preference_record=pref_record,
            salesperson_strategy_tag="reframe_cost_of_inaction",
            salesperson_strategy_source="strategic_engine",
        )

        assert bundle.salesperson_strategy_tag == "reframe_cost_of_inaction"
        assert bundle.salesperson_strategy_source == "strategic_engine"


class TestConversationStateSprint1CoreSchema:
    def test_schema_models_instantiation(self):
        stakeholder = DecisionStakeholder(
            name="Sarah",
            role="spouse",
            presence="absent",
            notes="Must consult before signing",
        )
        decision_struct = DecisionStructure(
            primary_decision_maker="John",
            decision_maker_present=True,
            stakeholders=[stakeholder],
            timeline_horizon="next_week",
            urgency_level="medium",
            access_constraints=["can't meet mornings"],
        )
        assert len(decision_struct.stakeholders) == 1
        assert decision_struct.stakeholders[0].role == "spouse"

        objection = ObjectionRecord(
            canonical_category="commission_fee",
            initial_statement="Your 6% fee is too high",
            latest_statement="I'm still hesitant on 6%",
            lifecycle_state="unresolved",
            first_turn_id=3,
            last_updated_turn_id=5,
            recurrence_count=2,
            attempted_strategies=["value_comparison"],
        )
        assert objection.lifecycle_state == "unresolved"
        assert objection.recurrence_count == 2
        assert objection.attempted_strategies == ["value_comparison"]

        snapshot = ConversationStateSnapshot(
            call_sid="CA_test_call_001",
            decision_structure=decision_struct,
            objections=[objection],
        )
        assert snapshot.state_version == 1
        assert len(snapshot.get_unresolved_objections()) == 1


class TestConversationStateSprint1PersistentFacts:
    def test_persistent_facts_retention_across_unrelated_intervening_turns(self):
        """Validates Client Principle #6 under active topic shifts:

        A fact stated in Turn 1 (spouse involvement) must remain 100% active and untouched
        across 25 turns where the conversation moves to budget, commission objections,
        timeline shifts, supersessions, and other property facts.
        """
        manager = PersistentFactsManager()

        # Turn 1: Prospect states spouse must be involved
        f_spouse = manager.record_fact(
            category="decision_maker",
            fact_key="spouse_involvement",
            fact_value="Wife Mary must be present for final agreement",
            source_turn_id=1,
            timestamp_ms=3000,
        )
        assert f_spouse.status == "active"
        assert f_spouse.source_turn_id == 1

        # Intervening Turn 4: Conversation shifts to budget & pricing
        f_budget = manager.record_fact(
            category="financial",
            fact_key="max_budget",
            fact_value="$650,000 all-in",
            source_turn_id=4,
            timestamp_ms=15000,
        )

        # Intervening Turn 7: Conversation shifts to move-in timeline
        f_timeline_orig = manager.record_fact(
            category="timeline",
            fact_key="target_closing",
            fact_value="November 1st before the holidays",
            source_turn_id=7,
            timestamp_ms=32000,
        )

        # Intervening Turn 11: Shift to property preferences
        f_property = manager.record_fact(
            category="property",
            fact_key="lot_preference",
            fact_value="Minimum quarter-acre with mature trees",
            source_turn_id=11,
            timestamp_ms=54000,
        )

        # Intervening Turn 15: Prospect supersedes timeline fact!
        f_timeline_old, f_timeline_new = manager.supersede_fact(
            old_fact_id=f_timeline_orig.fact_id,
            new_fact_value="Pushed to December 15th due to school schedule",
            source_turn_id=15,
            timestamp_ms=78000,
        )

        # Intervening Turn 19: Shift to communication channel preference
        f_channel = manager.record_fact(
            category="preference",
            fact_key="preferred_channel",
            fact_value="Email summaries only, no afternoon calls",
            source_turn_id=19,
            timestamp_ms=95000,
        )

        # Turn 25 (End of call / 24 turns later):
        # 1. Spouse fact from Turn 1 is STILL active and untouched
        spouse_fact_now = manager.get_active_fact("spouse_involvement")
        assert spouse_fact_now is not None
        assert spouse_fact_now.fact_id == f_spouse.fact_id
        assert spouse_fact_now.status == "active"
        assert spouse_fact_now.source_turn_id == 1
        assert spouse_fact_now.fact_value == "Wife Mary must be present for final agreement"

        # 2. Budget and Property facts are also active
        assert manager.get_active_fact("max_budget").fact_value == "$650,000 all-in"
        assert manager.get_active_fact("lot_preference").source_turn_id == 11

        # 3. Superseded timeline has exactly 2 history entries (old superseded, new active)
        timeline_history = manager.get_fact_history("target_closing")
        assert len(timeline_history) == 2
        assert timeline_history[0].status == "superseded"
        assert timeline_history[0].superseded_at_turn_id == 15
        assert timeline_history[1].status == "active"
        assert timeline_history[1].source_turn_id == 15

        # 4. Total active facts: 5 (spouse, budget, timeline_new, property, channel)
        assert len(manager.get_active_facts()) == 5

    def test_fact_supersession_preserves_history(self):
        manager = PersistentFactsManager()

        # Turn 2: Prospect says "We've decided to stay in the home"
        f1 = manager.record_fact(
            category="timeline",
            fact_key="moving_decision",
            fact_value="Decided to stay in the home",
            source_turn_id=2,
            timestamp_ms=10000,
        )
        assert f1.status == "active"

        # Turn 8: Prospect updates: "I'd still move if there was a better strategy"
        old_fact, new_fact = manager.supersede_fact(
            old_fact_id=f1.fact_id,
            new_fact_value="Would move conditionally if there was a better strategy",
            source_turn_id=8,
            timestamp_ms=45000,
        )

        # Invariant 1: Old fact is not deleted; status is 'superseded'
        assert old_fact.status == "superseded"
        assert old_fact.superseded_by_fact_id == new_fact.fact_id
        assert old_fact.superseded_at_turn_id == 8

        # Invariant 2: New fact is 'active' and points to the same key
        assert new_fact.status == "active"
        assert new_fact.fact_key == "moving_decision"
        assert new_fact.source_turn_id == 8

        # Invariant 3: History preserves both in chronological order
        history = manager.get_fact_history("moving_decision")
        assert len(history) == 2
        assert history[0].status == "superseded"
        assert history[1].status == "active"

        # Invariant 4: Active query returns the new one
        assert manager.get_active_fact("moving_decision").fact_id == new_fact.fact_id


class TestConversationStateManager:
    def test_monotonic_versioning_and_explainability(self):
        state_mgr = ConversationStateManager(call_sid="CA_test_001")
        assert state_mgr.current_state.state_version == 1

        # Turn 1
        inf1, sem1, pref1 = _create_mock_upstream_data(turn_id=1)
        bundle1 = extract_behavioral_bundle(1, "client", "Hello", inf1, sem1, pref1)
        snap1 = state_mgr.process_turn_bundle(bundle1)

        # Version must have incremented due to dimension & compliance updates
        assert snap1.state_version > 1
        assert snap1.last_updated_turn_id == 1
        assert len(snap1.change_history) > 0

        first_change = snap1.change_history[0]
        assert first_change.triggering_turn_id == 1
        assert first_change.state_version_before < first_change.state_version_after
        assert "ev_inf_001" in first_change.evidence_ids

    def test_stale_state_write_rejection_and_equal_turn_id_semantics(self):
        """Validates Client Principle #9 including the equal turn_id boundary condition:

        1. bundle.turn_id < last_updated_turn_id -> Strictly rejected as StaleStateUpdateError.
        2. bundle.turn_id == last_updated_turn_id with older timestamp -> Rejected as stale packet.
        3. bundle.turn_id == last_updated_turn_id with newer/equal timestamp -> Permitted as intra-turn
           refinement (e.g. late LLM enrichment or continuation fragment).
        """
        state_mgr = ConversationStateManager(call_sid="CA_test_002")

        # Process Turn 5 at timestamp 5000ms
        inf5, sem5, pref5 = _create_mock_upstream_data(turn_id=5, timestamp_ms=5000)
        bundle5 = extract_behavioral_bundle(5, "client", "Turn 5 initial segment", inf5, sem5, pref5)
        snap5 = state_mgr.process_turn_bundle(bundle5)
        assert snap5.last_updated_turn_id == 5
        assert snap5.last_updated_timestamp_ms == 5000
        v_after_turn5 = snap5.state_version

        # Case 1: Strictly older turn (Turn 3 arriving late) -> MUST reject
        inf3, sem3, pref3 = _create_mock_upstream_data(turn_id=3, timestamp_ms=3000)
        bundle3 = extract_behavioral_bundle(3, "client", "Delayed Turn 3 message", inf3, sem3, pref3)
        with pytest.raises(StaleStateUpdateError) as exc_info:
            state_mgr.process_turn_bundle(bundle3)
        assert "bundle turn 3 is older than current turn 5" in str(exc_info.value)

        # Case 2: Equal turn ID (Turn 5) but with strictly EARLIER timestamp (out-of-order race) -> MUST reject
        inf5_stale, sem5_stale, pref5_stale = _create_mock_upstream_data(turn_id=5, timestamp_ms=4500)
        bundle5_stale = extract_behavioral_bundle(5, "client", "Stale Turn 5 fragment", inf5_stale, sem5_stale, pref5_stale)
        with pytest.raises(StaleStateUpdateError) as exc_info:
            state_mgr.process_turn_bundle(bundle5_stale)
        assert "has earlier timestamp (4500ms) than current processed turn (5000ms)" in str(exc_info.value)

        # Case 3: Equal turn ID (Turn 5) with NEWER timestamp (5200ms) -> Permitted intra-turn refinement
        # (e.g. late LLM enrichment or continuation fragment with higher readiness)
        inf5_refined, sem5_refined, pref5_refined = _create_mock_upstream_data(turn_id=5, timestamp_ms=5200)
        inf5_refined.readiness.score = 0.95  # Enrichment updates readiness
        bundle5_refined = extract_behavioral_bundle(5, "client", "Turn 5 enriched", inf5_refined, sem5_refined, pref5_refined)

        snap5_refined = state_mgr.process_turn_bundle(bundle5_refined)
        assert snap5_refined.last_updated_turn_id == 5
        assert snap5_refined.last_updated_timestamp_ms == 5200
        assert snap5_refined.dimensions.readiness == 0.95
        assert snap5_refined.state_version > v_after_turn5
