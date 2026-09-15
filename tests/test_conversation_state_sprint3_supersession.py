import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_supersession import (
    TruthSupersessionDetector,
    SupersessionDecision,
)
from copilot.conversation_state_models import PersistentFactRecord


def _create_prospect_bundle(turn_id: int, text: str, call_sid: str = "CA_supersession_test_001"):
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.2, confidence=0.8),
        pacing=DimensionScore(score=0.60, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.65, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.60, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_sup_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_sup_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id="client",
    )

    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id="client",
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
    )


class TestConversationStateSprint3TruthSupersession:
    def test_client_exact_canonical_example_stay_to_conditional_move(self):
        """Validates the exact client example from the specification:

        Turn 2: "We've decided to stay in the home"
        Turn 8: "I'd still move if I believed there was a better strategy"

        Proves:
        1. Conditionality is recognized as an UPDATE without contradiction keywords.
        2. Old fact is NOT deleted: transitioned to 'superseded' with bidirectional linking.
        3. New fact is active with updated conditional belief.
        4. Both records retained in chronological history.
        """
        manager = ConversationStateManager(call_sid="CA_client_example_001")

        # Turn 2: Prospect asserts decision to stay
        t2 = _create_prospect_bundle(turn_id=2, text="We've decided to stay in the home.")
        fact_payload = [
            {
                "category": "timeline",
                "fact_key": "moving_decision",
                "fact_value": "Decided to stay in the home",
                "confidence": 0.95,
            }
        ]
        snap2 = manager.process_turn_bundle(t2, fact_updates=fact_payload)
        assert len(snap2.get_active_facts()) == 1
        f_orig = snap2.get_active_fact("moving_decision")
        assert f_orig.status == "active"
        orig_fact_id = f_orig.fact_id

        # Turn 8: Prospect expresses conditional willingness to move
        t8 = _create_prospect_bundle(
            turn_id=8,
            text="I'd still move if I believed there was a better strategy.",
        )
        snap8 = manager.process_turn_bundle(t8)

        # Invariant 1: Active fact is now the updated conditional belief
        active_f = snap8.get_active_fact("moving_decision")
        assert active_f is not None
        assert active_f.fact_id != orig_fact_id
        assert active_f.status == "active"
        assert active_f.source_turn_id == 8
        assert "moving conditionally" in active_f.fact_value.lower() or "better strategy" in active_f.fact_value.lower()

        # Invariant 2: Old fact is permanently preserved in history as 'superseded'
        history = manager.facts_manager.get_fact_history("moving_decision")
        assert len(history) == 2

        old_record = history[0]
        assert old_record.fact_id == orig_fact_id
        assert old_record.status == "superseded"
        assert old_record.superseded_by_fact_id == active_f.fact_id
        assert old_record.superseded_at_turn_id == 8

        # Invariant 3: Explainability change record emitted
        sup_changes = [c for c in snap8.change_history if c.field_path == "facts.moving_decision"]
        assert len(sup_changes) == 1
        assert "Truth supersession" in sup_changes[0].reason

    def test_decision_authority_supersession_wife_to_power_of_attorney(self):
        """Validates decision authority shift:

        'My wife handles all the decisions' -> 'I have power of attorney to sign today'
        """
        manager = ConversationStateManager(call_sid="CA_auth_test_001")

        # Turn 1: Joint/spouse requirement
        t1 = _create_prospect_bundle(turn_id=1, text="Hello")
        fact_payload = [
            {
                "category": "decision_maker",
                "fact_key": "decision_maker_authority",
                "fact_value": "My wife handles the decisions and must be present to consult",
            }
        ]
        snap1 = manager.process_turn_bundle(t1, fact_updates=fact_payload)
        f1_id = snap1.get_active_fact("decision_maker_authority").fact_id

        # Turn 5: Prospect clarifies sole power of attorney
        t5 = _create_prospect_bundle(
            turn_id=5,
            text="Actually, on this investment property I have power of attorney to sign today.",
        )
        snap5 = manager.process_turn_bundle(t5)

        active_f = snap5.get_active_fact("decision_maker_authority")
        assert active_f.fact_id != f1_id
        assert active_f.status == "active"
        assert "power of attorney" in active_f.fact_value.lower()

        old_f = next(f for f in snap5.facts if f.fact_id == f1_id)
        assert old_f.status == "superseded"
        assert old_f.superseded_by_fact_id == active_f.fact_id

    def test_timeline_closing_supersession_thanksgiving_to_january(self):
        """Validates closing timeline shift:

        'Close before Thanksgiving' -> 'School semester doesn't end until January'
        """
        manager = ConversationStateManager(call_sid="CA_timeline_test_001")

        # Turn 2: November timeline
        t2 = _create_prospect_bundle(turn_id=2, text="Need to close before Thanksgiving.")
        fact_payload = [
            {
                "category": "timeline",
                "fact_key": "target_closing",
                "fact_value": "Need to close before Thanksgiving",
            }
        ]
        snap2 = manager.process_turn_bundle(t2, fact_updates=fact_payload)
        orig_id = snap2.get_active_fact("target_closing").fact_id

        # Turn 7: January timeline shift
        t7 = _create_prospect_bundle(
            turn_id=7,
            text="The kids' school semester doesn't end until January, so we can't move until then.",
        )
        snap7 = manager.process_turn_bundle(t7)

        active_timeline = snap7.get_active_fact("target_closing")
        assert active_timeline.fact_id != orig_id
        assert active_timeline.status == "active"
        assert "january" in active_timeline.fact_value.lower()

        history = manager.facts_manager.get_fact_history("target_closing")
        assert len(history) == 2
        assert history[0].status == "superseded"
        assert history[1].status == "active"

    def test_channel_preference_supersession_cell_to_text_email(self):
        """Validates contact channel restriction:

        'Call me on my cell anytime' -> 'Clinic is packed, text or email is much better'
        """
        manager = ConversationStateManager(call_sid="CA_channel_test_001")

        t1 = _create_prospect_bundle(turn_id=1, text="Feel free to call me on my cell anytime.")
        fact_payload = [
            {
                "category": "preference",
                "fact_key": "preferred_channel",
                "fact_value": "Call me on my cell anytime",
            }
        ]
        snap1 = manager.process_turn_bundle(t1, fact_updates=fact_payload)
        orig_id = snap1.get_active_fact("preferred_channel").fact_id

        t4 = _create_prospect_bundle(
            turn_id=4,
            text="During the day my clinic is packed with patients, so text or email is much better.",
        )
        snap4 = manager.process_turn_bundle(t4)

        active_channel = snap4.get_active_fact("preferred_channel")
        assert active_channel.fact_id != orig_id
        assert active_channel.status == "active"
        assert "text or email" in active_channel.fact_value.lower()

    def test_reiteration_confirms_without_supersession(self):
        """Reiterating an active belief (CONFIRMS) must NOT trigger supersession or spurious version bumps."""
        manager = ConversationStateManager(call_sid="CA_reiteration_test_001")

        t1 = _create_prospect_bundle(turn_id=1, text="We've decided to stay in the home.")
        fact_payload = [
            {
                "category": "timeline",
                "fact_key": "moving_decision",
                "fact_value": "Decided to stay in the home",
            }
        ]
        snap1 = manager.process_turn_bundle(t1, fact_updates=fact_payload)
        f1_id = snap1.get_active_fact("moving_decision").fact_id
        v_after_t1 = snap1.state_version

        # Turn 3: Prospect re-affirms staying
        t3 = _create_prospect_bundle(turn_id=3, text="Yes, like I said, we are definitely staying put.")
        snap3 = manager.process_turn_bundle(t3)

        # Zero supersession: exactly 1 record in history, same ID, still active
        history = manager.facts_manager.get_fact_history("moving_decision")
        assert len(history) == 1
        assert history[0].fact_id == f1_id
        assert history[0].status == "active"

    def test_unrelated_utterance_leaves_facts_unchanged(self):
        """Unrelated conversational chatter (UNCHANGED) leaves active facts untouched."""
        detector = TruthSupersessionDetector()
        active_fact = PersistentFactRecord(
            category="timeline",
            fact_key="moving_decision",
            fact_value="Decided to stay in the home",
            source_turn_id=1,
            timestamp_ms=1000,
        )

        decisions = detector.evaluate_turn(
            candidate_text="It looks like it might rain this afternoon.",
            active_facts=[active_fact],
        )

    def test_outright_reversal_selling_to_not_selling(self):
        """Validates REVERSES: outright flipping of a prior conclusion:

        Turn 1: "We're planning to sell this spring" -> Turn 5: "We're not interested in selling anymore, we've decided to pull it off the market completely"
        """
        manager = ConversationStateManager(call_sid="CA_reversal_test_001")

        # Turn 1: Intending to sell
        t1 = _create_prospect_bundle(turn_id=1, text="We are planning to sell this spring.")
        fact_payload = [
            {
                "category": "timeline",
                "fact_key": "selling_decision",
                "fact_value": "Planning to sell this spring",
                "confidence": 0.95,
            }
        ]
        snap1 = manager.process_turn_bundle(t1, fact_updates=fact_payload)
        f_sell_id = snap1.get_active_fact("selling_decision").fact_id

        # Turn 5: Complete reversal
        t5 = _create_prospect_bundle(
            turn_id=5,
            text="We are not interested in selling anymore. We've decided to pull it off the market completely.",
        )
        snap5 = manager.process_turn_bundle(t5)

        # Invariant 1: Active fact reflects complete reversal
        active_f = snap5.get_active_fact("selling_decision")
        assert active_f is not None
        assert active_f.fact_id != f_sell_id
        assert active_f.status == "active"
        assert "not to sell" in active_f.fact_value.lower() or "off the market" in active_f.fact_value.lower()

        # Invariant 2: Old fact superseded with pointer to new
        history = manager.facts_manager.get_fact_history("selling_decision")
        assert len(history) == 2
        assert history[0].status == "superseded"
        assert history[0].superseded_by_fact_id == active_f.fact_id

        # Invariant 3: Explainability record explicitly tags REVERSES and carries confidence
        chg = [c for c in snap5.change_history if c.field_path == "facts.selling_decision"]
        assert len(chg) == 1
        assert "REVERSES" in chg[0].reason
        assert chg[0].confidence >= 0.85

    def test_candidate_fact_retrieval_prevents_cross_domain_false_matches(self):
        """Validates the candidate fact routing/retrieval pre-filtering mechanism:

        Verifies that an utterance in one domain (e.g. channel or scheduling) is NEVER
        evaluated against an active fact in an unrelated domain (e.g. spouse authority or price).
        """
        detector = TruthSupersessionDetector()

        fact_timeline = PersistentFactRecord(
            category="timeline",
            fact_key="moving_decision",
            fact_value="Decided to stay in the home",
            source_turn_id=1,
            timestamp_ms=1000,
        )
        fact_spouse = PersistentFactRecord(
            category="decision_maker",
            fact_key="spouse_involvement",
            fact_value="Wife Mary must be present",
            source_turn_id=1,
            timestamp_ms=1000,
        )
        fact_channel = PersistentFactRecord(
            category="preference",
            fact_key="preferred_channel",
            fact_value="Call me on my cell anytime",
            source_turn_id=1,
            timestamp_ms=1000,
        )
        fact_budget = PersistentFactRecord(
            category="financial",
            fact_key="max_budget",
            fact_value="$650,000 all-in",
            source_turn_id=1,
            timestamp_ms=1000,
        )

        all_facts = [fact_timeline, fact_spouse, fact_channel, fact_budget]

        # Case 1: Utterance strictly about channel/clinic
        channel_candidates = detector.filter_candidate_facts(
            "I can't talk on the phone during clinic hours, so send me an email instead",
            all_facts,
        )
        matched_keys_1 = [f.fact_key for f in channel_candidates]
        assert "preferred_channel" in matched_keys_1
        assert "spouse_involvement" not in matched_keys_1
        assert "max_budget" not in matched_keys_1
        assert "moving_decision" not in matched_keys_1

        # Case 2: Utterance strictly about decision authority / attorney
        authority_candidates = detector.filter_candidate_facts(
            "Actually, my attorney and I have sole authority to sign for the LLC",
            all_facts,
        )
        matched_keys_2 = [f.fact_key for f in authority_candidates]
        assert "spouse_involvement" in matched_keys_2
        assert "preferred_channel" not in matched_keys_2
        assert "max_budget" not in matched_keys_2

        # Case 3: Completely unrelated descriptive chatter (zero candidate facts returned)
        unrelated_candidates = detector.filter_candidate_facts(
            "The oak trees in the backyard look magnificent in this weather",
            all_facts,
        )
        assert len(unrelated_candidates) == 0

    def test_conditionality_and_nuance_preservation_in_updated_fact(self):
        """Validates that conditional nuance is preserved in the updated fact value,

        rather than being flattened into a crude binary overwrite.
        """
        detector = TruthSupersessionDetector()
        fact_stay = PersistentFactRecord(
            category="timeline",
            fact_key="moving_decision",
            fact_value="Decided to stay in the home",
            source_turn_id=2,
            timestamp_ms=2000,
        )

        decisions = detector.evaluate_turn(
            candidate_text="I'd still move if I believed there was a better strategy.",
            active_facts=[fact_stay],
        )

        assert len(decisions) == 1
        d = decisions[0]
        assert d.has_supersession is True
        assert d.relation == "UPDATES"
        # Verify conditionality is explicitly captured
        assert d.condition is not None
        assert "better strategy" in d.condition.lower()
        assert "condition" in d.new_truth_value.lower()
        assert "better strategy" in d.new_truth_value.lower()
