import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_materiality import MaterialityFilter
from copilot.conversation_state_models import PersistentFactRecord


def _create_bundle(
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
    question_type: str = "none",
    agreement: float = 0.0,
    call_sid: str = "CA_materiality_test_001",
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
        contributing_evidence_ids=[f"ev_mat_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_mat_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        recurrence_id=recurrence_id,
        question_type=question_type,
        agreement_score=agreement,
    )

    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inference,
        semantic_snapshot=semantic,
        salesperson_strategy_tag=salesperson_strategy_tag,
    )


class TestConversationStateSprint4Materiality:
    def test_spec_acceptance_dimension_stability_on_unrelated_fact(self):
        """Definition of Done Acceptance Test from ConversationState_Implementation_Plan:

        'A turn that only reveals a new fact does not also perturb Trust/Emotion scores;
        verified via a test where an unrelated fact is stated and dimension scores remain
        provably unchanged.'
        """
        manager = ConversationStateManager(call_sid="CA_stability_test_001")

        # Turn 1: Initial conversation baseline
        t1 = _create_bundle(
            turn_id=1,
            speaker_id="client",
            text="Hello, thanks for following up with me today.",
            trust=0.72,
            emotion_valence=0.10,
            emotion_tension=0.15,
            readiness=0.65,
        )
        snap1 = manager.process_turn_bundle(t1)
        initial_trust = snap1.dimensions.trust
        initial_valence = snap1.dimensions.emotion_valence
        initial_tension = snap1.dimensions.emotion_tension
        initial_readiness = snap1.dimensions.readiness

        # Turn 2: Prospect reveals an unrelated fact about spouse's schedule / work
        # Upstream inference engine reports slight acoustic noise (e.g. slight jitter: trust=0.74, valence=0.12),
        # but because this turn is purely a factual disclosure without a genuine behavioral shift,
        # the Materiality Filter gates out 'dimensions', preserving dimension stability!
        t2 = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="Just so you know, my spouse's schedule changes next month because of hospital shifts.",
            trust=0.74,
            emotion_valence=0.12,
            emotion_tension=0.16,
            readiness=0.67,
        )
        snap2 = manager.process_turn_bundle(
            t2,
            fact_updates=[
                {
                    "category": "logistical",
                    "fact_key": "spouse_work_shift",
                    "fact_value": "Hospital shifts change next month",
                }
            ],
        )

        # Fact was recorded
        assert snap2.get_active_fact("spouse_work_shift") is not None

        # Invariant: Dimension scores are 100% PROVABLY UNCHANGED from Turn 1!
        assert snap2.dimensions.trust == initial_trust
        assert snap2.dimensions.emotion_valence == initial_valence
        assert snap2.dimensions.emotion_tension == initial_tension
        assert snap2.dimensions.readiness == initial_readiness

        # Verify change history: Contains fact update, but ZERO dimension change record
        dim_changes = [c for c in snap2.change_history if c.field_path == "dimensions" and c.triggering_turn_id == 2]
        assert len(dim_changes) == 0

    def test_casual_pleasantries_true_negatives_zero_state_perturbation(self):
        """Verifies True Negatives: casual pleasantries and weather chatter are classified

        as non-material, resulting in ZERO state mutations and unchanged state_version.
        """
        manager = ConversationStateManager(call_sid="CA_pleasantries_001")

        # Turn 1: Establish baseline
        t1 = _create_bundle(turn_id=1, speaker_id="client", text="Hello.")
        snap1 = manager.process_turn_bundle(t1)
        version_after_t1 = snap1.state_version
        dims_after_t1 = snap1.dimensions.model_dump()

        # Turn 2: Casual weather chatter
        t2 = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="Looks like it might rain this afternoon.",
            trust=0.69,  # slight acoustic noise
            emotion_valence=0.02,
        )
        classification = manager.materiality_filter.classify_turn(t2, snap1)
        assert classification.is_material is False
        assert len(classification.affected_targets) == 0

        snap2 = manager.process_turn_bundle(t2)

        # State version strictly unchanged (Client Principle #3)
        assert snap2.state_version == version_after_t1
        # Dimensions strictly unchanged
        assert snap2.dimensions.model_dump() == dims_after_t1
        # Last turn tracked
        assert snap2.last_updated_turn_id == 2

    def test_adversarial_false_negative_prevention_embedded_facts(self):
        """Adversarial check: casual conversational chit-chat concealing structural facts

        (e.g. having coffee talking about brother-in-law on deed) must be correctly
        detected as MATERIAL for decision_structure and facts.
        """
        filter_engine = MaterialityFilter()

        # Utterance superficially starts with coffee chatter, but carries title & co-owner fact
        bundle = _create_bundle(
            turn_id=3,
            speaker_id="client",
            text="We were just having coffee talking about how my brother-in-law actually co-owns the title and deed.",
        )
        from copilot.conversation_state_models import ConversationStateSnapshot
        ongoing_state = ConversationStateSnapshot(call_sid="CA_materiality_test_001", state_version=2)
        ongoing_state.dimensions.trust = 0.70
        ongoing_state.dimensions.emotion_tension = 0.20
        ongoing_state.dimensions.emotion_valence = 0.0
        ongoing_state.dimensions.readiness = 0.60
        ongoing_state.dimensions.engagement = 0.65
        ongoing_state.dimensions.pacing = 0.60

        res = filter_engine.classify_turn(bundle, current_state=ongoing_state)

        assert res.is_material is True
        assert "decision_structure" in res.affected_targets
        assert "facts" in res.affected_targets
        # Because tone was neutral and state is already bootstrapped, dimensions is NOT perturbed
        assert "dimensions" not in res.affected_targets

    def test_objection_statement_isolates_to_objections_and_dimensions(self):
        """Validates that an objection utterance selectively targets objections and dimensions,

        without touching decision_structure or contact_compliance.
        """
        filter_engine = MaterialityFilter()

        bundle = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="Six percent is just way too high. That commission is way too steep.",
            recurrence_id="rec_commission_01",
        )
        res = filter_engine.classify_turn(bundle)

        assert res.is_material is True
        assert "objections" in res.affected_targets
        assert "dimensions" in res.affected_targets
        assert "decision_structure" not in res.affected_targets
        assert "contact_compliance" not in res.affected_targets

    def test_contact_boundary_targets_compliance_and_dimensions(self):
        """Validates that a hard contact boundary targets contact_compliance and dimensions

        to immediately enforce protection and collapse readiness.
        """
        filter_engine = MaterialityFilter()

        bundle = _create_bundle(
            turn_id=4,
            speaker_id="client",
            text="Stop calling me. Take my number off your list immediately or I will sue.",
            boundary=1.0,
        )
        res = filter_engine.classify_turn(bundle)

        assert res.is_material is True
        assert "contact_compliance" in res.affected_targets
        assert "dimensions" in res.affected_targets
        assert "objections" in res.affected_targets  # Hard boundary overrides active objections

    def test_significant_acoustic_emotional_shift_triggers_dimensions(self):
        """Validates that when a prospect experiences a genuine acoustic/emotional shift

        (e.g. sudden surge in tension and negative valence), 'dimensions' is flagged.
        """
        manager = ConversationStateManager(call_sid="CA_emotion_shift_001")

        # Turn 1: Calm baseline
        t1 = _create_bundle(
            turn_id=1,
            speaker_id="client",
            text="Good morning.",
            emotion_valence=0.10,
            emotion_tension=0.10,
        )
        snap1 = manager.process_turn_bundle(t1)

        # Turn 2: Sudden surge in tension (delta >= 0.08)
        t2 = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="I really don't appreciate being pressured like this.",
            emotion_valence=-0.45,
            emotion_tension=0.75,
            trust=0.30,
        )
        classification = manager.materiality_filter.classify_turn(t2, snap1)
        assert classification.is_material is True
        assert "dimensions" in classification.affected_targets

        snap2 = manager.process_turn_bundle(t2)
        assert snap2.dimensions.emotion_tension == 0.75
        assert snap2.dimensions.emotion_valence == -0.45

    def test_weakest_link_confidence_on_materiality(self):
        """Ensures the MaterialityFilter outputs a calibrated confidence following

        Client Principle #8 (Honest Uncertainty) bounded by upstream evidence.
        """
        filter_engine = MaterialityFilter()

        bundle = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="I might consider moving if we get the right price.",
        )
        res = filter_engine.classify_turn(bundle)
        assert 0.0 < res.confidence <= 1.0
        assert res.confidence == round(min(bundle.semantic_confidence, bundle.inference_confidence), 3)

    def test_hard_compliance_and_objection_deterministic_override_guarantee(self):
        """CRITICAL SAFETY TEST: Proves that an upstream hard compliance boundary

        (boundary_score >= 0.85), contact preference, or objection recurrence_id can NEVER
        be suppressed by the Materiality Filter, even on completely innocuous text.
        """
        filter_engine = MaterialityFilter()

        # Scenario 1: Utterance text looks like completely benign weather chatter,
        # but upstream acoustic/compliance pipeline fired a hard boundary (0.95).
        bundle_boundary = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="It is quite sunny outside today.",
            boundary=0.95,
        )
        res_boundary = filter_engine.classify_turn(bundle_boundary)
        assert res_boundary.is_material is True
        assert "contact_compliance" in res_boundary.affected_targets
        assert "objections" in res_boundary.affected_targets
        assert "dimensions" in res_boundary.affected_targets
        assert "Deterministic Override: boundary_score >= 0.70" in res_boundary.reasoning

        # Scenario 2: Benign text, but upstream contact preference fired
        bundle_pref = _create_bundle(
            turn_id=3,
            speaker_id="client",
            text="Have a good afternoon.",
        )
        bundle_pref.contact_preference = "reduced_frequency"
        res_pref = filter_engine.classify_turn(bundle_pref)
        assert res_pref.is_material is True
        assert "contact_compliance" in res_pref.affected_targets
        assert "facts" in res_pref.affected_targets
        assert "Deterministic Override: upstream contact_preference" in res_pref.reasoning

        # Scenario 3: Innocuous text, but recurrence_id is present from Sprint 2 engine
        bundle_obj = _create_bundle(
            turn_id=4,
            speaker_id="client",
            text="I'm still thinking about it.",
            recurrence_id="rec_commission_01",
        )
        res_obj = filter_engine.classify_turn(bundle_obj)
        assert res_obj.is_material is True
        assert "objections" in res_obj.affected_targets
        assert "dimensions" in res_obj.affected_targets
        assert "Deterministic Override: upstream recurrence_id" in res_obj.reasoning

    def test_adversarial_negative_figurative_idiom_brother_in_law_not_structural(self):
        """Adversarial check: casual idiom mentioning family with figurative hyperbole

        ("my brother-in-law always jokes he practically owns the place") must NOT
        be falsely classified as decision_structure or facts.
        """
        manager = ConversationStateManager(call_sid="CA_idiom_test_001")

        # Turn 1: Baseline established
        t1 = _create_bundle(turn_id=1, speaker_id="client", text="Hello.")
        snap1 = manager.process_turn_bundle(t1)
        v1 = snap1.state_version

        # Turn 2: Figurative joke / idiom
        t2 = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="We were just talking about how my brother-in-law always jokes he practically owns the place.",
            trust=0.70,
            emotion_valence=0.0,
            emotion_tension=0.2,
        )
        res = manager.materiality_filter.classify_turn(t2, snap1)

        # Invariant: Neither decision_structure nor facts is triggered
        assert "decision_structure" not in res.affected_targets
        assert "facts" not in res.affected_targets
        assert res.is_material is False

        snap2 = manager.process_turn_bundle(t2)
        # State version strictly unchanged; no bogus stakeholders added
        assert snap2.state_version == v1
        assert len(snap2.facts) == 0

    def test_full_chain_dimension_pass_through_on_real_behavioral_movement(self):
        """Validates the positive direction of Client Principle #3:

        When genuine emotional/acoustic movement occurs (delta >= 0.08), the filter
        correctly classifies 'dimensions' as material, updates the snapshot scores,
        increments state_version, and records an explainability StateChangeRecord.
        """
        manager = ConversationStateManager(call_sid="CA_dim_passthrough_001")

        # Turn 1: Baseline
        t1 = _create_bundle(
            turn_id=1,
            speaker_id="client",
            text="Hello.",
            trust=0.70,
            emotion_valence=0.10,
            emotion_tension=0.10,
        )
        snap1 = manager.process_turn_bundle(t1)
        v1 = snap1.state_version

        # Turn 2: Real acoustic/emotional shift: tension surges from 0.10 to 0.65, trust drops to 0.40
        t2 = _create_bundle(
            turn_id=2,
            speaker_id="client",
            text="Wait a second, this feels way too rushed and I'm very uncomfortable with this pressure.",
            trust=0.40,
            emotion_valence=-0.40,
            emotion_tension=0.65,
        )
        snap2 = manager.process_turn_bundle(t2)

        # Invariant 1: State version increments
        assert snap2.state_version > v1

        # Invariant 2: Dimensions updated to new inference state
        assert snap2.dimensions.trust == 0.40
        assert snap2.dimensions.emotion_tension == 0.65
        assert snap2.dimensions.emotion_valence == -0.40

        # Invariant 3: Explainability change record emitted
        dim_records = [c for c in snap2.change_history if c.field_path == "dimensions" and c.triggering_turn_id == 2]
        assert len(dim_records) == 1
        assert "Updated downstream inference dimension scores" in dim_records[0].reason

    def test_casual_idiom_filler_does_not_suppress_explicit_legal_deed(self):
        """Adversarial check in opposite direction:

        A genuine, literal ownership disclosure phrased casually with words like
        'practically' or 'acts like' (e.g. 'he practically lives here now that he's on the deed with us'
        or 'he acts like he owns the place because he actually is on the deed with us')
        must NEVER be rejected by the idiom filter because an explicit legal instrument is named.
        """
        filter_engine = MaterialityFilter()

        # Case 1: "he practically lives here now that he's on the deed with us"
        bundle1 = _create_bundle(
            turn_id=3,
            speaker_id="client",
            text="He practically lives here now that he's on the deed with us.",
        )
        res1 = filter_engine.classify_turn(bundle1)
        assert res1.is_material is True
        assert "decision_structure" in res1.affected_targets
        assert "facts" in res1.affected_targets

        # Case 2: "he acts like he owns the place because he actually is on the deed with us"
        bundle2 = _create_bundle(
            turn_id=4,
            speaker_id="client",
            text="He acts like he owns the place because he actually is on the deed with us.",
        )
        res2 = filter_engine.classify_turn(bundle2)
        assert res2.is_material is True
        assert "decision_structure" in res2.affected_targets
        assert "facts" in res2.affected_targets


