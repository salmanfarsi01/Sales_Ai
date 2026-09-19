import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_objections import classify_objection_label


def _create_turn_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    agreement: float = 0.0,
    readiness: float = 0.5,
    future_lang: float = 0.0,
    boundary: float = 0.0,
    question_type: str = "none",
    strategy_tag: str = None,
    strategy_source: str = "none",
    recurrence_type: str = "none",
    recurrence_id: str = None,
):
    inference = DownstreamInferenceState(
        call_sid="CA_scripted_call_001",
        timestamp_ms=2500 * turn_id,
        trust=DimensionScore(score=0.75, confidence=0.85, primary_horizon="last_20_30s"),
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
        call_sid="CA_scripted_call_001",
        speaker_id=speaker_id,
        question_type=question_type,
        agreement_score=agreement,
        boundary_score=boundary,
        future_language_score=future_lang,
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
        salesperson_strategy_source=strategy_source,
    )


class TestConversationStateSprint2Objections:
    def test_category_label_classification(self):
        assert classify_objection_label("Your 6 percent commission is too high") == "commission_fee"
        assert classify_objection_label("We want to wait until spring to sell") == "market_timing"
        assert classify_objection_label("My nephew is a realtor and might list it") == "broker_representation"
        assert classify_objection_label("I need to talk to my wife first") == "spouse_authority"
        assert classify_objection_label("That price is too low, we won't lowball") == "pricing_value"
        assert classify_objection_label("Just looking around right now, not ready") == "general_hesitation"
        assert classify_objection_label("The weather is nice today") is None

    def test_realistic_scripted_dialogue_lifecycle(self):
        """Exercises a realistic, multi-turn sales dialogue covering:

        1. Two distinct objections active simultaneously (commission fee vs. spouse authority).
        2. Speaker-filtered CLARIFICATION on Agent turn only.
        3. Agent reframe attempt DOES NOT prematurely trigger PARTIALLY_RESOLVED.
        4. Prospect pushback keeps objection in unresolved/clarified state.
        5. Genuine prospect partial acceptance triggers PARTIALLY_RESOLVED.
        6. Objection-scoped forward advance RESOLVES commission fee WITHOUT resolving spouse authority.
        7. Reactivation of commission fee preserves stable objection_id, increments recurrence_count,
           and retains full strategy history (Client Principle #5).
        """
        manager = ConversationStateManager(call_sid="CA_scripted_call_001")

        # Turn 1 (Prospect): Raises commission objection -> UNRESOLVED
        t1 = _create_turn_bundle(
            turn_id=1,
            speaker_id="client",
            text="Look, I don't know, 6 percent is way too high. That commission is just too steep for me.",
            recurrence_id="rec_obj_fee_01",
        )
        snap1 = manager.process_turn_bundle(t1)
        assert len(snap1.objections) == 1
        obj_fee = snap1.objections[0]
        fee_id = obj_fee.objection_id
        assert obj_fee.canonical_category == "commission_fee"
        assert obj_fee.lifecycle_state == "unresolved"
        assert obj_fee.recurrence_id == "rec_obj_fee_01"

        # Turn 2 (Prospect): Raises second distinct objection -> spouse authority
        t2 = _create_turn_bundle(
            turn_id=2,
            speaker_id="client",
            text="Plus my wife isn't on board yet. She wants to wait and talk about it first.",
            recurrence_id="rec_obj_spouse_02",
        )
        snap2 = manager.process_turn_bundle(t2)
        assert len(snap2.objections) == 2
        obj_spouse = snap2.objections[1]
        spouse_id = obj_spouse.objection_id
        assert obj_spouse.canonical_category == "spouse_authority"
        assert obj_spouse.lifecycle_state == "unresolved"

        # Turn 3 (Agent): Clarifying question directed at commission fee -> CLARIFIED
        t3 = _create_turn_bundle(
            turn_id=3,
            speaker_id="salesperson",
            text="I completely understand, John. Regarding the fee, are you comparing our full-service marketing to a discount broker, or is it the final net proceeds in your pocket you care most about?",
            question_type="clarifying",
        )
        snap3 = manager.process_turn_bundle(t3)
        # Commission fee must be CLARIFIED; Spouse authority must remain UNRESOLVED
        fee_turn3 = next(o for o in snap3.objections if o.objection_id == fee_id)
        spouse_turn3 = next(o for o in snap3.objections if o.objection_id == spouse_id)
        assert fee_turn3.lifecycle_state == "clarified"
        assert spouse_turn3.lifecycle_state == "unresolved"

        # Turn 4 (Prospect): Answers clarifying question
        t4 = _create_turn_bundle(
            turn_id=4,
            speaker_id="client",
            text="It's about the net. I need to walk away with at least $420,000 clean.",
            agreement=0.20,
        )
        snap4 = manager.process_turn_bundle(t4)
        fee_turn4 = next(o for o in snap4.objections if o.objection_id == fee_id)
        assert fee_turn4.lifecycle_state == "clarified"

        # Turn 5 (Agent): Attempts reframe strategy -> logs strategy, does NOT prematurely resolve!
        t5 = _create_turn_bundle(
            turn_id=5,
            speaker_id="salesperson",
            text="Understood. Our average listing sells for 3.8% closer to list price than market median, which actually nets you $14,000 more even after our commission.",
            strategy_tag="net_proceeds_comps",
            strategy_source="strategic_engine",
        )
        snap5 = manager.process_turn_bundle(t5)
        fee_turn5 = next(o for o in snap5.objections if o.objection_id == fee_id)
        # Invariant: Strategy is logged, but state remains CLARIFIED (awaiting prospect acceptance)
        assert "net_proceeds_comps" in fee_turn5.attempted_strategies
        assert fee_turn5.lifecycle_state == "clarified"

        # Turn 6 (Prospect): Pushes back against reframe -> remains CLARIFIED (does NOT flip to partially_resolved!)
        t6 = _create_turn_bundle(
            turn_id=6,
            speaker_id="client",
            text="I don't know if I buy that math. Sounds like typical salesman talk.",
            agreement=0.15,
        )
        snap6 = manager.process_turn_bundle(t6)
        fee_turn6 = next(o for o in snap6.objections if o.objection_id == fee_id)
        assert fee_turn6.lifecycle_state == "clarified"

        # Turn 7 (Agent): Tries second reframe with hyperlocal proof
        t7 = _create_turn_bundle(
            turn_id=7,
            speaker_id="salesperson",
            text="Fair enough John. I can show you the exact three comps on your street that closed last month where that math proved out.",
            strategy_tag="hyperlocal_comps",
            strategy_source="strategic_engine",
        )
        snap7 = manager.process_turn_bundle(t7)
        fee_turn7 = next(o for o in snap7.objections if o.objection_id == fee_id)
        assert "hyperlocal_comps" in fee_turn7.attempted_strategies

        # Turn 8 (Prospect): Prospect shows genuine partial acceptance -> PARTIALLY_RESOLVED!
        t8 = _create_turn_bundle(
            turn_id=8,
            speaker_id="client",
            text="Well... okay, I can see how that math might work if the comps hold up, but I'm still hesitant on paying that much.",
            agreement=0.52,  # Partial acceptance of reframe evidence
        )
        snap8 = manager.process_turn_bundle(t8)
        fee_turn8 = next(o for o in snap8.objections if o.objection_id == fee_id)
        assert fee_turn8.lifecycle_state == "partially_resolved"
        # Spouse objection remains UNRESOLVED or aged to DORMANT (not resolved!)
        assert next(o for o in snap8.objections if o.objection_id == spouse_id).lifecycle_state in ("unresolved", "dormant")

        # Turn 9 (Agent): Sets up meeting to inspect the proof
        t9 = _create_turn_bundle(
            turn_id=9,
            speaker_id="salesperson",
            text="Let's do this: I'll bring the actual closed HUD statements on Thursday at 3 PM so you can verify the numbers with your own eyes.",
        )
        manager.process_turn_bundle(t9)

        # Turn 10 (Prospect): Behavioral advance resolving commission fee -> RESOLVED!
        t10 = _create_turn_bundle(
            turn_id=10,
            speaker_id="client",
            text="Alright, that sounds fair. Show me those closed comps on Thursday at three.",
            agreement=0.85,
            future_lang=0.75,
            readiness=0.82,
        )
        snap10 = manager.process_turn_bundle(t10)
        fee_turn10 = next(o for o in snap10.objections if o.objection_id == fee_id)
        spouse_turn10 = next(o for o in snap10.objections if o.objection_id == spouse_id)

        # Invariant: Commission fee is RESOLVED
        assert fee_turn10.lifecycle_state == "resolved"
        assert fee_turn10.resolution_evidence is not None

        # Critical Scope Check: Spouse authority was NEVER addressed and MUST REMAIN UNRESOLVED / DORMANT (NOT RESOLVED!)
        # even though call-wide readiness is 0.82!
        assert spouse_turn10.lifecycle_state in ("unresolved", "dormant")
        assert spouse_turn10.lifecycle_state != "resolved"

        # Turn 11 (Prospect): Previously resolved objection returns -> REACTIVATED (Client Principle #5)
        t11 = _create_turn_bundle(
            turn_id=11,
            speaker_id="client",
            text="Wait, before we lock this in, I was just thinking... even with those comps, what if I still feel like 6 percent is too expensive on Thursday?",
            recurrence_type="same_objection_repeated",
            recurrence_id="rec_obj_fee_01",
        )
        snap11 = manager.process_turn_bundle(t11)
        fee_turn11 = next(o for o in snap11.objections if o.objection_id == fee_id)

        # Identity preserved (Client Principle #5)
        assert fee_turn11.objection_id == fee_id
        assert fee_turn11.lifecycle_state == "reactivated"
        assert fee_turn11.recurrence_count == 2
        # Full strategy history intact
        assert fee_turn11.attempted_strategies == ["net_proceeds_comps", "hyperlocal_comps"]
        assert "what if I still feel like 6 percent is too expensive" in fee_turn11.latest_statement

    def test_prospect_clarifying_question_does_not_clarify_objection(self):
        """Confirm CLARIFIED is speaker-filtered: a prospect asking a question cannot clarify their own objection."""
        manager = ConversationStateManager(call_sid="CA_speaker_filter_001")

        # Turn 1: Client raises objection
        t1 = _create_turn_bundle(turn_id=1, speaker_id="client", text="Your fee is too expensive.")
        manager.process_turn_bundle(t1)

        # Turn 2: Client asks clarifying question
        t2 = _create_turn_bundle(turn_id=2, speaker_id="client", text="What does that fee even cover?", question_type="clarifying")
        snap2 = manager.process_turn_bundle(t2)

        # Objection must remain UNRESOLVED
        assert snap2.objections[0].lifecycle_state == "unresolved"

    def test_boundary_cutoff_transitions_all_active_objections(self):
        manager = ConversationStateManager(call_sid="CA_boundary_test_001")

        # Turn 1: Client raises timing objection
        t1 = _create_turn_bundle(turn_id=1, speaker_id="client", text="We want to wait until next year.")
        manager.process_turn_bundle(t1)
        assert manager.current_state.objections[0].lifecycle_state == "unresolved"

        # Turn 2: Client drops hard boundary
        t2 = _create_turn_bundle(turn_id=2, speaker_id="client", text="Stop calling me. Take me off your list.", boundary=1.0)
        snap2 = manager.process_turn_bundle(t2)

        assert snap2.contact_compliance.hard_boundary_active is True
        assert snap2.objections[0].lifecycle_state == "boundary"

    def test_adversarial_label_taxonomy_false_positives(self):
        """Adversarial testing: ensure incidental vocabulary usage does NOT trigger false objection labels."""
        assert classify_objection_label("The property has an HOA fee of $120 per quarter") is None
        assert classify_objection_label("We planted beautiful spring flowers in the garden") is None
        assert classify_objection_label("My wife and I visited our friends in Denver") is None
        assert classify_objection_label("The house has a 4 car detached garage") is None
        assert classify_objection_label("The market for tech jobs in Austin is booming") is None
        assert classify_objection_label("I spoke with my friend yesterday about fishing") is None
