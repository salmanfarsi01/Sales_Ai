from copilot.conversation_state_models import (
    ConversationStage,
    ConversationStateSnapshot,
    MomentumBreakdown,
    ObjectionRecord,
)
from copilot.core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
)
from copilot.llm_response_gateway import Prompt
from copilot.observation_engine import PromptObservationEngine
from copilot.observation_models import (
    ObservationConfig,
    RepPromptBehavior,
)


def test_scenario_11_rep_paraphrases_prompt_effectively():
    """Spec 11 §13.1 Scenario 11: Rep paraphrases prompt effectively ->

    Strategic adherence recognized despite non-verbatim wording.
    """
    engine = PromptObservationEngine()
    snapshot = ConversationStateSnapshot(
        call_sid="call_obs_11",
        state_version=12,
        objections=[
            ObjectionRecord(
                objection_id="obj_comm",
                canonical_category="financial_commission",
                initial_statement="Your commission is too high",
                latest_statement="Your commission is too high",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="active",
            )
        ],
    )
    decision = StrategicDecision(
        call_id="call_obs_11",
        source_state_version=12,
        strategic_objective="Validate seller fee concern and transition to net walk-away",
        primary_action=StrategicAction.VALIDATE,
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=12,
        text="That makes complete sense—let's focus directly on what matters most for your specific situation.",
        strategic_action=StrategicAction.VALIDATE,
        strategic_objective=decision.strategic_objective,
        max_prompt_words=20,
    )

    # Rep paraphrases cleanly in their own conversational words
    rep_spoken = "I completely understand where you are coming from on the fee, that is totally natural to feel."
    prospect_reactions = [
        {"turn_id": "turn_101", "text": "Yeah, exactly, I just want to make sure I am not overpaying."}
    ]

    outcome = engine.observe_turn(
        prompt=prompt,
        decision=decision,
        rep_utterance=rep_spoken,
        rep_turn_id="turn_100",
        initial_snapshot=snapshot,
        prospect_reactions=prospect_reactions,
    )

    assert outcome.rep_behavior == RepPromptBehavior.STRATEGIC_PARAPHRASE
    assert 0.70 <= outcome.adherence_score <= 0.90
    assert outcome.structural_action_executed is True
    assert outcome.topic_continuity is True
    assert outcome.violation_reason is None
    assert "turn_101" in outcome.prospect_reaction_turn_ids


def test_scenario_12_rep_deviates_and_damages_trust():
    """Spec 11 §13.1 Scenario 12: Rep ignores prompt and damages trust ->

    Harmful deviation logic and negative state reaction captured.
    """
    engine = PromptObservationEngine()
    snapshot_initial = ConversationStateSnapshot(
        call_sid="call_obs_12",
        state_version=15,
        objections=[
            ObjectionRecord(
                objection_id="obj_price",
                canonical_category="financial_commission",
                initial_statement="I am not paying six percent",
                latest_statement="I am not paying six percent",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="active",
            )
        ],
    )
    snapshot_initial.dimensions.trust = 0.60
    snapshot_initial.momentum = MomentumBreakdown(momentum_score=55.0, trend="stable")

    decision = StrategicDecision(
        call_id="call_obs_12",
        source_state_version=15,
        strategic_objective="Validate seller fee concern",
        primary_action=StrategicAction.VALIDATE,
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=15,
        text="That makes complete sense—let's focus directly on what matters most for your specific situation.",
        strategic_action=StrategicAction.VALIDATE,
        strategic_objective=decision.strategic_objective,
    )

    # Rep deviates aggressively, attacking prospect's objection on topic
    rep_spoken = "Well if you cut the commission you get what you pay for and cheap agents fail."

    # Prospect reacts negatively: trust drops, momentum stalls
    snapshot_final = snapshot_initial.model_copy(deep=True)
    snapshot_final.dimensions.trust = 0.40  # Delta trust = -0.20
    snapshot_final.momentum = MomentumBreakdown(momentum_score=45.0, trend="regressing")  # Delta momentum = -10.0

    prospect_reactions = [
        {"turn_id": "turn_121", "text": "Excuse me? Are you calling me cheap?"}
    ]

    outcome = engine.observe_turn(
        prompt=prompt,
        decision=decision,
        rep_utterance=rep_spoken,
        rep_turn_id="turn_120",
        initial_snapshot=snapshot_initial,
        prospect_reactions=prospect_reactions,
        final_snapshot=snapshot_final,
    )

    assert outcome.rep_behavior == RepPromptBehavior.HARMFUL_DEVIATION
    assert outcome.adherence_score <= 0.20
    assert outcome.observed_state_delta.get("trust") == -0.20
    assert outcome.observed_state_delta.get("momentum") == -10.0


def test_priority1_unconditional_guardrail_override_defeats_warm_prospect():
    """Spec 01 §10 Priority Rule: Breach of do_not_do (premature_close) unconditionally forces

    HARMFUL_DEVIATION, even if single-turn prospect reaction was positive.
    """
    engine = PromptObservationEngine()
    snapshot = ConversationStateSnapshot(call_sid="call_obs_guard", state_version=8)
    snapshot.dimensions.trust = 0.50

    decision = StrategicDecision(
        call_id="call_obs_guard",
        source_state_version=8,
        strategic_objective="Explore timeline needs",
        primary_action=StrategicAction.QUESTION,
        do_not_do=["premature_close"],
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=8,
        text="What would be the most important outcome for you if you were to make a move?",
        strategic_action=StrategicAction.QUESTION,
        strategic_objective=decision.strategic_objective,
    )

    # Rep violates prohibition by pushing for an immediate listing agreement
    rep_spoken = "Look are you ready to list with me right now and sign the contract today?"

    # Prospect answers politely with positive sentiment
    snapshot_final = snapshot.model_copy(deep=True)
    snapshot_final.dimensions.trust = 0.55  # Positive delta!

    prospect_reactions = [
        {"turn_id": "turn_81", "text": "Haha well you are enthusiastic, let's see."}
    ]

    outcome = engine.observe_turn(
        prompt=prompt,
        decision=decision,
        rep_utterance=rep_spoken,
        rep_turn_id="turn_80",
        initial_snapshot=snapshot,
        prospect_reactions=prospect_reactions,
        final_snapshot=snapshot_final,
    )

    # Priority 1 MUST override: positive prospect delta is ignored when do_not_do breached
    assert outcome.rep_behavior == RepPromptBehavior.HARMFUL_DEVIATION
    assert outcome.adherence_score == 0.05
    assert "PROHIBITED_DIRECTIVE_BREACH" in outcome.violation_reason


def test_priority1_unverified_social_proof_improvisation_breach():
    """Spec 01 §9 & §11: Improvising unverified social proof claims triggers unconditional

    HARMFUL_DEVIATION in Priority 1.
    """
    engine = PromptObservationEngine()
    snapshot = ConversationStateSnapshot(call_sid="call_sp_breach", state_version=10)
    decision = StrategicDecision(
        call_id="call_sp_breach",
        source_state_version=10,
        strategic_objective="Validate situation",
        primary_action=StrategicAction.VALIDATE,
        do_not_do=["fabricate_client_results", "claim_prior_customer_outcomes"],
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=10,
        text="That makes complete sense—let's focus directly on what matters most for your specific situation.",
        strategic_action=StrategicAction.VALIDATE,
        strategic_objective=decision.strategic_objective,
    )

    # Rep invents unverified social proof testimonials
    rep_spoken = "We have helped several homeowners on your street achieve a clean profitable sale."

    outcome = engine.observe_turn(
        prompt=prompt,
        decision=decision,
        rep_utterance=rep_spoken,
        rep_turn_id="turn_90",
        initial_snapshot=snapshot,
        verified_facts_present=False,  # Unverified!
    )

    assert outcome.rep_behavior == RepPromptBehavior.HARMFUL_DEVIATION
    assert outcome.adherence_score == 0.05
    assert outcome.violation_reason == "FABRICATED_UNVERIFIED_CLAIM_BREACH"


def test_asr_safe_question_detection_without_question_mark():
    """Deepgram 8kHz ASR often drops trailing question marks.

    Question syntax must be recognized via auxiliary inversions and interrogative pronouns.
    """
    engine = PromptObservationEngine()
    snapshot = ConversationStateSnapshot(call_sid="call_asr_q", state_version=5)
    decision = StrategicDecision(
        call_id="call_asr_q",
        source_state_version=5,
        strategic_objective="Inquire about seller timeline",
        primary_action=StrategicAction.QUESTION,
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=5,
        text="What timeframe are you thinking about for making this move?",
        strategic_action=StrategicAction.QUESTION,
        strategic_objective=decision.strategic_objective,
    )

    # Deepgram transcript with auxiliary inversion and NO question mark
    rep_spoken_inversion = "would you like to walk through the numbers together on friday"
    assert engine._check_structural_action(rep_spoken_inversion, StrategicAction.QUESTION, snapshot) is True

    # Interrogative pronoun at start with NO question mark
    rep_spoken_interrogative = "how soon are you hoping to have the home on the market"
    assert engine._check_structural_action(rep_spoken_interrogative, StrategicAction.QUESTION, snapshot) is True


def test_useful_deviation_with_zero_trust_and_positive_momentum():
    """Uncovered combination: delta_trust = 0, delta_momentum > 0.

    When momentum rises while trust holds flat, unprompted on-topic execution is USEFUL_DEVIATION.
    """
    engine = PromptObservationEngine()
    snapshot_initial = ConversationStateSnapshot(
        call_sid="call_mom_rise",
        state_version=18,
        objections=[
            ObjectionRecord(
                objection_id="obj_timeline",
                canonical_category="timing_market",
                initial_statement="We want to wait until spring",
                latest_statement="We want to wait until spring",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="active",
            )
        ],
    )
    snapshot_initial.dimensions.trust = 0.65
    snapshot_initial.momentum = MomentumBreakdown(momentum_score=50.0, trend="stable")

    decision = StrategicDecision(
        call_id="call_mom_rise",
        source_state_version=18,
        strategic_objective="Explore spring timeline",
        primary_action=StrategicAction.QUESTION,
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=18,
        text="What is leading you to focus on the spring timeline specifically?",
        strategic_action=StrategicAction.QUESTION,
        strategic_objective=decision.strategic_objective,
    )

    # Rep provides an unprompted educational market stat on the spring timeline (action is EDUCATE, not QUESTION)
    rep_spoken = "Typically in this market the spring inventory surge increases buyer competition significantly."

    # Prospect reaction: trust holds flat (delta=0), momentum rises (+6.0)
    snapshot_final = snapshot_initial.model_copy(deep=True)
    snapshot_final.dimensions.trust = 0.65  # Delta = 0.0
    snapshot_final.momentum = MomentumBreakdown(momentum_score=56.0, trend="advancing")  # Delta = +6.0

    prospect_reactions = [
        {"turn_id": "turn_181", "text": "Oh really, I didn't know inventory was that competitive then."}
    ]

    outcome = engine.observe_turn(
        prompt=prompt,
        decision=decision,
        rep_utterance=rep_spoken,
        rep_turn_id="turn_180",
        initial_snapshot=snapshot_initial,
        prospect_reactions=prospect_reactions,
        final_snapshot=snapshot_final,
    )

    assert outcome.rep_behavior == RepPromptBehavior.USEFUL_DEVIATION
    assert 0.40 <= outcome.adherence_score <= 0.60
    assert outcome.observed_state_delta.get("trust") == 0.0
    assert outcome.observed_state_delta.get("momentum") == 6.0


def test_referential_topic_continuity_on_absent_spouse():
    """Tests that referential pronouns ('talk to her', 'both of us') correctly preserve

    topic continuity on absent spouse objections without needing literal keyword matches.
    """
    engine = PromptObservationEngine()
    snapshot = ConversationStateSnapshot(
        call_sid="call_spouse_ref",
        state_version=7,
        objections=[
            ObjectionRecord(
                objection_id="obj_spouse",
                canonical_category="decision_structure_spouse",
                initial_statement="I need to talk to my wife first",
                latest_statement="I need to talk to my wife first",
                first_turn_id=1,
                last_updated_turn_id=1,
                lifecycle_state="active",
            )
        ],
    )
    decision = StrategicDecision(
        call_id="call_spouse_ref",
        source_state_version=7,
        strategic_objective="Acknowledge absent decision maker",
        primary_action=StrategicAction.VALIDATE,
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=7,
        text="That makes complete sense—let's make sure both of you are completely aligned before deciding anything.",
        strategic_action=StrategicAction.VALIDATE,
        strategic_objective=decision.strategic_objective,
    )

    # Rep uses referential phrasing ("her") instead of literal "wife"
    rep_spoken = "That makes complete sense, you should definitely run everything by her first."

    outcome = engine.observe_turn(
        prompt=prompt,
        decision=decision,
        rep_utterance=rep_spoken,
        rep_turn_id="turn_70",
        initial_snapshot=snapshot,
    )

    assert outcome.topic_continuity is True
    assert outcome.rep_behavior == RepPromptBehavior.STRATEGIC_PARAPHRASE


def test_scenario_22_novel_objection_stored_as_learning_candidate():
    """Spec 11 §13.1 Scenario 22 / Spec 05:

    Novel objection appears once -> Stored as learning candidate, not immediate Core doctrine.
    """
    engine = PromptObservationEngine()
    candidate = engine.record_learning_candidate(
        call_id="call_learn_22",
        turn_id="turn_220",
        raw_utterance="We only work with agents who donate their commission to local cat shelters.",
        metadata={"lead_type": "expired", "detected_stage": "objection"},
    )

    assert candidate.isolated_from_core_doctrine is True
    assert candidate.candidate_type == "novel_objection"
    assert candidate.occurrence_count == 1
    assert candidate.call_id == "call_learn_22"
    assert "cat shelters" in candidate.raw_utterance
