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
    ConversionEventStatus,
    ComplianceEvent,
)
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_intelligence_models import StrategicAction
from copilot.llm_response_gateway import LLMResponseGateway, Prompt
from copilot.observation_engine import PromptObservationEngine
from copilot.observation_models import RepPromptBehavior


def _make_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.50,
    emotion_valence: float = 0.0,
    emotion_tension: float = 0.20,
    engagement: float = 0.50,
    agreement: float = 0.50,
    specificity: float = 0.50,
    future_lang: float = 0.50,
    boundary: float = 0.0,
    contact_preference: str = "none",
    call_sid: str = "CA_audit_defect_test",
):
    inference = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=3000 * turn_id,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=0.8),
        pacing=DimensionScore(score=0.50, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.50, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.50, confidence=0.85, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )

    semantic = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
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
    )


# -----------------------------------------------------------------------------
# 1. Teleprompter Spoken Copy Origin (Core vs. Gateway Separation)
# -----------------------------------------------------------------------------
def test_item1_core_intelligence_does_not_generate_spoken_words():
    """Validates that Core Intelligence outputs purely structural StrategicDecision
    parameters (Action, Objective, Constraints) and does NOT synthesize spoken words.
    The teleprompter wording is produced downstream by LLMResponseGateway.
    """
    manager = ConversationStateManager(call_sid="CA_item1_test")
    bundle = _make_bundle(
        turn_id=1,
        speaker_id="client",
        text="Actually, don't call me again.",
        boundary=1.0,
    )
    snapshot = manager.process_turn_bundle(bundle)

    core_engine = PitchProXCoreIntelligenceEngine()
    eval_result = core_engine.evaluate(snapshot=snapshot, turn_speaker="client", turn_text=bundle.utterance_text)

    # Core must output structural decision
    assert eval_result.decision.primary_action == StrategicAction.ACKNOWLEDGE
    assert eval_result.decision.push_strength == "respect_record_exit"
    assert "persuade" in eval_result.decision.do_not_do
    assert eval_result.decision.should_prompt is True
    # Core itself does NOT populate gateway_fallback_stub or final_prompt_text
    assert eval_result.decision.gateway_fallback_stub is None
    assert eval_result.decision.final_prompt_text is None

    # Gateway produces the deterministic fallback stub
    gateway = LLMResponseGateway()
    fallback_text = gateway._deterministic_fallback(eval_result.decision)
    assert "completely respect that" in fallback_text.lower()
    assert "take care" in fallback_text.lower()


# -----------------------------------------------------------------------------
# 2. Readiness = None on Clean Slate & Exclusion from Observation Engine Deltas
# -----------------------------------------------------------------------------
def test_item2_readiness_none_on_insufficient_evidence_and_observation_delta():
    """Validates that unmeasured baseline readiness is None (not 0.0), and that
    ObservationEngine excludes insufficient_evidence snapshots from observed_state_delta
    calculations, preventing phantom +45.0 jumps.
    """
    manager = ConversationStateManager(call_sid="CA_item2_test")
    # Clean slate turn 1
    t1 = _make_bundle(
        turn_id=1,
        speaker_id="salesperson",
        text="Would Thursday work for a quick call?",
    )
    s1 = manager.process_turn_bundle(t1)
    assert s1.readiness is not None
    assert s1.readiness.insufficient_evidence is True
    assert s1.readiness.readiness_score is None

    obs_engine = PromptObservationEngine()
    # Mock observation window with s1 as before_snapshot and an active measured s2 as after_snapshot
    s2 = s1.model_copy(deep=True)
    s2.readiness.insufficient_evidence = False
    s2.readiness.readiness_score = 45.0

    _, delta = obs_engine._attribute_reaction_window(
        prospect_reactions=[{"turn_id": "t2", "text": "okay"}],
        initial_snapshot=s1,
        final_snapshot=s2,
    )
    # When initial_snapshot has insufficient_evidence, readiness delta is excluded (not in delta)
    assert "readiness" not in delta


# -----------------------------------------------------------------------------
# 3. Observation Engine Detector Exemplar for violating_contact_preference & Block 6
# -----------------------------------------------------------------------------
def test_item3_observation_engine_detects_violating_contact_preference():
    """Validates that PromptObservationEngine detects violating_contact_preference
    as an unconditional guardrail breach (HARMFUL_DEVIATION) when the rep ignores
    prospect's contact restriction.
    """
    obs_engine = PromptObservationEngine()
    from copilot.core_intelligence_models import StrategicDecision
    decision = StrategicDecision(
        call_id="CA_item3_test",
        source_state_version=2,
        primary_action=StrategicAction.VALIDATE,
        strategic_objective="Respect contact frequency while checking property status",
        do_not_do=["violating_contact_preference", "prohibited_daily_texting"],
    )
    prompt = Prompt(
        decision_id=decision.decision_id,
        source_state_version=2,
        text="Understood, I'll keep notes and check back with you in a couple weeks.",
        strategic_action=StrategicAction.VALIDATE,
        strategic_objective=decision.strategic_objective,
    )
    snapshot = ConversationStateSnapshot(call_sid="CA_item3_test")

    outcome = obs_engine.observe_turn(
        prompt=prompt,
        decision=decision,
        rep_utterance="I will text you every single day with fresh market updates.",
        rep_turn_id="turn_3",
        initial_snapshot=snapshot,
    )
    assert outcome.rep_behavior == RepPromptBehavior.HARMFUL_DEVIATION


def test_item3_gateway_block_6_includes_contact_preference_directive():
    """Validates that LLMResponseGateway injects strict contact preference directives
    into Block 6 when violating_contact_preference is present in do_not_do.
    """
    gateway = LLMResponseGateway()
    from copilot.core_intelligence_models import StrategicDecision
    decision = StrategicDecision(
        call_id="CA_item3_gw_test",
        source_state_version=2,
        primary_action=StrategicAction.VALIDATE,
        strategic_objective="Acknowledge prospect feedback",
        do_not_do=["violating_contact_preference"],
    )
    snapshot = ConversationStateSnapshot(call_sid="CA_item3_gw_test")
    context = gateway.assemble_context(decision=decision, snapshot=snapshot, facts=[])
    assert "CONTACT PREFERENCE RESTRICTION (CRITICAL)" in context
    assert "Do NOT propose daily texting" in context


# -----------------------------------------------------------------------------
# 4. Negation Filter (Clause-Level) & Scheduling Disambiguation
# -----------------------------------------------------------------------------
def test_item4_negation_filter_clause_level_compound_boundary():
    """Validates that a compound utterance:
    'I don't mind you calling, but don't call me again.'
    is parsed at the clause level so the second clause triggers a confirmed hard boundary.
    """
    manager = ConversationStateManager(call_sid="CA_item4_neg_test")
    bundle = _make_bundle(
        turn_id=2,
        speaker_id="client",
        text="I don't mind you calling, but don't call me again.",
    )
    s = manager.process_turn_bundle(bundle)
    assert s.contact_compliance.hard_boundary_active is True
    assert s.compliance_event is not None
    assert s.compliance_event.event_type == "hard_boundary_confirmed"
    assert s.compliance_event.is_inferred_extension is True


def test_item4_scheduling_disambiguation_is_not_hard_boundary():
    """Validates that:
    'don't call me tomorrow, call Thursday'
    is disambiguated as a scheduling constraint rather than a hard boundary.
    """
    manager = ConversationStateManager(call_sid="CA_item4_sched_test")
    bundle = _make_bundle(
        turn_id=2,
        speaker_id="client",
        text="Don't call me tomorrow, call Thursday.",
    )
    s = manager.process_turn_bundle(bundle)
    assert s.contact_compliance.hard_boundary_active is False
    assert s.contact_compliance.boundary_suspected is False
    assert s.compliance_event is None
    # Gate should remain closed or handle Thursday as logistics constraint
    assert s.conversion_gate.is_open is False


# -----------------------------------------------------------------------------
# 5. Narrow boundary_suspected Scope & Lifecycle Exits
# -----------------------------------------------------------------------------
def test_item5_ordinary_objection_does_not_trigger_boundary_suspected():
    """Ordinary hesitation like 'I need to think about it' must stay in the
    objection lifecycle and NOT trigger boundary_suspected.
    """
    manager = ConversationStateManager(call_sid="CA_item5_obj_test")
    bundle = _make_bundle(
        turn_id=2,
        speaker_id="client",
        text="I need to think about it before making any decisions.",
    )
    s = manager.process_turn_bundle(bundle)
    assert s.contact_compliance.boundary_suspected is False
    assert s.contact_compliance.hard_boundary_active is False


def test_item5_contact_friction_triggers_boundary_suspected_and_expires():
    """Contact friction ('too many calls') triggers boundary_suspected,
    and expires after 2 turns if unconfirmed.
    """
    manager = ConversationStateManager(call_sid="CA_item5_sus_test")
    # Turn 2: contact friction
    t2 = _make_bundle(
        turn_id=2,
        speaker_id="client",
        text="Why are you calling me? I get too many calls.",
    )
    s2 = manager.process_turn_bundle(t2)
    assert s2.contact_compliance.boundary_suspected is True
    assert s2.contact_compliance.boundary_suspected_turn_id == 2

    # Turn 3: Rep addresses concern
    t3 = _make_bundle(turn_id=3, speaker_id="salesperson", text="I completely understand, we try to be respectful of your time.")
    s3 = manager.process_turn_bundle(t3)
    assert s3.contact_compliance.boundary_suspected is True

    # Turn 4: Prospect changes topic / engages normally
    t4 = _make_bundle(turn_id=4, speaker_id="client", text="Okay, what is the property estimate?")
    s4 = manager.process_turn_bundle(t4)
    # Cleared by prospect affirmative engagement or expired (turn 4 - turn 2 >= 2)
    assert s4.contact_compliance.boundary_suspected is False


# -----------------------------------------------------------------------------
# 6. ComplianceEvent Types §4.6 Extensions & Manual Review Requirement
# -----------------------------------------------------------------------------
def test_item6_compliance_event_extensions_and_suppression_review():
    """Validates that hard_boundary_confirmed and boundary_retracted carry
    is_inferred_extension=True and that boundary_retracted requires human compliance review.
    """
    manager = ConversationStateManager(call_sid="CA_item6_comp_test")
    # Confirm boundary
    t1 = _make_bundle(turn_id=1, speaker_id="client", text="Stop calling me.")
    s1 = manager.process_turn_bundle(t1)
    assert s1.compliance_event is not None
    assert s1.compliance_event.event_type == "hard_boundary_confirmed"
    assert s1.compliance_event.is_inferred_extension is True

    # Retract boundary
    t2 = _make_bundle(turn_id=2, speaker_id="client", text="Actually, you can call me, it's fine.")
    s2 = manager.process_turn_bundle(t2)
    assert s2.compliance_event is not None
    assert s2.compliance_event.event_type == "boundary_retracted"
    assert s2.compliance_event.is_inferred_extension is True
    assert s2.compliance_event.details.get("requires_human_compliance_review") is True
    assert s2.compliance_event.details.get("automatic_suppression_removal") is False


# -----------------------------------------------------------------------------
# Follow-up Audit Tests: Items 1, 2, 3, 4 & Smaller Items
# -----------------------------------------------------------------------------
def test_item1_push_strength_constrained_by_gate_status_and_model_validator():
    """Item 1: When gate is closed, push strength must NOT be close-style (two_window_choice, direct_ask).
    Validates ConversationState constraint and StrategicDecision model validator.
    """
    from copilot.core_intelligence_models import StrategicDecision
    from copilot.conversation_state_models import MeetingConversionGate

    manager = ConversationStateManager(call_sid="CA_item1_gate_push")
    # Clean slate turn: gate is closed
    t1 = _make_bundle(
        turn_id=1,
        speaker_id="salesperson",
        text="Would Thursday work for a quick call to go over some options?",
    )
    s1 = manager.process_turn_bundle(t1)
    assert s1.conversion_gate is None or s1.conversion_gate.is_open is False
    assert s1.push_strength.state not in ("two_window_choice", "direct_ask")
    assert s1.push_strength.state == "resolve_then_ask"

    # Turn 2: Prospect responds with contact preference (friction); gate evaluates to CLOSED
    t2 = _make_bundle(
        turn_id=2,
        speaker_id="client",
        text="Please don't start texting me every day.",
        contact_preference="reduced_frequency",
    )
    s2 = manager.process_turn_bundle(t2)
    assert s2.conversion_gate.is_open is False
    assert s2.push_strength.state not in ("two_window_choice", "direct_ask")
    assert s2.push_strength.state == "resolve_then_ask"

    # Model validator on StrategicDecision enforces invariant
    # 1. Closed gate + two_window_choice -> Must raise ValueError
    with pytest.raises(ValueError, match="Invariant violation.*meeting gate is closed"):
        StrategicDecision(
            call_id="CA_test",
            source_state_version=1,
            strategic_objective="Test close",
            primary_action=StrategicAction.QUESTION,
            push_strength="two_window_choice",
            meeting_gate_open=False,
        )

    # 2. Closed gate + COMMITMENT_CLOSE -> Must raise ValueError
    with pytest.raises(ValueError, match="Invariant violation.*meeting gate is closed"):
        StrategicDecision(
            call_id="CA_test",
            source_state_version=1,
            strategic_objective="Test close",
            primary_action=StrategicAction.COMMITMENT_CLOSE,
            push_strength="resolve_then_ask",
            meeting_gate_open=False,
        )

    # 3. Non-closing action + two_window_choice (even if gate None) -> Must raise ValueError
    with pytest.raises(ValueError, match="Invariant violation.*cannot accompany non-closing"):
        StrategicDecision(
            call_id="CA_test",
            source_state_version=1,
            strategic_objective="Test question",
            primary_action=StrategicAction.QUESTION,
            push_strength="two_window_choice",
        )

    # 4. Valid combination: QUESTION + resolve_then_ask + gate closed -> Valid!
    valid_dec = StrategicDecision(
        call_id="CA_test",
        source_state_version=1,
        strategic_objective="Discover preferences",
        primary_action=StrategicAction.QUESTION,
        push_strength="resolve_then_ask",
        meeting_gate_open=False,
    )
    assert valid_dec.push_strength == "resolve_then_ask"


def test_item2_why_are_you_calling_me_treated_as_objection_not_boundary_suspected():
    """Item 2: In cold outreach (expired/FSBO), 'Why are you calling me?' is a standard
    first reaction. It must NOT trigger boundary_suspected.
    """
    manager = ConversationStateManager(call_sid="CA_item2_cold_outreach")
    t1 = _make_bundle(
        turn_id=1,
        speaker_id="client",
        text="Why are you calling me? Who gave you this number?",
    )
    s1 = manager.process_turn_bundle(t1)
    # Must NOT trigger boundary_suspected!
    assert s1.contact_compliance.boundary_suspected is False
    assert s1.contact_compliance.hard_boundary_active is False

    # Core Intelligence must NOT issue a STOP_PERSUADING_CLARIFY
    core_engine = PitchProXCoreIntelligenceEngine()
    eval_result = core_engine.evaluate(snapshot=s1, turn_speaker="client", turn_text=t1.utterance_text)
    assert "BOUNDARY_SUSPECTED" not in eval_result.decision.reason_codes
    assert "STOP_PERSUADING_CLARIFY" not in eval_result.decision.reason_codes
    assert eval_result.decision.primary_action in (StrategicAction.QUESTION, StrategicAction.CLARIFY)


def test_item3_boundary_suspected_lifecycle_clearing_and_t2_expiry():
    """Item 3: boundary_suspected lifecycle:
    1. Naked backchannel ('okay', 'fine') does NOT clear.
    2. Explicit contact permission ('it's fine to call me') DOES clear.
    3. At T+2, confirm and clear are evaluated before expiry.
    4. On expiry, past_contact_friction is stored in facts, and Core applies light constraint.
    """
    # Test A: Naked backchannel does NOT clear, expires at T+2, stores friction fact
    mgr_a = ConversationStateManager(call_sid="CA_item3_lifecycle_a")
    # T0 (turn_id=0): prospect friction
    t0 = _make_bundle(turn_id=0, speaker_id="client", text="You keep calling too much.")
    s0 = mgr_a.process_turn_bundle(t0)
    assert s0.contact_compliance.boundary_suspected is True
    assert s0.contact_compliance.boundary_suspected_turn_id == 0

    # T1 (turn_id=1): rep clarify
    t1 = _make_bundle(turn_id=1, speaker_id="salesperson", text="Understood, I'll keep it light.")
    s1 = mgr_a.process_turn_bundle(t1)
    assert s1.contact_compliance.boundary_suspected is True

    # T2 (turn_id=2): prospect says naked 'okay'
    t2 = _make_bundle(turn_id=2, speaker_id="client", text="Okay.")
    s2 = mgr_a.process_turn_bundle(t2)
    # At T2 (turn_id 2 - 0 = 2 >= 2), naked 'okay' does NOT clear, so it expires!
    assert s2.contact_compliance.boundary_suspected is False
    # Memory fact must be persisted!
    fric_fact = next((f for f in s2.facts if f.fact_key == "past_contact_friction"), None)
    assert fric_fact is not None
    assert fric_fact.status == "active"

    # Core Intelligence must respect memory fact with light constraints
    core = PitchProXCoreIntelligenceEngine()
    eval_res = core.evaluate(snapshot=s2, turn_speaker="client", turn_text=t2.utterance_text)
    assert "high_pressure_closing" in eval_res.decision.do_not_do
    assert "scheduling_push" in eval_res.decision.do_not_do
    assert "PAST_CONTACT_FRICTION_HONORED" in eval_res.decision.reason_codes

    # Test B: Explicit contact statement clears cleanly at T+2
    mgr_b = ConversationStateManager(call_sid="CA_item3_lifecycle_b")
    mgr_b.process_turn_bundle(_make_bundle(turn_id=0, speaker_id="client", text="Stop calling so much."))
    mgr_b.process_turn_bundle(_make_bundle(turn_id=1, speaker_id="salesperson", text="Do you want me to stop calling?"))
    t2_clear = _make_bundle(turn_id=2, speaker_id="client", text="No, it's fine to call me.")
    s2_clear = mgr_b.process_turn_bundle(t2_clear)
    assert s2_clear.contact_compliance.boundary_suspected is False
    assert s2_clear.contact_compliance.hard_boundary_active is False

    # Test C: Confirm at T+2 evaluated before expiry
    mgr_c = ConversationStateManager(call_sid="CA_item3_lifecycle_c")
    mgr_c.process_turn_bundle(_make_bundle(turn_id=0, speaker_id="client", text="Stop calling so much."))
    mgr_c.process_turn_bundle(_make_bundle(turn_id=1, speaker_id="salesperson", text="Do you want me to stop calling?"))
    t2_confirm = _make_bundle(turn_id=2, speaker_id="client", text="Yes, stop calling me right now.")
    s2_confirm = mgr_c.process_turn_bundle(t2_confirm)
    # Confirm takes precedence over expiry!
    assert s2_confirm.contact_compliance.hard_boundary_active is True
    assert s2_confirm.contact_compliance.boundary_suspected is False


def test_item4_scheduling_alternative_bounded_and_adversarial_cases():
    """Item 4: Bounded scheduling alternative:
    1. 'Don't call me tomorrow, call Thursday' passes as scheduling alternative.
    2. Adversarial: 'Don't call me again, call my lawyer' must NOT pass as scheduling -> Hard boundary.
    3. Adversarial: 'Don't call me today, call my lawyer' must NOT pass as scheduling -> Hard boundary.
    4. Ambiguous: 'Don't call me right now, call someone else' falls back to boundary_suspected.
    """
    manager = ConversationStateManager(call_sid="CA_item4_scheduling_filter")

    # 1. Valid scheduling alternative
    t1 = _make_bundle(turn_id=1, speaker_id="client", text="Don't call me tomorrow, call Thursday.")
    s1 = manager.process_turn_bundle(t1)
    assert s1.contact_compliance.hard_boundary_active is False
    assert s1.contact_compliance.boundary_suspected is False

    # 2. Adversarial 1: 'Don't call me again, call my lawyer'
    mgr_adv1 = ConversationStateManager(call_sid="CA_item4_adv1")
    t_adv1 = _make_bundle(turn_id=1, speaker_id="client", text="Don't call me again, call my lawyer.")
    s_adv1 = mgr_adv1.process_turn_bundle(t_adv1)
    assert s_adv1.contact_compliance.hard_boundary_active is True

    # 3. Adversarial 2: 'Don't call me today, call my lawyer'
    mgr_adv2 = ConversationStateManager(call_sid="CA_item4_adv2")
    t_adv2 = _make_bundle(turn_id=1, speaker_id="client", text="Don't call me today, call my lawyer.")
    s_adv2 = mgr_adv2.process_turn_bundle(t_adv2)
    assert s_adv2.contact_compliance.hard_boundary_active is True

    # 4. Ambiguous case: 'Don't call me right now, call someone else'
    mgr_amb = ConversationStateManager(call_sid="CA_item4_amb")
    t_amb = _make_bundle(turn_id=1, speaker_id="client", text="Don't call me right now, call someone else.")
    s_amb = mgr_amb.process_turn_bundle(t_amb)
    # Does NOT pass as scheduling, falls back to boundary_suspected
    assert s_amb.contact_compliance.boundary_suspected is True
    assert s_amb.contact_compliance.hard_boundary_active is False


def test_smaller_items_block6_templating_and_compliance_defaults():
    """Smaller Items:
    1. Block 6 templates dynamically from ContactPreference fields ('email only', 'no calls before 10').
    2. ComplianceEvent class default for is_inferred_extension is False.
    """
    # 1. Verify ComplianceEvent class default
    event = ComplianceEvent(
        event_type="consent_issue",
        occurred_at="2026-09-28T10:00:00Z",
    )
    assert event.is_inferred_extension is False

    # 2. Test Block 6 templating for 'email only'
    gateway = LLMResponseGateway()
    mgr = ConversationStateManager(call_sid="CA_small_block6")
    t1 = _make_bundle(turn_id=1, speaker_id="client", text="Please communicate by email only.")
    s1 = mgr.process_turn_bundle(t1)

    core = PitchProXCoreIntelligenceEngine()
    eval_res = core.evaluate(snapshot=s1, turn_speaker="client", turn_text=t1.utterance_text)
    prompt_context = gateway.assemble_context(
        decision=eval_res.decision,
        snapshot=s1,
        facts=[],
    )
    assert "email only" in prompt_context.lower()

    # 3. Test Block 6 templating for 'no calls before 10'
    mgr2 = ConversationStateManager(call_sid="CA_small_timing")
    t2 = _make_bundle(turn_id=1, speaker_id="client", text="Don't call me before 10am.")
    s2 = mgr2.process_turn_bundle(t2)
    eval_res2 = core.evaluate(snapshot=s2, turn_speaker="client", turn_text=t2.utterance_text)
    prompt_context2 = gateway.assemble_context(
        decision=eval_res2.decision,
        snapshot=s2,
        facts=[],
    )
    assert "before 10" in prompt_context2.lower()


def test_simulation_invariant_action_gate_push():
    """Validates the architectural invariant across simulations:
    (primary_action, gate status, and push_strength must be a valid combination).
    1. If meeting gate is closed, push_strength cannot be direct_ask or two_window_choice.
    2. If meeting gate is closed, primary_action cannot be COMMITMENT_CLOSE.
    3. If primary_action is non-closing (QUESTION, CLARIFY, etc.), push_strength cannot be close-style.
    """
    import json
    import os
    import glob

    sim_files = glob.glob("reports/synthetic/conversation_state_sim_mukmxsle.json")
    assert sim_files, "mukmxsle simulation report must exist"

    for fpath in sim_files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        timeline = data.get("timeline", [])
        for step in timeline:
            tid = step.get("turn_id")
            state_after = step.get("state_after", {})
            gate = state_after.get("conversion_gate")
            gate_open = gate.get("is_open") if gate else False
            push = state_after.get("push_strength", {}).get("state")
            strat = step.get("strategic_decision") or {}
            action = strat.get("primary_action")

            if gate_open is False:
                assert push not in ("two_window_choice", "direct_ask"), (
                    f"Turn {tid} in {fpath} violated invariant: Gate is closed, but push is {push}!"
                )
                if action:
                    assert action != "commitment_close", (
                        f"Turn {tid} in {fpath} violated invariant: Gate is closed, but action is COMMITMENT_CLOSE!"
                    )

            if action in ("question", "clarify", "acknowledge", "validate", "educate"):
                assert push not in ("two_window_choice", "direct_ask"), (
                    f"Turn {tid} in {fpath} violated invariant: Non-closing action {action} paired with {push}!"
                )

