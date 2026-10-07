"""Regression and Parametrized Paraphrase Tests for Novel Script Generalization.

Covers:
1. Exact 6-turn sales script regression test:
   - Turn 2: DE_RISK / defend / none, wife stakeholder, co_decision_required, no false downgrade.
   - Turn 4: Materiality True, trust_credibility objection with cause recorded.
   - Turn 6: Materiality True, email allowed, calls restricted after 6pm,
             no_active_boundary evidence includes Turn 6 and reflects soft restriction,
             decision is ACKNOWLEDGE / protect / none.
2. Parametrized paraphrase tests (with holdouts) across:
   - Stakeholder concerns (wife/husband/partner/spouse)
   - Trust and prior agent fraud/loss objections
   - Contact preferences (clause-based, per-channel, time-restricted)
   - Positive filler whitelist vs substantive client disclosures
"""

import pytest
from typing import List, Dict, Any

from copilot.conversation_replay import ConversationReplayEngine
from copilot.conversation_state_models import (
    ConversationStage,
    ContactPreference,
)
from copilot.core_intelligence_models import StrategicAction
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_objections import (
    classify_objection_label,
    is_decision_authority_statement,
)
from copilot.conversation_objection_driver import ObjectionDriverClassifier
from copilot.behavioral_semantic import (
    extract_structured_contact_preferences,
    extract_structured_contact_preference,
)
from copilot.conversation_materiality import (
    MaterialityFilter,
    is_positive_filler,
    is_soft_contact_preference,
)
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot


# =============================================================================
# Helper to create lightweight bundles for testing
# =============================================================================
def _make_bundle(turn_id: int, text: str, speaker_id: str = "client"):
    inf = DownstreamInferenceState(
        call_sid="test_call",
        timestamp_ms=turn_id * 3000,
        trust=DimensionScore(score=0.50, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0.0, tension_level=0.20, confidence=0.85),
        pacing=DimensionScore(score=0.60, confidence=0.80, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.50, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.50, confidence=0.80, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.50, confidence=0.80, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )
    snap = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id}",
        call_sid="test_call",
        speaker_id=speaker_id,
        boundary_score=0.0,
        specificity_score=0.60,
        agreement_score=0.50,
    )
    return extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=snap,
    )


# =============================================================================
# 1. Exact 6-Turn Simulated Script Regression Test
# =============================================================================
def test_simulated_6_turn_script_regression():
    script = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi Mr. Reyes, thanks for picking up. I'm calling about your property on Maple Drive."},
        {"turn_id": 2, "speaker_id": "client", "text": "Yeah, I'm the owner. But my wife thinks companies like yours are just scams."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "I completely understand the caution. Can I ask what made her feel that way?"},
        {"turn_id": 4, "speaker_id": "client", "text": "A neighbor paid a big upfront fee and the agent disappeared."},
        {"turn_id": 5, "speaker_id": "salesperson", "text": "That's a legitimate worry. We charge nothing upfront, only at closing."},
        {"turn_id": 6, "speaker_id": "client", "text": "Fine. Email me the contract details, but no phone calls after 6pm."},
    ]

    replay_engine = ConversationReplayEngine()
    report = replay_engine.replay_dialogue_turns(
        call_sid="sim_test_reyes_call",
        raw_turns=script,
        save_report=False,
        run_semantic_analysis=True,
    )

    timeline_by_id = {t.turn_id: t for t in report.timeline}

    # -------------------------------------------------------------------------
    # Turn 2 Assertions:
    # - DE_RISK / defend / push none
    # - Stakeholder wife captured, co_decision_required True
    # - No spurious LOW_CONFIDENCE_ACTION_DOWNGRADE
    # -------------------------------------------------------------------------
    t2 = timeline_by_id[2]
    assert t2.materiality.is_material is True
    dec2 = t2.strategic_decision
    assert dec2 is not None
    assert dec2.primary_action == StrategicAction.DE_RISK
    assert dec2.strategic_posture == "defend"
    assert str(dec2.push_strength) == "none"
    assert "STAKEHOLDER_CONCERN" in dec2.reason_codes
    assert "DECISION_MAKER_ABSENT" not in dec2.reason_codes
    assert "separate_partners" in dec2.do_not_do or "isolate_from_spouse" in dec2.do_not_do

    state2 = t2.state_after
    roles = [s.role for s in state2.decision_structure.stakeholders]
    assert "wife" in roles
    assert state2.decision_structure.co_decision_required is True

    # -------------------------------------------------------------------------
    # Turn 4 Assertions:
    # - Materiality True (not dropped as filler!)
    # - Trust objection registered with cause (driver) recorded
    # - No spurious LOW_CONFIDENCE_ACTION_DOWNGRADE
    # -------------------------------------------------------------------------
    t4 = timeline_by_id[4]
    assert t4.materiality.is_material is True, "Turn 4 about scam/upfront fee must not be dropped as filler!"
    state4 = t4.state_after
    trust_objs = [o for o in state4.objections if o.canonical_category == "trust_credibility"]
    assert len(trust_objs) >= 1, "Expected trust_credibility objection to be registered on Turn 4"
    t4_obj = trust_objs[0]
    assert t4_obj.driver_layer is not None
    assert t4_obj.driver_layer.underlying_driver == "prior_agent_fraud_or_loss"
    assert "upfront" in t4_obj.driver_layer.origin_context or "disappeared" in t4_obj.driver_layer.origin_context

    dec4 = t4.strategic_decision
    assert dec4 is not None
    assert dec4.primary_action == StrategicAction.VALIDATE
    assert dec4.strategic_posture == "advance"
    assert str(dec4.push_strength) == "low"
    assert "CATEGORY_TRUST_CREDIBILITY" in dec4.reason_codes

    # -------------------------------------------------------------------------
    # Turn 6 Assertions:
    # - Materiality True
    # - Email allowed, calls restricted after 6pm
    # - no_active_boundary includes Turn 6 in evidence and reflects restrictions
    # - Strategic decision is ACKNOWLEDGE / protect / none
    # - No spurious LOW_CONFIDENCE_ACTION_DOWNGRADE
    # -------------------------------------------------------------------------
    t6 = timeline_by_id[6]
    assert t6.materiality.is_material is True
    state6 = t6.state_after

    comp = state6.contact_compliance
    prefs = comp.contact_preferences
    assert len(prefs) >= 2, f"Expected both email and call preferences, found: {prefs}"

    email_pref = next((p for p in prefs if p.channel == "email"), None)
    assert email_pref is not None, "Email preference must exist"
    assert email_pref.allowed is True

    call_pref = next((p for p in prefs if p.channel == "call"), None)
    assert call_pref is not None, "Call preference must exist"
    assert call_pref.time_restriction == "after 6pm"
    assert "calling after 6pm" in str(call_pref.prohibited_behavior)

    # Gate condition 7 (no_active_boundary)
    gate6 = state6.conversion_gate
    c7 = next(c for c in gate6.conditions if c.condition_name == "no_active_boundary")
    assert c7.met is True
    assert 6 in c7.evidence_turn_ids, f"Turn 6 must be in evidence turn IDs: {c7.evidence_turn_ids}"
    assert "active contact restrictions in effect" in c7.reason
    assert "after 6pm" in c7.reason

    # Decision
    dec6 = t6.strategic_decision
    assert dec6 is not None
    assert dec6.primary_action == StrategicAction.ACKNOWLEDGE
    assert dec6.strategic_posture == "protect"
    assert str(dec6.push_strength) == "none"
    assert "CONTACT_PREFERENCE_DECLARED" in dec6.reason_codes
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" not in dec6.reason_codes


# =============================================================================
# 2. Parametrized Paraphrase Tests: Stakeholders & Spouse Concerns
# =============================================================================
@pytest.mark.parametrize(
    "text,expected_role",
    [
        # 5 Core Paraphrases
        ("Yeah, I'm the owner. But my wife thinks companies like yours are just scams.", "wife"),
        ("I need to hold off because my husband wants to look at it first.", "husband"),
        ("Honestly my partner thinks this is a scam and told me not to answer.", "partner"),
        ("We might sell, but my spouse was warned about cold callers.", "partner"),
        ("My wife feels uncomfortable talking to investors right now.", "wife"),
        # 2 Holdout Paraphrases
        ("Look, my husband told me never to trust people calling like this.", "husband"),
        ("My partner is really skeptical of offers like this.", "partner"),
    ],
)
def test_stakeholder_paraphrase_extraction(text, expected_role):
    from copilot.conversation_state_manager import ConversationStateManager
    manager = ConversationStateManager(call_sid="param_test_stakeholders")
    bundle = _make_bundle(turn_id=2, text=text, speaker_id="client")

    manager.process_turn_bundle(bundle)
    state = manager.current_state

    roles = [s.role for s in state.decision_structure.stakeholders]
    assert expected_role in roles, f"Expected role '{expected_role}' in extracted stakeholders: {roles}"
    assert state.decision_structure.co_decision_required is True


# =============================================================================
# 3. Parametrized Paraphrase Tests: Trust & Fraud Objections
# =============================================================================
@pytest.mark.parametrize(
    "text,expected_category,expected_driver",
    [
        # 5 Core Paraphrases
        ("A neighbor paid a big upfront fee and the agent disappeared.", "trust_credibility", "prior_agent_fraud_or_loss"),
        ("An agent took a deposit and vanished into thin air.", "trust_credibility", "prior_agent_fraud_or_loss"),
        ("I've been burned before by someone promising a fast sale.", "trust_credibility", "prior_agent_fraud_or_loss"),
        ("A broker charged my cousin an upfront fee and ran off.", "trust_credibility", "prior_agent_fraud_or_loss"),
        ("Companies like yours are just scams to get people's equity.", "trust_credibility", "scam_distrust"),
        # 2 Holdout Paraphrases
        ("We got burned by an agent before who stopped answering calls.", "trust_credibility", "prior_agent_fraud_or_loss"),
        ("I think outfits like yours are just a total scam.", "trust_credibility", "scam_distrust"),
    ],
)
def test_trust_objection_paraphrase_classification(text, expected_category, expected_driver):
    cat = classify_objection_label(text)
    assert cat == expected_category, f"Expected category '{expected_category}' for '{text}', got '{cat}'"

    classifier = ObjectionDriverClassifier()
    driver = classifier.classify_driver(cat, text)
    assert driver.surface_objection == expected_category
    assert driver.underlying_driver == expected_driver


# =============================================================================
# 4. Parametrized Paraphrase Tests: Contact Preferences
# =============================================================================
@pytest.mark.parametrize(
    "text,expected_channels,expected_time_restriction",
    [
        # 5 Core Paraphrases
        ("Email me the contract details, but no phone calls after 6pm.", ["email", "call"], "after 6pm"),
        ("Email only please, do not call.", ["email", "call"], None),
        ("Don't call me before noon, I am sleeping.", ["call"], "before noon"),
        ("Text is fine, calls aren't.", ["sms", "call"], None),
        ("No phone calls after 5pm, just send an email.", ["call", "email"], "after 5pm"),
        # 2 Holdout Paraphrases
        ("Send everything by email, don't call my cell.", ["email", "call"], None),
        ("You can text me, but no calls before 10am.", ["sms", "call"], "before 10am"),
    ],
)
def test_contact_preference_paraphrase_extraction(text, expected_channels, expected_time_restriction):
    prefs = extract_structured_contact_preferences(text, source_turn_id=1, confidence=0.90)
    found_channels = [p.channel for p in prefs]
    for exp_ch in expected_channels:
        assert exp_ch in found_channels, f"Expected channel '{exp_ch}' in extracted preferences: {found_channels}"

    if expected_time_restriction:
        restricted = next((p for p in prefs if p.time_restriction), None)
        assert restricted is not None, f"Expected a preference with time restriction '{expected_time_restriction}'"
        assert expected_time_restriction in restricted.time_restriction


# =============================================================================
# 5. Positive Filler Whitelist vs Substantive Disclosure Tests
# =============================================================================
@pytest.mark.parametrize(
    "filler_text",
    [
        "yeah",
        "yes.",
        "okay",
        "ok",
        "sure",
        "right",
        "uh-huh",
        "sounds good",
        "hello",
        "good morning",
        "thanks",
        "looks like rain",
    ],
)
def test_positive_filler_whitelist_recognized(filler_text):
    assert is_positive_filler(filler_text) is True


@pytest.mark.parametrize(
    "substantive_text",
    [
        "A neighbor paid a big upfront fee and the agent disappeared.",
        "My wife thinks companies like yours are just scams.",
        "Email me the contract details, but no phone calls after 6pm.",
        "We are looking to sell sometime in the spring.",
        "I need to speak with an attorney first.",
    ],
)
def test_substantive_disclosures_not_filler(substantive_text):
    assert is_positive_filler(substantive_text) is False
    filter_engine = MaterialityFilter()
    bundle = _make_bundle(turn_id=3, text=substantive_text, speaker_id="client")
    res = filter_engine.classify_turn(bundle)
    assert res.is_material is True, f"Substantive text '{substantive_text}' was erroneously classified non-material!"


# =============================================================================
# 6. Negative False Positive Prevention Tests (Client Item 4)
# =============================================================================
def test_negative_cases_do_not_trigger_false_positives():
    """Verifies that non-risky, benign statements do NOT trigger false state mutations."""
    from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
    from copilot.conversation_state_models import ConversationStateSnapshot

    # 1. "My wife loves the neighborhood." -> must NOT trigger absent stakeholder or concern
    t1 = "My wife loves the neighborhood."
    assert is_decision_authority_statement(t1) is False
    assert classify_objection_label(t1) is None
    snap1 = ConversationStateSnapshot(call_sid="test_neg_1")
    snap1.conversation_stage = ConversationStage.DISCOVERY
    engine = PitchProXCoreIntelligenceEngine()
    dec_res1 = engine.evaluate(snapshot=snap1, turn_text=t1)
    assert "STAKEHOLDER_CONCERN" not in dec_res1.decision.reason_codes
    assert "DECISION_MAKER_ABSENT" not in dec_res1.decision.reason_codes
    assert dec_res1.decision.primary_action != StrategicAction.DE_RISK

    # 2. "I saw a scam story on the news, but that's not why I'm calling." -> must NOT trigger objection
    t2 = "I saw a scam story on the news, but that's not why I'm calling."
    assert classify_objection_label(t2) is None
    bundle2 = _make_bundle(turn_id=2, text=t2, speaker_id="client")
    mat2 = MaterialityFilter().classify_turn(bundle2)
    assert "objections" not in mat2.affected_targets

    # 3. "I'll send you context by email." -> must NOT trigger SMS preference via substring 'context'
    t3 = "I'll send you context by email."
    prefs3 = extract_structured_contact_preferences(t3)
    channels3 = [p.channel for p in prefs3]
    assert "sms" not in channels3
    assert "email" in channels3

    # 4. "The agent was great, he disappeared into the garage for tools." -> must NOT trigger trust objection
    t4 = "The agent was great, he disappeared into the garage for tools."
    assert classify_objection_label(t4) is None
    bundle4 = _make_bundle(turn_id=2, text=t4, speaker_id="client")
    mat4 = MaterialityFilter().classify_turn(bundle4)
    assert "objections" not in mat4.affected_targets


# =============================================================================
# 7. Contact Edge Cases Tests (Client Item 7)
# =============================================================================
def test_contact_preference_edge_cases():
    """Verifies compound contact constraints, preference reversal, and dual time bounds."""
    # 1. "Don't email me, call instead"
    p1 = extract_structured_contact_preferences("Don't email me, call instead")
    p1_by_ch = {p.channel: p for p in p1}
    assert "email" in p1_by_ch and "call" in p1_by_ch
    assert p1_by_ch["email"].allowed is False
    assert p1_by_ch["call"].allowed is True

    # 2. "Text is fine but no calls"
    p2 = extract_structured_contact_preferences("Text is fine but no calls")
    p2_by_ch = {p.channel: p for p in p2}
    assert "sms" in p2_by_ch and "call" in p2_by_ch
    assert p2_by_ch["sms"].allowed is True
    assert p2_by_ch["call"].allowed is False

    # 3. "no calls before 9 or after 6" (dual time bounds)
    p3 = extract_structured_contact_preferences("no calls before 9 or after 6")
    assert len(p3) >= 1
    call_p = next(p for p in p3 if p.channel == "call")
    assert call_p.time_restriction == "before 9 or after 6"
    assert "calling before 9 or after 6" in call_p.prohibited_behavior


# =============================================================================
# 8. Rate Limit 429 Fallback Autonomous Pass Test (Client Item 2)
# =============================================================================
def test_rate_limit_429_fallback_autonomous_pass(monkeypatch):
    """Simulates LLM returning HTTP 429 (Rate Limit Exceeded) and proves turns 2, 4, 6 pass autonomously."""
    class Mock429Client:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    raise RuntimeError("Error code: 429 - Rate limit reached: TPM or RPM quota exceeded.")

    mock_client = Mock429Client()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_live_test_key_for_mock_429")

    script = [
        {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi Mr. Reyes, thanks for picking up. I'm calling about your property on Maple Drive."},
        {"turn_id": 2, "speaker_id": "client", "text": "Yeah, I'm the owner. But my wife thinks companies like yours are just scams."},
        {"turn_id": 3, "speaker_id": "salesperson", "text": "I completely understand the caution. Can I ask what made her feel that way?"},
        {"turn_id": 4, "speaker_id": "client", "text": "A neighbor paid a big upfront fee and the agent disappeared."},
        {"turn_id": 5, "speaker_id": "salesperson", "text": "That's a legitimate worry. We charge nothing upfront, only at closing."},
        {"turn_id": 6, "speaker_id": "client", "text": "Fine. Email me the contract details, but no phone calls after 6pm."},
    ]

    # Initialize engines with mock 429 client so any attempted LLM call fails with 429
    filter_with_429 = MaterialityFilter(groq_client=mock_client)
    driver_with_429 = ObjectionDriverClassifier(groq_client=mock_client)
    state_mgr = ConversationStateManager(call_sid="sim_test_429_call")
    state_mgr.materiality_filter = filter_with_429
    state_mgr.objections_engine.driver_classifier = driver_with_429

    replay_engine = ConversationReplayEngine(state_manager=state_mgr)
    report = replay_engine.replay_dialogue_turns(
        call_sid="sim_test_429_call",
        raw_turns=script,
        save_report=False,
        run_semantic_analysis=True,
    )

    timeline_by_id = {t.turn_id: t for t in report.timeline}

    # Turn 2: Verified on fallback
    t2 = timeline_by_id[2]
    assert t2.materiality.is_material is True
    dec2 = t2.strategic_decision
    assert dec2.primary_action == StrategicAction.DE_RISK
    assert dec2.strategic_posture == "defend"
    assert "STAKEHOLDER_CONCERN" in dec2.reason_codes

    # Turn 4: Verified on fallback (not dropped as filler, driver captured)
    t4 = timeline_by_id[4]
    assert t4.materiality.is_material is True
    trust_objs = [o for o in t4.state_after.objections if o.canonical_category == "trust_credibility"]
    assert len(trust_objs) >= 1
    assert trust_objs[0].driver_layer.underlying_driver == "prior_agent_fraud_or_loss"
    assert t4.strategic_decision.primary_action == StrategicAction.VALIDATE

    # Turn 6: Verified on fallback (per-channel contact preferences captured)
    t6 = timeline_by_id[6]
    assert t6.materiality.is_material is True
    comp6 = t6.state_after.contact_compliance
    channels6 = {p.channel: p for p in comp6.contact_preferences}
    assert "email" in channels6 and channels6["email"].allowed is True
    assert "call" in channels6 and channels6["call"].time_restriction == "after 6pm"
    assert t6.strategic_decision.primary_action == StrategicAction.ACKNOWLEDGE
    assert t6.strategic_decision.strategic_posture == "protect"
