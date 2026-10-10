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
import json
from pathlib import Path
from typing import List, Dict, Any

from copilot.conversation_replay import ConversationReplayEngine
from copilot.conversation_state_models import (
    ConversationStage,
    ContactPreference,
    ConversionEventStatus,
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
from copilot.behavioral_semantic import (
    SemanticFeatureEngine,
    SemanticFeatureSnapshot,
)


@pytest.fixture(autouse=True)
def mock_groq_offline(monkeypatch):
    """Enforces deterministic offline execution across all tests in this module,
    preventing live network latency, external Groq API calls, or non-deterministic rate limits."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key_offline")


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


def test_turn_2_routing_and_stakeholder_isolation():
    """Restores standalone check: Turn 2 routes to DE_RISK/defend/none, records wife stakeholder,
    sets co_decision_required, and isolates stakeholder concerns without false downgrade."""
    bundle = _make_bundle(turn_id=2, text="Yeah, I'm the owner. But my wife thinks companies like yours are just scams.")
    manager = ConversationStateManager(call_sid="sim_test_stand_alone_t2")
    state = manager.process_turn_bundle(bundle)
    from copilot.core_decision_manager import CoreDecisionManager
    dm = CoreDecisionManager(call_sid="sim_test_stand_alone_t2")
    res = dm.evaluate_state(snapshot=state, turn_speaker="client", turn_text=bundle.utterance_text, turn_timestamp_ms=6000, turn_id=2)
    dec = res.decision
    assert dec.primary_action == StrategicAction.DE_RISK
    assert dec.strategic_posture == "defend"
    assert str(dec.push_strength) == "none"
    assert "STAKEHOLDER_CONCERN" in dec.reason_codes
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" not in dec.reason_codes
    roles = [s.role for s in state.decision_structure.stakeholders]
    assert "wife" in roles
    assert state.decision_structure.co_decision_required is True


def test_turn_4_trust_objection_and_cause_recorded():
    """Restores standalone check: Turn 4 is material, records trust_credibility objection,
    captures underlying driver (prior_agent_fraud_or_loss), and routes to VALIDATE/advance/low."""
    bundle = _make_bundle(turn_id=4, text="A neighbor paid a big upfront fee and the agent disappeared.")
    mat = MaterialityFilter().classify_turn(bundle)
    assert mat.is_material is True
    manager = ConversationStateManager(call_sid="sim_test_stand_alone_t4")
    state = manager.process_turn_bundle(bundle)
    trust_objs = [o for o in state.objections if o.canonical_category == "trust_credibility"]
    assert len(trust_objs) >= 1
    assert trust_objs[0].driver_layer.underlying_driver == "prior_agent_fraud_or_loss"
    from copilot.core_decision_manager import CoreDecisionManager
    dm = CoreDecisionManager(call_sid="sim_test_stand_alone_t4")
    res = dm.evaluate_state(snapshot=state, turn_speaker="client", turn_text=bundle.utterance_text, turn_timestamp_ms=12000, turn_id=4)
    dec = res.decision
    assert dec.primary_action == StrategicAction.VALIDATE
    assert dec.strategic_posture == "advance"
    assert str(dec.push_strength) in ("low", "moderate")
    assert "CATEGORY_TRUST_CREDIBILITY" in dec.reason_codes


def test_spoken_prompts_suppressed_on_rep_turns():
    """Restores standalone check: Spoken guidance prompts are strictly suppressed on salesperson turns."""
    from copilot.core_decision_manager import CoreDecisionManager
    from copilot.conversation_state_models import ConversationStateSnapshot
    dm = CoreDecisionManager(call_sid="test_rep_suppression")
    snap = ConversationStateSnapshot(call_sid="test_rep_suppression")
    for rep_text in [
        "I completely understand the caution. Can I ask what made her feel that way?",
        "That's a legitimate worry. We charge nothing upfront, only at closing.",
        "That's fair — a lot of people feel that way before they see the actual numbers.",
    ]:
        res = dm.evaluate_state(snapshot=snap, turn_speaker="salesperson", turn_text=rep_text, turn_timestamp_ms=1000, turn_id=3)
        assert res.decision.should_prompt is False
        assert getattr(res.decision, "spoken_guidance_prompt", None) is None
        assert res.decision.final_prompt_text is None


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

    # 5. Variation: "I read a scam story, but that's unrelated to my situation." -> must NOT trigger objection
    t5 = "I read a scam story, but that's unrelated to my situation."
    assert classify_objection_label(t5) is None
    bundle5 = _make_bundle(turn_id=2, text=t5, speaker_id="client")
    mat5 = MaterialityFilter().classify_turn(bundle5)
    assert "objections" not in mat5.affected_targets

    # 6. Variation: "The agent vanished into a meeting." -> must NOT trigger trust objection
    t6 = "The agent vanished into a meeting."
    assert classify_objection_label(t6) is None
    bundle6 = _make_bundle(turn_id=2, text=t6, speaker_id="client")
    mat6 = MaterialityFilter().classify_turn(bundle6)
    assert "objections" not in mat6.affected_targets

    # 7. Contrast case: "My wife loves the neighborhood but thinks we should wait." -> MUST trigger!
    t7 = "My wife loves the neighborhood but thinks we should wait."
    assert is_decision_authority_statement(t7) is True
    assert classify_objection_label(t7) == "spouse_authority"


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

    # 3. "no calls before 9 or after 6" (dual time bounds stored separately and normalized to 24h)
    p3 = extract_structured_contact_preferences("no calls before 9 or after 6")
    assert len(p3) >= 1
    call_p = next(p for p in p3 if p.channel == "call")
    assert call_p.allowed is True  # Allowed during window, restricted outside
    assert call_p.time_not_before == "09:00"
    assert call_p.time_not_after == "18:00"
    assert call_p.time_restriction == "before 9 or after 6"
    assert "calling before 9 or after 6" in call_p.prohibited_behavior

    # 4. WhatsApp channel detection: "Ping me on WhatsApp, do not ring my cell during work hours."
    p4 = extract_structured_contact_preferences("Ping me on WhatsApp, do not ring my cell during work hours.")
    p4_by_ch = {p.channel: p for p in p4}
    assert "whatsapp" in p4_by_ch and "call" in p4_by_ch
    assert p4_by_ch["whatsapp"].allowed is True
    assert p4_by_ch["call"].allowed is True  # Allowed outside work hours
    assert p4_by_ch["call"].cadence == "specific_times"


def test_contact_timing_enforcement_flags_after_hours():
    """Verifies that is_time_permitted correctly identifies after-hours violations (e.g. proposed 7pm call)."""
    prefs = extract_structured_contact_preferences("no calls before 9 or after 6")
    call_p = next(p for p in prefs if p.channel == "call")
    # Proposed call at 7pm (19:00) must be flagged as prohibited
    permitted_1900, reason_1900 = call_p.is_time_permitted("19:00")
    assert permitted_1900 is False
    assert "Proposed time 19:00 violates restriction: no call after 18:00" in reason_1900

    # Proposed call at 8am (08:00) must be flagged as prohibited
    permitted_0800, reason_0800 = call_p.is_time_permitted("08:00")
    assert permitted_0800 is False
    assert "Proposed time 08:00 violates restriction: no call before 09:00" in reason_0800

    # Proposed call at 2pm (14:00) is permitted
    permitted_1400, reason_1400 = call_p.is_time_permitted("14:00")
    assert permitted_1400 is True


def test_unclassified_material_turn_routing():
    """Verifies that an unclassified material statement defaults safely to CLARIFY / VALIDATE
    with low push rather than defaulting to a blind discovery question."""
    from copilot.core_decision_manager import CoreDecisionManager
    from copilot.conversation_state_models import ConversationStateSnapshot
    snap = ConversationStateSnapshot(call_sid="test_unclass_mat")
    snap.unclassified_material = True
    dm = CoreDecisionManager(call_sid="test_unclass_mat")
    res = dm.evaluate_state(
        snapshot=snap,
        turn_speaker="client",
        turn_text="Something vague but serious was mentioned.",
        turn_timestamp_ms=1000,
        turn_id=2,
    )
    assert res.decision.unclassified_material is True
    assert res.decision.primary_action in (StrategicAction.CLARIFY, StrategicAction.VALIDATE)
    assert str(res.decision.push_strength) == "low"
    assert "UNCLASSIFIED_MATERIAL_CONTENT" in res.decision.reason_codes


# =============================================================================
# 8. Rate Limit 429 Fallback Autonomous Pass Test (Client Item 2)
# =============================================================================
def test_rate_limit_429_fallback_autonomous_pass(monkeypatch):
    """Simulates LLM returning HTTP 429 (Rate Limit Exceeded) and proves turns 2, 4, 6 pass autonomously."""
    call_count = 0

    class Mock429Client:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
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
    sem_with_429 = SemanticFeatureEngine(groq_client=mock_client)

    state_mgr = ConversationStateManager(call_sid="sim_test_429_call")
    state_mgr.materiality_filter = filter_with_429
    state_mgr.objections_engine.driver_classifier = driver_with_429
    state_mgr.semantic_engine = sem_with_429

    replay_engine = ConversationReplayEngine(state_manager=state_mgr)
    report = replay_engine.replay_dialogue_turns(
        call_sid="sim_test_429_call",
        raw_turns=script,
        save_report=False,
        run_semantic_analysis=True,
    )

    # Prove that the LLM was actually called and raised the 429 error
    assert call_count > 0, "LLM was never called; test did not prove fallback under 429!"

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


# =============================================================================
# 9. Gate-Closed Invariant Test Across Real Script JSON Files (Item 5)
# =============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REAL_SCRIPT_FILES = {
    "canonical": DATA_DIR / "script_canonical.json",
    "script_d_filler": DATA_DIR / "script_d_filler.json",
    "script_h_mismatched": DATA_DIR / "script_h_mismatched.json",
    "script_b1_first_call": DATA_DIR / "script_b1_first_call.json",
    "script_b2_returning": DATA_DIR / "script_b2_returning.json",
    "sim_muy04bh0_fee_complaint": DATA_DIR / "script_sim_muy04bh0_fee_complaint.json",
}

SCRIPTS_FOR_INVARIANT = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in REAL_SCRIPT_FILES.items()
}


@pytest.mark.parametrize(
    "script_name",
    [
        "canonical",
        "script_d_filler",
        "script_h_mismatched",
        "script_b1_first_call",
        "script_b2_returning",
        "sim_muy04bh0_fee_complaint",
    ],
)
@pytest.mark.parametrize("run_semantic", [False, True])
def test_gate_closed_no_commitment_close_or_confirm_protect_invariant(script_name, run_semantic):
    """Proves that across all real script JSON files (Canonical, D filler, H mismatched, B1 first call, B2 returning, sim_muy04bh0 fee complaint)
    in either mode (semantic False or True), no turn ever selects commitment_close, high/direct push,
    or confirm_and_protect while the meeting gate is closed without an existing confirmed conversion event (Point 4).
    """
    dialogue = SCRIPTS_FOR_INVARIANT[script_name]
    engine = ConversationReplayEngine()
    report = engine.replay_dialogue_turns(
        call_sid=f"invariant_check_{script_name}_sem_{run_semantic}",
        raw_turns=dialogue,
        save_report=False,
        run_semantic_analysis=run_semantic,
    )

    for step in report.timeline:
        tid = step.turn_id
        dec = step.strategic_decision
        gate = step.state_after.conversion_gate
        gate_open = gate.is_open if gate else False

        if not gate_open:
            assert dec.primary_action != StrategicAction.COMMITMENT_CLOSE, (
                f"[{script_name}] Turn {tid} (semantic={run_semantic}) violated invariant: Gate is closed, but action is COMMITMENT_CLOSE!"
            )
            assert str(dec.push_strength) not in ("direct_ask", "two_window_choice", "high"), (
                f"[{script_name}] Turn {tid} (semantic={run_semantic}) violated invariant: Gate is closed, but push is {dec.push_strength}!"
            )
            is_confirmed = bool(
                step.state_after.conversion_event
                and getattr(step.state_after.conversion_event, "status", None) in ("confirmed", ConversionEventStatus.CONFIRMED)
            )
            if not is_confirmed:
                assert "CONFIRM_AND_PROTECT_ACTIVE" not in dec.reason_codes, (
                    f"[{script_name}] Turn {tid} (semantic={run_semantic}) violated invariant: Gate is closed without confirmed event, but CONFIRM_AND_PROTECT is active!"
                )


# =============================================================================
# 10. Point 19: Robust Mismatched Input / Empty Utterance Handling (Script H)
# =============================================================================
def test_point19_mismatched_decision_and_empty_utterance_handling():
    """Point 19: Validates robust handling of empty utterance, whitespace, and mismatched turn sequence.
    Proves:
    1. Zero crashes or unhandled exceptions across empty/whitespace/mismatched turns.
    2. Conversion gate remains closed throughout.
    3. Action is never COMMITMENT_CLOSE.
    4. Push strength is never high, direct_ask, or two_window_choice.
    5. Zero fabricated commitment close or confirm_and_protect activations.
    """
    raw_turns = json.loads((DATA_DIR / "script_h_mismatched.json").read_text(encoding="utf-8"))
    engine = ConversationReplayEngine()
    report = engine.replay_dialogue_turns(
        call_sid="sim_test_point19_mismatched",
        raw_turns=raw_turns,
        save_report=False,
        run_semantic_analysis=True,
    )
    assert len(report.timeline) == 6
    for step in report.timeline:
        gate_open = step.state_after.conversion_gate.is_open if step.state_after.conversion_gate else False
        assert gate_open is False, f"Turn {step.turn_id}: Gate must remain closed on mismatched/empty turns"
        assert step.strategic_decision.primary_action != StrategicAction.COMMITMENT_CLOSE
        assert str(step.strategic_decision.push_strength) not in ("high", "direct_ask", "two_window_choice")
        assert "CONFIRM_AND_PROTECT_ACTIVE" not in step.strategic_decision.reason_codes


# =============================================================================
# 11. Turn 15 Downgrade Edge: Contact Preference at Confidence < 0.65 Stays ACKNOWLEDGE / protect / none
# =============================================================================
def test_turn15_contact_preference_low_confidence_no_downgrade():
    """Verifies that contact preference turns with posture 'protect' at confidence < 0.65
    (e.g. 0.60) stay ACKNOWLEDGE / protect / none and are never downgraded to CLARIFY.
    """
    from copilot.core_decision_manager import CoreDecisionManager
    from copilot.conversation_state_models import (
        ConversationStateSnapshot,
        ContactComplianceState,
        ContactPreference,
        ConversationStage,
    )

    snap = ConversationStateSnapshot(
        call_sid="test_contact_downgrade_protect",
        overall_confidence=0.60,
        conversation_stage=ConversationStage.VALUE_WALKTHROUGH,
    )
    snap.contact_compliance = ContactComplianceState(
        contact_preferences=[
            ContactPreference(
                channel="sms",
                allowed=True,
                cadence="reduced",
                prohibited_behavior="texting every day",
                source_turn_id=15,
                confidence=0.60,
            )
        ]
    )
    dm = CoreDecisionManager(call_sid="test_contact_downgrade_protect")
    res = dm.evaluate_state(
        snapshot=snap,
        turn_speaker="client",
        turn_text="Please don't start texting me every day before we meet, by the way.",
        turn_timestamp_ms=15000,
    )
    dec = res.decision
    assert dec.primary_action == StrategicAction.ACKNOWLEDGE
    assert dec.strategic_posture == "protect"
    assert str(dec.push_strength) == "none"
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" not in dec.reason_codes
    assert dec.primary_action != StrategicAction.CLARIFY


# =============================================================================
# 12. Dedicated Tests for WhatsApp, Other Channels, and normalize_time_to_24h
# =============================================================================
def test_whatsapp_and_channel_normalization():
    """Dedicated tests for WhatsApp, other channels, and normalize_time_to_24h."""
    from copilot.behavioral_semantic import normalize_time_to_24h

    # 1. normalize_time_to_24h test cases
    assert normalize_time_to_24h("noon") == "12:00"
    assert normalize_time_to_24h("midnight") == "00:00"
    assert normalize_time_to_24h("8am") == "08:00"
    assert normalize_time_to_24h("7pm") == "19:00"
    assert normalize_time_to_24h("6") == "18:00"  # bare 1..7 defaults to PM
    assert normalize_time_to_24h("9") == "09:00"  # bare 8..11 defaults to AM
    assert normalize_time_to_24h("early morning") == "early morning"

    # 2. WhatsApp extraction
    prefs_wa = extract_structured_contact_preferences("Send me a message on WhatsApp only")
    assert len(prefs_wa) == 1
    assert prefs_wa[0].channel == "whatsapp"
    assert prefs_wa[0].allowed is True

    # 3. is_time_permitted behavior on bare numbers vs text windows
    pref_call = extract_structured_contact_preferences("no calls after 6pm")[0]
    # Evaluated with normalized HH:MM
    ok_1400, _ = pref_call.is_time_permitted("14:00")
    assert ok_1400 is True
    violation_1900, reason = pref_call.is_time_permitted("19:00")
    assert violation_1900 is False
    assert "after 18:00" in reason

    # Text window or non-HH:MM returns clarification requirement
    invalid_win, reason_win = pref_call.is_time_permitted("early morning")
    assert invalid_win is False
    assert "requires clarification" in reason_win


# =============================================================================
# 13. Golden-File Regression Test for the 18-Row Baseline (Semantic Layer Off)
# =============================================================================
def test_canonical_18_row_baseline_regression():
    """Golden-file regression test for the 18-row baseline (Semantic Layer Off).
    Guarantees exact matching of all approved actions, postures, push strengths,
    and prompt statuses across all 18 turns.
    """
    raw_turns = json.loads((DATA_DIR / "script_canonical.json").read_text(encoding="utf-8"))
    engine = ConversationReplayEngine()
    report = engine.replay_dialogue_turns(
        call_sid="sim_test_canon_baseline_golden",
        raw_turns=raw_turns,
        save_report=False,
        run_semantic_analysis=False,
    )

    expected_rows = [
        (1, "salesperson", StrategicAction.QUESTION, "explore", "low", False, False),
        (2, "client", StrategicAction.QUESTION, "explore", "low", False, True),
        (3, "salesperson", StrategicAction.QUESTION, "explore", "low", True, False),
        (4, "client", StrategicAction.QUESTION, "explore", "low", False, True),
        (5, "client", StrategicAction.VALIDATE, "advance", "moderate", False, True),
        (6, "salesperson", StrategicAction.VALIDATE, "advance", "moderate", True, False),
        (7, "client", StrategicAction.QUESTION, "explore", "low", False, True),
        (8, "salesperson", StrategicAction.QUESTION, "explore", "low", False, False),
        (9, "client", StrategicAction.QUESTION, "explore", "low", False, True),
        (10, "client", StrategicAction.CLARIFY, "coordinate", "low", False, True),
        (11, "salesperson", StrategicAction.CLARIFY, "coordinate", "low", True, False),
        (12, "client", StrategicAction.MIRROR, "advance", "low", False, True),
        (13, "salesperson", StrategicAction.MIRROR, "advance", "low", True, False),
        (14, "client", StrategicAction.EDUCATE, "explore", "low", False, True),
        (15, "client", StrategicAction.ACKNOWLEDGE, "protect", "none", False, True),
        (16, "client", StrategicAction.QUESTION, "explore", "low", False, True),
        (17, "salesperson", StrategicAction.QUESTION, "explore", "low", True, False),
        (18, "client", StrategicAction.ACKNOWLEDGE, "protect", "none", False, True),
    ]

    assert len(report.timeline) == 18
    for step, expected in zip(report.timeline, expected_rows):
        tid, spk, exp_action, exp_posture, exp_push, exp_carried, exp_prompt = expected
        dec = step.strategic_decision
        assert step.turn_id == tid
        assert step.evidence_bundle.speaker_id == spk
        assert dec.primary_action == exp_action, f"Turn {tid} action mismatch: {dec.primary_action} != {exp_action}"
        assert dec.strategic_posture == exp_posture, f"Turn {tid} posture mismatch: {dec.strategic_posture} != {exp_posture}"
        assert str(dec.push_strength) == exp_push, f"Turn {tid} push mismatch: {dec.push_strength} != {exp_push}"
        assert bool(dec.carried_forward_from_turn_id) == exp_carried, f"Turn {tid} carried mismatch"
        assert dec.should_prompt == exp_prompt, f"Turn {tid} prompt mismatch: {dec.should_prompt} != {exp_prompt}"

