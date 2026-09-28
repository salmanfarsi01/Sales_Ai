import os
import json
import pytest

from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_state_contract import extract_behavioral_bundle, BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_intelligence_models import (
    StrategicDecision,
    StrategicAction,
)
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversionEventStatus,
    ConversionEventObject,
)


def _create_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    trust: float = 0.50,
    engagement: float = 0.50,
    agreement: float = 0.50,
    specificity: float = 0.50,
    future_lang: float = 0.0,
    boundary: float = 0.0,
    contact_preference: str = "none",
    call_sid: str = "CA_audit_extended",
    emotion_valence: float = 0.0,
    emotion_tension: float = 0.2,
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
# 1. Minimum Dimensional Coverage Rule (Client Feedback Item 2)
# -----------------------------------------------------------------------------
def test_readiness_partial_published_and_overall_suppressed_when_coverage_under_50_percent():
    """When only 1 dimension is measured (coverage < 0.50), headline readiness_score
    must be None (with insufficient_evidence=True) and exposed as readiness_partial
    with coverage metadata to prevent misleading downstream consumers.
    """
    manager = ConversationStateManager("CA_coverage_test")
    # Turn 1: Agent intro
    t1 = _create_bundle(1, "salesperson", "Hi, are you looking to sell your home?")
    manager.process_turn_bundle(t1)

    # Turn 2: Prospect asserts sole decision authority, but nothing else
    t2 = _create_bundle(2, "client", "I'm the one making this decision, no one else needs to sign off.")
    snap = manager.process_turn_bundle(t2)

    # Decision dimension is measured at 90.0 (weight 0.25 -> coverage 0.25 < 0.50)
    assert snap.readiness.decision_readiness == 90.0
    assert snap.readiness.coverage == 0.25
    assert snap.readiness.insufficient_evidence is True
    assert snap.readiness.readiness_score is None
    assert snap.readiness.readiness_partial == 90.0
    assert "low_coverage" in snap.readiness.active_blocker_caps


def test_readiness_overall_published_when_coverage_at_or_above_50_percent():
    """When at least 2 dimensions are measured (coverage >= 0.50), headline
    readiness_score is published.
    """
    manager = ConversationStateManager("CA_coverage_passing")
    t1 = _create_bundle(1, "salesperson", "Would Thursday work for a quick call?")
    manager.process_turn_bundle(t1)

    # Turn 2: Prospect asserts decision authority (decision=90.0) AND hesitation (emotional=59.0)
    t2 = _create_bundle(
        turn_id=2,
        speaker_id="client",
        text="I'm the decision maker, but honestly we're not sure if this is the right time.",
        emotion_valence=-0.3,
        emotion_tension=0.6,
        specificity=0.70,
    )
    snap = manager.process_turn_bundle(t2)

    # Two dimensions measured: decision (0.25) + emotional (0.25) = 0.50
    assert snap.readiness.coverage >= 0.50
    assert snap.readiness.insufficient_evidence is False
    assert snap.readiness.readiness_score is not None
    assert snap.readiness.readiness_partial is not None


# -----------------------------------------------------------------------------
# 2. Fabricated Event Elimination (Client Feedback Item 3)
# -----------------------------------------------------------------------------
def test_no_fabricated_placeholder_tentative_or_confirmed_events():
    """Confirms no placeholder events with 'Tentative Time Slot' or 'Confirmed Time Slot'
    can be created. Tentative and confirmed events strictly require real time parsed
    from prospect evidence or accepted proposal.
    """
    manager = ConversationStateManager("CA_placeholder_elimination")
    t1 = _create_bundle(1, "salesperson", "Let's meet to discuss the home.")
    manager.process_turn_bundle(t1)

    # Prospect expresses hedged statement without any concrete time or proposal
    t2 = _create_bundle(2, "client", "Maybe, we are not sure if this is the right time.", agreement=0.30)
    snap = manager.process_turn_bundle(t2)

    # Must NOT fabricate a tentative event with 'Tentative Time Slot'!
    ce = snap.conversion_event
    if ce is not None:
        assert ce.status != ConversionEventStatus.TENTATIVE or ce.start_at != "Tentative Time Slot"
        assert ce.start_at not in ("Tentative Time Slot", "Confirmed Time Slot")


# -----------------------------------------------------------------------------
# 3. Permission Regex Mandatory Object (Smaller Items)
# -----------------------------------------------------------------------------
def test_permission_regex_requires_object_and_rejects_third_parties():
    """Permission regex must require an explicit object ('me'/'us'/'to you') and reject
    'it's fine to call my lawyer' or 'happy to talk to your manager'.
    """
    manager = ConversationStateManager("CA_permission_regex_test")
    # Induce suspected boundary first with contact-specific friction
    t1 = _create_bundle(1, "client", "Why are you calling me? I get too many calls.")
    s1 = manager.process_turn_bundle(t1)
    assert s1.contact_compliance.boundary_suspected is True

    # Prospect says "happy to talk to your manager" -> must NOT clear suspected boundary!
    t2 = _create_bundle(2, "client", "I'd be happy to talk to your manager about this.")
    s2 = manager.process_turn_bundle(t2)
    assert s2.contact_compliance.boundary_suspected is True

    # Prospect explicitly says "you can call me tomorrow" -> clears suspected boundary!
    t3 = _create_bundle(3, "client", "Actually, you can call me tomorrow.")
    s3 = manager.process_turn_bundle(t3)
    assert s3.contact_compliance.boundary_suspected is False


# -----------------------------------------------------------------------------
# 4. Time-Bounded Holds vs Daily Time Preferences (Smaller Items)
# -----------------------------------------------------------------------------
def test_time_bounded_hold_sets_contact_not_before_and_bypasses_boundary_suspected():
    """'Don't call me again until Thursday' is a temporal hold:
    - Stores contact_not_before = 'Thursday'
    - Does NOT trigger boundary_suspected
    - Core Intelligence Engine acknowledges the hold with no_push
    """
    manager = ConversationStateManager("CA_hold_test")
    t1 = _create_bundle(1, "salesperson", "Just following up on your property.")
    manager.process_turn_bundle(t1)

    t2 = _create_bundle(2, "client", "Don't call me again until Thursday.")
    s2 = manager.process_turn_bundle(t2)

    # 1. Stored in contact_compliance as absolute date
    assert s2.contact_compliance.contact_not_before == "2026-10-01"
    assert s2.contact_compliance.contact_not_before_turn_id == 2
    # 2. Does NOT trigger boundary_suspected!
    assert s2.contact_compliance.boundary_suspected is False
    assert s2.contact_compliance.hard_boundary_active is False

    # 3. Core Engine acknowledges hold with protect_and_shorten (not hard exit)
    core = PitchProXCoreIntelligenceEngine()
    res = core.evaluate(s2, turn_speaker="client", turn_text=t2.utterance_text)
    assert res.decision.primary_action == StrategicAction.ACKNOWLEDGE
    assert res.decision.push_strength == "protect_and_shorten"
    assert "HOLD_RESPECTED" in res.decision.reason_codes
    assert "prohibited_contact_before_2026-10-01" in res.decision.do_not_do


def test_daily_time_preference_is_not_a_hold_or_boundary():
    """'Don't call after 4pm' is a daily time preference:
    - Extracted as a structured ContactPreference (specific_times, prohibited_behavior='calling after 4pm')
    - Does NOT set contact_not_before
    - Does NOT trigger boundary_suspected or hard boundary
    """
    manager = ConversationStateManager("CA_daily_pref_test")
    t1 = _create_bundle(1, "client", "Don't call after 4pm, I work late.")
    s1 = manager.process_turn_bundle(t1)

    assert s1.contact_compliance.contact_not_before is None
    assert s1.contact_compliance.boundary_suspected is False
    assert s1.contact_compliance.hard_boundary_active is False

    # Stored in contact_preferences
    prefs = s1.contact_compliance.contact_preferences
    assert len(prefs) >= 1
    assert any(p.cadence == "specific_times" and "4pm" in (p.prohibited_behavior or "") for p in prefs)


# -----------------------------------------------------------------------------
# 5. Dual-Mode Invariant Validation (Strict Raise vs Coerce)
# -----------------------------------------------------------------------------
def test_strict_invariant_raise_mode_under_env():
    """When STRICT_INVARIANT_RAISE=1, any illegal gate/action/push combination raises ValueError."""
    os.environ["STRICT_INVARIANT_RAISE"] = "1"
    with pytest.raises(ValueError, match="Invariant violation"):
        StrategicDecision(
            call_id="CA_inv_test",
            source_state_version=1,
            strategic_objective="Test close",
            primary_action=StrategicAction.COMMITMENT_CLOSE,  # Cannot close with gate closed
            push_strength="direct_ask",
            meeting_gate_open=False,
        )

    # When STRICT_INVARIANT_RAISE=0, safe coercion takes place
    os.environ["STRICT_INVARIANT_RAISE"] = "0"
    dec = StrategicDecision(
        call_id="CA_inv_coerce",
        source_state_version=1,
        strategic_objective="Test close",
        primary_action=StrategicAction.COMMITMENT_CLOSE,
        push_strength="direct_ask",
        meeting_gate_open=False,
    )
    assert dec.push_strength == "resolve_then_ask"
    assert dec.primary_action == StrategicAction.QUESTION
    # Restore strict raise for test suite
    os.environ["STRICT_INVARIANT_RAISE"] = "1"


# -----------------------------------------------------------------------------
# 6. Turn 04 Neutral Defaults Evaluate as Unknown (Client Feedback Item 1)
# -----------------------------------------------------------------------------
def test_turn_04_neutral_defaults_evaluate_as_unknown():
    """Turn 4 in the 18-turn benchmark replay:
    - Neutral trust (0.50), engagement (0.50), value (50.0), no objections raised
    - Must evaluate as 'unknown' (not 'met' on neutral defaults)
    - Gate remains firmly CLOSED
    """
    with open("reports/synthetic/conversation_state_sim_mucj0p5s.json", encoding="utf-8") as f:
        data = json.load(f)

    mgr = ConversationStateManager("CA_t04_neutral_audit")
    for t in data["timeline"][:4]:
        eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
        snap = mgr.process_turn_bundle(eb)

    gate = snap.conversion_gate
    assert gate.is_open is False
    assert gate.status == "closed"

    cond_map = {c.condition_name: c for c in gate.conditions}
    assert cond_map["trust_not_collapsing"].status == "unknown"
    assert cond_map["engagement_on_topic"].status == "unknown"
    assert cond_map["clear_value_reason"].status == "unknown"
    assert cond_map["objections_resolved_or_partial"].status == "unknown"
    assert cond_map["plausible_logistics"].status == "unknown"

    # Only verified conditions pass
    assert cond_map["decision_maker_aligned"].status == "met"
    assert cond_map["decision_maker_aligned"].evidence_turn_ids == [2]
    assert cond_map["no_active_boundary"].status == "met"


def test_turn_09_gate_hardened_against_thin_evidence():
    """Turn 9 in the 18-turn benchmark replay:
    - Prospect hedges: 'Maybe next week could work, let me think about it.'
    - clear_value_reason is unknown because no value/problem goal has been articulated.
    - Gate remains CLOSED (is_open=False), preventing premature direct_ask.
    - Engagement evidence excludes rep line (turn 8).
    - plausible_logistics notes hedged/conditional feasibility.
    """
    with open("reports/synthetic/conversation_state_sim_mucj0p5s.json", encoding="utf-8") as f:
        data = json.load(f)

    mgr = ConversationStateManager("CA_t09_hardened_audit")
    for t in data["timeline"][:9]:
        eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
        snap = mgr.process_turn_bundle(eb)

    gate = snap.conversion_gate
    assert gate.is_open is False
    assert gate.status == "closed"
    assert snap.push_strength.state == "resolve_then_ask"

    cond_map = {c.condition_name: c for c in gate.conditions}
    assert cond_map["clear_value_reason"].status == "unknown"
    assert cond_map["plausible_logistics"].status == "met"
    assert "hedge phrase" in cond_map["plausible_logistics"].reason

    # Evidence turns must NEVER contain turn 8 (rep line)
    for c in gate.conditions:
        assert 8 not in c.evidence_turn_ids


# -----------------------------------------------------------------------------
# 7. Production Invariant Coercion Verification
# -----------------------------------------------------------------------------
def test_production_coercion_mode_when_strict_raise_disabled():
    """Explicitly verify that when STRICT_INVARIANT_RAISE=0 (production mode),
    illegal gate/action/push combinations are safely coerced without throwing.
    """
    os.environ["STRICT_INVARIANT_RAISE"] = "0"
    dec = StrategicDecision(
        call_id="CA_prod_coerce_test",
        source_state_version=1,
        strategic_objective="Test close in prod",
        primary_action=StrategicAction.COMMITMENT_CLOSE,
        push_strength="direct_ask",
        meeting_gate_open=False,
    )
    assert dec.push_strength == "resolve_then_ask"
    assert dec.primary_action == StrategicAction.QUESTION
    assert "INVARIANT_VIOLATION_COERCED" in dec.reason_codes
    # Restore strict raise for test suite
    os.environ["STRICT_INVARIANT_RAISE"] = "1"


# -----------------------------------------------------------------------------
# 8. Golden Snapshot Integrity
# -----------------------------------------------------------------------------
def test_golden_snapshot_integrity():
    """Validates the committed full-field golden snapshot for sim_mucj0p5s."""
    path = "reports/synthetic/conversation_state_sim_mucj0p5s_golden.json"
    assert os.path.exists(path), "Golden snapshot file must exist"
    with open(path, encoding="utf-8") as f:
        golden = json.load(f)

    assert golden["call_sid"] == "sim_mucj0p5s"
    assert len(golden["timeline"]) == 18

    # Verify T04 is closed with unknown conditions
    t4 = golden["timeline"][3]
    t4_gate = t4["state_snapshot"]["conversion_gate"]
    assert t4_gate["is_open"] is False

    # Verify T09 gate is hardened: closed because value justification was unarticulated
    t9 = golden["timeline"][8]
    t9_gate = t9["state_snapshot"]["conversion_gate"]
    assert t9_gate["is_open"] is False
    assert any(c["condition_name"] == "clear_value_reason" and c["status"] == "unknown" for c in t9_gate["conditions"])
    # Verify all evidence turns are strictly prospect turns (turn 8 excluded)
    for c in t9_gate["conditions"]:
        assert 8 not in c["evidence_turn_ids"]

    # Verify T18 closes cleanly with confirmed appointment
    t18 = golden["timeline"][17]
    assert t18["state_snapshot"]["conversion_event"]["status"] == "confirmed"
    assert t18["state_snapshot"]["conversion_event"]["start_at"] == "Thursday At 3"
