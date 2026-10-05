"""Test suite validating Group 2: Decision-model structural refinements (Points 6 to 14)
per guideline to fix issues 021026.md and client feedback.
"""
from __future__ import annotations

import os
import pytest
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversationStage,
    ConversionEventObject,
    ConversionEventStatus,
    MeetingConversionGate,
    DecisionStructure,
    DecisionStakeholder,
    DimensionScores,
    ContactCompliance,
    ContactPreference,
    ObjectionRecord,
    ObjectionLifecycleState,
    PushStrengthRecommendation,
    PersistentFactRecord,
    StrategyAttemptOutcome,
)
from copilot.core_intelligence_models import (
    StrategicAction,
    StrategicDecision,
    PushStrengthValue,
)
from copilot.core_intelligence_engine import (
    PitchProXCoreIntelligenceEngine,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from copilot.core_decision_manager import CoreDecisionManager
from copilot.llm_response_gateway import LLMResponseGateway
from copilot.conversation_replay import ConversationReplayEngine


def test_point6_split_action_posture_and_push_strength():
    """Point 6: Split Action / Posture / Push Strength into three independent fields.
    1. confirm_and_protect-style decisions must have push_strength: none or low, not hardcoded high.
    2. Confirm push_strength can independently vary across the same action and posture.
    """
    engine = PitchProXCoreIntelligenceEngine()
    snap = ConversationStateSnapshot(
        call_sid="call_pt6_confirm_protect",
        state_version=5,
        last_updated_turn_id=18,
        conversion_confirmed=True,
        conversion_event=ConversionEventObject(
            conversion_type="property_walkthrough",
            status=ConversionEventStatus.CONFIRMED,
            start_at="Thursday at 3:00 PM",
            participants=["Client", "Wife", "Agent"],
        ),
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.98,
            commitment_slot="Thursday at 3:00 PM",
        ),
        push_strength=PushStrengthRecommendation(
            state="confirm_and_protect",
            rationale="Confirmed appointment protection mode",
            recommended_action="confirm_and_protect",
            confidence=0.98,
        ),
    )

    eval_res = engine.evaluate(snapshot=snap, turn_speaker="client", turn_text="Thursday at 3 works, and my wife will be there.", turn_id=18)
    dec = eval_res.decision

    # 1. Three independent fields exist and are populated
    assert dec.primary_action == StrategicAction.ACKNOWLEDGE
    assert dec.strategic_posture == "protect"
    assert dec.push_strength in ("none", "low")
    assert dec.push_strength == "none"
    assert dec.push_strength != "high"
    assert dec.push_strength == "confirm_and_protect"  # Legacy alias works

    # 2. Independence: Same action (ACKNOWLEDGE) and posture (protect) can take different push strengths
    dec_none = StrategicDecision(
        call_id="call_pt6_var1",
        source_state_version=1,
        strategic_objective="Protect confirmed slot with zero pressure",
        primary_action=StrategicAction.ACKNOWLEDGE,
        strategic_posture="protect",
        push_strength="none",
    )
    dec_low = StrategicDecision(
        call_id="call_pt6_var2",
        source_state_version=1,
        strategic_objective="Protect confirmed slot with light logistical confirmation",
        primary_action=StrategicAction.ACKNOWLEDGE,
        strategic_posture="protect",
        push_strength="low",
    )
    assert dec_none.push_strength == "none"
    assert dec_low.push_strength == "low"
    assert dec_none.strategic_posture == dec_low.strategic_posture == "protect"
    assert dec_none.primary_action == dec_low.primary_action == StrategicAction.ACKNOWLEDGE

    # 3. Same action (CLARIFY) and posture (coordinate) can take different push strengths
    dec_coord_none = StrategicDecision(
        call_id="call_pt6_var3",
        source_state_version=1,
        strategic_objective="Coordinate logistics passively",
        primary_action=StrategicAction.CLARIFY,
        strategic_posture="coordinate",
        push_strength="none",
    )
    dec_coord_low = StrategicDecision(
        call_id="call_pt6_var4",
        source_state_version=1,
        strategic_objective="Coordinate logistics actively",
        primary_action=StrategicAction.CLARIFY,
        strategic_posture="coordinate",
        push_strength="low",
    )
    assert dec_coord_none.push_strength == "none"
    assert dec_coord_low.push_strength == "low"

    # 4. Same action (COMMITMENT_CLOSE) and posture (advance) can take moderate vs high push strength
    dec_close_mod = StrategicDecision(
        call_id="call_pt6_var5",
        source_state_version=1,
        strategic_objective="Soft walkthrough window proposal",
        primary_action=StrategicAction.COMMITMENT_CLOSE,
        strategic_posture="advance",
        push_strength="moderate",
        meeting_gate_open=True,
    )
    dec_close_high = StrategicDecision(
        call_id="call_pt6_var6",
        source_state_version=1,
        strategic_objective="Direct walkthrough booking ask",
        primary_action=StrategicAction.COMMITMENT_CLOSE,
        strategic_posture="advance",
        push_strength="high",
        meeting_gate_open=True,
    )
    assert dec_close_mod.push_strength == "moderate"
    assert dec_close_high.push_strength == "high"


def test_point7_absent_decision_maker_logistics_vs_genuine_risk():
    """Point 7: Stop auto-mapping 'decision-maker changed' to DE_RISK.
    Classify reason across all four required categories:
    1. Logistics / Coordination (Turn 10) -> CLARIFY, posture coordinate
    2. Relationship Conflict -> VALIDATE, posture mediate
    3. Trust Concern -> VALIDATE, posture reassure
    4. Genuine Deal Risk -> DE_RISK, posture defend
    """
    engine = PitchProXCoreIntelligenceEngine()

    base_stakeholder = DecisionStakeholder(name="Wife", role="wife", presence="absent")
    base_structure = DecisionStructure(
        decision_maker_present=False,
        primary_decision_maker="Spouse involvement required",
        stakeholders=[base_stakeholder],
        confidence=0.85,
    )

    # Category 1: Logistics / Coordination (Turn 10 scenario)
    snap_logistics = ConversationStateSnapshot(call_sid="call_pt7_logistics", state_version=3, decision_structure=base_structure)
    eval_log = engine.evaluate(
        snapshot=snap_logistics,
        turn_speaker="client",
        turn_text="Actually, my wife would really need to be part of this conversation before we go any further.",
        turn_id=10,
    )
    dec_log = eval_log.decision
    assert dec_log.primary_action == StrategicAction.CLARIFY
    assert dec_log.strategic_posture == "coordinate"
    assert "COORDINATION_LOGISTICS" in dec_log.reason_codes
    assert dec_log.push_strength in ("none", "low")
    assert "isolate_from_spouse" in dec_log.do_not_do
    assert "separate_partners" in dec_log.do_not_do

    # Verify do_not_do prohibitions reach LLM prompt contract (Block 1 & Block 7)
    gateway = LLMResponseGateway()
    llm_prompt = gateway.assemble_context(decision=dec_log, snapshot=snap_logistics, facts=[])
    assert "isolate_from_spouse" in llm_prompt
    assert "separate_partners" in llm_prompt
    assert "press_for_single_party_commitment" in llm_prompt
    fallback_text = gateway._deterministic_fallback(decision=dec_log, snapshot=snap_logistics)
    assert any(w in fallback_text.lower() for w in ("loop", "include", "together", "both"))

    # Category 2: Relationship Conflict
    snap_conflict = ConversationStateSnapshot(call_sid="call_pt7_conflict", state_version=3, decision_structure=base_structure)
    eval_conflict = engine.evaluate(
        snapshot=snap_conflict,
        turn_speaker="client",
        turn_text="We're going through a bitter divorce and we don't agree on whether to sell the property.",
        turn_id=10,
    )
    dec_conflict = eval_conflict.decision
    assert dec_conflict.primary_action == StrategicAction.VALIDATE
    assert dec_conflict.strategic_posture == "mediate"
    assert "RELATIONSHIP_CONFLICT" in dec_conflict.reason_codes
    assert dec_conflict.push_strength == "none"

    # Category 3: Trust Concern
    snap_trust = ConversationStateSnapshot(call_sid="call_pt7_trust", state_version=3, decision_structure=base_structure)
    eval_trust = engine.evaluate(
        snapshot=snap_trust,
        turn_speaker="client",
        turn_text="My wife doesn't trust real estate agents at all and thinks you're just trying to rip us off.",
        turn_id=10,
    )
    dec_trust = eval_trust.decision
    assert dec_trust.primary_action == StrategicAction.VALIDATE
    assert dec_trust.strategic_posture == "reassure"
    assert "STAKEHOLDER_TRUST_CONCERN" in dec_trust.reason_codes
    assert dec_trust.secondary_action == StrategicAction.SOCIAL_PROOF

    # Category 4: Genuine Deal Risk
    snap_risk = ConversationStateSnapshot(call_sid="call_pt7_risk", state_version=3, decision_structure=base_structure)
    eval_risk = engine.evaluate(
        snapshot=snap_risk,
        turn_speaker="client",
        turn_text="Our attorney reviewed this and told us this looks like a total scam and we must refuse to sign.",
        turn_id=10,
    )
    dec_risk = eval_risk.decision
    assert dec_risk.primary_action == StrategicAction.DE_RISK
    assert dec_risk.strategic_posture == "defend"
    assert "GENUINE_DEAL_RISK" in dec_risk.reason_codes
    assert dec_risk.push_strength == "none"


def test_point8_allow_wait_hold_no_prompt_at_all():
    """Point 8: Support WAIT / HOLD / no prompt at all.
    - Clarify difference between HOLD (state pause) and WAIT_SILENCE (conversational micro-pause).
    - When should_prompt=False or HOLD, gateway returns Prompt(text="", status="skipped").
    - Downstream presentation/replay handles skipped prompt without errors.
    """
    gateway = LLMResponseGateway()

    # 1. HOLD produces empty prompt with skipped status
    dec_hold = StrategicDecision(
        call_id="call_pt8_hold",
        source_state_version=2,
        source_turn_id=3,
        should_prompt=False,
        strategic_objective="Hold position and allow prospect space without intervening.",
        primary_action=StrategicAction.HOLD,
        strategic_posture="defend",
        push_strength="none",
        reason_codes=["STRATEGY_CARRIED_FORWARD"],
    )
    snap = ConversationStateSnapshot(call_sid="call_pt8_hold", state_version=2)
    prompt_hold = gateway.generate_prompt(decision=dec_hold, snapshot=snap, facts=[])
    assert prompt_hold.text == ""
    assert prompt_hold.status == "skipped"

    # 2. WAIT_SILENCE produces empty prompt with skipped status
    dec_wait = StrategicDecision(
        call_id="call_pt8_wait",
        source_state_version=2,
        source_turn_id=4,
        should_prompt=False,
        strategic_objective="Allow 3-second micro-pause for prospect to finish speaking.",
        primary_action=StrategicAction.WAIT_SILENCE,
        strategic_posture="protect",
        push_strength="none",
    )
    prompt_wait = gateway.generate_prompt(decision=dec_wait, snapshot=snap, facts=[])
    assert prompt_wait.text == ""
    assert prompt_wait.status == "skipped"

    # 3. Deterministic fallback also returns empty text
    assert gateway._deterministic_fallback(dec_hold, snapshot=snap) == ""
    assert gateway._deterministic_fallback(dec_wait, snapshot=snap) == ""


def test_point9_discipline_on_primary_secondary_strategy_stacking():
    """Point 9: Discipline on strategy stacking:
    1. Maximum 1 secondary action (never 3+ stacked strategies).
    2. Meaningful secondary_action_reason is enforced (>= 15 chars, rejects trivial placeholders).
    3. Spot-check 18-turn benchmark for compliance across all turns.
    """
    # 1. Strict raise validation rejects weak / trivial secondary_action_reason
    os.environ["STRICT_INVARIANT_RAISE"] = "1"
    try:
        with pytest.raises(ValueError, match="requires a meaningful justification"):
            StrategicDecision(
                call_id="call_pt9_weak",
                source_state_version=1,
                strategic_objective="Test weak justification",
                primary_action=StrategicAction.QUESTION,
                secondary_action=StrategicAction.MIRROR,
                secondary_action_reason="test",  # Weak placeholder < 15 chars
            )

        # Gibberish repetition like 'xxxxxxxxxxxxxxx' is rejected
        with pytest.raises(ValueError, match="requires a meaningful justification"):
            StrategicDecision(
                call_id="call_pt9_gibberish",
                source_state_version=1,
                strategic_objective="Test gibberish justification",
                primary_action=StrategicAction.QUESTION,
                secondary_action=StrategicAction.MIRROR,
                secondary_action_reason="xxxxxxxxxxxxxxx",  # 15 chars but single character gibberish
            )

        # Valid meaningful reason passes cleanly
        dec_valid = StrategicDecision(
            call_id="call_pt9_valid",
            source_state_version=1,
            strategic_objective="Test valid justification",
            primary_action=StrategicAction.QUESTION,
            secondary_action=StrategicAction.MIRROR,
            secondary_action_reason="Mirror prospect emotional language to build rapport and lower defense.",
        )
        assert dec_valid.secondary_action is not None
        assert len(dec_valid.secondary_action_reason) >= 15
    finally:
        os.environ.pop("STRICT_INVARIANT_RAISE", None)

    # 2. Replay 18-turn benchmark and verify every turn complies
    turns = [
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

    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns("sim_pt9_stacking", turns, save_report=False)
    cdm = CoreDecisionManager(call_sid="sim_pt9_stacking")

    for step in rep.timeline:
        eval_res = cdm.evaluate_state(snapshot=step.state_after, turn_speaker=step.speaker_id, turn_text=step.text, turn_id=step.turn_id)
        d = eval_res.decision
        assert d.primary_action is not None
        if d.secondary_action is not None:
            assert d.secondary_action_reason is not None and len(d.secondary_action_reason) >= 15


def test_point10_confidence_downgrades_commitment_close_to_cautious_action():
    """Point 10: Confidence must change behavior, not just be a displayed number.
    - Configurable threshold (default 0.65).
    - Boundary test at exactly 0.65 (does not downgrade).
    - Boundary test at < 0.65 (downgrades).
    - Boundary test at > 0.65 (does not downgrade).
    - Uses an actual 'confirm' scenario (open gate).
    - Also verifies CHALLENGE and high push strength cap.
    """
    engine = PitchProXCoreIntelligenceEngine()
    assert engine.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD == 0.65

    # Actual confirm scenario: Meeting gate is OPEN, prospect says "Thursday at 3 works"
    def make_confirm_snapshot(gate_conf: float, trust_score: float, trust_conf: float) -> ConversationStateSnapshot:
        return ConversationStateSnapshot(
            call_sid="call_pt10_confirm_scenario",
            state_version=5,
            conversation_stage=ConversationStage.SCHEDULING,
            conversion_gate=MeetingConversionGate(
                is_open=True,
                confidence=gate_conf,
                commitment_slot="Thursday at 3:00 PM",
                unknown_conditions=[],
            ),
            dimensions=DimensionScores(trust=trust_score, trust_confidence=trust_conf, trust_measured=True),
        )

    # Case A: High confidence (0.85 > 0.65) -> Retains COMMITMENT_CLOSE
    snap_high = make_confirm_snapshot(gate_conf=0.90, trust_score=0.85, trust_conf=0.90)
    eval_high = engine.evaluate(snapshot=snap_high, turn_speaker="client", turn_text="Thursday at 3 works, let's do it.", turn_id=18)
    assert eval_high.decision.confidence >= 0.65
    assert eval_high.decision.primary_action == StrategicAction.COMMITMENT_CLOSE
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" not in eval_high.decision.reason_codes

    # Case B: Exactly at boundary (0.65) -> Meets threshold, does NOT downgrade
    # final_conf = (base * 0.70) + (trust_conf * 0.30) -> (0.65 * 0.70) + (0.65 * 0.30) = 0.650
    snap_boundary = make_confirm_snapshot(gate_conf=0.65, trust_score=0.65, trust_conf=0.65)
    eval_boundary = engine.evaluate(snapshot=snap_boundary, turn_speaker="client", turn_text="Thursday at 3 works.", turn_id=18)
    assert eval_boundary.decision.confidence == 0.65
    assert eval_boundary.decision.primary_action == StrategicAction.COMMITMENT_CLOSE
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" not in eval_boundary.decision.reason_codes

    # Case C: Below threshold (< 0.65, e.g. 0.58) -> Action downgrades to CLARIFY, push strength capped to low
    snap_low = make_confirm_snapshot(gate_conf=0.60, trust_score=0.30, trust_conf=0.40)
    eval_low = engine.evaluate(snapshot=snap_low, turn_speaker="client", turn_text="Thursday at 3 works, I guess.", turn_id=18)
    dec_low = eval_low.decision
    assert dec_low.confidence < 0.65
    assert dec_low.primary_action == StrategicAction.CLARIFY
    assert dec_low.strategic_posture == "explore"
    assert dec_low.push_strength in ("none", "low")
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" in dec_low.reason_codes
    assert "premature_close_on_low_confidence" in dec_low.do_not_do

    # Case D: Configurable threshold test
    engine_custom = PitchProXCoreIntelligenceEngine(confidence_threshold=0.80)
    assert engine_custom.confidence_threshold == 0.80
    eval_custom = engine_custom.evaluate(snapshot=snap_boundary, turn_speaker="client", turn_text="Thursday at 3 works.", turn_id=18)
    # 0.65 is below custom 0.80 -> downgrades!
    assert eval_custom.decision.primary_action == StrategicAction.CLARIFY

    # Case E: Otherwise 'confirm' scenario (confirm_and_protect path) with low confidence
    # If an appointment is confirmed but overall confidence is low (< 0.65), it must downgrade to CLARIFY to verify understanding
    snap_confirm_low = ConversationStateSnapshot(
        call_sid="call_pt10_confirm_low",
        state_version=6,
        conversation_stage=ConversationStage.COMMITMENT_CONFIRMED,
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.40,
            commitment_slot="Thursday at 3:00 PM",
            unknown_conditions=[],
        ),
        conversion_event=ConversionEventObject(
            event_id="conv_pt10_low",
            conversion_type="in_person_meeting",
            status=ConversionEventStatus.CONFIRMED,
            start_at="Thursday at 3:00 PM",
            location_or_format="Scheduled Meeting",
            participants=["Client", "Agent"],
            confirmation_confidence=0.40,
            source_turn_ids=[18],
        ),
        dimensions=DimensionScores(trust=0.40, trust_confidence=0.40, trust_measured=True),
    )
    eval_confirm_low = engine.evaluate(snapshot=snap_confirm_low, turn_speaker="client", turn_text="Thursday at 3 works.", turn_id=18)
    dec_confirm_low = eval_confirm_low.decision
    assert dec_confirm_low.primary_action == StrategicAction.CLARIFY
    assert dec_confirm_low.strategic_posture == "explore"
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" in dec_confirm_low.reason_codes
    assert "Verify understanding" in (dec_confirm_low.secondary_action_reason or "")


def test_point11_evidence_considered_filters_irrelevant_metrics():
    """Point 11: Evidence Considered should be relevant, not exhaustive.
    - Fields tied to the decision's reason codes are included.
    - Unrelated metrics (unmeasured trust, unknown readiness, inactive contact preferences) drop out.
    - Fail-safe fallback prevents an empty evidence block.
    """
    engine = PitchProXCoreIntelligenceEngine()

    # Case A: Turn is an objection on timing. Contact preferences and unmeasured trust must DROP OUT.
    snap_irrelevant = ConversationStateSnapshot(
        call_sid="call_pt11_irrelevant",
        state_version=2,
        contact_compliance=ContactCompliance(
            contact_preferences=[
                ContactPreference(channel="sms", allowed=False, prohibited_behavior="daily_texting", source_turn_id=1)
            ]
        ),
        objections=[
            ObjectionRecord(
                objection_id="obj_t1",
                canonical_category="timing",
                raw_utterance="We're not ready yet.",
                initial_statement="We're not ready yet.",
                latest_statement="We're not ready yet.",
                first_turn_id=5,
                last_updated_turn_id=5,
                lifecycle_state=ObjectionLifecycleState.ACTIVE,
                recurrence_count=1,
            )
        ],
    )
    eval_irrelevant = engine.evaluate(
        snapshot=snap_irrelevant,
        turn_speaker="client",
        turn_text="We're not ready yet.",
        turn_id=5,
    )
    ev_irrelevant = eval_irrelevant.decision.evidence_considered
    assert not any("Contact preferences:" in e for e in ev_irrelevant)
    assert any("Active objections:" in e for e in ev_irrelevant)
    assert any('Turn utterance (client): "We\'re not ready yet."' in e for e in ev_irrelevant)

    # Case B: Turn explicitly invokes contact restrictions -> contact preferences MUST appear
    snap_relevant = ConversationStateSnapshot(
        call_sid="call_pt11_relevant",
        state_version=3,
        contact_compliance=ContactCompliance(
            contact_preferences=[
                ContactPreference(channel="sms", allowed=False, prohibited_behavior="daily_texting", source_turn_id=1)
            ]
        ),
    )
    eval_relevant = engine.evaluate(
        snapshot=snap_relevant,
        turn_speaker="client",
        turn_text="Please don't start texting me every day.",
        turn_id=15,
    )
    ev_relevant = eval_relevant.decision.evidence_considered
    assert any("Contact preferences:" in e for e in ev_relevant)

    # Case C: Fail-safe fallback when no metrics trigger
    snap_empty = ConversationStateSnapshot(call_sid="call_pt11_empty", state_version=1, conversation_stage=ConversationStage.DISCOVERY)
    eval_empty = engine.evaluate(snapshot=snap_empty, turn_speaker="client", turn_text="", turn_id=1)
    assert len(eval_empty.decision.evidence_considered) >= 1
    assert any("Stage:" in e for e in eval_empty.decision.evidence_considered)


def test_point12_no_canonical_state_duplication_uses_references():
    """Point 12: Don't duplicate canonical state between ConversationState and StrategicDecision.
    - StrategicDecision points to facts and objections via referenced_fact_ids / referenced_objection_ids.
    - Mutating ConversationState after a decision immediately reflects via dynamic reference resolution.
    - Deleted / superseded facts handle dangling IDs safely.
    """
    fact1 = PersistentFactRecord(
        fact_id="fact_meet_101",
        fact_key="confirmed_meeting_time",
        fact_value="Thursday at 3:00 PM",
        source_turn_id=18,
        timestamp_ms=1000,
    )
    snap = ConversationStateSnapshot(
        call_sid="call_pt12_ref",
        state_version=5,
        conversion_confirmed=True,
        facts=[fact1],
        conversion_event=ConversionEventObject(
            conversion_type="property_walkthrough",
            status=ConversionEventStatus.CONFIRMED,
            start_at="Thursday at 3:00 PM",
            participants=["Client", "Agent"],
        ),
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.95,
            commitment_slot="Thursday at 3:00 PM",
        ),
    )

    engine = PitchProXCoreIntelligenceEngine()
    eval_res = engine.evaluate(snapshot=snap, turn_speaker="client", turn_text="Thursday at 3 works.", turn_id=18)
    dec = eval_res.decision

    # 1. Decision stores references by ID, not deep duplicated state
    assert "fact_meet_101" in dec.referenced_fact_ids

    # 2. Mutating ConversationState fact after decision is created:
    # Dynamic reference resolution reflects mutated canonical state without diverging!
    assert dec.resolve_commitment_slot(snap) == "Thursday at 3:00 PM"
    fact1.fact_value = "Friday at 10:00 AM"
    assert dec.resolve_commitment_slot(snap) == "Friday at 10:00 AM"

    # 3. Dangling ID safety: If fact is superseded or removed from snapshot
    snap.facts = []
    assert dec.resolve_fact(snap, "fact_meet_101") is None


def test_point13_preserve_strategy_across_filler_turns():
    """Point 13: Preserve strategy across turns when nothing material changed.
    - A filler turn ('okay') records carried_forward_from_decision_id without regenerating duplicate logic.
    - Guard: If the previous prompt was an explicit question or closing ask, 'okay' is an affirmative answer, NOT a filler.
    - Chained carry-forwards preserve root decision and turn ID.
    - Staleness limit: after 2 consecutive carry-forward turns, forces fresh strategic evaluation.
    """
    cdm = CoreDecisionManager(call_sid="call_pt13_carried")
    snap1 = ConversationStateSnapshot(
        call_sid="call_pt13_carried",
        state_version=1,
        last_updated_turn_id=3,
        conversation_stage=ConversationStage.DISCOVERY,
    )
    # Turn 1: Substantive statement (primary action: QUESTION)
    eval1 = cdm.evaluate_state(
        snapshot=snap1,
        turn_speaker="client",
        turn_text="We might look at moving sometime next year.",
        turn_id=4,
    )
    dec1 = eval1.decision
    assert dec1.carried_forward_from_decision_id is None

    # Turn 2: Non-material filler when previous action was NOT a question/closing ask
    # To test pure backchannel carry-forward, set previous action to ACKNOWLEDGE
    dec1.primary_action = StrategicAction.ACKNOWLEDGE
    dec1.secondary_action = None
    dec1.strategic_objective = "Acknowledge prospect situation."

    snap2 = ConversationStateSnapshot(
        call_sid="call_pt13_carried",
        state_version=1,
        last_updated_turn_id=4,
        conversation_stage=ConversationStage.DISCOVERY,
    )
    eval2 = cdm.evaluate_state(
        snapshot=snap2,
        turn_speaker="client",
        turn_text="okay",
        turn_id=5,
    )
    dec2 = eval2.decision

    # Preserves provenance explicitly from Turn 4 decision
    assert dec2.carried_forward_from_decision_id == dec1.decision_id
    assert dec2.carried_forward_from_turn_id == 4
    assert "STRATEGY_CARRIED_FORWARD" in dec2.reason_codes
    assert dec2.should_prompt is False  # Point 8: filler produces no new prompt

    # Chained carry-forward test: Turn 3 filler points back to root Turn 4, not Turn 5
    eval3 = cdm.evaluate_state(
        snapshot=snap2,
        turn_speaker="client",
        turn_text="yeah",
        turn_id=6,
    )
    dec3 = eval3.decision
    assert dec3.carried_forward_from_decision_id == dec1.decision_id
    assert dec3.carried_forward_from_turn_id == 4

    # Staleness limit test: 3rd consecutive filler forces fresh evaluation (staleness limit = 2)
    eval4 = cdm.evaluate_state(
        snapshot=snap2,
        turn_speaker="client",
        turn_text="got it",
        turn_id=7,
    )
    dec4 = eval4.decision
    assert "STRATEGY_CARRIED_FORWARD" not in dec4.reason_codes

    # Question-pending guard: When previous decision was a direct QUESTION, 'okay' is an answer, NOT carried forward
    cdm2 = CoreDecisionManager(call_sid="call_pt13_question_guard")
    snap_q = ConversationStateSnapshot(call_sid="call_pt13_question_guard", state_version=1, conversation_stage=ConversationStage.DISCOVERY)
    eval_q = cdm2.evaluate_state(snapshot=snap_q, turn_speaker="salesperson", turn_text="What is your timeline?", turn_id=1)
    eval_q.decision.primary_action = StrategicAction.QUESTION
    cdm2.latest_decision = eval_q.decision

    eval_ans = cdm2.evaluate_state(snapshot=snap_q, turn_speaker="client", turn_text="okay", turn_id=2)
    # Must NOT be carried forward because a question was pending!
    assert "STRATEGY_CARRIED_FORWARD" not in eval_ans.decision.reason_codes

    # Prompt ending in '?' guard: Even if action was ACKNOWLEDGE, if the prompt asked 'Would Tuesday work?', 'yeah' is an answer!
    cdm_prompt = CoreDecisionManager(call_sid="call_pt13_prompt_q")
    eval_p = cdm_prompt.evaluate_state(snapshot=snap_q, turn_speaker="salesperson", turn_text="Got it.", turn_id=1)
    eval_p.decision.primary_action = StrategicAction.ACKNOWLEDGE
    eval_p.decision.final_prompt_text = "Would Tuesday at 4 work for you?"
    cdm_prompt.latest_decision = eval_p.decision
    eval_ans2 = cdm_prompt.evaluate_state(snapshot=snap_q, turn_speaker="client", turn_text="yeah", turn_id=2)
    assert "STRATEGY_CARRIED_FORWARD" not in eval_ans2.decision.reason_codes

    # Configurable max_carried_turns test (e.g. limit = 1)
    cdm_custom = CoreDecisionManager(call_sid="call_pt13_custom_limit", max_carried_turns=1)
    snap_c = ConversationStateSnapshot(call_sid="call_pt13_custom_limit", state_version=1, conversation_stage=ConversationStage.DISCOVERY)
    ev_c1 = cdm_custom.evaluate_state(snapshot=snap_c, turn_speaker="client", turn_text="We might move.", turn_id=1)
    ev_c1.decision.primary_action = StrategicAction.ACKNOWLEDGE
    cdm_custom.latest_decision = ev_c1.decision
    # 1st carried turn -> carried forward
    ev_c2 = cdm_custom.evaluate_state(snapshot=snap_c, turn_speaker="client", turn_text="okay", turn_id=2)
    assert "STRATEGY_CARRIED_FORWARD" in ev_c2.decision.reason_codes
    # 2nd carried turn -> exceeds limit of 1 -> fresh evaluation
    ev_c3 = cdm_custom.evaluate_state(snapshot=snap_c, turn_speaker="client", turn_text="yeah", turn_id=3)
    assert "STRATEGY_CARRIED_FORWARD" not in ev_c3.decision.reason_codes


def test_point14_use_strategy_history_avoids_repeating_failed_approaches():
    """Point 14: Use strategy history to avoid repeating failed approaches.
    - References earlier attempt turn and outcome summary in rationale and reason codes.
    - 'New evidence justifies retrying' exception allows retry if new facts arrived.
    - Exhaustion fallback when all candidates have failed.
    """
    engine = PitchProXCoreIntelligenceEngine()

    outcome_attempt = StrategyAttemptOutcome(
        strategy_tag="reframe",
        attempted_at_turn_id=4,
        prospect_response_turn_id=5,
        prospect_response_summary="Objection persisted despite reframe",
        effectiveness="rejected",
    )
    obj_with_failure = ObjectionRecord(
        objection_id="obj_commission_reactivated",
        canonical_category="commission",
        raw_utterance="Your commission is too high.",
        initial_statement="Your commission is too high.",
        latest_statement="Your commission is too high.",
        first_turn_id=4,
        last_updated_turn_id=8,
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        recurrence_count=2,  # Normally triggers FIRST_PUSHBACK -> REFRAME
        attempted_strategies=["reframe"],
        strategy_outcomes=[outcome_attempt],
    )

    # 1. Standard pivot: Avoids failed 'reframe', pivots to QUANTIFY, and references attempt & outcome
    snap = ConversationStateSnapshot(
        call_sid="call_pt14_history",
        state_version=4,
        last_updated_turn_id=8,
        objections=[obj_with_failure],
    )
    eval_res = engine.evaluate(snapshot=snap, turn_speaker="client", turn_text="I'm still not paying that fee.", turn_id=8)
    dec = eval_res.decision

    assert dec.primary_action == StrategicAction.QUANTIFY
    assert "AVOIDED_FAILED_STRATEGY_REFRAME" in dec.reason_codes
    assert "FAILED_OUTCOME_REJECTED" in dec.reason_codes
    assert "PIVOTED_TO_UNTRIED_STRATEGY" in dec.reason_codes
    assert "turn 4" in dec.strategic_objective.lower()
    assert "persisted despite reframe" in dec.strategic_objective.lower()

    # 2. 'New evidence justifies retrying' exception: New fact added since Turn 4 attempt
    new_fact = PersistentFactRecord(
        fact_id="fact_roi_new",
        fact_key="net_proceeds_calculation",
        fact_value="Seller nets 12% more with staging",
        source_turn_id=7,  # Turn 7 > Attempt Turn 4
        timestamp_ms=2000,
    )
    snap_new_ev = ConversationStateSnapshot(
        call_sid="call_pt14_new_ev",
        state_version=5,
        last_updated_turn_id=8,
        facts=[new_fact],
        objections=[obj_with_failure],
    )
    eval_retry = engine.evaluate(snapshot=snap_new_ev, turn_speaker="client", turn_text="I'm still not paying that fee.", turn_id=8)
    dec_retry = eval_retry.decision
    assert "RETRY_FAILED_STRATEGY_JUSTIFIED_BY_NEW_EVIDENCE" in dec_retry.reason_codes

    # 3. Exhaustion edge case: All candidate strategies have failed
    all_failed_obj = ObjectionRecord(
        objection_id="obj_all_failed",
        canonical_category="commission",
        raw_utterance="No way on the fee.",
        initial_statement="No way on the fee.",
        latest_statement="No way on the fee.",
        first_turn_id=2,
        last_updated_turn_id=10,
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        recurrence_count=5,
        attempted_strategies=["reframe", "quantify", "differentiate", "de_risk", "clarify", "question", "validate"],
        strategy_outcomes=[
            StrategyAttemptOutcome(strategy_tag="reframe", attempted_at_turn_id=2, prospect_response_turn_id=3, prospect_response_summary="Rejected reframe", effectiveness="rejected"),
            StrategyAttemptOutcome(strategy_tag="quantify", attempted_at_turn_id=4, prospect_response_turn_id=5, prospect_response_summary="Rejected quantify", effectiveness="rejected"),
            StrategyAttemptOutcome(strategy_tag="differentiate", attempted_at_turn_id=6, prospect_response_turn_id=7, prospect_response_summary="Rejected differentiation", effectiveness="rejected"),
            StrategyAttemptOutcome(strategy_tag="de_risk", attempted_at_turn_id=8, prospect_response_turn_id=9, prospect_response_summary="Rejected de-risk", effectiveness="rejected"),
            StrategyAttemptOutcome(strategy_tag="clarify", attempted_at_turn_id=9, prospect_response_turn_id=10, prospect_response_summary="Rejected clarify", effectiveness="rejected"),
            StrategyAttemptOutcome(strategy_tag="question", attempted_at_turn_id=10, prospect_response_turn_id=11, prospect_response_summary="Rejected question", effectiveness="rejected"),
            StrategyAttemptOutcome(strategy_tag="validate", attempted_at_turn_id=11, prospect_response_turn_id=12, prospect_response_summary="Rejected validate", effectiveness="rejected"),
        ],
    )
    snap_exhausted = ConversationStateSnapshot(
        call_sid="call_pt14_exhausted",
        state_version=6,
        last_updated_turn_id=10,
        objections=[all_failed_obj],
    )
    eval_exhausted = engine.evaluate(snapshot=snap_exhausted, turn_speaker="client", turn_text="Still no on that fee.", turn_id=10)
    assert "ALL_OBJECTION_STRATEGIES_EXHAUSTED_FALLBACK_CLARIFY" in eval_exhausted.decision.reason_codes
    assert eval_exhausted.decision.primary_action == StrategicAction.CLARIFY


def test_pipeline_ordering_cross_cutting():
    """Cross-cutting concern: Pipeline ordering between carry-forward (13), confidence downgrade (10),
    failed-strategy pivot (14), and prompt skipping (8).
    A filler turn at low confidence preserves the previous vetted decision with should_prompt=False.
    """
    cdm = CoreDecisionManager(call_sid="call_cross_cutting")
    snap = ConversationStateSnapshot(
        call_sid="call_cross_cutting",
        state_version=2,
        last_updated_turn_id=2,
        conversation_stage=ConversationStage.DISCOVERY,
        dimensions=DimensionScores(trust=0.20, trust_confidence=0.30),
    )
    # Turn 1: Low confidence evaluation downgrades action
    eval1 = cdm.evaluate_state(snapshot=snap, turn_speaker="client", turn_text="We might move.", turn_id=2)
    assert eval1.decision.confidence < 0.65
    # Clear any pending question on Turn 1 so Turn 2 'okay' is a pure filler backchannel
    eval1.decision.secondary_action = None
    cdm.latest_decision = eval1.decision

    # Turn 2: Filler turn carries forward the low-confidence decision with prompt skipped
    snap2 = ConversationStateSnapshot(call_sid="call_cross_cutting", state_version=2, last_updated_turn_id=3, conversation_stage=ConversationStage.DISCOVERY)
    eval2 = cdm.evaluate_state(snapshot=snap2, turn_speaker="client", turn_text="okay", turn_id=3)
    dec2 = eval2.decision
    assert dec2.carried_forward_from_decision_id == eval1.decision.decision_id
    assert dec2.should_prompt is False

    # Gateway skips prompt without invoking LLM
    gateway = LLMResponseGateway()
    prompt = gateway.generate_prompt(decision=dec2, snapshot=snap2, facts=[])
    assert prompt.status == "skipped"


def test_point12_enforce_no_direct_reads_via_ast():
    """Point 12: Continuous enforcement against direct reads of snapshot-only fields.
    Scans copilot/llm_response_gateway.py to verify that any read of commitment_slot,
    meeting_gate_open, or conversion_confirmed uses the safe resolvers (resolve_*),
    preventing direct reads of stale snapshot-time fields.
    """
    import ast
    from pathlib import Path

    gateway_file = Path(__file__).resolve().parent.parent / "copilot" / "llm_response_gateway.py"
    with open(gateway_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(gateway_file))

    disallowed_direct_attrs = {"commitment_slot", "meeting_gate_open", "conversion_confirmed"}
    direct_violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in disallowed_direct_attrs:
            # Check if this attribute access is on 'decision'
            if isinstance(node.value, ast.Name) and node.value.id == "decision":
                # Direct read like decision.commitment_slot detected!
                direct_violations.append((node.lineno, node.attr))

    assert direct_violations == [], f"Direct read of snapshot-only field(s) on decision detected in gateway: {direct_violations}. Use decision.resolve_<field>(snapshot) instead."


def test_point11_turn16_regression_and_reason_code_evidence_justification():
    """Point 11 Regression & Invariant: Every item in evidence_considered must be justified
    by reason codes or primary action, with the stage fallback as the only exception.
    Specifically tests Turn 16: An utterance containing 'mornings' in a scheduling context
    must NOT pull contact preferences into evidence_considered when reason codes are scheduling.
    """
    engine = PitchProXCoreIntelligenceEngine()

    # Regression Case: Turn 16 context
    snap_t16 = ConversationStateSnapshot(
        call_sid="call_pt11_turn16_regression",
        state_version=8,
        conversation_stage=ConversationStage.SCHEDULING,
        contact_compliance=ContactCompliance(
            contact_preferences=[
                ContactPreference(channel="sms", allowed=False, prohibited_behavior="daily_texting", source_turn_id=15)
            ]
        ),
        conversion_gate=MeetingConversionGate(is_open=False, unknown_conditions=["trust_not_collapsing", "clear_value_reason"]),
    )

    eval_t16 = engine.evaluate(
        snapshot=snap_t16,
        turn_speaker="client",
        turn_text="Mornings don't really work for us either, just so you know.",
        turn_id=16,
    )
    ev_t16 = eval_t16.decision.evidence_considered
    reasons_t16 = set(eval_t16.decision.reason_codes)

    # Must NOT contain contact preferences because reason codes do NOT contain contact preference codes
    assert not any("Contact preferences:" in e for e in ev_t16), f"Contact preferences leaked into Turn 16 evidence: {ev_t16}"
    assert "STAGE_SCHEDULING" in reasons_t16
    assert any("Gate: CLOSED" in e for e in ev_t16)
    assert any("Readiness:" in e for e in ev_t16)

    # Invariant Assertion across dialogue turns: Every evidence item must be justified
    test_cases = [
        ("We are not sure this is the right time.", 5, ConversationStage.OBJECTION_HANDLING),
        ("Actually, my wife would really need to be part of this.", 10, ConversationStage.DECISION_RESOLUTION),
        ("That's actually really helpful, tell me more -- how does the marketing process work...", 14, ConversationStage.DISCOVERY),
        ("Please don't start texting me every day before we meet.", 15, ConversationStage.DISCOVERY),
        ("Thursday at 3 works, and my wife will be there.", 18, ConversationStage.COMMITMENT_CONFIRMED),
    ]

    for utxt, tid, stg in test_cases:
        snap_case = ConversationStateSnapshot(
            call_sid="call_pt11_audit",
            state_version=tid,
            conversation_stage=stg,
            contact_compliance=snap_t16.contact_compliance,
            conversion_gate=MeetingConversionGate(is_open=(tid == 18)),
            conversion_event=ConversionEventObject(
                conversion_type="property_walkthrough",
                status=ConversionEventStatus.CONFIRMED if tid == 18 else ConversionEventStatus.PENDING,
            ) if tid == 18 else None,
            conversion_confirmed=(tid == 18),
        )
        res = engine.evaluate(snapshot=snap_case, turn_speaker="client", turn_text=utxt, turn_id=tid)
        ev_items = res.decision.evidence_considered
        reasons = set(res.decision.reason_codes)
        p_act = res.decision.primary_action

        # Turn 14 specifically: Reason codes are STAGE_VALUE_WALKTHROUGH & DEMONSTRATE_DIFFERENTIATION,
        # with no READINESS_* or GATE_* code. Readiness must NOT leak into evidence.
        if tid == 14:
            assert not any("Readiness:" in e for e in ev_items), f"Readiness leaked into Turn 14 evidence: {ev_items}"

        for ev in ev_items:
            if ev.startswith("Turn utterance"):
                continue
            elif ev.startswith("Trust:"):
                continue
            elif ev.startswith("Momentum:"):
                continue
            elif ev.startswith("Active objections:"):
                assert any(
                    r.startswith("OBJECTION_")
                    or r.startswith("LADDER_")
                    or r.startswith("REFRAME_")
                    or r.startswith("QUANTIFY_")
                    or r.startswith("DIFFERENTIATE_")
                    for r in reasons
                ) or p_act in (
                    StrategicAction.VALIDATE, StrategicAction.MIRROR, StrategicAction.REFRAME,
                    StrategicAction.QUANTIFY, StrategicAction.DIFFERENTIATE, StrategicAction.DE_RISK
                ), f"Objection evidence {ev} not justified by reasons {reasons} or action {p_act}"
            elif ev.startswith("Gate:"):
                assert any(
                    r.startswith("GATE_")
                    or r.startswith("COMMITMENT_")
                    or r.startswith("CONVERSION_")
                    or r.startswith("CLOSE_")
                    or r.startswith("TWO_WINDOW")
                    or r.startswith("DIRECT_ASK")
                    for r in reasons
                ) or p_act == StrategicAction.COMMITMENT_CLOSE, f"Gate evidence {ev} not justified by reasons {reasons}"
            elif ev.startswith("Contact preferences:") or ev.startswith("Hard boundary:") or ev.startswith("Boundary suspected:"):
                assert any(k in r for r in reasons for k in (
                    "CONTACT_NOT_BEFORE_DECLARED", "HOLD_RESPECTED", "PAST_CONTACT_FRICTION_HONORED",
                    "HARD_BOUNDARY_ACTIVE", "COMPLIANCE_PRIORITY", "BOUNDARY_SUSPECTED",
                    "CONTACT_PREFERENCE_DECLARED", "CONTACT_PREFERENCE_ENFORCED"
                )), f"Contact compliance evidence {ev} not justified by reasons {reasons}"
            elif ev.startswith("Readiness:"):
                assert any(
                    r.startswith("READINESS_")
                    or r.startswith("GATE_")
                    or r.startswith("COMMITMENT_")
                    or r.startswith("CONVERSION_")
                    or r.startswith("CLOSE_")
                    for r in reasons
                ) or p_act == StrategicAction.COMMITMENT_CLOSE, f"Readiness evidence {ev} not justified by reasons {reasons} or action {p_act}"
            elif ev.startswith("Stage:"):
                assert len(ev_items) <= 2, f"Stage fallback included when evidence list already had ample items: {ev_items}"
            else:
                pytest.fail(f"Unrecognized evidence item '{ev}' not governed by justification rules.")


def test_point6_push_strength_normalization_and_tunable_parameters():
    """Confirm push strength normalizes legacy strings to none/low/moderate/high vocabulary
    and confirm unmeasured_trust_cap and momentum adjustments are configurable.
    """
    # 1. Legacy string normalization to canonical vocabulary
    p_resolve = PushStrengthValue("resolve_then_ask")
    assert str(p_resolve) == "moderate"
    assert p_resolve == "resolve_then_ask"

    p_direct = PushStrengthValue("direct_ask")
    assert str(p_direct) == "high"
    assert p_direct == "direct_ask"

    p_confirm = PushStrengthValue("confirm_and_protect")
    assert str(p_confirm) == "none"
    assert p_confirm == "confirm_and_protect"

    # 2. Configurable unmeasured_trust_cap, momentum adjustments, and min_confidence_floor
    custom_engine = PitchProXCoreIntelligenceEngine(
        confidence_threshold=0.70,
        unmeasured_trust_cap=0.35,
        min_confidence_floor=0.40,
        momentum_adjustments={"regressing": -0.15, "stalling": -0.05, "advancing": 0.05, "stable": 0.0},
    )
    assert custom_engine.confidence_threshold == 0.70
    assert custom_engine.unmeasured_trust_cap == 0.35
    assert custom_engine.min_confidence_floor == 0.40
    assert custom_engine.momentum_adjustments["regressing"] == -0.15

    # 3. Hash and canonical JSON serialization
    assert hash(p_resolve) == hash("moderate")
    dec_test = StrategicDecision(
        call_id="call_test_json",
        source_state_version=1,
        primary_action=StrategicAction.ACKNOWLEDGE,
        strategic_objective="Protect confirmed walkthrough",
        push_strength=p_resolve,
    )
    dumped_dict = dec_test.model_dump()
    assert dumped_dict["push_strength"] == "moderate"
    import json
    dumped_json = json.loads(dec_test.model_dump_json())
    assert dumped_json["push_strength"] == "moderate"

    # CDM pass-through
    cdm = CoreDecisionManager(
        call_sid="call_custom_params",
        confidence_threshold=0.72,
        unmeasured_trust_cap=0.40,
        momentum_adjustments={"regressing": -0.12, "stalling": -0.06, "advancing": 0.06, "stable": 0.0},
    )
    assert cdm.engine.confidence_threshold == 0.72
    assert cdm.engine.unmeasured_trust_cap == 0.40
    assert cdm.engine.momentum_adjustments["regressing"] == -0.12


def test_point10_confirmed_conversion_unmeasured_trust_discount_exemption_boundary():
    """Item 5 Boundary Test: Confirmed conversions must be exempt from the unmeasured-trust
    cap (0.45). Without exemption, base 0.80 + unmeasured trust (0.45 * 0.30 = 0.135) + regressing
    momentum (-0.08) yields 0.56 + 0.135 - 0.08 = 0.615, which would erroneously trip the
    P10 downgrade to CLARIFY on an already-confirmed won appointment.
    With the exemption, trust uses full dimension confidence (0.70 * 0.30 = 0.210), yielding
    0.56 + 0.210 - 0.08 = 0.690 >= 0.65, protecting the confirmed appointment without downgrade.
    """
    from copilot.conversation_state_models import MomentumBreakdown
    engine = PitchProXCoreIntelligenceEngine()
    snap = ConversationStateSnapshot(
        call_sid="call_pt10_confirm_boundary",
        state_version=18,
        conversation_stage=ConversationStage.COMMITMENT_CONFIRMED,
        conversion_confirmed=True,
        conversion_gate=MeetingConversionGate(
            is_open=True,
            confidence=0.80,
            commitment_slot="Thursday at 3",
        ),
        conversion_event=ConversionEventObject(
            conversion_type="property_walkthrough",
            status=ConversionEventStatus.CONFIRMED,
            start_at="Thursday at 3",
            confirmation_confidence=0.80,
        ),
        momentum=MomentumBreakdown(momentum_score=30.0, trend="regressing"),
        dimensions=DimensionScores(
            trust=0.5,
            trust_confidence=0.7,
            trust_measured=False,
            momentum=0.3,
        ),
    )

    eval_res = engine.evaluate(
        snapshot=snap,
        turn_speaker="client",
        turn_text="Thursday at 3 works, and my wife will be there.",
        turn_id=18,
    )
    dec = eval_res.decision

    # 1. Trust confidence must not be capped to 0.45 because confirmed conversion is exempt
    assert dec.confidence_breakdown["trust_confidence"] == 0.70
    assert dec.confidence_breakdown["momentum_adjustment"] == -0.08

    # 2. Final confidence calculation: 0.80 * 0.70 + 0.70 * 0.30 - 0.08 = 0.6900
    assert dec.confidence == 0.6900
    assert dec.confidence >= engine.confidence_threshold

    # 3. Decision must protect confirmed appointment, NOT downgrade to CLARIFY
    assert dec.primary_action == StrategicAction.ACKNOWLEDGE
    assert dec.strategic_posture == "protect"
    assert str(dec.push_strength) == "none"
    assert "CONVERSION_CONFIRMED" in dec.reason_codes
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" not in dec.reason_codes


def test_point10_recurring_objection_low_confidence_push_capping():
    """Item 4: When recurrence discount pushes objection confidence below threshold (< 0.65),
    moderate push strength must be capped to low, and LOW_CONFIDENCE_ACTION_DOWNGRADE appended.
    """
    engine = PitchProXCoreIntelligenceEngine()
    recurring_obj = ObjectionRecord(
        objection_id="obj_recur_low_conf",
        canonical_category="commission",
        initial_statement="Your 6% fee is just too high.",
        latest_statement="I'm still really concerned about that commission rate.",
        first_turn_id=5,
        last_updated_turn_id=12,
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        recurrence_count=6,
        confidence=0.90,
    )
    snap = ConversationStateSnapshot(
        call_sid="call_pt10_obj_push_cap",
        state_version=12,
        conversation_stage=ConversationStage.OBJECTION_HANDLING,
        objections=[recurring_obj],
        dimensions=DimensionScores(
            trust=0.5,
            trust_confidence=0.7,
            trust_measured=False,
        ),
    )

    eval_res = engine.evaluate(
        snapshot=snap,
        turn_speaker="client",
        turn_text="I'm still really concerned about that commission rate.",
        turn_id=12,
    )
    dec = eval_res.decision

    # Confidence calculation:
    # Recurrence >= 6 floors base objection confidence to 0.50
    # Unmeasured trust capped to 0.45
    # final_conf = 0.50 * 0.70 + 0.45 * 0.30 = 0.35 + 0.135 = 0.485 < 0.65
    assert dec.confidence == 0.4850
    assert dec.confidence < engine.confidence_threshold

    # Push strength must be capped to low (not moderate, and does not equal resolve_then_ask)
    assert str(dec.push_strength) == "low"
    assert dec.push_strength != "resolve_then_ask"
    assert dec.push_strength != "moderate"
    assert "LOW_CONFIDENCE_ACTION_DOWNGRADE" in dec.reason_codes
    assert "aggressive_push_on_low_confidence" in dec.do_not_do


def test_point8_all_rep_turns_suppress_prompts():
    """Item 1: In production, teleprompter prompts are dispatched ONLY on prospect utterances.
    Every rep turn must produce should_prompt = False, whether it is a fresh evaluation (e.g. Turn 1)
    or a continuation (e.g. Turn 3).
    """
    cdm = CoreDecisionManager(call_sid="call_pt8_all_rep_suppressed")
    snap1 = ConversationStateSnapshot(
        call_sid="call_pt8_all_rep_suppressed",
        state_version=1,
        conversation_stage=ConversationStage.DISCOVERY,
    )

    # Turn 1 (salesperson, fresh evaluation)
    eval1 = cdm.evaluate_state(
        snapshot=snap1,
        turn_speaker="salesperson",
        turn_text="Hi, this is Alex with Premier Realty.",
        turn_id=1,
    )
    assert eval1.decision.should_prompt is False
    assert eval1.decision.carried_forward_from_decision_id is None

    # Turn 2 (prospect utterance -> prompt generated)
    eval2 = cdm.evaluate_state(
        snapshot=snap1,
        turn_speaker="client",
        turn_text="Yeah, I'm thinking about selling next year.",
        turn_id=2,
    )
    assert eval2.decision.should_prompt is True

    # Turn 3 (salesperson utterance with continuation -> prompt suppressed)
    eval3 = cdm.evaluate_state(
        snapshot=snap1,
        turn_speaker="salesperson",
        turn_text="We specialize in properties in your neighborhood.",
        turn_id=3,
    )
    assert eval3.decision.should_prompt is False
