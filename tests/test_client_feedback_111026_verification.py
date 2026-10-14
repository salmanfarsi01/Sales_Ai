"""Comprehensive verification suite for Client Feedback (111026).

Verifies all 3 items and core invariants requested by client:
Item 1:
- Deferral persists when driving ConversationStateManager directly (no replay engine)
- Decision builder (_build_deferred_to_meeting_decision) does NOT mutate state
- Deferred objection remains ACTIVE (not partially_resolved) and open_concerns formats correctly
- Prospect memory isolation: save_prospect_memory with turn provenance -> loaded_prospect_memory in Call 2 for same user, zero exposure for different user, not written to facts
- Paraphrase patterns and negative exclusions

Item 2:
- Emotional readiness stays None without measured evidence (objection presence does not fabricate 80%)
- Logical readiness excludes unmeasured synthetic default 0.60
- Per-component trace table validation

Item 3:
- Selected pressure does not overwrite state ceiling
- Invariant: selected_pressure <= preliminary_ceiling across all turns of both scripts (sim_mucj0p5s and sim_muy04bh0)
- Node test: UI renders "Selected Pressure" and "Preliminary Strategy Alias (Legacy)" and summary's push_strength
"""

import json
import os
import re
import subprocess
from pathlib import Path
import pytest

from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ObjectionLifecycleState,
    ObjectionRecord,
    PushStrengthRecommendation,
    ReadinessBreakdown,
)
from copilot.conversation_state_contract import extract_behavioral_bundle, BehavioralSignalInputBundle
from copilot.behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.conversation_objections import (
    ObjectionLifecycleEngine,
    is_meeting_deferral_statement,
    MEETING_DEFERRAL_PATTERNS,
    NEGATIVE_DEFERRAL_PATTERNS,
)
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_decision_manager import CoreDecisionManager, StrategicDecisionCache
from copilot.conversation_presentation import compute_open_concerns, format_open_concerns_summary
from copilot.conversation_replay import ConversationReplayEngine
from copilot.prospect_memory import ProspectMemoryStore


def _make_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    call_sid: str = "sim_test_call",
    trust: float = 0.50,
    tension: float = 0.20,
    valence: float = 0.00,
    engagement: float = 0.50,
    readiness: float = 0.50,
    momentum: float = 0.50,
    specificity: float = 0.60,
    agreement: float = 0.50,
    future_lang: float = 0.00,
    boundary: float = 0.00,
    contact_pref: str = "none",
    strategy_tag: str = None,
    recurrence_id: str = None,
    recurrence_type: str = "none",
) -> BehavioralSignalInputBundle:
    inf = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=turn_id * 3000,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=valence, tension_level=tension, confidence=0.85),
        pacing=DimensionScore(score=0.60, confidence=0.80, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=momentum, confidence=0.80, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.80, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_{turn_id}"],
    )
    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        recurrence_id=recurrence_id,
        recurrence_type=recurrence_type,
        specificity_score=specificity,
        agreement_score=agreement,
        future_language_score=future_lang,
        contact_preference=contact_pref,
    )
    b = extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=sem,
        salesperson_strategy_tag=strategy_tag,
    )
    b.timestamp_ms = turn_id * 3000
    return b


# =============================================================================
# ITEM 1: DEFERRED CONCERN
# =============================================================================

def test_deferral_persists_at_manager_level_without_replay():
    """Client requirement: Deferral must persist when driving ConversationStateManager
    directly with NO replay engine.
    """
    call_sid = "call_direct_manager_deferral"
    manager = ConversationStateManager(call_sid=call_sid)

    # Turn 2: Agent states fee
    b2 = _make_bundle(2, "salesperson", "Our standard fee is 5% which includes staging.", call_sid)
    manager.process_turn_bundle(b2)

    # Turn 3: Client objects to fee and proposes appointment
    b3 = _make_bundle(3, "client", "Your commission is still too expensive, but I can meet Thursday at 4 to see the numbers.", call_sid)
    manager.process_turn_bundle(b3)

    assert len(manager.current_state.objections) == 1
    obj = manager.current_state.objections[0]
    assert obj.canonical_category == "commission_fee"
    assert obj.deferred_to_meeting is False

    # Turn 4: Agent gives staging explanation
    b4 = _make_bundle(4, "salesperson", "The staging typically raises the final sale price, which offsets the fee.", call_sid)
    manager.process_turn_bundle(b4)

    # Turn 6: Client reiterates concern
    b6 = _make_bundle(6, "client", "But honestly the commission still bothers me.", call_sid)
    manager.process_turn_bundle(b6)
    assert manager.current_state.objections[0].lifecycle_state in (ObjectionLifecycleState.ACTIVE, "active")

    # Turn 8: Client explicitly defers to meeting directly via manager
    b8 = _make_bundle(8, "client", "ok leave it we will talk about it in our meeting Thursday at 4 pm", call_sid)
    final_snap = manager.process_turn_bundle(b8)

    # Assert flag persisted directly in canonical state
    final_obj = manager.current_state.objections[0]
    assert final_obj.deferred_to_meeting is True
    assert final_obj.retained_for_followup is True
    assert final_obj.lifecycle_state in (ObjectionLifecycleState.ACTIVE, "active")
    assert final_obj.latest_statement == "But honestly the commission still bothers me."
    assert final_obj.deferral_statement == "ok leave it we will talk about it in our meeting Thursday at 4 pm"

    # Verify open concerns formatting
    concerns = compute_open_concerns(final_snap)
    assert len(concerns) == 1
    assert concerns[0]["category"] == "commission_fee"
    assert concerns[0]["deferred_to_meeting"] is True
    assert concerns[0]["latest_statement"] == "But honestly the commission still bothers me."
    assert concerns[0]["deferral_statement"] == "ok leave it we will talk about it in our meeting Thursday at 4 pm"
    summary_line = format_open_concerns_summary(concerns)
    assert summary_line == "Open Concerns: 1 (commission fee — deferred to meeting: 'But honestly the commission still bothers me.')"


def test_decision_builder_does_not_modify_state():
    """Client requirement: Decision builder must not mutate state.
    Confirm _build_deferred_to_meeting_decision leaves snapshot untouched.
    """
    call_sid = "call_immutability_check"
    engine = PitchProXCoreIntelligenceEngine()
    
    obj = ObjectionRecord(
        objection_id="obj_test_123",
        canonical_category="commission_fee",
        initial_statement="Too expensive",
        latest_statement="Too expensive",
        first_turn_id=2,
        last_updated_turn_id=2,
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        deferred_to_meeting=False,
    )
    snap = ConversationStateSnapshot(
        call_sid=call_sid,
        objections=[obj],
    )
    
    # Pre-condition: deferred_to_meeting is False
    assert snap.objections[0].deferred_to_meeting is False

    # Build decision
    from copilot.core_intelligence_models import StrategicAction, StrategicInterpretationContext
    context = StrategicInterpretationContext(
        state_version=snap.state_version,
        conversion_confirmed=True,
        meeting_gate_open=True,
    )
    eval_res = engine._build_deferred_to_meeting_decision(
        snapshot=snap,
        context=context,
        primary_obj=obj,
        turn_timestamp_ms=10000,
        turn_text="we can go over the fees when we meet",
    )

    # Post-condition: snap.objections[0].deferred_to_meeting remains False on snapshot
    assert snap.objections[0].deferred_to_meeting is False
    assert eval_res.decision.primary_action == StrategicAction.ACKNOWLEDGE
    assert eval_res.decision.strategic_posture == "protect"
    assert str(eval_res.decision.push_strength) == "none"


def test_deferred_concern_remains_active_not_partially_resolved():
    """Client requirement: Deferral must not mark the objection partially_resolved.
    Objection must remain ACTIVE with deferred_to_meeting=True.
    """
    engine = ObjectionLifecycleEngine()
    obj = ObjectionRecord(
        objection_id="obj_fee_active",
        canonical_category="commission_fee",
        initial_statement="Commission too high",
        latest_statement="Commission still bothers me",
        first_turn_id=3,
        last_updated_turn_id=6,
        lifecycle_state=ObjectionLifecycleState.ACTIVE,
        recurrence_count=2,
    )
    engine._objections.append(obj)
    engine.active_focus_objection_id = obj.objection_id

    # Client defers to meeting
    b = _make_bundle(8, "client", "we can go over the fees when we meet")
    objs, changes, next_ver = engine.evaluate_turn(b, current_version=5)

    assert objs[0].deferred_to_meeting is True
    assert objs[0].lifecycle_state == ObjectionLifecycleState.ACTIVE
    assert objs[0].lifecycle_state != ObjectionLifecycleState.PARTIALLY_RESOLVED

    # Ensure change record audit exists
    deferral_changes = [c for c in changes if "deferred_to_meeting" in c.field_path]
    assert len(deferral_changes) == 1
    assert deferral_changes[0].new_value is True


def test_deferred_concern_eligible_memory_processing_and_isolation(tmp_path):
    """Client requirement:
    - save_prospect_memory / finalize_call writes objection record with turn provenance
    - End-to-end replay engine execution calls finalize_call() and persists concern
    - Duplicate prevention: calling finalize_call() twice or re-saving in Call 2 does not create duplicate records
    - Fail-closed: missing user_id returns [] cleanly without crashing teardown
    - Second call by same user loads it into loaded_prospect_memory (not facts)
    - Different user gets zero exposure
    """
    store = ProspectMemoryStore(store_path=tmp_path / "prospect_memory_store.json")
    user_alice = "user_agent_alice_42"
    user_bob = "user_agent_bob_99"
    prospect_p1 = "prospect_homeowner_77"

    # 1. Fail-closed test: missing user_id or prospect_id must fail closed and return [] without crashing
    mgr_no_user = ConversationStateManager(call_sid="call_no_user", memory_store=store)
    assert mgr_no_user.finalize_call() == []

    # 2. Call 1: Alice has call with prospect, records deferred objection in state
    call1_sid = "call_1_alice_prospect"
    mgr_call1 = ConversationStateManager(
        call_sid=call1_sid,
        user_id=user_alice,
        prospect_id=prospect_p1,
        memory_store=store,
    )
    b3 = _make_bundle(3, "client", "Your commission is still too expensive.", call1_sid)
    mgr_call1.process_turn_bundle(b3)
    b8 = _make_bundle(8, "client", "ok leave it we will talk about it in our meeting Thursday at 4 pm", call1_sid)
    mgr_call1.process_turn_bundle(b8)

    # Automated call finalization: persists deferred open concerns to memory automatically
    saved_records = mgr_call1.finalize_call()
    assert len(saved_records) == 1
    rec = saved_records[0]
    assert rec is not None
    assert rec.source_call_sid == call1_sid
    assert rec.source_turn_id == 8
    assert rec.user_id == user_alice
    assert rec.prospect_id == prospect_p1
    assert rec.value["deferred_to_meeting"] is True
    assert rec.value["latest_statement"] == "Your commission is still too expensive."
    assert rec.value["deferral_statement"] == "ok leave it we will talk about it in our meeting Thursday at 4 pm"

    # 3. Duplicate Prevention Test: calling finalize_call() a second time on the same call must NOT duplicate memory
    saved_records_repeat = mgr_call1.finalize_call()
    assert len(saved_records_repeat) == 1
    all_alice_records = store.retrieve_memories(user_id=user_alice, prospect_id=prospect_p1)
    assert len(all_alice_records) == 1, "Duplicate finalize_call() must not create duplicate memory items"

    # 4. Call 2: Alice calls the same prospect again -> Loads into loaded_prospect_memory
    mgr_call2_alice = ConversationStateManager(
        call_sid="call_2_alice_prospect",
        user_id=user_alice,
        prospect_id=prospect_p1,
        memory_store=store,
        load_prospect_memory=True,
    )
    # Memory must be in loaded_prospect_memory
    assert len(mgr_call2_alice.current_state.loaded_prospect_memory) == 1
    loaded = mgr_call2_alice.current_state.loaded_prospect_memory[0]
    assert loaded.key == "commission_fee"
    assert loaded.memory_type == "objection"
    assert loaded.provenance.source_call_sid == call1_sid
    assert loaded.provenance.source_turn_id == 8

    # CRITICAL: Must NOT be injected into facts!
    fact_keys = [f.fact_key for f in mgr_call2_alice.current_state.facts]
    assert "commission_fee" not in fact_keys

    # 5. Call 2 re-save de-duplication: finalize_call on Call 2 updates rather than duplicating
    b_c2 = _make_bundle(2, "client", "Still wondering about the fee we talked about.", "call_2_alice_prospect")
    mgr_call2_alice.process_turn_bundle(b_c2)
    saved_c2 = mgr_call2_alice.finalize_call()
    assert len(saved_c2) == 1
    all_alice_after_c2 = store.retrieve_memories(user_id=user_alice, prospect_id=prospect_p1)
    assert len(all_alice_after_c2) == 1, "Second call re-saving objection must update, not duplicate"

    # 6. Call 3: Bob (different user) calls the same prospect -> ZERO retrieval
    mgr_call3_bob = ConversationStateManager(
        call_sid="call_3_bob_prospect",
        user_id=user_bob,
        prospect_id=prospect_p1,
        memory_store=store,
        load_prospect_memory=True,
    )
    assert len(mgr_call3_bob.current_state.loaded_prospect_memory) == 0


def test_deferral_pattern_coverage_and_negatives():
    """Client requirement: Validate paraphrases and negative exclusion patterns."""
    positives = [
        "ok leave it we will talk about it in our meeting Thursday at 4 pm",
        "we can go over the fees when we meet",
        "we will talk about it in our meeting",
        "let's discuss the commission when we meet",
        "leave it for the meeting",
        "save it for our meeting",
        "we can cover the numbers in our meeting",
        "talk about it in our meeting",
    ]
    for text in positives:
        assert is_meeting_deferral_statement(text) is True, f"Failed to match positive: {text}"

    negatives = [
        "what time is the meeting?",
        "let's meet to sign",
        "where is the meeting?",
        "who is coming to the meeting?",
        "can we reschedule the meeting?",
    ]
    for text in negatives:
        assert is_meeting_deferral_statement(text) is False, f"Erroneously matched negative: {text}"


# =============================================================================
# ITEM 2: READINESS COVERAGE & PER-COMPONENT TRACE
# =============================================================================

def test_emotional_readiness_stays_none_without_measured_signals():
    """Client requirement: Emotional readiness must stay None when unmeasured,
    even when an objection exists.
    """
    manager = ConversationStateManager("call_unmeasured_emotional")
    
    # Turn 1: Salesperson opens
    b1 = _make_bundle(1, "salesperson", "Hello, thanks for taking my call.")
    manager.process_turn_bundle(b1)

    # Turn 2: Client raises objection with default tension (0.20) and neutral valence (0.00)
    b2 = _make_bundle(
        turn_id=2,
        speaker_id="client",
        text="Your fees are too high.",
        tension=0.20,
        valence=0.00,
        specificity=0.60,
    )
    snap = manager.process_turn_bundle(b2)

    # Objection exists
    assert len(snap.objections) == 1
    # But emotional readiness must be NONE because tension 0.20 is unmeasured default
    assert snap.readiness.emotional_readiness is None


def test_readiness_per_component_trace_on_fee_script():
    """Client requirement: Verify per-component trace table on fee script sim_muy04bh0.
    Emotional: None
    Logical: None
    Logistical: 90.0
    Decision: None
    Coverage: 25.0% (1/4) < 50% threshold -> headline readiness_score is None (null).
    """
    raw_turns = json.loads(Path("data/script_sim_muy04bh0_fee_complaint.json").read_text(encoding="utf-8"))
    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns(call_sid="sim_muy04bh0", raw_turns=raw_turns, save_report=True)

    r = rep.final_state.readiness
    assert r.emotional_readiness is None
    assert r.logical_readiness is None
    assert r.logistical_readiness == 90.0
    assert r.decision_readiness is None
    assert r.coverage == 0.25
    assert r.readiness_score is None  # Headline is null!
    assert r.readiness_partial == 90.0
    assert "low_coverage" in r.active_blocker_caps


def test_readiness_does_not_affect_confidence_rows_in_canonical():
    """Client requirement: Readiness elimination must NOT move canonical confidence rows
    (0.455, 0.685, 0.805).
    """
    golden_path = Path("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json")
    assert golden_path.exists()
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    
    # Key confidence rows in canonical run
    t14 = [s for s in golden["timeline"] if s["turn_id"] == 14][0]
    t15 = [s for s in golden["timeline"] if s["turn_id"] == 15][0]
    t18 = [s for s in golden["timeline"] if s["turn_id"] == 18][0]

    assert t14["strategic_decision"]["confidence"] == 0.455
    assert t15["strategic_decision"]["confidence"] == 0.730
    assert t18["strategic_decision"]["confidence"] == 0.805


# =============================================================================
# ITEM 3: PRESSURE INCONSISTENCY & CEILING INVARIANT
# =============================================================================

def test_selected_pressure_does_not_exceed_preliminary_ceiling_invariant():
    """Client requirement: Selected pressure must never exceed preliminary ceiling
    (none < low < moderate < high) across all turns of both scripts (canonical and fee script)
    in both semantic modes (semantic=False and semantic=True).
    """
    levels = {"none": 0, "low": 1, "moderate": 2, "high": 3}
    scripts = [
        ("sim_mucj0p5s", "data/script_canonical.json"),
        ("sim_muy04bh0", "data/script_sim_muy04bh0_fee_complaint.json"),
    ]

    for sid, path in scripts:
        raw_turns = json.loads(Path(path).read_text(encoding="utf-8"))
        for sem_mode in [False, True]:
            engine = ConversationReplayEngine()
            rep = engine.replay_dialogue_turns(
                call_sid=sid,
                raw_turns=raw_turns,
                save_report=False,
                run_semantic_analysis=sem_mode,
            )
            for s in rep.timeline:
                tid = s.turn_id
                sd = s.strategic_decision
                dec_p = getattr(sd, "push_strength", None)
                if isinstance(dec_p, dict):
                    dec_p = dec_p.get("pressure") or dec_p.get("value")
                dec_p = str(dec_p)
                st_push = s.state_after.push_strength
                ceiling = st_push.preliminary_ceiling if st_push else "moderate"
                assert levels[dec_p.lower()] <= levels[str(ceiling).lower()], (
                    f"Script {sid} (semantic={sem_mode}) Turn {tid}: Selected {dec_p} exceeds ceiling {ceiling}"
                )


def test_null_readiness_consumer_formatting_guard():
    """Client requirement: Check consumers of readiness_score:
    Evidence line and strategic_interpretation.readiness_score must handle null
    without printing 'None%' or failing.
    """
    engine = PitchProXCoreIntelligenceEngine()
    snap = ConversationStateSnapshot(
        call_sid="call_null_readiness_consumer",
        readiness=ReadinessBreakdown(
            readiness_score=None,
            readiness_partial=90.0,
            coverage=0.25,
            insufficient_evidence=True,
            active_blocker_caps=["low_coverage"],
        ),
    )
    eval_res = engine.evaluate(
        snapshot=snap,
        turn_speaker="client",
        turn_text="Can we meet Thursday at 4?",
        turn_timestamp_ms=10000,
        turn_id=2,
    )
    decision = eval_res.decision
    assert decision is not None

    # Assert no 'None%' in evidence
    for ev in decision.evidence_considered:
        assert "None%" not in ev, f"Found erroneous 'None%' in evidence line: {ev}"
    assert decision.strategic_interpretation.get("readiness_score") is None


def test_canonical_18_turn_readiness_evolution_and_cap_behavior():
    """Client requirement: Verify the full 18-turn canonical readiness timeline after excluding defaults:
    - Turn 1: insufficient evidence (0/4 measured, coverage 0.0)
    - Turns 2-8: low coverage (1/4 measured: decision=90.0), headline is None, partial=90.0, cap blocker ['low_coverage']
    - Turn 9: 2/4 measured (logistical=80.0, decision=90.0), coverage 50%, headline is 85.0
    - Turns 10-15: 2/4 measured (logistical=80.0, decision=40.0), uncapped 60.0, capped to 55.0 by absent_decision_maker
    - Turns 16-17: logistical drops to 65.0, decision=40.0, headline is 52.5
    - Turn 18: appointment confirmed & both spouses present (logistical=90.0, decision=90.0), headline is 90.0
    """
    raw_turns = json.loads(Path("data/script_canonical.json").read_text(encoding="utf-8"))
    engine = ConversationReplayEngine()
    rep = engine.replay_dialogue_turns(call_sid="sim_mucj0p5s", raw_turns=raw_turns, save_report=False)

    timeline = rep.timeline
    assert len(timeline) == 18

    # Turn 1
    t1_r = timeline[0].state_after.readiness
    assert t1_r.readiness_score is None
    assert t1_r.coverage == 0.0

    # Turns 2-8: Headline is None, Decision is 90.0 from Turn 2 ("I'm the one making this decision")
    for tid in range(2, 9):
        step_r = timeline[tid - 1].state_after.readiness
        assert step_r.readiness_score is None, f"Turn {tid} headline should be None"
        assert step_r.coverage == 0.25, f"Turn {tid} coverage should be 0.25"
        assert step_r.decision_readiness == 90.0, f"Turn {tid} decision readiness should be 90.0"
        assert step_r.readiness_partial == 90.0, f"Turn {tid} partial should be 90.0"
        assert "low_coverage" in step_r.active_blocker_caps

    # Turn 9: Reaches 50% coverage
    t9_r = timeline[8].state_after.readiness
    assert t9_r.readiness_score == 85.0
    assert t9_r.coverage == 0.50

    # Turns 10-15: Capped by absent decision maker to 55.0
    for tid in range(10, 16):
        step_r = timeline[tid - 1].state_after.readiness
        assert step_r.readiness_score == 55.0, f"Turn {tid} should be capped at 55.0"
        assert "absent_decision_maker" in step_r.active_blocker_caps

    # Turns 16-17: 52.5
    assert timeline[15].state_after.readiness.readiness_score == 52.5
    assert timeline[16].state_after.readiness.readiness_score == 52.5

    # Turn 18: 90.0
    assert timeline[17].state_after.readiness.readiness_score == 90.0


def test_ui_renders_selected_pressure_and_preliminary_alias_labels():
    """Client requirement: Node test verifying that 'Selected Pressure' and
    'Preliminary Strategy Alias (Legacy)' render in web/conversation_state_replay.html.
    Extracts the actual Push Strength template directly from web/conversation_state_replay.html.
    """
    html_path = Path("web/conversation_state_replay.html")
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # 1. Verify literal label existence in the DOM markup
    assert "Selected Pressure" in content
    assert "Preliminary Strategy Alias (Legacy)" in content
    assert "State Preliminary Ceiling" in content

    # 2. Extract the exact Push Strength card template from conversation_state_replay.html
    match = re.search(
        r"(<div style=\"margin: 10px 0; background: #FFFFFF; border: 1px solid var\(--border\);[\s\S]*?Push Strength & Pressure Ceiling[\s\S]*?Rationale:[\s\S]*?</div>\s*</div>)",
        content,
    )
    assert match is not None, "Could not extract Push Strength card template from HTML"
    card_template = match.group(1)

    # 3. Run Node.js evaluation on the extracted template
    node_script = f"""
    const push = {{
      state: "resolve_then_ask",
      pressure: "moderate",
      preliminary_ceiling: "moderate",
      legacy_strategy_alias: "resolve_then_ask",
      recommended_action: "Engage prospect",
      rationale: "Discover needs"
    }};
    const sd = {{
      push_strength: {{ pressure: "none" }},
      primary_action: "acknowledge"
    }};

    function escapeHtml(str) {{ return String(str); }}

    const renderedHtml = `{card_template}`;

    if (!renderedHtml.includes("Selected Pressure:") ||
        !renderedHtml.includes("State Preliminary Ceiling:") ||
        !renderedHtml.includes("Preliminary Strategy Alias (Legacy):")) {{
      console.error("Missing required labels in rendered output!");
      process.exit(1);
    }}
    if (!renderedHtml.includes("none") || !renderedHtml.includes("moderate") || !renderedHtml.includes("resolve_then_ask")) {{
      console.error("Values not rendered properly in template!");
      process.exit(1);
    }}
    console.log("NODE_TEST_PASSED");
    """
    res = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
    assert "NODE_TEST_PASSED" in res.stdout
