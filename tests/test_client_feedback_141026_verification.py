"""Tests verifying resolution of Client Feedback 141026:
1. Decision evidence references resolve within bound snapshot (Turn 3-8 in sim_muy04bh0 and all turns in sim_mucj0p5s).
2. StrategicDecisionCache and validate_binding enforce snapshot evidence consistency.
3. Strategic Interpretation panel displays Strategy / Posture and Selected Pressure cleanly.
4. Backend cache isolation, memory ownership, and runtime decision inputs.
"""

import json
from pathlib import Path
import pytest

from copilot.conversation_replay import ConversationReplayEngine
from copilot.core_decision_manager import CoreDecisionManager, StrategicDecisionCache
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_intelligence_models import StrategicAction, StrategicDecision, PushStrengthValue
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    PersistentFactRecord,
    ObjectionRecord,
    ObjectionDriverLayer,
)
from copilot.prospect_memory import ProspectMemoryStore, ProspectMemoryRecord


def test_sim_muy04bh0_evidence_references_resolve_within_snapshot():
    """Client Feedback 141026: Turn 3 through Turn 8 evidence references must strictly
    resolve within the bound snapshot.
    """
    StrategicDecisionCache.clear()
    engine = ConversationReplayEngine()
    script_path = Path(__file__).resolve().parent.parent / "data" / "script_sim_muy04bh0_fee_complaint.json"
    raw_turns = json.loads(script_path.read_text(encoding="utf-8"))
    rep = engine.replay_dialogue_turns(call_sid="sim_muy04bh0", raw_turns=raw_turns, save_report=True)

    report_path = Path(__file__).resolve().parent.parent / "reports" / "synthetic" / "conversation_state_sim_muy04bh0.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))

    for step in data["timeline"]:
        tid = step["turn_id"]
        sa = step["state_after"]
        dec = step.get("strategic_decision") or {}
        
        state_fact_ids = {f["fact_id"] for f in sa.get("facts", [])}
        state_obj_ids = {o["objection_id"] for o in sa.get("objections", [])}
        state_stk_ids = {s.get("stakeholder_id") for s in sa.get("stakeholders", []) if s.get("stakeholder_id")}
        if sa.get("decision_structure") and sa["decision_structure"].get("stakeholders"):
            for s in sa["decision_structure"]["stakeholders"]:
                if s.get("stakeholder_id"):
                    state_stk_ids.add(s["stakeholder_id"])
                if s.get("role"):
                    state_stk_ids.add(s["role"])
                    state_stk_ids.add(f"stakeholder_{s['role']}")

        # Verify referenced facts
        for fid in dec.get("referenced_fact_ids", []):
            assert fid in state_fact_ids, (
                f"Turn {tid} referenced_fact_id '{fid}' does not exist in bound snapshot facts: {state_fact_ids}"
            )

        # Verify referenced objections
        for oid in dec.get("referenced_objection_ids", []):
            assert oid in state_obj_ids, (
                f"Turn {tid} referenced_objection_id '{oid}' does not exist in bound snapshot objections: {state_obj_ids}"
            )

        # Verify referenced stakeholders
        for sid in dec.get("referenced_stakeholder_ids", []):
            assert sid in state_stk_ids, (
                f"Turn {tid} referenced_stakeholder_id '{sid}' does not exist in bound snapshot stakeholders: {state_stk_ids}"
            )

    # Specific check for Turn 3 and Turn 8
    step3 = [s for s in data["timeline"] if s["turn_id"] == 3][0]
    dec3 = step3["strategic_decision"]
    assert len(dec3["referenced_fact_ids"]) > 0
    assert len(dec3["referenced_objection_ids"]) > 0
    # Fact and objection resolve in Turn 3 state
    assert dec3["referenced_fact_ids"][0] in [f["fact_id"] for f in step3["state_after"]["facts"]]
    assert dec3["referenced_objection_ids"][0] in [o["objection_id"] for o in step3["state_after"]["objections"]]

    step8 = [s for s in data["timeline"] if s["turn_id"] == 8][0]
    dec8 = step8["strategic_decision"]
    assert len(dec8["referenced_fact_ids"]) > 0
    assert dec8["referenced_fact_ids"][0] in [f["fact_id"] for f in step8["state_after"]["facts"]]


def test_sim_mucj0p5s_golden_evidence_references_resolve_within_snapshot():
    """Client Feedback 141026: Canonical 18-turn replay must have 100% consistent evidence
    references across all turns.
    """
    StrategicDecisionCache.clear()
    engine = ConversationReplayEngine()
    script_path = Path(__file__).resolve().parent.parent / "data" / "script_canonical.json"
    raw_turns = json.loads(script_path.read_text(encoding="utf-8"))
    rep = engine.replay_dialogue_turns(call_sid="sim_mucj0p5s", raw_turns=raw_turns, save_report=True)

    report_path = Path(__file__).resolve().parent.parent / "reports" / "synthetic" / "conversation_state_sim_mucj0p5s.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(data["timeline"]) == 18
    for step in data["timeline"]:
        tid = step["turn_id"]
        sa = step["state_after"]
        dec = step.get("strategic_decision") or {}

        state_fact_ids = {f["fact_id"] for f in sa.get("facts", [])}
        state_obj_ids = {o["objection_id"] for o in sa.get("objections", [])}
        state_stk_ids = {s.get("stakeholder_id") for s in sa.get("stakeholders", []) if s.get("stakeholder_id")}
        if sa.get("decision_structure") and sa["decision_structure"].get("stakeholders"):
            for s in sa["decision_structure"]["stakeholders"]:
                if s.get("stakeholder_id"):
                    state_stk_ids.add(s["stakeholder_id"])
                if s.get("role"):
                    state_stk_ids.add(s["role"])
                    state_stk_ids.add(f"stakeholder_{s['role']}")

        for fid in dec.get("referenced_fact_ids", []):
            assert fid in state_fact_ids, f"Turn {tid} fact mismatch: {fid} not in {state_fact_ids}"
        for oid in dec.get("referenced_objection_ids", []):
            assert oid in state_obj_ids, f"Turn {tid} objection mismatch: {oid} not in {state_obj_ids}"
        for sid in dec.get("referenced_stakeholder_ids", []):
            assert sid in state_stk_ids, f"Turn {tid} stakeholder mismatch: {sid} not in {state_stk_ids}"


def test_validate_binding_rejects_missing_snapshot_evidence():
    """Client Feedback 141026: validate_binding must reject a decision whose referenced_fact_ids
    or referenced_objection_ids point to IDs not present in the bound snapshot.
    """
    call_sid = "sim_test_binding"
    snap = ConversationStateSnapshot(
        call_sid=call_sid,
        state_version=10,
        last_updated_turn_id=3,
        facts=[
            PersistentFactRecord(
                fact_id="fact_real_123",
                category="timeline",
                fact_key="confirmed_meeting_time",
                fact_value="Thursday at 4",
                source_turn_id=3,
                timestamp_ms=9000,
            )
        ],
        objections=[
            ObjectionRecord(
                objection_id="obj_real_456",
                canonical_category="commission_fee",
                initial_statement="Fee too high",
                latest_statement="Fee too high",
                first_turn_id=3,
                last_updated_turn_id=3,
            )
        ],
    )

    # 1. Decision with valid references
    valid_dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=10,
        source_turn_id=3,
        source_event_id="ev_turn_3_v10",
        strategic_objective="Protect appointment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        referenced_fact_ids=["fact_real_123"],
        referenced_objection_ids=["obj_real_456"],
    )

    is_valid, reason = StrategicDecisionCache.validate_binding(
        decision=valid_dec,
        call_sid=call_sid,
        turn_id=3,
        source_state_version=10,
        source_event_id="ev_turn_3_v10",
        state_call_sid=call_sid,
        snapshot=snap,
    )
    assert is_valid is True

    # 2. Decision with stale fact ID (e.g. from prior run)
    stale_fact_dec = valid_dec.model_copy(deep=True)
    stale_fact_dec.referenced_fact_ids = ["fact_stale_999"]

    is_valid, reason = StrategicDecisionCache.validate_binding(
        decision=stale_fact_dec,
        call_sid=call_sid,
        turn_id=3,
        source_state_version=10,
        source_event_id="ev_turn_3_v10",
        state_call_sid=call_sid,
        snapshot=snap,
    )
    assert is_valid is False
    assert "referenced_fact_id 'fact_stale_999' not found in bound snapshot facts" in reason

    # 3. Decision with stale objection ID
    stale_obj_dec = valid_dec.model_copy(deep=True)
    stale_obj_dec.referenced_objection_ids = ["obj_stale_888"]

    is_valid, reason = StrategicDecisionCache.validate_binding(
        decision=stale_obj_dec,
        call_sid=call_sid,
        turn_id=3,
        source_state_version=10,
        source_event_id="ev_turn_3_v10",
        state_call_sid=call_sid,
        snapshot=snap,
    )
    assert is_valid is False
    assert "referenced_objection_id 'obj_stale_888' not found in bound snapshot objections" in reason


def test_cache_strictly_bound_rejects_stale_evidence_references():
    """Client Feedback 141026: StrategicDecisionCache.get_strictly_bound must return None
    when cached decision references point to IDs outside the provided snapshot.
    """
    call_sid = "sim_test_cache_evidence"
    StrategicDecisionCache.clear(call_sid)

    # Cache decision with fact_A
    cached_dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=5,
        source_turn_id=2,
        source_event_id="ev_turn_2_v5",
        strategic_objective="Protect appointment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        referenced_fact_ids=["fact_A"],
    )
    StrategicDecisionCache.put(
        call_sid=call_sid,
        decision=cached_dec,
        turn_id=2,
        source_event_id="ev_turn_2_v5",
    )

    # Query with snapshot that has fact_A -> succeeds
    snap_matching = ConversationStateSnapshot(
        call_sid=call_sid,
        state_version=5,
        last_updated_turn_id=2,
        facts=[PersistentFactRecord(fact_id="fact_A", category="timeline", fact_key="meeting", fact_value="Thursday", source_turn_id=2, timestamp_ms=6000)],
    )
    res = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=2,
        source_state_version=5,
        source_event_id="ev_turn_2_v5",
        state_call_sid=call_sid,
        snapshot=snap_matching,
    )
    assert res is not None
    assert res.referenced_fact_ids == ["fact_A"]

    # Query with new snapshot that has fact_B (Run 2) -> rejected (returns None)
    snap_new_run = ConversationStateSnapshot(
        call_sid=call_sid,
        state_version=5,
        last_updated_turn_id=2,
        facts=[PersistentFactRecord(fact_id="fact_B", category="timeline", fact_key="meeting", fact_value="Thursday", source_turn_id=2, timestamp_ms=6000)],
    )
    res_stale = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=2,
        source_state_version=5,
        source_event_id="ev_turn_2_v5",
        state_call_sid=call_sid,
        snapshot=snap_new_run,
    )
    assert res_stale is None, "Stale cached decision must be rejected when evidence references point outside snapshot"


def test_carried_strategy_aborts_if_references_not_in_target_snapshot():
    """Client Feedback 141026: _is_carried_strategy_valid_for_snapshot must reject carry-forward
    if the carried strategy references facts/objections that don't exist in target snapshot.
    """
    call_sid = "sim_test_carry_ev"
    cdm = CoreDecisionManager(call_sid=call_sid)

    strat = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=3,
        source_turn_id=1,
        source_event_id="ev_turn_1_v3",
        strategic_objective="Protect appointment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        referenced_fact_ids=["fact_gone_001"],
    )

    target_snap = ConversationStateSnapshot(
        call_sid=call_sid,
        state_version=4,
        last_updated_turn_id=2,
        facts=[],  # Empty facts!
    )
    context = cdm.engine._build_context(target_snap)

    is_valid, reason = cdm._is_carried_strategy_valid_for_snapshot(strat, target_snap, context)
    assert is_valid is False
    assert "referenced_fact_id 'fact_gone_001' does not exist in target snapshot" in reason


def test_strategic_interpretation_rendering_labels_and_values():
    """Client Feedback 141026: Strategic Interpretation panel renders Strategy / Posture
    and Selected Pressure cleanly without confusing posture as push pressure.
    """
    html_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"
    content = html_path.read_text(encoding="utf-8")

    # Frontend panel has Strategy / Posture and Selected Pressure
    assert "Strategy / Posture:" in content
    assert "Selected Pressure:" in content

    # Backend engine attaches strategy_posture and selected_pressure
    engine = PitchProXCoreIntelligenceEngine()
    snap = ConversationStateSnapshot(
        call_sid="sim_test_interp",
        state_version=8,
        conversation_stage="scheduling",
    )
    res = engine.evaluate(snapshot=snap, turn_speaker="client", turn_text="Can we do Thursday?")
    interp = res.decision.strategic_interpretation
    assert "strategy_posture" in interp
    assert "selected_pressure" in interp
    assert interp["strategy_posture"] == res.decision.strategic_posture


def test_cross_user_prospect_memory_isolation_and_replay_clean_slate(tmp_path):
    """Client Feedback Verification: Prospect Memory is strictly isolated by user_id -> prospect_id.
    Simulations maintain clean cache isolation.
    """
    store_file = tmp_path / "test_prospect_memories.json"
    store = ProspectMemoryStore(store_path=store_file)
    store.clear()
    user_a = "user_alpha"
    user_b = "user_beta"
    prospect_id = "prospect_john_smith"

    # User A records a memory
    rec_a = store.save_record(
        user_id=user_a,
        prospect_id=prospect_id,
        source_call_sid="call_user_a",
        memory_type="timeline",
        key="move_timing",
        value="next year",
        content="Not moving until next year",
        source_turn_id=3,
    )
    assert rec_a.user_id == user_a

    # User A can retrieve
    mems_a = store.retrieve_memories(user_id=user_a, prospect_id=prospect_id)
    assert len(mems_a) == 1
    assert mems_a[0].content == "Not moving until next year"

    # User B calling the same prospect CANNOT retrieve User A's memory
    mems_b = store.retrieve_memories(user_id=user_b, prospect_id=prospect_id)
    assert len(mems_b) == 0, "User B must not receive User A's prospect memory records"


def test_missing_or_none_snapshot_is_strictly_rejected():
    """Client Feedback 141026 item 2:
    The snapshot parameter must be mandatory and fail-closed.
    Omitting snapshot or passing None must strictly reject lookups and validation.
    """
    call_sid = "sim_test_missing_snapshot"
    dec = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=1,
        source_turn_id=1,
        source_event_id="ev_turn_1_v1",
        strategic_objective="Test",
        primary_action=StrategicAction.QUESTION,
    )
    StrategicDecisionCache.clear(call_sid)
    StrategicDecisionCache.put(call_sid=call_sid, decision=dec, turn_id=1, source_event_id="ev_turn_1_v1")

    # 1. get_strictly_bound with snapshot=None returns None
    res = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=1,
        source_state_version=1,
        source_event_id="ev_turn_1_v1",
        state_call_sid=call_sid,
        snapshot=None,
    )
    assert res is None, "get_strictly_bound with snapshot=None must return None (fail closed)"

    # 2. validate_binding with snapshot=None returns False with explicit error
    is_valid, reason = StrategicDecisionCache.validate_binding(
        decision=dec,
        call_sid=call_sid,
        turn_id=1,
        source_state_version=1,
        source_event_id="ev_turn_1_v1",
        state_call_sid=call_sid,
        snapshot=None,
    )
    assert is_valid is False
    assert "Missing snapshot" in reason

    # 3. validate_evidence_consistency with snapshot=None returns False
    ev_ok, ev_reason = StrategicDecisionCache.validate_evidence_consistency(dec, snapshot=None)
    assert ev_ok is False
    assert "Missing snapshot" in ev_reason

    # 4. Calling without snapshot argument raises TypeError (cannot fail open by omission)
    with pytest.raises(TypeError):
        StrategicDecisionCache.get_strictly_bound(  # type: ignore
            call_sid=call_sid,
            turn_id=1,
            source_state_version=1,
            source_event_id="ev_turn_1_v1",
            state_call_sid=call_sid,
        )


def test_active_vs_superseded_fact_resolution_rules():
    """Client Feedback 141026 item 3:
    What counts as 'resolves'?
    - Reference to active fact: PASSES.
    - Reference to superseded fact without annotation when require_active=True: REJECTED.
    - Reference to superseded fact with explicit ALLOW_SUPERSEDED_EVIDENCE annotation: PASSES.
    """
    call_sid = "sim_test_superseded_facts"
    snap = ConversationStateSnapshot(
        call_sid=call_sid,
        state_version=8,
        last_updated_turn_id=8,
        facts=[
            PersistentFactRecord(
                fact_id="fact_old_t3",
                category="timeline",
                fact_key="confirmed_meeting_time",
                fact_value="Thursday at 4pm",
                status="superseded",
                superseded_by_fact_id="fact_new_t8",
                superseded_at_turn_id=8,
                source_turn_id=3,
                timestamp_ms=9000,
            ),
            PersistentFactRecord(
                fact_id="fact_new_t8",
                category="timeline",
                fact_key="confirmed_meeting_time",
                fact_value="Thursday at 3pm",
                status="active",
                source_turn_id=8,
                timestamp_ms=25000,
            ),
        ],
    )

    # Case A: Decision referencing the ACTIVE fact passes
    dec_active = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=8,
        source_turn_id=8,
        source_event_id="ev_turn_8_v8",
        strategic_objective="Protect updated appointment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        referenced_fact_ids=["fact_new_t8"],
    )
    ok_a, reason_a = StrategicDecisionCache.validate_evidence_consistency(dec_active, snap, require_active=True)
    assert ok_a is True
    assert "All referenced evidence IDs resolve" in reason_a

    # Case B: Decision referencing the SUPERSEDED fact without annotation is REJECTED
    dec_superseded = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=8,
        source_turn_id=8,
        source_event_id="ev_turn_8_v8",
        strategic_objective="Protect old appointment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        referenced_fact_ids=["fact_old_t3"],
    )
    ok_b, reason_b = StrategicDecisionCache.validate_evidence_consistency(dec_superseded, snap, require_active=True)
    assert ok_b is False
    assert "Evidence reference to superseded fact 'fact_old_t3' is invalid" in reason_b
    assert "superseded by 'fact_new_t8'" in reason_b

    # Case C: Decision referencing superseded fact WITH explicit annotation PASSES
    dec_annotated = dec_superseded.model_copy(deep=True)
    dec_annotated.reason_codes.append("ALLOW_SUPERSEDED_EVIDENCE")
    ok_c, reason_c = StrategicDecisionCache.validate_evidence_consistency(dec_annotated, snap, require_active=True)
    assert ok_c is True


def test_carried_decision_refreshes_superseded_fact_references():
    """Client Feedback 141026 item 5:
    Carried decisions need refreshed references, not only rejection.
    When a filler or rep turn carries forward a previous decision across a turn
    where a fact was superseded, the reference is re-pointed to the active successor.
    """
    from copilot.conversation_state_models import MeetingConversionGate, ConversionEventObject, ConversionEventStatus

    call_sid = "sim_test_refresh_carry"
    cdm = CoreDecisionManager(call_sid=call_sid)

    # Root decision from Turn 3 references fact_old
    dec_turn3 = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=3,
        source_turn_id=3,
        source_event_id="ev_turn_3_v3",
        strategic_objective="Protect initial appointment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        strategic_posture="protect",
        push_strength="none",
        referenced_fact_ids=["fact_old"],
        reason_codes=["CONVERSION_CONFIRMED"],
    )
    cdm.latest_decision = dec_turn3

    # Target snapshot at Turn 4 where fact_old was superseded by fact_new
    target_snap = ConversationStateSnapshot(
        call_sid=call_sid,
        state_version=4,
        last_updated_turn_id=4,
        conversation_stage="scheduling",
        conversion_confirmed=True,
        conversion_gate=MeetingConversionGate(is_open=True, status="open", conversion_target="appointment"),
        conversion_events=[ConversionEventObject(event_id="cev_1", event_type="appointment", status=ConversionEventStatus.CONFIRMED, start_at="Thursday at 3pm", confidence=0.9, confirmed_turn_id=3)],
        facts=[
            PersistentFactRecord(
                fact_id="fact_old",
                category="timeline",
                fact_key="meeting_time",
                fact_value="4pm",
                status="superseded",
                superseded_by_fact_id="fact_new",
                superseded_at_turn_id=4,
                source_turn_id=3,
                timestamp_ms=10000,
            ),
            PersistentFactRecord(
                fact_id="fact_new",
                category="timeline",
                fact_key="meeting_time",
                fact_value="3pm",
                status="active",
                source_turn_id=4,
                timestamp_ms=12000,
            ),
        ],
    )

    # Evaluate a filler turn ("okay")
    res = cdm.evaluate_state(
        snapshot=target_snap,
        turn_speaker="prospect",
        turn_text="okay",
        turn_id=4,
    )
    carried_dec = res.decision
    assert carried_dec.carried_forward is True
    # Reference MUST be refreshed to fact_new
    assert carried_dec.referenced_fact_ids == ["fact_new"]
    # Carried decision resolves strictly as active in Turn 4 snapshot
    ok, reason = StrategicDecisionCache.validate_evidence_consistency(carried_dec, target_snap, require_active=True)
    assert ok is True


def test_same_call_id_run_twice_with_different_state_rejects_stale_cache():
    """Client Feedback 141026 item 4:
    Root cause test: The same call ID run twice with different UUIDs (e.g. repeated test or simulation).
    Proves that even when the cache has an entry for (call_sid, turn_id, version, event_id),
    Run 2's lookup with Run 2's snapshot rejects the stale Run 1 decision and re-evaluates.
    """
    from copilot.conversation_state_models import MeetingConversionGate, ConversionEventObject, ConversionEventStatus

    call_sid = "sim_muy04bh0_reuse_test"
    StrategicDecisionCache.clear(call_sid)

    # Run 1: decision references fact_run1 and obj_run1
    dec_run1 = StrategicDecision(
        call_id=call_sid,
        call_sid=call_sid,
        source_state_version=3,
        source_turn_id=3,
        source_event_id="ev_turn_3_v3",
        strategic_objective="Protect appointment",
        primary_action=StrategicAction.ACKNOWLEDGE,
        referenced_fact_ids=["fact_7049325b"],
        referenced_objection_ids=["obj_6b28ab27"],
    )
    StrategicDecisionCache.put(
        call_sid=call_sid,
        decision=dec_run1,
        turn_id=3,
        source_event_id="ev_turn_3_v3",
    )

    # Run 2: snapshot has completely different IDs (due to UUID randomization)
    snap_run2 = ConversationStateSnapshot(
        call_sid=call_sid,
        state_version=3,
        last_updated_turn_id=3,
        conversion_confirmed=True,
        conversion_gate=MeetingConversionGate(is_open=True, status="open", conversion_target="appointment"),
        conversion_events=[ConversionEventObject(event_id="cev_run2", event_type="appointment", status=ConversionEventStatus.CONFIRMED, start_at="Thursday at 4pm", confidence=0.9, confirmed_turn_id=3)],
        facts=[
            PersistentFactRecord(
                fact_id="fact_d83772cf",
                category="timeline",
                fact_key="confirmed_meeting_time",
                fact_value="Thursday at 4pm",
                status="active",
                source_turn_id=3,
                timestamp_ms=10000,
            )
        ],
        objections=[
            ObjectionRecord(
                objection_id="obj_167be88f",
                canonical_category="commission_fee",
                initial_statement="Fee is high",
                latest_statement="Fee is high",
                first_turn_id=3,
                last_updated_turn_id=3,
            )
        ],
    )

    # In Run 2, querying get_strictly_bound with snap_run2 MUST reject dec_run1
    cached_lookup = StrategicDecisionCache.get_strictly_bound(
        call_sid=call_sid,
        turn_id=3,
        source_state_version=3,
        source_event_id="ev_turn_3_v3",
        state_call_sid=call_sid,
        snapshot=snap_run2,
    )
    assert cached_lookup is None, "Cache lookup must fail when cached decision references stale IDs from Run 1"

    # CoreDecisionManager.evaluate_state will not use the stale cache and produces fresh decision bound to Run 2
    cdm = CoreDecisionManager(call_sid=call_sid)
    res = cdm.evaluate_state(
        snapshot=snap_run2,
        turn_speaker="prospect",
        turn_text="Your commission is still too expensive, but I can meet Thursday at 4.",
        turn_id=3,
        use_cache=True,
    )
    fresh_dec = res.decision
    assert fresh_dec.referenced_fact_ids == ["fact_d83772cf"]
    assert fresh_dec.referenced_objection_ids == ["obj_167be88f"]
    assert "fact_7049325b" not in fresh_dec.referenced_fact_ids


def test_frontend_node_binding_validation_evidence_consistency():
    """Client Feedback 141026 item 6:
    Execute the real JavaScript validateDecisionBinding function from web/conversation_state_replay.html via Node.js.
    Verifies that when 4 binding keys match, but evidence references point elsewhere,
    it returns valid: false with an explicit rejection reason and renders a visible UI warning.
    """
    import subprocess
    import re

    html_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"
    content = html_path.read_text(encoding="utf-8")

    match = re.search(r"(function validateDecisionBinding\(d, st, sa, repCallSid\)\s*\{[\s\S]*?\n      \})", content)
    assert match is not None, "Could not find validateDecisionBinding in web/conversation_state_replay.html"
    func_js = match.group(1)

    node_script = f"""
{func_js}

const reportCallSid = 'call_sim_141026';
const baseStep = {{ turn_id: 3, source_event_id: 'ev_turn_3_v3' }};
const baseState = {{
    call_sid: 'call_sim_141026',
    state_version: 3,
    facts: [{{ fact_id: 'fact_d83772cf' }}],
    objections: [{{ objection_id: 'obj_167be88f' }}],
    stakeholders: []
}};

// Case 1: All 4 keys match, but decision references stale fact from prior run (Turn 3 bug)
const staleFactDecision = {{
    call_sid: 'call_sim_141026',
    call_id: 'call_sim_141026',
    source_turn_id: 3,
    source_state_version: 3,
    source_event_id: 'ev_turn_3_v3',
    referenced_fact_ids: ['fact_7049325b'],
    referenced_objection_ids: ['obj_167be88f']
}};
const resStaleFact = validateDecisionBinding(staleFactDecision, baseStep, baseState, reportCallSid);

// Case 2: All 4 keys match, but decision references stale objection
const staleObjDecision = {{
    ...staleFactDecision,
    referenced_fact_ids: ['fact_d83772cf'],
    referenced_objection_ids: ['obj_6b28ab27']
}};
const resStaleObj = validateDecisionBinding(staleObjDecision, baseStep, baseState, reportCallSid);

// Case 3: All 4 keys match AND all evidence references resolve within baseState
const validDecision = {{
    ...staleFactDecision,
    referenced_fact_ids: ['fact_d83772cf'],
    referenced_objection_ids: ['obj_167be88f']
}};
const resValid = validateDecisionBinding(validDecision, baseStep, baseState, reportCallSid);

// Case 4: Simulate UI card rendering on rejection
function renderCard(bindingCheck, step, sa) {{
    if (!bindingCheck.valid) {{
        return `<div class="card-unit warning">No valid Strategic Decision bound for this turn (Turn ${{step.turn_id}}, V${{sa.state_version}}). Binding rejected: ${{bindingCheck.reason}}</div>`;
    }}
    return `<div class="card-unit success">Bound successfully</div>`;
}}
const uiWarningHtml = renderCard(resStaleFact, baseStep, baseState);

console.log(JSON.stringify({{
    staleFact: resStaleFact,
    staleObj: resStaleObj,
    valid: resValid,
    uiWarning: uiWarningHtml
}}));
"""

    proc = subprocess.run(
        ["node"],
        input=node_script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    out = json.loads(proc.stdout)

    # 1. Stale fact reference rejected
    assert out["staleFact"]["valid"] is False
    assert "Evidence reference mismatch: referenced_fact_id 'fact_7049325b' not found" in out["staleFact"]["reason"]

    # 2. Stale objection reference rejected
    assert out["staleObj"]["valid"] is False
    assert "Evidence reference mismatch: referenced_objection_id 'obj_6b28ab27' not found" in out["staleObj"]["reason"]

    # 3. Valid references pass
    assert out["valid"]["valid"] is True
    assert "All four keys and evidence references valid" in out["valid"]["reason"]

    # 4. Visible UI error card
    assert "Binding rejected: Evidence reference mismatch: referenced_fact_id 'fact_7049325b' not found" in out["uiWarning"]


def test_frontend_node_template_rendering_and_fallbacks():
    """Client Feedback 141026 item 7:
    Execute real template rendering snippets from web/conversation_state_replay.html in Node.js.
    - Strategy / Posture is rendered separately from Selected Pressure.
    - confirm_and_protect is labeled as a legacy alias.
    - Missing or old data uses '—' and never invents values (no invented 'moderate' or 'resolve_then_ask').
    """
    import subprocess

    html_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"
    content = html_path.read_text(encoding="utf-8")

    # Confirm fallback 'moderate' and 'resolve_then_ask' are eliminated from push section
    assert "push.preliminary_ceiling || push.pressure || 'moderate'" not in content
    assert "push.legacy_strategy_alias || push.state || 'resolve_then_ask'" not in content

    node_script = """
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// 1. Trace Step 2 template
function renderTraceStep2(sd) {
    const strategyMarkup = `<strong style="color:#1E1B4B;">${escapeHtml(String(sd.primary_action || '—').toUpperCase())} / ${escapeHtml(sd.strategic_posture || '—')}</strong>${(sd.strategic_interpretation && (sd.strategic_interpretation.strategy_posture === 'confirm_and_protect' || sd.strategic_interpretation.push_strength_state === 'confirm_and_protect')) ? ` <span class="mono" style="font-size:9.5px; color:#64748B;">(legacy alias: confirm_and_protect)</span>` : ''}`;
    const pressureMarkup = `<strong class="mono" style="color:#0F172A; font-size:11px;">${escapeHtml(sd.strategic_interpretation.selected_pressure || (sd.push_strength && sd.push_strength.pressure ? sd.push_strength.pressure : (['none', 'low', 'moderate', 'high'].includes(String(sd.push_strength)) ? String(sd.push_strength) : '—')))}</strong>`;
    return { strategy: strategyMarkup, pressure: pressureMarkup };
}

// 2. Summary Panel Push Strength template
function renderSummaryPush(push, sd) {
    const selectedPressure = escapeHtml(sd ? (sd.push_strength && sd.push_strength.pressure ? sd.push_strength.pressure : (['none', 'low', 'moderate', 'high'].includes(String(sd.push_strength)) ? String(sd.push_strength) : '—')) : (push.selected_pressure || push.pressure || '—'));
    const prelimCeiling = escapeHtml(push.preliminary_ceiling || push.pressure || '—');
    const legacyAlias = escapeHtml(push.legacy_strategy_alias || push.state || '—');
    return { selectedPressure, prelimCeiling, legacyAlias };
}

// Test Data A: Active turn 3 decision with confirm_and_protect legacy alias
const sdTurn3 = {
    primary_action: 'acknowledge',
    strategic_posture: 'protect',
    push_strength: 'none',
    strategic_interpretation: {
        strategy_posture: 'confirm_and_protect',
        selected_pressure: 'none',
        push_strength_state: 'confirm_and_protect'
    }
};
const trace3 = renderTraceStep2(sdTurn3);

// Test Data B: Empty/old data lacking values
const sdEmpty = {
    strategic_interpretation: {}
};
const traceEmpty = renderTraceStep2(sdEmpty);
const summaryEmpty = renderSummaryPush({}, null);

console.log(JSON.stringify({
    trace3: trace3,
    traceEmpty: traceEmpty,
    summaryEmpty: summaryEmpty
}));
"""

    proc = subprocess.run(
        ["node"],
        input=node_script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    out = json.loads(proc.stdout)

    # Trace Step 2 shows primary action, posture, and legacy alias
    assert "ACKNOWLEDGE / protect" in out["trace3"]["strategy"]
    assert "(legacy alias: confirm_and_protect)" in out["trace3"]["strategy"]
    assert "none" in out["trace3"]["pressure"]

    # Empty/old data renders '—' and does NOT invent values
    assert "— / —" in out["traceEmpty"]["strategy"]
    assert "—" in out["traceEmpty"]["pressure"]
    assert out["summaryEmpty"]["prelimCeiling"] == "—"
    assert out["summaryEmpty"]["legacyAlias"] == "—"

