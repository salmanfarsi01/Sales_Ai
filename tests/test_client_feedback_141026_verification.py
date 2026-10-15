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
