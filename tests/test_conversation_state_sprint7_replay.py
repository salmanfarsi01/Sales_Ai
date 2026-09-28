"""Sprint 7 / Phase 9 Test Suite: ConversationState Inspection & Replay Console.

Validates:
1. Replay Engine chronological execution: State Before (V_t-1) -> Evidence -> Changes -> State After (V_t)
2. State version lineage unbroken across all turns
3. Report serialization and disk persistence in reports/conversation_state_<call_sid>.json
4. Loading and listing reports from disk
5. FastAPI endpoints:
   - GET /conversation-state-replay
   - GET /api/conversation-state/reports
   - GET /api/conversation-state/replay/{call_sid}
   - POST /api/conversation-state/replay-dialogue
6. Standalone test runner (run_behavioral_signal.py) integration
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from copilot.conversation_replay import (
    ConversationReplayEngine,
    ConversationStateReplayReport,
    TurnReplayStep,
)
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import (
    DownstreamInferenceState,
    DimensionScore,
    EmotionState,
)
from copilot.behavioral_semantic import SemanticFeatureSnapshot
from copilot.fastapi_app import app
from run_behavioral_signal import app as runner_app


def _create_mock_bundle(
    turn_id: int,
    speaker_id: str,
    text: str,
    call_sid: str = "CA_replay_test_001",
    trust: float = 0.70,
    emotion_tension: float = 0.15,
    emotion_valence: float = 0.20,
    engagement: float = 0.70,
    readiness: float = 0.60,
    momentum: float = 0.65,
    boundary: float = 0.0,
    recurrence_id: str = None,
    salesperson_strategy_tag: str = None,
    specificity: float = 0.70,
    agreement: float = 0.60,
):
    ts_ms = 3000 * turn_id
    inf = DownstreamInferenceState(
        call_sid=call_sid,
        timestamp_ms=ts_ms,
        trust=DimensionScore(score=trust, confidence=0.85, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=emotion_valence, tension_level=emotion_tension, confidence=0.85),
        pacing=DimensionScore(score=0.60, confidence=0.80, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=engagement, confidence=0.85, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=momentum, confidence=0.80, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=readiness, confidence=0.80, primary_horizon="last_60_90s"),
        overall_confidence=0.85,
        contributing_evidence_ids=[f"ev_rep_{turn_id}"],
    )
    sem = SemanticFeatureSnapshot(
        utterance_id=f"utt_rep_{turn_id:03d}",
        call_sid=call_sid,
        speaker_id=speaker_id,
        boundary_score=boundary,
        recurrence_id=recurrence_id,
        specificity_score=specificity,
        agreement_score=agreement,
    )
    bundle = extract_behavioral_bundle(
        turn_id=turn_id,
        speaker_id=speaker_id,
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=sem,
        salesperson_strategy_tag=salesperson_strategy_tag,
    )
    bundle.timestamp_ms = ts_ms
    return bundle


class TestConversationStateSprint7Replay:
    def test_replay_engine_chronological_state_before_and_after(self, tmp_path):
        """Verifies that the replay engine faithfully captures the 4-step sequence:
        State Before (V_t-1) -> Evidence -> Changes + Reasons -> State After (V_t).
        """
        engine = ConversationReplayEngine(reports_dir=tmp_path)
        call_sid = "CA_chronological_001"

        # Turn 1: Agent intro
        t1 = _create_mock_bundle(
            turn_id=1,
            speaker_id="salesperson",
            text="Hi Daniel, I noticed your property was listed previously. Are you still open to moving?",
            salesperson_strategy_tag="open_discovery",
        )
        # Turn 2: Prospect voices fee objection
        t2 = _create_mock_bundle(
            turn_id=2,
            speaker_id="client",
            text="I don't know, 6 percent is just too high. That commission is steep.",
            recurrence_id="rec_fee_01",
            trust=0.60,
            emotion_tension=0.35,
        )
        # Turn 3: Agent reframe
        t3 = _create_mock_bundle(
            turn_id=3,
            speaker_id="salesperson",
            text="I completely understand. If we could net you substantially more in your pocket, would that make sense?",
            salesperson_strategy_tag="financial_net_proceeds_reframe",
        )
        # Turn 4: Prospect agrees to a walkthrough
        t4 = _create_mock_bundle(
            turn_id=4,
            speaker_id="client",
            text="Yes, come by Thursday at 4 PM to walk through the property.",
            trust=0.85,
            emotion_tension=0.10,
            engagement=0.85,
            agreement=0.90,
            specificity=0.85,
        )

        report = engine.replay_call(call_sid=call_sid, bundles=[t1, t2, t3, t4], save_report=True)

        assert report.call_sid == call_sid
        assert report.total_turns == 4
        assert len(report.timeline) == 4

        # Verify Step 1: Initial Bootstrap
        step1 = report.timeline[0]
        assert step1.turn_id == 1
        assert step1.speaker_id == "salesperson"
        assert step1.state_before.state_version == 1
        assert step1.state_after.state_version > 1

        # Verify Step 2: Objection raised
        step2 = report.timeline[1]
        assert step2.turn_id == 2
        assert step2.speaker_id == "client"
        assert step2.state_before.state_version == step1.state_after.state_version
        assert len(step2.state_after.objections) == 1
        assert step2.state_after.objections[0].canonical_category == "commission_fee"
        assert step2.state_after.objections[0].lifecycle_state == "unresolved"

        # Verify Step 4: Walkthrough confirmed
        step4 = report.timeline[3]
        assert step4.turn_id == 4
        assert step4.speaker_id == "client"
        assert step4.state_before.state_version == report.timeline[2].state_after.state_version
        assert step4.state_after.conversion_event is not None
        assert step4.state_after.conversion_event.status == "confirmed"
        assert step4.state_after.conversion_event.conversion_type == "property_walkthrough"

        # Verify unbroken state version lineage across all 4 turns
        for idx in range(1, len(report.timeline)):
            assert report.timeline[idx].state_before.state_version == report.timeline[idx - 1].state_after.state_version

    def test_replay_engine_report_persistence_and_loading(self, tmp_path):
        """Verifies report is saved to reports/ and can be cleanly reloaded by call_sid."""
        engine = ConversationReplayEngine(reports_dir=tmp_path)
        call_sid = "CA_persist_test_002"

        t1 = _create_mock_bundle(turn_id=1, speaker_id="client", text="Hello there, tell me about your pricing.")
        t2 = _create_mock_bundle(turn_id=2, speaker_id="client", text="Send me an email with your fees.", specificity=0.20)

        report = engine.replay_call(call_sid=call_sid, bundles=[t1, t2], save_report=True)

        expected_file = tmp_path / f"conversation_state_{call_sid}.json"
        assert expected_file.exists()

        # Load back via load_replay_report
        loaded = engine.load_replay_report(call_sid)
        assert loaded is not None
        assert loaded.call_sid == call_sid
        assert loaded.total_turns == 2
        assert len(loaded.timeline) == 2
        assert loaded.timeline[0].text == "Hello there, tell me about your pricing."

        # List reports
        available = engine.list_available_reports()
        assert len(available) == 1
        assert available[0]["call_sid"] == call_sid
        assert available[0]["total_turns"] == 2

    def test_fastapi_endpoints_serve_console_and_reports(self, tmp_path, monkeypatch):
        """Verifies FastAPI GET /conversation-state-replay, /api/conversation-state/reports,
        and /api/conversation-state/replay/{call_sid} endpoints.
        """
        from copilot import conversation_replay

        # Point reports_dir to tmp_path
        engine = ConversationReplayEngine(reports_dir=tmp_path)
        monkeypatch.setattr(
            conversation_replay,
            "ConversationReplayEngine",
            lambda *args, **kwargs: ConversationReplayEngine(reports_dir=tmp_path),
        )

        call_sid = "CA_fastapi_test_003"
        t1 = _create_mock_bundle(turn_id=1, speaker_id="client", text="Yes, we are interested in selling.")
        engine.replay_call(call_sid=call_sid, bundles=[t1], save_report=True)

        client = TestClient(app)

        # 1. Console UI endpoint
        resp_ui = client.get("/conversation-state-replay")
        assert resp_ui.status_code == 200
        assert "ConversationState Replay Console" in resp_ui.text
        assert "State Before" in resp_ui.text
        assert "New Evidence" in resp_ui.text

        # 2. Reports list endpoint
        resp_list = client.get("/api/conversation-state/reports")
        assert resp_list.status_code == 200
        reports_data = resp_list.json()
        assert any(r["call_sid"] == call_sid for r in reports_data)

        # 3. Single report replay endpoint
        resp_rep = client.get(f"/api/conversation-state/replay/{call_sid}")
        assert resp_rep.status_code == 200
        rep_data = resp_rep.json()
        assert rep_data["call_sid"] == call_sid
        assert rep_data["total_turns"] == 1
        assert len(rep_data["timeline"]) == 1

        # 4. Non-existent report returns 404
        resp_404 = client.get("/api/conversation-state/replay/non_existent_call_999")
        assert resp_404.status_code == 404

    def test_fastapi_replay_dialogue_simulation_endpoint(self, tmp_path, monkeypatch):
        """Verifies POST /api/conversation-state/replay-dialogue simulates dialogue turns."""
        from copilot import conversation_replay

        monkeypatch.setattr(
            conversation_replay,
            "ConversationReplayEngine",
            lambda *args, **kwargs: ConversationReplayEngine(reports_dir=tmp_path),
        )

        client = TestClient(app)
        payload = {
            "call_sid": "CA_sim_endpoint_004",
            "turns": [
                {"turn_id": 1, "speaker_id": "salesperson", "text": "Hi, are you looking to sell?"},
                {"turn_id": 2, "speaker_id": "client", "text": "Yeah, that sounds nice. Sure, maybe sometime.", "specificity": 0.20, "agreement": 0.75},
            ],
        }

        resp = client.post("/api/conversation-state/replay-dialogue", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["call_sid"] == "CA_sim_endpoint_004"
        assert data["total_turns"] == 2
        assert len(data["timeline"]) == 2

        # Verify friendly-but-vague prospect outcome on turn 2
        step2 = data["timeline"][1]
        assert step2["state_after"]["push_strength"]["state"] == "resolve_then_ask"

    def test_standalone_runner_serves_replay_console(self):
        """Verifies that run_behavioral_signal.py also serves /conversation-state-replay."""
        client = TestClient(runner_app)
        resp = client.get("/conversation-state-replay")
        assert resp.status_code == 200
        assert "ConversationState Replay Console" in resp.text

    def test_replay_collision_preserves_archived_runs(self, tmp_path):
        """Verifies that replaying the same call_sid a second time preserves the original audit
        trail in reports/archive/ rather than overwriting it destructively.
        """
        engine = ConversationReplayEngine(reports_dir=tmp_path)
        call_sid = "CA_collision_audit_005"

        t1 = _create_mock_bundle(turn_id=1, speaker_id="client", text="First run turn 1.")
        r1 = engine.replay_call(call_sid=call_sid, bundles=[t1], save_report=True)
        assert r1.run_count == 1
        assert len(r1.archived_versions) == 0

        # Run 2 on identical call_sid
        t2 = _create_mock_bundle(turn_id=2, speaker_id="client", text="Second run turn 2.")
        r2 = engine.replay_call(call_sid=call_sid, bundles=[t1, t2], save_report=True)
        assert r2.run_count == 2
        assert len(r2.archived_versions) == 1

        # Check archived file exists
        archived_rel = r2.archived_versions[0]
        archived_path = tmp_path.parent / archived_rel
        assert archived_path.exists()
        old_data = json.loads(archived_path.read_text(encoding="utf-8"))
        assert old_data["run_count"] == 1
        assert old_data["total_turns"] == 1

        # Check active report reflects latest run
        active_report = engine.load_replay_report(call_sid)
        assert active_report.run_count == 2
        assert active_report.total_turns == 2

    def test_synthetic_simulation_isolation_and_tagging(self, tmp_path):
        """Verifies dialogue simulations are stored in synthetic/ and clearly tagged."""
        engine = ConversationReplayEngine(reports_dir=tmp_path)
        turns = [
            {"turn_id": 1, "speaker_id": "salesperson", "text": "Hypothetical scenario."},
            {"turn_id": 2, "speaker_id": "client", "text": "Testing synthetic flow."},
        ]
        rep = engine.replay_dialogue_turns(call_sid="sim_test_006", raw_turns=turns, save_report=True)
        assert rep.source == "synthetic_simulation"
        assert "synthetic" in rep.report_file

        available = engine.list_available_reports()
        sim_entry = next((item for item in available if item["call_sid"] == "sim_test_006"), None)
        assert sim_entry is not None
        assert sim_entry["is_synthetic"] is True
        assert sim_entry["source"] == "synthetic_simulation"

    @pytest.mark.asyncio
    async def test_pipeline_service_auto_generates_conversation_state_report(self, tmp_path):
        """Verifies that execute_behavioral_turn_pipeline automatically feeds ConversationStateManager
        and writes both behavioral_signal_<call_sid>.json and conversation_state_<call_sid>.json.
        """
        from copilot.behavioral_pipeline_service import execute_behavioral_turn_pipeline
        from copilot.behavioral_normalization import NormalizedUtterance
        from copilot.behavioral_timing import DeterministicTimingEngine
        from copilot.behavioral_baseline import BaselineAndChangePointEngine
        from copilot.behavioral_evidence import MultiWindowAggregator, SQLiteEvidenceLogStore
        from copilot.behavioral_inference import DownstreamInferenceEngine
        from copilot.behavioral_semantic import SemanticFeatureEngine

        call_sid = "test_pipe_conv_007"
        db_path = tmp_path / "test_pipe.db"
        store = SQLiteEvidenceLogStore(db_path=db_path)
        timing_engine = DeterministicTimingEngine()
        baseline_engine = BaselineAndChangePointEngine(call_sid=call_sid)
        aggregator = MultiWindowAggregator(call_sid, timing_engine, baseline_engine, store)
        inference_engine = DownstreamInferenceEngine()
        semantic_engine = SemanticFeatureEngine()

        u1 = NormalizedUtterance(
            utterance_id="u1",
            call_sid=call_sid,
            speaker_id="salesperson",
            text="Hi Daniel, wanted to follow up on your listing.",
            start_ms=0,
            end_ms=2500,
        )
        u2 = NormalizedUtterance(
            utterance_id="u2",
            call_sid=call_sid,
            speaker_id="client",
            text="Yes, come by Thursday at 3 PM to take a look.",
            start_ms=3000,
            end_ms=6500,
        )

        result = await execute_behavioral_turn_pipeline(
            segmented_utts=[u1, u2],
            call_sid=call_sid,
            prospect_id="client",
            agent_id="salesperson",
            transcript="Hi Daniel... Yes, come by Thursday at 3 PM...",
            timing_engine=timing_engine,
            semantic_engine=semantic_engine,
            baseline_engine=baseline_engine,
            aggregator=aggregator,
            inference_engine=inference_engine,
            reports_dir=tmp_path,
        )

        # 1. Returned payload includes conversation_state and report link
        assert "conversation_state" in result
        assert "conversation_state_report" in result
        assert result["conversation_state"]["state_version"] > 1

        # 2. Both report files exist on disk
        state_file = tmp_path / f"conversation_state_{call_sid}.json"
        signal_file = tmp_path / f"behavioral_signal_{call_sid}.json"
        assert state_file.exists()
        assert signal_file.exists()

        # 3. ConversationReplayEngine loads it faithfully
        engine = ConversationReplayEngine(reports_dir=tmp_path)
        loaded = engine.load_replay_report(call_sid)
        assert loaded is not None
        assert loaded.total_turns == 2
        assert loaded.final_state.state_version == result["conversation_state"]["state_version"]

        store.close()

    def test_retrospective_replay_from_historical_behavioral_signal_report(self, tmp_path):
        """Verifies that an existing behavioral_signal_<call_sid>.json on disk can be
        retrospectively ingested on the fly when requested by ConversationReplayEngine.
        """
        # Create mock historical behavioral_signal file
        call_sid = "test_historical_past_call_008"
        hist_file = tmp_path / f"behavioral_signal_{call_sid}.json"
        hist_data = {
            "call_sid": call_sid,
            "prospect_id": "client",
            "agent_id": "salesperson",
            "play_by_play_turns": [
                {
                    "turn_index": 1,
                    "speaker_id": "salesperson",
                    "text": "Hi, I am calling regarding your property.",
                    "start_ms": 1000,
                    "trust_score": 0.70,
                    "engagement_score": 0.70,
                    "momentum_score": 0.60,
                    "readiness_score": 0.50,
                    "expressed_valence": 0.1,
                    "tension_level": 0.15,
                    "turn_confidence": 0.85,
                },
                {
                    "turn_index": 2,
                    "speaker_id": "client",
                    "text": "That sounds good, let's schedule an appointment for Friday.",
                    "start_ms": 4000,
                    "trust_score": 0.85,
                    "engagement_score": 0.85,
                    "momentum_score": 0.75,
                    "readiness_score": 0.70,
                    "expressed_valence": 0.3,
                    "tension_level": 0.10,
                    "boundary_score": 0.0,
                    "specificity_score": 0.85,
                    "agreement_score": 0.90,
                    "turn_confidence": 0.85,
                },
            ],
        }
        hist_file.write_text(json.dumps(hist_data, indent=2), encoding="utf-8")

        engine = ConversationReplayEngine(reports_dir=tmp_path)

        # 1. Listing discovers historical report and flags it as audio replay
        available = engine.list_available_reports()
        entry = next((item for item in available if item["call_sid"] == call_sid), None)
        assert entry is not None
        assert entry["has_state_replay"] is False
        assert entry["source"] == "recording_replay"

        # 2. Loading retrospectively processes turns into ConversationState
        loaded = engine.load_replay_report(call_sid)
        assert loaded is not None
        assert loaded.call_sid == call_sid
        assert loaded.total_turns == 2
        assert loaded.final_state.state_version > 1

        # 3. Serialized conversation_state_<call_sid>.json now written to disk
        state_file = tmp_path / f"conversation_state_{call_sid}.json"
        assert state_file.exists()

