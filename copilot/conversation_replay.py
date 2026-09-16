from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
    StateChangeRecord,
)
from .conversation_materiality import MaterialityClassification
from .conversation_state_manager import ConversationStateManager

LOGGER = logging.getLogger("copilot.conversation_replay")


class TurnReplayStep(BaseModel):
    """Represents a single turn step in the chronological conversation replay.

    Implements the core client spec view:
    State Before (V_t-1) -> New Evidence (Turn t) -> State Changes + Reasons -> State After (V_t)
    """

    turn_id: int
    speaker_id: str
    text: str
    timestamp_ms: int
    state_before: Optional[ConversationStateSnapshot] = None
    evidence_bundle: BehavioralSignalInputBundle
    materiality: MaterialityClassification
    state_changes: List[StateChangeRecord] = Field(default_factory=list)
    state_after: ConversationStateSnapshot


class ConversationStateReplayReport(BaseModel):
    """Complete serialized replay report for an entire call.

    Persisted permanently to reports/conversation_state_<call_sid>.json
    for post-launch inspection and real-call debugging.
    """

    call_sid: str
    source: str = Field(
        default="live_call",
        description="Origin of replay report: 'live_call', 'recording_replay', or 'synthetic_simulation'",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_turns: int = 0
    run_count: int = Field(default=1, description="Number of times this call SID has been replayed")
    archived_versions: List[str] = Field(
        default_factory=list,
        description="Relative paths to prior archived versions of this call SID",
    )
    final_state: ConversationStateSnapshot
    timeline: List[TurnReplayStep] = Field(default_factory=list)
    conversion_summary: Dict[str, Any] = Field(default_factory=dict)
    report_file: Optional[str] = None


class ConversationReplayEngine:
    """Replay and inspection engine for ConversationState.

    Executes a sequential series of turns, faithfully capturing
    the complete state before each turn, upstream evidence bundle,
    materiality gating decision, state change records, and state after.
    """

    def __init__(
        self,
        reports_dir: Optional[Path] = None,
        state_manager: Optional[ConversationStateManager] = None,
    ):
        if reports_dir is not None:
            self.reports_dir = Path(reports_dir)
        else:
            self.reports_dir = Path(__file__).resolve().parent.parent / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_reports_dir = self.reports_dir / "synthetic"
        self.synthetic_reports_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.reports_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = self.reports_dir.parent
        self.state_manager = state_manager

    def replay_call(
        self,
        call_sid: str,
        bundles: List[BehavioralSignalInputBundle],
        initial_state: Optional[ConversationStateSnapshot] = None,
        save_report: bool = True,
        source: str = "live_call",
    ) -> ConversationStateReplayReport:
        """Replays an ordered list of turn bundles through ConversationStateManager.

        Captures:
        - state_before (snapshot copy prior to turn execution)
        - evidence_bundle (upstream signals)
        - materiality classification (with reasoning and permitted targets)
        - state_changes (exact StateChangeRecord entries produced by this turn)
        - state_after (snapshot copy after turn execution)
        - collision detection and non-destructive archival for repeat replays
        """
        manager = self.state_manager or ConversationStateManager(
            call_sid=call_sid,
            initial_snapshot=initial_state,
        )

        timeline: List[TurnReplayStep] = []

        for bundle in bundles:
            # Capture deep copy of state before processing this turn
            state_before = manager.current_state.model_copy(deep=True)

            # Evaluate materiality
            materiality = manager.materiality_filter.classify_turn(
                bundle=bundle,
                current_state=manager.current_state,
                has_explicit_fact_updates=False,
            )

            # Record change history length prior to processing
            hist_len_before = len(manager.current_state.change_history)

            # Process turn through state manager
            state_after_ref = manager.process_turn_bundle(bundle)
            state_after = state_after_ref.model_copy(deep=True)

            # Extract StateChangeRecord entries produced by this specific turn
            turn_changes = [
                c.model_copy(deep=True)
                for c in manager.current_state.change_history[hist_len_before:]
                if c.triggering_turn_id == bundle.turn_id
            ]

            step = TurnReplayStep(
                turn_id=bundle.turn_id,
                speaker_id=bundle.speaker_id,
                text=bundle.utterance_text,
                timestamp_ms=bundle.timestamp_ms,
                state_before=state_before,
                evidence_bundle=bundle.model_copy(deep=True),
                materiality=materiality,
                state_changes=turn_changes,
                state_after=state_after,
            )
            timeline.append(step)

        final_state = manager.current_state.model_copy(deep=True)

        # Build conversion summary
        conv_summary = {
            "gate_is_open": final_state.conversion_gate.is_open if final_state.conversion_gate else False,
            "gate_status": final_state.conversion_gate.status if final_state.conversion_gate else "closed",
            "push_strength": final_state.push_strength.state if final_state.push_strength else "protect_and_shorten",
            "recommended_action": final_state.push_strength.recommended_action if final_state.push_strength else "",
            "failed_conditions": final_state.conversion_gate.failed_conditions if final_state.conversion_gate else [],
            "blocking_reasons": final_state.conversion_gate.blocking_reasons if final_state.conversion_gate else [],
            "conversion_event": final_state.conversion_event.model_dump() if final_state.conversion_event else None,
            "momentum_score": final_state.momentum.momentum_score if final_state.momentum else 50.0,
            "momentum_trend": final_state.momentum.trend if final_state.momentum else "stable",
            "readiness_score": final_state.readiness.readiness_score if final_state.readiness else 50.0,
            "active_blockers": final_state.readiness.active_blocker_caps if final_state.readiness else [],
        }

        # Tag and isolate synthetic dialogue simulations from live call recordings
        is_synthetic = (source == "synthetic_simulation") or ("sim_" in call_sid.lower())
        if is_synthetic:
            source = "synthetic_simulation"
            target_dir = self.synthetic_reports_dir
        else:
            target_dir = self.reports_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        report_file_path = target_dir / f"conversation_state_{call_sid}.json"

        # Archival and collision handling: Preserve prior run without overwriting audit history
        archived_versions: List[str] = []
        run_count = 1
        if save_report and report_file_path.exists():
            try:
                old_raw = json.loads(report_file_path.read_text(encoding="utf-8"))
                prior_created = old_raw.get("created_at", "")
                prior_ts = prior_created.replace(":", "-").replace(".", "-")
                if not prior_ts:
                    mtime = report_file_path.stat().st_mtime
                    prior_ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

                archive_dir = target_dir / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_file = archive_dir / f"conversation_state_{call_sid}_{prior_ts}.json"
                archive_file.write_text(report_file_path.read_text(encoding="utf-8"), encoding="utf-8")

                prior_archives = old_raw.get("archived_versions", [])
                try:
                    rel_archive = str(archive_file.relative_to(self.base_dir)).replace("\\", "/")
                except Exception:
                    rel_archive = str(archive_file).replace("\\", "/")
                archived_versions = list(prior_archives) + [rel_archive]
                run_count = int(old_raw.get("run_count", 1)) + 1
                LOGGER.info("Archived prior report for %s to %s (run #%d)", call_sid, archive_file, run_count)
            except Exception as exc:
                LOGGER.warning("Could not archive prior report for %s: %s", call_sid, exc)

        try:
            rel_report_file = str(report_file_path.relative_to(self.base_dir)).replace("\\", "/")
        except Exception:
            rel_report_file = str(report_file_path).replace("\\", "/")

        report = ConversationStateReplayReport(
            call_sid=call_sid,
            source=source,
            total_turns=len(timeline),
            run_count=run_count,
            archived_versions=archived_versions,
            final_state=final_state,
            timeline=timeline,
            conversion_summary=conv_summary,
            report_file=rel_report_file,
        )

        if save_report:
            try:
                report_file_path.write_text(
                    report.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                LOGGER.info("Saved ConversationState replay report (%s) to %s", source, report_file_path)
            except Exception as exc:
                LOGGER.error("Failed to write replay report %s: %s", report_file_path, exc)

        return report

    def replay_behavioral_signal_report(
        self,
        signal_report_path: Path,
        save_report: bool = True,
    ) -> Optional[ConversationStateReplayReport]:
        """Ingests a historical behavioral signal report and replays it through ConversationState."""
        from .conversation_state_contract import extract_behavioral_bundle
        from .behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
        from .behavioral_semantic import SemanticFeatureSnapshot

        if not signal_report_path.exists():
            return None

        try:
            raw_data = json.loads(signal_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.error("Failed to read behavioral signal report %s: %s", signal_report_path, exc)
            return None

        call_sid = raw_data.get("call_sid", signal_report_path.stem.replace("behavioral_signal_", ""))
        turns_data = raw_data.get("play_by_play_turns", [])
        if not turns_data:
            return None

        bundles: List[BehavioralSignalInputBundle] = []
        for idx, t in enumerate(turns_data):
            turn_id = int(t.get("turn_index", idx + 1))
            speaker_id = str(t.get("speaker_id", "client"))
            text = str(t.get("text", ""))
            ts_ms = int(t.get("start_ms", idx * 3000))

            sem_snap = SemanticFeatureSnapshot(
                utterance_id=f"utt_{call_sid}_{turn_id:03d}",
                call_sid=call_sid,
                speaker_id=speaker_id,
                boundary_score=float(t.get("boundary_score", 0.0)),
                recurrence_type=t.get("recurrence_type", "none"),
                recurrence_id=t.get("recurrence_id"),
                recurrence_count=int(t.get("recurrence_count", 0)),
                question_type=t.get("question_type", "none"),
                specificity_score=float(t.get("specificity_score", 0.60)),
                future_language_score=float(t.get("future_language_score", 0.0)),
                agreement_score=float(t.get("agreement_score", 0.50)),
                contact_preference=t.get("contact_preference", "none"),
                contact_preference_confidence=float(t.get("contact_preference_confidence", 0.0)),
                contact_preference_details=t.get("contact_preference_details"),
            )

            trust_val = float(t.get("trust_score", 0.70))
            tension_val = float(t.get("tension_level", 0.20))
            valence_val = float(t.get("expressed_valence", 0.0))
            eng_val = float(t.get("engagement_score", 0.70))
            read_val = float(t.get("readiness_score", 0.50))
            mom_val = float(t.get("momentum_score", 0.60))
            conf_val = float(t.get("turn_confidence", 0.85))

            inf_state = DownstreamInferenceState(
                call_sid=call_sid,
                timestamp_ms=ts_ms,
                trust=DimensionScore(score=trust_val, confidence=conf_val, primary_horizon="last_20_30s"),
                emotion=EmotionState(expressed_valence=valence_val, tension_level=tension_val, confidence=conf_val),
                pacing=DimensionScore(score=float(t.get("pacing_score", 0.60)), confidence=conf_val, primary_horizon="current_utterance"),
                engagement=DimensionScore(score=eng_val, confidence=conf_val, primary_horizon="last_20_30s"),
                momentum=DimensionScore(score=mom_val, confidence=conf_val, primary_horizon="last_60_90s"),
                readiness=DimensionScore(score=read_val, confidence=conf_val, primary_horizon="last_60_90s"),
                overall_confidence=conf_val,
                contributing_evidence_ids=[f"ev_{call_sid}_{turn_id}"],
            )

            bundle = extract_behavioral_bundle(
                turn_id=turn_id,
                speaker_id=speaker_id,
                utterance_text=text,
                inference_state=inf_state,
                semantic_snapshot=sem_snap,
            )
            bundle.timestamp_ms = ts_ms
            bundles.append(bundle)

        source_tag = "synthetic_simulation" if "sim_" in call_sid.lower() else "recording_replay"
        return self.replay_call(
            call_sid=call_sid,
            bundles=bundles,
            save_report=save_report,
            source=source_tag,
        )

    def load_replay_report(self, call_sid: str) -> Optional[ConversationStateReplayReport]:
        """Loads a stored replay report from the reports directory or synthetic directory.

        If not yet generated, retrospectively replays from any available behavioral_signal_<call_sid>.json.
        """
        safe_sid = "".join(c for c in call_sid if c.isalnum() or c in ("-", "_"))
        # Search live reports first, then synthetic
        target = self.reports_dir / f"conversation_state_{safe_sid}.json"
        if target.exists():
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
                return ConversationStateReplayReport.model_validate(data)
            except Exception as exc:
                LOGGER.error("Error loading replay report for %s: %s", safe_sid, exc)

        synth_target = self.synthetic_reports_dir / f"conversation_state_{safe_sid}.json"
        if synth_target.exists():
            try:
                data = json.loads(synth_target.read_text(encoding="utf-8"))
                return ConversationStateReplayReport.model_validate(data)
            except Exception as exc:
                LOGGER.error("Error loading synthetic replay report for %s: %s", safe_sid, exc)

        # Retrospective ingestion: check for historical behavioral signal report
        sig_target = self.reports_dir / f"behavioral_signal_{safe_sid}.json"
        if sig_target.exists():
            LOGGER.info("Retrospectively generating ConversationState report for historical call %s", safe_sid)
            return self.replay_behavioral_signal_report(sig_target, save_report=True)

        return None

    def list_available_reports(self) -> List[Dict[str, Any]]:
        """Lists all existing conversation state replay reports and available historical recordings."""
        items: List[Dict[str, Any]] = []
        known_sids = set()

        search_dirs = [self.reports_dir, self.synthetic_reports_dir]
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for p in sorted(sdir.glob("conversation_state_*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
                try:
                    content = json.loads(p.read_text(encoding="utf-8"))
                    sid = content.get("call_sid", p.stem.replace("conversation_state_", ""))
                    known_sids.add(sid)
                    conv = content.get("conversion_summary", {})
                    src = content.get(
                        "source",
                        "synthetic_simulation" if ("sim_" in sid.lower() or "synthetic" in str(p)) else "live_call",
                    )
                    items.append({
                        "call_sid": sid,
                        "filename": p.name,
                        "created_at": content.get("created_at", ""),
                        "source": src,
                        "is_synthetic": src == "synthetic_simulation",
                        "has_state_replay": True,
                        "run_count": content.get("run_count", 1),
                        "archived_runs_count": len(content.get("archived_versions", [])),
                        "total_turns": content.get("total_turns", 0),
                        "gate_status": conv.get("gate_status", "closed"),
                        "push_strength": conv.get("push_strength", "protect_and_shorten"),
                        "momentum_score": conv.get("momentum_score", 50.0),
                        "readiness_score": conv.get("readiness_score", 50.0),
                        "mtime": p.stat().st_mtime,
                    })
                except Exception as exc:
                    LOGGER.warning("Could not read report file %s: %s", p, exc)
                    continue

        # Also discover historical behavioral signal reports that can be replayed on demand
        if self.reports_dir.exists():
            for p in sorted(self.reports_dir.glob("behavioral_signal_*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
                sid = p.stem.replace("behavioral_signal_", "")
                if sid in known_sids:
                    continue
                known_sids.add(sid)
                try:
                    content = json.loads(p.read_text(encoding="utf-8"))
                    turns = len(content.get("utterances", content.get("play_by_play_turns", [])))
                    items.append({
                        "call_sid": sid,
                        "filename": p.name,
                        "created_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                        "source": "recording_replay",
                        "is_synthetic": False,
                        "has_state_replay": False,
                        "run_count": 0,
                        "archived_runs_count": 0,
                        "total_turns": turns,
                        "gate_status": "unprocessed (click to replay)",
                        "push_strength": "ready",
                        "momentum_score": 50.0,
                        "readiness_score": 50.0,
                        "mtime": p.stat().st_mtime,
                    })
                except Exception:
                    continue

        items.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return items

    def replay_dialogue_turns(
        self,
        call_sid: str,
        raw_turns: List[Dict[str, Any]],
        save_report: bool = True,
    ) -> ConversationStateReplayReport:
        """Helper to convert raw turn dicts into bundles and run replay.

        Always tags reports as synthetic_simulation to guarantee zero mixing with real calls.
        """
        from .conversation_state_contract import extract_behavioral_bundle
        from .behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
        from .behavioral_semantic import SemanticFeatureEngine, SemanticFeatureSnapshot

        # Guarantee synthetic prefix if not present
        if not call_sid.lower().startswith("sim_") and "sim" not in call_sid.lower():
            call_sid = f"sim_{call_sid}"

        bundles: List[BehavioralSignalInputBundle] = []

        for t in raw_turns:
            tid = int(t.get("turn_id", len(bundles) + 1))
            spk = t.get("speaker_id", "client")
            txt = t.get("text", "")
            ts_ms = int(t.get("timestamp_ms", tid * 3000))

            sem_snap = SemanticFeatureSnapshot(
                utterance_id=f"utt_sim_{tid:03d}",
                call_sid=call_sid,
                speaker_id=spk,
                boundary_score=float(t.get("boundary", 0.0)),
                recurrence_id=t.get("recurrence_id"),
                specificity_score=float(t.get("specificity", 0.60)),
                agreement_score=float(t.get("agreement", 0.50)),
            )

            trust_val = float(t.get("trust", 0.70))
            tension_val = float(t.get("tension", 0.15))
            valence_val = float(t.get("valence", 0.20))
            eng_val = float(t.get("engagement", 0.70))
            read_val = float(t.get("readiness", 0.60))
            mom_val = float(t.get("momentum", 0.65))

            inf_state = DownstreamInferenceState(
                call_sid=call_sid,
                timestamp_ms=ts_ms,
                trust=DimensionScore(score=trust_val, confidence=0.85, primary_horizon="last_20_30s"),
                emotion=EmotionState(expressed_valence=valence_val, tension_level=tension_val, confidence=0.85),
                pacing=DimensionScore(score=0.60, confidence=0.80, primary_horizon="current_utterance"),
                engagement=DimensionScore(score=eng_val, confidence=0.85, primary_horizon="last_20_30s"),
                momentum=DimensionScore(score=mom_val, confidence=0.80, primary_horizon="last_60_90s"),
                readiness=DimensionScore(score=read_val, confidence=0.80, primary_horizon="last_60_90s"),
                overall_confidence=0.85,
                contributing_evidence_ids=[f"ev_sim_{tid}"],
            )

            bundle = extract_behavioral_bundle(
                turn_id=tid,
                speaker_id=spk,
                utterance_text=txt,
                inference_state=inf_state,
                semantic_snapshot=sem_snap,
                salesperson_strategy_tag=t.get("strategy_tag"),
            )
            bundle.timestamp_ms = ts_ms
            bundles.append(bundle)

        return self.replay_call(
            call_sid=call_sid,
            bundles=bundles,
            save_report=save_report,
            source="synthetic_simulation",
        )


class ReplayDialogueRequest(BaseModel):
    """Request payload for dialogue simulation endpoint."""
    call_sid: Optional[str] = None
    turns: List[Dict[str, Any]] = Field(default_factory=list)


def get_conversation_replay_router(
    reports_dir: Optional[Path] = None,
    html_file_path: Optional[Path] = None,
):
    """Shared-service FastAPI APIRouter exposing the ConversationState Replay & Inspection Console.

    Single source of truth mounted identically by both:
    1. copilot/fastapi_app.py (main production application server)
    2. run_behavioral_signal.py (standalone testing console runner)

    Guarantees zero route definition drift across the repository.
    """
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse
    import uuid

    router = APIRouter(tags=["conversation-state-replay"])

    if html_file_path is None:
        html_file_path = Path(__file__).resolve().parent.parent / "web" / "conversation_state_replay.html"

    @router.get("/conversation-state-replay")
    async def serve_conversation_state_replay():
        if not html_file_path.exists():
            raise HTTPException(status_code=404, detail="conversation_state_replay.html not found")
        return FileResponse(html_file_path)

    @router.get("/api/conversation-state/reports")
    async def list_conversation_state_reports():
        engine = ConversationReplayEngine(reports_dir=reports_dir)
        return engine.list_available_reports()

    @router.get("/api/conversation-state/replay/{call_sid}")
    async def get_conversation_state_replay(call_sid: str):
        engine = ConversationReplayEngine(reports_dir=reports_dir)
        report = engine.load_replay_report(call_sid)
        if not report:
            raise HTTPException(status_code=404, detail=f"Replay report for {call_sid} not found")
        return report.model_dump()

    @router.post("/api/conversation-state/replay-dialogue")
    async def replay_conversation_state_dialogue(req: ReplayDialogueRequest):
        call_sid = req.call_sid or f"sim_{uuid.uuid4().hex[:8]}"
        if not req.turns:
            raise HTTPException(status_code=400, detail="Must supply non-empty list of turns")
        engine = ConversationReplayEngine(reports_dir=reports_dir)
        report = engine.replay_dialogue_turns(call_sid=call_sid, raw_turns=req.turns, save_report=True)
        return report.model_dump()

    return router

