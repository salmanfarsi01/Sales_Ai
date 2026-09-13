from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .behavioral_normalization import NormalizedTurnEvent, NormalizedUtterance
from .behavioral_timing import DeterministicTimingEngine, TimingFeatureSnapshot
from .behavioral_semantic import SemanticFeatureSnapshot
from .behavioral_baseline import (
    BaselineAndChangePointEngine,
    ChangePointEvent,
    FeatureDeviation,
)

WindowHorizon = Literal[
    "current_utterance",
    "last_5_10s",
    "last_20_30s",
    "last_60_90s",
    "full_call",
]

HORIZON_DURATIONS_MS: Dict[WindowHorizon, int] = {
    "current_utterance": 0,
    "last_5_10s": 10000,
    "last_20_30s": 30000,
    "last_60_90s": 60000,
    "full_call": -1,
}


class ConfidencePillarConfig(BaseModel):
    """Named, versioned, and documented calibration thresholds for the 5-pillar confidence model.

    All thresholds are v1 provisional estimates established for honest uncertainty representation,
    pending empirical calibration against labeled telephony conversion datasets.
    """
    version: str = "1.0.0"

    baseline_unlocked_prior: float = Field(
        0.70,
        description=(
            "Confidence cap when a speaker's baseline profile is actively calibrating (<3 turns / <15 words / <60s). "
            "Timing metrics (WPM, pause durations, turn lengths) are true acoustic facts measured directly from "
            "word timestamps (not fabricated), and population norms (~150 WPM) provide a reasonable baseline prior. "
            "However, individual baseline divergence claims (z-scores) remain provisional until intra-call calibration "
            "locks against the speaker's own cadence, capping initial confidence at 0.70."
        ),
    )
    single_channel_diarization_certainty: float = Field(
        0.80,
        description=(
            "Certainty cap when relying on single-channel diarization rather than isolated dual-track audio. "
            "Single-channel acoustic speaker clustering exhibits an estimated 15-20% error rate on conversational "
            "telephony (especially during rapid turn-taking, backchannels, and simultaneous talk). While dual-track "
            "RTP audio achieves 1.0 (zero speaker ambiguity), single-channel diarization is capped at 0.80 pending "
            "empirical diarization error-rate (DER) benchmarks."
        ),
    )
    synthetic_timing_certainty: float = Field(
        0.50,
        description=(
            "Certainty cap when word-level timestamps are unavailable and timestamps are linearly interpolated from "
            "utterance boundaries. Ensures synthetic timing estimates cannot produce high-confidence downstream judgments."
        ),
    )
    heuristic_semantic_certainty: float = Field(
        0.65,
        description=(
            "Certainty cap for deterministic heuristic/regex semantic extraction fallback. Rule-based regex reliably "
            "detects explicit phrasing but lacks contextual nuance and pragmatic inference compared to LLM reasoning."
        ),
    )
    heuristic_bypass_certainty: float = Field(
        0.50,
        description=(
            "Certainty cap for short conversational fragments (1-2 words) that bypass deep semantic evaluation, "
            "leaving unmeasured dimensions at neutral priors."
        ),
    )
    high_confidence_ceiling: float = Field(
        0.95,
        description="Ceiling for pipeline confidence when all 5 pillars (dual-track, locked baseline, LLM, clean ASR) are satisfied.",
    )


class BehavioralEvidenceSnapshot(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    call_sid: str
    timestamp_ms: int
    horizon: WindowHorizon
    window_duration_ms: int
    timing_features: TimingFeatureSnapshot
    semantic_features: Optional[SemanticFeatureSnapshot] = None
    deviations: List[FeatureDeviation] = Field(default_factory=list)
    change_points: List[ChangePointEvent] = Field(default_factory=list)
    talk_ratio_client: float = Field(0.5, ge=0.0, le=1.0)
    contributing_utterance_ids: List[str] = Field(default_factory=list)
    contributing_event_ids: List[str] = Field(default_factory=list)
    evidence_confidence: float = Field(1.0, ge=0.0, le=1.0)
    confidence_driver: Optional[str] = None
    confidence_breakdown: Optional[Dict[str, float]] = None


class MultiWindowEvidenceFrame(BaseModel):
    frame_id: str = Field(default_factory=lambda: f"frm_{uuid.uuid4().hex[:12]}")
    call_sid: str
    timestamp_ms: int
    windows: Dict[str, BehavioralEvidenceSnapshot]


class EvidenceLogStore(ABC):
    @abstractmethod
    def append_frame(self, frame: MultiWindowEvidenceFrame) -> None:
        pass

    @abstractmethod
    def get_latest_frame(self, call_sid: str) -> Optional[MultiWindowEvidenceFrame]:
        pass

    @abstractmethod
    def get_window_snapshot(
        self, call_sid: str, horizon: str, timestamp_ms: Optional[int] = None
    ) -> Optional[BehavioralEvidenceSnapshot]:
        pass

    @abstractmethod
    def backward_trace(self, evidence_ids: List[str]) -> List[BehavioralEvidenceSnapshot]:
        pass

    @abstractmethod
    def query_time_range(
        self, call_sid: str, start_ms: int, end_ms: int, horizon: Optional[str] = None
    ) -> List[BehavioralEvidenceSnapshot]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SQLiteEvidenceLogStore(EvidenceLogStore):
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent / "knowledge"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(base_dir / "evidence_log.db")
        else:
            self.db_path = str(db_path)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_frames (
                        frame_id TEXT PRIMARY KEY,
                        call_sid TEXT NOT NULL,
                        timestamp_ms INTEGER NOT NULL,
                        frame_json TEXT NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_snapshots (
                        evidence_id TEXT PRIMARY KEY,
                        frame_id TEXT NOT NULL,
                        call_sid TEXT NOT NULL,
                        timestamp_ms INTEGER NOT NULL,
                        horizon TEXT NOT NULL,
                        snapshot_json TEXT NOT NULL,
                        FOREIGN KEY (frame_id) REFERENCES evidence_frames(frame_id)
                    )
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_frames_call_time ON evidence_frames(call_sid, timestamp_ms)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_snaps_lookup ON evidence_snapshots(call_sid, horizon, timestamp_ms)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_snaps_id ON evidence_snapshots(evidence_id)"
                )

    def append_frame(self, frame: MultiWindowEvidenceFrame) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO evidence_frames (frame_id, call_sid, timestamp_ms, frame_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (frame.frame_id, frame.call_sid, frame.timestamp_ms, frame.model_dump_json()),
                )
                for horizon, snapshot in frame.windows.items():
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO evidence_snapshots (evidence_id, frame_id, call_sid, timestamp_ms, horizon, snapshot_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot.evidence_id,
                            frame.frame_id,
                            frame.call_sid,
                            snapshot.timestamp_ms,
                            horizon,
                            snapshot.model_dump_json(),
                        ),
                    )

    def get_latest_frame(self, call_sid: str) -> Optional[MultiWindowEvidenceFrame]:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT frame_json FROM evidence_frames
                WHERE call_sid = ?
                ORDER BY timestamp_ms DESC LIMIT 1
                """,
                (call_sid,),
            )
            row = cursor.fetchone()
            if row:
                return MultiWindowEvidenceFrame.model_validate_json(row[0])
            return None

    def get_window_snapshot(
        self, call_sid: str, horizon: str, timestamp_ms: Optional[int] = None
    ) -> Optional[BehavioralEvidenceSnapshot]:
        with self._lock:
            cursor = self._conn.cursor()
            if timestamp_ms is None:
                cursor.execute(
                    """
                    SELECT snapshot_json FROM evidence_snapshots
                    WHERE call_sid = ? AND horizon = ?
                    ORDER BY timestamp_ms DESC LIMIT 1
                    """,
                    (call_sid, horizon),
                )
            else:
                cursor.execute(
                    """
                    SELECT snapshot_json FROM evidence_snapshots
                    WHERE call_sid = ? AND horizon = ? AND timestamp_ms <= ?
                    ORDER BY timestamp_ms DESC LIMIT 1
                    """,
                    (call_sid, horizon, timestamp_ms),
                )
            row = cursor.fetchone()
            if row:
                return BehavioralEvidenceSnapshot.model_validate_json(row[0])
            return None

    def backward_trace(self, evidence_ids: List[str]) -> List[BehavioralEvidenceSnapshot]:
        if not evidence_ids:
            return []
        placeholders = ",".join("?" for _ in evidence_ids)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                f"""
                SELECT snapshot_json FROM evidence_snapshots
                WHERE evidence_id IN ({placeholders})
                ORDER BY timestamp_ms ASC
                """,
                evidence_ids,
            )
            rows = cursor.fetchall()
            return [BehavioralEvidenceSnapshot.model_validate_json(r[0]) for r in rows]

    def query_time_range(
        self, call_sid: str, start_ms: int, end_ms: int, horizon: Optional[str] = None
    ) -> List[BehavioralEvidenceSnapshot]:
        with self._lock:
            cursor = self._conn.cursor()
            if horizon is not None:
                cursor.execute(
                    """
                    SELECT snapshot_json FROM evidence_snapshots
                    WHERE call_sid = ? AND horizon = ? AND timestamp_ms BETWEEN ? AND ?
                    ORDER BY timestamp_ms ASC
                    """,
                    (call_sid, horizon, start_ms, end_ms),
                )
            else:
                cursor.execute(
                    """
                    SELECT snapshot_json FROM evidence_snapshots
                    WHERE call_sid = ? AND timestamp_ms BETWEEN ? AND ?
                    ORDER BY timestamp_ms ASC
                    """,
                    (call_sid, start_ms, end_ms),
                )
            rows = cursor.fetchall()
            return [BehavioralEvidenceSnapshot.model_validate_json(r[0]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class MultiWindowAggregator:
    def __init__(
        self,
        call_sid: str,
        timing_engine: DeterministicTimingEngine,
        baseline_engine: Optional[BaselineAndChangePointEngine] = None,
        store: Optional[EvidenceLogStore] = None,
        confidence_config: Optional[ConfidencePillarConfig] = None,
    ):
        self.call_sid = call_sid
        self.timing_engine = timing_engine
        self.baseline_engine = baseline_engine
        self.store = store or SQLiteEvidenceLogStore()
        self.confidence_config = confidence_config or ConfidencePillarConfig()

        self.utterances: List[NormalizedUtterance] = []
        self.turn_events: List[NormalizedTurnEvent] = []
        self.semantic_snapshots: List[SemanticFeatureSnapshot] = []
        self.change_points: List[ChangePointEvent] = []
        self.latest_frame: Optional[MultiWindowEvidenceFrame] = None

    def _compute_talk_ratio(self, utts: List[NormalizedUtterance]) -> float:
        client_duration = sum(
            max(1, u.end_ms - u.start_ms) for u in utts if u.speaker_id == "client"
        )
        sales_duration = sum(
            max(1, u.end_ms - u.start_ms) for u in utts if u.speaker_id == "salesperson"
        )
        total = client_duration + sales_duration
        if total <= 0:
            return 0.5
        return round(client_duration / total, 3)

    def _aggregate_semantic_for_window(
        self,
        window_utts: List[NormalizedUtterance],
        fallback_snap: Optional[SemanticFeatureSnapshot] = None,
    ) -> Optional[SemanticFeatureSnapshot]:
        """Aggregates semantic evidence across all contributing utterances in a window horizon.
        Preserves active objections/recurrence, persistent boundary markers, and peak disclosure scores.
        """
        if not window_utts:
            return fallback_snap

        window_utt_ids = {u.utterance_id for u in window_utts}
        snaps = [s for s in self.semantic_snapshots if s.utterance_id in window_utt_ids]
        if not snaps:
            return fallback_snap

        if len(snaps) == 1:
            return snaps[0]

        latest = snaps[-1]

        # 1. Recurrence: find most recent active objection or recurrence in this window.
        # rec_count must come strictly from active_rec to prevent pairing a 1st occurrence with an older turn's count.
        active_rec = next((s for s in reversed(snaps) if s.recurrence_type != "none"), None)
        rec_type = active_rec.recurrence_type if active_rec else "none"
        rec_id = active_rec.recurrence_id if active_rec else None
        rec_count = active_rec.recurrence_count if active_rec else 0

        # 2. Boundary: persistent within the window (max score)
        snap_boundary = max(snaps, key=lambda s: s.boundary_score)
        max_boundary = snap_boundary.boundary_score
        boundary_conf = (
            snap_boundary.boundary_confidence if max_boundary > 0.0 else min(s.boundary_confidence for s in snaps)
        )

        # 3. Question: find latest active question in the window
        active_q = next((s for s in reversed(snaps) if s.question_type != "none"), None)
        q_type = active_q.question_type if active_q else "none"
        q_conf = active_q.question_type_confidence if active_q else min(s.question_type_confidence for s in snaps)

        # 4. Continuous scores: peak presence in this window horizon
        # Per-feature confidences track the snapshot that provided the peak measurement (or weakest link if unmeasured 0.0)
        snap_spec = max(snaps, key=lambda s: s.specificity_score)
        max_spec = snap_spec.specificity_score
        spec_conf = snap_spec.specificity_confidence if max_spec > 0.0 else min(s.specificity_confidence for s in snaps)

        snap_future = max(snaps, key=lambda s: s.future_language_score)
        max_future = snap_future.future_language_score
        future_conf = (
            snap_future.future_language_confidence if max_future > 0.0 else min(s.future_language_confidence for s in snaps)
        )

        snap_agree = max(snaps, key=lambda s: s.agreement_score)
        max_agree = snap_agree.agreement_score
        agree_conf = snap_agree.agreement_confidence if max_agree > 0.0 else min(s.agreement_confidence for s in snaps)

        rec_conf = active_rec.recurrence_confidence if active_rec else min(s.recurrence_confidence for s in snaps)

        # 5. Extraction mode:
        # - "llm" only if 100% of snaps in the window used LLM
        # - "mixed" if partial (both LLM and heuristic turns present)
        # - When 0% used LLM, peak severity wins: error > timeout > offline
        #   (An earlier API exception or timeout is not masked by a subsequent clean offline bypass)
        all_llm = all(s.extraction_mode == "llm" for s in snaps)
        has_llm = any(s.extraction_mode == "llm" for s in snaps)
        if all_llm:
            mode = "llm"
        elif has_llm:
            mode = "mixed"
        else:
            heuristic_modes = {s.extraction_mode for s in snaps}
            if "heuristic_error" in heuristic_modes:
                mode = "heuristic_error"
            elif "heuristic_timeout" in heuristic_modes:
                mode = "heuristic_timeout"
            else:
                mode = "heuristic_offline"

        window_confidences = [
            round(boundary_conf, 2),
            round(rec_conf, 2),
            round(q_conf, 2),
            round(spec_conf, 2),
            round(future_conf, 2),
            round(agree_conf, 2),
        ]
        window_sem_conf = round(min(window_confidences), 2)

        return SemanticFeatureSnapshot(
            utterance_id=latest.utterance_id,
            call_sid=self.call_sid,
            speaker_id=latest.speaker_id,
            extraction_mode=mode,
            recurrence_id=rec_id,
            recurrence_type=rec_type,
            recurrence_count=rec_count,
            question_type=q_type,
            specificity_score=round(max_spec, 2),
            future_language_score=round(max_future, 2),
            boundary_score=round(max_boundary, 2),
            agreement_score=round(max_agree, 2),
            boundary_confidence=window_confidences[0],
            recurrence_confidence=window_confidences[1],
            question_type_confidence=window_confidences[2],
            specificity_confidence=window_confidences[3],
            future_language_confidence=window_confidences[4],
            agreement_confidence=window_confidences[5],
            semantic_confidence=window_sem_conf,
        )

    def process_turn(
        self,
        utterance: NormalizedUtterance,
        timing_snapshot: TimingFeatureSnapshot,
        semantic_snapshot: Optional[SemanticFeatureSnapshot] = None,
        new_change_points: Optional[List[ChangePointEvent]] = None,
    ) -> MultiWindowEvidenceFrame:
        self.utterances.append(utterance)
        if semantic_snapshot is not None:
            self.semantic_snapshots.append(semantic_snapshot)
        if new_change_points:
            self.change_points.extend(new_change_points)

        timestamp_ms = utterance.end_ms
        windows: Dict[str, BehavioralEvidenceSnapshot] = {}

        # 1. Current Utterance Horizon
        curr_turn_duration = max(1, utterance.end_ms - utterance.start_ms)
        curr_deviations: List[FeatureDeviation] = []
        if self.baseline_engine is not None:
            curr_deviations = self.baseline_engine.compute_deviations(
                timing_snapshot, speaker_id=utterance.speaker_id
            )

        cfg = self.confidence_config

        # Comprehensive weakest-link confidence aggregation across 5 key factors:
        # 1. Timing measurement quality (word-level timestamps vs synthetic estimation)
        t_conf = cfg.synthetic_timing_certainty if utterance.is_estimated_timing else float(timing_snapshot.timing_confidence)

        # 2. ASR audio transcription quality (per-word Deepgram confidence average)
        asr_conf = max(0.0, min(1.0, float(utterance.asr_confidence)))

        # 3. Baseline readiness (calibrating prior vs locked normative 1.0)
        is_locked = self.baseline_engine.is_locked(utterance.speaker_id) if self.baseline_engine else True
        base_conf = 1.0 if is_locked else cfg.baseline_unlocked_prior

        # 4. Speaker separation certainty (dual-track isolated 1.0 vs single-channel diarized)
        meta = utterance.metadata or {}
        if "speaker_separation_confidence" in meta:
            sep_conf = float(meta["speaker_separation_confidence"])
        elif meta.get("is_dual_track", False):
            sep_conf = 1.0
        elif not meta.get("enforce_single_speaker", True):
            sep_conf = cfg.single_channel_diarization_certainty
        else:
            sep_conf = 1.0

        # 5. Semantic extraction certainty
        sem_conf = float(semantic_snapshot.semantic_confidence) if semantic_snapshot else cfg.high_confidence_ceiling

        curr_factors: List[tuple[str, float, str]] = [
            ("timing", t_conf, "Estimated word timing without word-level timestamps"),
            ("asr", asr_conf, f"ASR audio transcription quality ({int(round(asr_conf * 100))}%)"),
            ("baseline", base_conf, f"Uncalibrated speaker baseline ({utterance.speaker_id} calibrating)"),
            ("speaker_sep", sep_conf, f"Single-channel speaker diarization uncertainty ({int(round(sep_conf * 100))}%)"),
        ]
        if semantic_snapshot is not None:
            curr_factors.append(("semantic", sem_conf, f"Heuristic semantic extraction ({semantic_snapshot.extraction_mode})"))
        if new_change_points:
            cp_min = min(float(cp.confidence) for cp in new_change_points)
            curr_factors.append(("change_point", cp_min, f"Change point detection uncertainty ({int(round(cp_min * 100))}%)"))

        weakest_name, weakest_val, weakest_reason = min(curr_factors, key=lambda f: f[1])
        curr_confidence = max(0.0, min(1.0, weakest_val))
        curr_driver = "High confidence (dual-track, verified ASR, locked baseline, LLM)" if curr_confidence >= cfg.high_confidence_ceiling else weakest_reason
        curr_breakdown = {name: round(val, 2) for name, val, _ in curr_factors}

        curr_events = [
            e.event_id
            for e in self.timing_engine.turn_events
            if e.timestamp_ms >= utterance.start_ms and e.timestamp_ms <= utterance.end_ms
        ]

        windows["current_utterance"] = BehavioralEvidenceSnapshot(
            evidence_id=f"ev_{self.call_sid}_{timestamp_ms}_curr",
            call_sid=self.call_sid,
            timestamp_ms=timestamp_ms,
            horizon="current_utterance",
            window_duration_ms=curr_turn_duration,
            timing_features=timing_snapshot,
            semantic_features=semantic_snapshot,
            deviations=curr_deviations,
            change_points=new_change_points or [],
            talk_ratio_client=1.0 if utterance.speaker_id == "client" else 0.0,
            contributing_utterance_ids=[utterance.utterance_id],
            contributing_event_ids=curr_events,
            evidence_confidence=round(curr_confidence, 2),
            confidence_driver=curr_driver,
            confidence_breakdown=curr_breakdown,
        )

        # Multi-window horizons (last_5_10s, last_20_30s, last_60_90s, full_call)
        rolling_horizons: List[tuple[WindowHorizon, int]] = [
            ("last_5_10s", 10000),
            ("last_20_30s", 30000),
            ("last_60_90s", 60000),
            ("full_call", timestamp_ms),
        ]

        for horizon, duration_ms in rolling_horizons:
            window_start_ms = max(0, timestamp_ms - duration_ms)
            window_utts = [
                u
                for u in self.utterances
                if u.end_ms >= window_start_ms and u.start_ms <= timestamp_ms
            ]
            window_utt_ids = [u.utterance_id for u in window_utts]

            window_events = [
                e.event_id
                for e in self.timing_engine.turn_events
                if e.timestamp_ms >= window_start_ms and e.timestamp_ms <= timestamp_ms
            ]

            window_cps = [
                cp
                for cp in self.change_points
                if cp.timestamp_ms >= window_start_ms and cp.timestamp_ms <= timestamp_ms
            ]

            h_timing = self.timing_engine.compute_snapshot_for_window(
                window_ms=duration_ms,
                speaker_id=utterance.speaker_id,
                at_timestamp_ms=timestamp_ms,
            )

            h_deviations: List[FeatureDeviation] = []
            if self.baseline_engine is not None:
                h_deviations = self.baseline_engine.compute_deviations(
                    h_timing, speaker_id=utterance.speaker_id
                )

            # Strict weakest-link minimum confidence calculation in window
            h_t_conf = cfg.synthetic_timing_certainty if any(u.is_estimated_timing for u in window_utts) else float(h_timing.timing_confidence)
            h_asr_conf = min((float(u.asr_confidence) for u in window_utts), default=asr_conf)
            if self.baseline_engine is not None:
                speakers_in_win = set(u.speaker_id for u in window_utts)
                h_locked = all(self.baseline_engine.is_locked(s) for s in speakers_in_win) if speakers_in_win else is_locked
                h_base_conf = 1.0 if h_locked else cfg.baseline_unlocked_prior
            else:
                h_base_conf = 1.0

            h_sep_confs = []
            for u in window_utts:
                m = u.metadata or {}
                if "speaker_separation_confidence" in m:
                    h_sep_confs.append(float(m["speaker_separation_confidence"]))
                elif m.get("is_dual_track", False):
                    h_sep_confs.append(1.0)
                elif not m.get("enforce_single_speaker", True):
                    h_sep_confs.append(cfg.single_channel_diarization_certainty)
                else:
                    h_sep_confs.append(1.0)
            h_sep_conf = min(h_sep_confs) if h_sep_confs else sep_conf

            talk_ratio = self._compute_talk_ratio(window_utts)
            h_semantic = self._aggregate_semantic_for_window(
                window_utts, fallback_snap=semantic_snapshot
            )

            h_factors: List[tuple[str, float, str]] = [
                ("timing", h_t_conf, "Estimated word timing without word-level timestamps"),
                ("asr", h_asr_conf, f"ASR audio transcription quality ({int(round(h_asr_conf * 100))}%)"),
                ("baseline", h_base_conf, "Uncalibrated speaker baseline in window"),
                ("speaker_sep", h_sep_conf, f"Single-channel speaker diarization uncertainty ({int(round(h_sep_conf * 100))}%)"),
            ]
            if h_semantic is not None:
                h_factors.append(("semantic", float(h_semantic.semantic_confidence), f"Heuristic semantic extraction ({h_semantic.extraction_mode})"))
            if window_cps:
                h_cp_min = min(float(cp.confidence) for cp in window_cps)
                h_factors.append(("change_point", h_cp_min, f"Change point detection uncertainty ({int(round(h_cp_min * 100))}%)"))

            h_w_name, h_w_val, h_w_reason = min(h_factors, key=lambda f: f[1])
            h_confidence = max(0.0, min(1.0, h_w_val))
            h_driver = "High confidence (dual-track, verified ASR, locked baseline, LLM)" if h_confidence >= cfg.high_confidence_ceiling else h_w_reason
            h_breakdown = {name: round(val, 2) for name, val, _ in h_factors}

            windows[horizon] = BehavioralEvidenceSnapshot(
                evidence_id=f"ev_{self.call_sid}_{timestamp_ms}_{horizon}",
                call_sid=self.call_sid,
                timestamp_ms=timestamp_ms,
                horizon=horizon,
                window_duration_ms=duration_ms,
                timing_features=h_timing,
                semantic_features=h_semantic,
                deviations=h_deviations,
                change_points=window_cps,
                talk_ratio_client=talk_ratio,
                contributing_utterance_ids=window_utt_ids,
                contributing_event_ids=window_events,
                evidence_confidence=round(h_confidence, 2),
                confidence_driver=h_driver,
                confidence_breakdown=h_breakdown,
            )

        frame = MultiWindowEvidenceFrame(
            frame_id=f"frm_{self.call_sid}_{timestamp_ms}",
            call_sid=self.call_sid,
            timestamp_ms=timestamp_ms,
            windows=windows,
        )

        self.latest_frame = frame
        self.store.append_frame(frame)
        return frame
