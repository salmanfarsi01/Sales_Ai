from __future__ import annotations

import os
import json
import math
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field

from .behavioral_normalization import NormalizedUtterance
from .behavioral_timing import TimingFeatureSnapshot
from .behavioral_semantic import SemanticFeatureSnapshot

LOGGER = logging.getLogger("copilot.behavioral_baseline")


class SpeakerBaselineStats(BaseModel):
    mean: float = Field(..., ge=0.0)
    stddev: float = Field(..., ge=0.0)
    sample_count: int = Field(default=0, ge=0)


class BaselineProfile(BaseModel):
    prospect_id: Optional[str] = None
    speech_rate_wpm: SpeakerBaselineStats
    avg_pause_duration_ms: SpeakerBaselineStats
    response_latency_ms: SpeakerBaselineStats
    turn_length_words: SpeakerBaselineStats
    call_count: int = Field(default=1, ge=1)
    updated_at_ms: int = Field(default=0, ge=0)


class FeatureDeviation(BaseModel):
    feature_name: str
    observed_value: float
    baseline_mean: float
    baseline_stddev: float
    z_score: float
    is_significant: bool


class ChangePointEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"cp_{uuid.uuid4().hex[:10]}")
    call_sid: str
    timestamp_ms: int
    feature_name: str
    observed_value: float
    baseline_value: float
    delta: float
    z_score: float
    direction: Literal["surge", "drop"]
    baseline_source: Literal["intra_call", "historical_only", "blended"] = "intra_call"
    semantic_context: Optional[Dict[str, Any]] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ProspectBaselineStore:
    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            base_dir = Path(__file__).resolve().parent.parent / "knowledge"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.store_path = base_dir / "prospect_baselines.json"
        else:
            self.store_path = store_path

    def _read_data(self) -> Dict[str, Any]:
        if not self.store_path.exists():
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            LOGGER.warning("Could not read prospect baseline store: %s", exc)
            return {}

    def _write_data(self, data: Dict[str, Any]) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            LOGGER.warning("Could not write prospect baseline store: %s", exc)

    def get_baseline(self, prospect_id: str) -> Optional[BaselineProfile]:
        if not prospect_id:
            return None
        data = self._read_data()
        record = data.get(prospect_id)
        if not record:
            return None
        try:
            return BaselineProfile.model_validate(record)
        except Exception as exc:
            LOGGER.warning("Malformed baseline record for %s: %s", prospect_id, exc)
            return None

    def save_baseline(self, prospect_id: str, profile: BaselineProfile) -> None:
        if not prospect_id:
            return
        data = self._read_data()
        data[prospect_id] = profile.model_dump()
        self._write_data(data)


DEFAULT_FEATURE_STDDEV_FLOORS: Dict[str, Tuple[float, float]] = {
    # feature_name: (min_stddev_ratio, min_stddev_abs)
    "speech_rate_wpm": (0.18, 15.0),        # 18% of mean, at least 15 WPM floor
    "avg_pause_duration_ms": (0.25, 150.0), # 25% of mean, at least 150ms floor
    "response_latency_ms": (0.30, 200.0),   # 30% of mean, at least 200ms floor
    "turn_length_words": (0.35, 3.0),       # 35% of mean, at least 3 words floor
}


def compute_sample_stats(
    samples: List[float],
    min_stddev_ratio: float = 0.18,
    min_stddev_abs: float = 1.0,
) -> SpeakerBaselineStats:
    n = len(samples)
    if n == 0:
        return SpeakerBaselineStats(mean=0.0, stddev=1.0, sample_count=0)
    mean = sum(samples) / n
    if n == 1:
        stddev = max(min_stddev_abs, mean * min_stddev_ratio)
    else:
        variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
        stddev = math.sqrt(variance)
        stddev = max(stddev, max(min_stddev_abs, mean * min_stddev_ratio))
    return SpeakerBaselineStats(mean=round(mean, 2), stddev=round(stddev, 2), sample_count=n)


DEFAULT_INTRA_CALL_WINDOW_MS: int = 60000
DEFAULT_MAX_CALIBRATION_WINDOW_MS: int = 90000
DEFAULT_MIN_CALIBRATION_TURNS: int = 3
DEFAULT_MIN_CUMULATIVE_WORDS: int = 15
DEFAULT_Z_THRESHOLD: float = 2.0


class BaselineAndChangePointEngine:
    def __init__(
        self,
        call_sid: str,
        prospect_id: Optional[str] = None,
        store: Optional[ProspectBaselineStore] = None,
        intra_call_window_ms: int = DEFAULT_INTRA_CALL_WINDOW_MS,
        max_calibration_window_ms: int = DEFAULT_MAX_CALIBRATION_WINDOW_MS,
        min_calibration_turns: int = DEFAULT_MIN_CALIBRATION_TURNS,
        min_cumulative_words: int = DEFAULT_MIN_CUMULATIVE_WORDS,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
    ):
        self.call_sid = call_sid
        self.prospect_id = prospect_id
        self.store = store or ProspectBaselineStore()
        self.intra_call_window_ms = intra_call_window_ms
        self.max_calibration_window_ms = max_calibration_window_ms
        self.min_calibration_turns = min_calibration_turns
        self.min_cumulative_words = min_cumulative_words
        self.z_threshold = z_threshold

        self.historical_profile: Optional[BaselineProfile] = (
            self.store.get_baseline(prospect_id) if prospect_id else None
        )
        self.prospect_samples: Dict[str, List[float]] = {
            "speech_rate_wpm": [],
            "avg_pause_duration_ms": [],
            "response_latency_ms": [],
            "turn_length_words": [],
        }
        self.intra_call_profile: Optional[BaselineProfile] = None
        self.blended_profile: Optional[BaselineProfile] = None
        self.is_intra_call_locked: bool = False
        self.change_points: List[ChangePointEvent] = []
        self.earliest_sample_ms: Optional[int] = None

    def _blend_stats(
        self,
        intra: SpeakerBaselineStats,
        cross: Optional[SpeakerBaselineStats],
        call_count: int,
    ) -> SpeakerBaselineStats:
        if cross is None or cross.sample_count == 0:
            return intra

        w_cross = min(0.60, 0.15 * max(1, call_count))
        w_intra = 1.0 - w_cross

        blended_mean = (w_intra * intra.mean) + (w_cross * cross.mean)
        blended_var = (w_intra * (intra.stddev ** 2)) + (w_cross * (cross.stddev ** 2))
        blended_stddev = math.sqrt(max(0.1, blended_var))
        sample_total = intra.sample_count + cross.sample_count

        return SpeakerBaselineStats(
            mean=round(blended_mean, 2),
            stddev=round(blended_stddev, 2),
            sample_count=sample_total,
        )

    def _recalculate_profiles(self, timestamp_ms: int) -> None:
        speech_stats = compute_sample_stats(
            self.prospect_samples["speech_rate_wpm"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["speech_rate_wpm"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["speech_rate_wpm"][1],
        )
        pause_stats = compute_sample_stats(
            self.prospect_samples["avg_pause_duration_ms"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["avg_pause_duration_ms"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["avg_pause_duration_ms"][1],
        )
        latency_stats = compute_sample_stats(
            self.prospect_samples["response_latency_ms"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["response_latency_ms"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["response_latency_ms"][1],
        )
        turn_stats = compute_sample_stats(
            self.prospect_samples["turn_length_words"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["turn_length_words"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["turn_length_words"][1],
        )

        self.intra_call_profile = BaselineProfile(
            prospect_id=self.prospect_id,
            speech_rate_wpm=speech_stats,
            avg_pause_duration_ms=pause_stats,
            response_latency_ms=latency_stats,
            turn_length_words=turn_stats,
            call_count=1,
            updated_at_ms=timestamp_ms,
        )

        if self.historical_profile is not None:
            calls = self.historical_profile.call_count
            self.blended_profile = BaselineProfile(
                prospect_id=self.prospect_id,
                speech_rate_wpm=self._blend_stats(
                    speech_stats, self.historical_profile.speech_rate_wpm, calls
                ),
                avg_pause_duration_ms=self._blend_stats(
                    pause_stats, self.historical_profile.avg_pause_duration_ms, calls
                ),
                response_latency_ms=self._blend_stats(
                    latency_stats, self.historical_profile.response_latency_ms, calls
                ),
                turn_length_words=self._blend_stats(
                    turn_stats, self.historical_profile.turn_length_words, calls
                ),
                call_count=calls + 1,
                updated_at_ms=timestamp_ms,
            )
        else:
            self.blended_profile = self.intra_call_profile

    def _lock_intra_call_baseline(self, timestamp_ms: int) -> None:
        self._recalculate_profiles(timestamp_ms)
        self.is_intra_call_locked = True

    def get_active_profile_and_source(
        self,
    ) -> Tuple[Optional[BaselineProfile], Optional[Literal["intra_call", "historical_only", "blended"]]]:
        if self.is_intra_call_locked:
            if self.historical_profile is not None:
                return self.blended_profile, "blended"
            return self.intra_call_profile or self.blended_profile, "intra_call"
        if self.historical_profile is not None:
            return self.historical_profile, "historical_only"
        return None, None

    def get_active_profile(self) -> Optional[BaselineProfile]:
        profile, _ = self.get_active_profile_and_source()
        return profile

    def update_with_utterance(
        self,
        utterance: NormalizedUtterance,
        timing_snapshot: TimingFeatureSnapshot,
        semantic_snapshot: Optional[SemanticFeatureSnapshot] = None,
    ) -> List[ChangePointEvent]:
        if utterance.speaker_id != "client":
            return []

        timestamp_ms = utterance.end_ms
        if self.earliest_sample_ms is None:
            self.earliest_sample_ms = utterance.start_ms

        observed_features: Dict[str, float] = {
            "speech_rate_wpm": float(timing_snapshot.speech_rate_wpm),
            "avg_pause_duration_ms": float(timing_snapshot.avg_pause_duration_ms),
            "turn_length_words": float(timing_snapshot.turn_length_words),
        }
        if timing_snapshot.response_latency_ms is not None:
            observed_features["response_latency_ms"] = float(timing_snapshot.response_latency_ms)

        is_clean_speech = utterance.asr_confidence >= 0.70

        just_locked = False
        if not self.is_intra_call_locked:
            if is_clean_speech:
                for feat, val in observed_features.items():
                    if feat == "speech_rate_wpm" and val <= 0.0:
                        continue
                    self.prospect_samples[feat].append(val)

            elapsed_ms = timestamp_ms - self.earliest_sample_ms
            turn_count = len(self.prospect_samples["turn_length_words"])
            total_words = sum(self.prospect_samples["turn_length_words"])

            sufficient_speech = turn_count >= self.min_calibration_turns and total_words >= self.min_cumulative_words
            if (elapsed_ms >= self.intra_call_window_ms and sufficient_speech) or (
                elapsed_ms >= self.max_calibration_window_ms and turn_count >= self.min_calibration_turns
            ):
                self._lock_intra_call_baseline(timestamp_ms)
                just_locked = True

        active_profile, baseline_source = self.get_active_profile_and_source()
        if active_profile is None or baseline_source is None:
            return []

        new_change_points: List[ChangePointEvent] = []
        is_anomalous_turn = False
        for feat, val in observed_features.items():
            if feat == "speech_rate_wpm":
                if val <= 0.0:
                    continue
                # Turns with fewer than 4 words have excessive timestamp quantization variance
                # and should not trigger speech rate change-points on short confirmations
                turn_words = observed_features.get("turn_length_words", 0.0)
                if turn_words < 4.0:
                    continue

            stat: SpeakerBaselineStats = getattr(active_profile, feat)
            if stat.stddev <= 0.001:
                continue

            z_score = (val - stat.mean) / stat.stddev
            if abs(z_score) >= self.z_threshold:
                is_anomalous_turn = True
                direction = "surge" if z_score > 0 else "drop"
                delta = round(val - stat.mean, 2)

                context_dict: Dict[str, Any] = {}
                if semantic_snapshot is not None:
                    context_dict = {
                        "question_type": semantic_snapshot.question_type,
                        "recurrence_type": semantic_snapshot.recurrence_type,
                        "boundary_score": semantic_snapshot.boundary_score,
                        "agreement_score": semantic_snapshot.agreement_score,
                    }

                event = ChangePointEvent(
                    event_id=f"cp_{self.call_sid}_{timestamp_ms}_{feat}",
                    call_sid=self.call_sid,
                    timestamp_ms=timestamp_ms,
                    feature_name=feat,
                    observed_value=round(val, 2),
                    baseline_value=stat.mean,
                    delta=delta,
                    z_score=round(z_score, 2),
                    direction=direction,
                    baseline_source=baseline_source,
                    confidence=float(timing_snapshot.timing_confidence),
                    semantic_context=context_dict or None,
                )
                new_change_points.append(event)
                self.change_points.append(event)

        if self.is_intra_call_locked and not just_locked and not is_anomalous_turn and is_clean_speech:
            for feat, val in observed_features.items():
                if feat == "speech_rate_wpm" and val <= 0.0:
                    continue
                self.prospect_samples[feat].append(val)
            self._recalculate_profiles(timestamp_ms)

        return new_change_points

    def compute_deviations(
        self, timing_snapshot: TimingFeatureSnapshot
    ) -> List[FeatureDeviation]:
        active_profile = self.get_active_profile()
        if active_profile is None:
            return []

        observed_features: Dict[str, float] = {
            "speech_rate_wpm": float(timing_snapshot.speech_rate_wpm),
            "avg_pause_duration_ms": float(timing_snapshot.avg_pause_duration_ms),
            "turn_length_words": float(timing_snapshot.turn_length_words),
        }
        if timing_snapshot.response_latency_ms is not None:
            observed_features["response_latency_ms"] = float(timing_snapshot.response_latency_ms)

        deviations: List[FeatureDeviation] = []
        for feat, val in observed_features.items():
            stat: SpeakerBaselineStats = getattr(active_profile, feat)
            z = (val - stat.mean) / stat.stddev if stat.stddev > 0.001 else 0.0
            deviations.append(
                FeatureDeviation(
                    feature_name=feat,
                    observed_value=round(val, 2),
                    baseline_mean=stat.mean,
                    baseline_stddev=stat.stddev,
                    z_score=round(z, 2),
                    is_significant=abs(z) >= self.z_threshold,
                )
            )

        return deviations

    def finalize_and_persist(self) -> Optional[BaselineProfile]:
        turn_count = len(self.prospect_samples["turn_length_words"])
        if not self.is_intra_call_locked:
            if turn_count >= self.min_calibration_turns:
                self._lock_intra_call_baseline(timestamp_ms=0)
            else:
                return None

        final_profile = self.get_active_profile()
        if final_profile and self.prospect_id:
            self.store.save_baseline(self.prospect_id, final_profile)
        return final_profile
