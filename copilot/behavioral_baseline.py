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
    speaker_id: Optional[str] = None
    speaker_role: Optional[Literal["salesperson", "client"]] = "client"
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
    delta_percent: Optional[float] = Field(
        default=None,
        description="Percentage change from baseline mean: ((observed - mean) / mean) * 100",
    )
    is_significant: bool
    is_measured: bool = Field(
        default=True,
        description="True if the observed feature represents an actual measurement rather than absence of measurable data",
    )
    baseline_source: Optional[Literal["intra_call", "historical_only", "blended"]] = Field(
        default=None,
        description="Baseline distribution source against which this deviation was computed",
    )


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


class ContactPreferenceRecord(BaseModel):
    preference: Literal["none", "reduced_frequency", "channel_restriction", "timing_restriction"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: Optional[str] = None
    first_observed_turn: Optional[int] = None
    call_sid: Optional[str] = None
    timestamp_ms: Optional[int] = None
    # Client Feedback Issue #7 Wishlist Schema Fields (Cross-layer parity with ContactPreference)
    channel: Optional[Literal["sms", "call", "email"]] = None
    allowed: bool = True
    cadence: Optional[Literal["reduced", "specific_times", "no_preference"]] = None
    prohibited_behavior: Optional[str] = None
    boundary_strength: Literal["preference", "hard_restriction"] = "preference"


class ContactPreferenceStore:
    """DEPRECATED: Legacy unpartitioned prospect preference store.
    
    Superseded by copilot.prospect_memory.ProspectMemoryStore which enforces
    mandatory user_id partitioning and Point 15-18 isolation invariants.
    Retained solely for test fixture compatibility.
    """
    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            base_dir = Path(__file__).resolve().parent.parent / "knowledge"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.store_path = base_dir / "prospect_preferences.json"
            LOGGER.warning(
                "ContactPreferenceStore is deprecated and unpartitioned (Point 18). "
                "Use ProspectMemoryStore with mandatory user_id."
            )
        else:
            self.store_path = store_path

    def _read_data(self) -> Dict[str, Any]:
        if not self.store_path.exists():
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get("_STATUS") == "QUARANTINED_DEPRECATED":
                    return {}
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            LOGGER.warning("Could not read contact preference store: %s", exc)
            return {}

    def _write_data(self, data: Dict[str, Any]) -> None:
        if self.store_path.name == "prospect_preferences.json":
            LOGGER.warning("Refusing to write to quarantined legacy prospect_preferences.json")
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            LOGGER.warning("Could not write contact preference store: %s", exc)

    def get_preference(self, prospect_id: str) -> Optional[ContactPreferenceRecord]:
        if not prospect_id:
            return None
        data = self._read_data()
        rec = data.get(prospect_id)
        if not rec:
            return None
        try:
            return ContactPreferenceRecord.model_validate(rec)
        except Exception as exc:
            LOGGER.warning("Malformed preference record for %s: %s", prospect_id, exc)
            return None

    def save_preference(self, prospect_id: str, record: ContactPreferenceRecord) -> None:
        if not prospect_id:
            return
        data = self._read_data()
        data[prospect_id] = record.model_dump()
        self._write_data(data)


DEFAULT_FEATURE_STDDEV_FLOORS: Dict[str, Tuple[float, float]] = {
    # feature_name: (min_stddev_ratio, min_stddev_abs)
    #
    # Rationale for per-feature variance floor ratio differentiation:
    # 1. speech_rate_wpm (18%, floor 15.0 WPM):
    #    Speech rate has relatively tight physiological bounds in fluent adult speech (~10-15% CV).
    #    An 18% floor (e.g. +/-32.4 WPM at 90 WPM) provides sufficient breathing room for natural
    #    sentence-to-sentence cadence swings (75-115 WPM) without false-triggering |z| >= 2.0.
    # 2. avg_pause_duration_ms (25%, floor 150.0 ms):
    #    Intra-turn pauses are sensitive to ASR word-boundary alignment jitter (+/-50-100ms) and cognitive
    #    micro-hesitations, demanding a wider 25% relative margin and a 150ms absolute floor.
    # 3. response_latency_ms (30%, floor 200.0 ms):
    #    Turn-taking latency is inherently noisy and context-dependent (rapid backchannels vs. thoughtful
    #    deliberation), requiring a 30% margin and 200ms floor to prevent false hesitation alerts.
    # 4. turn_length_words (35%, floor 3.0 words):
    #    Turn length exhibits the highest natural dispersion in spontaneous dialogue (ranging from brief
    #    acknowledgments to multi-sentence disclosures), requiring the widest relative floor (35%).
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
        agent_id: Optional[str] = None,
        store: Optional[ProspectBaselineStore] = None,
        agent_store: Optional[ProspectBaselineStore] = None,
        intra_call_window_ms: int = DEFAULT_INTRA_CALL_WINDOW_MS,
        max_calibration_window_ms: int = DEFAULT_MAX_CALIBRATION_WINDOW_MS,
        min_calibration_turns: int = DEFAULT_MIN_CALIBRATION_TURNS,
        min_cumulative_words: int = DEFAULT_MIN_CUMULATIVE_WORDS,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
    ):
        self.call_sid = call_sid
        self.prospect_id = prospect_id
        self.agent_id = agent_id
        self.store = store or ProspectBaselineStore()
        self.agent_store = agent_store or self.store
        self.intra_call_window_ms = intra_call_window_ms
        self.max_calibration_window_ms = max_calibration_window_ms
        self.min_calibration_turns = min_calibration_turns
        self.min_cumulative_words = min_cumulative_words
        self.z_threshold = z_threshold

        self.historical_profile: Optional[BaselineProfile] = (
            self.store.get_baseline(prospect_id) if prospect_id else None
        )
        self.agent_historical_profile: Optional[BaselineProfile] = (
            self.agent_store.get_baseline(agent_id) if agent_id else None
        )

        # Strictly segregated samples per individual speaker
        self.speaker_samples: Dict[str, Dict[str, List[float]]] = {
            "client": {
                "speech_rate_wpm": [],
                "avg_pause_duration_ms": [],
                "response_latency_ms": [],
                "turn_length_words": [],
            },
            "salesperson": {
                "speech_rate_wpm": [],
                "avg_pause_duration_ms": [],
                "response_latency_ms": [],
                "turn_length_words": [],
            },
        }
        # Direct aliases for clean access
        self.prospect_samples = self.speaker_samples["client"]
        self.agent_samples = self.speaker_samples["salesperson"]

        self.intra_call_profiles: Dict[str, Optional[BaselineProfile]] = {
            "client": None,
            "salesperson": None,
        }
        self.blended_profiles: Dict[str, Optional[BaselineProfile]] = {
            "client": None,
            "salesperson": None,
        }
        self.is_locked_by_speaker: Dict[str, bool] = {
            "client": False,
            "salesperson": False,
        }
        self.earliest_sample_ms_by_speaker: Dict[str, Optional[int]] = {
            "client": None,
            "salesperson": None,
        }
        self.change_points: List[ChangePointEvent] = []

    # Backwards-compatibility properties pointing to prospect ("client")
    @property
    def is_intra_call_locked(self) -> bool:
        return self.is_locked_by_speaker["client"]

    @is_intra_call_locked.setter
    def is_intra_call_locked(self, val: bool) -> None:
        self.is_locked_by_speaker["client"] = val

    @property
    def intra_call_profile(self) -> Optional[BaselineProfile]:
        return self.intra_call_profiles["client"]

    @intra_call_profile.setter
    def intra_call_profile(self, val: Optional[BaselineProfile]) -> None:
        self.intra_call_profiles["client"] = val

    @property
    def blended_profile(self) -> Optional[BaselineProfile]:
        return self.blended_profiles["client"]

    @blended_profile.setter
    def blended_profile(self, val: Optional[BaselineProfile]) -> None:
        self.blended_profiles["client"] = val

    @property
    def earliest_sample_ms(self) -> Optional[int]:
        return self.earliest_sample_ms_by_speaker["client"]

    @earliest_sample_ms.setter
    def earliest_sample_ms(self, val: Optional[int]) -> None:
        self.earliest_sample_ms_by_speaker["client"] = val

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

    def _recalculate_profiles(self, timestamp_ms: int, speaker_id: Literal["client", "salesperson"] = "client") -> None:
        samples = self.speaker_samples[speaker_id]
        speech_stats = compute_sample_stats(
            samples["speech_rate_wpm"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["speech_rate_wpm"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["speech_rate_wpm"][1],
        )
        pause_stats = compute_sample_stats(
            samples["avg_pause_duration_ms"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["avg_pause_duration_ms"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["avg_pause_duration_ms"][1],
        )
        latency_stats = compute_sample_stats(
            samples["response_latency_ms"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["response_latency_ms"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["response_latency_ms"][1],
        )
        turn_stats = compute_sample_stats(
            samples["turn_length_words"],
            min_stddev_ratio=DEFAULT_FEATURE_STDDEV_FLOORS["turn_length_words"][0],
            min_stddev_abs=DEFAULT_FEATURE_STDDEV_FLOORS["turn_length_words"][1],
        )

        role_id = self.prospect_id if speaker_id == "client" else self.agent_id
        profile = BaselineProfile(
            prospect_id=self.prospect_id if speaker_id == "client" else None,
            speaker_id=role_id,
            speaker_role=speaker_id,
            speech_rate_wpm=speech_stats,
            avg_pause_duration_ms=pause_stats,
            response_latency_ms=latency_stats,
            turn_length_words=turn_stats,
            call_count=1,
            updated_at_ms=timestamp_ms,
        )
        self.intra_call_profiles[speaker_id] = profile

        hist = self.historical_profile if speaker_id == "client" else self.agent_historical_profile
        if hist is not None:
            calls = hist.call_count
            self.blended_profiles[speaker_id] = BaselineProfile(
                prospect_id=self.prospect_id if speaker_id == "client" else None,
                speaker_id=role_id,
                speaker_role=speaker_id,
                speech_rate_wpm=self._blend_stats(
                    speech_stats, hist.speech_rate_wpm, calls
                ),
                avg_pause_duration_ms=self._blend_stats(
                    pause_stats, hist.avg_pause_duration_ms, calls
                ),
                response_latency_ms=self._blend_stats(
                    latency_stats, hist.response_latency_ms, calls
                ),
                turn_length_words=self._blend_stats(
                    turn_stats, hist.turn_length_words, calls
                ),
                call_count=calls + 1,
                updated_at_ms=timestamp_ms,
            )
        else:
            self.blended_profiles[speaker_id] = profile

    def _lock_intra_call_baseline(self, timestamp_ms: int, speaker_id: Literal["client", "salesperson"] = "client") -> None:
        self._recalculate_profiles(timestamp_ms, speaker_id=speaker_id)
        self.is_locked_by_speaker[speaker_id] = True

    def is_locked(self, speaker_id: Literal["client", "salesperson"] = "client") -> bool:
        return self.is_locked_by_speaker.get(speaker_id, False)

    def get_active_profile_and_source(
        self,
        speaker_id: Literal["client", "salesperson"] = "client",
    ) -> Tuple[Optional[BaselineProfile], Optional[Literal["intra_call", "historical_only", "blended"]]]:
        hist = self.historical_profile if speaker_id == "client" else self.agent_historical_profile
        if self.is_locked_by_speaker.get(speaker_id, False):
            if hist is not None:
                return self.blended_profiles[speaker_id], "blended"
            return self.intra_call_profiles[speaker_id] or self.blended_profiles[speaker_id], "intra_call"
        if hist is not None:
            return hist, "historical_only"
        return None, None

    def get_active_profile(
        self,
        speaker_id: Literal["client", "salesperson"] = "client",
    ) -> Optional[BaselineProfile]:
        profile, _ = self.get_active_profile_and_source(speaker_id)
        return profile

    def update_with_utterance(
        self,
        utterance: NormalizedUtterance,
        timing_snapshot: TimingFeatureSnapshot,
        semantic_snapshot: Optional[SemanticFeatureSnapshot] = None,
    ) -> List[ChangePointEvent]:
        speaker_id = utterance.speaker_id
        if speaker_id not in ("client", "salesperson"):
            return []

        timestamp_ms = utterance.end_ms
        if self.earliest_sample_ms_by_speaker[speaker_id] is None:
            self.earliest_sample_ms_by_speaker[speaker_id] = utterance.start_ms

        observed_features: Dict[str, float] = {
            "speech_rate_wpm": float(timing_snapshot.speech_rate_wpm),
            "avg_pause_duration_ms": float(timing_snapshot.avg_pause_duration_ms),
            "turn_length_words": float(timing_snapshot.turn_length_words),
        }
        if timing_snapshot.response_latency_ms is not None:
            observed_features["response_latency_ms"] = float(timing_snapshot.response_latency_ms)

        is_clean_speech = utterance.asr_confidence >= 0.70

        just_locked = False
        samples = self.speaker_samples[speaker_id]
        if not self.is_locked_by_speaker[speaker_id]:
            if is_clean_speech:
                turn_words = observed_features.get("turn_length_words", 0.0)
                for feat, val in observed_features.items():
                    # Skip rate calculation on sub-4-word utterances due to denominator quantization noise
                    if feat == "speech_rate_wpm" and (val <= 0.0 or turn_words < 4.0):
                        continue
                    # Skip pause duration when no intra-turn pauses occurred (absence of pause, not 0ms duration)
                    if feat == "avg_pause_duration_ms" and val <= 0.0:
                        continue
                    # Skip turn length on sub-2-word fragments to prevent 1-word breath tokens from deflating baseline mean
                    if feat == "turn_length_words" and val < 2.0:
                        continue
                    # Skip negative response latencies (speech overlaps) from silence latency baseline
                    if feat == "response_latency_ms" and val < 0.0:
                        continue
                    samples[feat].append(val)

            elapsed_ms = timestamp_ms - (self.earliest_sample_ms_by_speaker[speaker_id] or 0)
            turn_count = len(samples["turn_length_words"])
            total_words = sum(samples["turn_length_words"])

            sufficient_speech = (
                turn_count >= self.min_calibration_turns
                and total_words >= self.min_cumulative_words
            )
            # Baseline sufficiency gate:
            # Enforce elapsed time, turn count, and cumulative words per individual speaker
            if elapsed_ms >= self.intra_call_window_ms and sufficient_speech:
                self._lock_intra_call_baseline(timestamp_ms, speaker_id=speaker_id)
                just_locked = True

        active_profile, baseline_source = self.get_active_profile_and_source(speaker_id)
        if active_profile is None or baseline_source is None:
            return []

        new_change_points: List[ChangePointEvent] = []
        is_anomalous_turn = False
        for feat, val in observed_features.items():
            if feat == "speech_rate_wpm":
                if val <= 0.0:
                    continue
                turn_words = observed_features.get("turn_length_words", 0.0)
                if turn_words < 4.0:
                    continue

            if feat == "avg_pause_duration_ms":
                if val <= 0.0:
                    continue
                turn_words = observed_features.get("turn_length_words", 0.0)
                if turn_words < 4.0:
                    continue

            if feat == "turn_length_words":
                if val < 4.0:
                    stat: SpeakerBaselineStats = getattr(active_profile, feat)
                    if val < stat.mean:
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
                    event_id=f"cp_{self.call_sid}_{speaker_id}_{timestamp_ms}_{feat}",
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

        if self.is_locked_by_speaker[speaker_id] and not just_locked and not is_anomalous_turn and is_clean_speech:
            turn_words = observed_features.get("turn_length_words", 0.0)
            for feat, val in observed_features.items():
                if feat == "speech_rate_wpm" and (val <= 0.0 or turn_words < 4.0):
                    continue
                if feat == "avg_pause_duration_ms" and val <= 0.0:
                    continue
                if feat == "turn_length_words" and val < 2.0:
                    continue
                if feat == "response_latency_ms" and val < 0.0:
                    continue
                samples[feat].append(val)
            self._recalculate_profiles(timestamp_ms, speaker_id=speaker_id)

        return new_change_points

    def compute_deviations(
        self,
        timing_snapshot: TimingFeatureSnapshot,
        speaker_id: Optional[Literal["client", "salesperson"]] = None,
    ) -> List[FeatureDeviation]:
        target_speaker: Literal["client", "salesperson"] = (
            speaker_id or getattr(timing_snapshot, "speaker_id", None) or "client"
        )
        active_profile, baseline_source = self.get_active_profile_and_source(target_speaker)
        if active_profile is None or baseline_source is None:
            return []

        observed_features: Dict[str, float] = {
            "speech_rate_wpm": float(timing_snapshot.speech_rate_wpm),
            "avg_pause_duration_ms": float(timing_snapshot.avg_pause_duration_ms),
            "turn_length_words": float(timing_snapshot.turn_length_words),
        }
        if timing_snapshot.response_latency_ms is not None:
            observed_features["response_latency_ms"] = float(timing_snapshot.response_latency_ms)

        deviations: List[FeatureDeviation] = []
        turn_words = observed_features.get("turn_length_words", 0.0)
        for feat, val in observed_features.items():
            stat: SpeakerBaselineStats = getattr(active_profile, feat)

            if feat == "avg_pause_duration_ms" and val <= 0.0:
                z = 0.0
                is_sig = False
                is_meas = False
            elif feat == "speech_rate_wpm" and (val <= 0.0 or turn_words < 4.0):
                z = 0.0
                is_sig = False
                is_meas = False
            elif feat == "turn_length_words" and val < 4.0 and val < stat.mean:
                z = (val - stat.mean) / stat.stddev if stat.stddev > 0.001 else 0.0
                is_sig = False
                is_meas = True
            else:
                z = (val - stat.mean) / stat.stddev if stat.stddev > 0.001 else 0.0
                is_sig = abs(z) >= self.z_threshold
                is_meas = True

            delta_pct = (
                round(((val - stat.mean) / stat.mean) * 100.0, 1)
                if (is_meas and stat.mean > 0.001)
                else None
            )

            deviations.append(
                FeatureDeviation(
                    feature_name=feat,
                    observed_value=round(val, 2),
                    baseline_mean=stat.mean,
                    baseline_stddev=stat.stddev,
                    z_score=round(z, 2),
                    delta_percent=delta_pct,
                    is_significant=is_sig,
                    is_measured=is_meas,
                    baseline_source=baseline_source,
                )
            )

        return deviations

    def finalize_and_persist(
        self,
        speaker_id: Literal["client", "salesperson"] = "client",
    ) -> Optional[BaselineProfile]:
        samples = self.speaker_samples[speaker_id]
        turn_count = len(samples["turn_length_words"])
        total_words = sum(samples["turn_length_words"])
        sufficient_speech = (
            turn_count >= self.min_calibration_turns
            and total_words >= self.min_cumulative_words
        )
        if not self.is_locked_by_speaker[speaker_id]:
            if sufficient_speech:
                self._lock_intra_call_baseline(timestamp_ms=0, speaker_id=speaker_id)
            else:
                return None

        final_profile = self.get_active_profile(speaker_id)
        if final_profile:
            if speaker_id == "client" and self.prospect_id:
                self.store.save_baseline(self.prospect_id, final_profile)
            elif speaker_id == "salesperson" and self.agent_id:
                self.agent_store.save_baseline(self.agent_id, final_profile)
        return final_profile

