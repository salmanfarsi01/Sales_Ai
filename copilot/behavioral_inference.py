from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .behavioral_evidence import (
    BehavioralEvidenceSnapshot,
    MultiWindowEvidenceFrame,
    WindowHorizon,
)


class DimensionScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    primary_horizon: WindowHorizon
    contributing_evidence_ids: List[str] = Field(default_factory=list)
    drivers: List[str] = Field(default_factory=list)


class EmotionState(BaseModel):
    expressed_valence: float = Field(0.0, ge=-1.0, le=1.0)
    tension_level: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    tone_claim_made: bool = Field(False)
    observable_signals: List[str] = Field(default_factory=list)
    contributing_evidence_ids: List[str] = Field(default_factory=list)


class DownstreamInferenceState(BaseModel):
    inference_id: str = Field(default_factory=lambda: f"inf_{uuid.uuid4().hex[:12]}")
    call_sid: str
    timestamp_ms: int
    state_version: int = Field(1, ge=1)
    inference_latency_ms: float = Field(0.0, ge=0.0)
    trust: DimensionScore
    emotion: EmotionState
    pacing: DimensionScore
    engagement: DimensionScore
    momentum: DimensionScore
    readiness: DimensionScore
    acoustic_evidence: Literal["unavailable", "available"] = "unavailable"
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    contributing_evidence_ids: List[str] = Field(default_factory=list)
    baseline_sources: List[str] = Field(default_factory=list)


class InferenceScoringConfig(BaseModel):
    """Named, versioned, and documented calibration configuration for downstream inference.

    All weights, multipliers, and thresholds are v1 provisional estimates established
    for baseline behavioral scoring, intended to be calibrated against empirical call
    conversion outcomes.
    """
    version: str = "1.0.0"

    # Pacing calibration weights
    pacing_base_score: float = Field(0.80, description="Baseline pacing score assuming natural cadence until deviations occur")
    pacing_baseline_aligned_boost: float = Field(0.10, description="Reward for speech rate remaining within |z| < 1.0 of baseline")
    pacing_moderate_departure_penalty: float = Field(0.15, description="Penalty for moderate speech rate departure 1.0 <= |z| < 2.0")
    pacing_significant_departure_penalty: float = Field(0.35, description="Penalty for statistically significant departure |z| >= 2.0")
    pacing_prolonged_latency_penalty: float = Field(0.20, description="Penalty for response latency > 2000ms indicating conversational drag")
    pacing_rushed_turnaround_penalty: float = Field(0.10, description="Penalty for response latency < 150ms suggesting speaker steamrolling")
    pacing_interruption_penalty: float = Field(0.15, description="Penalty per cross-speaker interruption in 60s window (max 3 counted)")
    pacing_score_min: float = Field(0.05, description="Lower bound for pacing score")
    pacing_score_max: float = Field(1.0, description="Upper bound for pacing score")

    # Emotion / Tension calibration weights
    emotion_base_tension: float = Field(0.20, description="Baseline ambient conversational tension in standard discovery")
    emotion_base_valence: float = Field(0.0, description="Neutral starting valence")
    emotion_interruption_tension_boost: float = Field(0.25, description="Tension elevation per cross-speaker interruption")
    emotion_latency_tension_boost: float = Field(0.20, description="Tension elevation for hesitant latency > 1800ms")
    emotion_rate_drop_tension_boost: float = Field(0.20, description="Tension elevation for sudden speech rate deceleration change point")
    emotion_boundary_tension_boost: float = Field(0.40, description="Tension elevation when hard stop-contact boundary is encountered")
    emotion_boundary_valence_penalty: float = Field(0.60, description="Negative valence impact of stop-contact boundary statement")
    emotion_recurrence_tension_boost: float = Field(0.25, description="Tension elevation when objections or concerns repeat without resolution")
    emotion_recurrence_valence_penalty: float = Field(0.35, description="Negative valence impact of unresolved recurrent objections")
    emotion_failed_reframe_tension_boost: float = Field(0.15, description="Additional tension boost when concern repeats after failed rep reframe")
    emotion_failed_reframe_valence_penalty: float = Field(0.15, description="Additional valence penalty when concern repeats after failed rep reframe")
    emotion_substantive_agreement_valence_boost: float = Field(0.50, description="Positive valence elevation for substantive agreement >= 0.70")
    emotion_substantive_agreement_tension_relief: float = Field(0.20, description="Tension relief when substantive agreement is achieved")
    emotion_polite_agreement_valence_boost: float = Field(0.15, description="Modest positive valence elevation for polite acknowledgments")

    # Engagement calibration weights
    engagement_base_score: float = Field(0.50, description="Neutral midpoint for engagement before evaluating participation balance")
    engagement_balanced_dialogue_boost: float = Field(0.25, description="Boost for balanced prospect dialogue (talk ratio in 35-65% sweet spot)")
    engagement_low_participation_penalty: float = Field(0.25, description="Penalty for passive or monosyllabic prospect (talk ratio < 15%)")
    engagement_active_share_boost: float = Field(0.10, description="Positive contribution for active prospect vocal share outside extremes")
    engagement_high_inquiry_boost: float = Field(0.20, description="Boost for high inquiry velocity (question rate >= 1.5/min)")
    engagement_steady_inquiry_boost: float = Field(0.10, description="Boost for steady inquiry rate (question rate >= 0.5/min)")
    engagement_specificity_boost: float = Field(0.15, description="Boost for high detail/fact specificity (score >= 0.50)")

    # Trust calibration weights
    trust_base_score: float = Field(0.60, description="Professional rapport baseline assuming provisional good faith")
    trust_boundary_penalty: float = Field(0.45, description="Heavy trust penalty when stop-contact or representation boundary is hit")
    trust_positive_echo_boost: float = Field(0.20, description="Trust reinforcement when prospect adopts rep's strategic terminology")
    trust_unresolved_objection_penalty: float = Field(0.20, description="Erosion of trust when core objection recurs unresolved")
    trust_substantive_agreement_boost: float = Field(0.15, description="Trust boost from substantive agreement on value propositions")
    trust_sustained_disclosure_boost: float = Field(0.10, description="Boost for sustained disclosure depth (turn length >= 12 words)")
    trust_constrained_response_penalty: float = Field(0.10, description="Penalty for constrained monosyllabic responses (turn length <= 3 words)")

    # Momentum calibration weights
    momentum_base_score: float = Field(0.55, description="Baseline forward momentum assuming ongoing conversation")
    momentum_low_friction_boost: float = Field(0.15, description="Forward momentum boost when conversational friction is minimal (tension <= 0.20)")
    momentum_high_friction_penalty: float = Field(0.30, description="Momentum deceleration penalty when tension is elevated (tension >= 0.60)")
    momentum_transactional_question_boost: float = Field(0.20, description="Forward momentum acceleration for process/timeline next-step questions")
    momentum_hostile_question_penalty: float = Field(0.35, description="Momentum stall penalty for combative or hostile challenges")

    # Readiness calibration weights
    readiness_base_score: float = Field(0.30, description="Conservative closing readiness baseline; commitment is earned, not assumed")
    readiness_scheduling_recurrence_boost: float = Field(0.35, description="Major readiness boost when prospect repeats concrete milestone dates/times")
    readiness_future_language_boost: float = Field(0.25, description="Readiness boost for operational future commitment language (score >= 0.60)")
    readiness_agreement_boost: float = Field(0.20, description="Readiness boost for substantive value agreement (score >= 0.70)")
    readiness_unresolved_objection_penalty: float = Field(0.20, description="Readiness block when unresolved core objections repeat")
    readiness_trust_foundation_boost: float = Field(0.10, description="Readiness boost when supported by strong trust foundation (trust >= 0.75)")
    readiness_hard_boundary_override_score: float = Field(0.0, description="Deterministic zero override score when stop-contact boundary occurs")

    # Bounding ranges
    score_floor_active: float = Field(0.05, description="Active non-zero floor for continuous dimensions (pacing, engagement, trust, momentum)")
    score_ceiling: float = Field(1.0, description="Standard maximum ceiling for all dimensions")
    readiness_floor: float = Field(0.0, description="Absolute zero floor for readiness")
    valence_min: float = Field(-1.0, description="Lower bound for emotional valence")
    valence_max: float = Field(1.0, description="Upper bound for emotional valence")
    tension_min: float = Field(0.0, description="Lower bound for conversational tension")
    tension_max: float = Field(1.0, description="Upper bound for conversational tension")


DEFAULT_INFERENCE_CONFIG = InferenceScoringConfig()


class AcousticEvidenceProvider(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def extract_acoustic_features(self, audio_chunk: bytes) -> Optional[Dict[str, Any]]:
        pass


class NullAcousticProvider(AcousticEvidenceProvider):
    def is_available(self) -> bool:
        return False

    def extract_acoustic_features(self, audio_chunk: bytes) -> Optional[Dict[str, Any]]:
        return None


class DownstreamInferenceEngine:
    def __init__(
        self,
        acoustic_provider: Optional[AcousticEvidenceProvider] = None,
        config: Optional[InferenceScoringConfig] = None,
    ):
        self.acoustic_provider = acoustic_provider or NullAcousticProvider()
        self.config = config or DEFAULT_INFERENCE_CONFIG
        self._call_state_versions: Dict[str, int] = {}

    def compute_inference(
        self,
        call_sid: str,
        current_frame: MultiWindowEvidenceFrame,
        recent_frames: Optional[List[MultiWindowEvidenceFrame]] = None,
    ) -> DownstreamInferenceState:
        start_t = time.monotonic()
        version = self._call_state_versions.get(call_sid, 0) + 1
        self._call_state_versions[call_sid] = version
        frames = (recent_frames or []) + [current_frame]

        any_window = next(iter(current_frame.windows.values())) if current_frame.windows else None
        w_curr = current_frame.windows.get("current_utterance") or any_window
        w_10s = current_frame.windows.get("last_5_10s") or w_curr
        w_30s = current_frame.windows.get("last_20_30s") or w_10s
        w_60s = current_frame.windows.get("last_60_90s") or w_30s
        w_full = current_frame.windows.get("full_call") or w_60s

        if w_curr is None:
            raise ValueError(f"MultiWindowEvidenceFrame {current_frame.frame_id} contains no window snapshots")

        # 1. Pacing Calculation (Primary: last_5_10s, secondary: last_20_30s)
        pacing_evidence = [w_10s, w_30s]
        pacing_conf = min(w.evidence_confidence for w in pacing_evidence)
        pacing_drivers: List[str] = []
        pacing_score = self.config.pacing_base_score

        rate_dev = next(
            (d for w in (w_10s, w_curr, w_30s) if w for d in w.deviations if d.feature_name == "speech_rate_wpm"),
            None,
        )
        if rate_dev is not None:
            z = rate_dev.z_score
            if abs(z) < 1.0:
                pacing_score += self.config.pacing_baseline_aligned_boost
                pacing_drivers.append(f"Speech rate aligned with baseline (z={z:.1f})")
            elif abs(z) >= 2.0:
                pacing_score -= self.config.pacing_significant_departure_penalty
                pacing_drivers.append(f"Significant speech rate departure from baseline (z={z:.1f})")
            else:
                pacing_score -= self.config.pacing_moderate_departure_penalty
                pacing_drivers.append(f"Moderate speech rate deviation (z={z:.1f})")

        latency = w_10s.timing_features.response_latency_ms
        if latency is not None:
            if latency > 2000:
                pacing_score -= self.config.pacing_prolonged_latency_penalty
                pacing_drivers.append(f"Prolonged response latency ({latency}ms)")
            elif latency < 150:
                pacing_score -= self.config.pacing_rushed_turnaround_penalty
                pacing_drivers.append(f"Rushed response turnaround ({latency}ms)")
            else:
                pacing_drivers.append("Response latency within natural cadence")

        interruptions = w_30s.timing_features.interruptions_60s
        if interruptions > 1:
            pacing_score -= self.config.pacing_interruption_penalty * min(3, interruptions)
            pacing_drivers.append(f"Cross-speaker interruptions detected ({interruptions})")

        pacing_score = max(self.config.score_floor_active, min(self.config.score_ceiling, round(pacing_score, 2)))
        pacing = DimensionScore(
            score=pacing_score,
            confidence=round(pacing_conf, 2),
            primary_horizon="last_5_10s",
            contributing_evidence_ids=[w_10s.evidence_id, w_30s.evidence_id],
            drivers=pacing_drivers,
        )

        # 2. Emotion / Tension State (Primary: last_5_10s & last_20_30s)
        emotion_evidence = [w_10s, w_30s, w_curr]
        emotion_conf = min(w.evidence_confidence for w in emotion_evidence)
        emotion_signals: List[str] = []
        tension_level = self.config.emotion_base_tension
        valence = self.config.emotion_base_valence

        if interruptions > 0:
            tension_level += self.config.emotion_interruption_tension_boost * min(3, interruptions)
            emotion_signals.append(f"Interruption overlaps ({interruptions})")

        if latency is not None and latency > 1800:
            tension_level += self.config.emotion_latency_tension_boost
            emotion_signals.append("Delayed response pause clustering")

        cps_30s = w_30s.change_points
        if any(cp.direction == "drop" and cp.feature_name == "speech_rate_wpm" for cp in cps_30s):
            tension_level += self.config.emotion_rate_drop_tension_boost
            emotion_signals.append("Speech rate sudden deceleration")

        sem_curr = (
            w_curr.semantic_features
            if (w_curr and w_curr.semantic_features)
            else (w_10s.semantic_features if w_10s else None)
        )
        # Emotion / Tension considers evidence in local horizons (last_5_10s, last_20_30s)
        sem_emotion = (
            w_30s.semantic_features
            if (w_30s and w_30s.semantic_features)
            else sem_curr
        )
        if sem_emotion is not None:
            if sem_emotion.boundary_score >= 0.80:
                tension_level += self.config.emotion_boundary_tension_boost
                valence -= self.config.emotion_boundary_valence_penalty
                emotion_signals.append("Hard boundary / stop-contact marker")
            if sem_emotion.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe"):
                tension_level += self.config.emotion_recurrence_tension_boost
                valence -= self.config.emotion_recurrence_valence_penalty
                emotion_signals.append(f"Recurrent concern ({sem_emotion.recurrence_type})")
                if sem_emotion.recurrence_type == "concern_after_failed_reframe":
                    tension_level += self.config.emotion_failed_reframe_tension_boost
                    valence -= self.config.emotion_failed_reframe_valence_penalty
                    emotion_signals.append("Prospect repeated concern following failed rep reframe attempt")
            if sem_emotion.agreement_score >= 0.70:
                valence += self.config.emotion_substantive_agreement_valence_boost
                tension_level = max(0.05, tension_level - self.config.emotion_substantive_agreement_tension_relief)
                emotion_signals.append(f"Substantive agreement ({sem_emotion.agreement_score:.2f})")
            elif sem_emotion.agreement_score >= 0.30:
                valence += self.config.emotion_polite_agreement_valence_boost
                emotion_signals.append("Polite acknowledgment agreement")

        tension_level = max(self.config.tension_min, min(self.config.tension_max, round(tension_level, 2)))
        valence = max(self.config.valence_min, min(self.config.valence_max, round(valence, 2)))
        emotion = EmotionState(
            expressed_valence=valence,
            tension_level=tension_level,
            confidence=round(emotion_conf, 2),
            tone_claim_made=False,
            observable_signals=emotion_signals,
            contributing_evidence_ids=[w_curr.evidence_id, w_10s.evidence_id, w_30s.evidence_id],
        )

        # 3. Engagement Calculation (Primary: last_20_30s & last_60_90s)
        eng_evidence = [w_30s, w_60s]
        eng_conf = min(w.evidence_confidence for w in eng_evidence)
        eng_drivers: List[str] = []
        eng_score = self.config.engagement_base_score

        talk_ratio = w_60s.talk_ratio_client
        if 0.35 <= talk_ratio <= 0.65:
            eng_score += self.config.engagement_balanced_dialogue_boost
            eng_drivers.append(f"Balanced conversational dialogue (client talk ratio: {talk_ratio:.1%})")
        elif talk_ratio < 0.15:
            eng_score -= self.config.engagement_low_participation_penalty
            eng_drivers.append(f"Low client participation (client talk ratio: {talk_ratio:.1%})")
        else:
            eng_score += self.config.engagement_active_share_boost
            eng_drivers.append(f"Active client vocal share ({talk_ratio:.1%})")

        q_rate = w_60s.timing_features.question_rate_per_min
        if q_rate >= 1.5:
            eng_score += self.config.engagement_high_inquiry_boost
            eng_drivers.append(f"High inquiry velocity ({q_rate:.1f} questions/min)")
        elif q_rate >= 0.5:
            eng_score += self.config.engagement_steady_inquiry_boost
            eng_drivers.append(f"Steady inquiry rate ({q_rate:.1f} questions/min)")

        sem_eng = (
            w_60s.semantic_features
            if (w_60s and w_60s.semantic_features)
            else sem_curr
        )
        if sem_eng is not None and sem_eng.specificity_score >= 0.50:
            eng_score += self.config.engagement_specificity_boost
            eng_drivers.append(f"Specific details and facts provided (score: {sem_eng.specificity_score:.2f})")

        eng_score = max(self.config.score_floor_active, min(self.config.score_ceiling, round(eng_score, 2)))
        engagement = DimensionScore(
            score=eng_score,
            confidence=round(eng_conf, 2),
            primary_horizon="last_60_90s",
            contributing_evidence_ids=[w_30s.evidence_id, w_60s.evidence_id],
            drivers=eng_drivers,
        )

        # 4. Trust Calculation (Primary: last_60_90s & full_call)
        trust_evidence = [w_60s, w_full]
        trust_conf = min(w.evidence_confidence for w in trust_evidence)
        trust_drivers: List[str] = []
        trust_score = self.config.trust_base_score

        sem_trust = (
            w_60s.semantic_features
            if (w_60s and w_60s.semantic_features)
            else (w_full.semantic_features if w_full else sem_curr)
        )
        if sem_trust is not None:
            if sem_trust.boundary_score >= 0.80:
                trust_score -= self.config.trust_boundary_penalty
                trust_drivers.append("Boundary statement suppresses trust")
            if sem_trust.recurrence_type == "positive_echo":
                trust_score += self.config.trust_positive_echo_boost
                trust_drivers.append("Client echoes rep strategic framing")
            elif sem_trust.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe"):
                trust_score -= self.config.trust_unresolved_objection_penalty
                trust_drivers.append(f"Unresolved objection recurrence ({sem_trust.recurrence_type})")
            if sem_trust.agreement_score >= 0.70:
                trust_score += self.config.trust_substantive_agreement_boost
                trust_drivers.append("Substantive alignment on strategic points")

        t_len = w_60s.timing_features.turn_length_words
        if t_len >= 12:
            trust_score += self.config.trust_sustained_disclosure_boost
            trust_drivers.append(f"Sustained disclosure depth ({t_len} words/turn)")
        elif t_len <= 3:
            trust_score -= self.config.trust_constrained_response_penalty
            trust_drivers.append(f"Constrained monosyllabic responses ({t_len} words/turn)")

        trust_score = max(self.config.score_floor_active, min(self.config.score_ceiling, round(trust_score, 2)))
        trust = DimensionScore(
            score=trust_score,
            confidence=round(trust_conf, 2),
            primary_horizon="last_60_90s",
            contributing_evidence_ids=[w_60s.evidence_id, w_full.evidence_id],
            drivers=trust_drivers,
        )

        # 5. Momentum Calculation (Primary: last_20_30s transitioning into last_60_90s)
        mom_evidence = [w_30s, w_60s]
        mom_conf = min(w.evidence_confidence for w in mom_evidence)
        mom_drivers: List[str] = []
        mom_score = self.config.momentum_base_score

        if tension_level <= 0.20:
            mom_score += self.config.momentum_low_friction_boost
            mom_drivers.append("Conversational friction remains minimal")
        elif tension_level >= 0.60:
            mom_score -= self.config.momentum_high_friction_penalty
            mom_drivers.append(f"Elevated friction dampens forward momentum (tension: {tension_level:.2f})")

        if sem_curr is not None:
            if sem_curr.question_type == "transactional":
                mom_score += self.config.momentum_transactional_question_boost
                mom_drivers.append("Client inquires on operational process / next steps")
            elif sem_curr.question_type == "hostile":
                mom_score -= self.config.momentum_hostile_question_penalty
                mom_drivers.append("Hostile challenge stalls progress")

        mom_score = max(self.config.score_floor_active, min(self.config.score_ceiling, round(mom_score, 2)))
        momentum = DimensionScore(
            score=mom_score,
            confidence=round(mom_conf, 2),
            primary_horizon="last_20_30s",
            contributing_evidence_ids=[w_30s.evidence_id, w_60s.evidence_id],
            drivers=mom_drivers,
        )

        # 6. Readiness Calculation (Primary: full_call & last_60_90s)
        read_evidence = [w_60s, w_full, w_curr]
        read_drivers: List[str] = []

        # Hard boundary instantly sets readiness to zero
        if sem_curr is not None and sem_curr.boundary_score >= 0.80:
            read_score = self.config.readiness_hard_boundary_override_score
            read_drivers.append("Hard boundary encountered: readiness overridden to 0.0")
            # The override is driven deterministically by the boundary detection alone.
            # Its confidence is inherited from the boundary detection, bypassing smoothed multi-horizon timing.
            read_conf = getattr(sem_curr, "boundary_confidence", sem_curr.semantic_confidence)
            read_evidence_ids = [w_curr.evidence_id]
        else:
            read_score = self.config.readiness_base_score
            read_conf = min(w.evidence_confidence for w in read_evidence)
            read_evidence_ids = [w_curr.evidence_id, w_60s.evidence_id, w_full.evidence_id]

            if sem_curr is not None:
                if sem_curr.recurrence_type == "scheduling_detail_repeated":
                    read_score += self.config.readiness_scheduling_recurrence_boost
                    read_drivers.append("Client repeats scheduling/next-step milestone details")
                if sem_curr.future_language_score >= 0.60:
                    read_score += self.config.readiness_future_language_boost
                    read_drivers.append(f"Operational future commitment language ({sem_curr.future_language_score:.2f})")
                if sem_curr.agreement_score >= 0.70:
                    read_score += self.config.readiness_agreement_boost
                    read_drivers.append("Substantive agreement on value proposition")
                elif sem_curr.agreement_score < 0.30 and sem_curr.recurrence_type in (
                    "same_objection_repeated",
                    "concern_after_failed_reframe",
                ):
                    read_score -= self.config.readiness_unresolved_objection_penalty
                    read_drivers.append("Unresolved core objection blocks closing readiness")

            if trust_score >= 0.75:
                read_score += self.config.readiness_trust_foundation_boost
                read_drivers.append(f"Supported by strong trust foundation ({trust_score:.2f})")

        read_score = max(self.config.readiness_floor, min(self.config.score_ceiling, round(read_score, 2)))
        readiness = DimensionScore(
            score=read_score,
            confidence=round(read_conf, 2),
            primary_horizon="full_call",
            contributing_evidence_ids=read_evidence_ids,
            drivers=read_drivers,
        )

        # Overall confidence is strict minimum across all 6 dimensions
        overall_conf = min(
            pacing.confidence,
            emotion.confidence,
            engagement.confidence,
            trust.confidence,
            momentum.confidence,
            readiness.confidence,
        )

        # Collect unique contributing evidence IDs
        all_evidence_ids: List[str] = []
        for d in (pacing, engagement, trust, momentum, readiness):
            for eid in d.contributing_evidence_ids:
                if eid not in all_evidence_ids:
                    all_evidence_ids.append(eid)
        for eid in emotion.contributing_evidence_ids:
            if eid not in all_evidence_ids:
                all_evidence_ids.append(eid)

        # Collect unique baseline sources from change points and active deviations
        baseline_sources: List[str] = []
        for f in frames:
            for snap in f.windows.values():
                for cp in snap.change_points:
                    if cp.baseline_source and cp.baseline_source not in baseline_sources:
                        baseline_sources.append(cp.baseline_source)
                for dev in snap.deviations:
                    if dev.is_measured and dev.baseline_source and dev.baseline_source not in baseline_sources:
                        baseline_sources.append(dev.baseline_source)

        acoustic_flag: Literal["unavailable", "available"] = (
            "available" if self.acoustic_provider.is_available() else "unavailable"
        )

        latency_ms = round((time.monotonic() - start_t) * 1000.0, 3)

        return DownstreamInferenceState(
            call_sid=call_sid,
            timestamp_ms=current_frame.timestamp_ms,
            state_version=version,
            inference_latency_ms=latency_ms,
            trust=trust,
            emotion=emotion,
            pacing=pacing,
            engagement=engagement,
            momentum=momentum,
            readiness=readiness,
            acoustic_evidence=acoustic_flag,
            overall_confidence=round(overall_conf, 2),
            contributing_evidence_ids=all_evidence_ids,
            baseline_sources=baseline_sources,
        )
