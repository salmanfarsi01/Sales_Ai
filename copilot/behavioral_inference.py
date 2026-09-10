from __future__ import annotations

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
    trust: DimensionScore
    emotion: EmotionState
    pacing: DimensionScore
    engagement: DimensionScore
    momentum: DimensionScore
    readiness: DimensionScore
    acoustic_evidence: Literal["unavailable", "partial", "available"] = "unavailable"
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    contributing_evidence_ids: List[str] = Field(default_factory=list)
    baseline_sources: List[str] = Field(default_factory=list)


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
    def __init__(self, acoustic_provider: Optional[AcousticEvidenceProvider] = None):
        self.acoustic_provider = acoustic_provider or NullAcousticProvider()

    def compute_inference(
        self,
        call_sid: str,
        current_frame: MultiWindowEvidenceFrame,
        recent_frames: Optional[List[MultiWindowEvidenceFrame]] = None,
    ) -> DownstreamInferenceState:
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
        pacing_score = 0.80

        rate_dev = next(
            (d for w in (w_10s, w_curr, w_30s) if w for d in w.deviations if d.feature_name == "speech_rate_wpm"),
            None,
        )
        if rate_dev is not None:
            z = rate_dev.z_score
            if abs(z) < 1.0:
                pacing_score += 0.10
                pacing_drivers.append(f"Speech rate aligned with baseline (z={z:.1f})")
            elif abs(z) >= 2.0:
                pacing_score -= 0.35
                pacing_drivers.append(f"Significant speech rate departure from baseline (z={z:.1f})")
            else:
                pacing_score -= 0.15
                pacing_drivers.append(f"Moderate speech rate deviation (z={z:.1f})")

        latency = w_10s.timing_features.response_latency_ms
        if latency is not None:
            if latency > 2000:
                pacing_score -= 0.20
                pacing_drivers.append(f"Prolonged response latency ({latency}ms)")
            elif latency < 150:
                pacing_score -= 0.10
                pacing_drivers.append(f"Rushed response turnaround ({latency}ms)")
            else:
                pacing_drivers.append("Response latency within natural cadence")

        interruptions = w_30s.timing_features.interruptions_60s
        if interruptions > 1:
            pacing_score -= 0.15 * min(3, interruptions)
            pacing_drivers.append(f"Cross-speaker interruptions detected ({interruptions})")

        pacing_score = max(0.05, min(1.0, round(pacing_score, 2)))
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
        tension_level = 0.20
        valence = 0.0

        if interruptions > 0:
            tension_level += 0.25 * min(3, interruptions)
            emotion_signals.append(f"Interruption overlaps ({interruptions})")

        if latency is not None and latency > 1800:
            tension_level += 0.20
            emotion_signals.append("Delayed response pause clustering")

        cps_30s = w_30s.change_points
        if any(cp.direction == "drop" and cp.feature_name == "speech_rate_wpm" for cp in cps_30s):
            tension_level += 0.20
            emotion_signals.append("Speech rate sudden deceleration")

        sem_curr = (
            w_curr.semantic_features
            if (w_curr and w_curr.semantic_features)
            else (w_10s.semantic_features if w_10s else None)
        )
        if sem_curr is not None:
            if sem_curr.boundary_score > 0.0:
                tension_level += 0.40
                valence -= 0.60
                emotion_signals.append("Hard boundary / stop-contact marker")
            if sem_curr.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe"):
                tension_level += 0.25
                valence -= 0.35
                emotion_signals.append(f"Recurrent concern ({sem_curr.recurrence_type})")
            if sem_curr.agreement_score >= 0.70:
                valence += 0.50
                tension_level = max(0.05, tension_level - 0.20)
                emotion_signals.append(f"Substantive agreement ({sem_curr.agreement_score:.2f})")
            elif sem_curr.agreement_score >= 0.30:
                valence += 0.15
                emotion_signals.append("Polite acknowledgment agreement")

        tension_level = max(0.0, min(1.0, round(tension_level, 2)))
        valence = max(-1.0, min(1.0, round(valence, 2)))
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
        eng_score = 0.50

        talk_ratio = w_60s.talk_ratio_client
        if 0.35 <= talk_ratio <= 0.65:
            eng_score += 0.25
            eng_drivers.append(f"Balanced conversational dialogue (client talk ratio: {talk_ratio:.1%})")
        elif talk_ratio < 0.15:
            eng_score -= 0.25
            eng_drivers.append(f"Low client participation (client talk ratio: {talk_ratio:.1%})")
        else:
            eng_score += 0.10
            eng_drivers.append(f"Active client vocal share ({talk_ratio:.1%})")

        q_rate = w_60s.timing_features.question_rate_per_min
        if q_rate >= 1.5:
            eng_score += 0.20
            eng_drivers.append(f"High inquiry velocity ({q_rate:.1f} questions/min)")
        elif q_rate >= 0.5:
            eng_score += 0.10
            eng_drivers.append(f"Steady inquiry rate ({q_rate:.1f} questions/min)")

        if sem_curr is not None and sem_curr.specificity_score >= 0.50:
            eng_score += 0.15
            eng_drivers.append(f"Specific details and facts provided (score: {sem_curr.specificity_score:.2f})")

        eng_score = max(0.05, min(1.0, round(eng_score, 2)))
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
        trust_score = 0.60

        if sem_curr is not None:
            if sem_curr.boundary_score > 0.0:
                trust_score -= 0.45
                trust_drivers.append("Boundary statement suppresses trust")
            if sem_curr.recurrence_type == "positive_echo":
                trust_score += 0.20
                trust_drivers.append("Client echoes rep strategic framing")
            elif sem_curr.recurrence_type in ("same_objection_repeated", "concern_after_failed_reframe"):
                trust_score -= 0.20
                trust_drivers.append(f"Unresolved objection recurrence ({sem_curr.recurrence_type})")
            if sem_curr.agreement_score >= 0.70:
                trust_score += 0.15
                trust_drivers.append("Substantive alignment on strategic points")

        t_len = w_60s.timing_features.turn_length_words
        if t_len >= 12:
            trust_score += 0.10
            trust_drivers.append(f"Sustained disclosure depth ({t_len} words/turn)")
        elif t_len <= 3:
            trust_score -= 0.10
            trust_drivers.append(f"Constrained monosyllabic responses ({t_len} words/turn)")

        trust_score = max(0.05, min(1.0, round(trust_score, 2)))
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
        mom_score = 0.55

        if tension_level <= 0.20:
            mom_score += 0.15
            mom_drivers.append("Conversational friction remains minimal")
        elif tension_level >= 0.60:
            mom_score -= 0.30
            mom_drivers.append(f"Elevated friction dampens forward momentum (tension: {tension_level:.2f})")

        if sem_curr is not None:
            if sem_curr.question_type == "transactional":
                mom_score += 0.20
                mom_drivers.append("Client inquires on operational process / next steps")
            elif sem_curr.question_type == "hostile":
                mom_score -= 0.35
                mom_drivers.append("Hostile challenge stalls progress")

        mom_score = max(0.05, min(1.0, round(mom_score, 2)))
        momentum = DimensionScore(
            score=mom_score,
            confidence=round(mom_conf, 2),
            primary_horizon="last_20_30s",
            contributing_evidence_ids=[w_30s.evidence_id, w_60s.evidence_id],
            drivers=mom_drivers,
        )

        # 6. Readiness Calculation (Primary: full_call & last_60_90s)
        read_evidence = [w_60s, w_full, w_curr]
        read_conf = min(w.evidence_confidence for w in read_evidence)
        read_drivers: List[str] = []
        read_score = 0.30

        # Hard boundary instantly sets readiness to zero
        if sem_curr is not None and sem_curr.boundary_score > 0.0:
            read_score = 0.0
            read_drivers.append("Hard boundary encountered: readiness overridden to 0.0")
        else:
            if sem_curr is not None:
                if sem_curr.recurrence_type == "scheduling_detail_repeated":
                    read_score += 0.35
                    read_drivers.append("Client repeats scheduling/next-step milestone details")
                if sem_curr.future_language_score >= 0.60:
                    read_score += 0.25
                    read_drivers.append(f"Operational future commitment language ({sem_curr.future_language_score:.2f})")
                if sem_curr.agreement_score >= 0.70:
                    read_score += 0.20
                    read_drivers.append("Substantive agreement on value proposition")
                elif sem_curr.agreement_score < 0.30 and sem_curr.recurrence_type in (
                    "same_objection_repeated",
                    "concern_after_failed_reframe",
                ):
                    read_score -= 0.20
                    read_drivers.append("Unresolved core objection blocks closing readiness")

            if trust_score >= 0.75:
                read_score += 0.10
                read_drivers.append(f"Supported by strong trust foundation ({trust_score:.2f})")

        read_score = max(0.0, min(1.0, round(read_score, 2)))
        readiness = DimensionScore(
            score=read_score,
            confidence=round(read_conf, 2),
            primary_horizon="full_call",
            contributing_evidence_ids=[w_curr.evidence_id, w_60s.evidence_id, w_full.evidence_id],
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

        # Collect unique baseline sources from change points
        baseline_sources: List[str] = []
        for f in frames:
            for snap in f.windows.values():
                for cp in snap.change_points:
                    if cp.baseline_source not in baseline_sources:
                        baseline_sources.append(cp.baseline_source)
        if not baseline_sources:
            baseline_sources = ["intra_call"]

        acoustic_flag: Literal["unavailable", "partial", "available"] = (
            "available" if self.acoustic_provider.is_available() else "unavailable"
        )

        return DownstreamInferenceState(
            call_sid=call_sid,
            timestamp_ms=current_frame.timestamp_ms,
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
