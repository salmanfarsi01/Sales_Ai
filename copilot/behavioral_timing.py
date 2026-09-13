from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from .behavioral_normalization import (
    NormalizedUtterance,
    NormalizedTurnEvent,
    detect_turn_events,
)


class TimingFeatureSnapshot(BaseModel):
    timestamp_ms: int = Field(..., ge=0)
    window_ms: int = Field(default=60000, ge=1000)
    speaker_id: Literal["salesperson", "client"] = Field(...)
    speech_rate_wpm: float = Field(0.0, ge=0.0)
    turn_speech_rate_wpm: float = Field(
        default=0.0,
        ge=0.0,
        description="Instantaneous speech rate of this specific turn in WPM",
    )
    avg_pause_duration_ms: float = Field(0.0, ge=0.0)
    intra_turn_pause_count: int = Field(0, ge=0)
    pause_measured: bool = Field(
        default=False,
        description="Explicit boolean distinguishing whether intra-turn pauses (>= 250ms) were measured versus absence of pauses",
    )
    response_latency_ms: Optional[int] = None
    interruptions_60s: int = Field(0, ge=0)
    turn_length_words: int = Field(0, ge=0)
    turn_duration_ms: int = Field(0, ge=0)
    question_count_60s: int = Field(0, ge=0)
    question_rate_per_min: float = Field(0.0, ge=0.0)
    timing_confidence: float = Field(1.0, ge=0.0, le=1.0)


class DeterministicTimingEngine:
    def __init__(self, window_ms: int = 60000, min_pause_threshold_ms: int = 250):
        self.window_ms = window_ms
        self.min_pause_threshold_ms = min_pause_threshold_ms
        self.utterances: List[NormalizedUtterance] = []
        self.turn_events: List[NormalizedTurnEvent] = []

    def _is_question(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if cleaned.endswith("?"):
            return True
        first_word = cleaned.split()[0].lower().rstrip("?,.")
        question_words = {
            "what", "when", "where", "which", "who", "whom", "whose",
            "why", "how", "is", "are", "was", "were", "do", "does",
            "did", "can", "could", "should", "would", "will", "won't",
            "haven't", "hasn't", "isn't", "aren't", "wasn't", "weren't"
        }
        return first_word in question_words

    def get_raw_events(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> List[NormalizedTurnEvent]:
        events = self.turn_events
        if start_ms is not None:
            events = [e for e in events if e.timestamp_ms >= start_ms]
        if end_ms is not None:
            events = [e for e in events if e.timestamp_ms <= end_ms]
        return events

    def get_raw_utterances(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> List[NormalizedUtterance]:
        utts = self.utterances
        if start_ms is not None:
            utts = [u for u in utts if u.end_ms >= start_ms]
        if end_ms is not None:
            utts = [u for u in utts if u.start_ms <= end_ms]
        return utts

    def process_utterance(self, utterance: NormalizedUtterance) -> TimingFeatureSnapshot:
        if self.utterances:
            prev_utt = self.utterances[-1]
            inter_events = detect_turn_events(utterance, prev_utt)
            self.turn_events.extend(inter_events)

        self.turn_events.append(
            NormalizedTurnEvent(
                event_id=f"start_{utterance.utterance_id}",
                call_sid=utterance.call_sid,
                event_type="turn_start",
                speaker_id=utterance.speaker_id,
                timestamp_ms=utterance.start_ms,
                duration_ms=0,
            )
        )
        self.turn_events.append(
            NormalizedTurnEvent(
                event_id=f"end_{utterance.utterance_id}",
                call_sid=utterance.call_sid,
                event_type="turn_end",
                speaker_id=utterance.speaker_id,
                timestamp_ms=utterance.end_ms,
                duration_ms=max(0, utterance.end_ms - utterance.start_ms),
            )
        )

        if not utterance.is_estimated_timing and len(utterance.words) > 1:
            for i in range(len(utterance.words) - 1):
                w1 = utterance.words[i]
                w2 = utterance.words[i + 1]
                if not w1.is_estimated and not w2.is_estimated:
                    gap = w2.start_ms - w1.end_ms
                    if gap >= self.min_pause_threshold_ms:
                        self.turn_events.append(
                            NormalizedTurnEvent(
                                event_id=f"pause_{utterance.utterance_id}_{i}",
                                call_sid=utterance.call_sid,
                                event_type="silence",
                                speaker_id=utterance.speaker_id,
                                timestamp_ms=w1.end_ms,
                                duration_ms=gap,
                                metadata={"source": "intra_turn_pause"},
                            )
                        )

        self.utterances.append(utterance)
        return self.compute_snapshot_for_window(
            window_ms=self.window_ms,
            speaker_id=utterance.speaker_id,
            at_timestamp_ms=utterance.end_ms,
        )

    def compute_snapshot_for_window(
        self,
        window_ms: int,
        speaker_id: Literal["salesperson", "client"],
        at_timestamp_ms: Optional[int] = None,
    ) -> TimingFeatureSnapshot:
        if not self.utterances:
            raise ValueError("Cannot compute timing snapshot with empty utterance history")

        current_utt = self.utterances[-1]
        target_time = at_timestamp_ms if at_timestamp_ms is not None else current_utt.end_ms
        window_start_ms = max(0, target_time - window_ms)

        turn_duration_ms = max(1, current_utt.end_ms - current_utt.start_ms)
        turn_words = len(current_utt.words) if current_utt.words else len(current_utt.text.split())
        turn_minutes = turn_duration_ms / 60000.0
        turn_speech_rate_wpm = round(turn_words / turn_minutes, 1) if turn_minutes > 0 else 0.0

        response_latency_ms: Optional[int] = None
        for prev in reversed(self.utterances[:-1]):
            if prev.speaker_id != current_utt.speaker_id:
                response_latency_ms = current_utt.start_ms - prev.end_ms
                break

        intra_turn_pause_count = 0
        total_pause_duration_ms = 0
        if not current_utt.is_estimated_timing and len(current_utt.words) > 1:
            for i in range(len(current_utt.words) - 1):
                w1 = current_utt.words[i]
                w2 = current_utt.words[i + 1]
                if not w1.is_estimated and not w2.is_estimated:
                    gap = w2.start_ms - w1.end_ms
                    if gap >= self.min_pause_threshold_ms:
                        intra_turn_pause_count += 1
                        total_pause_duration_ms += gap

        avg_pause_duration_ms = (
            total_pause_duration_ms / intra_turn_pause_count
            if intra_turn_pause_count > 0
            else 0.0
        )

        recent_utts = [
            u for u in self.utterances
            if u.end_ms >= window_start_ms and u.start_ms <= target_time
        ]

        speaker_recent = [u for u in recent_utts if u.speaker_id == speaker_id]
        total_speaker_words = sum(
            len(u.words) if u.words else len(u.text.split())
            for u in speaker_recent
        )
        total_voiced_duration_ms = sum(
            max(1, u.end_ms - u.start_ms)
            for u in speaker_recent
        )
        voiced_minutes = total_voiced_duration_ms / 60000.0
        speech_rate_wpm = (
            round(total_speaker_words / voiced_minutes, 1)
            if voiced_minutes > 0
            else 0.0
        )

        interruptions_window = 0
        for i in range(len(recent_utts)):
            for j in range(i + 1, len(recent_utts)):
                u1 = recent_utts[i]
                u2 = recent_utts[j]
                if u1.speaker_id != u2.speaker_id:
                    if u1.start_ms < u2.start_ms < u1.end_ms:
                        interruptions_window += 1

        speaker_questions = [u for u in speaker_recent if self._is_question(u.text)]
        question_count_window = len(speaker_questions)
        call_start_ms = self.utterances[0].start_ms
        elapsed_window_minutes = min(window_ms, max(1000, target_time - call_start_ms)) / 60000.0
        question_rate_per_min = (
            round(question_count_window / elapsed_window_minutes, 2)
            if elapsed_window_minutes > 0
            else 0.0
        )

        timing_confidence = 0.5 if current_utt.is_estimated_timing else 1.0

        return TimingFeatureSnapshot(
            timestamp_ms=target_time,
            window_ms=window_ms,
            speaker_id=speaker_id,
            speech_rate_wpm=speech_rate_wpm,
            turn_speech_rate_wpm=turn_speech_rate_wpm,
            avg_pause_duration_ms=round(avg_pause_duration_ms, 1),
            intra_turn_pause_count=intra_turn_pause_count,
            pause_measured=intra_turn_pause_count > 0,
            response_latency_ms=response_latency_ms,
            interruptions_60s=interruptions_window,
            turn_length_words=turn_words,
            turn_duration_ms=turn_duration_ms,
            question_count_60s=question_count_window,
            question_rate_per_min=question_rate_per_min,
            timing_confidence=timing_confidence,
        )
