from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal
import uuid
from pydantic import BaseModel, Field


class CallMetadata(BaseModel):
    call_id: str
    lead_type: str = "Expired Listings"
    user_id: str = "sales_rep"
    prospect_id: Optional[str] = None
    stage: str = "Discovery"
    active_playbook: Optional[str] = None
    calibration_profile: Optional[str] = None
    semantic_timeout_sec: Optional[float] = None


class NormalizedWord(BaseModel):
    word: str
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    punctuated_word: Optional[str] = None
    is_estimated: bool = Field(default=False)


class NormalizedUtterance(BaseModel):
    utterance_id: str = Field(default_factory=lambda: f"utt_{uuid.uuid4().hex[:10]}")
    call_sid: str = Field(default="")
    speaker_id: Literal["salesperson", "client"] = Field(...)
    text: str = Field(...)
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    asr_confidence: float = Field(1.0, ge=0.0, le=1.0)
    is_final: bool = Field(default=True)
    speech_final: bool = Field(default=True)
    is_estimated_timing: bool = Field(default=False)
    words: List[NormalizedWord] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedTurnEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    call_sid: str = Field(default="")
    event_type: Literal["turn_start", "turn_end", "speaker_switch", "silence", "overlap"] = Field(...)
    speaker_id: Literal["salesperson", "client"] = Field(...)
    timestamp_ms: int = Field(..., ge=0)
    duration_ms: Optional[int] = Field(default=None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def normalize_deepgram_result(
    raw_result: Dict[str, Any],
    call_sid: str,
    speaker_id: Literal["salesperson", "client"],
    stream_offset_ms: int = 0,
) -> Optional[NormalizedUtterance]:
    if not isinstance(raw_result, dict):
        return None

    channel = raw_result.get("channel", {})
    alternatives = channel.get("alternatives", [])
    if not alternatives or not isinstance(alternatives, list):
        return None

    alt = alternatives[0]
    if not isinstance(alt, dict):
        return None

    text = (alt.get("transcript") or "").strip()
    if not text:
        return None

    raw_start = raw_result.get("start")
    raw_duration = raw_result.get("duration")
    asr_confidence = float(alt.get("confidence", 1.0))
    is_final = bool(raw_result.get("is_final", False))
    speech_final = bool(raw_result.get("speech_final", False))

    raw_words = alt.get("words", [])
    normalized_words: List[NormalizedWord] = []

    if isinstance(raw_words, list) and raw_words:
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            w_text = item.get("word", "")
            w_start = item.get("start", 0.0)
            w_end = item.get("end", 0.0)
            w_conf = float(item.get("confidence", asr_confidence))
            w_punc = item.get("punctuated_word")

            w_start_ms = max(0, int(round(w_start * 1000))) + stream_offset_ms
            w_end_ms = max(w_start_ms, int(round(w_end * 1000)) + stream_offset_ms)

            normalized_words.append(
                NormalizedWord(
                    word=w_text,
                    start_ms=w_start_ms,
                    end_ms=w_end_ms,
                    confidence=max(0.0, min(1.0, w_conf)),
                    punctuated_word=w_punc,
                )
            )

    is_estimated_timing = not bool(normalized_words)
    if normalized_words:
        utt_start_ms = normalized_words[0].start_ms
        utt_end_ms = normalized_words[-1].end_ms
    else:
        start_sec = float(raw_start) if raw_start is not None else 0.0
        dur_sec = float(raw_duration) if raw_duration is not None else 0.0
        utt_start_ms = max(0, int(round(start_sec * 1000))) + stream_offset_ms
        utt_end_ms = utt_start_ms + max(0, int(round(dur_sec * 1000)))

        words_list = text.split()
        if words_list and utt_end_ms > utt_start_ms:
            step = (utt_end_ms - utt_start_ms) // len(words_list)
            for idx, w in enumerate(words_list):
                w_s = utt_start_ms + idx * step
                w_e = utt_start_ms + (idx + 1) * step if idx < len(words_list) - 1 else utt_end_ms
                normalized_words.append(
                    NormalizedWord(
                        word=w,
                        start_ms=w_s,
                        end_ms=w_e,
                        confidence=asr_confidence,
                        is_estimated=True,
                    )
                )

    return NormalizedUtterance(
        call_sid=call_sid,
        speaker_id=speaker_id,
        text=text,
        start_ms=utt_start_ms,
        end_ms=utt_end_ms,
        asr_confidence=max(0.0, min(1.0, asr_confidence)),
        is_final=is_final,
        speech_final=speech_final,
        is_estimated_timing=is_estimated_timing,
        words=normalized_words,
        metadata={
            "raw_start_sec": raw_start,
            "raw_duration_sec": raw_duration,
        },
    )


def normalize_generic_transcript(
    text: str,
    speaker_id: Literal["salesperson", "client"],
    start_ms: int,
    end_ms: int,
    call_sid: str = "",
    confidence: float = 1.0,
    is_final: bool = True,
    speech_final: bool = True,
    is_estimated_timing: bool = True,
) -> NormalizedUtterance:
    cleaned_text = text.strip()
    words_list = cleaned_text.split()
    normalized_words: List[NormalizedWord] = []

    if words_list:
        total_duration = max(0, end_ms - start_ms)
        step = total_duration // len(words_list) if total_duration > 0 else 0
        for idx, word in enumerate(words_list):
            w_start = start_ms + idx * step
            w_end = start_ms + (idx + 1) * step if idx < len(words_list) - 1 else end_ms
            normalized_words.append(
                NormalizedWord(
                    word=word,
                    start_ms=w_start,
                    end_ms=max(w_start, w_end),
                    confidence=confidence,
                    is_estimated=is_estimated_timing,
                )
            )

    return NormalizedUtterance(
        call_sid=call_sid,
        speaker_id=speaker_id,
        text=cleaned_text,
        start_ms=start_ms,
        end_ms=max(start_ms, end_ms),
        asr_confidence=max(0.0, min(1.0, confidence)),
        is_final=is_final,
        speech_final=speech_final,
        is_estimated_timing=is_estimated_timing,
        words=normalized_words,
    )


def detect_turn_events(
    current_utterance: NormalizedUtterance,
    previous_utterance: Optional[NormalizedUtterance],
) -> List[NormalizedTurnEvent]:
    events: List[NormalizedTurnEvent] = []

    events.append(
        NormalizedTurnEvent(
            call_sid=current_utterance.call_sid,
            event_type="turn_start",
            speaker_id=current_utterance.speaker_id,
            timestamp_ms=current_utterance.start_ms,
        )
    )

    if previous_utterance is not None:
        if previous_utterance.speaker_id != current_utterance.speaker_id:
            events.append(
                NormalizedTurnEvent(
                    call_sid=current_utterance.call_sid,
                    event_type="speaker_switch",
                    speaker_id=current_utterance.speaker_id,
                    timestamp_ms=current_utterance.start_ms,
                    metadata={
                        "previous_speaker": previous_utterance.speaker_id,
                    },
                )
            )

            delta = current_utterance.start_ms - previous_utterance.end_ms
            if delta > 0:
                events.append(
                    NormalizedTurnEvent(
                        call_sid=current_utterance.call_sid,
                        event_type="silence",
                        speaker_id=current_utterance.speaker_id,
                        timestamp_ms=previous_utterance.end_ms,
                        duration_ms=delta,
                    )
                )
            elif delta < 0:
                events.append(
                    NormalizedTurnEvent(
                        call_sid=current_utterance.call_sid,
                        event_type="overlap",
                        speaker_id=current_utterance.speaker_id,
                        timestamp_ms=current_utterance.start_ms,
                        duration_ms=abs(delta),
                        metadata={
                            "overridden_speaker": previous_utterance.speaker_id,
                        },
                    )
                )

    events.append(
        NormalizedTurnEvent(
            call_sid=current_utterance.call_sid,
            event_type="turn_end",
            speaker_id=current_utterance.speaker_id,
            timestamp_ms=current_utterance.end_ms,
            duration_ms=current_utterance.end_ms - current_utterance.start_ms,
        )
    )

    return events


def segment_audio_transcript_turns(
    words_raw: List[Dict[str, Any]],
    transcript: str = "",
    call_sid: str = "",
    speaker_id: Literal["salesperson", "client"] = "client",
    raw_utterances: Optional[List[Dict[str, Any]]] = None,
    min_pause_split_ms: int = 500,
) -> List[NormalizedUtterance]:
    """Segments batch audio transcription into distinct sequential NormalizedUtterance turns.
    Uses Deepgram's pre-segmented utterances if present (>1), or segments word timestamps
    whenever adjacent words have a silence gap >= min_pause_split_ms (or sentence boundary + >=400ms).
    """
    # 1. If Deepgram returned multiple pre-segmented utterances
    if raw_utterances and len(raw_utterances) > 1:
        segmented: List[NormalizedUtterance] = []
        for u in raw_utterances:
            u_text = (u.get("transcript") or "").strip()
            if not u_text:
                continue
            u_words_raw = u.get("words", [])
            norm_words = []
            for w in u_words_raw:
                w_text = w.get("punctuated_word") or w.get("word", "")
                w_start = int(round(float(w.get("start", 0.0)) * 1000))
                w_end = int(round(float(w.get("end", 0.0)) * 1000))
                norm_words.append(
                    NormalizedWord(
                        word=w.get("word") or w_text,
                        start_ms=w_start,
                        end_ms=max(w_start, w_end),
                        confidence=float(w.get("confidence", 0.95)),
                        punctuated_word=w_text,
                    )
                )
            u_start = norm_words[0].start_ms if norm_words else int(round(float(u.get("start", 0.0)) * 1000))
            u_end = norm_words[-1].end_ms if norm_words else int(round(float(u.get("end", 0.0)) * 1000))
            segmented.append(
                NormalizedUtterance(
                    call_sid=call_sid,
                    speaker_id=speaker_id,
                    text=u_text,
                    start_ms=u_start,
                    end_ms=max(u_start + 100, u_end),
                    asr_confidence=float(u.get("confidence", 0.95)),
                    words=norm_words,
                )
            )
        if segmented:
            return segmented

    # 2. Fallback / Word Timestamp Segmentation
    norm_words_all = []
    for w in (words_raw or []):
        w_text = w.get("punctuated_word") or w.get("word", "")
        w_start = int(round(float(w.get("start", 0.0)) * 1000))
        w_end = int(round(float(w.get("end", 0.0)) * 1000))
        norm_words_all.append(
            NormalizedWord(
                word=w.get("word") or w_text,
                start_ms=w_start,
                end_ms=max(w_start, w_end),
                confidence=float(w.get("confidence", 0.95)),
                punctuated_word=w_text,
            )
        )

    if not norm_words_all:
        clean_text = transcript.strip() if transcript else "I see, thanks."
        return [
            NormalizedUtterance(
                call_sid=call_sid,
                speaker_id=speaker_id,
                text=clean_text,
                start_ms=0,
                end_ms=max(1000, len(clean_text.split()) * 300),
                words=[],
                is_estimated_timing=True,
            )
        ]

    chunks: List[List[NormalizedWord]] = []
    current_chunk: List[NormalizedWord] = [norm_words_all[0]]

    for i in range(len(norm_words_all) - 1):
        w_curr = norm_words_all[i]
        w_next = norm_words_all[i + 1]
        gap_ms = w_next.start_ms - w_curr.end_ms
        punc = w_curr.punctuated_word or w_curr.word
        ends_sentence = any(punc.rstrip().endswith(ch) for ch in [".", "?", "!"])

        if gap_ms >= min_pause_split_ms or (ends_sentence and gap_ms >= 400):
            chunks.append(current_chunk)
            current_chunk = [w_next]
        else:
            current_chunk.append(w_next)

    if current_chunk:
        chunks.append(current_chunk)

    segmented: List[NormalizedUtterance] = []
    for chunk in chunks:
        chunk_text = " ".join(w.punctuated_word or w.word for w in chunk).strip()
        segmented.append(
            NormalizedUtterance(
                call_sid=call_sid,
                speaker_id=speaker_id,
                text=chunk_text,
                start_ms=chunk[0].start_ms,
                end_ms=max(chunk[0].start_ms + 100, chunk[-1].end_ms),
                words=chunk,
                asr_confidence=sum(w.confidence for w in chunk) / len(chunk),
            )
        )
    return segmented
