# Input Contract Validation & Gap Report (Spec 07 Section 2)

Date: September 2026  
Status: Phase 0 Deliverable  
Target: PitchProX Behavioral Signal Engine  

---

## 1. Executive Summary

This report validates the input data delivered by the active PitchProX telephony and automated speech recognition (ASR) pipeline against the required input contract in Spec 07 Section 2.

The active pipeline consists of:
1. Twilio Voice Media Streams: WebSocket delivering raw 8kHz mu-law audio packets split by track (inbound vs outbound).
2. Deepgram Nova-3 Streaming ASR: WebSocket connection (`wss://api.deepgram.com/v1/listen`) processing live mu-law audio with interim results, endpointing, voice activity detection (VAD), and smart formatting.

The evaluation confirms that all required inputs for transcript-derived timing and semantic feature extraction are either fully available or derivable from existing streams. Acoustic signal modeling is explicitly marked as deferred per the project scope.

---

## 2. Input Contract Gap Matrix

| Input Contract Item | Required Fields | Spec 07 Supported Capability | PitchProX Status | Pipeline Source & Verification Notes |
| :--- | :--- | :--- | :--- | :--- |
| Speaker-separated transcript | `speaker_id`, `utterance text`, `utterance start/end`, `ASR confidence` | Meaning, intent, objection, questions, repetition, disclosure depth | Available | Dual-track media stream in Twilio protocol. Inbound packets represent salesperson (`track == "inbound"`), outbound packets represent client (`track == "outbound"`). Deepgram instance per track provides independent start/end offsets and confidence scores. |
| Word timestamps | `word`, `start_ms`, `end_ms`, `confidence` | Speech rate, pauses, overlap, response latency, emphasis timing | Available | Deepgram Nova-3 returns a `words` array on every alternative in the `Results` event payload. Each word object includes `word`, `start` (seconds float), `end` (seconds float), and `confidence` (float). Previously discarded by the copilot consumer; extracted in Phase 1 normalization. |
| Turn events | Speaker changes, interruption/overlap flags, silence windows | Turn-taking, dominance, pacing alignment, engagement | Derivable | Determined by comparing the start and end millisecond boundaries of consecutive and concurrent utterances across the salesperson and client channels. |
| Call metadata | `call_id`, `lead_type`, `user_id`, `prospect_id`, `stage`, active Playbook, Calibration profile | Contextual normalization | Available | Provided via query parameters on WebSocket connection (`tenant_id`, `salesman_id`, `call_sid`) and linked through `PreCallFolder` and active `Playbook` state. |
| Acoustic features | Pitch contour, RMS/energy, speaking-rate estimate, voicing, prosody embedding | Tone/arousal confidence; excitement vs agitation; calm vs flatness | Deferred (Architectural Slot) | Raw audio chunks flow through Twilio queues, and digital signal processing algorithms exist in `copilot/calibration.py` (`compute_energy_dynamics`). However, live acoustic tone classification is deferred per scope agreement until Phase 7. Downstream confidence calculations must not claim acoustic support. |
| Historical baseline | Prospect prior calls, user calibration baseline | Personalized change detection rather than population-only assumptions | Partial | Salesperson baseline is available from voice calibration profiles (`speaking_pace`, `energy_dynamics`). Prospect cross-call baseline persistence is deferred to Phase 4; in-call prospect baseline will be initialized during the first 60 to 90 seconds of clean speech. |

---

## 3. Technical Findings & Action Items for Phase 1

1. Deepgram Word Extraction:
   The streaming STT handler in `copilot/fastapi_app.py` previously extracted only `event["channel"]["alternatives"][0]["transcript"]`. Phase 1 extracts the `words` array and converts floating-point timestamps into integer milliseconds to prevent precision drift.

2. Clock Synchronization:
   Deepgram returns timestamps relative to the start of each audio stream (`start` and `end` in seconds). To ensure multi-channel alignment across salesperson and client tracks, the normalization layer tracks stream initialization time (`call_started` monotonic epoch) and produces normalized timestamps referenced to call inception.

3. Graceful Handling of Unpunctuated / Partial Frames:
   Interim ASR results may contain incomplete word arrays. Only final or endpointed utterances (`is_final == True` or `speech_final == True`) commit to the persistent behavioral buffer. Interim events are buffered for real-time latency measurements.

---

## 4. Phase 1 Verification & Closure Checklist

1. Cross-Track Timestamp Synchronization:
   - Twilio media events deliver `media.timestamp` (elapsed RTP milliseconds from stream inception).
   - `first_packet_twilio_ms` captures the exact arrival offset of the first packet on each track.
   - Deepgram zero-based timestamps for each track are shifted by `stream_offset_ms = first_packet_twilio_ms[role]`, locking both tracks into a single unified call timeline.

2. Call Metadata Field Mapping:
   - `CallMetadata` model explicitly defines: `call_id`, `lead_type`, `user_id`, `prospect_id`, `stage`, `active_playbook`, `calibration_profile`.
   - Extracted directly from WebSocket query parameters and `PreCallFolder` at session initialization.

3. Synthetic Timing Distinction:
   - `NormalizedWord` carries `is_estimated: bool = False`.
   - `NormalizedUtterance` carries `is_estimated_timing: bool = False`.
   - When ASR omits word timing and synthetic spacing is applied, `is_estimated_timing` and word `is_estimated` are set to `True`, preventing downstream feature engines from treating synthetic intervals as observed acoustic data.

4. Test Suite Baseline:
   - 95 passed, 1 skipped.
   - The single skipped test is `tests/test_file_extraction.py::TestPDFExtraction::test_extract_pdf_basic`, which gracefully skips when optional dependency `PyMuPDF` (`fitz`) is not installed. All behavioral engine and core telephony tests run and pass at 100%.
