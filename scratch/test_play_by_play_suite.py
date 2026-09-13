import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.behavioral_normalization import NormalizedUtterance, NormalizedWord
from copilot.behavioral_timing import DeterministicTimingEngine
from copilot.behavioral_semantic import SemanticFeatureEngine
from copilot.behavioral_baseline import BaselineAndChangePointEngine, ProspectBaselineStore
from copilot.behavioral_evidence import MultiWindowAggregator, SQLiteEvidenceLogStore
from copilot.behavioral_inference import DownstreamInferenceEngine

async def run_client_play_by_play():
    call_sid = "test_play_by_play_call_001"
    prospect_id = "client_eval_prospect"

    timing_engine = DeterministicTimingEngine()
    semantic_engine = SemanticFeatureEngine()
    baseline_store = ProspectBaselineStore(Path("knowledge/test_play_by_play_baselines.json"))
    evidence_store = SQLiteEvidenceLogStore(":memory:")
    
    baseline_engine = BaselineAndChangePointEngine(
        call_sid=call_sid,
        prospect_id=prospect_id,
        store=baseline_store,
        intra_call_window_ms=60000,
        min_calibration_turns=3,
        min_cumulative_words=15,
    )
    aggregator = MultiWindowAggregator(
        call_sid=call_sid,
        timing_engine=timing_engine,
        baseline_engine=baseline_engine,
        store=evidence_store,
    )
    inference_engine = DownstreamInferenceEngine()

    def make_utt(turn_idx, text, start_ms, end_ms, speaker_id="client"):
        words_list = text.strip().split()
        dur = end_ms - start_ms
        per_word = dur / max(1, len(words_list))
        norm_words = []
        for i, w in enumerate(words_list):
            w_start = start_ms + int(i * per_word)
            w_end = w_start + int(per_word * 0.85)
            norm_words.append(NormalizedWord(word=w, start_ms=w_start, end_ms=w_end, confidence=0.98))
        return NormalizedUtterance(
            utterance_id=f"utt_{turn_idx:02d}",
            call_sid=call_sid,
            speaker_id=speaker_id,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            words=norm_words,
            confidence=0.98,
        )

    # Segments matching client script:
    turns_data = [
        # Segment A: Baseline calibration (turns 1, 2, 3 covering 64 seconds)
        (1, "client", "We've been looking at homes for a couple months now. My husband and I aren't in a huge rush, but our current lease ends in about three months, so we do need to make a decision before then.", 1000, 17000),
        (2, "client", "We're mainly looking for something with at least three bedrooms and a bit of outdoor space, since we have two dogs.", 25000, 34000),
        (3, "client", "We toured a place two weeks ago that we both really liked, but it ended up going to another buyer before we could put an offer in.", 43000, 53000),
        
        # Segment B: [TEST 1: Pace change fast + TEST 4: Real question]
        # Spoken noticeably fast (24 words in 6.8s -> ~212 WPM)
        (4, "client", "Okay so what would the total commission actually work out to and how soon could we realistically close if we found something this week?", 65000, 71800),
        
        # Rep response
        (5, "salesperson", "Our full service fee is six percent, and we can generally close within thirty days of an accepted offer.", 72000, 76000),
        
        # [TEST 2: Pause - 4.5s hesitation latency before Segment C]
        # Segment C: [TEST 3: Vague concern]
        (6, "client", "Honestly, that commission just feels like a lot for what's actually involved.", 80500, 85000),
        
        # Rep attempts reframe
        (7, "salesperson", "I completely understand, but our team manages every step from staging and photography to all contract negotiations.", 85500, 89500),
        
        # Segment D: [TEST 3: Recurrence - same concern, different words after rep reframe]
        (8, "client", "I hear you, but I still think we're paying too much for what you're actually doing here.", 91400, 97900),
        
        # Segment E: [TEST 5: Specificity increasing]
        (9, "client", "Our budget is around four hundred and thirty thousand, we'd want to close by the fifteenth of next month, and my brother-in-law, who's a contractor, would want to walk through anything before we sign.", 101000, 113500),
        
        # Segment F: [TEST 6: Clear boundary]
        (10, "client", "Actually, please don't contact me about this anymore.", 115500, 118500),
    ]

    play_by_play = []
    context_history = []

    for t_idx, speaker, text, s_ms, e_ms in turns_data:
        utt = make_utt(t_idx, text, s_ms, e_ms, speaker_id=speaker)
        timing_snap = timing_engine.process_utterance(utt)
        sem_snap = await semantic_engine.analyze_turn_semantic(utt, context_history=context_history)
        context_history.append(utt)

        new_cps = baseline_engine.update_with_utterance(utt, timing_snap, sem_snap)
        frame = aggregator.process_turn(utt, timing_snap, sem_snap, new_change_points=new_cps)
        inf = inference_engine.compute_inference(call_sid=call_sid, current_frame=frame)

        if speaker == "salesperson":
            continue

        active_profile = baseline_engine.get_active_profile()
        base_wpm = round(active_profile.speech_rate_wpm.mean, 1) if active_profile and active_profile.speech_rate_wpm.sample_count > 0 else None
        curr_wpm = round(timing_snap.speech_rate_wpm, 1)
        pace_delta = round(((curr_wpm - base_wpm) / base_wpm) * 100.0, 1) if (base_wpm and base_wpm > 0) else None

        play_by_play.append({
            "turn": t_idx,
            "time_sec": f"{round(s_ms/1000.0, 1)}s - {round(e_ms/1000.0, 1)}s",
            "text": text,
            "wpm": curr_wpm,
            "baseline_wpm": base_wpm,
            "pace_delta_pct": f"{pace_delta:+0.1f}%" if pace_delta is not None else "calibrating",
            "pause_ms": round(timing_snap.avg_pause_duration_ms, 0),
            "latency_ms": timing_snap.response_latency_ms if timing_snap.response_latency_ms is not None else "—",
            "question": sem_snap.question_type,
            "recurrence": f"{sem_snap.recurrence_type} (cnt: {sem_snap.recurrence_count})" if sem_snap.recurrence_type != "none" else "none",
            "boundary": sem_snap.boundary_score,
            "specificity": sem_snap.specificity_score,
            "future_lang": sem_snap.future_language_score,
            "confidence": sem_snap.semantic_confidence,
            "trust": round(inf.trust.score * 100),
            "pacing": round(inf.pacing.score * 100),
            "readiness": round(inf.readiness.score * 100),
            "locked": baseline_engine.is_intra_call_locked,
        })

    print(json.dumps(play_by_play, indent=2))

if __name__ == "__main__":
    asyncio.run(run_client_play_by_play())
