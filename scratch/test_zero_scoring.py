import asyncio
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from copilot.calibration import CalibrationService


async def test_zero_scoring():
    service = CalibrationService()
    stage = "Security & Compliance"
    target = "Our platform uses tenant-isolated storage, end-to-end encryption, and built-in SOC 2 Type II controls with real-time audit logs for GDPR."

    # 1. Silence / Empty
    silence_eval = await service._run_llm_evaluation(
        round_num=3,
        stage=stage,
        target_teleprompt=target,
        user_transcript="",
        duration_seconds=3.0,
    )
    print("=== SCENARIO 1: SILENCE (SPOKE NOTHING) ===")
    print("Overall Score:", silence_eval["overall_score"])
    print("Breakdown:", silence_eval["score_breakdown"])
    print("Feedback:", silence_eval["feedback"])
    assert silence_eval["overall_score"] == 0
    assert silence_eval["score_breakdown"]["word_choice"] == 0
    print(" Passed silence test (0%)!\n")

    # 2. Irrelevant / Off-topic
    irrelevant_transcript = "I love eating pizza with extra cheese while watching football on Sunday."
    irrelevant_eval = await service._run_llm_evaluation(
        round_num=3,
        stage=stage,
        target_teleprompt=target,
        user_transcript=irrelevant_transcript,
        duration_seconds=5.0,
    )
    print("=== SCENARIO 2: IRRELEVANT / DISCONNECTED WORDS ===")
    print("Overall Score:", irrelevant_eval["overall_score"])
    print("Breakdown:", irrelevant_eval["score_breakdown"])
    print("Feedback:", irrelevant_eval["feedback"])
    assert irrelevant_eval["overall_score"] == 0
    assert irrelevant_eval["score_breakdown"]["word_choice"] == 0
    print(" Passed irrelevant test (0%)!\n")

    # 3. Partial Fragment (only 2 words)
    partial_transcript = "Our platform"
    partial_eval = await service._run_llm_evaluation(
        round_num=3,
        stage=stage,
        target_teleprompt=target,
        user_transcript=partial_transcript,
        duration_seconds=2.0,
    )
    print("=== SCENARIO 3: TINY PARTIAL FRAGMENT ===")
    print("Overall Score:", partial_eval["overall_score"])
    print("Breakdown:", partial_eval["score_breakdown"])
    assert partial_eval["overall_score"] < 15
    print(" Passed partial fragment test (<15%)!\n")

    # 4. Accurate Delivery
    accurate_transcript = "Our platform uses tenant-isolated storage, end-to-end encryption, and built-in SOC 2 Type II controls with real-time audit logs for GDPR compliance."
    accurate_eval = await service._run_llm_evaluation(
        round_num=3,
        stage=stage,
        target_teleprompt=target,
        user_transcript=accurate_transcript,
        duration_seconds=7.5,
    )
    print("=== SCENARIO 4: ACCURATE TELEPROMPT DELIVERY ===")
    print("Overall Score:", accurate_eval["overall_score"])
    print("Breakdown:", accurate_eval["score_breakdown"])
    assert accurate_eval["overall_score"] >= 85
    print(" Passed accurate delivery test (>=85%)!\n")

    print("ALL 4 SCENARIOS PASSED WITH EXACT STRICT EVALUATION!")

asyncio.run(test_zero_scoring())
