import asyncio
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from copilot.calibration import CalibrationService


async def test_strict_eval():
    service = CalibrationService()
    stage = "Security & Compliance"
    target = "Our platform uses tenant-isolated storage, end-to-end encryption, and built-in SOC 2 Type II controls with real-time audit logs for GDPR."

    # Test Case 1: Sloppy / Partial / Fillers
    sloppy_transcript = "Um, yeah, so our platform has like encryption and stuff, and uh, we have SOC 2."
    sloppy_eval = await service._run_llm_evaluation(
        round_num=2,
        stage=stage,
        target_teleprompt=target,
        user_transcript=sloppy_transcript,
        duration_seconds=3.0,
    )
    print("=== TEST CASE 1: SLOPPY / FILLERS / INCOMPLETE ===")
    print("Overall Score:", sloppy_eval["overall_score"])
    print("Breakdown:", sloppy_eval["score_breakdown"])
    print("Detected Fillers:", sloppy_eval["detected_fillers"])
    print("Feedback:", sloppy_eval["feedback"])
    print()

    # Test Case 2: Near Perfect Enterprise Delivery
    perfect_transcript = "Our platform uses tenant-isolated storage, end-to-end encryption, and built-in SOC 2 Type II controls with real-time audit logs for GDPR compliance."
    perfect_eval = await service._run_llm_evaluation(
        round_num=2,
        stage=stage,
        target_teleprompt=target,
        user_transcript=perfect_transcript,
        duration_seconds=8.0,
    )
    print("=== TEST CASE 2: HIGH PRECISION ENTERPRISE DELIVERY ===")
    print("Overall Score:", perfect_eval["overall_score"])
    print("Breakdown:", perfect_eval["score_breakdown"])
    print("Feedback:", perfect_eval["feedback"])

asyncio.run(test_strict_eval())
