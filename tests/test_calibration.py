"""Tests for the AI Voice Calibration system."""

import pytest
import asyncio
from copilot.calibration import CalibrationService, create_standalone_calibration_app


@pytest.mark.asyncio
async def test_calibration_session_lifecycle():
    service = CalibrationService()
    
    # 1. Create Dynamic 8-Round Session with Groq
    session = await service.create_dynamic_session(user_id="test_sales_rep", industry="B2B Enterprise AI")
    session_id = session["session_id"]
    assert session_id.startswith("calib_")
    assert session["current_round"] == 1
    assert session["status"] == "in_progress"
    assert session["total_rounds"] == 8

    # 2. Generate on-demand Round 1 ElevenLabs Voice
    round_1 = await service.generate_round_voice_on_demand(session_id, 1)
    assert round_1["round_number"] == 1
    assert "question_text" in round_1
    assert "teleprompt_text" in round_1
    assert round_1["audio_id"] is not None
    assert round_1["audio_url"] == f"/api/calibration/audio/{round_1['audio_id']}"

    # 3. Evaluate Round 1
    evaluated_round = await service.evaluate_round(
        session_id=session_id,
        round_num=1,
        transcript_override=round_1["teleprompt_text"],
        duration_seconds=5.2,
    )
    assert evaluated_round["completed"] is True
    eval_result = evaluated_round["evaluation"]
    assert eval_result is not None
    assert eval_result["overall_score"] >= 0
    assert "score_breakdown" in eval_result
    
    breakdown = eval_result["score_breakdown"]
    assert 0 <= breakdown["word_choice"] <= 100
    assert 0 <= breakdown["pacing"] <= 100
    assert 0 <= breakdown["sentiment"] <= 100
    assert 0 <= breakdown["tone_emphasis"] <= 100
    assert 0 <= breakdown["pause_filters"] <= 100
    assert 0 <= breakdown["energy_inflection"] <= 100

    # 4. Progress through remaining rounds (2 to 8)
    for r in range(2, 9):
        await service.generate_round_voice_on_demand(session_id, r)
        r_eval = await service.evaluate_round(
            session_id=session_id,
            round_num=r,
            transcript_override=session["rounds"][r]["teleprompt_text"],
            duration_seconds=4.8,
        )
        assert r_eval["completed"] is True

    # 5. Verify final summary report
    session_status = service.get_session(session_id)
    assert session_status["status"] == "completed"
    
    report = session_status["summary_report"]
    assert report is not None
    assert report["rounds_completed"] == 8
    assert report["total_rounds"] == 8
    assert 0 <= report["overall_score"] <= 100
    assert len(report["round_performance"]) == 8
    assert "word_choice" in report["score_breakdown"]
    assert "pacing" in report["score_breakdown"]
    assert "sentiment" in report["score_breakdown"]
    assert "tone_emphasis" in report["score_breakdown"]
    assert "pause_filters" in report["score_breakdown"]
    assert "energy_inflection" in report["score_breakdown"]
    assert len(report["what_this_means"]) == 4
    assert len(report["what_is_next"]) == 3


def test_standalone_app_initialization():
    app = create_standalone_calibration_app()
    assert app is not None
    assert app.title == "AI Voice Calibration Studio"
