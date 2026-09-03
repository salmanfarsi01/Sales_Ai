import asyncio
import json
import os
from pathlib import Path
from copilot.calibration import CalibrationService


async def test_report_storage():
    service = CalibrationService()
    session = await service.create_dynamic_session('test_rep_1', 'Enterprise Cyber Defense')
    session_id = session['session_id']
    print('Testing session:', session_id)

    for r in range(1, 9):
        await service.generate_round_voice_on_demand(session_id, r)
        await service.evaluate_round(
            session_id=session_id,
            round_num=r,
            transcript_override="We provide automated compliance, end-to-end encryption, and dedicated engineering support with guaranteed 4x ROI.",
            duration_seconds=7.5,
        )

    # Check session status
    updated_session = service.get_session(session_id)
    assert updated_session['status'] == 'completed'
    assert updated_session['summary_report'] is not None
    report = updated_session['summary_report']

    # Verify files on disk
    reports_dir = Path(__file__).resolve().parent.parent / 'reports'
    session_file = reports_dir / f'calibration_{session_id}.json'
    latest_file = reports_dir / 'latest_calibration_report.json'

    print('Session file exists:', session_file.exists(), 'Path:', session_file)
    print('Latest file exists:', latest_file.exists(), 'Path:', latest_file)

    assert session_file.exists()
    assert latest_file.exists()

    content = json.loads(session_file.read_text(encoding='utf-8'))
    print('Loaded JSON Overall Score:', content['overall_score'])
    print('Loaded JSON Score Breakdown:', content['score_breakdown'])
    print('Loaded JSON Rounds Detail Count:', len(content['rounds_detail']))
    print('Successfully verified local JSON report storage!')

asyncio.run(test_report_storage())
