import asyncio
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from copilot.calibration import CalibrationService


async def test():
    s = CalibrationService()
    session = await s.create_dynamic_session('sales_rep_1', 'Cybersecurity & Zero Trust Platform')
    session_id = session['session_id']
    print('Session ID:', session_id)
    print('Industry:', session['industry'])
    print('=' * 60)

    for r in range(1, 5):
        rd = await s.generate_round_voice_on_demand(session_id, r)
        print(f"ROUND {r} [{rd['stage']}] - Buyer: {rd['persona_name']} ({rd['persona_title']})")
        print(f"Voice: {rd['voice_name']} (ID: {rd['voice_id']})")
        print(f"Question:   {rd['question_text']}")
        print(f"Teleprompt: {rd['teleprompt_text']}")
        print('-' * 60)

asyncio.run(test())
