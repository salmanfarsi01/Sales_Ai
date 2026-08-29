import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from copilot.call_report import REPORT_KEYS, CallReportGenerator, complete_report, validate_report


class FakeGroq:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
            )
        )


def sample_report() -> dict[str, object]:
    return {
        "call_summary": {"overview": "A productive call", "key_points": [], "summary_disclosure": ""},
        "call_status": {"status": "completed", "reason": "Both parties completed the discussion"},
        "outcome": {"category": "follow_up", "confidence": 0.9, "reason": "", "next_action": "", "follow_up": ""},
        "key_moments_log": [],
        "performance_metrics": {
            "call_duration_seconds": 12.0,
            "response_timing": {},
            "prompt_utilization": {},
            "coaching_insights": [],
        },
        "conversion_indicators": {},
        "agent_tone_delivery_feedback": {},
        "agent_sentiment_responsiveness": {},
    }


class CallReportTests(unittest.TestCase):
    def test_validates_required_sections(self):
        report = validate_report(sample_report())
        self.assertEqual(set(REPORT_KEYS), set(report))

    def test_completes_sections_omitted_by_model(self):
        report = complete_report({"call_summary": {"overview": "Short call"}})
        self.assertEqual(set(REPORT_KEYS), set(report))
        self.assertEqual(report["conversion_indicators"]["status"], "unknown")

    def test_generates_and_saves_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = CallReportGenerator(
                FakeGroq(json.dumps(sample_report())), "test-model", Path(directory)
            )
            report = asyncio.run(generator.generate(
                "CA123", [{"speaker": "client", "text": "Hello", "elapsed_seconds": 0.1}], 1.2
            ))
            self.assertEqual(report["call_id"], "CA123")
            self.assertEqual(report["call_summary"]["overview"], "A productive call")


if __name__ == "__main__":
    unittest.main()