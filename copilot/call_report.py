from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_KEYS = (
    "call_summary",
    "call_status",
    "outcome",
    "key_moments_log",
    "performance_metrics",
    "conversion_indicators",
    "agent_tone_delivery_feedback",
    "agent_sentiment_responsiveness",
    "lead_stage",
    "lead_status",
    "lead_outcome",
)


def _json_text(text: str) -> str:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else cleaned


def validate_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("Call report must be a JSON object")
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"Call report is missing sections: {', '.join(missing)}")
    if not isinstance(report["key_moments_log"], list):
        raise ValueError("key_moments_log must be an array")
    for moment in report["key_moments_log"]:
        if not isinstance(moment, dict) or not isinstance(moment.get("timestamp"), (int, float)):
            raise ValueError("key_moments_log timestamps must be numbers in seconds")
    if not isinstance(report["performance_metrics"], dict):
        raise ValueError("performance_metrics must be an object")
    return report


def complete_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("Call report must be a JSON object")
    defaults: dict[str, Any] = {
        "call_summary": {"overview": "Not available", "key_points": [], "summary_disclosure": "Not available"},
        "call_status": {"status": "unknown", "reason": "Not available"},
        "outcome": {"category": "unknown", "confidence": 0, "reason": "Not available", "next_action": "Not available", "follow_up": "Not available"},
        "key_moments_log": [],
        "performance_metrics": {"call_duration_seconds": 0, "response_timing": "Not available", "prompt_utilization": "Not available", "coaching_insights": "Not available"},
        "conversion_indicators": {"status": "unknown", "evidence": "Not available", "strengths": [], "risks": []},
        "agent_tone_delivery_feedback": {"tone_alignment": "Not available", "energy_profile": "Not available", "strengths": [], "improvements": []},
        "agent_sentiment_responsiveness": {"sentiment_score": None, "adaptability_moments": [], "coaching_notes": "Not available"},
        "lead_stage": "Unknown",
        "lead_status": "Unknown",
        "lead_outcome": "Unknown",
    }
    for key, default in defaults.items():
        report.setdefault(key, default)
    return validate_report(report)


class CallReportGenerator:
    def __init__(self, client: Any, model: str, directory: str | Path = "reports"):
        self.client = client
        self.model = model
        self.directory = Path(directory)

    async def generate(
        self,
        call_sid: str,
        transcript: list[dict[str, Any]],
        duration_seconds: float,
    ) -> dict[str, Any]:
        conversation = "\n".join(
            f"[{item['elapsed_seconds']:.3f} seconds] {item['speaker']}: {item['text']}"
            for item in transcript
        )
        prompt = f"""Analyze this complete salesperson/client call and return only valid JSON.

The JSON must contain exactly these top-level sections:
1. call_summary: overview, key_points, summary_disclosure
2. call_status: status, reason
3. outcome: category, confidence, reason, next_action, follow_up
4. key_moments_log: array of timestamp, type, description, impact; timestamp must be a number in seconds from call start
5. performance_metrics: call_duration_seconds, response_timing, prompt_utilization, coaching_insights
6. conversion_indicators: status, evidence, strengths, risks
7. agent_tone_delivery_feedback: tone_alignment, energy_profile, strengths, improvements
8. agent_sentiment_responsiveness: sentiment_score, adaptability_moments, coaching_notes
9. lead_stage: Must be exactly one of: "New", "Attempted", "Contacted", "In Conversation", "Follow-Up", "Qualified", "Appointment Set", "Archived"
10. lead_status: Must be exactly one of: "Hot", "Warm", "Cold", "Unknown"
11. lead_outcome: Must be exactly one of: "Appointment Set", "Follow-Up Required", "Information Requested", "Not Interested", "Not a Fit", "No Answer", "Voicemail"

Use only evidence from the transcript. If a value cannot be known exactly, estimate it and state
that it is an estimate. Prompt utilization means how closely the salesperson followed the AI
teleprompts; if no teleprompt text is available, mark it as an estimate based on the transcript.
Use confidence between 0 and 1. Keep all narrative concise.

Measured call duration in seconds: {duration_seconds:.3f}
Complete transcript:
{conversation or '[No final transcript was captured.]'}
"""

        def request() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You produce strictly valid JSON call reports."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=1800,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or "{}"

        raw = await asyncio.to_thread(request)
        report = complete_report(json.loads(_json_text(raw)))
        report["call_sid"] = call_sid
        report["generated_at"] = datetime.now(timezone.utc).isoformat()

        # Save the JSON report to the configured directory
        self.directory.mkdir(parents=True, exist_ok=True)
        report_file = self.directory / f"{call_sid}.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return report