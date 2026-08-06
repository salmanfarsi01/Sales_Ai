from __future__ import annotations

import logging
import os

from aiohttp import web

from .config import Settings
from .context_recovery import recover_query
from .twilio_diagnostic import DiagnosticTwilioCopilot


class ContextAwareTwilioCopilot(DiagnosticTwilioCopilot):
    async def stream_suggestion(self, call_id, question, context) -> None:
        prior_client_turns = [
            message["content"] for message in context if message.get("role") == "user"
        ]
        reconstructed = recover_query(question, prior_client_turns)
        if reconstructed != question.strip():
            await self.broadcast({
                "type": "status",
                "message": f'Context recovered: "{question}" → "{reconstructed}"',
            })
        await super().stream_suggestion(call_id, reconstructed, context)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    port = int(os.getenv("COPILOT_WEB_PORT", "8000"))
    logging.getLogger("copilot.twilio.context").info("dashboard http://127.0.0.1:%d", port)
    web.run_app(ContextAwareTwilioCopilot(Settings.from_env()).app(), host="127.0.0.1", port=port, print=None)
