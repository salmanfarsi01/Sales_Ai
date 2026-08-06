from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

from .config import Settings
from .knowledge_upload import KnowledgeTwilioCopilot


class AllQuestionsCopilot(KnowledgeTwilioCopilot):
    """Ensure even very short final client questions trigger context-aware answers."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._client_history: dict[str, list[str]] = {}

    async def broadcast(self, event: dict[str, object]) -> None:
        await super().broadcast(event)
        event_type = event.get("type")
        call_id = str(event.get("call_id", "active"))
        if event_type == "call_start":
            self._client_history[call_id] = []
            return
        if event_type == "call_end":
            self._client_history.pop(call_id, None)
            return
        if (
            event_type == "transcript"
            and event.get("role") == "client"
            and event.get("final") is True
        ):
            text = str(event.get("text", "")).strip()
            history = self._client_history.setdefault(call_id, [])
            if text and len(text) < 8:
                context = [{"role": "user", "content": prior} for prior in history[-6:]]
                asyncio.create_task(self.stream_suggestion(call_id, text, context))
            if text:
                history.append(text)
                del history[:-12]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    port = int(os.getenv("COPILOT_WEB_PORT", "8000"))
    logging.getLogger("copilot.knowledge").info("dashboard http://127.0.0.1:%d", port)
    web.run_app(AllQuestionsCopilot(Settings.from_env()).app(), host="127.0.0.1", port=port, print=None)
