from __future__ import annotations

import logging
import os
from pathlib import Path

from aiohttp import web

from .config import Settings
from .twilio_fast import FastTwilioCopilot


class DiagnosticTwilioCopilot(FastTwilioCopilot):
    async def dashboard(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(Path(__file__).resolve().parent.parent / "web" / "twilio_fast.html")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    port = int(os.getenv("COPILOT_WEB_PORT", "8000"))
    logging.getLogger("copilot.twilio.fast").info("dashboard http://127.0.0.1:%d", port)
    web.run_app(DiagnosticTwilioCopilot(Settings.from_env()).app(), host="127.0.0.1", port=port, print=None)
