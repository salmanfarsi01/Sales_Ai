from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from pathlib import Path
from time import time

from aiohttp import web

from .config import Settings
from .twilio_context import ContextAwareTwilioCopilot

LOGGER = logging.getLogger("copilot.knowledge")
KNOWLEDGE = Path("knowledge")
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_CHARS = 2_000_000


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return (cleaned or "document")[:80]


def extract_pdf(data: bytes) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support is not installed. Run: python -m pip install pypdf") from exc
    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported")
    pages = []
    total = 0
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            block = f"## Page {number}\n\n{text}"
            total += len(block)
            if total > MAX_EXTRACTED_CHARS:
                raise ValueError("Extracted PDF text is too large")
            pages.append(block)
    if not pages:
        raise ValueError("No selectable text found; scanned PDFs require OCR")
    return "\n\n".join(pages), len(reader.pages)


class KnowledgeTwilioCopilot(ContextAwareTwilioCopilot):
    def app(self) -> web.Application:
        app = super().app()
        app.router.add_get("/knowledge", self.list_knowledge)
        app.router.add_post("/knowledge/upload", self.upload_pdf)
        return app

    async def dashboard(self, request: web.Request) -> web.FileResponse:
        static = Path(__file__).resolve().parent.parent / "web"
        return web.FileResponse(static / "twilio_knowledge.html")

    async def list_knowledge(self, request: web.Request) -> web.Response:
        documents = []
        if KNOWLEDGE.exists():
            for path in sorted(KNOWLEDGE.glob("*.pdf")):
                documents.append({"name": path.name, "bytes": path.stat().st_size})
        return web.json_response({"documents": documents, "indexed_chunks": len(self.knowledge._chunks)})

    async def upload_pdf(self, request: web.Request) -> web.Response:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file" or not field.filename:
            raise web.HTTPBadRequest(text="A PDF file is required")
        if Path(field.filename).suffix.casefold() != ".pdf":
            raise web.HTTPBadRequest(text="Only PDF files are accepted")
        data = bytearray()
        while True:
            chunk = await field.read_chunk(64 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_PDF_BYTES:
                raise web.HTTPRequestEntityTooLarge(max_size=MAX_PDF_BYTES, actual_size=len(data))
        if not bytes(data).startswith(b"%PDF"):
            raise web.HTTPBadRequest(text="The uploaded file is not a valid PDF")
        try:
            text, pages = await asyncio.to_thread(extract_pdf, bytes(data))
        except (ValueError, RuntimeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        KNOWLEDGE.mkdir(parents=True, exist_ok=True)
        name = f"{safe_stem(field.filename)}_{int(time())}"
        pdf_path = KNOWLEDGE / f"{name}.pdf"
        extracted_path = KNOWLEDGE / f"{name}.pdf.md"
        await asyncio.to_thread(pdf_path.write_bytes, bytes(data))
        await asyncio.to_thread(
            extracted_path.write_text,
            f"# Source: {field.filename}\n\n{text}",
            encoding="utf-8",
        )
        await asyncio.to_thread(self.knowledge.reload)
        event = {
            "type": "knowledge_uploaded", "name": pdf_path.name,
            "pages": pages, "characters": len(text), "indexed_chunks": len(self.knowledge._chunks),
        }
        await self.broadcast(event)
        LOGGER.info("indexed PDF %s pages=%d characters=%d", pdf_path.name, pages, len(text))
        return web.json_response(event)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    port = int(os.getenv("COPILOT_WEB_PORT", "8000"))
    LOGGER.info("dashboard http://127.0.0.1:%d", port)
    web.run_app(KnowledgeTwilioCopilot(Settings.from_env()).app(), host="127.0.0.1", port=port, print=None)
