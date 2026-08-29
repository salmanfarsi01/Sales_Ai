"""FastAPI-based Twilio Sales Copilot backend server.

Handles Twilio call streaming, Deepgram STT, context recovery, Groq suggestion streaming,
local PDF parsing, and Pinecone RAG integration.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import suppress
from pathlib import Path
from time import monotonic, time

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from groq import Groq

from .config_rag import Settings
from .call_report import CallReportGenerator
from .retrieval import LocalKnowledgeBase
from .rag_integration import CopilotRAGRetriever
from .rag_api import RAGAPIHandler
from .knowledge_upload import (
    KNOWLEDGE,
    VALID_TARGETS,
    MAX_PDF_BYTES,
    MAX_EXTRACTED_CHARS,
    is_pdf_signature,
    extract_pdf,
    resolve_target,
    safe_stem,
)

LOGGER = logging.getLogger("copilot.fastapi")
STATIC = Path(__file__).resolve().parent.parent / "web"
STOP = object()


class FastAPICopilot:
    """Manages FastAPI router and lifecycle for the Twilio Sales Copilot."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.groq = Groq(api_key=settings.groq_api_key)
        self.call_reports = CallReportGenerator(self.groq, settings.llm_model)
        self.knowledge = LocalKnowledgeBase()
        self.dashboards: set[WebSocket] = set()
        self._client_history: dict[str, list[str]] = {}

        if settings.rag and settings.rag.rag_enabled:
            LOGGER.info("Initializing Pinecone RAG system")
            self.rag_retriever = CopilotRAGRetriever(
                pinecone_api_key=settings.rag.pinecone_api_key,
                openai_api_key=settings.rag.openai_api_key,
                admin_mode=settings.rag.admin_mode,
            )
            self.rag_api_handler = RAGAPIHandler(self.rag_retriever)
        else:
            LOGGER.info("RAG disabled, using local knowledge base")
            self.rag_retriever = None
            self.rag_api_handler = None

    async def broadcast(self, event: dict[str, object]) -> None:
        """Broadcast events to all connected dashboards."""
        stale = []
        for socket in list(self.dashboards):
            try:
                await socket.send_json(event)
            except Exception:
                stale.append(socket)
        for socket in stale:
            self.dashboards.discard(socket)

    async def component(self, name: str, state: str, detail: str = "") -> None:
        """Helper to broadcast component status."""
        await self.broadcast({"type": "component", "component": name, "state": state, "detail": detail})

    async def stream_suggestion(self, call_id: str, question: str, context: list[dict[str, str]]) -> None:
        """Retrieve context and stream Groq recommendation."""
        # Context recovery logic
        from .context_recovery import recover_query

        prior_client_turns = [
            message["content"] for message in context if message.get("role") == "user"
        ]
        reconstructed = recover_query(question, prior_client_turns)
        if reconstructed != question.strip():
            await self.broadcast({
                "type": "status",
                "message": f'Context recovered: "{question}" → "{reconstructed}"',
            })

        suggestion_id = str(uuid.uuid4())
        started = monotonic()

        # Retrieve knowledge context (using RAG if configured, fallback to local)
        if self.rag_retriever:
            tenant_id = self.settings.default_tenant_id
            context_list = self.rag_retriever.get_context(
                query=reconstructed,
                tenant_id=tenant_id,
                top_k=self.settings.rag.search_top_k,
                min_score=self.settings.rag.min_score_threshold,
            )
            evidence = "\n\n".join(context_list)
            # Reconstruct list of sources for suggestion UI
            sources = []
            for item in context_list:
                if "From " in item and ":\n" in item:
                    sources.append(item.split(":\n")[0].replace("From ", ""))
        else:
            matched = self.knowledge.search(reconstructed, limit=3)
            evidence = "\n\n".join(f"[{item.source}]\n{item.text}" for item in matched)
            sources = [item.source for item in matched]

        await self.broadcast({
            "type": "suggestion_start",
            "id": suggestion_id,
            "call_id": call_id,
            "question": reconstructed,
            "sources": sources,
        })

        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def generate() -> None:
            try:
                stream = self.groq.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a live sales copilot helping the salesperson answer the client. "
                            "Return at most three concise sentences with the recommended response only. "
                            "Answer as if you are advising the salesperson, not the client. "
                            "Use supplied knowledge whenever it matches the question. "
                            "If the knowledge contains a relevant fact, do not say you have no information. "
                            "Only admit that a fact is unavailable when the supplied knowledge truly does not contain it."
                        )},
                        *context[-self.settings.transcript_window:],
                        {
                            "role": "user",
                            "content": f"Client is asking: {reconstructed}\n\nKnowledge:\n{evidence or 'No matching local knowledge.'}"
                        },
                    ],
                    temperature=0.2,
                    max_completion_tokens=140,
                    stream=True,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        asyncio.create_task(asyncio.to_thread(generate))
        first_token_ms = None
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if first_token_ms is None:
                first_token_ms = int((monotonic() - started) * 1000)
            await self.broadcast({"type": "suggestion_delta", "id": suggestion_id, "text": item})

        await self.broadcast({
            "type": "suggestion_end",
            "id": suggestion_id,
            "ttft_ms": first_token_ms,
            "total_ms": int((monotonic() - started) * 1000),
        })

    def get_app(self) -> FastAPI:
        """Constructs and configures the FastAPI application router."""
        app = FastAPI(title="Twilio Sales Copilot", version="2.0.0")

        @app.get("/")
        async def dashboard():
            dashboard_file = STATIC / "twilio_knowledge.html"
            if not dashboard_file.exists():
                raise HTTPException(status_code=404, detail="Dashboard file not found")
            return FileResponse(dashboard_file)

        @app.get("/health")
        async def health():
            return {"status": "ok", "dashboards": len(self.dashboards)}

        @app.websocket("/events")
        async def dashboard_events(websocket: WebSocket):
            await websocket.accept()
            self.dashboards.add(websocket)
            await websocket.send_json({"type": "status", "message": "Dashboard connected; waiting for a Twilio call."})
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self.dashboards.discard(websocket)

        @app.websocket("/twilio")
        async def twilio_stream(websocket: WebSocket):
            await websocket.accept()
            call_id = str(uuid.uuid4())
            queue_size = max(256, self.settings.audio_queue_size)
            queues: dict[str, asyncio.Queue[bytes | object]] = {
                "salesperson": asyncio.Queue(queue_size),
                "client": asyncio.Queue(queue_size),
            }
            packets = {"salesperson": 0, "client": 0}
            dropped = {"salesperson": 0, "client": 0}
            context: list[dict[str, str]] = []
            full_transcript: list[dict[str, object]] = []
            call_started = monotonic()
            state: dict[str, object] = {
                "generation_task": None,
                "last_generation": 0.0,
                "last_client_question": "",
            }
            await self.component("twilio", "connected", "Media WebSocket accepted")
            LOGGER.info("Twilio connected %s", call_id)

            async def receive_twilio() -> None:
                nonlocal call_id
                try:
                    async for message in websocket.iter_text():
                        event = json.loads(message)
                        kind = event.get("event")
                        if kind == "connected":
                            await self.component("twilio", "connected", "Twilio protocol connected")
                        elif kind == "start":
                            start = event.get("start", {})
                            call_id = start.get("callSid", call_id)
                            await self.broadcast({"type": "call_start", "call_id": call_id})
                            await self.component("twilio", "streaming", f"Call {call_id}")
                        elif kind == "media":
                            media = event.get("media", {})
                            track = media.get("track")
                            role = "salesperson" if track == "inbound" else "client" if track == "outbound" else None
                            if role is None:
                                continue
                            chunk = base64.b64decode(media.get("payload", ""))
                            packets[role] += 1
                            try:
                                queues[role].put_nowait(chunk)
                            except asyncio.QueueFull:
                                dropped[role] += 1
                            if packets[role] == 1 or packets[role] % 50 == 0:
                                await self.component(
                                    f"audio_{role}", "receiving",
                                    f"{packets[role]} packets · queue {queues[role].qsize()} · dropped {dropped[role]}",
                                )
                        elif kind == "stop":
                            await self.component("twilio", "stopping", "Twilio sent stop")
                            await asyncio.gather(*(queue.put(STOP) for queue in queues.values()))
                            return
                except WebSocketDisconnect:
                    LOGGER.info("Twilio WebSocket disconnected")
                    await asyncio.gather(*(queue.put(STOP) for queue in queues.values()))

            async def stt(role: str) -> None:
                language = os.getenv("DEEPGRAM_LANGUAGE", "en")
                url = (
                    "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1"
                    f"&language={language}&interim_results=true&endpointing=120&vad_events=true"
                    "&smart_format=true&model=nova-3"
                )
                headers = {"Authorization": f"Token {self.settings.deepgram_api_key}"}
                await self.component(f"stt_{role}", "connecting", "Opening Deepgram stream")
                async with websockets.connect(url, additional_headers=headers, open_timeout=10) as deepgram:
                    await self.component(f"stt_{role}", "ready", "Deepgram connected")

                    async def send_audio() -> None:
                        while True:
                            chunk = await queues[role].get()
                            if chunk is STOP:
                                await deepgram.send(json.dumps({"type": "CloseStream"}))
                                return
                            await deepgram.send(chunk)

                    async def receive_text() -> None:
                        async for raw in deepgram:
                            event = json.loads(raw)
                            event_type = event.get("type")
                            if event_type == "Error":
                                raise RuntimeError(event.get("description", "Deepgram error"))
                            if event_type != "Results":
                                continue
                            text = event.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "").strip()
                            if not text:
                                continue
                            final = bool(event.get("is_final"))
                            speech_final = bool(event.get("speech_final"))
                            await self.component(f"stt_{role}", "transcribing", "Speech detected")
                            await self.broadcast({
                                "type": "transcript",
                                "call_id": call_id,
                                "role": role,
                                "text": text,
                                "final": final,
                            })
                            if final:
                                context.append({
                                    "role": "assistant" if role == "salesperson" else "user",
                                    "content": text,
                                })
                                del context[:-self.settings.transcript_window]
                                full_transcript.append({
                                    "speaker": role,
                                    "text": text,
                                    "elapsed_seconds": monotonic() - call_started,
                                })

                                # AllQuestionsCopilot functionality (short questions handling)
                                if role == "client":
                                    client_text = text.strip()
                                    client_history = self._client_history.setdefault(call_id, [])
                                    if client_text and len(client_text) < 8:
                                        hist_context = [{"role": "user", "content": prior} for prior in client_history[-6:]]
                                        asyncio.create_task(self.stream_suggestion(call_id, client_text, hist_context))
                                    if client_text:
                                        client_history.append(client_text)
                                        del client_history[:-12]

                            if role == "client" and len(text) >= 8 and (final or speech_final):
                                task = state.get("generation_task")
                                due = monotonic() - float(state["last_generation"]) >= 0.35
                                duplicate = text == state["last_client_question"]
                                can_start = task is None or task.done()
                                if task and not task.done() and (speech_final or final):
                                    task.cancel()
                                    can_start = True
                                if not duplicate and (due or final) and can_start:
                                    state["last_client_question"] = text
                                    state["last_generation"] = monotonic()
                                    prompt_context = list(context)
                                    if final and prompt_context and prompt_context[-1]["content"] == text:
                                        prompt_context.pop()
                                    await self.component("llm", "generating", "Client speech triggered suggestion")
                                    new_task = asyncio.create_task(
                                        self.stream_suggestion(call_id, text, prompt_context)
                                    )
                                    state["generation_task"] = new_task
                                    new_task.add_done_callback(
                                        lambda completed: asyncio.create_task(
                                            self.component("llm", "ready" if not completed.exception() else "error", "Suggestion complete")
                                        ) if not completed.cancelled() else None
                                    )

                    tasks = {asyncio.create_task(send_audio()), asyncio.create_task(receive_text())}
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

            tasks = {
                asyncio.create_task(receive_twilio()),
                asyncio.create_task(stt("salesperson")),
                asyncio.create_task(stt("client")),
            }
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            except Exception as exc:
                LOGGER.exception("call pipeline failed %s", call_id)
                await self.broadcast({"type": "error", "message": str(exc), "call_id": call_id})
                await self.component("pipeline", "error", str(exc))
            finally:
                generation_task = state.get("generation_task")
                if generation_task:
                    generation_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await generation_task
                await websocket.close()
                try:
                    report = await self.call_reports.generate(
                        call_id,
                        full_transcript,
                        monotonic() - call_started,
                    )
                    await self.broadcast({"type": "call_report", "call_id": call_id, "report": report})
                except Exception as exc:
                    LOGGER.exception("call report failed %s", call_id)
                    await self.broadcast({"type": "error", "message": f"Call report failed: {exc}", "call_id": call_id})
                await self.broadcast({
                    "type": "call_end",
                    "call_id": call_id,
                    "dropped_audio_chunks": dropped["salesperson"] + dropped["client"],
                })
                await self.component(
                    "twilio", "ended",
                    f"packets salesperson={packets['salesperson']}, client={packets['client']}; dropped={dropped}",
                )

        @app.get("/knowledge")
        async def list_knowledge():
            documents = []
            for target in sorted(VALID_TARGETS):
                target_dir = KNOWLEDGE / target
                if not target_dir.exists():
                    continue
                for path in sorted(target_dir.glob("*.pdf")):
                    documents.append({"name": path.name, "bytes": path.stat().st_size, "target": target})
            return {"documents": documents, "indexed_chunks": len(self.knowledge._chunks)}

        @app.post("/knowledge/upload")
        async def upload_pdf(
            file: UploadFile = File(...),
            target: str = Form("sales"),
        ):
            if not file.filename:
                raise HTTPException(status_code=400, detail="A PDF file is required")

            if Path(file.filename).suffix.casefold() != ".pdf":
                raise HTTPException(status_code=400, detail="Only PDF files are accepted")

            payload = await file.read()
            if len(payload) == 0:
                raise HTTPException(status_code=400, detail="Selected file is empty. Please choose a valid PDF file.")

            if not is_pdf_signature(payload):
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")

            try:
                resolved_target = resolve_target(target)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            try:
                text, pages = await asyncio.to_thread(extract_pdf, payload)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            target_dir = KNOWLEDGE / resolved_target
            target_dir.mkdir(parents=True, exist_ok=True)
            name = f"{safe_stem(file.filename)}_{int(time())}"
            pdf_path = target_dir / f"{name}.pdf"
            extracted_path = target_dir / f"{name}.pdf.md"

            await asyncio.to_thread(pdf_path.write_bytes, payload)
            await asyncio.to_thread(
                extracted_path.write_text,
                f"# Source: {file.filename}\n\n{text}",
                encoding="utf-8",
            )
            await asyncio.to_thread(self.knowledge.reload)

            # Upload to Pinecone if configured
            pinecone_key = os.getenv("PINECONE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            pinecone_success = False
            if pinecone_key and openai_key:
                try:
                    from .rag_pinecone import PineconeRAG
                    admin_mode = (resolved_target == "admin")
                    rag = PineconeRAG(
                        pinecone_api_key=pinecone_key,
                        openai_api_key=openai_key,
                        admin_mode=admin_mode,
                    )
                    await asyncio.to_thread(
                        rag.upload_file,
                        file_path=str(pdf_path),
                        tenant_id=resolved_target,
                        file_content=payload,
                    )
                    LOGGER.info("Successfully uploaded %s to Pinecone index %s (namespace: %s)", file.filename, rag.index_name, resolved_target)
                    pinecone_success = True
                except Exception as exc:
                    LOGGER.error("Failed to upload %s to Pinecone: %s", file.filename, exc)

            event = {
                "type": "knowledge_uploaded",
                "name": pdf_path.name,
                "target": resolved_target,
                "pages": pages,
                "characters": len(text),
                "indexed_chunks": len(self.knowledge._chunks),
                "pinecone_synced": pinecone_success,
            }
            await self.broadcast(event)
            LOGGER.info("indexed PDF %s target=%s pages=%d characters=%d pinecone_synced=%s", pdf_path.name, resolved_target, pages, len(text), pinecone_success)
            return event

        # Register standard Pinecone RAG APIs if RAG is enabled
        if self.rag_api_handler:
            @app.post("/api/rag/upload")
            async def upload_rag_file(
                file: UploadFile = File(...),
                tenant_id: str = Form("default"),
            ):
                file_data = await file.read()
                result = self.rag_retriever.upload_knowledge(
                    file_path=file.filename,
                    tenant_id=tenant_id,
                    file_content=file_data,
                )
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.delete("/api/rag/file/{file_name}")
            async def delete_rag_file(file_name: str, tenant_id: str = "default"):
                result = self.rag_retriever.delete_knowledge_file(
                    file_name=file_name,
                    tenant_id=tenant_id,
                )
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.get("/api/rag/files")
            async def list_rag_files(tenant_id: str = "default"):
                result = self.rag_retriever.list_knowledge_files(tenant_id)
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.delete("/api/rag/tenant/{tenant_id}")
            async def delete_rag_tenant(tenant_id: str):
                result = self.rag_retriever.delete_tenant(tenant_id)
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.get("/api/rag/status")
            async def get_rag_status_api():
                return {
                    "status": "ok",
                    "rag_enabled": True,
                    "admin_mode": self.rag_retriever.admin_mode,
                    "embedding_model": "text-embedding-3-small",
                    "pinecone_indexes": ["admin-kb", "subscriber-kb"],
                }

        return app


# Module-level instance creation
settings = Settings.from_env()
copilot = FastAPICopilot(settings)
app = copilot.get_app()
