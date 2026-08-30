"""FastAPI-based user sales backend server.

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
from typing import Optional, Any

import httpx
import jwt
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq

from .config_rag import Settings
from .call_report import CallReportGenerator
from .retrieval import LocalKnowledgeBase
from .rag_integration import CopilotRAGRetriever
from .rag_api import RAGAPIHandler
from .file_extraction import FileExtractor
from .knowledge_upload import (
    KNOWLEDGE,
    VALID_TARGETS,
    is_pdf_signature,
    resolve_target,
    safe_stem,
)

LOGGER = logging.getLogger("copilot.fastapi")
STATIC = Path(__file__).resolve().parent.parent / "web"
STOP = object()


class FastAPICopilot:
    """Manages FastAPI router and lifecycle for user sales."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.groq = Groq(api_key=settings.groq_api_key)
        self.call_reports = CallReportGenerator(self.groq, settings.llm_model)
        self.knowledge = LocalKnowledgeBase()
        self.dashboards: dict[str, list[dict[str, Any]]] = {}
        self._client_history: dict[str, list[str]] = {}
        self.ingestion_jobs: dict[str, dict[str, Any]] = {}

        if settings.rag and settings.rag.rag_enabled:
            LOGGER.info("Initializing Pinecone RAG system (shared index: %s)", settings.rag.pinecone_index_name)
            self.rag_retriever = CopilotRAGRetriever(
                pinecone_api_key=settings.rag.pinecone_api_key,
                openai_api_key=settings.rag.openai_api_key,
                pinecone_index_name=settings.rag.pinecone_index_name,
            )
            self.rag_api_handler = RAGAPIHandler(self.rag_retriever)
        else:
            LOGGER.info("RAG disabled, using local knowledge base")
            self.rag_retriever = None
            self.rag_api_handler = None

    async def broadcast(self, tenant_id: str, event: dict[str, object], call_sid: Optional[str] = None) -> None:
        """Broadcast events to connected dashboards filtered by tenant_id and call_sid."""
        connections = self.dashboards.get(tenant_id, [])
        stale = []
        for conn in list(connections):
            socket = conn["socket"]
            conn_call_sid = conn["call_sid"]
            # If client is subscribed to a specific call, filter out other calls.
            if conn_call_sid and call_sid and conn_call_sid != call_sid:
                continue
            try:
                await socket.send_json(event)
            except Exception:
                stale.append(conn)
        for conn in stale:
            if conn in connections:
                connections.remove(conn)

    async def component(self, tenant_id: str, name: str, state: str, detail: str = "", call_sid: Optional[str] = None) -> None:
        """Helper to broadcast component status in standard event envelop format."""
        await self.broadcast(tenant_id, {
            "type": "component",
            "data": {
                "component": name,
                "state": state,
                "detail": detail
            }
        }, call_sid=call_sid)

    async def send_to_laravel_webhook(self, event_type: str, data: dict[str, Any], tenant_id: str) -> None:
        """Post event payloads asynchronously to the Laravel webhook endpoint if configured."""
        webhook_url = self.settings.laravel_webhook_url
        if not webhook_url:
            LOGGER.debug("Laravel webhook not configured. Skipping event payload transmission.")
            return

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                payload = {
                    "event": event_type,
                    "tenant_id": tenant_id,
                    "timestamp": int(time()),
                    "data": data,
                }
                resp = await client.post(webhook_url, json=payload)
                if resp.status_code >= 400:
                    LOGGER.error("Laravel webhook returned status %d: %s", resp.status_code, resp.text)
                else:
                    LOGGER.info("Successfully posted event %s to Laravel webhook for tenant %s", event_type, tenant_id)
        except Exception as exc:
            LOGGER.error("Failed to POST to Laravel webhook for event %s: %s", event_type, exc)

    def verify_jwt_token(self, token: str, tenant_id: str, call_sid: Optional[str] = None) -> bool:
        """Verify the JWT token from Laravel for WebSockets authorization."""
        secret = self.settings.jwt_secret
        if not secret:
            LOGGER.warning("JWT_SECRET is not configured. WebSockets access authorization checks are bypassed.")
            return True
        try:
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            
            # Validate tenant matches
            token_tenant = payload.get("tenant_id")
            if token_tenant != tenant_id:
                LOGGER.error("JWT tenant_id mismatch: token=%s, query=%s", token_tenant, tenant_id)
                return False
            
            token_call = payload.get("call_sid")
            token_role = payload.get("role", "salesperson")
            
            if call_sid:
                # Salesperson is only authorized to monitor their specific call_sid
                if token_role == "salesperson":
                    if token_call != call_sid:
                        LOGGER.error("JWT salesperson call_sid mismatch: token=%s, query=%s", token_call, call_sid)
                        return False
                # Supervisors are authorized to monitor any call within the tenant
                elif token_role == "supervisor":
                    pass
            else:
                # Requesting a supervisor global stream (monitoring all calls)
                if token_role != "supervisor":
                    LOGGER.error("JWT user role '%s' is not authorized to stream global events (supervisor role required)", token_role)
                    return False
            
            return True
        except jwt.ExpiredSignatureError:
            LOGGER.error("JWT connection token has expired")
            return False
        except jwt.InvalidTokenError as exc:
            LOGGER.error("JWT connection token verification failed: %s", exc)
            return False

    async def _process_ingestion_background(
        self,
        doc_id: str,
        filename: str,
        payload: bytes,
        target: str,
        tenant_id: str,
        salesman_id: Optional[str]
    ) -> None:
        """Background worker to extract, index, and sync PDF upload progress."""
        try:
            job = self.ingestion_jobs[doc_id]
            job["status"] = "processing"
            job["progress"] = 10
            
            try:
                resolved_target = resolve_target(target)
            except ValueError as exc:
                job["status"] = "failed"
                job["progress"] = 100
                job["error"] = str(exc)
                return
            
            job["progress"] = 20
            
            try:
                # Use multi-format FileExtractor
                extracted = await asyncio.to_thread(FileExtractor.extract, filename, payload)
                text = extracted.text
                pages = int(extracted.metadata.get("pages", "1"))
            except Exception as exc:
                job["status"] = "failed"
                job["progress"] = 100
                job["error"] = f"Extraction failed: {exc}"
                return
            
            job["progress"] = 40
            job["pages"] = pages
            job["characters"] = len(text)
            
            target_dir = KNOWLEDGE / tenant_id / resolved_target
            target_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(filename).suffix.lower()
            name = f"{safe_stem(filename)}_{int(time())}"
            dest_path = target_dir / f"{name}{suffix}"
            extracted_path = target_dir / f"{name}{suffix}.md"

            await asyncio.to_thread(dest_path.write_bytes, payload)
            await asyncio.to_thread(
                extracted_path.write_text,
                f"# Source: {filename}\n\n{text}",
                encoding="utf-8",
            )
            await asyncio.to_thread(self.knowledge.reload)
            
            job["progress"] = 50
            
            pinecone_key = os.getenv("PINECONE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            pinecone_success = False
            
            if pinecone_key and openai_key and self.rag_retriever:
                job["progress"] = 60
                
                def progress_cb(uploaded: int, total: int):
                    job["chunks_total"] = total
                    job["chunks_uploaded"] = uploaded
                    ratio = uploaded / max(1, total)
                    job["progress"] = int(60 + 40 * ratio)
                
                try:
                    await asyncio.to_thread(
                        self.rag_retriever.upload_knowledge,
                        file_path=str(dest_path),
                        tenant_id=tenant_id,
                        file_content=payload,
                        scope=resolved_target,
                        owner_id=salesman_id,
                        progress_callback=progress_cb,
                    )
                    pinecone_success = True
                except Exception as exc:
                    LOGGER.error("Failed to upload %s to Pinecone: %s", filename, exc)
                    job["status"] = "failed"
                    job["progress"] = 100
                    job["error"] = f"Pinecone sync failed: {exc}"
                    return
            
            job["progress"] = 100
            job["status"] = "completed"
            
            # Broadcast the completed event to visual channel (tenant-wide, no call_sid)
            event = {
                "type": "knowledge_uploaded",
                "data": {
                    "doc_id": doc_id,
                    "name": dest_path.name,
                    "target": resolved_target,
                    "pages": pages,
                    "characters": len(text),
                    "indexed_chunks": len(self.knowledge._chunks),
                    "pinecone_synced": pinecone_success,
                }
            }
            await self.broadcast(tenant_id, event)
            LOGGER.info("indexed File %s tenant=%s target=%s pages=%d characters=%d pinecone_synced=%s", dest_path.name, tenant_id, resolved_target, pages, len(text), pinecone_success)
            
        except Exception as exc:
            LOGGER.exception("Unhandled error in background ingestion task %s", doc_id)
            if doc_id in self.ingestion_jobs:
                self.ingestion_jobs[doc_id]["status"] = "failed"
                self.ingestion_jobs[doc_id]["progress"] = 100
                self.ingestion_jobs[doc_id]["error"] = str(exc)

    async def stream_suggestion(
        self,
        call_sid: str,
        question: str,
        context: list[dict[str, str]],
        tenant_id: str,
        scope: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> None:
        """Retrieve context and stream Groq recommendation."""
        from .context_recovery import recover_query

        prior_client_turns = [
            message["content"] for message in context if message.get("role") == "user"
        ]
        reconstructed = recover_query(question, prior_client_turns)
        if reconstructed != question.strip():
            await self.broadcast(tenant_id, {
                "type": "status",
                "data": {
                    "message": f'Context recovered: "{question}" → "{reconstructed}"'
                }
            }, call_sid=call_sid)

        suggestion_id = str(uuid.uuid4())
        started = monotonic()

        if self.rag_retriever:
            context_list = self.rag_retriever.get_context(
                query=reconstructed,
                tenant_id=tenant_id,
                top_k=self.settings.rag.search_top_k,
                min_score=self.settings.rag.min_score_threshold,
                scope=scope,
                owner_id=owner_id,
            )
            evidence = "\n\n".join(context_list)
            sources = []
            for item in context_list:
                if "From " in item and ":\n" in item:
                    sources.append(item.split(":\n")[0].replace("From ", ""))
        else:
            matched = self.knowledge.search(reconstructed, limit=3, tenant_id=tenant_id)
            evidence = "\n\n".join(f"[{item.source}]\n{item.text}" for item in matched)
            sources = [item.source for item in matched]

        await self.broadcast(tenant_id, {
            "type": "suggestion_start",
            "data": {
                "id": suggestion_id,
                "call_sid": call_sid,
                "question": reconstructed,
                "sources": sources,
            }
        }, call_sid=call_sid)

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
        full_suggestion = ""
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if first_token_ms is None:
                first_token_ms = int((monotonic() - started) * 1000)
            await self.broadcast(tenant_id, {
                "type": "suggestion_delta",
                "data": {
                    "id": suggestion_id,
                    "text": item
                }
            }, call_sid=call_sid)
            full_suggestion += item

        # Stream suggestion completed event
        await self.broadcast(tenant_id, {
            "type": "suggestion_end",
            "data": {
                "id": suggestion_id,
                "ttft_ms": first_token_ms,
                "total_ms": int((monotonic() - started) * 1000),
            }
        }, call_sid=call_sid)

        # Emit log suggestions record for delivery scoring
        suggestion_log = {
            "call_sid": call_sid,
            "suggestion_text": full_suggestion,
            "timestamp": int(time()),
        }
        await self.broadcast(tenant_id, {
            "type": "suggestion_log",
            "data": suggestion_log
        }, call_sid=call_sid)
        asyncio.create_task(self.send_to_laravel_webhook("suggestion_logged", suggestion_log, tenant_id))

    def get_app(self) -> FastAPI:
        """Constructs and configures the FastAPI application router."""
        app = FastAPI(title="user sales", version="2.0.0")

        # Standardized Error Handler for HTTPException
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "error_code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                }
            )

        # Standardized Error Handler for General Server Exceptions
        @app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            LOGGER.exception("Unhandled server error: %s", exc)
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": str(exc),
                }
            )

        @app.get("/")
        async def dashboard():
            dashboard_file = STATIC / "twilio_knowledge.html"
            if not dashboard_file.exists():
                raise HTTPException(status_code=404, detail="Dashboard file not found")
            return FileResponse(dashboard_file)

        @app.get("/health")
        async def health():
            total_dashboards = sum(len(conns) for conns in self.dashboards.values())
            return {"status": "ok", "dashboards": total_dashboards}

        @app.websocket("/events")
        async def dashboard_events(websocket: WebSocket):
            tenant_id = websocket.query_params.get("tenant_id")
            call_sid = websocket.query_params.get("call_sid")
            token = websocket.query_params.get("token")
            
            if not tenant_id:
                if self.settings.copilot_env == "production":
                    await websocket.close(code=4000, reason="Missing required query parameter: tenant_id")
                    return
                else:
                    tenant_id = self.settings.default_tenant_id
            
            # Authenticate the connection via signed JWT token
            if self.settings.jwt_secret:
                if not token:
                    await websocket.close(code=4001, reason="Missing WebSockets authorization token")
                    return
                if not self.verify_jwt_token(token, tenant_id, call_sid):
                    await websocket.close(code=4003, reason="Unauthorized connection request")
                    return

            await websocket.accept()
            conn = {"socket": websocket, "call_sid": call_sid}
            self.dashboards.setdefault(tenant_id, []).append(conn)
            
            scope_msg = f"for tenant {tenant_id}"
            if call_sid:
                scope_msg += f" and call {call_sid}"
                
            await websocket.send_json({
                "type": "status",
                "data": {
                    "message": f"Dashboard connected {scope_msg}; waiting for call."
                }
            })
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                conns = self.dashboards.get(tenant_id, [])
                if conn in conns:
                    conns.remove(conn)

        @app.websocket("/twilio")
        async def twilio_stream(websocket: WebSocket):
            tenant_id = websocket.query_params.get("tenant_id")
            salesman_id = websocket.query_params.get("salesman_id")
            call_sid_param = websocket.query_params.get("call_sid")

            if not tenant_id:
                if self.settings.copilot_env == "production":
                    await websocket.close(code=4000, reason="Missing required parameter: tenant_id")
                    return
                else:
                    tenant_id = self.settings.default_tenant_id

            await websocket.accept()
            call_sid = call_sid_param or str(uuid.uuid4())
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
            await self.component(tenant_id, "twilio", "connected", "Media WebSocket accepted", call_sid=call_sid)
            LOGGER.info("Twilio connected. tenant_id=%s, call_sid=%s", tenant_id, call_sid)

            async def receive_twilio() -> None:
                nonlocal call_sid
                try:
                    async for message in websocket.iter_text():
                        event = json.loads(message)
                        kind = event.get("event")
                        if kind == "connected":
                            await self.component(tenant_id, "twilio", "connected", "Twilio protocol connected", call_sid=call_sid)
                        elif kind == "start":
                            start = event.get("start", {})
                            call_sid = call_sid_param or start.get("callSid", call_sid)
                            await self.broadcast(tenant_id, {
                                "type": "call_start",
                                "data": {
                                    "call_sid": call_sid
                                }
                            }, call_sid=call_sid)
                            await self.component(tenant_id, "twilio", "streaming", f"Call {call_sid}", call_sid=call_sid)
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
                                    tenant_id,
                                    f"audio_{role}", "receiving",
                                    f"{packets[role]} packets · queue {queues[role].qsize()} · dropped {dropped[role]}",
                                    call_sid=call_sid,
                                )
                        elif kind == "stop":
                            await self.component(tenant_id, "twilio", "stopping", "Twilio sent stop", call_sid=call_sid)
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
                await self.component(tenant_id, f"stt_{role}", "connecting", "Opening Deepgram stream", call_sid=call_sid)
                async with websockets.connect(url, additional_headers=headers, open_timeout=10) as deepgram:
                    await self.component(tenant_id, f"stt_{role}", "ready", "Deepgram connected", call_sid=call_sid)

                    async def send_audio() -> None:
                        while True:
                            try:
                                chunk = await asyncio.wait_for(queues[role].get(), timeout=3.0)
                                if chunk is STOP:
                                    await deepgram.send(json.dumps({"type": "CloseStream"}))
                                    return
                                await deepgram.send(chunk)
                            except asyncio.TimeoutError:
                                try:
                                    await deepgram.send(json.dumps({"type": "KeepAlive"}))
                                except Exception as exc:
                                    LOGGER.warning("Failed to send KeepAlive to Deepgram for %s: %s", role, exc)
                                    return

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
                            await self.component(tenant_id, f"stt_{role}", "transcribing", "Speech detected", call_sid=call_sid)
                            await self.broadcast(tenant_id, {
                                "type": "transcript",
                                "data": {
                                    "call_sid": call_sid,
                                    "role": role,
                                    "text": text,
                                    "final": final,
                                }
                            }, call_sid=call_sid)
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

                                # Context recovery for short queries
                                if role == "client":
                                    client_text = text.strip()
                                    client_history = self._client_history.setdefault(call_sid, [])
                                    if client_text and len(client_text) < 8:
                                        hist_context = [{"role": "user", "content": prior} for prior in client_history[-6:]]
                                        asyncio.create_task(
                                            self.stream_suggestion(
                                                call_sid, client_text, hist_context, tenant_id,
                                                scope="sales", owner_id=salesman_id
                                            )
                                        )
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
                                    await self.component(tenant_id, "llm", "generating", "Client speech triggered suggestion", call_sid=call_sid)
                                    new_task = asyncio.create_task(
                                        self.stream_suggestion(
                                            call_sid, text, prompt_context, tenant_id,
                                            scope="sales", owner_id=salesman_id
                                        )
                                    )
                                    state["generation_task"] = new_task
                                    new_task.add_done_callback(
                                        lambda completed: asyncio.create_task(
                                            self.component(
                                                tenant_id, "llm",
                                                "ready" if not completed.exception() else "error",
                                                "Suggestion complete",
                                                call_sid=call_sid
                                            )
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
                LOGGER.exception("call pipeline failed %s", call_sid)
                await self.broadcast(tenant_id, {
                    "type": "error",
                    "data": {
                        "call_sid": call_sid,
                        "message": str(exc),
                    }
                }, call_sid=call_sid)
                await self.component(tenant_id, "pipeline", "error", str(exc), call_sid=call_sid)
            finally:
                generation_task = state.get("generation_task")
                if generation_task:
                    generation_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await generation_task
                with suppress(Exception):
                    await websocket.close()
                try:
                    report = await self.call_reports.generate(
                        call_sid,
                        full_transcript,
                        monotonic() - call_started,
                    )
                    await self.broadcast(tenant_id, {
                        "type": "call_report",
                        "data": {
                            "call_sid": call_sid,
                            "report": report
                        }
                    }, call_sid=call_sid)
                    
                    # POST report payload directly to Laravel Webhook
                    report_data = {
                        "call_sid": call_sid,
                        "report": report,
                    }
                    asyncio.create_task(self.send_to_laravel_webhook("call_report_generated", report_data, tenant_id))
                except Exception as exc:
                    LOGGER.exception("call report failed %s", call_sid)
                    await self.broadcast(tenant_id, {
                        "type": "error",
                        "data": {
                            "call_sid": call_sid,
                            "message": f"Call report failed: {exc}",
                        }
                    }, call_sid=call_sid)
                
                await self.broadcast(tenant_id, {
                    "type": "call_end",
                    "data": {
                        "call_sid": call_sid,
                        "dropped_audio_chunks": dropped["salesperson"] + dropped["client"],
                    }
                }, call_sid=call_sid)
                await self.component(
                    tenant_id, "twilio", "ended",
                    f"packets salesperson={packets['salesperson']}, client={packets['client']}; dropped={dropped}",
                    call_sid=call_sid
                )

        @app.get("/knowledge")
        async def list_knowledge(tenant_id: str):
            if not tenant_id:
                raise HTTPException(status_code=400, detail="Missing required query parameter: tenant_id")
            
            documents = []
            for target in sorted(VALID_TARGETS):
                target_dir = KNOWLEDGE / tenant_id / target
                if not target_dir.exists():
                    continue
                for path in sorted(target_dir.iterdir()):
                    if path.is_file() and not path.name.endswith(".md"):
                        documents.append({"name": path.name, "bytes": path.stat().st_size, "target": target})
            return {"documents": documents, "indexed_chunks": len(self.knowledge._chunks)}

        @app.post("/knowledge/upload")
        async def upload_pdf(
            file: UploadFile = File(...),
            target: str = Form("sales"),
            tenant_id: str = Form(...),
            salesman_id: Optional[str] = Form(None),
        ):
            if not tenant_id:
                raise HTTPException(status_code=400, detail="tenant_id form parameter is required")
            
            if not file.filename:
                raise HTTPException(status_code=400, detail="A file is required")

            suffix = Path(file.filename).suffix.lower()
            if suffix not in FileExtractor.SUPPORTED_FORMATS:
                supported_str = ", ".join(FileExtractor.SUPPORTED_FORMATS.keys())
                raise HTTPException(status_code=400, detail=f"Unsupported file format '{suffix}'. Supported formats: {supported_str}")

            payload = await file.read()
            if len(payload) == 0:
                raise HTTPException(status_code=400, detail="Selected file is empty.")

            if suffix == ".pdf" and not is_pdf_signature(payload):
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")

            doc_id = str(uuid.uuid4())
            self.ingestion_jobs[doc_id] = {
                "doc_id": doc_id,
                "filename": file.filename,
                "status": "pending",
                "progress": 0,
                "pages": 0,
                "characters": 0,
                "chunks_total": 0,
                "chunks_uploaded": 0,
                "error": None,
            }
            
            asyncio.create_task(
                self._process_ingestion_background(
                    doc_id=doc_id,
                    filename=file.filename,
                    payload=payload,
                    target=target,
                    tenant_id=tenant_id,
                    salesman_id=salesman_id,
                )
            )
            
            return JSONResponse(
                status_code=202,
                content={
                    "doc_id": doc_id,
                    "status": "pending",
                    "progress": 0,
                    "message": "Ingestion job started in the background",
                }
            )

        @app.get("/knowledge/status/{doc_id}")
        async def check_ingestion_status(doc_id: str):
            job = self.ingestion_jobs.get(doc_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Ingestion job not found for ID: {doc_id}")
            return job

        # Register standard Pinecone RAG APIs if RAG is enabled
        if self.rag_api_handler:
            @app.post("/api/rag/upload")
            async def upload_rag_file(
                file: UploadFile = File(...),
                tenant_id: str = Form(...),
                scope: str = Form("sales"),
                owner_id: Optional[str] = Form(None),
            ):
                if not tenant_id:
                    raise HTTPException(status_code=400, detail="tenant_id form parameter is required")
                file_data = await file.read()
                result = self.rag_retriever.upload_knowledge(
                    file_path=file.filename,
                    tenant_id=tenant_id,
                    file_content=file_data,
                    scope=scope,
                    owner_id=owner_id,
                )
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.delete("/api/rag/file/{file_name}")
            async def delete_rag_file(file_name: str, tenant_id: str):
                if not tenant_id:
                    raise HTTPException(status_code=400, detail="tenant_id query parameter is required")
                result = self.rag_retriever.delete_knowledge_file(
                    file_name=file_name,
                    tenant_id=tenant_id,
                )
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.get("/api/rag/files")
            async def list_rag_files(tenant_id: str):
                if not tenant_id:
                    raise HTTPException(status_code=400, detail="tenant_id query parameter is required")
                result = self.rag_retriever.list_knowledge_files(tenant_id)
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.delete("/api/rag/tenant/{tenant_id}")
            async def delete_rag_tenant(tenant_id: str):
                if not tenant_id:
                    raise HTTPException(status_code=400, detail="tenant_id path parameter is required")
                result = self.rag_retriever.delete_tenant(tenant_id)
                if result.get("status") == "error":
                    raise HTTPException(status_code=400, detail=result.get("error"))
                return result

            @app.get("/api/rag/status")
            async def get_rag_status_api():
                return {
                    "status": "ok",
                    "rag_enabled": True,
                    "admin_mode": False,
                    "embedding_model": "text-embedding-3-small",
                    "pinecone_indexes": [self.settings.rag.pinecone_index_name],
                }

        return app


# Module-level instance creation
settings = Settings.from_env()
copilot = FastAPICopilot(settings)
app = copilot.get_app()
