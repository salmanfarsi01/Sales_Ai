from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import suppress
from pathlib import Path
from time import monotonic

import websockets
from aiohttp import WSMsgType, web
from groq import Groq

from .call_report import CallReportGenerator
from .config import Settings
from .retrieval import LocalKnowledgeBase

LOGGER = logging.getLogger("copilot.twilio")
STATIC = Path(__file__).resolve().parent.parent / "web"
STOP = object()


class TwilioCopilot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.groq = Groq(api_key=settings.groq_api_key)
        self.call_reports = CallReportGenerator(self.groq, settings.llm_model)
        self.knowledge = LocalKnowledgeBase()
        self.dashboards: set[web.WebSocketResponse] = set()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.dashboard)
        app.router.add_get("/events", self.dashboard_events)
        app.router.add_get("/twilio", self.twilio_stream)
        app.router.add_get("/health", self.health)
        return app

    async def dashboard(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "twilio.html")

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "dashboards": len(self.dashboards)})

    async def dashboard_events(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        self.dashboards.add(socket)
        await socket.send_json({"type": "status", "message": "Dashboard connected; waiting for a Twilio call."})
        try:
            async for _ in socket:
                pass
        finally:
            self.dashboards.discard(socket)
        return socket

    async def broadcast(self, event: dict[str, object]) -> None:
        stale = []
        for socket in tuple(self.dashboards):
            if socket.closed:
                stale.append(socket)
                continue
            try:
                await socket.send_json(event)
            except (ConnectionError, RuntimeError):
                stale.append(socket)
        for socket in stale:
            self.dashboards.discard(socket)

    async def twilio_stream(self, request: web.Request) -> web.WebSocketResponse:
        twilio = web.WebSocketResponse(heartbeat=20, max_msg_size=2 * 1024 * 1024)
        await twilio.prepare(request)
        call_id = str(uuid.uuid4())
        audio: asyncio.Queue[bytes | object] = asyncio.Queue(self.settings.audio_queue_size)
        context: list[dict[str, str]] = []
        generation_task: asyncio.Task[None] | None = None
        last_generation = 0.0
        dropped = 0
        url = (
            "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000"
            "&channels=2&multichannel=true&interim_results=true&endpointing=150"
            "&vad_events=true&smart_format=true&model=nova-3"
        )
        headers = {"Authorization": f"Token {self.settings.deepgram_api_key}"}
        await self.broadcast({"type": "status", "message": "Twilio connected; opening speech recognition."})
        LOGGER.info("Twilio media connection opened %s", call_id)
        try:
            async with websockets.connect(url, additional_headers=headers) as deepgram:
                async def audio_sender() -> None:
                    while True:
                        chunk = await audio.get()
                        if chunk is STOP:
                            await deepgram.send(json.dumps({"type": "CloseStream"}))
                            return
                        await deepgram.send(chunk)

                async def twilio_receiver() -> None:
                    nonlocal call_id, dropped
                    frames: dict[str, dict[int, bytes]] = {"inbound": {}, "outbound": {}}
                    async for message in twilio:
                        if message.type != WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        kind = event.get("event")
                        if kind == "start":
                            call_id = event.get("start", {}).get("callSid", call_id)
                            await self.broadcast({"type": "call_start", "call_id": call_id})
                        elif kind == "media":
                            media = event["media"]
                            track = media.get("track")
                            if track not in frames:
                                continue
                            frames[track][int(media["timestamp"])] = base64.b64decode(media["payload"])
                            for chunk in self.stereo_chunks(frames):
                                try:
                                    audio.put_nowait(chunk)
                                except asyncio.QueueFull:
                                    dropped += 1
                        elif kind == "stop":
                            await audio.put(STOP)
                            return

                async def transcript_receiver() -> None:
                    nonlocal generation_task, last_generation
                    async for raw in deepgram:
                        event = json.loads(raw)
                        if event.get("type") != "Results":
                            continue
                        text = event.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "").strip()
                        if not text:
                            continue
                        channel = event.get("channel_index", [0])[0]
                        # For <Start track=both_tracks> before <Dial>, inbound is the salesperson
                        # talking into Twilio; outbound is the remote client's audio sent to them.
                        role = "salesperson" if channel == 0 else "client"
                        final = bool(event.get("is_final"))
                        speech_final = bool(event.get("speech_final"))
                        await self.broadcast({
                            "type": "transcript", "call_id": call_id, "role": role,
                            "text": text, "final": final,
                        })
                        if final:
                            context.append({
                                "role": "assistant" if role == "salesperson" else "user",
                                "content": text,
                            })
                            del context[:-self.settings.transcript_window]
                        if role != "client" or len(text) < 12:
                            continue
                        due = monotonic() - last_generation >= self.settings.suggestion_interval_ms / 1000
                        if speech_final or due:
                            last_generation = monotonic()
                            if generation_task:
                                generation_task.cancel()
                            prompt_context = list(context)
                            if final and prompt_context and prompt_context[-1]["content"] == text:
                                prompt_context = prompt_context[:-1]
                            generation_task = asyncio.create_task(
                                self.stream_suggestion(call_id, text, prompt_context)
                            )

                tasks = {
                    asyncio.create_task(audio_sender()),
                    asyncio.create_task(twilio_receiver()),
                    asyncio.create_task(transcript_receiver()),
                }
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:
            LOGGER.exception("call failed %s", call_id)
            await self.broadcast({"type": "error", "message": str(exc), "call_id": call_id})
        finally:
            if generation_task:
                generation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await generation_task
            await twilio.close()
            await self.broadcast({"type": "call_end", "call_id": call_id, "dropped_audio_chunks": dropped})
            LOGGER.info("Twilio media connection closed %s; dropped=%d", call_id, dropped)
        return twilio

    @staticmethod
    def stereo_chunks(frames: dict[str, dict[int, bytes]]) -> list[bytes]:
        chunks = []
        common = sorted(frames["inbound"].keys() & frames["outbound"].keys())
        for timestamp in common:
            inbound = frames["inbound"].pop(timestamp)
            outbound = frames["outbound"].pop(timestamp)
            chunks.append(bytes(value for pair in zip(inbound, outbound) for value in pair))
        # Prevent an absent/silent track from causing unbounded memory growth.
        all_times = sorted(frames["inbound"].keys() | frames["outbound"].keys())
        if len(all_times) > 25:
            cutoff = all_times[-25]
            for timestamp in [stamp for stamp in all_times if stamp < cutoff]:
                inbound = frames["inbound"].pop(timestamp, b"\xff" * 160)
                outbound = frames["outbound"].pop(timestamp, b"\xff" * 160)
                chunks.append(bytes(value for pair in zip(inbound, outbound) for value in pair))
        return chunks

    async def stream_suggestion(self, call_id, question, context) -> None:
        suggestion_id = str(uuid.uuid4())
        started = monotonic()
        sources = self.knowledge.search(question)
        evidence = "\n\n".join(f"[{item.source}]\n{item.text}" for item in sources)
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
                        {"role": "user", "content": f"Client is asking: {question}\n\nKnowledge:\n{evidence or 'No matching local knowledge.'}"},
                    ],
                    temperature=0.2, max_completion_tokens=140, stream=True,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        await self.broadcast({
            "type": "suggestion_start", "id": suggestion_id, "call_id": call_id,
            "question": question, "sources": [item.source for item in sources],
        })
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
            "type": "suggestion_end", "id": suggestion_id,
            "ttft_ms": first_token_ms, "total_ms": int((monotonic() - started) * 1000),
        })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    port = int(os.getenv("COPILOT_WEB_PORT", "8000"))
    web.run_app(TwilioCopilot(settings).app(), host="127.0.0.1", port=port, print=None)
