from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import suppress
from time import monotonic

import websockets
from aiohttp import WSMsgType, web

from .twilio_app import STOP, TwilioCopilot

LOGGER = logging.getLogger("copilot.twilio.fast")


class FastTwilioCopilot(TwilioCopilot):
    """Two independent mono STT streams; avoids stereo alignment and buffering."""

    async def component(self, name: str, state: str, detail: str = "") -> None:
        await self.broadcast({"type": "component", "component": name, "state": state, "detail": detail})

    async def twilio_stream(self, request: web.Request) -> web.WebSocketResponse:
        twilio = web.WebSocketResponse(heartbeat=20, max_msg_size=2 * 1024 * 1024)
        await twilio.prepare(request)
        call_id = str(uuid.uuid4())
        queue_size = max(256, self.settings.audio_queue_size)
        queues: dict[str, asyncio.Queue[bytes | object]] = {
            "salesperson": asyncio.Queue(queue_size), "client": asyncio.Queue(queue_size)
        }
        packets = {"salesperson": 0, "client": 0}
        dropped = {"salesperson": 0, "client": 0}
        context: list[dict[str, str]] = []
        state: dict[str, object] = {"generation_task": None, "last_generation": 0.0}
        await self.component("twilio", "connected", "Media WebSocket accepted")
        LOGGER.info("Twilio connected %s", call_id)

        async def receive_twilio() -> None:
            nonlocal call_id
            async for message in twilio:
                if message.type == WSMsgType.ERROR:
                    raise twilio.exception() or RuntimeError("Twilio WebSocket failed")
                if message.type != WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
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
                    import base64
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
                            "type": "transcript", "call_id": call_id, "role": role,
                            "text": text, "final": final,
                        })
                        if final:
                            context.append({
                                "role": "assistant" if role == "salesperson" else "user",
                                "content": text,
                            })
                            del context[:-self.settings.transcript_window]
                        if role == "client" and len(text) >= 8:
                            task = state.get("generation_task")
                            due = monotonic() - float(state["last_generation"]) >= 0.35
                            can_start = task is None or task.done()
                            if speech_final and task and not task.done():
                                task.cancel()
                                can_start = True
                            if due and can_start:
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
                with suppress(asyncio.CancelledError):
                    await generation_task
            await twilio.close()
            await self.broadcast({
                "type": "call_end", "call_id": call_id,
                "dropped_audio_chunks": dropped["salesperson"] + dropped["client"],
            })
            await self.component(
                "twilio", "ended",
                f"packets salesperson={packets['salesperson']}, client={packets['client']}; dropped={dropped}",
            )
        return twilio


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from .config import Settings
    settings = Settings.from_env()
    port = int(os.getenv("COPILOT_WEB_PORT", "8000"))
    LOGGER.info("dashboard http://127.0.0.1:%d", port)
    LOGGER.info("Twilio WebSocket endpoint /twilio")
    web.run_app(FastTwilioCopilot(settings).app(), host="127.0.0.1", port=port, print=None)
