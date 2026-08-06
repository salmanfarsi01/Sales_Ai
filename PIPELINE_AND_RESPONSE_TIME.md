# Pipeline and Response-Time Optimization

## Pipeline Used in This Project

This project uses a real-time sales-call copilot pipeline:

```text
Salesperson + Client
        ↓
Twilio phone call
        ↓ both audio tracks
ngrok WebSocket tunnel
        ↓
Python copilot server
        ↓ split by speaker
Two Deepgram streams
        ↓ transcripts
Context recovery + knowledge retrieval
        ↓
Groq LLM
        ↓ streamed suggestion
Browser dashboard
```

1. `run_call_copilot.py` starts the final `AllQuestionsCopilot` server on port 8000.

2. `start_call.py` tells Twilio to call the salesperson, dial the client, and stream `both_tracks` to `/twilio` through ngrok.

3. The server separates the Twilio audio tracks:

   - `inbound` → salesperson
   - `outbound` → client

   Each track gets its own bounded audio queue, preventing temporary slowdowns from blocking the call.

4. Two independent Deepgram WebSockets transcribe the tracks using 8 kHz μ-law audio, Nova-3, interim and final transcripts, voice activity detection, and endpointing.

5. Final transcripts from both speakers enter a rolling conversation window. Only client speech triggers an AI suggestion.

6. Short or incomplete follow-ups, such as “what about that?”, are reconstructed using recent client turns. Very short final questions also receive special handling so they are not ignored.

7. Before generation, the client’s question is matched against local knowledge:

   - `.txt` and `.md` files are divided into chunks.
   - Matching uses word overlap and query coverage.
   - The best three chunks are supplied to the model.
   - Uploaded PDFs are converted into Markdown and indexed immediately.

8. Groq generates a response using the recent conversation, reconstructed client question, relevant knowledge chunks, and a system instruction limiting the answer to three concise sentences. The default model is `llama-3.1-8b-instant`.

9. Generated tokens are streamed through `/events` to the browser. The dashboard also receives transcripts, component statuses, sources, timing information, errors, and call lifecycle events.

In short: **Twilio captures the call → Deepgram transcribes each person → the client’s question is contextualized and grounded in local documents → Groq generates a concise response → the dashboard shows it to the salesperson in real time.**

## How Response Time Was Optimized

1. **Separated audio streams:** Salesperson and client audio use two independent Deepgram connections, avoiding delays caused by aligning both tracks into stereo audio.

2. **Interim transcripts:** Deepgram sends partial speech results before the client finishes speaking, allowing suggestion generation to begin sooner.

3. **Aggressive endpointing:** `endpointing=120` detects the end of speech after approximately 120 ms of silence.

4. **Early generation:** Client transcripts can trigger generation every 350 ms:

   ```python
   due = monotonic() - last_generation >= 0.35
   ```

   This balances responsiveness against excessive LLM requests.

5. **Cancellation of obsolete generation:** When a newer final transcript arrives, the older generation task is cancelled so the model focuses on the latest and most complete question.

6. **Streamed Groq output:** Groq uses `stream=True`, so tokens appear on the dashboard immediately instead of waiting for the complete answer.

7. **Fast model:** The default `llama-3.1-8b-instant` model prioritizes low latency.

8. **Small prompts:** The system retains only the latest 12 transcript turns, retrieves at most three knowledge chunks, limits generation to 140 tokens, and requests no more than three sentences.

9. **Lightweight local retrieval:** Knowledge retrieval uses local token-overlap matching instead of making calls to an embedding service or remote vector database.

10. **Asynchronous processing:** Audio reception, two transcription streams, LLM generation, and dashboard updates run concurrently with `asyncio`. Blocking Groq work runs in a worker thread.

11. **Bounded queues:** Audio queues prevent backpressure from growing indefinitely. Under overload, packets can be dropped instead of continuously increasing end-to-end latency.

12. **Latency measurements:** The application records time to first token (`ttft_ms`) and total generation time, making performance regressions easier to identify.

The largest latency improvements come from **independent STT streams, interim transcription, early generation, the small Groq model, local retrieval, and streamed tokens**.
