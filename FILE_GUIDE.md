# Project File Guide

This document explains what each meaningful file in the project does and how the files connect.

## Top-Level Files

### `run_call_copilot.py`

The main server entry point. It loads variables from `.env` and starts `AllQuestionsCopilot`, the most complete version of the application. Run it with:

```powershell
python run_call_copilot.py
```

### `start_call.py`

Starts the actual Twilio phone call. It reads the Twilio credentials, phone numbers, and public WebSocket URL from `.env`, creates TwiML, calls the salesperson, dials the client, and requests both audio tracks.

### `README.md`

The main setup and operating guide. It explains dependency installation, server startup, ngrok configuration, call startup, dashboard statuses, knowledge files, and tests.

### `PDF_KNOWLEDGE.md`

Instructions for installing PDF support and uploading PDF knowledge through the dashboard. It also documents limitations such as unsupported encrypted or scanned PDFs.

### `PIPELINE_AND_RESPONSE_TIME.md`

Describes the complete call-processing pipeline and the techniques used to reduce response latency.

### `FILE_GUIDE.md`

This file. It describes the purpose of every meaningful project file.

### `requirements.txt`

Contains the pinned Python dependencies for the core application, including `aiohttp`, `websockets`, `twilio`, `groq`, and `python-dotenv`.

### `requirements-pdf.txt`

Adds optional PDF support. It installs the normal requirements plus `pypdf`.

## `copilot/` — Application Code

### `copilot/__init__.py`

Marks `copilot` as a Python package. It does not contain runtime logic.

### `copilot/config.py`

Defines the `Settings` data class and loads configuration from environment variables. This includes API keys, model selection, queue size, transcript history size, and suggestion timing.

### `copilot/twilio_app.py`

Provides the base `TwilioCopilot` class and shared application behavior:

- Creates the HTTP and WebSocket routes.
- Manages connected dashboards.
- Broadcasts transcripts, statuses, and suggestions.
- Loads and searches the local knowledge base.
- Calls Groq and streams generated tokens.
- Contains an older stereo-track Twilio/Deepgram implementation.

The newer runtime inherits its shared behavior but replaces the audio-processing method.

### `copilot/twilio_fast.py`

Contains the optimized real-time audio pipeline. It:

- Separates Twilio inbound and outbound tracks.
- Maintains independent audio queues.
- Opens one Deepgram connection per speaker.
- processes audio and transcripts asynchronously.
- Generates suggestions only from client speech.
- Cancels outdated generation tasks.
- Reports detailed component states to the dashboard.

### `copilot/twilio_diagnostic.py`

Extends the optimized pipeline and selects `web/twilio_fast.html` as the diagnostic dashboard. Most of its behavior comes from `FastTwilioCopilot`.

### `copilot/twilio_context.py`

Adds context recovery before LLM generation. If a client asks an incomplete follow-up, it combines the question with relevant recent client speech before searching knowledge and calling Groq.

### `copilot/context_recovery.py`

Contains the pure text-processing functions used by `twilio_context.py`:

- `needs_context()` detects incomplete, referential, or continuation questions.
- `recover_query()` prepends enough recent client speech to make a question understandable.

### `copilot/retrieval.py`

Implements the local knowledge retrieval system. It loads `.txt` and `.md` files, splits them into chunks, tokenizes their content, scores chunks by word overlap and query coverage, and returns the three best matches.

### `copilot/knowledge_upload.py`

Adds PDF knowledge support and the knowledge dashboard. It:

- Adds `/knowledge` and `/knowledge/upload` routes.
- Validates uploaded PDF files.
- Rejects oversized, encrypted, invalid, or image-only PDFs.
- Extracts text with `pypdf`.
- Saves the original PDF and extracted Markdown.
- Reloads the knowledge index without restarting the server.

### `copilot/knowledge_all_questions.py`

Defines the final application class used by `run_call_copilot.py`. It keeps recent client history and ensures very short final questions can also trigger context-aware suggestions.

## Runtime Inheritance Chain

The final application is assembled through inheritance:

```text
TwilioCopilot
    ↓ shared routes, dashboard events, retrieval, Groq generation
FastTwilioCopilot
    ↓ separate low-latency audio and STT streams
DiagnosticTwilioCopilot
    ↓ diagnostic dashboard
ContextAwareTwilioCopilot
    ↓ incomplete-question recovery
KnowledgeTwilioCopilot
    ↓ PDF upload and immediate indexing
AllQuestionsCopilot
    ↓ very-short-question handling
run_call_copilot.py
```

## `web/` — Browser Dashboards

### `web/twilio_fast.html`

The diagnostic real-time dashboard used by the intermediate diagnostic/context-aware versions. It displays connection states, audio activity, transcripts, and streamed suggestions.

### `web/twilio_knowledge.html`

The final dashboard used by `KnowledgeTwilioCopilot` and `AllQuestionsCopilot`. It includes the live call interface plus PDF upload and knowledge-source information.

## `knowledge/` — Grounding Documents

### `knowledge/README.md`

Explains what kinds of `.txt` and `.md` files can be placed in the directory for local retrieval.

### Uploaded and manually added documents

User-provided `.txt` and `.md` files are indexed when the server starts. A PDF uploaded from the dashboard creates:

- The original `.pdf` file.
- A `.pdf.md` file containing its extracted text, which is used for retrieval.

## `tests/` — Automated Tests

### `tests/test_retrieval.py`

Tests that the local knowledge system returns a relevant document and safely handles queries with no useful tokens.

### `tests/test_pdf_knowledge.py`

Tests sanitization of uploaded PDF filenames and the fallback filename behavior.

### `tests/test_context_recovery.py`

Tests incomplete-question detection, reference resolution, continuation merging, standalone questions, and history limits.

Run all tests with:

```powershell
python -m unittest discover -s tests -v
```

## Generated Files

Directories named `__pycache__/` and files ending in `.pyc` are generated automatically by Python. They cache compiled bytecode to make imports faster and should not be edited manually.

## Configuration Not Shown in the File List

The application expects a local `.env` file containing values such as:

- `DEEPGRAM_API_KEY`
- `GROQ_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER`
- `TWILIO_TO_NUMBER`
- `TWILIO_DIAL_NUMBER`
- `TWILIO_STREAM_URL`

This file contains secrets and should not be committed to source control.
