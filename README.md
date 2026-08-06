# Twilio Real-Time Sales Copilot

The application receives both sides of a Twilio call, transcribes each side independently with Deepgram, and generates streamed Groq suggestions only from the client's speech. The browser is a read-only dashboard; it does not use its microphone.

## One-time setup

Create and activate the virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add real credentials, phone numbers, and the current ngrok URL.

## Run every session

Use three PowerShell terminals in this order.

### 1. Start the copilot server

```powershell
python run_call_copilot.py
```

Why: serves the dashboard on port 8000, accepts Twilio audio at `/twilio`, opens separate Deepgram streams for both speakers, and streams AI suggestions.

Open the dashboard at `http://127.0.0.1:8000`.

### 2. Expose it to Twilio

```powershell
ngrok http 8000
```

Why: Twilio cannot connect to localhost. Copy ngrok's current HTTPS domain into `.env` as a WSS URL ending in `/twilio`:

```env
TWILIO_STREAM_URL=wss://YOUR-DOMAIN.ngrok-free.app/twilio
```

Restarting ngrok normally changes this URL, so update `.env` before each call when necessary.

### 3. Start the phone call

```powershell
python start_call.py
```

Why: asks Twilio to call the salesperson, dial the client, and stream both audio tracks to the copilot.

## Dashboard status order

A healthy call progresses through:

1. Twilio: `streaming`
2. Sales audio and Client audio: `receiving`
3. Sales STT and Client STT: `ready`, then `transcribing`
4. AI answer: `generating`, then `ready`

If Twilio never reaches `streaming`, check `TWILIO_STREAM_URL`. If one audio track remains `Waiting`, that participant is not connected or Twilio is not sending that track. If audio is received but STT never says `transcribing`, check the Deepgram key and language.

## Knowledge grounding

Place `.txt` or `.md` files in `knowledge/` before starting the server. Relevant passages and source filenames are included with suggestions.

## Tests

```powershell
python -m unittest discover -s tests -v
```

