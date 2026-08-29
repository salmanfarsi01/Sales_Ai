from dotenv import load_dotenv
import uvicorn
import os

load_dotenv(override=True)

if __name__ == "__main__":
    port = int(os.getenv("COPILOT_PORT", os.getenv("COPILOT_WEB_PORT", "5000")))
    host = os.getenv("COPILOT_HOST", "127.0.0.1")
    try:
        print(f"Starting FastAPI Sales Copilot server on http://{host}:{port}")
        uvicorn.run("copilot.fastapi_app:app", host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("Twilio copilot stopped.")
