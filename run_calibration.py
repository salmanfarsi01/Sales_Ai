"""Standalone launcher for the AI Voice Calibration Studio.

Runs independently on port 5001 without modifying or interrupting any existing copilot services.
"""

import sys
import os

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import uvicorn
from dotenv import load_dotenv
from copilot.calibration import create_standalone_calibration_app

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("CALIBRATION_HOST", "127.0.0.1")
    port = int(os.getenv("CALIBRATION_PORT", "5001"))
    
    print("==================================================")
    print(">> AI Voice Calibration Studio running at:")
    print(f"   http://{host}:{port}/")
    print("==================================================")
    
    app = create_standalone_calibration_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
