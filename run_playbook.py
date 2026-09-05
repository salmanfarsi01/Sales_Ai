"""Standalone launcher for the PitchProX Custom Playbook Studio.

Runs independently on port 5002 without modifying or interrupting any existing copilot services.
"""

import sys
import os

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import uvicorn
from dotenv import load_dotenv
from copilot.playbook import create_standalone_playbook_app

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("PLAYBOOK_HOST", "127.0.0.1")
    port = int(os.getenv("PLAYBOOK_PORT", "5002"))
    
    print("==================================================")
    print(">> PitchProX Custom Playbook Studio running at:")
    print(f"   http://{host}:{port}/")
    print("==================================================")
    
    app = create_standalone_playbook_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
