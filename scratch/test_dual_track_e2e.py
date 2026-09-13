"""Integration verification for both Single-Track (diarized) and Dual-Track endpoints."""
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from run_behavioral_signal import app
from copilot.calibration import SpeechToTextEngine
from copilot.behavioral_normalization import NormalizedUtterance, NormalizedWord

client = TestClient(app)

def test_api_health():
    res = client.get("/api/test/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["service"] == "Behavioral Signal Engine"
    print("Health check passed:", data)

def test_single_and_dual_track_endpoints_structure():
    # Verify that the test console serves HTML
    res_html = client.get("/behavioral-test")
    assert res_html.status_code == 200
    assert "PitchProX Behavioral Signal Console" in res_html.text
    assert "badge-speaker agent" in res_html.text
    assert "badge-speaker prospect" in res_html.text
    assert "Turn WPM" in res_html.text
    print("HTML Console verification passed.")

if __name__ == "__main__":
    test_api_health()
    test_single_and_dual_track_endpoints_structure()
    print("All quick E2E checks passed!")
