"""Global Pytest Configuration & Test Fixtures."""

import os
import sys
from pathlib import Path
import pytest

# Ensure repository root is on sys.path for test imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Provide fallback environment variables for test execution
os.environ["RAG_ENABLED"] = "false"
os.environ.setdefault("GROQ_API_KEY", "mock_groq_api_key_test_12345")
os.environ.setdefault("DEEPGRAM_API_KEY", "mock_deepgram_api_key_test_12345")
os.environ.setdefault("ELEVENLABS_API_KEY", "mock_elevenlabs_api_key_test_12345")
os.environ.setdefault("PINECONE_API_KEY", "mock_pinecone_api_key_test_12345")
os.environ.setdefault("OPENAI_API_KEY", "mock_openai_api_key_test_12345")
os.environ.setdefault("PINECONE_INDEX_NAME", "mock_pinecone_index_test")
os.environ.setdefault("COPILOT_ENV", "development")
