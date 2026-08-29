"""
Example: Complete RAG Integration in run_call_copilot.py

This shows the exact changes needed to your main entry point.
Copy and adapt for your use case.
"""

import asyncio
import logging
from aiohttp import web

# OLD IMPORTS (replace with NEW IMPORTS below)
# from copilot.twilio_app import AllQuestionsCopilot
# from copilot.config import Settings

# NEW IMPORTS - RAG-Enabled
from copilot.twilio_rag import RAGTwilioCopilot as AllQuestionsCopilot
from copilot.config_rag import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

LOGGER = logging.getLogger("copilot.main")


async def main():
    """Start the RAG-enabled Twilio copilot server."""
    
    # Load configuration from environment
    LOGGER.info("Loading configuration...")
    try:
        settings = Settings.from_env()
    except RuntimeError as e:
        LOGGER.error(f"Configuration error: {e}")
        raise
    
    LOGGER.info(
        f"Starting RAG Copilot on {settings.host}:{settings.port} "
        f"(RAG={'enabled' if settings.rag.rag_enabled else 'disabled'})"
    )
    
    # Create copilot instance
    # This now includes RAG support by default
    copilot = AllQuestionsCopilot(settings)
    
    LOGGER.info(f"RAG System Status:")
    LOGGER.info(f"  - Enabled: {settings.rag.rag_enabled}")
    LOGGER.info(f"  - Admin Mode: {settings.rag.admin_mode}")
    LOGGER.info(f"  - Chunk Size: {settings.rag.chunk_size} chars")
    LOGGER.info(f"  - Search Top-K: {settings.rag.search_top_k}")
    LOGGER.info(f"  - Min Score: {settings.rag.min_score_threshold}")
    LOGGER.info(f"  - Default Tenant: {settings.default_tenant_id}")
    
    # Create and start web application
    app = copilot.app()
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, settings.host, settings.port)
    await site.start()
    
    print(f"\n{'='*60}")
    print(f"✅ Copilot server is running!")
    print(f"{'='*60}")
    print(f"Dashboard:     http://{settings.host}:{settings.port}")
    print(f"Health check:  http://{settings.host}:{settings.port}/health")
    print(f"RAG Status:    http://{settings.host}:{settings.port}/api/rag/status")
    print(f"\nTwilio webhook: wss://your-ngrok-url/twilio")
    print(f"{'='*60}\n")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down...")
        await runner.cleanup()
        print("\n✅ Copilot stopped.")


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================================
# MIGRATION NOTES
# ============================================================================

"""
Changes from old run_call_copilot.py:

1. IMPORT CHANGE (Line 1-2)
   OLD: from copilot.twilio_app import AllQuestionsCopilot
   NEW: from copilot.twilio_rag import RAGTwilioCopilot as AllQuestionsCopilot

2. CONFIG CHANGE (Line 3)
   OLD: from copilot.config import Settings
   NEW: from copilot.config_rag import Settings

3. LOGGING ADDED (Line 12-14)
   NEW: Better logging during startup

4. SETTINGS DISPLAY ADDED (Line 30-38)
   NEW: Shows RAG configuration on startup

5. EVERYTHING ELSE STAYS THE SAME!

The RAGTwilioCopilot class:
- ✅ Extends FastTwilioCopilot (all existing functionality)
- ✅ Adds RAG support
- ✅ Maintains backward compatibility
- ✅ Drops in as AllQuestionsCopilot replacement
- ✅ Adds /api/rag/* endpoints
- ✅ Improves suggestion generation

No changes needed to:
- Twilio WebSocket handling
- Deepgram transcription
- Groq LLM calls
- Dashboard broadcasting
- Call reporting
- Any other existing functionality
"""


# ============================================================================
# DEBUGGING & TROUBLESHOOTING
# ============================================================================

"""
If you see errors during startup:

1. "Missing required environment variable: PINECONE_API_KEY"
   Solution: Add PINECONE_API_KEY to .env
   
2. "Missing required environment variable: OPENAI_API_KEY"
   Solution: Add OPENAI_API_KEY to .env
   
3. "Pinecone connection failed"
   Solution: Verify API key is correct and account is active
   
4. "OpenAI API key invalid"
   Solution: Verify API key is correct and has credits
   
5. "RAG system is running but suggests from old knowledge base"
   Solution: Clear old knowledge/ folder or set RAG_ENABLED=false

If you want to DISABLE RAG:
- Set RAG_ENABLED=false in .env
- The system will use LocalKnowledgeBase (old behavior)
- Useful for gradual migration or fallback
"""


# ============================================================================
# OPTIONAL: CONDITIONAL IMPORT (For gradual migration)
# ============================================================================

"""
If you want to keep BOTH old and new systems during migration:

import os

if os.getenv("RAG_ENABLED", "false").lower() == "true":
    # Use new RAG system
    from copilot.twilio_rag import RAGTwilioCopilot as AllQuestionsCopilot
    from copilot.config_rag import Settings
    LOGGER.info("RAG system enabled")
else:
    # Use old local knowledge system
    from copilot.twilio_app import AllQuestionsCopilot
    from copilot.config import Settings
    LOGGER.info("Using local knowledge base (RAG disabled)")

# Rest of code stays the same!
"""


# ============================================================================
# OPTIONAL: MONITORING & HEALTH CHECKS
# ============================================================================

"""
Add this to check system health during startup:

async def check_rag_health(copilot):
    '''Verify RAG system is working (optional)'''
    if not copilot.rag_retriever:
        LOGGER.info("RAG system is disabled")
        return
    
    try:
        # Try to list files in default tenant
        files = copilot.rag_retriever.list_knowledge_files("default")
        LOGGER.info(f"RAG health check: OK ({len(files.get('files', []))} files)")
    except Exception as e:
        LOGGER.warning(f"RAG health check failed: {e}")

# Call before event().wait():
# await check_rag_health(copilot)
"""
