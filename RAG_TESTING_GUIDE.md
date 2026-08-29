"""
Complete RAG System Testing Guide
From Setup to Verification - Step by Step
"""

# ============================================================================
# PART 1: PRE-TESTING CHECKLIST (5 minutes)
# ============================================================================

"""
Before you start testing:

☐ Install dependencies:
  pip install -r requirements-rag-full.txt

☐ Create Pinecone account: https://www.pinecone.io
  - Sign up (free serverless available)
  - Create index
  - Copy API key

☐ Create OpenAI account: https://openai.com
  - Sign up
  - Create API key at https://platform.openai.com/api-keys
  - Ensure account has credits (small charges)

☐ Update .env file with keys (see template below)

☐ Have sample files ready:
  - sample.pdf (or any PDF)
  - test.txt or test.md
  - sample.csv

☐ Verify venv is activated
"""


# ============================================================================
# PART 2: ENVIRONMENT SETUP (.env)
# ============================================================================

"""
Add to your .env file:

# Existing keys (already have)
DEEPGRAM_API_KEY=your_key
GROQ_API_KEY=your_key

# NEW: RAG Configuration
PINECONE_API_KEY=pc-xxxxx-xxxxx-xxxxx  # Get from Pinecone dashboard
OPENAI_API_KEY=sk-proj-xxxxx          # Get from OpenAI dashboard

# RAG Settings (defaults, adjust after testing)
RAG_ENABLED=true
ADMIN_MODE=false
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=150
RAG_SEARCH_TOP_K=5
RAG_MIN_SCORE=0.5
DEFAULT_TENANT_ID=test-tenant-001
"""


# ============================================================================
# PART 3: UNIT TESTS (10 minutes)
# ============================================================================

"""
TEST 1: File Extraction
========================

Run:
  pytest tests/test_file_extraction.py -v

What it tests:
  ✓ PDF extraction
  ✓ Text file extraction
  ✓ CSV extraction
  ✓ Error handling

Expected output:
  test_extract_pdf PASSED
  test_extract_txt PASSED
  test_extract_csv PASSED
  test_extract_markdown PASSED
  test_extract_from_blob PASSED
  test_unsupported_format PASSED

Result: If all ✓ PASSED, file extraction works!


TEST 2: RAG Pinecone Integration
==================================

Run:
  pytest tests/test_rag_pinecone.py -v

What it tests:
  ✓ Pinecone initialization
  ✓ Admin mode vs subscriber mode
  ✓ Chunk ID generation
  ✓ Text chunking algorithm

Expected output:
  test_initialization PASSED
  test_admin_mode_initialization PASSED
  test_generate_chunk_id PASSED
  test_split_into_chunks PASSED

Result: If all ✓ PASSED, Pinecone integration structure works!


TEST 3: RAG Integration
========================

Run:
  pytest tests/test_rag_integration.py -v

What it tests:
  ✓ Tenant context creation
  ✓ Retriever initialization
  ✓ Context formatting

Expected output:
  test_tenant_context_creation PASSED
  test_tenant_context_admin PASSED
  test_initialization PASSED
  test_get_context_formatting PASSED

Result: If all ✓ PASSED, integration layer works!
"""


# ============================================================================
# PART 4: INTEGRATION TEST (5 minutes)
# ============================================================================

"""
TEST 4: Upload and Search
===========================

This tests the complete pipeline:
File → Extract → Chunk → Embed → Upload → Search

Step 1: Create test file
  cat > test_sample.txt << EOF
  Our product features:
  - Cloud-based storage
  - Real-time collaboration
  - 99.9% uptime guarantee
  - 24/7 customer support
  EOF

Step 2: Start copilot server
  In Terminal 1:
  python run_call_copilot.py
  
  Expected output:
  ✅ Copilot server is running!
  Dashboard: http://127.0.0.1:5000

Step 3: Upload test file via API
  In Terminal 2:
  curl -X POST http://localhost:5000/api/rag/upload \
    -F "file=@test_sample.txt" \
    -F "tenant_id=test-tenant-001"

  Expected output:
  {
    "status": "success",
    "chunks_uploaded": 1,
    "total_size": 156,
    "file_type": "txt"
  }

  ✓ If you see "success", upload works!

Step 4: List uploaded files
  curl "http://localhost:5000/api/rag/files?tenant_id=test-tenant-001"

  Expected output:
  {
    "status": "success",
    "tenant_id": "test-tenant-001",
    "files": [
      {
        "file_name": "test_sample.txt",
        "file_type": "txt",
        "chunk_count": 1
      }
    ],
    "total_files": 1
  }

  ✓ If you see your file listed, storage works!

Step 5: Check Pinecone dashboard
  1. Log into Pinecone console
  2. Open "subscriber-kb" index
  3. Look for namespace "test-tenant-001"
  4. Should see 1 vector uploaded

  ✓ If you see the vector, Pinecone integration works!
"""


# ============================================================================
# PART 5: REAL-TIME CALL TEST (15 minutes)
# ============================================================================

"""
TEST 5: Complete Call with RAG
================================

This tests the full system end-to-end with actual Twilio call.

Prerequisites:
  ✓ Copilot server running (Terminal 1)
  ✓ ngrok tunnel running (Terminal 2)
  ✓ Test file uploaded to RAG (test_sample.txt)

Step 1: Verify copilot is running
  curl http://localhost:5000/health
  
  Expected:
  {"status": "ok", "dashboards": 0}

Step 2: Check RAG status
  curl http://localhost:5000/api/rag/status
  
  Expected:
  {
    "status": "ok",
    "rag_enabled": true,
    "admin_mode": false,
    "embedding_model": "text-embedding-3-small",
    "pinecone_indexes": ["admin-kb", "subscriber-kb"]
  }

Step 3: Start ngrok (Terminal 2)
  ngrok http 5000
  
  Copy the HTTPS URL (e.g., https://abc123.ngrok-free.app)

Step 4: Update .env with ngrok URL
  TWILIO_STREAM_URL=wss://abc123.ngrok-free.app/twilio

Step 5: Open dashboard
  http://127.0.0.1:5000
  
  Should see:
  - Call Status: "Waiting for Twilio call"
  - Knowledge Panel (if UI updated)

Step 6: Make call (Terminal 3)
  python start_call.py
  
  Expected:
  - Twilio connects
  - Dashboard shows: "streaming"
  - Audio tracks show: "receiving"

Step 7: During call, ask question about knowledge
  Salesperson: "Tell them about our product features"
  or
  Client: "What features do you have?"
  
  Dashboard should show:
  - AI answer: "generating"
  - Then: "ready" with suggestion
  - Suggestion should mention: cloud storage, collaboration, uptime, support
  
  ✓ If suggestion uses your knowledge, RAG WORKS!

Step 8: End call
  Hang up
  
  Dashboard shows:
  - Call report generated
  - JSON saved to reports/CA[call_id].json

Step 9: Check call report
  cat reports/CA[call_id].json | python -m json.tool
  
  Look for:
  - "call_summary"
  - "key_moments_log"
  - "performance_metrics"
  - Verify questions are answered using your knowledge
"""


# ============================================================================
# PART 6: VALIDATION CHECKLIST
# ============================================================================

"""
Complete Validation Checklist:

File Extraction:
  ☐ pytest tests/test_file_extraction.py -v passes
  ☐ Upload txt file via API returns success
  ☐ Upload PDF file via API returns success

Pinecone Integration:
  ☐ pytest tests/test_rag_pinecone.py -v passes
  ☐ Uploaded files appear in Pinecone console
  ☐ Correct namespace used (tenant_id)

API Endpoints:
  ☐ POST /api/rag/upload works
  ☐ GET /api/rag/files returns list
  ☐ DELETE /api/rag/file/{name} works
  ☐ GET /api/rag/status shows correct info

Real Call:
  ☐ Dashboard opens without errors
  ☐ RAG status shows "enabled": true
  ☐ Call connects and streams audio
  ☐ Suggestions use knowledge (not generic)
  ☐ Call report generated with content

End-to-End:
  ☐ Multi-file upload works (PDF, CSV, TXT)
  ☐ Different tenants isolated (use different tenant_ids)
  ☐ Delete file removes from search results
  ☐ System works for 30+ minutes without errors
"""


# ============================================================================
# PART 7: TROUBLESHOOTING
# ============================================================================

"""
Common Issues & Solutions:

1. "PINECONE_API_KEY not found"
   ✗ Issue: Missing from .env
   ✓ Solution: Add to .env and restart copilot

2. "OPENAI_API_KEY not found"
   ✗ Issue: Missing from .env
   ✓ Solution: Add to .env and restart copilot

3. Upload returns 400 error
   ✗ Issue: Missing tenant_id in request
   ✓ Solution: Add -F "tenant_id=test-tenant-001"

4. Upload succeeds but file not in /api/rag/files
   ✗ Issue: Querying wrong tenant_id
   ✓ Solution: Use same tenant_id in query

5. Suggestions don't mention uploaded content
   ✗ Issue: RAG not integrated in suggestion generation
   ✓ Solution: Verify twilio_rag.py is being used (check imports)

6. Slow embeddings (>500ms)
   ✗ Issue: Network latency or slow API
   ✓ Solution: Check internet, retry

7. High costs
   ✗ Issue: Embedding every upload multiple times
   ✓ Solution: Delete test files, batch uploads

8. Pinecone index not showing vectors
   ✗ Issue: Namespace is case-sensitive
   ✓ Solution: Use lowercase tenant_ids consistently
"""


# ============================================================================
# PART 8: PERFORMANCE BENCHMARKS
# ============================================================================

"""
Expected Performance (after setup):

File Upload:
  - Small file (< 1MB): 100-500ms
  - PDF page: 50-100ms per page
  - CSV row: 10-20ms per 1000 rows

Vector Search:
  - Embedding query: 50-100ms
  - Pinecone search: 20-50ms
  - Total latency: 100-200ms

Suggestion Generation:
  - Retrieve context: 100-200ms
  - Generate with Groq: 500-2000ms
  - Total suggestion latency: 600-2200ms

Cost:
  - Embedding: $0.02 per 1M tokens (~$0.0002 per file)
  - Search: <$0.0001 per query
  - Storage: ~$0.04-0.08 per 100K vectors/month
"""


# ============================================================================
# PART 9: QUICK VERIFICATION SCRIPT
# ============================================================================

"""
Copy and save as: test_rag_quick.sh

#!/bin/bash

echo "=== RAG System Quick Test ==="
echo

# Test 1: Config
echo "1. Checking environment variables..."
if [ -z "$PINECONE_API_KEY" ]; then
  echo "❌ PINECONE_API_KEY not set"
else
  echo "✓ PINECONE_API_KEY found"
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "❌ OPENAI_API_KEY not set"
else
  echo "✓ OPENAI_API_KEY found"
fi

# Test 2: Dependencies
echo
echo "2. Checking Python packages..."
python -c "import pinecone; print('✓ pinecone')" 2>/dev/null || echo "❌ pinecone not installed"
python -c "import openai; print('✓ openai')" 2>/dev/null || echo "❌ openai not installed"
python -c "import fitz; print('✓ PyMuPDF')" 2>/dev/null || echo "❌ PyMuPDF not installed"

# Test 3: Server
echo
echo "3. Testing copilot server..."
curl -s http://localhost:5000/health > /dev/null && echo "✓ Server running" || echo "❌ Server not responding"

# Test 4: RAG
echo
echo "4. Testing RAG status..."
curl -s http://localhost:5000/api/rag/status | grep -q "rag_enabled" && echo "✓ RAG API working" || echo "❌ RAG API not responding"

echo
echo "=== Test Complete ==="

Run:
  chmod +x test_rag_quick.sh
  ./test_rag_quick.sh
"""


# ============================================================================
# PART 10: TESTING TIMELINE
# ============================================================================

"""
Recommended Testing Order (total: ~1 hour):

0:00-0:10  Setup
           ✓ Install dependencies
           ✓ Add API keys to .env
           ✓ Verify keys are correct

0:10-0:20  Unit Tests
           ✓ pytest tests/test_file_extraction.py -v
           ✓ pytest tests/test_rag_pinecone.py -v
           ✓ pytest tests/test_rag_integration.py -v

0:20-0:30  API Tests
           ✓ Start copilot server
           ✓ Upload test file
           ✓ List files
           ✓ Delete file

0:30-0:40  Pinecone Verification
           ✓ Check Pinecone console
           ✓ Verify vectors created
           ✓ Check namespaces

0:40-1:00  Real Call Test
           ✓ Open dashboard
           ✓ Start ngrok
           ✓ Make test call
           ✓ Verify suggestions use knowledge
           ✓ Check call report

Result: If all pass, RAG system is working! 🎉
"""
