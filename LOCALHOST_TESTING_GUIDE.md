"""
RAG Testing on Localhost - Complete Guide
Test everything on your local machine before production
"""

# ============================================================================
# STEP 1: START THE COPILOT SERVER (Terminal 1)
# ============================================================================

"""
Command:
  python run_call_copilot.py

Expected Output:
  ============================================================
  ✅ Copilot server is running!
  ============================================================
  Dashboard:     http://127.0.0.1:5000
  Health check:  http://127.0.0.1:5000/health
  RAG Status:    http://127.0.0.1:5000/api/rag/status
  
  Twilio webhook: wss://your-ngrok-url/twilio
  ============================================================

If you see errors:
  - Check .env has PINECONE_API_KEY and OPENAI_API_KEY
  - Check keys are correct
  - Check internet connection
"""


# ============================================================================
# STEP 2: CHECK SERVER IS RUNNING (Terminal 2)
# ============================================================================

"""
Test health endpoint:

Command:
  curl http://localhost:5000/health

Expected Response:
  {"status": "ok", "dashboards": 0}

If you see:
  - ✓ connection refused: Server not started yet
  - ✓ timeout: Check firewall settings
  - ✓ {"status": "ok"}: Server is running! ✅
"""


# ============================================================================
# STEP 3: CHECK RAG SYSTEM STATUS
# ============================================================================

"""
Test RAG API endpoint:

Command:
  curl http://localhost:5000/api/rag/status

Expected Response:
  {
    "status": "ok",
    "rag_enabled": true,
    "admin_mode": false,
    "embedding_model": "text-embedding-3-small",
    "pinecone_indexes": ["admin-kb", "subscriber-kb"]
  }

Meaning:
  ✓ RAG is enabled and working
  ✓ Pinecone connection successful
  ✓ OpenAI embeddings ready
  ✓ Admin mode is OFF (subscriber index active)
"""


# ============================================================================
# STEP 4: CREATE TEST FILES
# ============================================================================

"""
Create test.txt:

Command (Windows PowerShell):
  @"
Our premium plan includes:
- Cloud storage with 1TB capacity
- Real-time collaboration for teams
- 99.9% uptime guarantee
- 24/7 customer support
- Advanced security with encryption
- Monthly analytics and reporting
"@ | Out-File test.txt -Encoding UTF8

Command (Mac/Linux):
  cat > test.txt << EOF
Our premium plan includes:
- Cloud storage with 1TB capacity
- Real-time collaboration for teams
- 99.9% uptime guarantee
- 24/7 customer support
- Advanced security with encryption
- Monthly analytics and reporting
EOF

Verify:
  cat test.txt
"""


# ============================================================================
# STEP 5: UPLOAD TEST FILE TO RAG
# ============================================================================

"""
Upload TXT file:

Command:
  curl -X POST http://localhost:5000/api/rag/upload ^
    -F "file=@test.txt" ^
    -F "tenant_id=test-tenant-001"

(Windows: Use ^ to continue lines)
(Mac/Linux: Use \ to continue lines)

Expected Response:
  {
    "status": "success",
    "chunks_uploaded": 1,
    "total_size": 289,
    "file_type": "txt"
  }

If you see "success":
  ✅ File uploaded successfully!
  ✅ 1 chunk created and embedded
  ✅ Stored in Pinecone under tenant_id "test-tenant-001"

If you see error:
  ❌ "Missing file or tenant_id"
    → Make sure both -F flags are present
  ❌ "Connection refused"
    → Server not running (check Step 1)
  ❌ "PINECONE_API_KEY error"
    → Check .env file has correct key
"""


# ============================================================================
# STEP 6: LIST UPLOADED FILES
# ============================================================================

"""
Check what files are stored:

Command:
  curl "http://localhost:5000/api/rag/files?tenant_id=test-tenant-001"

Expected Response:
  {
    "status": "success",
    "tenant_id": "test-tenant-001",
    "files": [
      {
        "file_name": "test.txt",
        "file_type": "txt",
        "chunk_count": 1
      }
    ],
    "total_files": 1
  }

Meaning:
  ✓ Your file is stored in Pinecone
  ✓ 1 chunk created from the text
  ✓ Stored under correct tenant_id
  ✓ Ready for search!
"""


# ============================================================================
# STEP 7: UPLOAD DIFFERENT TENANT FILES
# ============================================================================

"""
Test multi-tenant isolation:

Create another test file:
  cat > test2.txt << EOF
Enterprise solution features:
- Unlimited storage
- Advanced API access
- Dedicated account manager
- Custom integrations
- Priority support (1-hour SLA)
EOF

Upload for different tenant:
  curl -X POST http://localhost:5000/api/rag/upload ^
    -F "file=@test2.txt" ^
    -F "tenant_id=acme-corp"

Verify tenant 1 still only has its file:
  curl "http://localhost:5000/api/rag/files?tenant_id=test-tenant-001"
  
Response should show only test.txt

Verify tenant 2 has its file:
  curl "http://localhost:5000/api/rag/files?tenant_id=acme-corp"
  
Response should show only test2.txt

✅ This proves tenant isolation works!
"""


# ============================================================================
# STEP 8: UPLOAD PDF FILE
# ============================================================================

"""
Download a sample PDF or create one:
  - Use any PDF you have
  - Name it: sample.pdf
  - For testing, PDFs work great

Upload PDF:
  curl -X POST http://localhost:5000/api/rag/upload ^
    -F "file=@sample.pdf" ^
    -F "tenant_id=test-tenant-001"

Expected Response:
  {
    "status": "success",
    "chunks_uploaded": 15,
    "total_size": 45000,
    "file_type": "pdf"
  }

Meaning:
  ✓ PDF extracted successfully
  ✓ 15 chunks created (splits by page)
  ✓ Each chunk embedded with OpenAI
  ✓ All chunks stored in Pinecone

List files again:
  curl "http://localhost:5000/api/rag/files?tenant_id=test-tenant-001"

Should now show TWO files:
  - test.txt (1 chunk)
  - sample.pdf (15 chunks)
"""


# ============================================================================
# STEP 9: UPLOAD CSV FILE
# ============================================================================

"""
Create test.csv:

Command (PowerShell):
  @"
Product,Price,Storage,Users,Support
Starter,$29,100GB,1,Email
Professional,$99,1TB,5,Phone
Enterprise,$299,Unlimited,Unlimited,24/7
"@ | Out-File test.csv -Encoding UTF8

Upload CSV:
  curl -X POST http://localhost:5000/api/rag/upload ^
    -F "file=@test.csv" ^
    -F "tenant_id=test-tenant-001"

Expected Response:
  {
    "status": "success",
    "chunks_uploaded": 1,
    "total_size": 180,
    "file_type": "csv"
  }

Meaning:
  ✓ CSV parsed successfully
  ✓ Formatted as searchable text
  ✓ Ready for RAG queries
"""


# ============================================================================
# STEP 10: DELETE A FILE
# ============================================================================

"""
Remove a file from knowledge base:

Command:
  curl -X DELETE "http://localhost:5000/api/rag/file/test.txt?tenant_id=test-tenant-001"

Expected Response:
  {
    "status": "success",
    "deleted_count": 1
  }

Verify it's deleted:
  curl "http://localhost:5000/api/rag/files?tenant_id=test-tenant-001"

Should now show only 2 files (PDF and CSV), not test.txt

Meaning:
  ✓ File removed from Pinecone
  ✓ No longer returned in searches
  ✓ Frees up vector storage
"""


# ============================================================================
# STEP 11: OPEN DASHBOARD IN BROWSER
# ============================================================================

"""
Access the web interface:

URL:
  http://localhost:5000

Expected to see:
  ✓ Dashboard page loads
  ✓ "Waiting for Twilio call" message
  ✓ Real-time status indicators
  ✓ (Optional) Knowledge panel if UI updated

This is the same dashboard that displays:
  - Call status
  - Audio transcripts
  - AI suggestions
  - Call metrics
"""


# ============================================================================
# STEP 12: VERIFY EVERYTHING WITH SCRIPT
# ============================================================================

"""
Save as: test_localhost.sh or test_localhost.bat

PowerShell Version (Windows):

$BaseURL = "http://localhost:5000"
$TenantID = "test-tenant-001"

Write-Host "=== Testing RAG System on Localhost ===" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health
Write-Host "1. Testing server health..." -ForegroundColor Yellow
$health = curl -s "$BaseURL/health"
Write-Host $health
Write-Host ""

# Test 2: RAG Status
Write-Host "2. Testing RAG status..." -ForegroundColor Yellow
$rag_status = curl -s "$BaseURL/api/rag/status"
Write-Host $rag_status
Write-Host ""

# Test 3: List files
Write-Host "3. Listing uploaded files..." -ForegroundColor Yellow
$files = curl -s "$BaseURL/api/rag/files?tenant_id=$TenantID"
Write-Host $files
Write-Host ""

Write-Host "=== All Tests Complete ===" -ForegroundColor Green

Run:
  powershell -ExecutionPolicy Bypass -File test_localhost.bat
"""


# ============================================================================
# STEP 13: COMMON CURL COMMANDS (QUICK REFERENCE)
# ============================================================================

"""
Basic Commands:

1. Check server running:
   curl http://localhost:5000/health

2. Check RAG status:
   curl http://localhost:5000/api/rag/status

3. Upload file:
   curl -X POST http://localhost:5000/api/rag/upload ^
     -F "file=@filename.pdf" ^
     -F "tenant_id=my-tenant"

4. List files for tenant:
   curl "http://localhost:5000/api/rag/files?tenant_id=my-tenant"

5. Delete file:
   curl -X DELETE "http://localhost:5000/api/rag/file/filename.pdf?tenant_id=my-tenant"

6. Delete all tenant data:
   curl -X DELETE "http://localhost:5000/api/rag/tenant/my-tenant"

Replace:
  - localhost:5000 with your server URL
  - filename.pdf with your actual file
  - my-tenant with your tenant ID
"""


# ============================================================================
# STEP 14: EXPECTED SEQUENCE (FIRST TIME)
# ============================================================================

"""
Complete Testing Flow (30 minutes):

0:00  Start server
      python run_call_copilot.py
      
0:01  Verify health
      curl http://localhost:5000/health
      → {"status": "ok"}

0:02  Check RAG status
      curl http://localhost:5000/api/rag/status
      → {"status": "ok", "rag_enabled": true}

0:03  Create test file
      cat > test.txt << EOF
      (paste content)
      EOF

0:05  Upload test.txt
      curl -X POST http://localhost:5000/api/rag/upload ^
        -F "file=@test.txt" ^
        -F "tenant_id=test-tenant-001"
      → {"status": "success", "chunks_uploaded": 1}

0:07  Verify upload
      curl "http://localhost:5000/api/rag/files?tenant_id=test-tenant-001"
      → Shows test.txt in list

0:10  Upload PDF
      curl -X POST http://localhost:5000/api/rag/upload ^
        -F "file=@sample.pdf" ^
        -F "tenant_id=test-tenant-001"
      → {"status": "success", "chunks_uploaded": 15}

0:15  List all files
      curl "http://localhost:5000/api/rag/files?tenant_id=test-tenant-001"
      → Shows 2 files

0:20  Test multi-tenant
      curl -X POST http://localhost:5000/api/rag/upload ^
        -F "file=@test2.txt" ^
        -F "tenant_id=acme-corp"

0:25  Verify isolation
      curl "http://localhost:5000/api/rag/files?tenant_id=acme-corp"
      → Shows only test2.txt

0:30  ✅ All tests passed! RAG system is working!
"""


# ============================================================================
# STEP 15: TROUBLESHOOTING
# ============================================================================

"""
Issue: "Connection refused" or "Could not resolve host"
  ✗ Problem: Server not running
  ✓ Solution: Run python run_call_copilot.py in Terminal 1

Issue: {"status": "error", "error": "Missing PINECONE_API_KEY"}
  ✗ Problem: .env doesn't have API key
  ✓ Solution: Add PINECONE_API_KEY=xxx to .env and restart server

Issue: {"status": "error", "error": "Missing OPENAI_API_KEY"}
  ✗ Problem: .env doesn't have API key
  ✓ Solution: Add OPENAI_API_KEY=xxx to .env and restart server

Issue: Upload returns 400 error
  ✗ Problem: Missing -F parameters
  ✓ Solution: Ensure both -F "file=@..." and -F "tenant_id=..." are present

Issue: Files uploaded but not in list
  ✗ Problem: Using different tenant_id in query
  ✓ Solution: Use same tenant_id in query as upload

Issue: Very slow uploads (>2 seconds)
  ✗ Problem: Network latency to OpenAI/Pinecone
  ✓ Solution: Normal for first upload, check internet speed

Issue: "No such file or directory" error
  ✗ Problem: File path incorrect
  ✓ Solution: Ensure file exists and path is correct
           Use full path if needed: C:\\path\\to\\file.pdf
"""


# ============================================================================
# NEXT STEP: REAL TWILIO CALL TEST
# ============================================================================

"""
Once localhost testing passes, test with real call:

1. Keep copilot running on localhost:5000
2. Run ngrok to expose to internet:
   ngrok http 5000
   
3. Copy ngrok URL and add to .env:
   TWILIO_STREAM_URL=wss://your-ngrok-id.ngrok-free.app/twilio

4. Start another terminal:
   python start_call.py

5. Open dashboard:
   http://localhost:5000

6. During call, ask question related to uploaded knowledge:
   "What are your product features?"
   "What's your pricing?"
   
7. AI should use RAG to answer with knowledge from uploaded files!

✅ RAG is working in production!
"""
