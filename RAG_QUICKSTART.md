"""
RAG System Quick Start Guide

Get your SaaS platform's AI knowledge base up and running in 15 minutes.
"""

# ============================================================================
# PART 1: SETUP (5 minutes)
# ============================================================================

# Step 1: Install dependencies
# Terminal:
pip install -r requirements-rag-full.txt

# Step 2: Create accounts and get API keys
# 1. Pinecone (https://www.pinecone.io)
#    - Sign up → Create serverless index
#    - Copy API key from dashboard
#
# 2. OpenAI (https://openai.com)
#    - Sign up → Create API key
#    - Ensure you have credits


# Step 3: Update .env file
# Add these lines:
cat >> .env << EOF

# === RAG Configuration ===
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

RAG_ENABLED=true
ADMIN_MODE=false
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=150
RAG_SEARCH_TOP_K=5
RAG_MIN_SCORE=0.5
DEFAULT_TENANT_ID=default
EOF


# ============================================================================
# PART 2: INTEGRATE INTO COPILOT (3 minutes)
# ============================================================================

# Step 4: Update run_call_copilot.py
# Change these lines:

# OLD:
# from copilot.twilio_app import AllQuestionsCopilot

# NEW:
from copilot.twilio_rag import RAGTwilioCopilot as AllQuestionsCopilot


# Step 5: Restart server
python run_call_copilot.py

# Open browser:
# http://127.0.0.1:5000


# ============================================================================
# PART 3: UPLOAD KNOWLEDGE FILES (5 minutes)
# ============================================================================

# Option A: Use Dashboard UI (Recommended)
# 1. Open dashboard at http://127.0.0.1:5000
# 2. Find "Knowledge Base" section
# 3. Click "Choose Files"
# 4. Select: PDF, Word, Excel, CSV, images, etc.
# 5. Click "Upload"
# 6. Wait for "Success" message

# Supported formats:
# ✅ PDF files (.pdf)
# ✅ Text files (.txt, .md)
# ✅ Word documents (.docx, .doc)
# ✅ Excel spreadsheets (.xlsx, .xls)
# ✅ CSV files (.csv)
# ✅ Images with text (.jpg, .png) - requires Tesseract
# ⚠️ Video files (.mp4) - requires transcription service
# ⚠️ Audio files (.mp3, .wav) - requires transcription service


# Option B: Use API (Programmatic)
# Upload a file:
curl -X POST http://localhost:5000/api/rag/upload \
  -F "file=@sample.pdf" \
  -F "tenant_id=default"

# Response:
# {
#   "status": "success",
#   "chunks_uploaded": 42,
#   "total_size": 15234,
#   "file_type": "pdf"
# }

# List uploaded files:
curl "http://localhost:5000/api/rag/files?tenant_id=default"

# Response:
# {
#   "status": "success",
#   "tenant_id": "default",
#   "files": [
#     {
#       "file_name": "sample.pdf",
#       "file_type": "pdf",
#       "chunk_count": 42
#     }
#   ],
#   "total_files": 1
# }

# Delete a file:
curl -X DELETE "http://localhost:5000/api/rag/file/sample.pdf?tenant_id=default"


# ============================================================================
# PART 4: TEST & VERIFY (2 minutes)
# ============================================================================

# Method 1: Manual Test
# 1. Call your Twilio number
# 2. Say something related to uploaded knowledge
# 3. Check dashboard for "Knowledge retrieved" indicator
# 4. Listen for AI suggestion using your documents

# Method 2: API Test
# Create test file
cat > test_question.txt << EOF
What is your pricing model?
EOF

# Query RAG directly (simulated)
# Note: There's no direct query endpoint; queries happen during calls

# Method 3: Check Pinecone Dashboard
# 1. Log into Pinecone console
# 2. View "subscriber-kb" index
# 3. Check "default" namespace
# 4. Should see vectors with your uploaded content


# ============================================================================
# PART 5: ADVANCED USAGE
# ============================================================================

# Multi-Tenant Setup
# For each tenant, just use a different tenant_id:

# Upload for Tenant A:
curl -X POST http://localhost:5000/api/rag/upload \
  -F "file=@tenant_a_docs.pdf" \
  -F "tenant_id=tenant_a"

# Upload for Tenant B:
curl -X POST http://localhost:5000/api/rag/upload \
  -F "file=@tenant_b_docs.pdf" \
  -F "tenant_id=tenant_b"

# Each tenant's data is isolated in separate Pinecone namespaces


# Admin Mode
# For company-wide knowledge:
ADMIN_MODE=true  # In .env
# Upload to admin index
# All tenants can access (read admin-kb index)


# Fine-Tune Retrieval Quality
# In .env:

# Retrieve more chunks:
RAG_SEARCH_TOP_K=10  # Default: 5

# Lower quality threshold:
RAG_MIN_SCORE=0.3  # Default: 0.5 (0=any, 1=perfect match)

# Adjust chunk size:
RAG_CHUNK_SIZE=2000  # Larger = broader context
RAG_CHUNK_SIZE=800   # Smaller = more precise

# More overlap for continuity:
RAG_CHUNK_OVERLAP=300  # Default: 150


# ============================================================================
# PART 6: TROUBLESHOOTING
# ============================================================================

# Problem: "Pinecone API key not found"
# Solution: Check .env has PINECONE_API_KEY=xxx (no spaces)

# Problem: "OpenAI API key not found"
# Solution: Check .env has OPENAI_API_KEY=xxx (no spaces)

# Problem: Upload returns error "Unsupported file format"
# Solution: Check file extension matches supported types
#          Use .pdf not .PDF (lowercase required)

# Problem: Poor suggestion quality
# Solution:
#   - Try RAG_MIN_SCORE=0.4
#   - Increase RAG_SEARCH_TOP_K=10
#   - Adjust RAG_CHUNK_SIZE (try 1000 or 2000)
#   - Ensure files have relevant content

# Problem: Embeddings very slow
# Solution:
#   - Normal: ~50-100ms per query
#   - If >500ms: Check internet connection
#   - Batch uploads instead of single uploads

# Problem: High Pinecone bill
# Solution:
#   - Delete unused files: DELETE /api/rag/file/{name}
#   - Smaller chunks: RAG_CHUNK_SIZE=1000
#   - Batch operations
#   - Use Pinecone serverless (cheaper)


# ============================================================================
# PART 7: MIGRATION FROM LOCAL KB
# ============================================================================

# If you previously used LocalKnowledgeBase (knowledge/ folder):

# Step 1: List your old files
ls knowledge/
# Example output:
# product_guide.md
# pricing.txt
# case_studies.pdf

# Step 2: Bulk upload all files
for file in knowledge/*; do
  curl -X POST http://localhost:5000/api/rag/upload \
    -F "file=@$file" \
    -F "tenant_id=default"
  echo "Uploaded: $file"
done

# Step 3: Verify in Pinecone
curl "http://localhost:5000/api/rag/files?tenant_id=default"

# Step 4: Update config to disable local KB
# In .env or config:
RAG_ENABLED=true  # Use Pinecone
# RAG_ENABLED=false  # Keep using local (backup)

# Step 5: Archive old files
mv knowledge knowledge.backup


# ============================================================================
# PART 8: PERFORMANCE BENCHMARKS
# ============================================================================

# Upload Performance:
# - Single file: 100-500ms (depends on size)
# - PDF: ~50ms per page (1-2MB files)
# - CSV: ~100ms per 1000 rows
# - Batch: ~100 chunks/second

# Query Performance:
# - Embedding generation: ~50-100ms
# - Vector search: ~20-50ms
# - Chunk retrieval: ~10-20ms
# - Total latency: 100-200ms added to suggestions

# Cost (Approximate):
# - Upload: $0.001-0.01 per file (OpenAI embeddings)
# - Query: <$0.0001 per query (very cheap)
# - Storage: $0.04-0.08 per 100K vectors/month (Pinecone)
# - Total: $20-50/month for typical SaaS


# ============================================================================
# PART 9: NEXT STEPS
# ============================================================================

# Immediate:
# ✅ Complete setup in Part 1
# ✅ Integrate in Part 2
# ✅ Upload first document in Part 3
# ✅ Test with real call in Part 4

# Within a week:
# ✅ Upload all customer documentation
# ✅ Test with multiple customers
# ✅ Fine-tune RAG parameters
# ✅ Monitor Pinecone costs

# Production:
# ✅ Set up monitoring/logging
# ✅ Configure backups
# ✅ Plan indexing strategy for multiple tenants
# ✅ Create admin dashboard for tenant management
# ✅ Implement audit logging for knowledge access

# ============================================================================
# PART 10: GETTING HELP
# ============================================================================

# Pinecone Docs: https://docs.pinecone.io/
# OpenAI Docs: https://platform.openai.com/docs/
# This Project Docs: See RAG_ARCHITECTURE.md, RAG_INTEGRATION_GUIDE.md

# Common Issues & Solutions: See RAG_ARCHITECTURE.md "Troubleshooting"
# Integration Details: See RAG_INTEGRATION_GUIDE.md
"""
