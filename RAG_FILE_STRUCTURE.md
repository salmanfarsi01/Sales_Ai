"""
RAG System - Complete File Structure Reference

This document shows all RAG-related files and their purpose.
"""

# ============================================================================
# CORE RAG FILES (NEW)
# ============================================================================

copilot/file_extraction.py
├─ Purpose: Multi-format file parsing
├─ Supports: PDF, TXT, MD, DOCX, DOC, CSV, XLSX, XLS, JPG, PNG, MP4, MP3
├─ Main Class: FileExtractor
├─ Key Methods:
│  ├─ extract(file_path, file_content) → ExtractedContent
│  ├─ _extract_pdf()
│  ├─ _extract_docx()
│  ├─ _extract_excel()
│  ├─ _extract_image() [OCR]
│  └─ _extract_video() [metadata]
└─ Usage: Called by upload pipeline to parse files

copilot/rag_pinecone.py
├─ Purpose: Direct Pinecone integration
├─ Main Class: PineconeRAG
├─ Handles:
│  ├─ Multi-tenant index management (admin-kb, subscriber-kb)
│  ├─ Chunk generation and embedding
│  ├─ Upsert to Pinecone with namespaces
│  ├─ Vector similarity search
│  ├─ File and tenant lifecycle
├─ Key Methods:
│  ├─ upload_file(file_path, tenant_id, file_content)
│  ├─ search(query, tenant_id, top_k, min_score)
│  ├─ delete_file(file_name, tenant_id)
│  ├─ delete_tenant_data(tenant_id)
│  └─ list_tenant_files(tenant_id)
└─ Internal:
   ├─ _get_embedding(text) - calls OpenAI
   ├─ _generate_chunk_id(tenant_id, file_name, chunk_idx)
   ├─ _split_into_chunks(text, chunk_size, overlap)
   └─ _get_or_create_index()

copilot/rag_integration.py
├─ Purpose: High-level RAG API for copilot
├─ Main Class: CopilotRAGRetriever
├─ Wraps PineconeRAG for simpler usage
├─ Key Methods:
│  ├─ get_context(query, tenant_id) → List[str]
│  ├─ upload_knowledge(file_path, tenant_id, file_content)
│  ├─ delete_knowledge_file(file_name, tenant_id)
│  ├─ list_knowledge_files(tenant_id)
│  └─ delete_tenant(tenant_id)
└─ Usage: Replace LocalKnowledgeBase in copilot

copilot/rag_api.py
├─ Purpose: HTTP REST API endpoints
├─ Main Class: RAGAPIHandler
├─ Endpoints:
│  ├─ POST   /api/rag/upload
│  ├─ DELETE /api/rag/file/{file_name}
│  ├─ GET    /api/rag/files
│  ├─ DELETE /api/rag/tenant/{tenant_id}
│  └─ GET    /api/rag/status
└─ Used by: Dashboard frontend, mobile apps, admin panel

copilot/twilio_rag.py
├─ Purpose: Complete copilot with RAG integration
├─ Main Class: RAGTwilioCopilot (extends FastTwilioCopilot)
├─ Replaces: LocalKnowledgeBase with RAG
├─ Changes:
│  ├─ Initializes CopilotRAGRetriever
│  ├─ Adds RAG API routes
│  ├─ Overrides _get_knowledge_context() to use RAG
│  ├─ Overrides _generate_suggestion() to use RAG context
│  └─ Backward compatible with LocalKnowledgeBase if RAG_ENABLED=false
└─ Usage: Drop-in replacement for existing copilot

copilot/config_rag.py
├─ Purpose: Configuration for RAG system
├─ Adds: RAGSettings dataclass with Pinecone config
├─ Environment Variables:
│  ├─ PINECONE_API_KEY
│  ├─ OPENAI_API_KEY
│  ├─ RAG_ENABLED
│  ├─ ADMIN_MODE
│  ├─ RAG_CHUNK_SIZE
│  ├─ RAG_CHUNK_OVERLAP
│  ├─ RAG_SEARCH_TOP_K
│  ├─ RAG_MIN_SCORE
│  └─ DEFAULT_TENANT_ID
└─ Extends: Original Settings class for backward compatibility

# ============================================================================
# TEST FILES (NEW)
# ============================================================================

tests/test_file_extraction.py
├─ Tests for FileExtractor
├─ Classes:
│  ├─ TestPDFExtraction
│  ├─ TestTextExtraction
│  ├─ TestCSVExtraction
│  ├─ TestUnsupportedFormat
│  └─ TestBlobExtraction
└─ Run: pytest tests/test_file_extraction.py -v

tests/test_rag_pinecone.py
├─ Tests for PineconeRAG
├─ Classes:
│  ├─ TestPineconeRAG
│  └─ TestRAGChunk
├─ Mocks: Pinecone, OpenAI clients
└─ Run: pytest tests/test_rag_pinecone.py -v

tests/test_rag_integration.py
├─ Tests for CopilotRAGRetriever
├─ Classes:
│  ├─ TestTenantContext
│  └─ TestCopilotRAGRetriever
└─ Run: pytest tests/test_rag_integration.py -v

# ============================================================================
# DOCUMENTATION FILES (NEW)
# ============================================================================

RAG_ARCHITECTURE.md
├─ Complete architecture overview
├─ Pipeline diagrams and flows
├─ Multi-tenant strategy
├─ File extraction reference
├─ Integration guide
├─ Security considerations
├─ Cost breakdown
└─ Troubleshooting guide

RAG_INTEGRATION_GUIDE.md
├─ Step-by-step integration instructions
├─ Environment setup
├─ Code changes required
├─ Dashboard UI updates
├─ Migration strategies
├─ Performance benchmarks
└─ Troubleshooting

RAG_QUICKSTART.md
├─ Quick 15-minute setup
├─ Part-by-part walkthrough
├─ Common tasks (upload, list, delete)
├─ Multi-tenant examples
├─ Advanced configurations
├─ Troubleshooting quick reference

requirements-rag-full.txt
├─ All RAG dependencies
├─ Optional file extraction libraries
├─ Development and testing packages
└─ Usage: pip install -r requirements-rag-full.txt

# ============================================================================
# EXISTING FILES (MODIFIED)
# ============================================================================

run_call_copilot.py
├─ Change: Import RAGTwilioCopilot instead of AllQuestionsCopilot
├─ Change: Use config_rag.Settings instead of config.Settings
└─ Comment: Can keep backward compatible with conditional import

.env (Example)
├─ Add: PINECONE_API_KEY
├─ Add: OPENAI_API_KEY
├─ Add: RAG_ENABLED
├─ Add: ADMIN_MODE
├─ Add: RAG_CHUNK_SIZE
└─ Add: RAG_SEARCH_TOP_K, RAG_MIN_SCORE, RAG_CHUNK_OVERLAP, DEFAULT_TENANT_ID

# ============================================================================
# ARCHITECTURE SUMMARY
# ============================================================================

┌─────────────────────────────────────────────────────────┐
│                  Dashboard / API Client                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         RAGAPIHandler (rag_api.py)                      │
│  - /api/rag/upload                                      │
│  - /api/rag/file/{name}                                 │
│  - /api/rag/files                                       │
│  - /api/rag/status                                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│     CopilotRAGRetriever (rag_integration.py)            │
│  - get_context()                                        │
│  - upload_knowledge()                                   │
│  - delete_knowledge_file()                              │
│  - list_knowledge_files()                               │
└────────────────┬────────────────────────────────────────┘
                 │
      ┌──────────┼──────────┐
      ▼                     ▼
 ┌─────────┐         ┌──────────────────┐
 │FileExtr │         │PineconeRAG       │
 │actor    │         │(rag_pinecone.py) │
 └────┬────┘         └────────┬─────────┘
      │                       │
      │ PDF,TXT,XLS           │
      │ DOCX,CSV,IMG          │
      │                       │ Chunks + 
      ├──────────┬────────────┤ Embeddings
      │          ▼            │
      │     ┌─────────────────┴─────┐
      │     │  OpenAI Embeddings    │
      │     │  text-embedding-      │
      │     │  3-small (1536-dim)   │
      │     └───────────┬───────────┘
      │                 │
      └─────────────────┼──────────────┐
                        │              │
                        ▼              ▼
                   ┌──────────────────────────┐
                   │  Pinecone Index          │
                   ├──────────────────────────┤
                   │ admin-kb   subscriber-kb │
                   │ [global]   [tenant-A]    │
                   │ [admin]    [tenant-B]    │
                   │            [tenant-C]    │
                   └──────────────────────────┘

# ============================================================================
# DEPENDENCY TREE
# ============================================================================

RAGTwilioCopilot
├─ Extends: FastTwilioCopilot
│  ├─ Deepgram (transcription)
│  └─ Groq (LLM generation)
│
├─ CopilotRAGRetriever
│  └─ PineconeRAG
│     ├─ pinecone-client (vector DB)
│     ├─ OpenAI API (embeddings)
│     │  └─ openai (library)
│     │
│     └─ FileExtractor
│        ├─ PyMuPDF (PDF)
│        ├─ python-docx (DOCX)
│        ├─ openpyxl (Excel)
│        ├─ csv (CSV - builtin)
│        ├─ pytesseract (OCR)
│        ├─ Pillow (images)
│        └─ moviepy (video metadata)
│
└─ RAGAPIHandler
   └─ RAGTwilioCopilot.rag_retriever

# ============================================================================
# DATA FLOW DURING CALLS
# ============================================================================

Call arrives
    ↓
Audio split by speaker
    ↓
Deepgram transcription
    ↓
Client speech detected?
    ├─ YES → Context Recovery (optional)
    └─ NO → Skip RAG
           ↓
         Query embedding (OpenAI)
           ↓
         Pinecone search with namespace
           ↓
         Retrieve top-5 chunks
           ↓
         Format context
           ↓
         Groq LLM generation with context
           ↓
         Stream response to dashboard
           ↓
         Log to call report

# ============================================================================
# FILE UPLOAD FLOW
# ============================================================================

User uploads file (web/API)
    ↓
RAGAPIHandler.upload_knowledge_file()
    ↓
FileExtractor.extract()
    ├─ Detect format
    ├─ Parse content
    ├─ Extract text + metadata
    └─ Return ExtractedContent
    ↓
PineconeRAG.upload_file()
    ├─ Split into chunks (1500 char + overlap)
    ├─ For each chunk:
    │  ├─ Generate embedding (OpenAI)
    │  ├─ Create chunk ID (SHA256)
    │  └─ Prepare vector record
    └─ Batch upsert to Pinecone (namespace=tenant_id)
    ↓
Return status: chunks_uploaded, total_size, file_type
    ↓
Dashboard shows: "✅ 42 chunks uploaded from sample.pdf"
"""
