"""
RAG Architecture Documentation for SaaS Platform

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│     Pinecone Vector Database (SaaS)        │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │  ADMIN INDEX │      │ SUBSCRIBER   │   │
│  │              │      │    INDEX     │   │
│  │ Admin's own  │      │              │   │
│  │   Knowledge  │      │ All tenants' │   │
│  │              │      │  knowledge   │   │
│  └──────────────┘      └──────┬───────┘   │
│                               │           │
│                    ┌──────────┼──────────┐ │
│                    │          │          │ │
│            ┌───────▼──┐ ┌────▼─────┐   │ │
│            │ Tenant A │ │ Tenant B │...│ │
│            │namespace │ │namespace │   │ │
│            └──────────┘ └──────────┘   │ │
│                    (Pinecone namespaces)│ │
│                                         │ │
└─────────────────────────────────────────┘
         ▲                              ▲
         │                              │
    [Upload]                      [Search/Retrieve]
         │                              │
    ┌────┴──────────────────────────────┴──┐
    │  Copilot RAG Integration Layer        │
    │  - File Extraction                    │
    │  - Chunking & Embedding               │
    │  - Search & Retrieval                 │
    └────┬────────────────────┬─────────────┘
         │                    │
    ┌────▼────┐          ┌────▼─────────┐
    │Deepgram │          │  Groq LLM    │
    │(STT)    │          │  (Generation)│
    └─────────┘          └──────────────┘
         ▲                      │
         │                      │
    ┌────┴──────────────────────▼──┐
    │   Twilio Real-Time Copilot    │
    │                                │
    │ Call Flow:                     │
    │ 1. Audio from Twilio           │
    │ 2. Transcribe with Deepgram    │
    │ 3. Extract client question     │
    │ 4. Search Pinecone for context │
    │ 5. Generate with Groq + context│
    │ 6. Stream to browser dashboard │
    └────────────────────────────────┘
```

## Multi-Tenant Strategy

### Index Organization

**ADMIN_INDEX (admin-kb)**
- Purpose: Centralized knowledge for admin/super-users
- Namespaces: 
  - Global (system-wide knowledge)
  - Per-admin (admin-specific knowledge)
- Use Case: Company-wide product docs, sales methodology

**SUBSCRIBER_INDEX (subscriber-kb)**
- Purpose: Tenant-specific knowledge
- Namespaces: One per tenant (Tenant A, Tenant B, etc.)
- Use Case: Client-specific docs, contracts, pricing, case studies
- Isolation: Complete data isolation via namespaces

### Benefits of This Design

1. **Scalability**: Subscribers share one index; admin has separate
2. **Cost Efficiency**: Single index per type = lower operational cost
3. **Data Isolation**: Pinecone namespaces provide logical separation
4. **Security**: Tenants can only query their own namespace
5. **Performance**: Larger index means more relevance signal

## File Extraction Pipeline

### Supported Formats

| Format | Handler | Extraction Method | Status |
|--------|---------|-------------------|--------|
| PDF | PyMuPDF | Text + page markers | ✅ Complete |
| TXT/MD | Built-in | Direct text read | ✅ Complete |
| DOCX | python-docx | Paragraphs + tables | ✅ Complete |
| DOC | python-docx | Convert → extract | ✅ Complete |
| CSV | csv module | Row-wise structure | ✅ Complete |
| XLSX/XLS | openpyxl | Sheet + row structure | ✅ Complete |
| JPG/PNG | pytesseract | OCR text extraction | ✅ Complete |
| MP4/WebM | moviepy | Metadata + audio track | ⚠️ Requires transcription |
| MP3/WAV | pydub | Metadata + audio track | ⚠️ Requires transcription |

### Optional Dependencies

Install based on your needs:

```bash
# PDF support
pip install PyMuPDF

# Word documents
pip install python-docx

# Excel files
pip install openpyxl

# OCR for images
pip install pytesseract pillow
# Also requires Tesseract-OCR system package

# Video/audio processing
pip install moviepy pydub

# All-in-one
pip install -r requirements-rag-full.txt
```

## Upload Workflow

```
User/Admin selects file
         │
         ▼
File extraction (FileExtractor)
├─ Detect format by extension
├─ Parse content based on type
├─ Extract text + metadata
         │
         ▼
Text chunking
├─ Split into overlapping 1500-char chunks
├─ Preserve context across boundaries
├─ Default 150-char overlap
         │
         ▼
Embedding generation
├─ Send each chunk to OpenAI text-embedding-3-small
├─ Get 1536-dimensional vectors
├─ Batch process for efficiency (100 chunks/batch)
         │
         ▼
Pinecone upsert
├─ Namespace: tenant_id
├─ Vector ID: SHA256(tenant_id + filename + chunk_idx)
├─ Metadata: source, file_type, chunk_index, etc.
├─ Store full chunk text for context
         │
         ▼
Dashboard confirmation
└─ Show chunks uploaded, file size, processing time
```

## Search & Retrieval Workflow

```
Client speaks question
         │
         ▼
Transcribed by Deepgram
         │
         ▼
Context recovery (optional)
├─ If incomplete: "what about X?"
├─ Prepend recent client speech
         │
         ▼
Pinecone search
├─ Embed query with OpenAI (same model as chunks)
├─ Cosine similarity in vector space
├─ Filter by tenant namespace
├─ Return top-5 results (configurable)
├─ Min score threshold: 0.5 (configurable)
         │
         ▼
Retrieve chunks
├─ Full chunk text from metadata
├─ Source filename
├─ File type and metadata
         │
         ▼
Format for LLM prompt
├─ "From filename.pdf: [chunk text]"
├─ Up to 3 top chunks (configurable)
├─ Preserve source attribution
         │
         ▼
Groq generation
├─ System prompt limits response to 3 sentences
├─ Include retrieved context
├─ Include conversation history
├─ Stream tokens to dashboard
         │
         ▼
Browser dashboard
└─ Display suggestion in real-time
```

## Integration with Existing Copilot

### Changes to twilio_app.py

```python
# OLD: Local knowledge base
from .retrieval import LocalKnowledgeBase
self.knowledge = LocalKnowledgeBase()

# NEW: RAG integration
from .rag_integration import CopilotRAGRetriever
self.rag_retriever = CopilotRAGRetriever(
    pinecone_api_key=settings.rag.pinecone_api_key,
    openai_api_key=settings.rag.openai_api_key,
    admin_mode=settings.rag.admin_mode,
)
```

### Changes to suggestion generation

```python
# OLD: Local search
context_chunks = self.knowledge.search(question, limit=3)

# NEW: RAG search
tenant_id = call_context.get("tenant_id", settings.default_tenant_id)
context_chunks = self.rag_retriever.get_context(
    query=question,
    tenant_id=tenant_id,
    top_k=settings.rag.search_top_k,
    min_score=settings.rag.min_score_threshold,
)
```

## Environment Configuration

Add to `.env`:

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# RAG Settings
RAG_ENABLED=true
ADMIN_MODE=false
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=150
RAG_SEARCH_TOP_K=5
RAG_MIN_SCORE=0.5

# Multi-tenant
DEFAULT_TENANT_ID=default
```

## Performance Optimizations

1. **Batch Upserts**: Upload 100 chunks at once to Pinecone
2. **Chunk Overlaps**: 150-char overlap preserves context
3. **Min Score Filtering**: Only return chunks with similarity > 0.5
4. **Namespace Filtering**: Reduce search space per tenant
5. **Metadata Indexing**: Filter by file_type, date if needed
6. **Caching Layer** (optional): Cache embeddings for common queries

## Cost Breakdown (Monthly Estimate)

**OpenAI Embeddings:**
- $0.02 per 1M tokens
- Typical: 100 pages × 500 tokens = 50K tokens = $0.001 per upload
- Search: 1 query = ~100 tokens = negligible

**Pinecone Vector DB:**
- Starter plan: ~$19/month for small scale
- Pod plan: $20-70 per pod-month (1M vectors)
- Serverless: $0.04 per 100K vectors written + usage

**Total for 10 active tenants, 100 documents each:**
- Embeddings: ~$10/month
- Pinecone: ~$40-80/month
- Total: ~$50-90/month

## Security & Isolation

1. **Tenant Isolation**:
   - Different namespaces = logical isolation
   - No cross-tenant queries by design
   - Validate tenant_id before every search

2. **API Key Security**:
   - Store in environment variables
   - Never expose Pinecone/OpenAI keys to frontend
   - Only backend can upload/search

3. **RBAC** (Future):
   - Admin mode: Can upload to admin index
   - Tenant mode: Can only search own namespace
   - User mode: Read-only access to suggestions

## Migration Path

If migrating from LocalKnowledgeBase:

1. Keep both systems during transition
2. Set RAG_ENABLED=false initially
3. Gradually migrate files to Pinecone
4. Run parallel searches (local + Pinecone) in logs
5. Compare results quality
6. Once confident, switch RAG_ENABLED=true
7. Archive local knowledge/ folder

## Testing

```bash
# Test file extraction
python -m pytest tests/test_file_extraction.py -v

# Test RAG integration
python -m pytest tests/test_rag_pinecone.py -v

# Test end-to-end with Twilio
python -m pytest tests/test_rag_integration.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Embeddings too slow | Batch multiple queries, use smaller chunks |
| Low retrieval quality | Increase top_k, lower min_score, adjust chunk size |
| High costs | Use smaller embedding model, batch operations |
| Tenant data leaking | Verify namespace filters, audit Pinecone queries |
| Duplicate chunks | Use deterministic IDs (SHA256 of content) |
"""
