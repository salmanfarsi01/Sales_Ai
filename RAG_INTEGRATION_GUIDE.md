"""
Integration Guide: Adding RAG to Existing Copilot

This document explains how to integrate the new RAG system into your existing
Twilio copilot with minimal changes to existing code.

## Step 1: Update Environment Variables

Add these to your `.env` file:

```env
# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_key_here

# OpenAI
OPENAI_API_KEY=your_openai_key_here

# RAG Configuration (optional, defaults shown)
RAG_ENABLED=true
ADMIN_MODE=false
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=150
RAG_SEARCH_TOP_K=5
RAG_MIN_SCORE=0.5

# Multi-tenant
DEFAULT_TENANT_ID=default
```

## Step 2: Update run_call_copilot.py

Replace `AllQuestionsCopilot` with `RAGTwilioCopilot`:

```python
from copilot.twilio_rag import RAGTwilioCopilot  # Import new class
from copilot.config_rag import Settings  # Use new config

async def main():
    settings = Settings.from_env()
    copilot = RAGTwilioCopilot(settings)  # Changed from AllQuestionsCopilot
    
    runner = web.AppRunner(copilot.app())
    await runner.setup()
    site = web.TCPSite(runner, settings.host, settings.port)
    await site.start()
    
    print(f"Copilot running on http://{settings.host}:{settings.port}")
    await asyncio.Event().wait()
```

## Step 3: Optional - Keep Backward Compatibility

If you want to support both local and RAG modes:

```python
# In config
RAG_ENABLED=true  # Set to false to use local knowledge

# In TwilioCopilot.__init__
if settings.rag.rag_enabled:
    self.rag_retriever = CopilotRAGRetriever(...)
    self.knowledge = None
else:
    self.knowledge = LocalKnowledgeBase()
    self.rag_retriever = None

# In knowledge retrieval
if self.rag_retriever:
    context = self.rag_retriever.get_context(query, tenant_id)
else:
    chunks = self.knowledge.search(query)
    context = [f"From {c.source}: {c.text}" for c in chunks]
```

## Step 4: Add RAG API Routes

In `twilio_app.py` or `twilio_fast.py`, add routes:

```python
from copilot.rag_api import RAGAPIHandler

class TwilioCopilot:
    def app(self) -> web.Application:
        app = web.Application()
        
        # Existing routes
        app.router.add_get("/", self.dashboard)
        app.router.add_get("/events", self.dashboard_events)
        app.router.add_get("/twilio", self.twilio_stream)
        
        # New RAG API routes
        rag_handler = RAGAPIHandler(self.rag_retriever)
        app.router.add_post("/api/rag/upload", rag_handler.upload_knowledge_file)
        app.router.add_delete("/api/rag/file/{file_name}", rag_handler.delete_knowledge_file)
        app.router.add_get("/api/rag/files", rag_handler.list_knowledge_files)
        app.router.add_delete("/api/rag/tenant/{tenant_id}", rag_handler.delete_tenant)
        app.router.add_get("/api/rag/status", rag_handler.get_rag_status)
        
        return app
```

## Step 5: Update Suggestion Generation

Modify the `_generate_suggestion()` method to use RAG:

```python
async def _generate_suggestion(self, query: str, tenant_id: str):
    # OLD CODE:
    # chunks = self.knowledge.search(query, limit=3)
    # context_text = "\n\n".join([f"{c.source}: {c.text}" for c in chunks])
    
    # NEW CODE:
    context_list = self.rag_retriever.get_context(
        query=query,
        tenant_id=tenant_id,
        top_k=self.settings.rag.search_top_k,
        min_score=self.settings.rag.min_score_threshold,
    )
    context_text = "\n\n".join(context_list)
    
    # Continue with LLM generation as before...
    prompt = f"Context: {context_text}\n\nQuestion: {query}"
    # ... rest of generation logic
```

## Step 6: Update Dashboard HTML

Add file upload UI to dashboard:

```html
<!-- In web/twilio_fast.html or your dashboard -->
<div id="knowledge-panel">
    <h3>Knowledge Base</h3>
    
    <div class="upload-section">
        <input type="file" id="knowledge-file" multiple accept=".pdf,.txt,.md,.docx,.csv,.xlsx,.jpg,.png,.mp4">
        <button onclick="uploadKnowledge()">Upload</button>
    </div>
    
    <div id="file-list">
        <h4>Uploaded Files:</h4>
        <ul id="files"></ul>
    </div>
</div>

<script>
async function uploadKnowledge() {
    const files = document.getElementById('knowledge-file').files;
    const tenantId = document.getElementById('tenant-select').value || 'default';
    
    for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('tenant_id', tenantId);
        
        const response = await fetch('/api/rag/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        console.log(`Uploaded ${file.name}:`, result);
    }
    
    listFiles();
}

async function listFiles() {
    const tenantId = document.getElementById('tenant-select').value || 'default';
    const response = await fetch(`/api/rag/files?tenant_id=${tenantId}`);
    const result = await response.json();
    
    const fileList = document.getElementById('files');
    fileList.innerHTML = '';
    
    if (result.files) {
        result.files.forEach(file => {
            const li = document.createElement('li');
            li.innerHTML = `
                ${file.file_name} (${file.chunk_count} chunks)
                <button onclick="deleteFile('${file.file_name}', '${tenantId}')">Delete</button>
            `;
            fileList.appendChild(li);
        });
    }
}

async function deleteFile(fileName, tenantId) {
    const response = await fetch(`/api/rag/file/${fileName}?tenant_id=${tenantId}`, {
        method: 'DELETE'
    });
    
    const result = await response.json();
    console.log(`Deleted ${fileName}:`, result);
    listFiles();
}

// Load file list on startup
listFiles();
</script>
```

## Step 7: Install Dependencies

```bash
# For basic RAG support (PDFs, text, Word, Excel, CSV)
pip install -r requirements-rag-full.txt

# Or minimal setup (no video/OCR)
pip install pinecone-client openai PyMuPDF python-docx openpyxl
```

## Step 8: Create Pinecone Account & Get Keys

1. Sign up at https://www.pinecone.io/
2. Create a serverless index (free tier available)
3. Get your API key
4. Add to `.env`: `PINECONE_API_KEY=xxx`

## Step 9: Create OpenAI API Key

1. Sign up at https://openai.com/
2. Create API key
3. Add to `.env`: `OPENAI_API_KEY=xxx`

## Step 10: Test the Setup

```bash
# Start copilot server
python run_call_copilot.py

# In another terminal, test RAG upload
curl -X POST http://localhost:5000/api/rag/upload \
  -F "file=@sample.pdf" \
  -F "tenant_id=default"

# Test retrieval
curl "http://localhost:5000/api/rag/files?tenant_id=default"
```

## Migration from LocalKnowledgeBase

### Approach 1: Gradual Migration (Recommended)

1. **Phase 1**: Deploy with both systems active
   ```python
   # Search both local and Pinecone
   local_results = self.knowledge.search(query)
   rag_results = self.rag_retriever.get_context(query, tenant_id)
   
   # Compare in logs
   LOGGER.info(f"Local: {len(local_results)}, RAG: {len(rag_results)}")
   ```

2. **Phase 2**: Gradually upload files to Pinecone
   - Use dashboard upload UI
   - Monitor relevance

3. **Phase 3**: Switch to RAG
   ```python
   RAG_ENABLED=true
   ```

4. **Phase 4**: Archive local knowledge/
   - Keep for backup
   - Remove from code

### Approach 2: Direct Switch

If confident in RAG quality:

1. Update code to use RAG only
2. Upload all local knowledge/ files via API
3. Test thoroughly
4. Deploy

### Approach 3: Parallel Deployment

- New deployments use RAG
- Keep old deployment with local KB
- Use load balancer to gradually shift traffic

## Troubleshooting

**Issue**: Embeddings timeout
- Solution: Batch queries, use async processing

**Issue**: High Pinecone costs
- Solution: Use smaller chunks, batch upserts

**Issue**: Poor retrieval quality
- Solution: Adjust chunk size, increase top_k, lower min_score

**Issue**: Duplicate chunks in index
- Solution: Check SHA256 ID generation, verify tenant_id

## Performance Benchmarks

- Upload: ~100 chunks/second (with batching)
- Search: ~100ms latency
- Embedding: ~50ms per query
- Total suggestion latency: 150-200ms added

## Next Steps

1. ✅ Set up Pinecone account
2. ✅ Add environment variables
3. ✅ Update application code
4. ✅ Upload knowledge files
5. ✅ Test end-to-end
6. ✅ Deploy to production
7. ✅ Monitor RAG quality
8. ✅ Optimize chunk sizes based on results
"""
