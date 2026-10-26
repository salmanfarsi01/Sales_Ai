"""Live Pinecone Tenant-Namespace Isolation Test (Constraint C6).

Verifies at the Pinecone index level that when User A and User B share a tenant namespace,
mandatory in-index filter {owner_user_id: user, owner_scope: 'user'} guarantees
that User B gets ZERO of User A's vectors.
"""

import os
import time
import pytest
import dotenv
from pinecone import Pinecone

from copilot.rag_pinecone import PineconeRAG

from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
dotenv.load_dotenv(env_path, override=True)

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

@pytest.mark.skipif(not PINECONE_KEY or not OPENAI_KEY or PINECONE_KEY.startswith("mock"), reason="Pinecone/OpenAI keys not configured")
def test_pinecone_tenant_namespace_isolation_live():
    rag = PineconeRAG(
        pinecone_api_key=PINECONE_KEY,
        openai_api_key="sk-dummy",
        index_name="subscriber-kb",
    )
    import hashlib, random
    def deterministic_embedding(text: str, timeout: float = 0.35) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        raw = [rng.gauss(0, 1) for _ in range(1536)]
        norm = sum(x**2 for x in raw) ** 0.5
        return [x / norm for x in raw]

    rag._get_embedding = deterministic_embedding
    test_namespace = "tenant_test_org"

    # Clean namespace prior to test
    try:
        rag.index.delete(delete_all=True, namespace=test_namespace)
        time.sleep(2)
    except Exception:
        pass

    # Upsert Chunk A for rep_john
    rag.upsert_text_chunk(
        chunk_id="chunk_rep_john_track_record",
        text="I have closed 42 homes in Westlake over the past 3 years.",
        tenant_id=test_namespace,
        metadata={
            "owner_user_id": "rep_john",
            "owner_id": "rep_john",
            "owner_scope": "user",
            "scope": "user",
            "source_title": "John Westlake Record",
        },
    )

    # Upsert Chunk B for rep_sarah
    rag.upsert_text_chunk(
        chunk_id="chunk_rep_sarah_rebate",
        text="Exclusive $5,000 cash back buyer rebate given at closing.",
        tenant_id=test_namespace,
        metadata={
            "owner_user_id": "rep_sarah",
            "owner_id": "rep_sarah",
            "owner_scope": "user",
            "scope": "user",
            "source_title": "Sarah Buyer Rebate",
        },
    )

    # Wait for Pinecone eventual consistency
    time.sleep(3)

    # 1. Query as rep_john: must find John's chunk, and ZERO of Sarah's chunk
    john_chunks = rag.search(
        query="I have closed 42 homes in Westlake over the past 3 years.",
        tenant_id=test_namespace,
        owner_id="rep_john",
        min_score=0.50,
    )
    john_ids = [c.id for c in john_chunks]
    assert "chunk_rep_john_track_record" in john_ids
    assert "chunk_rep_sarah_rebate" not in john_ids, "Cross-user data breach: Sarah's vector leaked to John!"

    # 2. Query as rep_sarah for John's text: Pinecone filter must return ZERO of John's vectors
    sarah_cross_chunks = rag.search(
        query="I have closed 42 homes in Westlake over the past 3 years.",
        tenant_id=test_namespace,
        owner_id="rep_sarah",
        min_score=0.50,
    )
    sarah_cross_ids = [c.id for c in sarah_cross_chunks]
    assert "chunk_rep_john_track_record" not in sarah_cross_ids, "Cross-user data breach: John's vector leaked to Sarah!"

    # 3. Query as rep_sarah for Sarah's text: must find Sarah's chunk, and ZERO of John's chunk
    sarah_chunks = rag.search(
        query="Exclusive $5,000 cash back buyer rebate given at closing.",
        tenant_id=test_namespace,
        owner_id="rep_sarah",
        min_score=0.50,
    )
    sarah_ids = [c.id for c in sarah_chunks]
    assert "chunk_rep_sarah_rebate" in sarah_ids
    assert "chunk_rep_john_track_record" not in sarah_ids, "Cross-user data breach: John's vector leaked to Sarah!"

    # 4. Clean up test namespace
    rag.index.delete(delete_all=True, namespace=test_namespace)
