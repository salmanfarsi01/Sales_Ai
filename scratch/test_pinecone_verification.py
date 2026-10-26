import os
import time
import random
import dotenv
from pinecone import Pinecone

dotenv.load_dotenv("C:/dev/Sales_Ai/.env")
pc_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pc_key)
idx = pc.Index("subscriber-kb")

print("=== 1. Index Stats BEFORE ===")
stats_before = idx.describe_index_stats()
print("Total vectors:", stats_before.total_vector_count)
print("Namespaces:", stats_before.namespaces)

# Ingest test chunk for rep_john with 1536-dim vector
test_chunk_id = "chunk_westlake_sales_live"
text = "I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market."
random.seed(42)
raw_vec = [random.gauss(0, 1) for _ in range(1536)]
norm = sum(x**2 for x in raw_vec) ** 0.5
vector = [x / norm for x in raw_vec]

meta = {
    "text": text,
    "source_title": "Westlake Production Record",
    "source_version": "1.0",
    "knowledge_class": "performance_past_sales",
    "verification_status": "company_approved",
    "sensitivity": "normal",
    "owner_user_id": "rep_john",
    "owner_id": "rep_john",
    "scope": "user",
    "owner_scope": "user",
    "topic_tags": ["westlake", "sales", "experience"],
}

idx.upsert(vectors=[(test_chunk_id, vector, meta)], namespace="rep_john")
time.sleep(3)

print("\n=== 2. Index Stats AFTER ===")
stats_after = idx.describe_index_stats()
print("Total vectors:", stats_after.total_vector_count)
print("Namespaces:", stats_after.namespaces)

print("\n=== 3. Fetched Vector by ID with Metadata ===")
fetched = idx.fetch(ids=[test_chunk_id], namespace="rep_john")
vec_data = fetched.vectors.get(test_chunk_id)
if vec_data:
    print("Vector ID:", vec_data.id)
    print("Metadata keys:", list(vec_data.metadata.keys()))
    print("Metadata:", vec_data.metadata)

print("\n=== 4. Live Query on Pinecone with Filter ===")
t1 = time.perf_counter()
query_res = idx.query(
    vector=vector,
    namespace="rep_john",
    top_k=3,
    include_metadata=True,
    filter={"$and": [{"owner_user_id": "rep_john"}, {"owner_scope": "user"}]},
)
t_search = (time.perf_counter() - t1) * 1000

print(f"Pinecone query latency: {t_search:.1f}ms")
for match in query_res.matches:
    print(f"Match ID: {match.id} | Score: {match.score:.4f} | Title: {match.metadata.get('source_title')}")
