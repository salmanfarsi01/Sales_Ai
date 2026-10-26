import os
import time
import json
import hashlib
import random
from datetime import datetime, timezone
import dotenv
from pinecone import Pinecone

dotenv.load_dotenv('C:/dev/Sales_Ai/.env', override=True)
pk = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pk)
idx = pc.Index('subscriber-kb')

def get_vec(text):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(1536)]
    norm = sum(x**2 for x in raw) ** 0.5
    return [x / norm for x in raw]

# Upsert John chunk
idx.upsert(
    vectors=[(
        'chunk_westlake_sales',
        get_vec('I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market.'),
        {
            'text': 'I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market.',
            'source_title': 'Westlake Production Record',
            'owner_user_id': 'rep_john',
            'owner_scope': 'user',
        }
    )],
    namespace='tenant_production'
)

# Upsert Sarah chunk into same tenant namespace
idx.upsert(
    vectors=[(
        'chunk_sarah_rebate',
        get_vec('Exclusive $5,000 cash back buyer rebate given at closing.'),
        {
            'text': 'Exclusive $5,000 cash back buyer rebate given at closing.',
            'source_title': 'Sarah VIP Buyer Rebate',
            'owner_user_id': 'rep_sarah',
            'owner_scope': 'user',
        }
    )],
    namespace='tenant_production'
)
time.sleep(3)

query_text = 'What is your track record in Westlake?'
filter_used = {'owner_user_id': {'$eq': 'rep_john'}, 'owner_scope': {'$eq': 'user'}}

t0 = time.perf_counter()
res = idx.query(
    vector=get_vec('I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market.'),
    namespace='tenant_production',
    top_k=5,
    include_metadata=True,
    filter=filter_used,
)
lat = (time.perf_counter() - t0) * 1000

log_entry = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'index_name': 'subscriber-kb',
    'namespace': 'tenant_production',
    'query_text': query_text,
    'filter_used': filter_used,
    'latency_ms': round(lat, 2),
    'matches_returned': len(res.matches),
    'matches': [
        {
            'id': m.id,
            'score': round(m.score, 4),
            'source_title': m.metadata.get('source_title'),
            'owner_user_id': m.metadata.get('owner_user_id'),
            'owner_scope': m.metadata.get('owner_scope'),
            'text_snippet': m.metadata.get('text', '')[:60],
        }
        for m in res.matches
    ]
}
print(json.dumps(log_entry, indent=2))

# Also demonstrate Sarah's query (confirming John's chunk is filtered out)
res_sarah = idx.query(
    vector=get_vec('I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market.'),
    namespace='tenant_production',
    top_k=5,
    include_metadata=True,
    filter={'owner_user_id': {'$eq': 'rep_sarah'}, 'owner_scope': {'$eq': 'user'}},
)
print("\n--- Sarah Query with filter {'owner_user_id': 'rep_sarah', 'owner_scope': 'user'} ---")
print(f"Matches for Sarah when querying John's text: {[m.id for m in res_sarah.matches]}")
