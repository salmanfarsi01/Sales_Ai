import os
import sys
import json
from copilot.conditional_retrieval import KnowledgeStore, ConditionalRetrievalEngine
from copilot.conditional_retrieval_router import seed_canonical_dummy_documents

store = KnowledgeStore()
seed_canonical_dummy_documents(store, user_id="rep_john")
engine = ConditionalRetrievalEngine(store=store)

canonical_lines = [
    "Hi John, thanks for calling. We're thinking about selling.",
    "What is your actual sales experience and track record in Westlake?",
    "Okay, makes sense. That sounds pretty good.",
    "What is your standard commission fee to list our home?",
    "Another agent told me Sarah offers a $5,000 cash back rebate. Can you match that?",
    "Do you guarantee you can sell it in 7 days? I heard someone claim that.",
    "What are your consultation hours? Can we meet tomorrow afternoon at 3:00 PM?",
    "How long does the residential home inspection contingency take?",
]

for idx, line in enumerate(canonical_lines, 1):
    candidates = store.search_candidates(
        query=line,
        user_id="rep_john",
        top_k=5,
        min_score=0.0
    )
    print(f"--- Turn {idx}: '{line}' ---")
    if not candidates:
        print("  No candidate chunks matched keywords/tokens.")
    for rec, score, raw_sem in candidates[:3]:
        print(f"  Chunk: {rec.knowledge_id} | Raw Semantic: {raw_sem:.4f} | Weighted Score: {score:.4f} | Title: {rec.source_title}")
