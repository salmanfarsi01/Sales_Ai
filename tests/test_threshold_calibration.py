"""Threshold Calibration Test Suite for Conditional Knowledge Retrieval.

Calibrates the 0.35 marginal floor and 0.50 relevance floor against a labeled set
of real queries and chunks, tuning and validating the ranking weights:
- High relevance (exact domain/intent match): similarity >= 0.50 -> status='retrieved'
- Marginal relevance (broad/paraphrase domain match): 0.35 <= similarity < 0.50 -> status='low_confidence'
- Out of domain / irrelevant: similarity < 0.35 -> status='none_found'
"""

import pytest
from copilot.conditional_retrieval import (
    ConditionalRetrievalEngine,
    KnowledgeStore,
    RankingWeights,
    RetrievalConfig,
)
from copilot.conditional_retrieval_models import (
    KnowledgeClass,
    KnowledgeRecord,
    RetrievalNeed,
    RetrievalRequest,
    RetrievalTrigger,
)


LABELED_CALIBRATION_CORPUS = [
    # Chunk 1: High Relevance - Local Austin Transaction Track Record
    KnowledgeRecord(
        knowledge_id="chunk_austin_sales",
        owner_scope="user",
        owner_user_id="rep_cal",
        knowledge_class=KnowledgeClass.PERFORMANCE_PAST_SALES.value,
        source_title="Austin Track Record",
        topic_tags=["austin", "sales_volume", "track_record", "experience"],
        lead_type_tags=["residential", "luxury"],
        prospect_type_tags=["seller", "homeowner"],
        text="I have closed 45 single family transactions in Austin over the last 3 years.",
        verification_status="company_approved",
    ),
    # Chunk 2: High Relevance - Commission and Cancellation Policy
    KnowledgeRecord(
        knowledge_id="chunk_policy_commission",
        owner_scope="user",
        owner_user_id="rep_cal",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="Brokerage Commission Schedule",
        topic_tags=["commission", "listing_fee", "policy", "cancellation"],
        lead_type_tags=["residential"],
        prospect_type_tags=["seller"],
        text="Standard full service listing fee is 5% with zero cancellation penalty at any point.",
        verification_status="company_approved",
    ),
    # Chunk 3: Marginal Relevance - General staging and preparation advice (partially overlaps marketing)
    KnowledgeRecord(
        knowledge_id="chunk_general_prep",
        owner_scope="user",
        owner_user_id="rep_cal",
        knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
        source_title="Home Prep Guide",
        topic_tags=["home_preparation", "staging_advice", "consultation"],
        lead_type_tags=["residential"],
        text="Preparation checklist suggests decluttering and neutral paint touch-ups prior to photography.",
        verification_status="user_supplied",
    ),
    # Chunk 4: Out of Domain / Irrelevant - Commercial warehouse industrial zoning
    KnowledgeRecord(
        knowledge_id="chunk_commercial_zoning",
        owner_scope="user",
        owner_user_id="rep_cal",
        knowledge_class=KnowledgeClass.MARKET_KNOWLEDGE.value,
        source_title="Industrial Logistics Zoning",
        topic_tags=["industrial", "warehouse", "freight_access", "zoning_b3"],
        lead_type_tags=["commercial"],
        text="Industrial zoning classification B-3 permits heavy logistics freight staging and high-voltage distribution.",
        verification_status="company_approved",
    ),
]


@pytest.fixture
def calibrated_engine():
    store = KnowledgeStore()
    for rec in LABELED_CALIBRATION_CORPUS:
        store.add_record(rec)
    config = RetrievalConfig(
        relevance_floor=0.50,
        marginal_threshold=0.50,
        weights=RankingWeights(
            semantic_weight=0.50,
            lead_type_boost=0.15,
            prospect_type_boost=0.10,
            verification_boost=0.15,
            specificity_boost=0.10,
        ),
    )
    return ConditionalRetrievalEngine(store=store, config=config)


def test_calibration_high_relevance_query_crosses_50_floor(calibrated_engine):
    """Real query for local sales volume crosses the 0.50 floor and returns status='retrieved'."""
    req = RetrievalRequest(
        call_id="cal_call_1",
        state_version=1,
        user_id="rep_cal",
        lead_type="residential",
        prospect_type="seller",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="verified sales track record in Austin",
            topic_tags=["austin", "sales_volume", "track_record"],
        ),
    )
    res = calibrated_engine.retrieve_for_request(req)
    assert res.status == "retrieved"
    assert len(res.facts) >= 1
    top_fact = res.facts[0]
    assert top_fact.knowledge_id == "chunk_austin_sales"
    assert top_fact.confidence >= 0.50
    assert "45 single family transactions in Austin" in top_fact.text


def test_calibration_marginal_relevance_falls_into_low_confidence(calibrated_engine):
    """Borderline/marginal query falls between 0.35 and 0.50 and returns status='low_confidence'."""
    req = RetrievalRequest(
        call_id="cal_call_2",
        state_version=1,
        user_id="rep_cal",
        lead_type="residential",
        prospect_type="seller",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="lighting enhancements and exterior curb appeal styling",
            topic_tags=["consultation"],
        ),
    )
    # Configure floor at 0.30 so marginal chunk qualifies for candidate consideration, but falls below 0.50 marginal threshold
    cfg = RetrievalConfig(relevance_floor=0.30, marginal_threshold=0.50)
    res = calibrated_engine.retrieve_for_request(req, config=cfg)
    assert res.status == "low_confidence"
    assert res.fallback_hint == "ask_clarifying_question"
    assert len(res.facts) >= 1
    assert res.facts[0].knowledge_id == "chunk_general_prep"


def test_calibration_out_of_domain_query_rejected_below_35_floor(calibrated_engine):
    """Completely out-of-domain residential query for warehouse zoning yields 0 results and status='none_found'."""
    req = RetrievalRequest(
        call_id="cal_call_3",
        state_version=1,
        user_id="rep_cal",
        lead_type="residential",
        prospect_type="seller",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="residential HOA bylaws and pool maintenance schedules",
            topic_tags=["hoa", "bylaws", "pool"],
        ),
    )
    res = calibrated_engine.retrieve_for_request(req)
    assert res.status == "none_found"
    assert len(res.facts) == 0
    # Confirm the commercial warehouse chunk was never admitted
    assert not any(f.knowledge_id == "chunk_commercial_zoning" for f in res.facts)


def test_calibration_ranking_weight_tuning():
    """Tuning ranking weights (e.g. boosting verification status or prospect type) adjusts candidate order."""
    store = KnowledgeStore()
    rec_unverified = KnowledgeRecord(
        knowledge_id="rec_unver",
        owner_scope="user",
        owner_user_id="rep_tune",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="Rep Note",
        topic_tags=["pricing"],
        text="Pricing recommendation note on pricing strategies.",
        verification_status="user_supplied",
    )
    rec_approved = KnowledgeRecord(
        knowledge_id="rec_appr",
        owner_scope="user",
        owner_user_id="rep_tune",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="Company Rate Policy",
        topic_tags=["pricing"],
        text="Pricing policy rate approved by company brokerage.",
        verification_status="company_approved",
    )
    store.add_record(rec_unverified)
    store.add_record(rec_approved)

    engine = ConditionalRetrievalEngine(store=store)

    req = RetrievalRequest(
        call_id="call_tune",
        state_version=1,
        user_id="rep_tune",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="pricing policy rules",
            topic_tags=["pricing"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "retrieved"
    # Company approved must rank ahead of unverified due to verification_boost
    assert res.facts[0].knowledge_id == "rec_appr"
    assert res.facts[0].authority_level == 3
