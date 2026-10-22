import pytest
from datetime import datetime, timezone, timedelta

from copilot.conditional_retrieval_models import (
    KnowledgeClass,
    KnowledgeRecord,
    RankingWeights,
    RetrievalConfig,
    RetrievalNeed,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrigger,
    RetrievedFact,
    WithheldFact,
)
from copilot.conditional_retrieval import (
    ConditionalRetrievalEngine,
    KnowledgeStore,
    QueryBuilder,
    RetrievalGate,
)
from copilot.conversation_state_models import ConversationStage, ConversationStateSnapshot
from copilot.core_intelligence_models import StrategicAction, StrategicDecision
from copilot.llm_response_gateway import LLMResponseGateway, validate_prompt_hallucinations


# ==============================================================================
# Phase 3: Configurable Ranking Tests (Doc 02 §7)
# ==============================================================================


def test_configurable_ranking_weights_influence_scoring():
    """Doc 02 §7: Ranking weights are configurable and dynamically influence candidate scores."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    rec_a = KnowledgeRecord(
        knowledge_id="rec_general",
        owner_scope="user",
        owner_user_id="user_ranker",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="General Value",
        lead_type_tags=["investor"],
        topic_tags=["marketing"],
        text="Digital targeted marketing campaigns for properties.",
        verification_status="user_supplied",
    )
    rec_b = KnowledgeRecord(
        knowledge_id="rec_approved",
        owner_scope="user",
        owner_user_id="user_ranker",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="Approved Value",
        lead_type_tags=[],
        topic_tags=["marketing"],
        text="Digital targeted marketing campaigns for properties.",
        verification_status="company_approved",
    )
    store.add_record(rec_a)
    store.add_record(rec_b)

    req = RetrievalRequest(
        call_id="call_rank",
        state_version=1,
        user_id="user_ranker",
        lead_type="investor",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="marketing campaigns",
            topic_tags=["marketing"],
        ),
    )

    # Config 1: Heavily weight lead_type_boost (0.35) over verification (0.05)
    config_lead = RetrievalConfig(
        weights=RankingWeights(
            semantic_weight=0.40,
            lead_type_boost=0.35,
            verification_boost=0.05,
        )
    )
    res_lead = engine.retrieve_for_request(req, config=config_lead)
    assert len(res_lead.facts) == 2
    # rec_a matches 'investor' lead type so it gets the big lead_type_boost and comes first
    assert res_lead.facts[0].knowledge_id == "rec_general"

    # Config 2: Heavily weight verification_boost (0.40) over lead_type (0.02)
    config_verif = RetrievalConfig(
        weights=RankingWeights(
            semantic_weight=0.40,
            lead_type_boost=0.02,
            verification_boost=0.40,
        )
    )
    res_verif = engine.retrieve_for_request(req, config=config_verif)
    assert len(res_verif.facts) == 2
    # rec_b has company_approved status so it gets the big verification_boost and comes first
    assert res_verif.facts[0].knowledge_id == "rec_approved"


def test_recency_boost_only_applies_when_time_sensitive():
    """Doc 02 §7: Recency boost applies ONLY if time_sensitive = True."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    recent_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    rec = KnowledgeRecord(
        knowledge_id="rec_recent",
        owner_scope="user",
        owner_user_id="user_recency",
        knowledge_class=KnowledgeClass.MARKET_KNOWLEDGE.value,
        source_title="Recent Market Inventory",
        uploaded_at=recent_date,
        topic_tags=["inventory"],
        text="Active local inventory dropped 15% this quarter.",
        verification_status="company_approved",
    )
    store.add_record(rec)

    # Case A: time_sensitive = False -> No recency boost
    req_non_ts = RetrievalRequest(
        call_id="call_ts_false",
        state_version=1,
        user_id="user_recency",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="inventory changes",
            topic_tags=["inventory"],
            time_sensitive=False,
        ),
    )
    res_non_ts = engine.retrieve_for_request(req_non_ts)

    # Case B: time_sensitive = True -> Recency boost applied
    req_ts = RetrievalRequest(
        call_id="call_ts_true",
        state_version=2,
        user_id="user_recency",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="inventory changes",
            topic_tags=["inventory"],
            time_sensitive=True,
        ),
    )
    res_ts = engine.retrieve_for_request(req_ts)

    assert len(res_non_ts.facts) == 1
    assert len(res_ts.facts) == 1
    # Score with time_sensitive=True must be strictly higher due to recency_boost
    assert res_ts.facts[0].confidence > res_non_ts.facts[0].confidence


# ==============================================================================
# Phase 3: Relevance Floor & Marginal Threshold Tests (Doc 02 §7)
# ==============================================================================


def test_relevance_floor_drops_low_similarity_candidates_returns_none_found():
    """Doc 02 §7 & C4: Chunks below the relevance floor are dropped, returning none_found."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    rec = KnowledgeRecord(
        knowledge_id="rec_unrelated",
        owner_scope="user",
        owner_user_id="user_floor",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="Office Parking Policy",
        topic_tags=["parking", "office"],
        text="Employee visitor parking is reserved on level two.",
        verification_status="company_approved",
    )
    store.add_record(rec)

    req = RetrievalRequest(
        call_id="call_floor",
        state_version=1,
        user_id="user_floor",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="Average list to sale price ratio",
            topic_tags=["pricing"],
        ),
    )
    # Relevance floor set to 0.40; parking chunk cannot clear it
    res = engine.retrieve_for_request(req, relevance_floor=0.40)
    assert res.status == "none_found"
    assert len(res.facts) == 0
    assert res.fallback_hint == "ask_clarifying_question"


def test_marginal_threshold_triggers_low_confidence_status():
    """Doc 02 §7: Best candidate clearing floor marginally returns status=low_confidence."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    rec = KnowledgeRecord(
        knowledge_id="rec_weak",
        owner_scope="user",
        owner_user_id="user_marginal",
        knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
        source_title="General Timeline",
        topic_tags=["timeline"],
        text="A home sale timeline typically involves multiple stages.",
        verification_status="user_supplied",
    )
    store.add_record(rec)

    req = RetrievalRequest(
        call_id="call_marginal",
        state_version=1,
        user_id="user_marginal",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="detailed closing escrow timeline breakdown",
            topic_tags=["timeline"],
        ),
    )
    # Floor = 0.20 (rec clears it), Marginal = 0.85 (rec does not clear it)
    config = RetrievalConfig(
        relevance_floor=0.20,
        marginal_threshold=0.85,
    )
    res = engine.retrieve_for_request(req, config=config)
    assert res.status == "low_confidence"
    assert len(res.facts) == 1
    assert res.fallback_hint == "use_generic_truthful"


def test_facts_capped_to_max_facts():
    """Doc 02 §7: Returned facts are capped (1-3 facts) to avoid prompt dumping."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    for i in range(5):
        store.add_record(
            KnowledgeRecord(
                knowledge_id=f"rec_cap_{i}",
                owner_scope="user",
                owner_user_id="user_cap",
                knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
                source_title=f"Value Prop {i}",
                topic_tags=["marketing"],
                text=f"Marketing proof point number {i} with high resolution virtual tours.",
                verification_status="company_approved",
            )
        )

    req = RetrievalRequest(
        call_id="call_cap",
        state_version=1,
        user_id="user_cap",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="virtual tours marketing proof",
            topic_tags=["marketing"],
        ),
    )
    config = RetrievalConfig(max_facts=2)
    res = engine.retrieve_for_request(req, config=config)
    assert len(res.facts) == 2


# ==============================================================================
# Phase 3: Conflict Resolution Table Tests (Doc 02 §8)
# ==============================================================================


def test_conflict_table_unverified_performance_claim_withheld():
    """Doc 02 §8: Historical performance claim lacking company approval is withheld."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    rec = KnowledgeRecord(
        knowledge_id="rec_perf_unverified",
        owner_scope="user",
        owner_user_id="user_agent",
        knowledge_class=KnowledgeClass.PERFORMANCE_PAST_SALES.value,
        source_title="Self-Reported Sales",
        topic_tags=["sales_volume"],
        text="I closed 85 homes last year with $40M in transaction volume.",
        verification_status="user_supplied",  # Not company_approved or system_validated!
    )
    store.add_record(rec)

    req = RetrievalRequest(
        call_id="call_perf",
        state_version=1,
        user_id="user_agent",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="sales volume transaction proof",
            topic_tags=["sales_volume"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert len(res.facts) == 0
    assert len(res.withheld) == 1
    assert res.withheld[0].knowledge_id == "rec_perf_unverified"
    assert res.withheld[0].reason == "unverified"
    assert res.status == "none_found"


def test_conflict_table_newer_version_supersedes_older_version():
    """Doc 02 §8: Newest effective approved source version wins; older version is withheld as stale."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    v1 = KnowledgeRecord(
        knowledge_id="rec_fee_v1",
        owner_scope="user",
        owner_user_id="user_fee",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Listing Fee Schedule",
        source_version="1.0",
        topic_tags=["commission", "fee"],
        text="Standard listing fee is 6% commission.",
        verification_status="company_approved",
    )
    v2 = KnowledgeRecord(
        knowledge_id="rec_fee_v2",
        owner_scope="user",
        owner_user_id="user_fee",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Listing Fee Schedule",
        source_version="2.0",
        topic_tags=["commission", "fee"],
        text="Standard listing fee is 5% commission.",
        verification_status="company_approved",
    )
    store.add_record(v1)
    store.add_record(v2)

    req = RetrievalRequest(
        call_id="call_version",
        state_version=1,
        user_id="user_fee",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="Standard listing fee commission schedule",
            topic_tags=["commission", "fee"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert len(res.facts) == 1
    assert res.facts[0].knowledge_id == "rec_fee_v2"
    assert "5% commission" in res.facts[0].text
    assert len(res.withheld) == 1
    assert res.withheld[0].knowledge_id == "rec_fee_v1"
    assert res.withheld[0].reason == "stale"


def test_conflict_table_contradictory_facts_authority_win():
    """Doc 02 §8: When two documents disagree, company_approved authority (3) overrides user_supplied (1)."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    rep_claim = KnowledgeRecord(
        knowledge_id="rec_rep_claim",
        owner_scope="user",
        owner_user_id="user_auth",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="Rep Personal Notes",
        topic_tags=["cancellation"],
        text="Clients may cancel listing contract with free cancellation anytime.",
        verification_status="user_supplied",
    )
    company_policy = KnowledgeRecord(
        knowledge_id="rec_co_policy",
        owner_scope="user",
        owner_user_id="user_auth",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="Official Company Policy",
        topic_tags=["cancellation"],
        text="Early termination incurs an administrative fee required for expenses.",
        verification_status="company_approved",
    )
    store.add_record(rep_claim)
    store.add_record(company_policy)

    req = RetrievalRequest(
        call_id="call_auth",
        state_version=1,
        user_id="user_auth",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.POLICY_WORDING,
            need_statement="listing cancellation policy",
            topic_tags=["cancellation"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert len(res.facts) == 1
    assert res.facts[0].knowledge_id == "rec_co_policy"
    assert len(res.withheld) == 1
    assert res.withheld[0].knowledge_id == "rec_rep_claim"
    assert res.withheld[0].reason == "disputed"


def test_conflict_table_contradictory_facts_tie_withholds_both_do_not_assert():
    """Doc 02 §8: Contradictory documents with equal authority are withheld (do not assert disputed fact)."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    doc_a = KnowledgeRecord(
        knowledge_id="rec_tie_a",
        owner_scope="user",
        owner_user_id="user_tie",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Brokerage Fee Sheet A",
        source_version="1.0",
        topic_tags=["fee"],
        text="Listing commission is 5% total.",
        verification_status="company_approved",
    )
    doc_b = KnowledgeRecord(
        knowledge_id="rec_tie_b",
        owner_scope="user",
        owner_user_id="user_tie",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Brokerage Fee Sheet B",
        source_version="1.0",
        topic_tags=["fee"],
        text="Listing commission is 6% total.",
        verification_status="company_approved",
    )
    store.add_record(doc_a)
    store.add_record(doc_b)

    req = RetrievalRequest(
        call_id="call_tie",
        state_version=1,
        user_id="user_tie",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="listing commission fee rate",
            topic_tags=["fee"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "conflict_unresolved"
    assert len(res.facts) == 0
    assert len(res.withheld) == 2
    assert all(w.reason == "disputed" for w in res.withheld)
    assert res.fallback_hint == "use_generic_truthful"


# ==============================================================================
# Phase 3 & 4: Hallucination Guardrails & LLM Integration (Doc 02 §9 & §10)
# ==============================================================================


def test_validate_prompt_hallucinations_detects_ungrounded_claims():
    """Doc 02 §9: Post-generation check detects ungrounded percentages, volumes, and guarantees."""
    facts = [
        RetrievedFact(
            knowledge_id="k_valid",
            text="We provide HDR professional photography on 100% of our listings.",
            knowledge_class="value_proposition",
            verification_status="company_approved",
            source_title="Value Sheet",
            confidence=0.85,
        )
    ]

    # Test 1: Grounded prompt passes
    valid_prompt = "We provide HDR professional photography on 100% of our listings."
    is_valid, violations = validate_prompt_hallucinations(valid_prompt, facts)
    assert is_valid
    assert len(violations) == 0

    # Test 2: Ungrounded percentage detected
    fake_pct_prompt = "We deliver a 98% satisfaction rate for every homeowner."
    is_valid, violations = validate_prompt_hallucinations(fake_pct_prompt, facts)
    assert not is_valid
    assert any("98%" in v for v in violations)

    # Test 3: Ungrounded volume metric detected
    fake_vol_prompt = "Our team closed 50 homes in your specific neighborhood."
    is_valid, violations = validate_prompt_hallucinations(fake_vol_prompt, facts)
    assert not is_valid
    assert any("50 homes" in v for v in violations)

    # Test 4: Ungrounded guarantee detected
    fake_guar_prompt = "We guarantee top dollar with zero risk."
    is_valid, violations = validate_prompt_hallucinations(fake_guar_prompt, facts)
    assert not is_valid
    assert any("guarantee" in v.lower() for v in violations)


def test_llm_gateway_post_generation_guardrail_replaces_hallucinated_text():
    """Doc 02 §9: When generation produces ungrounded numbers, gateway replaces it with truthful fallback."""
    gateway = LLMResponseGateway(
        generator_func=lambda ctx, max_w: "We closed 120 homes in this zip code and guarantee a 10% premium!"
    )
    decision = StrategicDecision(
        call_id="call_guard",
        source_state_version=5,
        strategic_objective="Differentiate track record",
        primary_action=StrategicAction.DIFFERENTIATE,
        max_prompt_words=20,
    )
    snapshot = ConversationStateSnapshot(call_sid="call_guard", state_version=5)
    # Empty facts
    facts = []

    prompt = gateway.generate_prompt(
        decision=decision,
        snapshot=snapshot,
        facts=facts,
    )
    # The hallucinated claim must NOT be present in prompt.text
    assert "120 homes" not in prompt.text
    assert "10% premium" not in prompt.text
    assert "guarantee" not in prompt.text.lower()
    # It must have fallen back to a clean, prospect-focused response
    assert prompt.status == "generated"
    assert len(prompt.text) > 0


def test_phase4_prompt_assembly_priority_hierarchy_and_structured_blocks():
    """Doc 01 §10 & Doc 02 §10: 6-block context assembly verifies strict priority ordering."""
    gateway = LLMResponseGateway()
    decision = StrategicDecision(
        call_id="call_p4",
        source_state_version=10,
        strategic_objective="Educate on commission net sheet",
        primary_action=StrategicAction.EDUCATE,
        reason_codes=["STAGE_DISCOVERY"],
        do_not_do=["misquote_rates", "pressure_close"],
        max_prompt_words=22,
    )
    snapshot = ConversationStateSnapshot(
        call_sid="call_p4",
        state_version=10,
        conversation_stage=ConversationStage.DISCOVERY,
    )
    facts = [
        RetrievedFact(
            knowledge_id="know_p4",
            text="Net sheet calculation accounts for title fees and local transfer taxes.",
            knowledge_class="pricing_packaging",
            verification_status="company_approved",
            source_title="Closing Guide",
            confidence=0.88,
        )
    ]

    context = gateway.assemble_context(
        decision=decision,
        snapshot=snapshot,
        facts=facts,
    )

    # 1. Verify all 6 structured context blocks exist
    assert "### BLOCK 1: STRATEGIC DECISION CONTRACT" in context
    assert "### BLOCK 2: CURRENT CONVERSATION TRUTH" in context
    assert "### BLOCK 3: VERIFIED KNOWLEDGE" in context
    assert "### BLOCK 4: PLAYBOOK CONSTRAINTS" in context
    assert "### BLOCK 5: CALIBRATION DELIVERY CONSTRAINTS" in context
    assert "### BLOCK 6: MANDATORY GENERATION DIRECTIVES" in context

    # 2. Verify Doc 01 §10 Priority Hierarchy is explicitly documented in Block 6
    assert "Priority 1: Safety/compliance" in context
    assert "Priority 2: Core strategic intent" in context
    assert "Priority 3: Live conversation truth" in context
    assert "Priority 4: Playbook sales methodology" in context
    assert "Priority 5: Relevant retrieved facts" in context
    assert "Priority 6: Calibration delivery profile" in context
    assert "Priority 7: Stylistic polish" in context

    # 3. Verify no-invention rule
    assert "Do NOT invent unverified facts, credentials, statistics, volume, or numbers" in context
    assert "Use only verified facts from Block 3" in context
