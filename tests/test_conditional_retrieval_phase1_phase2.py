import pytest
from datetime import datetime, timezone, timedelta

from copilot.conditional_retrieval_models import (
    KnowledgeClass,
    KnowledgeRecord,
    RetrievalNeed,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrigger,
    RetrievedFact,
    WithheldFact,
    GateDecisionLog,
)
from copilot.conditional_retrieval import (
    ConditionalRetrievalEngine,
    KnowledgeStore,
    QueryBuilder,
    RetrievalGate,
)


# ==============================================================================
# Phase 1: Contracts and Gate Tests (Doc 02 §6)
# ==============================================================================


def test_phase1_retrieval_gate_no_trigger_flagged():
    """Acceptance Test 8: Turn with no trigger -> status=none_needed, no retrieval executed."""
    gate = RetrievalGate()
    request = RetrievalRequest(
        call_id="call_test_1",
        state_version=1,
        user_id="user_alice",
        strategic_action="question",
        retrieval_need=None,
    )
    passed, status, fallback, reason = gate.evaluate(request)
    assert not passed
    assert status == "none_needed"
    assert fallback is None
    assert "NO_TRIGGER_FLAGGED" in reason
    assert len(gate.decision_logs) == 1
    assert gate.decision_logs[0].gate_passed is False


def test_phase1_retrieval_gate_excluded_strategic_actions():
    """Gate Rule: Strategic actions in {Wait, Hold, Acknowledge} bypass retrieval completely."""
    gate = RetrievalGate()
    for action in ["wait_silence", "wait", "hold", "acknowledge"]:
        request = RetrievalRequest(
            call_id="call_test_2",
            state_version=2,
            user_id="user_alice",
            strategic_action=action,
            retrieval_need=RetrievalNeed(
                trigger=RetrievalTrigger.FACTUAL_QUESTION,
                need_statement="Explain commission breakdown",
            ),
        )
        passed, status, fallback, reason = gate.evaluate(request)
        assert not passed
        assert status == "none_needed"
        assert fallback is None
        assert "ACTION_EXCLUDED_FROM_RETRIEVAL" in reason


def test_phase1_retrieval_gate_no_in_scope_knowledge():
    """Acceptance Test 1: User has no active knowledge -> none_found, fallback to generic truthful."""
    gate = RetrievalGate()
    empty_store = KnowledgeStore()
    request = RetrievalRequest(
        call_id="call_test_3",
        state_version=3,
        user_id="user_bob",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="Verify local sales track record",
        ),
    )
    passed, status, fallback, reason = gate.evaluate(request, knowledge_store=empty_store)
    assert not passed
    assert status == "none_found"
    assert fallback == "use_generic_truthful"
    assert "USER_HAS_ZERO_ACTIVE_IN_SCOPE_KNOWLEDGE" in reason


def test_phase1_retrieval_gate_playbook_evidence_no_proof():
    """Acceptance Test 10: Playbook requests evidence but no verified proof exists -> choose_alternate_move."""
    gate = RetrievalGate()
    empty_store = KnowledgeStore()
    request = RetrievalRequest(
        call_id="call_test_4",
        state_version=4,
        user_id="user_carol",
        strategic_action="social_proof",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.PLAYBOOK_EVIDENCE,
            need_statement="Show case study of neighbor selling at record price",
        ),
    )
    passed, status, fallback, reason = gate.evaluate(request, knowledge_store=empty_store)
    assert not passed
    assert status == "none_found"
    assert fallback == "choose_alternate_move"


def test_phase1_engine_stub_empty_pipeline():
    """Pipeline works end-to-end with no retrieval (returns empty facts without failing)."""
    engine = ConditionalRetrievalEngine()
    request = RetrievalRequest(
        call_id="call_test_stub",
        state_version=5,
        user_id="user_david",
        strategic_action="acknowledge",
        retrieval_need=None,
    )
    result = engine.retrieve_for_request(request)
    assert isinstance(result, RetrievalResult)
    assert result.status == "none_needed"
    assert len(result.facts) == 0
    assert result.latency_ms >= 0.0


# ==============================================================================
# Phase 2: Safe Retrieval & Pre-Filter Tests (Doc 02 §5, §7, Constraint C6)
# ==============================================================================


def test_phase2_team_scope_rejected_on_write():
    """Acceptance Test 14 & Constraint C6: Writing team/brokerage/company scope is rejected."""
    for forbidden_scope in ["team", "brokerage", "company"]:
        with pytest.raises(ValueError, match="Team, brokerage, and company scopes are not enabled"):
            KnowledgeRecord(
                owner_scope=forbidden_scope,  # type: ignore
                owner_user_id="user_alice",
                knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
                source_title="Brokerage Handbook",
                text="Standard 6% commission rate.",
            )


def test_phase2_cross_user_isolation_zero_leakage():
    """Acceptance Test 6: User A's knowledge is NEVER retrieved for User B."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    # User A uploads a verified track record
    record_user_a = KnowledgeRecord(
        knowledge_id="know_alice_1",
        owner_scope="user",
        owner_user_id="user_alice",
        knowledge_class=KnowledgeClass.PERFORMANCE_PAST_SALES.value,
        source_title="Alice's Verified Sales 2025",
        topic_tags=["north_hills", "past_sales", "volume"],
        text="Alice sold 42 homes in North Hills averaging 11 days on market.",
        verification_status="company_approved",
    )
    store.add_record(record_user_a)

    # User B queries the exact same topic and neighborhood
    request_user_b = RetrievalRequest(
        call_id="call_user_b",
        state_version=1,
        user_id="user_bob",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="North Hills past sales volume",
            topic_tags=["north_hills", "past_sales"],
        ),
    )

    result_b = engine.retrieve_for_request(request_user_b)
    # Zero retrieval for User B!
    assert result_b.status == "none_found"
    assert len(result_b.facts) == 0

    # User A queries their own topic -> Retrieved!
    request_user_a = RetrievalRequest(
        call_id="call_user_a",
        state_version=1,
        user_id="user_alice",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="North Hills past sales volume",
            topic_tags=["north_hills", "past_sales"],
        ),
    )
    result_a = engine.retrieve_for_request(request_user_a)
    assert result_a.status == "retrieved"
    assert len(result_a.facts) == 1
    assert "Alice sold 42 homes" in result_a.facts[0].text


def test_phase2_deleted_tombstone_pre_filter():
    """Acceptance Test 5: Soft-deleted chunks stop being retrieved immediately."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    record = KnowledgeRecord(
        knowledge_id="know_old_policy",
        owner_scope="user",
        owner_user_id="user_dan",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="2024 Fee Policy",
        topic_tags=["fees", "commission"],
        text="Discount fee of 3.5% applies to all listings.",
    )
    store.add_record(record)

    # Mark deleted / tombstone
    store.delete_record("know_old_policy", soft=True)

    request = RetrievalRequest(
        call_id="call_dan_1",
        state_version=1,
        user_id="user_dan",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="Discount commission fee policy",
            topic_tags=["fees"],
        ),
    )
    result = engine.retrieve_for_request(request)
    assert result.status == "none_found"
    assert len(result.facts) == 0


def test_phase2_supersession_pre_filter():
    """Superseded knowledge is excluded from search results."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    old_rec = KnowledgeRecord(
        knowledge_id="know_v1",
        owner_scope="user",
        owner_user_id="user_eve",
        knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
        source_title="Inspection Process v1",
        source_version="1.0",
        topic_tags=["inspection"],
        text="Old process: 14-day turnaround.",
    )
    new_rec = KnowledgeRecord(
        knowledge_id="know_v2",
        owner_scope="user",
        owner_user_id="user_eve",
        knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
        source_title="Inspection Process v2",
        source_version="2.0",
        topic_tags=["inspection"],
        text="New approved process: 5-day expedited turnaround.",
        verification_status="company_approved",
    )
    store.add_record(old_rec)
    store.add_record(new_rec)
    store.supersede_record(old_knowledge_id="know_v1", new_knowledge_id="know_v2")

    request = RetrievalRequest(
        call_id="call_eve_1",
        state_version=1,
        user_id="user_eve",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="Inspection turnaround timeframe",
            topic_tags=["inspection"],
        ),
    )
    result = engine.retrieve_for_request(request)
    assert result.status == "retrieved"
    assert len(result.facts) == 1
    assert result.facts[0].knowledge_id == "know_v2"
    assert "5-day expedited" in result.facts[0].text


def test_phase2_validity_window_pre_filter():
    """Chunks outside their effective validity window are excluded."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    now = datetime.now(timezone.utc)
    expired_rec = KnowledgeRecord(
        knowledge_id="know_expired",
        owner_scope="user",
        owner_user_id="user_frank",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Spring 2024 Special",
        effective_from=(now - timedelta(days=60)).isoformat(),
        effective_to=(now - timedelta(days=1)).isoformat(),  # Expired yesterday
        topic_tags=["promotion", "pricing"],
        text="Spring promotion: Free professional staging.",
    )
    store.add_record(expired_rec)

    request = RetrievalRequest(
        call_id="call_frank_1",
        state_version=1,
        user_id="user_frank",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="Free professional staging promotion",
            topic_tags=["promotion"],
        ),
    )
    result = engine.retrieve_for_request(request)
    assert result.status == "none_found"


def test_phase2_sensitivity_pre_filter():
    """Restricted chunks are strictly excluded; confidential requires explicit allow."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    restricted_rec = KnowledgeRecord(
        knowledge_id="know_restricted",
        owner_scope="user",
        owner_user_id="user_grace",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="Internal Legal Settlement",
        sensitivity="restricted",
        topic_tags=["dispute", "legal"],
        text="Confidential settlement term notes.",
    )
    store.add_record(restricted_rec)

    request = RetrievalRequest(
        call_id="call_grace_1",
        state_version=1,
        user_id="user_grace",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.POLICY_WORDING,
            need_statement="Legal dispute policy",
            topic_tags=["legal"],
        ),
    )
    result = engine.retrieve_for_request(request)
    assert result.status == "none_found"


def test_phase2_lead_type_pre_filter():
    """Chunks with lead_type_tags are filtered out when lead_type mismatches."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    investor_chunk = KnowledgeRecord(
        knowledge_id="know_investor",
        owner_scope="user",
        owner_user_id="user_heidi",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="Investor Cap Rates",
        lead_type_tags=["investor", "commercial"],
        topic_tags=["cap_rate", "roi"],
        text="Multi-family portfolio yields 8.4% cap rate.",
    )
    store.add_record(investor_chunk)

    # Request is for a residential homeowner lead
    request_residential = RetrievalRequest(
        call_id="call_heidi_1",
        state_version=1,
        user_id="user_heidi",
        lead_type="residential_homeowner",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="Average portfolio yield cap rate",
            topic_tags=["roi"],
        ),
    )
    result = engine.retrieve_for_request(request_residential)
    # Excluded because lead_type doesn't match
    assert result.status == "none_found"

    # Request with matching lead_type
    request_investor = RetrievalRequest(
        call_id="call_heidi_2",
        state_version=2,
        user_id="user_heidi",
        lead_type="investor",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="Average portfolio yield cap rate",
            topic_tags=["roi"],
        ),
    )
    result_inv = engine.retrieve_for_request(request_investor)
    assert result_inv.status == "retrieved"
    assert len(result_inv.facts) == 1
    assert "8.4% cap rate" in result_inv.facts[0].text


def test_phase2_query_builder_structure():
    """Query builder constructs trimmed query and maps triggers to knowledge classes without dumping transcript."""
    request = RetrievalRequest(
        call_id="call_qb_1",
        state_version=10,
        user_id="user_ivan",
        strategic_action="social_proof",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="Do you have experience selling luxury homes in Oakwood?",
            topic_tags=["luxury_homes", "oakwood"],
        ),
        conversation_evidence="Prospect asked: How many luxury homes did you sell here last year? I don't know if you're the right fit.",
    )
    query, target_classes = QueryBuilder.build_query(request)
    assert "Do you have experience selling luxury homes in Oakwood?" in query
    assert "luxury homes" in query
    assert "oakwood" in query
    assert "Prospect asked" in query
    # Candidate classes derived from credibility_objection
    assert KnowledgeClass.SALESPERSON_PROFILE.value in target_classes
    assert KnowledgeClass.PERFORMANCE_PAST_SALES.value in target_classes


# ==============================================================================
# Critical Fix & Edge-Case Verification Tests
# ==============================================================================


def test_untagged_chunk_passes_lead_type_filter():
    """Requirement: An untagged chunk (lead_type_tags=[]) must ALWAYS pass, regardless of lead_type."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    untagged_chunk = KnowledgeRecord(
        knowledge_id="know_untagged_1",
        owner_scope="user",
        owner_user_id="user_john",
        knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
        source_title="General Agency Guarantee",
        lead_type_tags=[],  # Untagged chunk
        topic_tags=["guarantee", "cancel_anytime"],
        text="Clients can cancel their listing agreement at any time with 24 hours notice.",
        verification_status="company_approved",
    )
    store.add_record(untagged_chunk)

    # Test with residential_buyer
    req_buyer = RetrievalRequest(
        call_id="call_john_buyer",
        state_version=1,
        user_id="user_john",
        lead_type="residential_buyer",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.POLICY_WORDING,
            need_statement="Cancel agreement policy guarantee",
            topic_tags=["guarantee"],
        ),
    )
    res_buyer = engine.retrieve_for_request(req_buyer)
    assert res_buyer.status == "retrieved"
    assert len(res_buyer.facts) == 1
    assert "cancel their listing agreement" in res_buyer.facts[0].text

    # Test with commercial_investor
    req_investor = RetrievalRequest(
        call_id="call_john_investor",
        state_version=2,
        user_id="user_john",
        lead_type="commercial_investor",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.POLICY_WORDING,
            need_statement="Cancel agreement policy guarantee",
            topic_tags=["guarantee"],
        ),
    )
    res_investor = engine.retrieve_for_request(req_investor)
    assert res_investor.status == "retrieved"
    assert len(res_investor.facts) == 1


def test_confidential_chunk_excluded_without_authorization_and_included_with_it():
    """Requirement: Confidential chunks are excluded by default, but included when explicitly authorized."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    confidential_chunk = KnowledgeRecord(
        knowledge_id="know_conf_1",
        owner_scope="user",
        owner_user_id="user_kate",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="VIP Negotiated Tier Matrix",
        sensitivity="confidential",
        topic_tags=["vip_rate", "commission"],
        text="Executive VIP tier qualifies for 3.0% full-service rate.",
        verification_status="company_approved",
    )
    store.add_record(confidential_chunk)

    req = RetrievalRequest(
        call_id="call_kate_vip",
        state_version=1,
        user_id="user_kate",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="Executive VIP commission rate tier",
            topic_tags=["vip_rate"],
        ),
    )

    # 1. Without authorization -> excluded (none_found)
    res_unauthorized = engine.retrieve_for_request(req, allow_confidential=False)
    assert res_unauthorized.status == "none_found"
    assert len(res_unauthorized.facts) == 0

    # 2. With authorization -> included (retrieved)
    res_authorized = engine.retrieve_for_request(req, allow_confidential=True)
    assert res_authorized.status == "retrieved"
    assert len(res_authorized.facts) == 1
    assert "3.0% full-service" in res_authorized.facts[0].text


def test_null_validity_window_fields():
    """Requirement: effective_from=None and effective_to=None are null-safe and always considered valid."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    null_window_chunk = KnowledgeRecord(
        knowledge_id="know_evergreen_1",
        owner_scope="user",
        owner_user_id="user_leo",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="Evergreen Value Proposition",
        effective_from=None,  # Null validity window
        effective_to=None,    # Null validity window
        topic_tags=["evergreen", "value"],
        text="We provide HDR professional photography on 100% of our listings.",
        verification_status="company_approved",
    )
    store.add_record(null_window_chunk)

    req = RetrievalRequest(
        call_id="call_leo_1",
        state_version=1,
        user_id="user_leo",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="HDR professional photography standard",
            topic_tags=["value"],
        ),
    )
    result = engine.retrieve_for_request(req)
    assert result.status == "retrieved"
    assert len(result.facts) == 1
    assert "HDR professional photography" in result.facts[0].text


def test_legacy_retrieve_for_decision_hits_gate_and_ownership_filter():
    """Requirement: retrieve_for_decision(decision) wraps the Gate and ownership pre-filters.
    1. Bypassed action (WAIT_SILENCE) cannot trigger retrieval, even if retrieval_needed=True.
    2. User B cannot retrieve User A's chunk via legacy path (cross-user isolation).
    """
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    # User A has verified knowledge in store
    store.add_record(
        KnowledgeRecord(
            knowledge_id="know_priv_1",
            owner_scope="user",
            owner_user_id="user_alice",
            knowledge_class=KnowledgeClass.PERFORMANCE_PAST_SALES.value,
            source_title="Alice's Verified Comps",
            topic_tags=["market_comps"],
            text="Alice's average days on market is 12 days.",
            verification_status="company_approved",
        )
    )

    from copilot.core_intelligence_models import StrategicAction, StrategicDecision, RequiredFactScope

    # Test 1: Even if retrieval_needed=True, Gate blocks excluded actions (WAIT_SILENCE)
    dec_wait = StrategicDecision(
        call_id="call_gate_block",
        source_state_version=5,
        strategic_objective="Tactical pause for prospect to vent",
        primary_action=StrategicAction.WAIT_SILENCE,
        retrieval_needed=True,  # Malicious/buggy caller set this to True
        required_facts=[RequiredFactScope(topic="market_comps")],
    )
    report_wait = engine.retrieve_for_decision(dec_wait, user_id="user_alice")
    assert not report_wait.retrieval_needed
    assert len(report_wait.facts) == 0
    assert len(report_wait.queries_executed) == 0
    assert report_wait.status == "none_needed"

    # Test 2: User B calls retrieve_for_decision for market_comps -> Zero exposure of User A's data!
    dec_user_b = StrategicDecision(
        call_id="call_user_b",
        source_state_version=6,
        strategic_objective="Quantify market comps",
        primary_action=StrategicAction.QUANTIFY,
        retrieval_needed=True,
        required_facts=[RequiredFactScope(topic="market_comps")],
    )
    report_b = engine.retrieve_for_decision(dec_user_b, user_id="user_bob")
    # Fails closed, returns 0 chunks for user_bob
    assert len(report_b.facts) == 0


def test_social_proof_no_verified_proof_returns_choose_alternate_move():
    """Requirement: When the Core wants Social Proof and nothing verified exists,
    Gate returns fallback_hint = 'choose_alternate_move' (Doc 02 §9).
    """
    gate = RetrievalGate()
    empty_store = KnowledgeStore()

    # Case A: action is 'social_proof'
    req_sp = RetrievalRequest(
        call_id="call_sp_empty",
        state_version=7,
        user_id="user_novice",
        strategic_action="social_proof",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="Share case study of 500k sale nearby",
        ),
    )
    passed, status, fallback, reason = gate.evaluate(req_sp, knowledge_store=empty_store)
    assert not passed
    assert status == "none_found"
    assert fallback == "choose_alternate_move"

    # Case B: trigger is 'factual_question'
    req_fq = RetrievalRequest(
        call_id="call_fq_empty",
        state_version=8,
        user_id="user_novice",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="What is your average list-to-sale price ratio?",
        ),
    )
    passed, status, fallback, reason = gate.evaluate(req_fq, knowledge_store=empty_store)
    assert not passed
    assert status == "none_found"
    assert fallback == "ask_clarifying_question"


def test_none_found_handled_cleanly_by_llm_assembly_for_every_status():
    """Requirement: LLM Response Gateway handles every status value cleanly (none_found,
    none_needed, low_confidence, conflict_unresolved, timed_out, error) without fabricating.
    """
    from copilot.llm_response_gateway import LLMResponseGateway
    from copilot.conversation_state_models import ConversationStateSnapshot
    from copilot.core_intelligence_models import StrategicAction, StrategicDecision

    gateway = LLMResponseGateway()
    decision = StrategicDecision(
        call_id="call_gateway_test",
        source_state_version=14,
        strategic_objective="Quantify average commission savings",
        primary_action=StrategicAction.QUANTIFY,
        retrieval_needed=True,
    )
    snapshot = ConversationStateSnapshot(call_sid="call_gateway_test", state_version=14)

    statuses = [
        "none_found",
        "none_needed",
        "low_confidence",
        "conflict_unresolved",
        "timed_out",
        "error",
    ]

    for stat in statuses:
        # Create an empty RetrievalResult for each status
        ret_result = RetrievalResult(
            request_id=f"req_{stat}",
            state_version=14,
            status=stat,
            facts=[],
            withheld=[],
            fallback_hint="use_generic_truthful",
        )

        # Context assembly must succeed and activate the Fact Unavailability Guard
        context = gateway.assemble_context(
            decision=decision,
            snapshot=snapshot,
            facts=ret_result.facts,
        )
        assert "FACT UNAVAILABILITY GUARD (CRITICAL)" in context
        assert "Do NOT fabricate, estimate, or invent" in context
        assert "No verified external facts available" in context

        # Prompt generation must produce a truthful response without crashing
        prompt = gateway.generate_prompt(
            decision=decision,
            snapshot=snapshot,
            facts=ret_result.facts,
        )
        assert prompt.text != ""
        assert prompt.status == "generated"

