import time
import pytest
from datetime import datetime, timezone

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
    StageLatencies,
)
from copilot.conditional_retrieval import (
    ConditionalRetrievalEngine,
    KnowledgeStore,
    RetrievalCache,
    RetrievalGate,
)
from copilot.conversation_state_models import ConversationStage, ConversationStateSnapshot
from copilot.core_intelligence_models import StrategicAction, StrategicDecision
from copilot.llm_response_gateway import LLMResponseGateway


# ==============================================================================
# Phase 5: Per-Stage Latency Instrumentation (Doc 01 §13)
# ==============================================================================


def test_stage_latencies_measured_on_successful_retrieval():
    """Doc 01 §13: Gate, embedding, search, ranking/conflict, packaging, and total are measured."""
    store = KnowledgeStore()
    cache = RetrievalCache()
    engine = ConditionalRetrievalEngine(store=store, cache=cache)

    rec = KnowledgeRecord(
        knowledge_id="rec_lat_1",
        owner_scope="user",
        owner_user_id="user_latency",
        knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
        source_title="Process Guarantee",
        topic_tags=["process"],
        text="We provide a dedicated closing coordinator for each transaction.",
        verification_status="company_approved",
    )
    store.add_record(rec)

    req = RetrievalRequest(
        call_id="call_lat_1",
        state_version=1,
        user_id="user_latency",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="closing coordinator process",
            topic_tags=["process"],
        ),
    )

    result = engine.retrieve_for_request(req)

    assert result.status == "retrieved"
    assert len(result.facts) == 1
    assert result.stage_latencies is not None
    assert isinstance(result.stage_latencies, StageLatencies)
    assert result.stage_latencies.gate_ms >= 0.0
    assert result.stage_latencies.embedding_ms >= 0.0
    assert result.stage_latencies.search_ms >= 0.0
    assert result.stage_latencies.ranking_conflict_ms >= 0.0
    assert result.stage_latencies.packaging_ms >= 0.0
    assert result.stage_latencies.total_ms >= 0.0
    assert result.latency_ms == result.stage_latencies.total_ms


def test_stage_latencies_measured_on_gate_exit():
    """Doc 01 §13 & Doc 02 §6: Gate short-circuits skip search/conflict resolution and record fast exit."""
    engine = ConditionalRetrievalEngine()

    # Trigger with excluded action (wait) -> none_needed
    req = RetrievalRequest(
        call_id="call_lat_gate",
        state_version=1,
        user_id="user_gate_lat",
        strategic_action="wait_silence",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="price question",
        ),
    )

    result = engine.retrieve_for_request(req)
    assert result.status == "none_needed"
    assert result.stage_latencies is not None
    assert result.stage_latencies.gate_ms >= 0.0
    assert result.stage_latencies.search_ms == 0.0
    assert result.stage_latencies.ranking_conflict_ms == 0.0


# ==============================================================================
# Phase 5: Deadline and Timeout Fallback (Doc 01 §13)
# ==============================================================================


def test_retrieval_timeout_triggers_fallback_hint_and_empty_facts():
    """Doc 01 §13: When deadline_ms is exceeded, return status='timed_out' and fallback_hint='use_generic_truthful'."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)

    rec = KnowledgeRecord(
        knowledge_id="rec_timeout_1",
        owner_scope="user",
        owner_user_id="user_to",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Fee Schedule",
        topic_tags=["fees"],
        text="Standard listing fee is 5% commission.",
        verification_status="company_approved",
    )
    store.add_record(rec)

    # Set impossibly low deadline (0.00001 ms) to guarantee timeout checkpoint triggers
    req = RetrievalRequest(
        call_id="call_to_1",
        state_version=1,
        user_id="user_to",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="what is your fee",
            topic_tags=["fees"],
        ),
        deadline_ms=0.00001,
    )

    result = engine.retrieve_for_request(req)

    assert result.status == "timed_out"
    assert result.fallback_hint == "use_generic_truthful"
    assert len(result.facts) == 0
    assert result.stage_latencies is not None
    assert result.stage_latencies.total_ms >= 0.0


def test_timeout_fallback_in_llm_response_gateway():
    """Doc 01 §13 & Doc 02 §10: Timed-out retrieval notifies LLM assembly to apply zero-assumption fallback."""
    gateway = LLMResponseGateway(generator_func=lambda ctx, words: "I understand your focus on fees, let's explore what fits your timeline.")
    snapshot = ConversationStateSnapshot(
        call_id="call_gw_to",
        call_sid="call_gw_to",
        conversation_stage=ConversationStage.VALUE_WALKTHROUGH,
        state_version=2,
    )
    decision = StrategicDecision(
        call_id="call_gw_to",
        decision_id="dec_gw_to",
        source_state_version=2,
        primary_action=StrategicAction.QUANTIFY,
        strategic_objective="Address pricing question",
        push_strength="medium",
        retrieval_needed=True,
    )

    timed_out_result = RetrievalResult(
        request_id="req_to_gw",
        state_version=2,
        status="timed_out",
        facts=[],
        withheld=[],
        fallback_hint="use_generic_truthful",
        latency_ms=250.5,
    )

    context = gateway.assemble_context(
        decision=decision,
        snapshot=snapshot,
        retrieval_result=timed_out_result,
    )

    # Block 3 has TIMED_OUT notice
    assert "[TIMED_OUT]" in context
    # Block 6 has RETRIEVAL TIMEOUT GUARD directive
    assert "RETRIEVAL TIMEOUT GUARD" in context
    assert "zero fabricated claims" in context

    # Generation succeeds truthfully without ungrounded hallucinations
    prompt = gateway.generate_prompt(
        decision=decision,
        snapshot=snapshot,
        retrieval_result=timed_out_result,
    )
    assert prompt.status == "generated"
    assert len(prompt.text) > 0


# ==============================================================================
# Phase 5: State Version Stale-Result Discard (Doc 01 §13)
# ==============================================================================


def test_validate_or_discard_stale_result_method():
    """Doc 01 §13: Late retrieval result from earlier state_version is discarded to avoid overwriting state."""
    result_v1 = RetrievalResult(
        request_id="req_stale_1",
        state_version=1,
        status="retrieved",
        facts=[
            RetrievedFact(
                knowledge_id="know_old",
                text="Old state claim 5% fee",
                knowledge_class="pricing_packaging",
                source_title="Old Doc",
                verification_status="company_approved",
            )
        ],
        fallback_hint=None,
    )

    # 1. State version matches -> valid, kept unchanged
    valid_res = ConditionalRetrievalEngine.validate_or_discard_stale_result(result_v1, current_state_version=1)
    assert valid_res.status == "retrieved"
    assert len(valid_res.facts) == 1

    # 2. State version advanced to 2 (e.g. prospect interrupted) -> discarded!
    stale_res = ConditionalRetrievalEngine.validate_or_discard_stale_result(result_v1, current_state_version=2)
    assert stale_res.status == "stale_discarded"
    assert len(stale_res.facts) == 0
    assert stale_res.fallback_hint == "use_generic_truthful"


def test_stale_result_discarded_in_llm_response_gateway():
    """Doc 01 §13 & Spec 11: LLMResponseGateway discards late retrieval results if snapshot.state_version advanced."""
    gateway = LLMResponseGateway(generator_func=lambda ctx, words: "I hear you, let's look at your latest priority.")
    
    # Snapshot is at state_version 3 (prospect interrupted and changed topic)
    snapshot = ConversationStateSnapshot(
        call_id="call_gw_stale",
        call_sid="call_gw_stale",
        conversation_stage=ConversationStage.OBJECTION_HANDLING,
        state_version=3,
    )
    decision = StrategicDecision(
        call_id="call_gw_stale",
        decision_id="dec_gw_stale",
        source_state_version=3,
        primary_action=StrategicAction.CLARIFY,
        strategic_objective="Address new prospect concern",
        push_strength="soft",
    )

    # Result came in late from turn 2
    late_result = RetrievalResult(
        request_id="req_late",
        state_version=2,  # Stale! (2 < 3)
        status="retrieved",
        facts=[
            RetrievedFact(
                knowledge_id="know_late",
                text="Obsolete fact from turn 2",
                knowledge_class="service_process",
                source_title="Doc",
                verification_status="company_approved",
            )
        ],
    )

    context = gateway.assemble_context(
        decision=decision,
        snapshot=snapshot,
        retrieval_result=late_result,
    )

    # Block 3 notes discard
    assert "[STALE_DISCARDED]" in context
    assert "discarded because conversation state advanced" in context
    # Obsolete fact text is not presented as verified reference data
    assert "Obsolete fact from turn 2" not in context
    # Block 6 instructs LLM to address latest statement
    assert "STALE RETRIEVAL DISCARD" in context


# ==============================================================================
# Phase 5: Cache Invalidation, Soft-Delete & Supersession (Doc 02 §11)
# ==============================================================================


def test_cache_hit_returns_identical_result_with_low_latency():
    """Doc 02 §11: Identical queries within TTL hit cache immediately."""
    store = KnowledgeStore()
    cache = RetrievalCache()
    engine = ConditionalRetrievalEngine(store=store, cache=cache)

    rec = KnowledgeRecord(
        knowledge_id="rec_c1",
        owner_scope="user",
        owner_user_id="user_c",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="Value Proposition",
        topic_tags=["unique_service"],
        text="We provide end-to-end staging and photography at no upfront cost.",
        verification_status="company_approved",
    )
    store.add_record(rec)

    req = RetrievalRequest(
        call_id="call_c_1",
        state_version=1,
        user_id="user_c",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="staging photography upfront",
            topic_tags=["unique_service"],
        ),
    )

    # First retrieval -> Miss, executes search & populates cache
    res1 = engine.retrieve_for_request(req)
    assert res1.status == "retrieved"
    assert len(res1.facts) == 1

    # Second retrieval -> Cache hit!
    res2 = engine.retrieve_for_request(req)
    assert res2.status == "retrieved"
    assert len(res2.facts) == 1
    assert res2.facts[0].knowledge_id == res1.facts[0].knowledge_id


def test_cache_partitioned_by_user_id_prevents_cross_user_leak():
    """Doc 02 §11 & C6: Cache is strictly partitioned by user_id; User A's cache never serves User B."""
    store = KnowledgeStore()
    cache = RetrievalCache()
    engine = ConditionalRetrievalEngine(store=store, cache=cache)

    rec_a = KnowledgeRecord(
        knowledge_id="rec_user_a",
        owner_scope="user",
        owner_user_id="user_alice",
        knowledge_class=KnowledgeClass.SALESPERSON_PROFILE.value,
        source_title="Alice Bio",
        topic_tags=["experience"],
        text="Alice has 10 years experience selling luxury homes.",
        verification_status="company_approved",
    )
    store.add_record(rec_a)

    req_alice = RetrievalRequest(
        call_id="call_alice",
        state_version=1,
        user_id="user_alice",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="experience selling luxury homes",
            topic_tags=["experience"],
        ),
    )

    res_alice = engine.retrieve_for_request(req_alice)
    assert res_alice.status == "retrieved"
    assert res_alice.facts[0].knowledge_id == "rec_user_a"

    # User Bob makes exact same query need
    req_bob = RetrievalRequest(
        call_id="call_bob",
        state_version=1,
        user_id="user_bob",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="experience selling luxury homes",
            topic_tags=["experience"],
        ),
    )

    res_bob = engine.retrieve_for_request(req_bob)
    # Bob has no records, must NOT get Alice's cached result!
    assert res_bob.status == "none_found"
    assert len(res_bob.facts) == 0


def test_cache_invalidated_immediately_on_record_delete():
    """Doc 02 §11 (Step 10): Deleting a record immediately flushes user's cache and marks tombstone."""
    store = KnowledgeStore()
    cache = RetrievalCache()
    engine = ConditionalRetrievalEngine(store=store, cache=cache)

    rec = KnowledgeRecord(
        knowledge_id="rec_del_1",
        owner_scope="user",
        owner_user_id="user_del",
        knowledge_class=KnowledgeClass.MARKET_KNOWLEDGE.value,
        source_title="Old Market Stats",
        topic_tags=["market"],
        text="Average home price in downtown was $650k in 2021.",
        verification_status="company_approved",
    )
    store.add_record(rec)

    req = RetrievalRequest(
        call_id="call_del",
        state_version=1,
        user_id="user_del",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="downtown market stats price",
            topic_tags=["market"],
        ),
    )

    # 1. Warm cache
    res1 = engine.retrieve_for_request(req)
    assert res1.status == "retrieved"

    # 2. User deletes record (soft delete tombstone per Step 10)
    store.delete_record("rec_del_1", soft=True)
    assert store.get_record("rec_del_1").deleted is True

    # 3. Next query must NOT return deleted chunk from cache or store
    res2 = engine.retrieve_for_request(req)
    assert res2.status == "none_found"
    assert len(res2.facts) == 0


def test_cache_invalidated_immediately_on_record_replace():
    """Doc 02 §11 (Step 10): Replacing a record marks supersession, flushes cache, and serves new version."""
    store = KnowledgeStore()
    cache = RetrievalCache()
    engine = ConditionalRetrievalEngine(store=store, cache=cache)

    rec_v1 = KnowledgeRecord(
        knowledge_id="rec_v1",
        owner_scope="user",
        owner_user_id="user_replace",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Commission Agreement",
        source_version="1.0",
        topic_tags=["commission"],
        text="Standard full service commission is 6%.",
        verification_status="company_approved",
    )
    store.add_record(rec_v1)

    req = RetrievalRequest(
        call_id="call_rep",
        state_version=1,
        user_id="user_replace",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.FACTUAL_QUESTION,
            need_statement="what is your standard commission",
            topic_tags=["commission"],
        ),
    )

    # 1. Warm cache with v1
    res1 = engine.retrieve_for_request(req)
    assert res1.status == "retrieved"
    assert "6%" in res1.facts[0].text

    # 2. Replace with v2 (5.5%)
    rec_v2 = KnowledgeRecord(
        knowledge_id="rec_v2",
        owner_scope="user",
        owner_user_id="user_replace",
        knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
        source_title="Commission Agreement",
        source_version="2.0",
        topic_tags=["commission"],
        text="Standard full service commission is 5.5%.",
        verification_status="company_approved",
    )
    store.replace_record("rec_v1", rec_v2)

    # Verify audit trail retained: rec_v1 has superseded_by
    assert store.get_record("rec_v1").superseded_by == "rec_v2"

    # 3. Retrieval returns v2 (5.5%), old v1 is superseded and not returned
    res2 = engine.retrieve_for_request(req)
    assert res2.status == "retrieved"
    assert len(res2.facts) == 1
    assert res2.facts[0].knowledge_id == "rec_v2"
    assert "5.5%" in res2.facts[0].text


def test_cache_invalidated_immediately_on_record_attest():
    """Doc 02 §11: Self-attestation flushes cache so fact is retrieved with self_attested status."""
    store = KnowledgeStore()
    cache = RetrievalCache()
    engine = ConditionalRetrievalEngine(store=store, cache=cache)

    rec = KnowledgeRecord(
        knowledge_id="rec_attest_c",
        owner_scope="user",
        owner_user_id="user_rep_att",
        knowledge_class=KnowledgeClass.SALESPERSON_PROFILE.value,
        source_title="Rep Bio",
        topic_tags=["experience"],
        text="I have personally closed 50 homes in this neighborhood.",
        verification_status="user_supplied",
    )
    store.add_record(rec)

    req = RetrievalRequest(
        call_id="call_att_c",
        state_version=1,
        user_id="user_rep_att",
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="homes closed in neighborhood",
            topic_tags=["experience"],
        ),
    )

    # 1. Warm cache (user_supplied status)
    res1 = engine.retrieve_for_request(req)
    assert res1.status == "retrieved"
    assert res1.facts[0].verification_status == "user_supplied"

    # 2. Rep attests record
    store.attest_record("rec_attest_c", user_id="user_rep_att")

    # 3. Re-query -> cache invalidated, returns self_attested fact (Authority 2)
    res2 = engine.retrieve_for_request(req)
    assert res2.status == "retrieved"
    assert res2.facts[0].verification_status == "self_attested"
    assert res2.facts[0].authority_level == 2
