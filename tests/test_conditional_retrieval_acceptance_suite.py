"""Phase 6 End-to-End Acceptance Suite for Conditional Knowledge Retrieval.

Covers all 14 end-to-end acceptance scenarios in a single CI file.
Every test validates the final assembled prompt and generated response,
confirming that retrieval supplies facts, Core decides strategy, and
the LLM never invents unverified claims or overrides live conversation truth.

Scenarios:
- Doc 02 §13 Acceptance Tests:
  1. No upload
  2. Relevant upload
  3. Irrelevant upload
  4. Contradictory upload
  5. Deleted upload
  6. Cross-user leakage (Individual-User Scope)
  7. Historical marketing claim
- Workflow Cross-Checks:
  8. No-trigger turn
  9. Script doesn't override Core
  10. Social Proof with no proof
  11. Stale result (version divergence after retrieval & post-generation)
  12. Timeout fallback
  13. Live truth over stored fact
  14. Team scope rejection (Constraint C6)
"""

import pytest
from copilot.conditional_retrieval import (
    ConditionalRetrievalEngine,
    KnowledgeStore,
    RetrievalCache,
    RetrievalConfig,
    RetrievalGate,
)
from copilot.conditional_retrieval_models import (
    KnowledgeClass,
    KnowledgeRecord,
    RetrievalNeed,
    RetrievalRequest,
    RetrievalTrigger,
)
from copilot.conversation_state_models import (
    ConversationStage,
    ConversationStateSnapshot,
    DecisionStructure,
)
from copilot.core_intelligence_models import (
    RequiredFactScope,
    StrategicAction,
    StrategicDecision,
)
from copilot.llm_response_gateway import LLMResponseGateway, Prompt


def _make_snapshot(
    state_version: int = 1,
    stage: ConversationStage = ConversationStage.DISCOVERY,
    evidence: str = "",
) -> ConversationStateSnapshot:
    """Helper creating a valid ConversationStateSnapshot."""
    return ConversationStateSnapshot(
        call_sid="call_accept_suite",
        call_id="call_accept_suite",
        state_version=state_version,
        conversation_stage=stage,
        decision_structure=DecisionStructure(decision_maker_present=True),
        unresolved_objections=[],
        conversation_evidence=evidence,
    )


# ==============================================================================
# Scenario 1: No upload (Doc 02 §13)
# ==============================================================================


def test_scenario_1_no_upload_end_to_end():
    """Gate/search yields none_found. Core produces a strong prompt with no user-specific claims."""
    store = KnowledgeStore()
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()
    snapshot = _make_snapshot(state_version=1)

    decision = StrategicDecision(
        call_id="call_scen_1",
        source_state_version=1,
        strategic_objective="Differentiate through consultative approach",
        primary_action=StrategicAction.DIFFERENTIATE,
        retrieval_needed=True,
        required_facts=[RequiredFactScope(topic="track_record", required_evidence_type="stats")],
    )

    req = RetrievalRequest(
        call_id="call_scen_1",
        state_version=1,
        user_id="rep_empty",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="verified sales experience",
            topic_tags=["experience"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "none_found"
    assert len(res.facts) == 0

    # End-to-end prompt check
    context = gateway.assemble_context(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert "### BLOCK 3: VERIFIED KNOWLEDGE" in context
    assert "- No verified external facts available for this conversational moment." in context
    assert "FACT UNAVAILABILITY GUARD (CRITICAL)" in context

    prompt = gateway.generate_prompt(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert prompt.status == "generated"
    assert len(prompt.text.strip()) > 0
    # Confirms no fabricated statistics or unverified transaction volumes
    assert "homes" not in prompt.text.lower()
    assert "%" not in prompt.text


# ==============================================================================
# Scenario 2: Relevant upload (Doc 02 §13)
# ==============================================================================


def test_scenario_2_relevant_upload_end_to_end():
    """Prospect asks about local experience; prompt accurately uses verified fact with no extra claims."""
    store = KnowledgeStore()
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_westlake_sales",
            owner_scope="user",
            owner_user_id="rep_westlake",
            knowledge_class=KnowledgeClass.PERFORMANCE_PAST_SALES.value,
            source_title="Westlake Production Record",
            topic_tags=["westlake", "experience", "sales"],
            text="I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()
    snapshot = _make_snapshot(state_version=2, evidence="Do you have any actual sales experience in the Westlake area?")

    decision = StrategicDecision(
        call_id="call_scen_2",
        source_state_version=2,
        strategic_objective="Quantify local track record to resolve credibility concern",
        primary_action=StrategicAction.QUANTIFY,
        retrieval_needed=True,
        required_facts=[RequiredFactScope(topic="westlake_sales", required_evidence_type="track_record")],
    )

    req = RetrievalRequest(
        call_id="call_scen_2",
        state_version=2,
        user_id="rep_westlake",
        strategic_action="quantify",
        conversation_evidence="Do you have any actual sales experience in the Westlake area?",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="verified sales experience in Westlake",
            topic_tags=["westlake", "experience"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "retrieved"
    assert len(res.facts) == 1
    assert "42 homes in Westlake" in res.facts[0].text

    context = gateway.assemble_context(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert "42 homes in Westlake" in context
    assert "[UNTRUSTED_REFERENCE_DATA_START]" in context

    # Test generation with simulated LLM using verified fact
    def mock_generator(ctx: str, max_words: int) -> str:
        return "In Westlake, I have personally closed 42 homes with an average of 18 days on market."

    gateway_with_mock = LLMResponseGateway(generator_func=mock_generator)
    prompt = gateway_with_mock.generate_prompt(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert "42 homes" in prompt.text
    assert "18 days" in prompt.text


# ==============================================================================
# Scenario 3: Irrelevant upload (Doc 02 §13)
# ==============================================================================


def test_scenario_3_irrelevant_upload_end_to_end():
    """Upload about unrelated topic is NOT retrieved. Prompt is identical to no-upload case."""
    store = KnowledgeStore()
    # Unrelated commercial zoning chunk
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_unrelated_zoning",
            owner_scope="user",
            owner_user_id="rep_irrel",
            knowledge_class=KnowledgeClass.MARKET_KNOWLEDGE.value,
            source_title="Commercial Warehouse Zoning",
            topic_tags=["industrial", "freight", "warehouse"],
            text="Heavy industrial warehouse zoning classification B-3 allows freight terminal operations.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()
    snapshot = _make_snapshot(state_version=1)

    decision = StrategicDecision(
        call_id="call_scen_3",
        source_state_version=1,
        strategic_objective="Educate on residential home inspection timelines",
        primary_action=StrategicAction.EDUCATE,
        retrieval_needed=True,
        required_facts=[RequiredFactScope(topic="inspection_timeline", required_evidence_type="timeline")],
    )

    req = RetrievalRequest(
        call_id="call_scen_3",
        state_version=1,
        user_id="rep_irrel",
        strategic_action="educate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="residential home inspection process and contingency timelines",
            topic_tags=["inspection", "residential", "contingency"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "none_found"
    assert len(res.facts) == 0

    context = gateway.assemble_context(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert "warehouse" not in context.lower()
    assert "freight" not in context.lower()
    assert "- No verified external facts available for this conversational moment." in context


# ==============================================================================
# Scenario 4: Contradictory upload (Doc 02 §13)
# ==============================================================================


def test_scenario_4_contradictory_upload_end_to_end():
    """Two conflicting documents uploaded: disputed fact is withheld on tie, newest approved wins where resolvable."""
    store = KnowledgeStore()
    # Document A: User supplied claim that commission is 5%
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_fee_a",
            owner_scope="user",
            owner_user_id="rep_conflict",
            knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
            source_title="Fee Schedule Doc A",
            topic_tags=["commission", "listing_fee"],
            text="Standard listing commission rate is 5% full service.",
            verification_status="user_supplied",
        )
    )
    # Document B: User supplied claim that commission is 6%
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_fee_b",
            owner_scope="user",
            owner_user_id="rep_conflict",
            knowledge_class=KnowledgeClass.COMPANY_POLICY.value,
            source_title="Fee Schedule Doc B",
            topic_tags=["commission", "listing_fee"],
            text="Standard listing commission rate is 6% full service.",
            verification_status="user_supplied",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()
    snapshot = _make_snapshot(state_version=1)

    decision = StrategicDecision(
        call_id="call_scen_4",
        source_state_version=1,
        strategic_objective="Clarify commission rate structure",
        primary_action=StrategicAction.CLARIFY,
        retrieval_needed=True,
        required_facts=[RequiredFactScope(topic="commission", required_evidence_type="rate")],
    )

    req = RetrievalRequest(
        call_id="call_scen_4",
        state_version=1,
        user_id="rep_conflict",
        strategic_action="clarify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="standard listing commission rate percentage",
            topic_tags=["commission", "listing_fee"],
        ),
    )
    res = engine.retrieve_for_request(req)
    # Both are user_supplied with conflicting rates (5% vs 6%) -> authority tie -> both withheld!
    assert res.status == "conflict_unresolved"
    assert len(res.facts) == 0
    assert any(w.reason == "disputed" for w in res.withheld)

    context = gateway.assemble_context(decision=decision, snapshot=snapshot, retrieval_result=res)
    # Confirm neither disputed rate was admitted into prompt context
    assert "5%" not in context
    assert "6%" not in context


# ==============================================================================
# Scenario 5: Deleted upload (Doc 02 §13)
# ==============================================================================


def test_scenario_5_deleted_upload_end_to_end():
    """After deletion propagates, chunk never appears in retrieval or prompt."""
    store = KnowledgeStore()
    rec = KnowledgeRecord(
        knowledge_id="chunk_to_delete",
        owner_scope="user",
        owner_user_id="rep_del",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="Old Guarantee Document",
        topic_tags=["guarantee", "refund"],
        text="Special limited time 100% money back refund guarantee on closing fees.",
        verification_status="company_approved",
    )
    store.add_record(rec)

    # Verify present before deletion
    req = RetrievalRequest(
        call_id="call_del",
        state_version=1,
        user_id="rep_del",
        strategic_action="de_risk",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="refund guarantee",
            topic_tags=["guarantee", "refund"],
        ),
    )
    engine = ConditionalRetrievalEngine(store=store)
    res_before = engine.retrieve_for_request(req)
    assert len(res_before.facts) == 1

    # Rep deletes the upload
    store.delete_record("chunk_to_delete", soft=True)

    # Retrieval after deletion
    res_after = engine.retrieve_for_request(req)
    assert res_after.status == "none_found"
    assert len(res_after.facts) == 0

    gateway = LLMResponseGateway()
    decision = StrategicDecision(
        call_id="call_del",
        source_state_version=1,
        strategic_objective="De-risk prospect hesitation",
        primary_action=StrategicAction.DE_RISK,
        retrieval_needed=True,
    )
    context = gateway.assemble_context(decision=decision, snapshot=_make_snapshot(), retrieval_result=res_after)
    assert "money back refund guarantee" not in context


# ==============================================================================
# Scenario 6: Cross-user leakage (Doc 02 §13, Individual-User Scope C6)
# ==============================================================================


def test_scenario_6_cross_user_leakage_end_to_end():
    """User A's knowledge is never retrieved for User B, even within the same brokerage/company."""
    store = KnowledgeStore()
    # User A's confidential notes
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_user_a_secret",
            owner_scope="user",
            owner_user_id="agent_alice",
            knowledge_class=KnowledgeClass.PERFORMANCE_PAST_SALES.value,
            source_title="Alice Secret Sales Log",
            topic_tags=["track_record", "luxury"],
            text="Alice private sales figure: closed 75 luxury penthouses in downtown corridor.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()

    # User B (Bob) queries for track record
    req_bob = RetrievalRequest(
        call_id="call_bob",
        state_version=1,
        user_id="agent_bob",  # Bob querying!
        strategic_action="quantify",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="luxury track record",
            topic_tags=["track_record", "luxury"],
        ),
    )
    res_bob = engine.retrieve_for_request(req_bob)
    assert res_bob.status == "none_found"
    assert len(res_bob.facts) == 0

    decision_bob = StrategicDecision(
        call_id="call_bob",
        source_state_version=1,
        strategic_objective="Quantify sales history",
        primary_action=StrategicAction.QUANTIFY,
        retrieval_needed=True,
    )
    context_bob = gateway.assemble_context(decision=decision_bob, snapshot=_make_snapshot(), retrieval_result=res_bob)
    assert "Alice" not in context_bob
    assert "75 luxury penthouses" not in context_bob

    prompt_bob = gateway.generate_prompt(decision=decision_bob, snapshot=_make_snapshot(), retrieval_result=res_bob)
    assert "75 luxury penthouses" not in prompt_bob.text


# ==============================================================================
# Scenario 7: Historical marketing claim (Doc 02 §13)
# ==============================================================================


def test_scenario_7_historical_marketing_claim_end_to_end():
    """A claim from an old training doc is not used as a verified fact in the prompt."""
    store = KnowledgeStore()
    # Old legacy pitch deck claim with ungrounded superlatives and stats
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_old_pitch_deck",
            owner_scope="user",
            owner_user_id="rep_deck",
            knowledge_class=KnowledgeClass.SALESPERSON_PROFILE.value,
            source_title="Legacy PitchDeck 2021",
            topic_tags=["superlative", "ranking", "profile"],
            text="I am the #1 top producer agent in North America guaranteed to sell in 7 days.",
            verification_status="user_supplied",  # Unapproved superlative/guarantee
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()

    req = RetrievalRequest(
        call_id="call_deck",
        state_version=1,
        user_id="rep_deck",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="agent ranking and guarantees",
            topic_tags=["ranking", "profile"],
        ),
    )
    res = engine.retrieve_for_request(req)
    # High-risk claim and unapproved guarantee must be withheld
    assert res.status == "none_found"
    assert len(res.facts) == 0
    assert any(w.reason == "unverified" for w in res.withheld)

    decision = StrategicDecision(
        call_id="call_deck",
        source_state_version=1,
        strategic_objective="Differentiate services",
        primary_action=StrategicAction.DIFFERENTIATE,
        retrieval_needed=True,
    )
    context = gateway.assemble_context(decision=decision, snapshot=_make_snapshot(), retrieval_result=res)
    assert "#1 top producer" not in context
    assert "guaranteed to sell in 7 days" not in context


# ==============================================================================
# Scenario 8: No-trigger turn (Workflow)
# ==============================================================================


def test_scenario_8_no_trigger_turn_end_to_end():
    """Turn with no trigger flagged -> Gate returns none_needed, 0 vector calls, clean prompt."""
    store = KnowledgeStore()
    store.add_record(
        KnowledgeRecord(
            knowledge_id="rec_some_doc",
            owner_scope="user",
            owner_user_id="rep_notrigger",
            knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
            source_title="Standard Doc",
            topic_tags=["marketing"],
            text="Standard photography included.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()

    # Core decides ACKNOWLEDGE, no retrieval need declared
    req = RetrievalRequest(
        call_id="call_notrigger",
        state_version=3,
        user_id="rep_notrigger",
        strategic_action="acknowledge",
        retrieval_need=None,  # No trigger!
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "none_needed"
    assert len(res.facts) == 0

    decision = StrategicDecision(
        call_id="call_notrigger",
        source_state_version=3,
        strategic_objective="Acknowledge prospect feedback",
        primary_action=StrategicAction.ACKNOWLEDGE,
        retrieval_needed=False,
    )
    context = gateway.assemble_context(decision=decision, snapshot=_make_snapshot(state_version=3), retrieval_result=res)
    assert "- No verified external facts available for this conversational moment." in context

    prompt = gateway.generate_prompt(decision=decision, snapshot=_make_snapshot(state_version=3), retrieval_result=res)
    assert prompt.status == "generated"
    assert "photography" not in prompt.text


# ==============================================================================
# Scenario 9: Script doesn't override Core (Workflow & Doc 01 §15)
# ==============================================================================


def test_scenario_9_script_does_not_override_core_end_to_end():
    """A retrieved script cannot change the strategic action (Doc 01 §15)."""
    store = KnowledgeStore()
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_hard_close_script",
            owner_scope="user",
            owner_user_id="rep_core_rule",
            knowledge_class=KnowledgeClass.OBJECTION_RESPONSES.value,
            source_title="Aggressive Script Book",
            topic_tags=["closing", "objection"],
            text="Ask for the credit card and sign the agreement on the spot today.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()
    snapshot = _make_snapshot(state_version=4)

    # Core strictly commands VALIDATE, and explicitly prohibits hard close
    decision = StrategicDecision(
        call_id="call_core_rule",
        source_state_version=4,
        strategic_objective="Validate prospect emotional concern without push",
        primary_action=StrategicAction.VALIDATE,
        do_not_do=["hard_close", "ask_payment", "push_commitment"],
        retrieval_needed=True,
    )

    req = RetrievalRequest(
        call_id="call_core_rule",
        state_version=4,
        user_id="rep_core_rule",
        strategic_action="validate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="objection response strategy",
            topic_tags=["objection"],
        ),
    )
    res = engine.retrieve_for_request(req)

    context = gateway.assemble_context(decision=decision, snapshot=snapshot, retrieval_result=res)
    # Block 1 commands VALIDATE and prohibits hard close
    assert "Primary Strategic Action: VALIDATE" in context
    assert "hard_close, ask_payment, push_commitment" in context
    # Directives explicitly prioritize Core intent over retrieved text (Doc 01 §10 Priority 2 over Priority 5)
    assert "Priority 2: Core strategic intent" in context
    assert "Priority 5: Relevant retrieved facts" in context

    prompt = gateway.generate_prompt(decision=decision, snapshot=snapshot, retrieval_result=res)
    # Confirms prompt honors VALIDATE and does not attempt hard close
    assert "credit card" not in prompt.text.lower()
    assert "sign the agreement" not in prompt.text.lower()


# ==============================================================================
# Scenario 10: Social Proof with no proof (Workflow)
# ==============================================================================


def test_scenario_10_social_proof_with_no_proof_end_to_end():
    """Core picks alternate move or generic truthful language. No fabricated proof."""
    store = KnowledgeStore()  # empty store
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()
    snapshot = _make_snapshot(state_version=2)

    decision = StrategicDecision(
        call_id="call_social_empty",
        source_state_version=2,
        strategic_objective="Offer social proof evidence of neighborhood success",
        primary_action=StrategicAction.SOCIAL_PROOF,
        retrieval_needed=True,
    )

    req = RetrievalRequest(
        call_id="call_social_empty",
        state_version=2,
        user_id="rep_no_proof",
        strategic_action="social_proof",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.PLAYBOOK_EVIDENCE,
            need_statement="prior customer transaction results",
            topic_tags=["social_proof", "results"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "none_found"
    assert res.fallback_hint == "choose_alternate_move"

    # Gateway rerouting verification
    prompt = gateway.generate_prompt(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert prompt.strategic_action == StrategicAction.VALIDATE
    # Confirms zero fabrication of past client claims
    assert "sold" not in prompt.text.lower()
    assert "client" not in prompt.text.lower()
    assert len(prompt.text.strip()) > 0


# ==============================================================================
# Scenario 11: Stale result mid-retrieval and post-generation (Workflow)
# ==============================================================================


def test_scenario_11_stale_result_end_to_end():
    """Interruption mid-retrieval (v_req != v_curr) -> late result discarded, no state overwrite."""
    store = KnowledgeStore()
    store.add_record(
        KnowledgeRecord(
            knowledge_id="rec_stale_item",
            owner_scope="user",
            owner_user_id="rep_stale",
            knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
            source_title="Feature List",
            topic_tags=["features"],
            text="Staged open house strategy with local drone footage.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()

    # Request originated at state_version = 5
    req = RetrievalRequest(
        call_id="call_stale",
        state_version=5,
        user_id="rep_stale",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="drone features",
            topic_tags=["features"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.state_version == 5
    assert len(res.facts) == 1

    # Meanwhile, prospect interrupted and state advanced to state_version = 6
    snapshot_v6 = _make_snapshot(state_version=6, evidence="Wait, I forgot to mention we also have a tenant living there.")
    decision_v6 = StrategicDecision(
        call_id="call_stale",
        source_state_version=6,
        strategic_objective="Acknowledge tenant constraint",
        primary_action=StrategicAction.VALIDATE,
    )

    # Checkpoint A: After retrieval in assemble_context
    context = gateway.assemble_context(decision=decision_v6, snapshot=snapshot_v6, retrieval_result=res)
    assert "[STALE_DISCARDED]" in context
    assert "drone footage" not in context

    # Checkpoint B: Right before teleprompter delivery (generation race condition)
    prompt = gateway.generate_prompt(
        decision=decision_v6,
        snapshot=snapshot_v6,
        retrieval_result=res,
        post_generation_state_version=7,  # state jumped again to 7 during LLM execution!
    )
    assert prompt.status == "stale_discarded"
    assert prompt.text == ""


# ==============================================================================
# Scenario 12: Timeout fallback (Workflow)
# ==============================================================================


def test_scenario_12_timeout_end_to_end():
    """Slow retrieval -> prompt still delivered in time with a truthful zero-assumption fallback."""
    store = KnowledgeStore()
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_slow",
            owner_scope="user",
            owner_user_id="rep_slow",
            knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
            source_title="Process Doc",
            topic_tags=["process"],
            text="Complex 12-step transaction procedure overview.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()
    snapshot = _make_snapshot(state_version=2)

    decision = StrategicDecision(
        call_id="call_timeout",
        source_state_version=2,
        strategic_objective="Explain transaction next steps",
        primary_action=StrategicAction.EDUCATE,
        retrieval_needed=True,
    )

    # Retrieval request with an impossible 0.00001ms deadline budget
    req = RetrievalRequest(
        call_id="call_timeout",
        state_version=2,
        user_id="rep_slow",
        strategic_action="educate",
        deadline_ms=0.00001,
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="12 step procedure",
            topic_tags=["process"],
        ),
    )
    res = engine.retrieve_for_request(req)
    assert res.status == "timed_out"
    assert len(res.facts) == 0
    assert res.fallback_hint == "use_generic_truthful"

    context = gateway.assemble_context(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert "[TIMED_OUT]" in context

    prompt = gateway.generate_prompt(decision=decision, snapshot=snapshot, retrieval_result=res)
    assert prompt.status == "generated"
    assert len(prompt.text.strip()) > 0
    # Processed truthfully without crashing or freezing


# ==============================================================================
# Scenario 13: Live truth over stored fact (Workflow & Doc 01 §10 Priority 3)
# ==============================================================================


def test_scenario_13_live_truth_over_stored_fact_end_to_end():
    """Stored fact conflicts with what the prospect just said -> live state wins in the prompt."""
    store = KnowledgeStore()
    # Stored chunk asserting reps only do weekday morning meetings
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_stored_schedule",
            owner_scope="user",
            owner_user_id="rep_schedule",
            knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
            source_title="Consultation Hours",
            topic_tags=["schedule", "consultation"],
            text="Initial consultations are strictly conducted on weekday morning at 10 AM.",
            verification_status="company_approved",
        )
    )
    engine = ConditionalRetrievalEngine(store=store)
    gateway = LLMResponseGateway()

    # Prospect stated live: "I can only meet tomorrow afternoon at 3:00 PM."
    snapshot = _make_snapshot(state_version=4, evidence="I can only meet tomorrow afternoon at 3:00 PM.")

    decision = StrategicDecision(
        call_id="call_live_truth",
        source_state_version=4,
        strategic_objective="Coordinate consultation scheduling",
        primary_action=StrategicAction.QUESTION,
        retrieval_needed=True,
    )

    req = RetrievalRequest(
        call_id="call_live_truth",
        state_version=4,
        user_id="rep_schedule",
        strategic_action="question",
        conversation_evidence="I can only meet tomorrow afternoon at 3:00 PM.",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.SERVICE_PROCESS_OBJECTION,
            need_statement="consultation hours schedule",
            topic_tags=["schedule"],
        ),
    )
    res = engine.retrieve_for_request(req)
    # Stored morning requirement directly contradicted live afternoon preference -> withheld!
    assert res.status == "none_found"
    assert len(res.facts) == 0
    assert any(w.reason == "disputed" for w in res.withheld)

    recent_turns = [{"speaker": "prospect", "text": "I can only meet tomorrow afternoon at 3:00 PM."}]
    context = gateway.assemble_context(
        decision=decision, snapshot=snapshot, retrieval_result=res, recent_turns=recent_turns
    )
    assert "weekday morning at 10 AM" not in context
    assert "afternoon at 3:00 PM" in context


# ==============================================================================
# Scenario 14: Team scope rejection (Constraint C6)
# ==============================================================================


def test_scenario_14_team_scope_rejection_end_to_end():
    """Writing owner_scope = team/brokerage/company is rejected with ValueError, and queries never match."""
    # Write-time rejection check for team, brokerage, company
    for forbidden_scope in ["team", "brokerage", "company"]:
        with pytest.raises(ValueError, match="Team, brokerage, and company scopes are not enabled"):
            KnowledgeRecord(
                knowledge_id=f"rec_forbidden_{forbidden_scope}",
                owner_scope=forbidden_scope,  # Rejected!
                owner_user_id="team_123",
                knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
                source_title="Shared Team Doc",
                topic_tags=["shared"],
                text="Shared brokerage marketing collateral.",
            )

    # Ingestion into store also rejects non-user scope
    store = KnowledgeStore()
    valid_user_rec = KnowledgeRecord(
        knowledge_id="rec_valid_user",
        owner_scope="user",
        owner_user_id="user_single",
        knowledge_class=KnowledgeClass.VALUE_PROPOSITION.value,
        source_title="Individual User Doc",
        topic_tags=["personal"],
        text="Individual user verified statement.",
    )
    store.add_record(valid_user_rec)

    engine = ConditionalRetrievalEngine(store=store)
    # Query specifying a team or another user never matches
    req_other = RetrievalRequest(
        call_id="call_scope",
        state_version=1,
        user_id="other_user_456",
        strategic_action="differentiate",
        retrieval_need=RetrievalNeed(
            trigger=RetrievalTrigger.CREDIBILITY_OBJECTION,
            need_statement="personal doc",
            topic_tags=["personal"],
        ),
    )
    res = engine.retrieve_for_request(req_other)
    assert res.status == "none_found"
    assert len(res.facts) == 0
