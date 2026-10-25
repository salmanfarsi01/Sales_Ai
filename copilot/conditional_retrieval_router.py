"""FastAPI router and API endpoints for Conditional Knowledge Retrieval testing & inspection.

Mounts the dedicated testing console and provides live simulation, document
management, and stage-by-stage pipeline inspection for conditional retrieval.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from copilot.conditional_retrieval import (
    ConditionalRetrievalEngine,
    KnowledgeStore,
    LatencyMetricsAggregator,
    RetrievalCache,
    RetrievalConfig,
    RetrievalGate,
)
from copilot.conditional_retrieval_models import (
    KnowledgeClass,
    KnowledgeRecord,
    RetrievalNeed,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrigger,
)
from copilot.conversation_state_models import (
    ConversationStage,
    ConversationStateSnapshot,
    DecisionStructure,
    ObjectionLifecycleState,
    ObjectionRecord,
)
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_intelligence_models import (
    RequiredFactScope,
    StrategicAction,
    StrategicDecision,
)
from copilot.llm_response_gateway import LLMResponseGateway, validate_prompt_hallucinations

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class IngestChunkRequest(BaseModel):
    knowledge_id: Optional[str] = None
    owner_user_id: str = "rep_john"
    owner_scope: str = "user"
    knowledge_class: str = KnowledgeClass.PERFORMANCE_PAST_SALES.value
    source_title: str = "Knowledge Chunk"
    source_version: str = "1.0"
    topic_tags: List[str] = Field(default_factory=list)
    text: str
    verification_status: str = "company_approved"
    sensitivity: str = "normal"
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    lead_types: Optional[List[str]] = None
    deleted: bool = False
    superseded_by: Optional[str] = None
    self_attest: bool = False


class DeleteRecordRequest(BaseModel):
    user_id: str
    knowledge_id: str
    soft_tombstone: bool = True  # True sets deleted=True, False removes from memory


class AttestRecordRequest(BaseModel):
    user_id: str
    knowledge_id: str
    tamper_text: Optional[str] = None  # If provided, modifies text to simulate hash mismatch


class SimulationRequest(BaseModel):
    user_id: str = "rep_john"
    call_id: str = "sim_test_call"
    state_version: int = 1
    conversation_evidence: str = "Do you have any actual sales experience in Westlake?"
    strategic_action: str = "quantify"
    trigger: Optional[str] = "credibility_objection"
    need_statement: str = "verified sales experience in Westlake"
    topic_tags: List[str] = Field(default_factory=lambda: ["westlake", "experience", "sales"])
    time_sensitive: bool = False
    lead_type: Optional[str] = "residential_seller"
    prospect_type: Optional[str] = None
    deadline_ms: Optional[float] = 250.0
    allow_confidential: bool = False
    simulate_interruption: bool = False  # If True, increments state_version post-retrieval
    prospect_live_override: Optional[str] = None  # E.g. prospect says afternoon when doc says morning


class MultiTurnScriptRequest(BaseModel):
    user_id: str = "rep_john"
    call_id: str = "multiturn_script_001"
    script_name: Optional[str] = "Canonical 8-Turn Benchmark"
    prospect_lines: List[str] = Field(default_factory=list)
    lead_type: Optional[str] = "residential_seller"
    prospect_type: Optional[str] = None
    deadline_ms: Optional[float] = 250.0
    allow_confidential: bool = False


# -----------------------------------------------------------------------------
# Helper: Seed Canonical Dummy Documents (Rep Profile, Sales, Pricing, Policy & Traps)
# -----------------------------------------------------------------------------

def seed_canonical_dummy_documents(store: KnowledgeStore, user_id: str = "rep_john") -> None:
    """Seeds the canonical dummy documents covering profile, past sales, pricing, policy, and traps."""
    # 0. Salesperson Profile Document
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_salesperson_profile",
            owner_scope="user",
            owner_user_id=user_id,
            knowledge_class=KnowledgeClass.SALESPERSON_PROFILE.value,
            source_title="Agent John Miller Bio",
            source_version="1.0",
            topic_tags=["profile", "experience", "bio", "background", "credentials"],
            text="John Miller has 12 years of residential real estate experience and is a certified luxury home marketing specialist.",
            verification_status="company_approved",
            sensitivity="normal",
        )
    )

    # 1. Verified Local Track Record (Positive match for Credibility objection)
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_westlake_sales",
            owner_scope="user",
            owner_user_id=user_id,
            knowledge_class=KnowledgeClass.PERFORMANCE_PAST_SALES.value,
            source_title="Westlake Production Record",
            source_version="1.0",
            topic_tags=["westlake", "experience", "sales", "track_record"],
            text="I have closed 42 homes in Westlake over the past 3 years with an average of 18 days on market.",
            verification_status="company_approved",
            sensitivity="normal",
        )
    )

    # 2. Irrelevant / Out-of-Domain Document (Relevance floor test)
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_unrelated_zoning",
            owner_scope="user",
            owner_user_id=user_id,
            knowledge_class=KnowledgeClass.MARKET_KNOWLEDGE.value,
            source_title="Commercial Warehouse Zoning",
            source_version="1.0",
            topic_tags=["industrial", "freight", "warehouse", "zoning"],
            text="Heavy industrial warehouse zoning classification B-3 allows freight terminal operations with 24-foot clearance.",
            verification_status="company_approved",
            sensitivity="normal",
        )
    )

    # 3A. Legacy Commission Policy (Old version)
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_commission_v1",
            owner_scope="user",
            owner_user_id=user_id,
            knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
            source_title="Listing Agreement Schedule 2023",
            source_version="1.0",
            topic_tags=["commission", "fees", "pricing"],
            text="Standard residential listing commission rate is 6.0% of gross purchase price.",
            verification_status="company_approved",
            sensitivity="normal",
        )
    )

    # 3B. Current Commission Policy (Newer version, supersedes 6%)
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_commission_v2",
            owner_scope="user",
            owner_user_id=user_id,
            knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
            source_title="Listing Agreement Schedule 2026",
            source_version="2.0",
            topic_tags=["commission", "fees", "pricing"],
            text="Standard residential listing commission rate is 5.0% of gross purchase price.",
            verification_status="company_approved",
            sensitivity="normal",
        )
    )

    # 4. Unverified Superlative & Guarantee (Guardrail test)
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_old_pitch_deck",
            owner_scope="user",
            owner_user_id=user_id,
            knowledge_class=KnowledgeClass.SALESPERSON_PROFILE.value,
            source_title="Legacy PitchDeck 2021",
            source_version="1.0",
            topic_tags=["ranking", "guarantee", "profile", "speed"],
            text="I am the #1 top producer agent in North America guaranteed to sell your home in 7 days.",
            verification_status="user_supplied",
            sensitivity="normal",
        )
    )

    # 5. Another User's Document (Constraint C6 Isolation test)
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_sarah_rebate",
            owner_scope="user",
            owner_user_id="rep_sarah",  # Different user
            knowledge_class=KnowledgeClass.PRICING_PACKAGING.value,
            source_title="Sarah VIP Buyer Rebate",
            source_version="1.0",
            topic_tags=["rebate", "cashback", "buyer", "incentive"],
            text="Exclusive $5,000 cash back buyer rebate credit given at closing for all executive listings.",
            verification_status="company_approved",
            sensitivity="normal",
        )
    )

    # 6. Stored Consultation Hours (Live truth conflict test)
    store.add_record(
        KnowledgeRecord(
            knowledge_id="chunk_stored_schedule",
            owner_scope="user",
            owner_user_id=user_id,
            knowledge_class=KnowledgeClass.SERVICE_PROCESS.value,
            source_title="Consultation Hours Policy",
            source_version="1.0",
            topic_tags=["schedule", "consultation", "hours", "appointment"],
            text="Initial consultations are strictly conducted on weekday mornings at 10:00 AM.",
            verification_status="company_approved",
            sensitivity="normal",
        )
    )


# -----------------------------------------------------------------------------
# Factory to Mount the Router
# -----------------------------------------------------------------------------

def get_conditional_retrieval_router(
    html_file_path: Optional[Path] = None,
    engine: Optional[ConditionalRetrievalEngine] = None,
    gateway: Optional[LLMResponseGateway] = None,
) -> APIRouter:
    """Builds and returns the APIRouter for conditional retrieval."""
    router = APIRouter(tags=["conditional-retrieval"])

    if html_file_path is None:
        html_file_path = Path(__file__).resolve().parent.parent / "web" / "conditional_retrieval_test.html"

    # Shared instances for test UI session
    shared_engine = engine or ConditionalRetrievalEngine()
    shared_gateway = gateway or LLMResponseGateway()

    # Seed initial dummy docs for rep_john
    seed_canonical_dummy_documents(shared_engine.store, user_id="rep_john")

    # -------------------------------------------------------------------------
    # UI Web Endpoints
    # -------------------------------------------------------------------------

    @router.get("/conditional-test")
    @router.get("/conditional-retrieval")
    async def serve_conditional_test_ui():
        if not html_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"UI file {html_file_path.name} not found. Ensure web/{html_file_path.name} exists.",
            )
        return FileResponse(html_file_path)

    # -------------------------------------------------------------------------
    # 14 Acceptance Presets Catalog
    # -------------------------------------------------------------------------

    @router.get("/api/conditional-retrieval/presets")
    async def get_acceptance_presets() -> List[Dict[str, Any]]:
        return [
            {
                "id": "scenario_1",
                "title": "Scenario 1: No Upload (Doc 02 §13)",
                "category": "Basic Functionality",
                "user_id": "rep_empty",
                "prospect_utterance": "Why should I pick you over the big commercial agency?",
                "strategic_action": "differentiate",
                "trigger": "credibility_objection",
                "need_statement": "verified sales experience and track record",
                "topic_tags": ["experience", "track_record"],
                "state_version": 1,
                "expected": "Gate yields none_found. Zero invented facts; strong truthful prompt generated without claims.",
            },
            {
                "id": "scenario_2",
                "title": "Scenario 2: Relevant Upload (Doc 02 §13)",
                "category": "Grounding & Quotes",
                "user_id": "rep_john",
                "prospect_utterance": "Do you have any actual sales experience in the Westlake area?",
                "strategic_action": "quantify",
                "trigger": "credibility_objection",
                "need_statement": "verified sales experience in Westlake",
                "topic_tags": ["westlake", "experience", "sales"],
                "state_version": 2,
                "expected": "Status retrieved. Accurately cites '42 homes in Westlake' and '18 days' without extra claims.",
            },
            {
                "id": "scenario_3",
                "title": "Scenario 3: Irrelevant Upload (Doc 02 §13)",
                "category": "Relevance Floor",
                "user_id": "rep_john",
                "prospect_utterance": "How long does the residential home inspection contingency take?",
                "strategic_action": "educate",
                "trigger": "service_process_objection",
                "need_statement": "residential home inspection contingency process and timeline",
                "topic_tags": ["inspection", "contingency", "timeline"],
                "state_version": 1,
                "expected": "Warehouse zoning chunk dropped below 0.35 floor -> none_found. Zero warehouse mentions in prompt.",
            },
            {
                "id": "scenario_4",
                "title": "Scenario 4: Contradictory Upload (Doc 02 §13)",
                "category": "Conflict Table",
                "user_id": "rep_john",
                "prospect_utterance": "What commission rate do you charge to list my home?",
                "strategic_action": "educate",
                "trigger": "policy_wording",
                "need_statement": "standard residential listing commission fee",
                "topic_tags": ["commission", "fees", "pricing"],
                "state_version": 1,
                "expected": "Newer v2.0 approved doc (5.0%) wins; older v1.0 (6.0%) withheld with reason='disputed'. Cites 5% only.",
            },
            {
                "id": "scenario_5",
                "title": "Scenario 5: Deleted Upload (Eventual Consistency)",
                "category": "Eventual Consistency",
                "user_id": "rep_john",
                "prospect_utterance": "Can you share your past sales volume numbers in Westlake?",
                "strategic_action": "quantify",
                "trigger": "credibility_objection",
                "need_statement": "sales volume numbers in Westlake",
                "topic_tags": ["westlake", "sales"],
                "state_version": 1,
                "pre_action": "soft_delete_westlake",
                "expected": "Authoritative store re-check intercepts candidate at 0ms (deleted=True) -> none_found. No stale leak.",
            },
            {
                "id": "scenario_6",
                "title": "Scenario 6: Cross-User Leakage Defense (C6)",
                "category": "Security & Privacy",
                "user_id": "rep_john",
                "prospect_utterance": "Do you offer any closing rebate credits or incentives?",
                "strategic_action": "educate",
                "trigger": "factual_question",
                "need_statement": "closing rebate credit or incentive programs",
                "topic_tags": ["rebate", "credit", "incentive"],
                "state_version": 1,
                "expected": "rep_sarah's $5,000 rebate chunk is filtered out by in-query owner_user_id pre-filter -> none_found.",
            },
            {
                "id": "scenario_7",
                "title": "Scenario 7: Historical Marketing Superlative",
                "category": "Guardrails & Attestation",
                "user_id": "rep_john",
                "prospect_utterance": "Can you guarantee you will sell my home in under a week?",
                "strategic_action": "differentiate",
                "trigger": "credibility_objection",
                "need_statement": "agent speed guarantee and ranking claims",
                "topic_tags": ["ranking", "guarantee"],
                "state_version": 1,
                "expected": "Legacy '#1 producer' and '7-day guarantee' are user_supplied -> withheld with reason='unverified'.",
            },
            {
                "id": "scenario_8",
                "title": "Scenario 8: No-Trigger Turn (Workflow)",
                "category": "Fast Short-Circuit",
                "user_id": "rep_john",
                "prospect_utterance": "Yeah, okay, that makes sense. I see what you mean.",
                "strategic_action": "acknowledge",
                "trigger": "none",
                "need_statement": "",
                "topic_tags": [],
                "state_version": 1,
                "expected": "Gate identifies no trigger and excluded action -> fast exit status='none_needed' (0ms, 0 vector calls).",
            },
            {
                "id": "scenario_9",
                "title": "Scenario 9: Script Cannot Override Core Strategy",
                "category": "Core Primacy",
                "user_id": "rep_john",
                "prospect_utterance": "I'm still comparing agents and want to understand your process.",
                "strategic_action": "educate",
                "trigger": "service_process_objection",
                "need_statement": "listing process walkthrough",
                "topic_tags": ["process"],
                "state_version": 1,
                "expected": "Retrieved aggressive closing scripts cannot alter Core's primary action EDUCATE to a hard close.",
            },
            {
                "id": "scenario_10",
                "title": "Scenario 10: Social Proof with No Proof",
                "category": "Truthful Fallbacks",
                "user_id": "rep_john",
                "prospect_utterance": "Do you have testimonials from luxury home sellers nearby?",
                "strategic_action": "social_proof",
                "trigger": "playbook_evidence",
                "need_statement": "luxury seller client testimonials",
                "topic_tags": ["testimonials", "luxury"],
                "state_version": 1,
                "expected": "No testimonials uploaded -> Gate returns fallback_hint='choose_alternate_move'. Zero fabricated proof.",
            },
            {
                "id": "scenario_11",
                "title": "Scenario 11: Stale Result Discard (Interruption)",
                "category": "Version Safety",
                "user_id": "rep_john",
                "prospect_utterance": "Actually, wait, never mind that, my spouse just arrived.",
                "strategic_action": "acknowledge",
                "trigger": "factual_question",
                "need_statement": "pricing question",
                "topic_tags": ["pricing"],
                "state_version": 1,
                "simulate_interruption": True,
                "expected": "Conversation state advanced to version 2 during generation -> result marked stale_discarded before delivery.",
            },
            {
                "id": "scenario_12",
                "title": "Scenario 12: Retrieval Timeout Fallback",
                "category": "Latency Safety",
                "user_id": "rep_john",
                "prospect_utterance": "Can you check the exact county tax parcel assessment details?",
                "strategic_action": "educate",
                "trigger": "factual_question",
                "need_statement": "county tax assessment parcel data",
                "topic_tags": ["taxes", "assessment"],
                "state_version": 1,
                "deadline_ms": 1.0,  # Forces timeout
                "expected": "Retrieval exceeded 1ms deadline -> status='timed_out'. Prompt generated using generic truthful fallback.",
            },
            {
                "id": "scenario_13",
                "title": "Scenario 13: Live Truth Over Stored Fact",
                "category": "Live Truth Priority",
                "user_id": "rep_john",
                "prospect_utterance": "I can only meet tomorrow afternoon at 3:00 PM.",
                "strategic_action": "question",
                "trigger": "service_process_objection",
                "need_statement": "consultation hours schedule",
                "topic_tags": ["schedule"],
                "state_version": 4,
                "prospect_live_override": "I can only meet tomorrow afternoon at 3:00 PM.",
                "expected": "Stored morning chunk contradicts prospect's live statement -> stored chunk withheld (disputed). Live truth wins.",
            },
            {
                "id": "scenario_14",
                "title": "Scenario 14: Team Scope Rejection (C6)",
                "category": "Security & Privacy",
                "user_id": "rep_john",
                "prospect_utterance": "What are your company-wide statistics?",
                "strategic_action": "quantify",
                "trigger": "credibility_objection",
                "need_statement": "brokerage company statistics",
                "topic_tags": ["brokerage"],
                "state_version": 1,
                "expected": "Pydantic validator strictly rejects team/brokerage scopes with ValueError. Zero organizational sharing.",
            },
        ]

    # -------------------------------------------------------------------------
    # Knowledge Store Management Endpoints
    # -------------------------------------------------------------------------

    @router.get("/api/conditional-retrieval/records")
    async def list_user_records(user_id: str = Query("rep_john")) -> Dict[str, Any]:
        """Lists records for a given user from the authoritative store."""
        records = [
            r for r in shared_engine.store._records.values()
            if r.owner_user_id == user_id
        ]
        user_version = shared_engine.store.get_user_knowledge_version(user_id)
        return {
            "user_id": user_id,
            "user_knowledge_version": user_version,
            "total_records": len(records),
            "records": [r.model_dump() for r in records],
        }

    @router.post("/api/conditional-retrieval/records")
    async def ingest_or_update_record(req: IngestChunkRequest) -> Dict[str, Any]:
        """Adds or updates a chunk in the authoritative store."""
        kid = req.knowledge_id or f"chunk_{uuid.uuid4().hex[:8]}"
        try:
            record = KnowledgeRecord(
                knowledge_id=kid,
                owner_scope=req.owner_scope,
                owner_user_id=req.owner_user_id,
                knowledge_class=req.knowledge_class,
                source_title=req.source_title,
                source_version=req.source_version,
                topic_tags=req.topic_tags,
                text=req.text,
                verification_status=req.verification_status,
                sensitivity=req.sensitivity,
                effective_from=req.effective_from,
                effective_to=req.effective_to,
                lead_types=req.lead_types,
                deleted=req.deleted,
                superseded_by=req.superseded_by,
            )
            shared_engine.store.add_record(record)
            if req.self_attest:
                shared_engine.store.attest_record(
                    knowledge_id=kid,
                    user_id=req.owner_user_id,
                    attestation_notes="Self-attested via Test Console UI",
                )
            return {"status": "success", "knowledge_id": kid, "record": record.model_dump()}
        except Exception as e:
            LOGGER.exception("Failed to ingest record: %s", e)
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/api/conditional-retrieval/records/delete")
    async def delete_record(req: DeleteRecordRequest) -> Dict[str, Any]:
        """Soft-deletes (tombstones) or purges a record."""
        rec = shared_engine.store.get_record(req.knowledge_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Record not found")
        if rec.owner_user_id != req.user_id:
            raise HTTPException(status_code=403, detail="Cross-user deletion forbidden (Constraint C6)")

        if req.soft_tombstone:
            shared_engine.store.delete_record(req.knowledge_id)
            return {"status": "success", "action": "soft_tombstoned", "knowledge_id": req.knowledge_id}
        else:
            shared_engine.store._records.pop(req.knowledge_id, None)
            shared_engine.store.bump_user_knowledge_version(req.user_id)
            return {"status": "success", "action": "purged", "knowledge_id": req.knowledge_id}

    @router.post("/api/conditional-retrieval/records/attest")
    async def attest_or_tamper_record(req: AttestRecordRequest) -> Dict[str, Any]:
        """Attests a record or deliberately tampers text to test SHA-256 hash invalidation."""
        rec = shared_engine.store.get_record(req.knowledge_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Record not found")

        if req.tamper_text is not None:
            # Tamper the text without updating hash
            rec.text = req.tamper_text
            shared_engine.store._records[req.knowledge_id] = rec
            shared_engine.store.bump_user_knowledge_version(req.user_id)
            return {
                "status": "success",
                "action": "text_tampered",
                "message": "Text modified while retaining old attestation hash. Authoritative re-check will detect mismatch.",
            }
        else:
            res = shared_engine.store.attest_record(
                knowledge_id=req.knowledge_id,
                user_id=req.user_id,
                attestation_notes="Console UI Attestation",
            )
            return {"status": "success", "action": "attested", "record": res.model_dump()}

    @router.post("/api/conditional-retrieval/load-canonical-dummies")
    async def load_canonical_dummies(user_id: str = Query("rep_john")) -> Dict[str, Any]:
        """Resets the store and reloads the 6 canonical dummy documents."""
        # Purge existing for this user
        for kid in list(shared_engine.store._records.keys()):
            if shared_engine.store._records[kid].owner_user_id in (user_id, "rep_sarah"):
                del shared_engine.store._records[kid]

        seed_canonical_dummy_documents(shared_engine.store, user_id=user_id)
        shared_engine.store.bump_user_knowledge_version(user_id)
        return {"status": "success", "message": f"Reloaded canonical dummy docs for {user_id} and rep_sarah."}

    # -------------------------------------------------------------------------
    # Pipeline Turn Simulation Endpoint
    # -------------------------------------------------------------------------

    @router.post("/api/conditional-retrieval/simulate")
    async def simulate_turn_pipeline(req: SimulationRequest) -> Dict[str, Any]:
        """Runs the entire conditional retrieval and prompt generation pipeline."""
        start_sim_time = time.monotonic()

        # Step 1: Map trigger & need (supports Autonomous Core Decision mode)
        core_decision_info = None
        action_to_use = req.strategic_action

        if req.trigger and req.trigger.lower() in ("auto", "core_autonomous", "core_decides"):
            from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
            core_engine_instance = PitchProXCoreIntelligenceEngine()
            eval_snapshot = ConversationStateSnapshot(
                call_sid=req.call_id,
                call_id=req.call_id,
                state_version=req.state_version,
                conversation_stage=ConversationStage.OBJECTION_HANDLING if any(w in req.conversation_evidence.lower() for w in ("experience", "fee", "rate", "cost", "how long")) else ConversationStage.DISCOVERY,
                decision_structure=DecisionStructure(decision_maker_present=True),
                unresolved_objections=[],
            )
            eval_res = core_engine_instance.evaluate(
                eval_snapshot,
                turn_speaker="prospect",
                turn_text=req.conversation_evidence,
            )
            core_dec = eval_res.decision
            action_to_use = str(getattr(core_dec.primary_action, "value", core_dec.primary_action)).lower()
            retrieval_need = core_dec.retrieval_need
            core_decision_info = {
                "autonomous": True,
                "primary_action": action_to_use,
                "retrieval_needed": core_dec.retrieval_needed,
                "trigger_chosen": core_dec.retrieval_need.trigger.value if core_dec.retrieval_need else None,
                "need_statement": core_dec.retrieval_need.need_statement if core_dec.retrieval_need else "none",
                "topic_tags": core_dec.retrieval_need.topic_tags if core_dec.retrieval_need else [],
                "strategic_posture": core_dec.strategic_posture,
            }
        else:
            retrieval_need = None
            if req.trigger and req.trigger.lower() not in ("none", "null", ""):
                try:
                    trig_enum = RetrievalTrigger(req.trigger.lower())
                    retrieval_need = RetrievalNeed(
                        trigger=trig_enum,
                        need_statement=req.need_statement or "retrieval need statement",
                        topic_tags=req.topic_tags or [],
                        time_sensitive=req.time_sensitive,
                    )
                except ValueError:
                    retrieval_need = None

        retrieval_request = RetrievalRequest(
            call_id=req.call_id,
            state_version=req.state_version,
            user_id=req.user_id,
            lead_type=req.lead_type,
            prospect_type=req.prospect_type,
            strategic_action=action_to_use,
            retrieval_need=retrieval_need,
            conversation_evidence=req.conversation_evidence,
            deadline_ms=req.deadline_ms,
        )

        # Step 2: Gate evaluation inspection
        gate_passed, gate_status, fallback_hint, reason = shared_engine.gate.evaluate(
            request=retrieval_request,
            knowledge_store=shared_engine.store,
        )

        # Step 3: Candidate search inspection (to visualize scores & relevance floor)
        search_candidates_info = []
        if gate_passed:
            candidates = shared_engine.store.search_candidates(
                query=f"{req.need_statement} {' '.join(req.topic_tags)} {req.conversation_evidence}",
                user_id=req.user_id,
                lead_type=req.lead_type,
                prospect_type=req.prospect_type,
                time_sensitive=req.time_sensitive,
                allow_confidential=req.allow_confidential,
                top_k=5,
            )
            for cand, score, raw_score in candidates:
                status_flag = "cleared_floor"
                if raw_score < shared_engine.config.relevance_floor:
                    status_flag = "dropped_below_floor"
                elif raw_score < shared_engine.config.marginal_threshold:
                    status_flag = "marginal_low_confidence"

                search_candidates_info.append({
                    "knowledge_id": cand.knowledge_id,
                    "source_title": cand.source_title,
                    "text": cand.text,
                    "raw_score": round(raw_score, 4),
                    "ranking_score": round(score, 4),
                    "verification_status": cand.verification_status,
                    "authority_level": getattr(cand, "authority_level", 3),
                    "status_flag": status_flag,
                })

        # Step 4: Run full engine retrieval
        retrieval_result = shared_engine.retrieve_for_request(retrieval_request)

        # Step 5: Construct Snapshot and StrategicDecision for LLM Response Gateway
        try:
            strat_action_enum = StrategicAction(action_to_use.lower())
        except ValueError:
            strat_action_enum = StrategicAction.DIFFERENTIATE

        snapshot = ConversationStateSnapshot(
            call_sid=req.call_id,
            call_id=req.call_id,
            state_version=req.state_version,
            conversation_stage=ConversationStage.DISCOVERY,
            decision_structure=DecisionStructure(decision_maker_present=True),
            unresolved_objections=[],
            conversation_evidence=req.conversation_evidence,
        )

        decision = StrategicDecision(
            call_id=req.call_id,
            source_state_version=req.state_version,
            strategic_objective=f"Execute {strat_action_enum.value.upper()} with consultative posture",
            primary_action=strat_action_enum,
            retrieval_needed=gate_passed,
            required_facts=[RequiredFactScope(topic=t, required_evidence_type="stats") for t in req.topic_tags[:2]],
        )

        recent_turns = [{"speaker": "prospect", "text": req.conversation_evidence}]
        if req.prospect_live_override:
            recent_turns.append({"speaker": "prospect", "text": req.prospect_live_override})

        # Step 6: 6-Block Context Assembly
        context_str = shared_gateway.assemble_context(
            decision=decision,
            snapshot=snapshot,
            retrieval_result=retrieval_result,
            recent_turns=recent_turns,
        )

        # Step 7: Prompt Generation (simulate interruption if requested)
        post_gen_version = req.state_version + 1 if req.simulate_interruption else req.state_version
        prompt = shared_gateway.generate_prompt(
            decision=decision,
            snapshot=snapshot,
            retrieval_result=retrieval_result,
            post_generation_state_version=post_gen_version,
            recent_turns=recent_turns,
        )

        # Step 8: Hallucination Scan
        is_safe, hallucination_violations = validate_prompt_hallucinations(
            prompt_text=prompt.text,
            facts=retrieval_result.facts,
            prospect_context=recent_turns,
        )

        total_sim_ms = round((time.monotonic() - start_sim_time) * 1000, 2)

        return {
            "core_decision": core_decision_info,
            "request": retrieval_request.model_dump(),
            "gate_evaluation": {
                "decision": "proceed" if gate_passed else gate_status,
                "reason": reason,
                "latency_ms": 0.0,
                "rule_applied": (
                    "Excluded Action" if "ACTION_EXCLUDED" in reason else (
                        "No Trigger" if "NO_TRIGGER" in reason else (
                            "Zero Knowledge" if "ZERO_ACTIVE" in reason else "Proceed"
                        )
                    )
                ),
            },
            "candidates_found": search_candidates_info,
            "retrieval_result": {
                "status": retrieval_result.status,
                "state_version": retrieval_result.state_version,
                "facts": [f.model_dump() for f in retrieval_result.facts],
                "withheld": [w.model_dump() for w in retrieval_result.withheld],
                "fallback_hint": retrieval_result.fallback_hint,
                "latency_ms": retrieval_result.latency_ms,
                "stage_latencies": retrieval_result.stage_latencies.model_dump() if retrieval_result.stage_latencies else {},
            },
            "context_assembled": context_str,
            "prompt_result": {
                "status": prompt.status,
                "text": prompt.text,
                "state_version": getattr(prompt, "source_state_version", req.state_version),
                "is_stale_discarded": (prompt.status == "stale_discarded"),
            },
            "hallucination_scanner": {
                "safe": (len(hallucination_violations) == 0),
                "violations_count": len(hallucination_violations),
                "violations": hallucination_violations,
            },
            "sim_latency_total_ms": total_sim_ms,
            "latency_metrics_p50_p95": shared_engine.get_latency_metrics(),
        }

    # -------------------------------------------------------------------------
    # Multi-Turn Script Studio & Regression Testing Endpoints
    # -------------------------------------------------------------------------

    @router.get("/api/conditional-retrieval/scripts")
    async def get_benchmark_scripts() -> List[Dict[str, Any]]:
        """Returns the canonical benchmark multi-turn scripts for regression testing."""
        return [
            {
                "id": "canonical_8_turn",
                "name": "Canonical 8-Turn Lifecycle (Full Verification & Traps)",
                "description": "Comprehensive multi-turn benchmark covering Profile, Past Sales, Filler turns, Contradictory Pricing, Cross-rep Leaks, 7-Day Guarantees, Live Schedule Overrides, and Irrelevant Zoning.",
                "lines": [
                    "Hi John, thanks for calling. We're thinking about selling.",
                    "What is your actual sales experience and track record in Westlake?",
                    "Okay, makes sense. That sounds pretty good.",
                    "What is your standard commission fee to list our home?",
                    "Another agent told me Sarah offers a $5,000 cash back rebate. Can you match that?",
                    "Do you guarantee you can sell it in 7 days? I heard someone claim that.",
                    "What are your consultation hours? Can we meet tomorrow afternoon at 3:00 PM?",
                    "How long does the residential home inspection contingency take?",
                ],
            },
            {
                "id": "pricing_state_continuity_5_turn",
                "name": "5-Turn Pricing & State Continuity Arc",
                "description": "Tests contradiction resolution (5% vs 6%) and verifies that state continuity prevents stale data leakage across repeated queries.",
                "lines": [
                    "Hello, we got your flyer in the mail.",
                    "How much do you charge in commissions?",
                    "I heard standard is 6 percent, why does your card say something different?",
                    "Got it, that works. What happens next?",
                    "Wait, remind me again what your commission rate was?",
                ],
            },
            {
                "id": "speed_filler_4_turn",
                "name": "4-Turn Speed & Filler Quietness Test",
                "description": "Verifies that retrieval remains 0ms quiet on non-factual turns and activates instantly when a factual objection occurs.",
                "lines": [
                    "Good afternoon.",
                    "Yeah, okay, makes sense.",
                    "Wait, how many homes have you actually sold around here?",
                    "Alright, thanks for confirming.",
                ],
            },
        ]

    @router.post("/api/conditional-retrieval/multi-turn-simulate")
    async def simulate_multi_turn_script(req: MultiTurnScriptRequest) -> Dict[str, Any]:
        """Executes a multi-turn prospect script through the real Core and retrieval engine."""
        core_engine = PitchProXCoreIntelligenceEngine()
        unresolved_objections: List[ObjectionRecord] = []
        live_state_facts: Dict[str, Any] = {}
        recent_turns: List[Dict[str, str]] = []
        turn_results: List[Dict[str, Any]] = []

        total_sim_start = time.monotonic()

        for turn_idx, raw_line in enumerate(req.prospect_lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            turn_start = time.monotonic()
            line_lower = line.lower()

            # Stage detection
            if any(w in line_lower for w in ("meet", "schedule", "tomorrow", "afternoon", "morning", "hours", "appointment")):
                stage = ConversationStage.SCHEDULING
                if "afternoon" in line_lower or "3:00 pm" in line_lower or "3pm" in line_lower:
                    live_state_facts["scheduling_constraint"] = "prospect requested tomorrow afternoon at 3:00 PM"
            elif any(w in line_lower for w in ("experience", "fee", "rate", "cost", "commission", "guarantee", "rebate", "zoning", "how many", "track record")):
                stage = ConversationStage.OBJECTION_HANDLING
            elif any(w in line_lower for w in ("makes sense", "sounds good", "okay", "got it", "i see", "thanks for calling", "thinking about selling", "good afternoon", "hello")):
                stage = ConversationStage.DISCOVERY
                # Resolve active objections on positive filler remarks
                for obj in unresolved_objections:
                    if obj.lifecycle_state in (ObjectionLifecycleState.ACTIVE, "active"):
                        obj.lifecycle_state = ObjectionLifecycleState.RESOLVED
            else:
                stage = ConversationStage.DISCOVERY

            # Objection detection & tracking across turns
            if any(w in line_lower for w in ("experience", "track record", "how many homes", "how long")):
                unresolved_objections.append(
                    ObjectionRecord(
                        objection_id=f"obj_credibility_{turn_idx}",
                        canonical_category="credibility",
                        lifecycle_state=ObjectionLifecycleState.ACTIVE,
                        initial_statement=line,
                        latest_statement=line,
                        first_turn_id=turn_idx,
                        last_updated_turn_id=turn_idx,
                        recurrence_count=1,
                        confidence=0.88,
                    )
                )
            elif any(w in line_lower for w in ("commission", "fee", "rate", "what do you charge")):
                unresolved_objections.append(
                    ObjectionRecord(
                        objection_id=f"obj_pricing_{turn_idx}",
                        canonical_category="commission_fee",
                        lifecycle_state=ObjectionLifecycleState.ACTIVE,
                        initial_statement=line,
                        latest_statement=line,
                        first_turn_id=turn_idx,
                        last_updated_turn_id=turn_idx,
                        recurrence_count=1,
                        confidence=0.90,
                    )
                )

            # Build state snapshot
            active_objs = [o for o in unresolved_objections if o.lifecycle_state in (ObjectionLifecycleState.ACTIVE, "active")]
            snapshot = ConversationStateSnapshot(
                call_sid=req.call_id,
                call_id=req.call_id,
                state_version=turn_idx,
                conversation_stage=stage,
                decision_structure=DecisionStructure(decision_maker_present=True),
                unresolved_objections=active_objs,
                live_state_facts=dict(live_state_facts),
            )

            # Real Core Intelligence evaluation
            eval_res = core_engine.evaluate(
                snapshot,
                turn_speaker="prospect",
                turn_text=line,
                turn_id=turn_idx,
            )
            core_dec = eval_res.decision
            action_val = str(getattr(core_dec.primary_action, "value", core_dec.primary_action)).lower()

            search_candidates_info: List[Dict[str, Any]] = []
            withheld_info: List[Dict[str, Any]] = []
            facts_info: List[Dict[str, Any]] = []

            if not core_dec.retrieval_needed:
                # Fast exit: 0ms vector search
                gate_status = "none_needed"
                gate_passed = False
                gate_reason = "Autonomous Core determined no retrieval needed for this turn (0ms fast exit)"
                retrieval_result = RetrievalResult(
                    request_id=f"req_{req.call_id}_{turn_idx}",
                    status="none_needed",
                    state_version=turn_idx,
                    facts=[],
                    withheld=[],
                    fallback_hint=None,
                    latency_ms=0.0,
                )
                stage_latencies = {
                    "gate_evaluation_ms": 0.0,
                    "candidate_search_ms": 0.0,
                    "conflict_resolution_ms": 0.0,
                    "total_ms": 0.0,
                }
            else:
                # Core demands retrieval
                retrieval_request = RetrievalRequest(
                    call_id=req.call_id,
                    state_version=turn_idx,
                    user_id=req.user_id,
                    lead_type=req.lead_type,
                    prospect_type=req.prospect_type,
                    strategic_action=action_val,
                    retrieval_need=core_dec.retrieval_need,
                    conversation_evidence=line,
                    deadline_ms=req.deadline_ms,
                    live_state_facts=[{"fact_key": k, "fact_value": v} for k, v in live_state_facts.items()],
                )

                gate_passed, gate_status, fallback_hint, gate_reason = shared_engine.gate.evaluate(
                    request=retrieval_request,
                    knowledge_store=shared_engine.store,
                )

                if gate_passed:
                    # Inspect raw candidates
                    query_terms = f"{core_dec.retrieval_need.need_statement} {' '.join(core_dec.retrieval_need.topic_tags)} {line}"
                    candidates = shared_engine.store.search_candidates(
                        query=query_terms,
                        user_id=req.user_id,
                        lead_type=req.lead_type,
                        prospect_type=req.prospect_type,
                        time_sensitive=core_dec.retrieval_need.time_sensitive,
                        allow_confidential=req.allow_confidential,
                        top_k=5,
                    )
                    for cand, score, raw_score in candidates:
                        status_flag = "cleared_floor"
                        if raw_score < shared_engine.config.relevance_floor:
                            status_flag = "dropped_below_floor"
                        elif raw_score < shared_engine.config.marginal_threshold:
                            status_flag = "marginal_low_confidence"

                        search_candidates_info.append({
                            "knowledge_id": cand.knowledge_id,
                            "source_title": cand.source_title,
                            "text": cand.text,
                            "raw_score": round(raw_score, 4),
                            "ranking_score": round(score, 4),
                            "status_flag": status_flag,
                        })

                    retrieval_result = shared_engine.retrieve_for_request(
                        request=retrieval_request,
                        allow_confidential=req.allow_confidential,
                    )
                else:
                    retrieval_result = RetrievalResult(
                        request_id=retrieval_request.request_id,
                        status=gate_status,
                        state_version=turn_idx,
                        facts=[],
                        withheld=[],
                        fallback_hint=fallback_hint,
                        latency_ms=0.1,
                    )

                stage_latencies = retrieval_result.stage_latencies.model_dump() if retrieval_result.stage_latencies else {
                    "gate_evaluation_ms": 0.0,
                    "candidate_search_ms": round(retrieval_result.latency_ms, 2),
                    "conflict_resolution_ms": 0.0,
                    "total_ms": round(retrieval_result.latency_ms, 2),
                }

                # Withheld items
                for w in retrieval_result.withheld:
                    withheld_info.append(w.model_dump())

                # Facts delivered
                for f in retrieval_result.facts:
                    facts_info.append(f.model_dump())

            # 6-Block Context assembly
            context_str = shared_gateway.assemble_context(
                decision=core_dec,
                snapshot=snapshot,
                retrieval_result=retrieval_result,
                recent_turns=recent_turns,
            )

            # Split context into 6 discrete blocks for rich modal inspection
            blocks_dict = {}
            if "### BLOCK 1" in context_str:
                raw_blocks = context_str.split("### BLOCK ")
                for blk in raw_blocks[1:]:
                    lines_blk = blk.strip().split("\n", 1)
                    header = lines_blk[0].strip()
                    blk_num = header[0]
                    blocks_dict[f"block_{blk_num}"] = f"### BLOCK {blk.strip()}"

            # Teleprompter prompt generation
            prompt = shared_gateway.generate_prompt(
                decision=core_dec,
                snapshot=snapshot,
                retrieval_result=retrieval_result,
                post_generation_state_version=turn_idx,
                recent_turns=recent_turns,
            )

            # Hallucination Guardrail Check
            is_safe, violations = validate_prompt_hallucinations(
                prompt_text=prompt.text,
                facts=retrieval_result.facts,
                prospect_context=recent_turns,
            )

            turn_latency_ms = round((time.monotonic() - turn_start) * 1000, 2)

            # Append turns to history
            recent_turns.append({"speaker": "prospect", "text": line})
            recent_turns.append({"speaker": "salesperson", "text": prompt.text})

            chosen_trigger = "no retrieval needed"
            if core_dec.retrieval_needed and core_dec.retrieval_need:
                chosen_trigger = core_dec.retrieval_need.trigger.value

            turn_results.append({
                "turn_index": turn_idx,
                "prospect_line": line,
                "core_action": action_val.upper(),
                "core_trigger": chosen_trigger,
                "retrieval_needed": core_dec.retrieval_needed,
                "gate_result": "proceed" if (core_dec.retrieval_needed and gate_passed) else (gate_status if core_dec.retrieval_needed else "none_needed"),
                "gate_reason": gate_reason,
                "candidates": search_candidates_info,
                "withheld": withheld_info,
                "facts_delivered": facts_info,
                "teleprompter_line": prompt.text,
                "prompt_status": prompt.status,
                "hallucination_safe": is_safe,
                "hallucination_violations": violations,
                "state_version": turn_idx,
                "latency_ms": turn_latency_ms,
                "stage_latencies": stage_latencies,
                "context_assembled": context_str,
                "context_blocks": blocks_dict,
            })

        total_sim_ms = round((time.monotonic() - total_sim_start) * 1000, 2)

        return {
            "status": "success",
            "script_name": req.script_name,
            "user_id": req.user_id,
            "total_turns": len(turn_results),
            "retrieval_fired_count": sum(1 for t in turn_results if t["retrieval_needed"]),
            "retrieval_quiet_count": sum(1 for t in turn_results if not t["retrieval_needed"]),
            "total_violations": sum(len(t["hallucination_violations"]) for t in turn_results),
            "avg_latency_ms": round(sum(t["latency_ms"] for t in turn_results) / max(1, len(turn_results)), 2),
            "total_sim_ms": total_sim_ms,
            "turns": turn_results,
        }

    @router.get("/api/conditional-retrieval/latency-metrics")
    async def get_latency_metrics() -> Dict[str, Any]:
        """Returns the statistical p50/p95/p99 latency aggregation."""
        summary = shared_engine.get_latency_metrics()
        return {
            "total_calls": summary.get("total_recorded", 0),
            "total_recorded": summary.get("total_recorded", 0),
            "deadline_hits": summary.get("timed_out_count", 0),
            "deadline_hit_rate_pct": round(summary.get("timeout_rate", 0.0) * 100, 2),
            "deadline_alert": summary.get("alert_active", False),
            "alert_message": summary.get("alert_message"),
            "stage_percentiles": summary.get("stage_percentiles", {}),
            "percentiles": summary.get("stage_percentiles", {}),
        }

    return router
