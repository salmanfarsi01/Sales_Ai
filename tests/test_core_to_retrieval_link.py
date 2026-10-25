"""Tests proving autonomous Core Intelligence decision and trigger selection linked to Conditional Retrieval.

Verifies that the Core Intelligence Engine (Doc 01 §3, Doc 02 §6):
1. Autonomously inspects prospect utterances, conversation stages, and objections.
2. Chooses the primary strategic action (QUANTIFY, DIFFERENTIATE, EDUCATE, SOCIAL_PROOF, etc.).
3. Autonomously determines whether retrieval is needed (retrieval_needed=True/False).
4. Autonomously picks the canonical RetrievalTrigger (credibility_objection, policy_wording,
   service_process_objection, playbook_evidence, prior_claim_reference, factual_question).
5. Passes the decision to ConditionalRetrievalEngine.retrieve_for_decision() with zero manual
   trigger injection, correctly retrieving verified subscriber facts or fast-exiting at 0ms.
"""

from __future__ import annotations

import pytest

from copilot.conditional_retrieval import ConditionalRetrievalEngine, KnowledgeStore
from copilot.conditional_retrieval_models import (
    KnowledgeClass,
    KnowledgeRecord,
    RetrievalTrigger,
)
from copilot.conditional_retrieval_router import seed_canonical_dummy_documents
from copilot.conversation_state_models import (
    ConversationStage,
    ConversationStateSnapshot,
    DecisionStructure,
    ObjectionLifecycleState,
    ObjectionRecord,
)
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_intelligence_models import StrategicAction


@pytest.fixture
def core_engine() -> PitchProXCoreIntelligenceEngine:
    return PitchProXCoreIntelligenceEngine()


@pytest.fixture
def retrieval_engine() -> ConditionalRetrievalEngine:
    store = KnowledgeStore()
    seed_canonical_dummy_documents(store, user_id="rep_john")
    return ConditionalRetrievalEngine(store=store, default_user_id="rep_john")


class TestCoreToRetrievalAutonomousLink:
    """Verifies that Core Intelligence autonomously determines retrieval needs and triggers."""

    def test_core_autonomously_picks_credibility_objection_and_retrieves_fact(
        self,
        core_engine: PitchProXCoreIntelligenceEngine,
        retrieval_engine: ConditionalRetrievalEngine,
    ):
        """Scenario: Prospect challenges sales track record in Westlake.
        Core must autonomously select CREDIBILITY_OBJECTION, and retrieval must fetch 42 homes.
        Zero manual trigger supplied!
        """
        snapshot = ConversationStateSnapshot(
            call_sid="call_westlake_test",
            call_id="call_westlake_test",
            state_version=1,
            conversation_stage=ConversationStage.OBJECTION_HANDLING,
            decision_structure=DecisionStructure(decision_maker_present=True),
            unresolved_objections=[
                ObjectionRecord(
                    objection_id="obj_cred_1",
                    canonical_category="credibility_experience",
                    initial_statement="Do you have any actual sales experience in Westlake?",
                    latest_statement="Do you have any actual sales experience in Westlake?",
                    first_turn_id=1,
                    last_updated_turn_id=1,
                    lifecycle_state=ObjectionLifecycleState.ACTIVE,
                    raw_transcript_excerpt="Do you have any actual sales experience in Westlake?",
                )
            ],
        )

        turn_text = "Do you have any actual sales experience in Westlake?"
        eval_res = core_engine.evaluate(snapshot, turn_speaker="prospect", turn_text=turn_text)
        decision = eval_res.decision

        # 1. Verify Core Autonomous Decision
        assert decision.should_prompt is True
        assert decision.primary_action in (StrategicAction.QUANTIFY, StrategicAction.DIFFERENTIATE, StrategicAction.EDUCATE, StrategicAction.QUESTION)
        assert decision.retrieval_needed is True
        assert decision.retrieval_need is not None
        assert decision.retrieval_need.trigger == RetrievalTrigger.CREDIBILITY_OBJECTION
        assert "westlake" in decision.retrieval_need.topic_tags or "sales" in decision.retrieval_need.topic_tags
        assert len(decision.required_facts) > 0

        # 2. Feed decision directly to ConditionalRetrievalEngine (no hand-set trigger!)
        report = retrieval_engine.retrieve_for_decision(decision, user_id="rep_john")

        # 3. Verify retrieval results
        assert report.status == "retrieved"
        assert len(report.facts) >= 1
        fact_text = report.facts[0].text
        assert "42 homes in Westlake" in fact_text
        assert report.facts[0].topic == KnowledgeClass.PERFORMANCE_PAST_SALES.value

    def test_core_autonomously_picks_policy_wording_for_commission_query(
        self,
        core_engine: PitchProXCoreIntelligenceEngine,
        retrieval_engine: ConditionalRetrievalEngine,
    ):
        """Scenario: Prospect asks about commission rate.
        Core must autonomously select POLICY_WORDING, retrieving the 5.0% commission chunk.
        """
        snapshot = ConversationStateSnapshot(
            call_sid="call_comm_test",
            call_id="call_comm_test",
            state_version=2,
            conversation_stage=ConversationStage.DISCOVERY,
            decision_structure=DecisionStructure(decision_maker_present=True),
            unresolved_objections=[],
        )

        turn_text = "What commission rate do you charge to list my home?"
        eval_res = core_engine.evaluate(snapshot, turn_speaker="prospect", turn_text=turn_text)
        decision = eval_res.decision

        assert decision.retrieval_needed is True
        assert decision.retrieval_need is not None
        assert decision.retrieval_need.trigger == RetrievalTrigger.POLICY_WORDING
        assert "commission" in decision.retrieval_need.topic_tags

        # Execute retrieval via Core decision
        report = retrieval_engine.retrieve_for_decision(decision, user_id="rep_john")
        assert report.status == "retrieved"
        assert any("5.0%" in f.text for f in report.facts)
        # Verify older 6.0% policy was withheld (in disputed_topics)
        assert "chunk_commission_v1" in report.disputed_topics

    def test_core_autonomously_picks_service_process_objection(
        self,
        core_engine: PitchProXCoreIntelligenceEngine,
        retrieval_engine: ConditionalRetrievalEngine,
    ):
        """Scenario: Prospect asks about home inspection contingencies.
        Core autonomously sets trigger to SERVICE_PROCESS_OBJECTION.
        """
        snapshot = ConversationStateSnapshot(
            call_sid="call_process_test",
            call_id="call_process_test",
            state_version=1,
            conversation_stage=ConversationStage.DISCOVERY,
            decision_structure=DecisionStructure(decision_maker_present=True),
            unresolved_objections=[],
        )

        turn_text = "How long does the residential home inspection contingency take?"
        eval_res = core_engine.evaluate(snapshot, turn_speaker="prospect", turn_text=turn_text)
        decision = eval_res.decision

        assert decision.retrieval_needed is True
        assert decision.retrieval_need is not None
        assert decision.retrieval_need.trigger == RetrievalTrigger.SERVICE_PROCESS_OBJECTION
        assert "inspection" in decision.retrieval_need.topic_tags or "contingency" in decision.retrieval_need.topic_tags

    def test_core_autonomously_picks_playbook_evidence_for_testimonials(
        self,
        core_engine: PitchProXCoreIntelligenceEngine,
        retrieval_engine: ConditionalRetrievalEngine,
    ):
        """Scenario: Prospect asks for seller client testimonials.
        Core autonomously sets trigger to PLAYBOOK_EVIDENCE.
        """
        snapshot = ConversationStateSnapshot(
            call_sid="call_social_test",
            call_id="call_social_test",
            state_version=1,
            conversation_stage=ConversationStage.DISCOVERY,
            decision_structure=DecisionStructure(decision_maker_present=True),
            unresolved_objections=[],
        )

        turn_text = "Do you have testimonials from other clients who sold luxury homes?"
        eval_res = core_engine.evaluate(snapshot, turn_speaker="prospect", turn_text=turn_text)
        decision = eval_res.decision

        assert decision.retrieval_needed is True
        assert decision.retrieval_need is not None
        assert decision.retrieval_need.trigger == RetrievalTrigger.PLAYBOOK_EVIDENCE

    def test_core_autonomously_skips_retrieval_on_filler_or_acknowledgment(
        self,
        core_engine: PitchProXCoreIntelligenceEngine,
        retrieval_engine: ConditionalRetrievalEngine,
    ):
        """Scenario: Prospect gives filler acknowledgment ('Yeah okay, that makes sense').
        Core must decide retrieval_needed=False, and retrieval must fast-exit in 0ms with none_needed.
        """
        snapshot = ConversationStateSnapshot(
            call_sid="call_filler_test",
            call_id="call_filler_test",
            state_version=3,
            conversation_stage=ConversationStage.DISCOVERY,
            decision_structure=DecisionStructure(decision_maker_present=True),
            unresolved_objections=[],
        )

        turn_text = "Yeah, okay, that makes sense. I see what you mean."
        eval_res = core_engine.evaluate(snapshot, turn_speaker="prospect", turn_text=turn_text)
        decision = eval_res.decision

        # 1. Core decides no retrieval is needed
        assert decision.retrieval_needed is False
        assert decision.retrieval_need is None
        assert len(decision.required_facts) == 0

        # 2. Retrieval fast-exits immediately
        report = retrieval_engine.retrieve_for_decision(decision, user_id="rep_john")
        assert report.status == "none_needed"
        assert report.retrieval_needed is False
        assert len(report.facts) == 0
        assert report.fallback_hint in (None, "use_generic_truthful")

    def test_core_autonomously_skips_retrieval_on_hard_boundary(
        self,
        core_engine: PitchProXCoreIntelligenceEngine,
        retrieval_engine: ConditionalRetrievalEngine,
    ):
        """Scenario: Prospect sets a hard boundary ('Take me off your list and do not call again').
        Core sets hard boundary exit with retrieval_needed=False.
        """
        snapshot = ConversationStateSnapshot(
            call_sid="call_boundary_test",
            call_id="call_boundary_test",
            state_version=1,
            conversation_stage=ConversationStage.DISCOVERY,
            decision_structure=DecisionStructure(decision_maker_present=True),
            unresolved_objections=[],
        )
        snapshot.contact_compliance.hard_boundary_active = True

        turn_text = "Do not call this number ever again. Remove me."
        eval_res = core_engine.evaluate(snapshot, turn_speaker="prospect", turn_text=turn_text)
        decision = eval_res.decision

        assert decision.retrieval_needed is False
        report = retrieval_engine.retrieve_for_decision(decision, user_id="rep_john")
        assert report.status == "none_needed"
        assert len(report.facts) == 0
