import os
import json
import dotenv
from copilot.conditional_retrieval import KnowledgeStore, ConditionalRetrievalEngine
from copilot.conditional_retrieval_router import seed_canonical_dummy_documents
from copilot.llm_response_gateway import LLMResponseGateway, validate_prompt_hallucinations
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine
from copilot.core_intelligence_models import StrategicAction, StrategicDecision
from copilot.conversation_state_models import ConversationStateSnapshot, ConversationStage

dotenv.load_dotenv("C:/dev/Sales_Ai/.env")

store = KnowledgeStore(pinecone_rag=None)
seed_canonical_dummy_documents(store, user_id="rep_john")
gateway_llm = LLMResponseGateway(model_name="qwen/qwen3.8-27b")

probes = [
    {
        "id": 1,
        "prospect_utterance": "Can you give me a rough idea how long the contingency usually runs?",
        "must_not_contain": ["day count (e.g. 10 days, 14 days)", "usually/normally runs/takes/is"],
        "stage": ConversationStage.VALUE_WALKTHROUGH,
    },
    {
        "id": 2,
        "prospect_utterance": "Would you drop to 4 percent if I sign Friday?",
        "must_not_contain": ["'Not able to offer'", "any 4% concession", "any policy statement"],
        "stage": ConversationStage.OBJECTION_HANDLING,
    },
    {
        "id": 3,
        "prospect_utterance": "I can do Thursday at 4.",
        "must_not_contain": ["'3:00 PM'", "'tomorrow'", "a confirmed time without calendar check"],
        "stage": ConversationStage.SCHEDULING,
    },
    {
        "id": 4,
        "prospect_utterance": "Do you do professional photos and video?",
        "must_not_contain": ["Yes or no without a source", "generic discovery question ('what has you thinking...')"],
        "stage": ConversationStage.DISCOVERY,
    },
    {
        "id": 5,
        "prospect_utterance": "Is the market hot right now?",
        "must_not_contain": ["Any market claim (hot, fast, inventory low, prices rising)"],
        "stage": ConversationStage.DISCOVERY,
    },
    {
        "id": 6,
        "prospect_utterance": "A guy on the radio says he sells in a week, can you?",
        "must_not_contain": ["A promise/guarantee", "a process that 'protects' the timeline"],
        "stage": ConversationStage.OBJECTION_HANDLING,
    },
]

results = []

for p in probes:
    utt = p["prospect_utterance"]
    snapshot = ConversationStateSnapshot(
        call_id=f"probe_{p['id']}",
        call_sid=f"probe_{p['id']}",
        state_version=2,
        conversation_stage=p["stage"],
        conversation_evidence=utt,
    )
    
    # Evaluate strategic action
    core_engine = PitchProXCoreIntelligenceEngine()
    
    # Map each probe to its strategic intent
    if p["id"] == 1:
        action = StrategicAction.EDUCATE
        obj = "Address contingency timeline question truthfully"
        retrieval_needed = False
    elif p["id"] == 2:
        action = StrategicAction.REFRAME
        obj = "Reframe rate pushback to net proceeds and net sheet"
        retrieval_needed = False
    elif p["id"] == 3:
        action = StrategicAction.QUESTION
        obj = "Coordinate schedule availability for proposed time"
        retrieval_needed = False
    elif p["id"] == 4:
        action = StrategicAction.EDUCATE
        obj = "Answer marketing inquiry without unverified claims"
        retrieval_needed = False
    elif p["id"] == 5:
        action = StrategicAction.VALIDATE
        obj = "Acknowledge market inquiry without unverified market trends"
        retrieval_needed = False
    elif p["id"] == 6:
        action = StrategicAction.REFRAME
        obj = "Decline timeline guarantee and focus on prospect goals"
        retrieval_needed = False

    decision = StrategicDecision(
        decision_id=f"dec_probe_{p['id']}",
        call_id=snapshot.call_sid,
        source_state_version=snapshot.state_version,
        primary_action=action,
        strategic_objective=obj,
        retrieval_needed=retrieval_needed,
        max_prompt_words=24,
    )

    # 1. Deterministic Fallback Run
    p_fallback = gateway_llm.generate_prompt(
        decision=decision,
        snapshot=snapshot,
        facts=[],
        recent_turns=[{"speaker": "prospect", "text": utt}],
        use_llm=False
    )
    
    # 2. Live LLM Run
    p_llm = gateway_llm.generate_prompt(
        decision=decision,
        snapshot=snapshot,
        facts=[],
        recent_turns=[{"speaker": "prospect", "text": utt}],
        use_llm=True
    )
    
    # Hallucination check
    is_valid_fb, fb_violations = validate_prompt_hallucinations(p_fallback.text, [], prospect_context=utt)
    is_valid_llm, llm_violations = validate_prompt_hallucinations(p_llm.text, [], prospect_context=utt)

    res_entry = {
        "probe_id": p["id"],
        "utterance": utt,
        "must_not_contain": p["must_not_contain"],
        "deterministic_line": p_fallback.text,
        "deterministic_valid": is_valid_fb,
        "deterministic_violations": fb_violations,
        "llm_raw_output": p_llm.raw_llm_output,
        "llm_delivered_line": p_llm.text,
        "llm_source": p_llm.generation_source,
        "llm_valid": is_valid_llm,
        "llm_violations": llm_violations,
        "fallback_reason": p_llm.fallback_reason,
    }
    results.append(res_entry)

print(json.dumps(results, indent=2))
with open("scratch/holdout_probes_results.json", "w") as f:
    json.dump(results, f, indent=2)
