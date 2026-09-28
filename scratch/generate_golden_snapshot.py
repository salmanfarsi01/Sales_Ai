import json, sys
sys.path.insert(0, ".")
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_manager import ConversationStateManager
from copilot.core_intelligence_engine import PitchProXCoreIntelligenceEngine

with open("reports/synthetic/conversation_state_sim_mucj0p5s.json", encoding="utf-8") as f:
    data = json.load(f)

mgr = ConversationStateManager(call_sid="sim_mucj0p5s")
core = PitchProXCoreIntelligenceEngine()

timeline_records = []
for t in data["timeline"]:
    eb = BehavioralSignalInputBundle(**t["evidence_bundle"])
    snap = mgr.process_turn_bundle(eb)
    eval_res = core.evaluate(snap, turn_speaker=eb.speaker_id, turn_text=eb.utterance_text)
    
    rec = {
        "turn_id": eb.turn_id,
        "speaker_id": eb.speaker_id,
        "utterance_text": eb.utterance_text,
        "evidence_bundle": eb.model_dump(),
        "state_snapshot": snap.model_dump(),
        "strategic_decision": eval_res.decision.model_dump() if eval_res and eval_res.decision else None,
        "strategic_context": eval_res.context.model_dump() if eval_res and eval_res.context else None,
    }
    timeline_records.append(rec)

golden_data = {
    "call_sid": "sim_mucj0p5s",
    "description": "Full-field corrected golden snapshot of 18-turn benchmark conversation",
    "timeline": timeline_records,
    "final_state": mgr.current_state.model_dump(),
}

with open("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json", "w", encoding="utf-8") as f:
    json.dump(golden_data, f, indent=2, default=str)

print("Generated reports/synthetic/conversation_state_sim_mucj0p5s_golden.json successfully!")
