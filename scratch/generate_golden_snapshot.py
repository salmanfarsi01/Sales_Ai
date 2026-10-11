import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from copilot.conversation_replay import ConversationReplayEngine

raw_turns = json.loads(Path("data/script_canonical.json").read_text(encoding="utf-8"))
engine = ConversationReplayEngine()
report = engine.replay_dialogue_turns(call_sid="sim_mucj0p5s", raw_turns=raw_turns, save_report=True, run_semantic_analysis=False)

timeline_records = []
for step in report.timeline:
    rec = {
        "turn_id": step.turn_id,
        "speaker_id": step.evidence_bundle.speaker_id,
        "utterance_text": step.evidence_bundle.utterance_text,
        "evidence_bundle": step.evidence_bundle.model_dump(),
        "state_snapshot": step.state_after.model_dump(),
        "strategic_decision": step.strategic_decision.model_dump() if step.strategic_decision else None,
    }
    timeline_records.append(rec)

golden_data = {
    "call_sid": "sim_mucj0p5s",
    "description": "Full-field corrected golden snapshot of 18-turn benchmark conversation",
    "timeline": timeline_records,
    "final_state": report.final_state.model_dump(),
}

with open("reports/synthetic/conversation_state_sim_mucj0p5s_golden.json", "w", encoding="utf-8") as f:
    json.dump(golden_data, f, indent=2, default=str)

print("Generated reports/synthetic/conversation_state_sim_mucj0p5s_golden.json successfully!")
