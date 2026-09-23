import sys
sys.path.insert(0, '.')
import json
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_replay import ConversationReplayEngine

# 1. Pipeline A: Trace Bundles
with open('reports/synthetic/conversation_state_sim_mucj0p5s.json') as f:
    data = json.load(f)

mgr_a = ConversationStateManager(call_sid='sim_mucj0p5s_pipeline_a')
print("=== PIPELINE A (Trace Bundles from sim_mucj0p5s.json) ===")
headers = ["Turn", "Stage", "prob_goal", "val_rec", "obj_mov", "trust_eng", "dec_clar", "fut_op", "commit", "Mom", "Delta", "Trend"]
print(f"{headers[0]:4s} | {headers[1]:20s} | {headers[2]:9s} | {headers[3]:7s} | {headers[4]:7s} | {headers[5]:9s} | {headers[6]:8s} | {headers[7]:6s} | {headers[8]:6s} | {headers[9]:5s} | {headers[10]:6s} | {headers[11]:10s}")
print("-" * 125)

for t in data['timeline']:
    eb = BehavioralSignalInputBundle(**t['evidence_bundle'])
    snap = mgr_a.process_turn_bundle(eb)
    tid = t['turn_id']
    mom = snap.momentum
    fs = mom.family_scores
    stg = snap.conversation_stage.value if snap.conversation_stage else "None"
    print(f"T{tid:02d}  | {stg:20s} | {fs['problem_goal_clarity']:9.1f} | {fs['value_recognition']:7.1f} | {fs['objection_movement']:7.1f} | {fs['trust_engagement_trend']:9.1f} | {fs['decision_structure_clarity']:8.1f} | {fs['future_operational_behavior']:6.1f} | {fs['commitment_behavior']:6.1f} | {mom.momentum_score:5.1f} | {mom.trend_delta:+6.1f} | {mom.trend:10s}")

# 2. Pipeline B: Raw Dialogue Replay
print("\n=== PIPELINE B (Raw Dialogue Replay via ConversationReplayEngine) ===")
dialogue = [
    {"turn_id": t["turn_id"], "speaker_id": t["speaker_id"], "text": t["text"]}
    for t in data["timeline"]
]
replay_engine = ConversationReplayEngine()
rep_result = replay_engine.replay_dialogue_turns(
    call_sid="sim_mucj0p5s_pipeline_b",
    raw_turns=dialogue,
    save_report=False,
)

print(f"{headers[0]:4s} | {headers[1]:20s} | {headers[2]:9s} | {headers[3]:7s} | {headers[4]:7s} | {headers[5]:9s} | {headers[6]:8s} | {headers[7]:6s} | {headers[8]:6s} | {headers[9]:5s} | {headers[10]:6s} | {headers[11]:10s}")
print("-" * 125)

for step in rep_result.timeline:
    tid = step.turn_id
    sa = step.state_after
    if sa and sa.momentum:
        mom = sa.momentum
        fs = mom.family_scores
        stg = sa.conversation_stage.value if sa.conversation_stage else "None"
        print(f"T{tid:02d}  | {stg:20s} | {fs['problem_goal_clarity']:9.1f} | {fs['value_recognition']:7.1f} | {fs['objection_movement']:7.1f} | {fs['trust_engagement_trend']:9.1f} | {fs['decision_structure_clarity']:8.1f} | {fs['future_operational_behavior']:6.1f} | {fs['commitment_behavior']:6.1f} | {mom.momentum_score:5.1f} | {mom.trend_delta:+6.1f} | {mom.trend:10s}")
    else:
        stg = sa.conversation_stage.value if sa and sa.conversation_stage else "None"
        print(f"T{tid:02d}  | {stg:20s} |       --- |     --- |     --- |       --- |      --- |    --- |    --- |   --- |    --- | non-material")
