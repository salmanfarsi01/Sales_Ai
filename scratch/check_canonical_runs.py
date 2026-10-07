import sys
sys.path.insert(0, ".")
import json
from copilot.conversation_replay import ConversationReplayEngine

raw = json.loads(open("data/script_canonical.json").read())
eng = ConversationReplayEngine()
rep_base = eng.replay_dialogue_turns("sim_test_canon_base", raw, save_report=False, run_semantic_analysis=False)
rep_sem = eng.replay_dialogue_turns("sim_test_canon_sem", raw, save_report=False, run_semantic_analysis=True)

print("=== CANONICAL COMPARISON: BASELINE vs SEMANTIC HEURISTIC ON ===")
print(f"{'Turn':4s} | {'Speaker':7s} | {'Baseline Action':16s} | {'Base Push':10s} | {'Base Conf':9s} | {'Sem Action':16s} | {'Sem Push':10s} | {'Sem Conf':9s} | {'Base_Src':8s} | {'Mom':5s}")
for b, s in zip(rep_base.timeline, rep_sem.timeline):
    b_dec = b.strategic_decision
    s_dec = s.strategic_decision
    s_breakdown = s_dec.confidence_breakdown
    base_src = s_breakdown.get("base_source", 0.0)
    mom_adj = s_breakdown.get("momentum_adj", 0.0)
    spk = b.evidence_bundle.speaker_id
    print(f"{b.turn_id:4d} | {spk:7s} | {str(b_dec.primary_action.value):16s} | {str(b_dec.push_strength):10s} | {b_dec.confidence:9.3f} | {str(s_dec.primary_action.value):16s} | {str(s_dec.push_strength):10s} | {s_dec.confidence:9.3f} | {base_src:8.3f} | {mom_adj:+5.2f}")
