import sys
sys.path.insert(0, ".")
from copilot.conversation_state_manager import ConversationStateManager
from copilot.core_decision_manager import CoreDecisionManager
from copilot.conversation_state_contract import extract_behavioral_bundle
from copilot.behavioral_inference import DownstreamInferenceState, DimensionScore, EmotionState
from copilot.behavioral_semantic import SemanticFeatureSnapshot

holdout_phrases = [
    (1, "My financial advisor handles all property sales for the trust.", "authority/stakeholder"),
    (2, "The last brokerage took our listing fee and did zero showings.", "trust/prior_loss"),
    (3, "We won't consider any offers below the recent tax assessment.", "price/valuation"),
    (4, "Texting during the workday is fine, but never call while I'm at the hospital.", "contact/channel_time"),
    (5, "Can you email the breakdown to my accountant first?", "contact/stakeholder_channel"),
    (6, "We had an agent promise a cash buyer that didn't even exist.", "trust/false_promise"),
    (7, "I'm only free on weekends between 10am and 2pm.", "contact/time_window"),
    (8, "I'm having coffee right now, it's pretty cold outside.", "filler/non_material"),
    (9, "My sister co-owns the deed and she lives in Spain.", "authority/co_owner"),
    (10, "No contact on Sundays under any circumstances.", "contact/hard_boundary"),
]

print("=== 10 PHRASES THROUGH FULL STATE MANAGER & CORE DECISION ===")
for idx, text, expected_type in holdout_phrases:
    mgr = ConversationStateManager(call_sid=f"test_phrase_{idx}")
    dm = CoreDecisionManager(call_sid=f"test_phrase_{idx}")
    
    from tests.test_novel_script_and_paraphrase_generalization import _make_bundle
    bundle = _make_bundle(turn_id=idx, text=text)
    res = mgr.process_turn_bundle(bundle)
    
    dec = dm.evaluate_state(
        snapshot=mgr.current_state,
        turn_speaker="client",
        turn_text=text,
        turn_timestamp_ms=1000,
    )
    
    print(f"Phrase {idx}: \"{text}\"")
    print(f"  Expected: {expected_type}")
    print(f"  boundary_suspected: {mgr.current_state.contact_compliance.boundary_suspected} (reason: {mgr.current_state.contact_compliance.boundary_suspected_reason})")
    print(f"  unclassified_material: {mgr.current_state.unclassified_material}")
    d = dec.decision
    print(f"  Action: {d.primary_action} | Posture: {d.strategic_posture} | Push: {d.push_strength}")
    print(f"  Reason codes: {d.reason_codes}")
    print(f"  Gateway prompt: \"{d.gateway_fallback_stub}\"")
    print()
