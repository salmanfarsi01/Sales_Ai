import sys
sys.path.insert(0, ".")
import json
from copilot.conversation_objections import classify_objection_label, is_decision_authority_statement
from copilot.conversation_objection_driver import ObjectionDriverClassifier
from copilot.behavioral_semantic import extract_structured_contact_preferences
from copilot.conversation_materiality import MaterialityFilter, is_positive_filler
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

filter_engine = MaterialityFilter()
driver_clf = ObjectionDriverClassifier()

print("=== 10 TRUE HOLDOUT PHRASES EVALUATION ===")
for idx, text, expected_type in holdout_phrases:
    is_filler = is_positive_filler(text)
    obj_cat = classify_objection_label(text)
    is_auth = is_decision_authority_statement(text)
    prefs = extract_structured_contact_preferences(text)
    
    inf = DownstreamInferenceState(
        call_sid="t",
        timestamp_ms=1000,
        trust=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_20_30s"),
        emotion=EmotionState(expressed_valence=0, tension_level=0, confidence=0.8),
        pacing=DimensionScore(score=0.6, confidence=0.8, primary_horizon="current_utterance"),
        engagement=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_20_30s"),
        momentum=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        readiness=DimensionScore(score=0.5, confidence=0.8, primary_horizon="last_60_90s"),
        overall_confidence=0.8,
        contributing_evidence_ids=[]
    )
    snap = SemanticFeatureSnapshot(
        utterance_id="u",
        call_sid="t",
        speaker_id="client",
        boundary_score=0.0,
        specificity_score=0.6,
        agreement_score=0.5
    )
    bundle = extract_behavioral_bundle(
        turn_id=idx,
        speaker_id="client",
        utterance_text=text,
        inference_state=inf,
        semantic_snapshot=snap
    )
    mat = filter_engine.classify_turn(bundle)
    
    print(f"[{idx}] \"{text}\"")
    print(f"    Expected: {expected_type}")
    print(f"    Material: {mat.is_material} (targets={mat.affected_targets})")
    print(f"    Objection: {obj_cat}")
    print(f"    Authority: {is_auth}")
    print(f"    Prefs: {[(p.channel, p.allowed, p.time_restriction, p.time_not_before, p.time_not_after) for p in prefs]}")
