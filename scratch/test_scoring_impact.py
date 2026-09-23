import sys
sys.path.insert(0, '.')
import json
import re, uuid
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle
from copilot.conversation_state_models import ConversationStage, DecisionStakeholder, MomentumBreakdown
from copilot.conversation_materiality import ABSENT_DECISION_MAKER_PATTERNS
from copilot.conversation_state_manager import PRESENCE_CONFIRMATION_PATTERNS
from copilot import conversation_state_manager, conversation_scoring

with open('reports/synthetic/conversation_state_sim_mucj0p5s.json') as f:
    data = json.load(f)

# Patch _extract_autonomous_decision_updates
def patched_extract(self, bundle, materiality):
    if bundle.speaker_id != "client":
        return None
    text = bundle.utterance_text.lower().strip()
    result = {}

    if any(re.search(pat, text) for pat in PRESENCE_CONFIRMATION_PATTERNS):
        existing = list(self.current_state.decision_structure.stakeholders)
        target_role = "wife" if "wife" in text else ("husband" if "husband" in text else ("spouse" if "spouse" in text else ("partner" if "partner" in text else None)))
        updated = False
        new_stakeholders = []
        for s in existing:
            match_role = (
                (target_role and s.role in (target_role, "spouse", "partner", "co_decision_maker", "female_decision_maker" if target_role == "wife" else "male_decision_maker"))
                or (not target_role and s.is_decision_maker)
            )
            if match_role:
                s_up = s.model_copy(deep=True)
                s_up.presence = "confirmed_attending"
                s_up.notes = f"Confirmed attending by client: '{bundle.utterance_text[:60]}'"
                new_stakeholders.append(s_up)
                updated = True
            else:
                new_stakeholders.append(s.model_copy(deep=True))
        if not updated and target_role:
            new_stakeholders.append(
                DecisionStakeholder(
                    stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
                    role=target_role,
                    is_decision_maker=True,
                    presence="confirmed_attending",
                    notes=f"Confirmed attending by client: '{bundle.utterance_text[:60]}'",
                    confidence=0.90,
                )
            )
        result["stakeholders"] = new_stakeholders
        has_absent = any(s.presence == "absent" for s in new_stakeholders)
        if not has_absent:
            result["decision_maker_present"] = True

    elif any(re.search(pat, text) for pat in ABSENT_DECISION_MAKER_PATTERNS):
        role = "co_decision_maker"
        if "husband" in text:
            role = "husband"
        elif "wife" in text:
            role = "wife"
        elif "partner" in text or "spouse" in text:
            role = "partner"
        elif "attorney" in text or "lawyer" in text:
            role = "attorney"
        elif "he" in text:
            role = "male_decision_maker"
        elif "she" in text:
            role = "female_decision_maker"

        stakeholder = DecisionStakeholder(
            stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
            role=role,
            is_decision_maker=True,
            presence="absent",
            notes=f"Identified as absent decision maker from client disclosure: '{bundle.utterance_text[:60]}'",
            confidence=0.85,
        )
        existing = list(self.current_state.decision_structure.stakeholders)
        if not any(s.role == role and s.presence == "absent" for s in existing):
            existing.append(stakeholder)

        result["decision_maker_present"] = False
        result["stakeholders"] = existing

    elif self.prior_bundle and self.prior_bundle.speaker_id == "salesperson" and (
        any(w in self.prior_bundle.utterance_text.lower() for w in ["him involved", "her involved", "them involved", "husband", "wife", "partner", "spouse", "decision maker", "sign off"])
        and (bundle.agreement_score >= 0.60 or any(re.search(rf"\b{aff}\b", text) for aff in ["yeah", "yes", "definitely", "sure", "absolutely", "correct", "of course"]))
    ):
        prior_text_l = self.prior_bundle.utterance_text.lower()
        role = "male_decision_maker" if "him" in prior_text_l else ("female_decision_maker" if "her" in prior_text_l else "co_decision_maker")
        stakeholder = DecisionStakeholder(
            stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
            role=role,
            is_decision_maker=True,
            presence="absent",
            notes=f"Confirmed stakeholder involvement in response to agent inquiry: '{bundle.utterance_text[:60]}'",
            confidence=0.80,
        )
        existing = list(self.current_state.decision_structure.stakeholders)
        if not any(s.role == role and s.presence == "absent" for s in existing):
            existing.append(stakeholder)
        result["decision_maker_present"] = False
        result["stakeholders"] = existing

    elif any(re.search(p, text) for p in [
        r"\b(?:i'm|i\s+am)\s+the\s+(?:one|sole\s+person)\s+(?:making|who\s+makes)\b",
        r"\bno\s+one\s+else\s+needs?\s+to\s+sign\b",
        r"\bi\s+make\s+the\s+decisions?\s+alone\b",
    ]):
        if not self.current_state.decision_structure.stakeholders:
            result["decision_maker_present"] = True
            result["primary_decision_maker"] = "sole_decision_maker"

    if any(re.search(p, text) for p in [
        r"\b(?:sometime\s+next\s+year|moving\s+next\s+year|look\s+at\s+moving|next\s+year)\b",
    ]):
        result["timeline_horizon"] = "sometime next year"
    if any(re.search(p, text) for p in [
        r"\bnothing\s+urgent\b",
        r"\bnot\s+urgent\b",
        r"\bno\s+rush\b",
    ]):
        result["urgency_level"] = "low"

    if any(re.search(p, text) for p in [
        r"\bmornings?\s+(?:don['’]?t|do\s+not)\s+(?:really\s+)?work\b",
        r"\bnot\s+available\s+in\s+the\s+mornings?\b",
    ]):
        cur_constraints = list(self.current_state.decision_structure.access_constraints)
        if "Mornings unavailable" not in cur_constraints:
            cur_constraints.append("Mornings unavailable")
            result["access_constraints"] = cur_constraints

    return result or None

conversation_state_manager.ConversationStateManager._extract_autonomous_decision_updates = patched_extract

orig_compute_momentum = conversation_scoring.ConversationScoringEngine.compute_momentum

def patched_compute_momentum(self, bundle, current_state):
    cfg = self.config
    dims = current_state.dimensions

    # 1. Problem / Goal Clarity
    spec_factor = bundle.specificity_score * 100.0 if bundle.speaker_id == "client" else 50.0
    has_goal_facts = any(f.category in ("timeline", "financial") for f in current_state.facts if f.status == "active")
    problem_goal_score = round(0.6 * spec_factor + (40.0 if has_goal_facts else 15.0), 1)
    problem_goal_score = min(100.0, max(0.0, problem_goal_score))

    # 2. Value Recognition
    agreement_factor = bundle.agreement_score * 100.0
    value_score = round(0.7 * agreement_factor + 0.3 * (dims.trust * 100.0), 1)
    value_score = min(100.0, max(0.0, value_score))

    # 3. Objection Movement
    if not current_state.objections:
        objection_score = 80.0
    else:
        obj_states = [o.lifecycle_state for o in current_state.objections]
        has_dead_end_supersession = any(
            o.lifecycle_state == "superseded" and o.superseded_by_objection_id == "decision_to_stay"
            for o in current_state.objections
        ) or any(
            f.fact_key == "decision_to_stay" and f.status == "active"
            for f in current_state.facts
        )
        if "boundary" in obj_states or has_dead_end_supersession:
            objection_score = 0.0
        elif all(s in ("resolved", "superseded") for s in obj_states):
            objection_score = 100.0
        elif any(s == "partially_resolved" for s in obj_states):
            objection_score = 70.0
        elif any(s == "clarified" for s in obj_states):
            objection_score = 55.0
        else:
            objection_score = 30.0

    # 4. Trust / Engagement Trend
    trust_pts = dims.trust * 100.0
    eng_pts = dims.engagement * 100.0
    tension_penalty = dims.emotion_tension * 30.0
    trust_eng_score = round(0.5 * trust_pts + 0.5 * eng_pts - tension_penalty, 1)
    trust_eng_score = min(100.0, max(0.0, trust_eng_score))

    # 5. Decision Structure Clarity (Responsive to Presence & Alignment)
    dec = current_state.decision_structure
    has_primary = bool(dec.primary_decision_maker) or dec.decision_maker_present
    has_absent_stakeholder = any(s.presence == "absent" for s in dec.stakeholders)
    if has_primary and dec.decision_maker_present and not has_absent_stakeholder:
        dec_score = 95.0
    elif has_primary and not has_absent_stakeholder:
        dec_score = 75.0
    elif has_primary:
        dec_score = 65.0
    else:
        dec_score = 40.0

    # 6. Future / Operational Behavior (Responsive to Confirmed Meeting / Scheduling)
    future_pts = bundle.future_language_score * 100.0
    has_closing_target = any(f.fact_key in ("target_closing", "move_in_date") for f in current_state.facts if f.status == "active")
    has_confirmed_meeting = (
        bool(current_state.conversion_event and getattr(current_state.conversion_event, "status", "") == "confirmed")
        or any(f.fact_key == "confirmed_meeting_time" and f.status == "active" for f in current_state.facts)
    )
    has_tentative_meeting = any(f.fact_key == "tentative_meeting_time" and f.status == "active" for f in current_state.facts)

    if has_confirmed_meeting:
        base_future = 85.0
    elif has_tentative_meeting or current_state.conversation_stage == ConversationStage.SCHEDULING:
        base_future = 55.0
    elif has_closing_target:
        base_future = 40.0
    else:
        base_future = 15.0

    future_score = round(0.3 * future_pts + 0.7 * base_future, 1)
    future_score = min(100.0, max(0.0, future_score))

    # 7. Commitment Behavior (Responsive to Confirmed Conversion Next Step)
    pacing_pts = dims.pacing * 100.0
    raw_commit = round(0.6 * agreement_factor + 0.4 * pacing_pts, 1)
    if has_confirmed_meeting or current_state.conversation_stage == ConversationStage.COMMITMENT_CONFIRMED:
        commit_score = round(max(raw_commit, 85.0 + 0.15 * agreement_factor), 1)
    else:
        commit_score = raw_commit
    commit_score = min(100.0, max(0.0, commit_score))

    # Composite Weighted Momentum Score
    composite_momentum = (
        problem_goal_score * cfg.problem_goal_clarity_weight
        + value_score * cfg.value_recognition_weight
        + objection_score * cfg.objection_movement_weight
        + trust_eng_score * cfg.trust_engagement_trend_weight
        + dec_score * cfg.decision_structure_clarity_weight
        + future_score * cfg.future_operational_behavior_weight
        + commit_score * cfg.commitment_behavior_weight
    )
    composite_momentum = round(min(100.0, max(0.0, composite_momentum)), 1)

    trend_delta = 0.0
    if self._momentum_history:
        prev_momentum = self._momentum_history[-1]
        trend_delta = round(composite_momentum - prev_momentum, 1)

    if trend_delta >= cfg.trend_advancing_delta:
        trend = "advancing"
    elif trend_delta <= cfg.trend_regressing_delta:
        trend = "regressing"
    elif objection_score <= 35.0:
        trend = "stalling"
    else:
        trend = "stable"

    self._momentum_history.append(composite_momentum)

    families = {
        "problem_goal_clarity": problem_goal_score,
        "value_recognition": value_score,
        "objection_movement": objection_score,
        "trust_engagement_trend": trust_eng_score,
        "decision_structure_clarity": dec_score,
        "future_operational_behavior": future_score,
        "commitment_behavior": commit_score,
    }

    effective_conf = round(min(bundle.inference_confidence, bundle.semantic_confidence), 3)

    return MomentumBreakdown(
        momentum_score=composite_momentum,
        trend=trend,
        trend_delta=trend_delta,
        family_scores=families,
        confidence=effective_conf,
    )

conversation_scoring.ConversationScoringEngine.compute_momentum = patched_compute_momentum

mgr = ConversationStateManager(call_sid='sim_mucj0p5s')
print(f"{'Turn':4s} | {'Stage':20s} | {'dec_clarity':11s} | {'future_op':9s} | {'commit':6s} | {'Mom':5s} | {'Delta':6s} | {'Trend':10s}")
print("-" * 85)

for t in data['timeline']:
    eb = BehavioralSignalInputBundle(**t['evidence_bundle'])
    snap = mgr.process_turn_bundle(eb)
    tid = t['turn_id']
    mom = snap.momentum
    fs = mom.family_scores
    stg = snap.conversation_stage.value if snap.conversation_stage else "None"
    print(f"T{tid:02d}  | {stg:20s} | {fs.get('decision_structure_clarity'):11.1f} | {fs.get('future_operational_behavior'):9.1f} | {fs.get('commitment_behavior'):6.1f} | {mom.momentum_score:5.1f} | {mom.trend_delta:+6.1f} | {mom.trend:10s}")
