# ConversationState — Implementation Plan
**PitchProX | Consumes Behavioral Signal Engine Output | Agile-Ready**

---

## 0. Scope Statement

**Purpose:** ConversationState is the middle layer between the Behavioral Signal Engine (what just happened) and the Strategic Decision Engine (what to do next). It answers one question at every point in the call: *"Given everything so far, what is currently true about this conversation?"*

**Upstream dependency:** Every phase below consumes the already-built `DownstreamInferenceState`, `SemanticFeatureSnapshot`, `MultiWindowEvidenceFrame`, and `ContactPreferenceRecord` objects from the Behavioral Signal Engine. Nothing here re-derives raw signals — this layer only interprets and maintains truth from what's already measured.

**Two capabilities flagged as genuinely new, not reused from prior work:**
1. **Truth supersession** (Phase 4) — detecting when a new statement updates or reverses an earlier one, even without contradictory wording.
2. **Materiality filtering** (Phase 5) — deciding which state fields a given turn should actually touch, instead of recomputing everything on every turn.
Both are real engineering work, not small extensions — scope and test them accordingly.

---

## 0.5 Ownership Boundary — AI Engine vs. Backend

Same split established for the Behavioral Signal Engine.

| Owned by AI Engine developer | Owned by Backend developer |
|---|---|
| State schema, field definitions, versioning logic | Persistent storage/database for ConversationState history |
| Objection lifecycle logic, momentum/readiness scoring | Cross-call persistence infrastructure (if state needs to survive beyond one call) |
| Materiality filter, truth-supersession logic | Deployment, scaling, access control |
| The explainability/audit-trail data structure | Wiring ConversationState into the Strategic Decision Engine's consumption pattern |
| The inspection/replay API contract | The actual replay UI hosting, if separate from the AI engine's test console |

---

## 1. Architecture at a Glance

```
Behavioral Signal Engine (per turn)
        ↓
Phase 1: State Schema (current truth object)
        ↓
Phase 2: Persistent Facts Store (never expires until superseded)
        ↓
Phase 3: Objection Lifecycle Engine
        ↓
Phase 4: Truth Supersession Detector  [NEW CAPABILITY]
        ↓
Phase 5: Materiality Filter (what actually needs updating)  [NEW CAPABILITY]
        ↓
Phase 6: Momentum & Readiness Scoring
        ↓
Phase 7: Meeting/Conversion Gate & Push Strength
        ↓
Phase 8: Explainability & Versioning (wraps every phase above)
        ↓
Phase 9: Inspection/Replay Console
        ↓
 [ Strategic Decision Engine — next component, out of scope ]
```

---

## 2. Phase-by-Phase Plan

### PHASE 0 — Input Contract Validation
**Objective:** Confirm exactly which Behavioral Signal Engine fields ConversationState will consume, before building against assumptions.

| | |
|---|---|
| Inputs to verify | `DownstreamInferenceState` (all 6 dimensions + confidence), `SemanticFeatureSnapshot` (recurrence, specificity, future language, boundary, contact preference), `MultiWindowEvidenceFrame` |
| Output | A short written contract: exact field names/types ConversationState will read, confirmed against the real schema, not memory |
| Definition of Done | Every field ConversationState needs has a confirmed source and type |
| Sprint | Sprint 0 |

---

### PHASE 1 — Core State Schema
**Objective:** Define the actual shape of "current truth."

| Field group | Contents |
|---|---|
| Decision structure | Decision-makers, roles, timing constraints, access constraints |
| Objection registry | List of tracked objections, each with lifecycle state (see Phase 3) |
| Persistent facts | Long-lived facts that don't expire (Phase 2) |
| Dimension scores | Trust, Emotion, Engagement, Momentum, Readiness — kept independent, never auto-coupled |
| Contact/compliance state | Hard boundary flag, soft contact preferences (inherited directly from Behavioral Signal Engine) |
| Metadata | `state_version`, `last_updated_turn_id`, per-field confidence |

| | |
|---|---|
| Tools needed | Pydantic models, same discipline as `DownstreamInferenceState` |
| Definition of Done | Schema reviewed against all 11 of the client's requirements — each requirement maps to a concrete field or mechanism, not left implicit |
| Sprint | Sprint 1 |

---

### PHASE 2 — Persistent Facts Store
**Objective:** Facts that stay true until explicitly superseded, regardless of what the conversation moves on to.

| | |
|---|---|
| Examples | "Spouse is a decision-maker," "can't meet mornings," "prefers email," "wants reduced contact frequency" |
| How it works | Each fact stored with: value, source turn ID, timestamp, and a status (`active` / `superseded`). Superseding a fact keeps the old one in history, doesn't delete it |
| Reuses | `ContactPreferenceRecord`'s persistence pattern from the Behavioral Signal Engine — same shape, extended to more fact types |
| Definition of Done | A fact stated early in a call is still retrievable and correctly marked active 20+ turns later, untouched by unrelated conversation in between |
| Sprint | Sprint 1-2 |

---

### PHASE 3 — Objection Lifecycle Engine
**Objective:** Track each objection through its full life story, not as isolated repeated flags.

| Lifecycle state | Definition (per spec) |
|---|---|
| Unresolved | Concern active; repeats or behavior doesn't move |
| Clarified | Meaning understood, not yet solved |
| Partially Resolved | Accepts some reframe, same issue remains |
| Resolved | Stops recurring, behavior advances |
| Reactivated | Previously resolved concern returns |
| Boundary | Stop/representation/legal/privacy — not a persuasion target |

| | |
|---|---|
| Reuses | Semantic recurrence detection from Behavioral Signal Engine — same matching logic, now attached to a persistent objection entity rather than a one-off flag |
| New logic needed | Each objection gets a stable ID, a history of attempted strategies (so the same failed reframe isn't reused), and lifecycle transitions triggered by evidence, not just repetition count |
| Known risk | Inherits the same fragility semantic recurrence showed in the Behavioral Signal Engine (false positives on incidental overlap, false negatives on filler words) — budget for the same iterative adversarial testing |
| Definition of Done | An objection raised, addressed, and reactivated later in different words is tracked as one entity with full history, not three disconnected events |
| Sprint | Sprint 2-3 |

---

### PHASE 4 — Truth Supersession Detector ⚠️ NEW CAPABILITY
**Objective:** Detect when a new statement updates or reverses an earlier one, even when there's no obvious contradiction in wording.

| | |
|---|---|
| The hard part | "We've decided to stay" → "I'd still move if I believed there was a better strategy" doesn't contain contradictory keywords. Detecting the second statement supersedes the first requires reasoning about *meaning*, not matching text |
| Approach | LLM-based comparison: given a new statement and the current state's relevant field, ask "does this update, reverse, or leave unchanged the existing belief about X?" — scoped narrowly per field, not a general-purpose contradiction detector |
| Guardrail | Old value is never deleted — moved to history with a timestamp and superseding turn ID, exactly as required |
| Definition of Done | The exact client example (staying → conditional on strategy) correctly updates decision state while preserving the original statement in history |
| Sprint | Sprint 3-4 (treat as its own mini-project, not a quick add-on) |

---

### PHASE 5 — Materiality Filter ⚠️ NEW CAPABILITY
**Objective:** Decide which state fields a given turn actually affects — don't recompute everything on every utterance.

| | |
|---|---|
| The hard part | A statement about the spouse's schedule should touch "decision structure," not Trust or Emotion. This requires classifying *what kind of information* a turn carries before deciding what to update |
| Approach | Lightweight LLM classification step per turn: "which of [decision structure / objection / trust / readiness / facts] does this turn's evidence materially affect?" — output is a small set of field names, not a full re-score |
| Definition of Done | A turn that only reveals a new fact does not also perturb Trust/Emotion scores; verified via a test where an unrelated fact is stated and dimension scores remain provably unchanged |
| Sprint | Sprint 4 |

---

### PHASE 6 — Momentum & Readiness Scoring
**Objective:** Implement the weighted scoring model from the second spec document.

| Momentum evidence family | Weight |
|---|---|
| Problem/goal clarity | 15% |
| Value recognition | 15% |
| Objection movement | 20% |
| Trust/engagement trend | 15% |
| Decision structure clarity | 10% |
| Future/operational behavior | 15% |
| Commitment behavior | 10% |

| | |
|---|---|
| Output | Momentum 0-100 + trend label (advancing / stable / stalling / regressing) |
| Readiness | Weighted mean across emotional, logical, logistical, decision-confidence dimensions, with **blocker caps** — e.g., readiness cannot exceed a configured ceiling (default 60) if logistical readiness is below a threshold, regardless of enthusiasm elsewhere |
| Reuses | `InferenceScoringConfig` pattern — named, versioned, documented weights, not bare numbers |
| Definition of Done | Acceptance test from the spec passes: high enthusiasm + absent decision-maker does not produce high overall readiness |
| Sprint | Sprint 5 |

---

### PHASE 7 — Meeting/Conversion Gate & Push Strength
**Objective:** Determine when the state supports pushing for a confirmed next step, and how hard.

| | |
|---|---|
| Gate conditions | Trust not collapsing, engagement on-topic, major objection resolved/partial, clear value reason, correct decision-maker identified, plausible logistics, no active boundary |
| Push strength states | Low trust/high threat → protect & shorten; moderate trust + unresolved concern → resolve then ask; rising trust + clear value → direct ask; vague but agreeable → two-window choice; soft hesitation → reduce friction, re-ask; hard boundary → respect, record, exit |
| Conversion Event Object | `conversion_type`, `status`, `start_at`, `location_or_format`, `participants`, `confirmation_confidence`, `source_turn_ids`, `blocking_items`, `followup_is_conversion` |
| Definition of Done | All 5 acceptance tests from the spec pass (friendly-but-vague ≠ high readiness; confirmed walkthrough = success even if short; "send me something" ≠ success; repeated unresolved objection blocks despite polite "okay"; absent decision-maker caps readiness regardless of enthusiasm) |
| Sprint | Sprint 5-6 |

---

### PHASE 8 — Explainability & Versioning
**Objective:** Every meaningful state change must be traceable to its cause.

| | |
|---|---|
| Mechanism | Every field change stores: old value, new value, triggering turn ID, evidence reference, timestamp |
| Versioning | `state_version` increments on every meaningful mutation — same monotonic pattern as `DownstreamInferenceState.state_version` |
| Stale-write protection | Downstream consumers (Strategic Decision Engine, teleprompter) must check state version before acting, so a delayed async result can never overwrite a materially newer state |
| Definition of Done | Any field value can be traced backward to the exact turn and evidence that produced it — same standard as the Behavioral Signal Engine's evidence log |
| Sprint | Woven through every phase above, not a separate sprint |

---

### PHASE 9 — Inspection/Replay Console
**Objective:** The client's explicit request — select a turn, see the full before/after story.

| | |
|---|---|
| View | `State Before → New Evidence → State Changes + Reasons → State After` |
| Reuses | Same pattern as the Behavioral Signal Engine's test console, one layer up — turn-by-turn replay, but showing interpreted state instead of raw signals |
| Client note | Explicitly requested to be preserved permanently, including post-launch, for real-call debugging — build this as a durable tool, not a throwaway test page |
| Definition of Done | Any past call can be replayed turn-by-turn, showing exactly what ConversationState believed at each step and why |
| Sprint | Sprint 6-7 |

---

### PHASE 10 — Acceptance Testing Against Both Specs
**Objective:** Prove the build against the client's actual written criteria.

| Test | From |
|---|---|
| Current truth supersedes old truth, old preserved in history | Client's principle #1 |
| Evidence separate from interpretation, traceable | Client's principles #2, #4 |
| Not every field updates on every turn | Client's principle #3 (Phase 5) |
| Objection reactivated in new words retains history | Client's principle #5 (Phase 3) |
| Persistent facts survive unrelated turns | Client's principle #6 (Phase 2) |
| Dimensions move independently | Client's principle #7 |
| Uncertainty preserved, not forced to certainty | Client's principle #8 |
| Stale async result cannot overwrite newer state | Client's principle #9 |
| Friendly-but-passive prospect → high engagement, moderate readiness only | Spec doc acceptance test |
| Confirmed walkthrough = success even if call was short | Spec doc acceptance test |
| "Send me something" alone ≠ conversion | Spec doc acceptance test |
| Repeated commission objection stays unresolved despite polite "okay" | Spec doc acceptance test |
| Absent decision-maker caps readiness despite high enthusiasm | Spec doc acceptance test |

| | |
|---|---|
| Sprint | Sprint 7-8 (hardening/QA sprint) |

---

## 3. Suggested Sprint Mapping (2-week sprints)

| Sprint | Focus |
|---|---|
| 0 | Input contract validation |
| 1 | Core schema, begin persistent facts store |
| 2 | Finish persistent facts, begin objection lifecycle |
| 3 | Finish objection lifecycle, begin truth supersession |
| 4 | Finish truth supersession, materiality filter |
| 5 | Momentum & readiness scoring, begin conversion gate |
| 6 | Finish conversion gate, begin inspection console |
| 7 | Finish inspection console, acceptance testing |
| 8 | Hardening, adversarial testing on objection lifecycle & truth supersession specifically |

---

## 4. Cross-Cutting Rules (apply to every phase)

- **Never force certainty.** Every interpreted field carries a confidence value — same weakest-link discipline as the Behavioral Signal Engine.
- **Never silently overwrite history.** Superseded facts and resolved-then-reactivated objections keep their full trail.
- **Never blend evidence and interpretation in the same field.** Evidence is what Behavioral Signal reported; interpretation is what ConversationState concluded from it — always two distinct, linked things.
- **Every state mutation is versioned and explainable**, with zero exceptions — this is the property the client has been testing for hardest throughout the Behavioral Signal Engine validation, and it will be tested for here too.
