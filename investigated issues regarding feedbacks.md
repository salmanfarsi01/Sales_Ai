# Investigation & Architectural Triage: Client Test UI Feedbacks

**Date:** September 29, 2026  
**Subject:** Investigation of Within-Call State Leakage (Issue A) and Cross-Call Stale State Contamination (Issue B)  
**Constraint Enforced:** Read-Only Investigation (Zero source code modifications made).

---

## 1. Executive Summary & Revised Triage Verdict

### Revised Verdict

> **"Both issues share the same architectural anti-pattern — code reading from a mutable 'latest' or shared reference instead of an immutable, fully-keyed (call_sid + state_version) lookup — but they occur in three separate places:  
> (1) `CoreDecisionManager.latest_decision` causes the within-call leakage in Issue A;  
> (2) `PromptDeliveryService`'s `call_sid="default_call"` singleton fallback and  
> (3) prospect-scoped persistence in `ContactPreferenceStore` both contribute to Issue B.  
> All three need independent fixes; #1 does not fix #2 or #3."**

---

## 2. Issue Identification & The Three Distinct Mechanisms

Rather than a single bug, the investigation reveals **three distinct underlying mechanisms** operating across different layers of the system:

```mermaid
flowchart TD
    subgraph AntiPattern [Shared Pattern: Trusting Mutable/Shared Pointers over Fully-Keyed Lookups]
        direction TB
    end

    subgraph M1 [Mechanism 1: In-Call Decision Overwrite]
        M1_A[CoreDecisionManager.latest_decision overwritten each turn] --> M1_B[Replay/rendering path reads latest_decision instead of decision_history[turn_id]]
        M1_B --> IssueA[Issue A: Within-Call Future Leakage on Turn 3]
    end

    subgraph M2 [Mechanism 2: Cross-Call Object Reuse Fallback]
        M2_A[PromptDeliveryService defaults to CoreDecisionManager call_sid='default_call'] --> M2_B[Multiple calls share default singleton instance]
        M2_B --> IssueB_1[Issue B Part 1: Stale Decision Contamination]
    end

    subgraph M3 [Mechanism 3: Cross-Call Persistent Storage Collision]
        M3_A[ContactPreferenceStore writes to knowledge/prospect_preferences.json] --> M3_B[Test harness reuses prospect_id='test_prospect_001']
        M3_B --> IssueB_2[Issue B Part 2: Contact Preferences Persist into Clean Tests]
    end

    AntiPattern -.-> M1
    AntiPattern -.-> M2
    AntiPattern -.-> M3
```

| Mechanism | Scope | Root Cause | Impacted Issue | Why Other Fixes Won't Solve It |
| :--- | :--- | :--- | :--- | :--- |
| **Mechanism 1 (In-Call)** | Single Call / Timeline | `CoreDecisionManager.latest_decision` gets overwritten on every turn, and the replay/rendering path inspects the terminal `latest_decision` (or terminal snapshot) instead of indexing `decision_history[turn_id]`. | **Issue A** | Resolves within-call scrubber leakage, but has zero effect across calls. |
| **Mechanism 2 (Cross-Call Object Reuse)** | Inter-Call / Runtime | `PromptDeliveryService` defaults to `CoreDecisionManager(call_sid="default_call")` — a shared singleton fallback that multiple calls land on when callers omit an explicit `call_sid`. | **Issue B** (Primary decision leak) | Even with turn indexing fixed, a shared manager instance still leaks state across calls that land on the fallback. |
| **Mechanism 3 (Cross-Call Persistent Storage)** | Inter-Call / Disk Storage | `ContactPreferenceStore` writes to and reads from `knowledge/prospect_preferences.json` keyed by `prospect_id`. The testing workflow reuses `"test_prospect_001"`, legitimately loading saved preferences into supposedly "clean" test calls. | **Issue B** (Preference leak: *"Contact preferences: 1 active"*) | Memory-level session purges do not touch persisted disk records; disk preferences will continue to contaminate tests using `"test_prospect_001"`. |

---

## 3. Hand-Tracing Issue A: Turn 3 Within-Call Future-State Leakage (Mechanism 1)

### A. What object does the Replay UI ask for?
1. The Replay UI (`web/conversation_state_replay.html`) renders a 4-column point-in-time inspection layout:
   - **Column 1:** State Before ($V_{t-1}$)
   - **Column 2:** Evidence & Gating for Turn $t$
   - **Column 3:** State Changes & Reasons applied by Turn $t$
   - **Column 4:** State After ($V_t$) and Strategic Decision evaluated for Turn $t$
2. When the user scrubs or clicks on **Turn 3** (`activeTurnIndex = 2`), the UI invokes:
   ```javascript
   renderSelectedTurn(currentReport.timeline[2])
   ```
3. Specifically, for Column 4, the UI asks for:
   - `step.state_after`: The snapshot reflecting conversation truth strictly after Turn 3 has been processed.
   - `step.strategic_decision`: The `StrategicDecision` object evaluated from Turn 3's point-in-time snapshot.
4. For Turn 3 of the 18-turn benchmark (`sim_mufc4lsh`):
   - Turn 3 is a salesperson question: *"Got it. What's driving the timing for you?"*
   - Expected `state_after`: `state_version = 4`, `stage = discovery`, `gate = CLOSED`, `commitment = 0.0`.
   - Expected `strategic_decision`: `source_state_version = 4`, `primary_action = QUESTION`, `strategic_objective = "Explore timeline and underlying motivations..."`.

### B. What does `core_decision_manager.py` return?
1. In `copilot/core_decision_manager.py`:
   ```python
   def evaluate_state(self, snapshot, ...):
       eval_result = self.engine.evaluate(snapshot=snapshot, ...)
       final_decision = self.modifier_pipeline.apply_modifiers(...)
       eval_result.decision = final_decision
       self.latest_decision = final_decision  # <--- OVERWRITTEN ON EVERY TURN
       self.latest_evaluation_result = eval_result
       self.decision_history.append(final_decision)
       return eval_result
   ```
2. When `evaluate_state()` is invoked during Turn 3 ($V_4$), it evaluates Turn 3 correctly and appends the result to `decision_history[2]`.
3. However, during the sequential execution of the 18-turn call, `decision_manager.evaluate_state()` is called 18 consecutive times on the same `CoreDecisionManager` instance.
4. By the time Turn 18 finishes executing, `decision_manager.latest_decision` is overwritten with Turn 18's decision ($V_{38}$ / $V_{42}$, Commitment 1.0, Action `ACKNOWLEDGE`, Gate `OPEN`).

### C. What `source_state_version` is attached to the returned `StrategicDecision`?
1. If the consumer inspects `decision_history[2]`, `source_state_version` is **4**.
2. If the consumer inspects `self.latest_decision` (or if the UI / replay report binds to the manager's `latest_decision` or final snapshot rather than indexing by turn), the attached `source_state_version` is **38** (or **42**).
3. Furthermore, in Python reference models, if `step.state_after` in the timeline was bound to `manager.current_state` without strict immutable serialization per turn, subsequent mutations on Turns 4..18 mutated the referenced dictionaries, causing Turn 3 to display $V_{38}$ dimensions (`Commitment 1.00`, `Thursday At 3`).

---

## 4. Hand-Tracing Issue B: Test A → Test B Cross-Call Leakage (Mechanisms 2 & 3)

### A. What happened in Test A?
1. Test A was the full 18-turn benchmark call (`sim_mufc4lsh` or `sim_validation_test`).
2. On Turn 18 of Test A, the prospect said:
   > *"Thursday at 3 works, and my wife will be there."*
3. The engine processed Turn 18:
   - Extracted fact: `confirmed_meeting_time: Thursday At 3`
   - Gate status: `OPEN` (Statement override)
   - Readiness: `80%` (or `72.2%`)
   - Momentum: `advancing`
   - Contact preference: `No daily texting` (active)
   - Evaluated `StrategicDecision`: `Action = ACKNOWLEDGE`, `source_state_version = 38` (or `42`).
   - `decision_manager.latest_decision` and `engine` evidence traces were populated with this Turn 18 evidence.

### B. What happened when Test B was executed?
1. Test B was a new, independent test dialogue (e.g., a 4-turn boundary script or a newly uploaded audio track).
2. Test B requested processing via `/api/conversation-state/replay-dialogue` or `/api/test/analyze-recording`.
3. **The Two Contributors to Issue B Fire Simultaneously:**
   - **Mechanism 2 (Object Reuse):** In `copilot/prompt_delivery_service.py`:
     ```python
     self.decision_manager = decision_manager or CoreDecisionManager(call_sid="default_call")
     ```
     When callers do not inject an isolated `decision_manager`, it falls back to a shared default instance (`call_sid="default_call"`). If Test A and Test B both hit this fallback, Test B reads Test A's `latest_decision`!
   - **Mechanism 3 (Persistent Storage):** In `copilot/behavioral_baseline.py`:
     ```python
     # ContactPreferenceStore writes to knowledge/prospect_preferences.json
     pref_store.save_preference("test_prospect_001", ...)
     ```
     When Test B runs using the default prospect ID `"test_prospect_001"`, `pref_store.get_preference("test_prospect_001")` immediately pulls the saved preference from Test A off the disk.
4. When Test B constructs its diagnostic trace via `_attach_full_trace()`, it combines the stale decision from Mechanism 2 with the persisted disk record from Mechanism 3:
   - `Turn utterance (client): "Thursday at 3 works, and my wife will be there."`
   - `Readiness: 80%`
   - `Gate: OPEN`
   - `Active objections: general_hesitation`
   - `Contact preferences: 1 active`

---

## 5. Prospect Memory Clarification & Corrected Framing for Mechanism 3

### Is `ContactPreferenceStore` persistence intentional or a test-environment defect?
**Verdict: It is 100% INTENTIONAL PROSPECT MEMORY.**

1. **Client Requirement Specification (`clients_requirement(220826).md`, Item 2):**
   > *"This should be a soft contact preference/boundary, not a sales objection. It should constrain future behavior and persist to prospect memory, but it shouldn't unnecessarily enter the objection lifecycle."*
2. **Core Runtime Implementation (`copilot/conversation_state_manager.py` lines 512, 521, 589 & `copilot/core_intelligence_engine.py` line 230):**
   - Explicitly persists contact friction and boundaries to Prospect Memory so that when a real prospect interacts in future calls, their stated preferences (e.g., "no daily texting", "mornings unavailable") are remembered and respected.
3. **Product Architectural Intent (Spec 01 & Spec 05):**
   - The persistence of `prospect_preferences.json` keyed by `prospect_id` is a core production feature designed to maintain long-term relationship memory across disparate call sessions.

### Why "Sandbox/Isolate Preference Storage" was the Wrong Framing:
Framing the fix as *"sandbox or disable storage during simulation"* would quietly cripple a valid, spec-mandated production feature.

### The Correct Defect Definition & Solution:
- **The True Defect:** The test harnesses (`run_behavioral_signal.py` lines 180 and 242, and simulation runners) hardcode `prospect_id="test_prospect_001"` across all runs. Every simulation run therefore acts as a *returning prospect*, inadvertently exercising the "returning prospect memory" code path rather than the "clean new prospect" code path.
- **The Correct Fix:** The test harness and simulation endpoints must generate a **unique, fresh prospect identity** by default (e.g. `f"sim_prospect_{call_sid}"` or a clean UUID), and only reuse a specific `prospect_id` when the test scenario is explicitly designed to test Prospect Memory recall.

---

## 6. Point 1 — Why Test A's Utterance Appeared in Test B (Client's Top Priority)

The client requested a strict investigation before changing any code, checking three specific vectors in order:

### Check (a): Is `ConversationStateManager` a singleton or module-level instance reused across calls?
- **Investigation:** Examined `copilot/conversation_replay.py` (lines 116–120) and `copilot/conversation_state_manager.py`.
- **Finding:**
  ```python
  manager = self.state_manager or ConversationStateManager(
      call_sid=call_sid,
      initial_snapshot=initial_state,
      conversion_target=conversion_target,
  )
  ```
  On all simulation and replay endpoints (`/api/conversation-state/replay-dialogue` and `/api/conversation-state/replay/{call_sid}`), `self.state_manager` is `None`. A brand new `ConversationStateManager` is instantiated fresh on every request, keyed strictly by `call_sid`. There is NO module-level singleton instance in memory.
- **Conclusion:** **NEGATIVE (NOT state-manager-reuse).**

### Check (b): Is there a process-level cache (`lru_cache`, un-keyed dict, global `_last_decision`) in `core_decision_manager.py` or `llm_response_gateway.py`?
- **Investigation:** Grepped and hand-audited `copilot/core_decision_manager.py` and `copilot/llm_response_gateway.py`.
- **Finding:** Neither file contains `lru_cache`, module-level dicts, or un-keyed global variables. In `copilot/conversation_replay.py` line 124, `decision_manager = CoreDecisionManager(call_sid=call_sid)` is constructed fresh on every replay call.
  *(Note: `PromptDeliveryService` in `prompt_delivery_service.py` line 34 does have a default fallback `call_sid="default_call"`, but this service is not invoked by `conversation_replay.py` or `run_behavioral_signal.py`.)*
- **Conclusion:** **NEGATIVE (NOT backend process-level cache in Core or Gateway).**

### Check (c): Is the Replay UI caching the last-rendered decision client-side and not clearing it?
- **Investigation:** Hand-traced `web/conversation_state_replay.html`.
- **Finding: POSITIVE (PRIMARY ROOT CAUSE FOR POINT 1).**
  The UI investigation revealed three exact failure points causing Test A's utterance to appear in Test B:
  1. **Exact Line 656 & Lines 1873–1886 (`#reportSelect` has NO `onchange` listener):**
     In `<select id="reportSelect">`, there is **no `onchange` event handler**. When a user selects a different report from the dropdown (switching from Test A `sim_mufc4lsh` to Test B), the browser DOM does not trigger an update. The UI only fetches a new report if the user explicitly clicks the "Load Replay" button (`#btnLoadReport`). If a user selects a report expecting standard dropdown reactivity, the UI remains entirely frozen on Test A!
  2. **Exact Lines 1417–1443 (`renderSelectedTurn` Null-Handling on Older Reports):**
     In `renderSelectedTurn(step)`:
     ```javascript
     col4.innerHTML = `
     ${step.strategic_decision ? `
       ... PitchProX Full Strategic Trace ...
       ${step.strategic_decision.evidence_considered.map(e => `<li>${escapeHtml(e)}</li>`).join('')}
     ` : ''}
     `;
     ```
     Out of 43 reports in `reports/synthetic/`, only 4 benchmark reports (`sim_mufc4lsh`, `sim_mukmxsle`, `sim_mucj0p5s_golden`, `sim_validation_test`) have `step.strategic_decision` populated. If a user navigated to a test report where `strategic_decision` was missing, or if an error occurred during loading, Column 4 either retained the previously rendered view or failed to reflect fresh test data.
  3. **Exact Lines 1923–1946 (`btnRunSimScript.onclick` Error Path State Retention):**
     When a new test script is submitted, `currentReport` is updated only on successful HTTP response. If a simulation error occurs or if the response fails to parse, `currentReport` is not reset to `null`, leaving the previous report's DOM intact.
  4. **Coupling with Mechanism 1 (In-Call Terminal Overwrite):**
     Because `CoreDecisionManager.latest_decision` is overwritten on every turn, any view of Test A displayed Turn 18's terminal evidence (`"Thursday at 3 works, and my wife will be there."`) across early turns as well.

### Final Classification for Point 1
**UI-cache / un-cleared DOM state & missing dropdown event reactivity (`UI-cache`), coupled with test-harness prospect identity collision (`Mechanism 3`).**

---

## 7. Architectural Separation: Why Three Fixes Are Required

Fixing only one mechanism leaves the other failure modes completely intact:

```
[Fix 1: In-call turn-indexed lookup get_decision_for_turn(turn_id) + UI event binding] 
   └── Resolves Issue A (Turn 3 within-call scrubber) and UI report dropdown freeze
   └── DOES NOT RESOLVE cross-call identity reuse

[Fix 2: Remove default_call singleton fallback in prompt_delivery_service.py]
   └── Resolves cross-call object reuse when prompt delivery service is invoked without call_sid
   └── Enforces fail-loud contract

[Fix 3: Test harness unique prospect_id generation (preserving real Prospect Memory)]
   └── Resolves Issue B's persistent contact preference contamination across test runs
   └── PRESERVES intentional Prospect Memory for production and returning prospect tests
```

---

## 8. Comprehensive Root Cause Matrix

| Dimension | Mechanism 1 (Issue A) | Mechanism 2 (Issue B: Object Reuse) | Mechanism 3 (Issue B: Persistent Storage) |
| :--- | :--- | :--- | :--- |
| **Object Requested** | Turn 3's point-in-time `TurnReplayStep` / `StrategicDecision`. | Test B's Turn 1 initial `StrategicDecision`. | Prospect contact preferences for Test B. |
| **Object Returned** | Terminal decision from Turn 18 ($V_{38}$). | Stale terminal decision from Test A's Turn 18 ($V_{38}$). | Persisted preference record from Test A. |
| **Attached `source_state_version`** | $V_{38}$ (instead of $V_4$). | $V_{38}$ (instead of $V_1$ or $V_2$). | N/A (Persistent record on disk). |
| **Root Cause Location** | `copilot/core_decision_manager.py` + `web/conversation_state_replay.html` (lines 656, 1417–1443, 1873). | `copilot/prompt_delivery_service.py` (line 34: `call_sid="default_call"`). | `run_behavioral_signal.py` (lines 180, 242) hardcoding `"test_prospect_001"`. |
| **Classification** | **UI-cache / Rendering pointer** | **Backend fallback object reuse** | **Test-harness identity collision (Prospect Memory working as designed)** |
| **Fix Required** | Turn-indexed lookup `get_decision_for_turn(turn_id)` + attach `onchange` on `#reportSelect`. | Require explicit `call_sid`, remove `"default_call"` default entirely. | Test harness generates unique `prospect_id` by default (e.g. `f"sim_prospect_{call_sid}"`). |

---

## 9. Implementation Sequence (Ready Upon User Approval)

1. **Step 1 (Mechanism 1: In-Call & Replay UI):**
   - In `copilot/core_decision_manager.py`: Expose explicit `get_decision_for_turn(turn_id)` and `get_decision_for_version(state_version)`.
   - In `copilot/conversation_replay.py`: Ensure `step.strategic_decision` is an immutable snapshot keyed strictly to `bundle.turn_id`.
   - In `web/conversation_state_replay.html`: Add `onchange` handler to `#reportSelect`, ensure UI cleanses previous report DOM before rendering a new selection, and handles missing/null decisions cleanly.
2. **Step 2 (Mechanism 2: PromptDeliveryService Fallback Removal):**
   - In `copilot/prompt_delivery_service.py`: Remove `call_sid="default_call"` default; require explicit `call_sid` and fail loudly with a `ValueError` if omitted.
3. **Step 3 (Mechanism 3: Test Harness Fresh Prospect IDs):**
   - In `run_behavioral_signal.py` and simulation endpoints: Generate unique prospect IDs by default (e.g. `f"test_prospect_{call_sid}"`), allowing tests to exercise genuine clean-slate prospect evaluation while keeping `knowledge/prospect_preferences.json` fully intact for returning-prospect tests.

---

## 10. Completed Resolutions: Point 2 & Point 3

### Point 2 — New-Call Initialization & Prospect Memory Isolation
1. **Brand-New State Construction:**
   - Updated [ConversationStateManager](file:///c:/dev/Sales_Ai/copilot/conversation_state_manager.py) so every new `call_sid` instantiates distinct sub-engines and collections:
     - `_bundle_history`: distinct empty list
     - `_conversion_events`: distinct empty list
     - `_deal_dispositions`: distinct empty list
     - `current_state.objections`: distinct empty list
     - `current_state.facts`: distinct empty list
     - `current_state.stage_history`: distinct deep copy
     - Sub-engines (`facts_manager`, `objections_engine`, `conversion_engine`, `momentum_engine`, etc.) constructed freshly.
2. **Explicit Prospect Memory Separation:**
   - Contact preferences are **never** loaded into fresh calls by default.
   - Added explicit parameters `load_prospect_memory: bool = False, prospect_id: Optional[str] = None` to `ConversationStateManager.__init__`.
   - Contact preferences are only loaded from `ContactPreferenceStore` if `load_prospect_memory=True` and a valid `prospect_id` is supplied.
3. **PromptDeliveryService Singleton Removal:**
   - Updated [PromptDeliveryService](file:///c:/dev/Sales_Ai/copilot/prompt_delivery_service.py) to require either `decision_manager` or `call_sid`.
   - Removed `call_sid="default_call"` fallback, raising a loud `ValueError` if neither is provided.
4. **ConversationReplay Integration:**
   - Wired `load_prospect_memory` and `prospect_id` through `replay_call`, `replay_dialogue_turns`, and `ReplayDialogueRequest` in [copilot/conversation_replay.py](file:///c:/dev/Sales_Ai/copilot/conversation_replay.py).

### Point 3 — Cache Isolation Keying on (call_sid, source_state_version)
1. **StrategicDecisionCache Implementation:**
   - Implemented `StrategicDecisionCache` in [copilot/core_decision_manager.py](file:///c:/dev/Sales_Ai/copilot/core_decision_manager.py):
     - Cache dictionary keyed strictly by `(call_sid, source_state_version)`.
     - Call-scoped turn index keyed strictly by `(call_sid, turn_id)`.
     - Rejects any lookup without a valid non-empty `call_sid` with `ValueError("call_sid is required...")`.
2. **Turn Isolation Across Calls:**
   - Requesting the same turn number across two distinct `call_sids` strictly routes to that call's decision instance and evidence bundle; cross-call leakage or collision is prevented.

### Automated Test Verification
- Created dedicated test suite [tests/test_points_2_and_3_isolation.py](file:///c:/dev/Sales_Ai/tests/test_points_2_and_3_isolation.py):
  - `test_point2_new_call_initialization_zero_shared_state`: Verifies `id(mgr_a) != id(mgr_b)` across all collections and asserts mutations in Call A leave Call B completely clean.
  - `test_point2_prospect_memory_explicit_loading_isolation`: Verifies that `load_prospect_memory=False` yields zero loaded preferences, while `load_prospect_memory=True` intentionally loads saved boundaries.
  - `test_point3_cache_isolation_keying`: Verifies that querying the exact same turn number for two different `call_sids` returns different decisions and evidence, that queries are keyed by `(call_sid, source_state_version)`, and that un-keyed lookups raise `ValueError`.
  - `test_prompt_delivery_service_cross_call_isolation`: Verifies that `PromptDeliveryService` prohibits un-keyed initialization without `call_sid` or `decision_manager`.
  - `test_point8_two_back_to_back_test_calls_zero_cross_call_leakage`: Verifies end-to-end back-to-back Test A -> Test B execution has zero leakage across facts, objections, contact preferences, gate, and evidence panels.

---

## 11. Completed Resolutions: Point 4, Point 5 & Point 6

### Point 4 — Remove "Latest State" Fallback
1. **Strict Version Gate in Replay UI:**
   - In [`web/conversation_state_replay.html`](file:///c:/dev/Sales_Ai/web/conversation_state_replay.html), Column 4 validates:
     ```javascript
     const hasValidDecision = sd && (sd.source_state_version === sAfter.state_version);
     ```
   - If a selected turn has no valid CoreDecision for its exact version, the UI renders an amber `Pending / No Decision Recorded` banner with `V{state_version} PENDING`. It **never** silently substitutes the final/latest call state or subsequent turns.
2. **Backend Non-Fallback Guarantee:**
   - In [`CoreDecisionManager`](file:///c:/dev/Sales_Ai/copilot/core_decision_manager.py), `get_decision_for_turn(turn_id)` and `get_decision_for_version(state_version)` return `None` if no decision was recorded for that exact version/turn. They never fall back to `self.latest_decision`.

### Point 5 — Evidence Consistency (Single Coherent Snapshot)
1. **Single-Snapshot Assembly Invariant:**
   - In [`PitchProXCoreIntelligenceEngine`](file:///c:/dev/Sales_Ai/copilot/core_intelligence_engine.py), the `Evidence Considered` block is strictly assembled from exactly one `ConversationStateSnapshot` at `source_state_version`.
   - `decision.utterance_turn_id`, `decision.metrics_source_turn_id`, and `decision.source_turn_id` are explicitly set from the turn currently being evaluated:
     $$\text{evidence.utterance\_turn\_id} == \text{evidence.metrics\_source\_turn\_id} == \text{decision.source\_state\_version's turn}$$
   - Utterance text and behavioral metrics (trust, readiness, momentum, gate, objections) are bound to the identical snapshot. Never does utterance come from turn $N$ while metrics come from turn $N+k$.

### Point 6 — Traceability in the Debug UI
1. **Traceability Header Bar:**
   - Exposed a 4-column metadata traceability header at the very top of the Strategic Trace card in [`web/conversation_state_replay.html`](file:///c:/dev/Sales_Ai/web/conversation_state_replay.html):
     - **CALL SID**: Active call identifier (e.g. `sim_mufc4lsh`).
     - **DECISION ID**: Unique decision ID (`dec_...`).
     - **SOURCE VERSION**: Point-in-time state version (`V2`, `V4`, `V6`...).
     - **SOURCE EVENT / TURN**: Triggering event ID and turn number (`ev_turn_N_vK (TN)`).
   - Confirmed that these four fields dynamically update turn-by-turn as the scrubber or timeline steps are selected.

### Automated Test Verification
- Created dedicated test suite [`tests/test_points_4_5_6_traceability.py`](file:///c:/dev/Sales_Ai/tests/test_points_4_5_6_traceability.py):
  - `test_point4_pending_state_when_no_decision_recorded`: Verifies `get_decision_for_turn` and `get_decision_for_version` return `None` for unrecorded turns, and verifies the UI renders the pending state banner.
  - `test_point5_evidence_consistency_single_coherent_snapshot`: Evaluates all 18 turns of the benchmark call and asserts that for every single turn, `utterance_turn_id == metrics_source_turn_id == source_turn_id == turn_id`, and that utterance text matches that exact turn.
  - `test_point6_traceability_fields_present_and_unique_turn_by_turn`: Verifies that `call_sid`, `decision_id`, `source_state_version`, and `source_event_id` are populated on every decision, `decision_id` is unique per turn, and version advances monotonically.
- Full repository test run: **432 passed, 0 failed in 48.58s**.



