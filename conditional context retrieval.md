# Conditional Knowledge Retrieval: Implementation Workflow

**Component:** `[ CONDITIONAL KNOWLEDGE RETRIEVAL: ONLY WHEN NEEDED ]`
**Primary reference:** Doc 02, AI Training & Retrieval Specification (§6–§13)
**Supporting references:** Doc 01, Core Runtime & State Engine (§10 context priority, §13 latency); Doc 06, Source Hierarchy & Supersession Index (authority resolution)
**Status:** Build plan for developers

---

## 0. One-paragraph summary

After the Core has chosen a strategic action, this component decides whether personalized knowledge would improve the next response. If yes, it retrieves only authorized, current, verified knowledge belonging to the individual user. It ranks and conflict-checks the results and hands the LLM a structured fact block. If no suitable knowledge exists, it returns an empty result and the pipeline continues truthfully without invented facts. **The Core decides the strategy. Retrieval supplies facts. The LLM verbalizes.** Retrieval never decides strategy, never overrides live conversation truth, and never runs just because a retrieval step exists in the pipeline.

---

## 1. Client constraints (apply to every task below)

| # | Constraint | Implementation consequence |
|---|---|---|
| C1 | Retrieval is **conditional**, triggered by a Core-identified need | A Retrieval Gate sits in front of the vector search. "No retrieval" is a normal, first-class outcome. |
| C2 | Retrieved content **supports** the selected strategy | Output goes into the "facts" block only. It cannot change the strategic action. |
| C3 | Retrieved content must not override **current conversation truth** | Live state wins in every conflict (Doc 01 §10 priority 2–3 over priority 5). |
| C4 | Never force irrelevant material into the response | Relevance threshold plus an "empty result" path. |
| C5 | If knowledge is unavailable, continue truthfully, no invention | Fallback: a different strategic move, generic truthful language, or a clarifying question. |
| C6 | **Individual-user ownership only** | Every query is hard-filtered to `owner_user_id = current user`. Team, brokerage and company scopes are **not** enabled. No shared Prospect Memory. No team-sharing features. |
| C7 | Doc 06 decides which source wins | Where a historical file conflicts, Level A docs win. Level C/D content is never treated as verified fact. |

> **Scope note (C6).** Doc 02 mentions `team / brokerage / company` scopes in §3, §4, §5 and §13. For this build, treat those as **reserved enum values that are rejected at write time and never matched at read time**. Do not build team-share UI, shared indexes or cross-user sharing. The acceptance test "same company, different users" becomes a strict **no cross-user leakage** test.

---

## 2. Where this component sits

```
STRATEGIC DECISION (Core output)
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. RETRIEVAL GATE        (need? yes/no) │──no──► empty RetrievalResult
└─────────────────────────────────────────┘                │
        │ yes                                              │
        ▼                                                  │
┌─────────────────────────────────────────┐                │
│ 2. QUERY BUILDER                        │                │
│ 3. PERMISSION + VALIDITY PRE-FILTER     │                │
│ 4. VECTOR SEARCH (candidates)           │                │
│ 5. RANKING + RELEVANCE THRESHOLD        │                │
│ 6. CONFLICT RESOLUTION                  │                │
│ 7. GUARDRAIL CHECK                      │                │
│ 8. RESULT PACKAGING                     │                │
└─────────────────────────────────────────┘                │
        ▼                                                  ▼
        └──────────────► LLM RESPONSE CONSTRUCTION ◄───────┘
                    (structured context blocks, §10)
```

**Inputs from upstream:** Core strategic decision, ConversationState snapshot (with version id), active Playbook constraints, Calibration constraints, `user_id`, `lead_type`, `prospect_type`.
**Output downstream:** one `RetrievalResult` object (possibly empty) consumed by LLM Response Construction.

### Pipeline-timing note (flag for client)
Doc 01 §3 lists "Context Assembly" (stage 5, which loads AI Training) *before* "Strategic Decision" (stage 6). The client's workflow diagram places conditional retrieval *after* Strategic Decision. **Follow the client's diagram.** Stage 5 may load lead type, Playbook, Calibration and prospect history, but **must not run AI Training retrieval**. Only the Core, once it has chosen an action, may request retrieval. This is consistent with Doc 02 §6 ("Retrieve only when the Core determines personalized knowledge can improve the next decision or language"). Confirm with the client in writing.

---

## 3. Build workflow

### Step 1: Define interfaces and contracts

**1.1 `RetrievalRequest` (Core → Retrieval)**

```json
{
  "request_id": "uuid",
  "call_id": "…",
  "state_version": 0,
  "user_id": "…",
  "lead_type": "…",
  "prospect_type": "…",
  "strategic_action": "Social Proof | Differentiate | De-Risk | Educate | Quantify | Clarify | …",
  "retrieval_need": {
    "trigger": "factual_question | credibility_objection | service_process_objection | playbook_evidence | policy_wording | prior_claim_reference",
    "need_statement": "e.g. verified local-market experience to de-risk credibility objection",
    "topic_tags": ["…"],
    "time_sensitive": false
  },
  "conversation_evidence": "short excerpt or paraphrase of what the prospect said",
  "deadline_ms": 0
}
```

**1.2 `RetrievalResult` (Retrieval → LLM assembly)**

```json
{
  "request_id": "…",
  "state_version": 0,
  "status": "retrieved | none_needed | none_found | low_confidence | conflict_unresolved | timed_out | error",
  "facts": [
    {
      "knowledge_id": "…",
      "text": "…",
      "knowledge_class": "…",
      "verification_status": "…",
      "source_title": "…",
      "source_version": "…",
      "confidence": 0.0
    }
  ],
  "withheld": [{ "knowledge_id": "…", "reason": "disputed | stale | unverified | restricted" }],
  "fallback_hint": "use_generic_truthful | ask_clarifying_question | choose_alternate_move | null",
  "latency_ms": 0
}
```

**Rules**
- Every request and result carries `state_version`. A result for an older version is discarded (see Step 9).
- `status != retrieved` is not an error. Downstream must handle each status explicitly.
- `withheld` is for audit and debugging. It is never shown to the LLM as usable fact.

---

### Step 2: Build the Retrieval Gate (Doc 02 §6)

The Gate is a deterministic check on the Core's decision. It is **not** a similarity search.

**Retrieve only if the Core has flagged one of these triggers:**

| Trigger ID | Doc 02 §6 rule |
|---|---|
| `factual_question` | Prospect asks a factual question about the salesperson or company |
| `credibility_objection` | A credibility objection needs a real, relevant proof point |
| `service_process_objection` | A service/process objection needs user-specific information |
| `playbook_evidence` | Playbook/Core strategy calls for evidence **and** verified user-specific evidence exists |
| `policy_wording` | A company policy or approved wording is relevant |
| `prior_claim_reference` | Prospect references a prior promise, service, claim or fact stored in the user's knowledge base |

**Gate logic**

```
IF no trigger flagged                           → none_needed (STOP)
IF strategic_action IN {Wait, Acknowledge}      → none_needed (STOP)   # no facts needed
IF user has zero active, in-scope knowledge     → none_found  (STOP, skip vector call)
ELSE                                            → proceed to Query Builder
```

**Design notes**
- The trigger comes from the Core's decision object. The retrieval service must not re-derive strategy.
- For `playbook_evidence`, check that verified evidence exists *before* committing to the strategy. A cheap metadata existence check (`verification_status` + `knowledge_class` count for this user) avoids selecting Social Proof when none is available. If none exists, return `fallback_hint = choose_alternate_move` so the Core can pick a different action (Doc 02 §9).
- Log every gate decision with its trigger and reason, including the "no" decisions.

---

### Step 3: Query Builder

1. Build the query from `need_statement` + `topic_tags` + a trimmed `conversation_evidence`. Do **not** pass the whole transcript.
2. Embed the query with the same embedding model used at ingestion. Pin the model version in config, and re-embed on model change.
3. Derive metadata filters:
   - `knowledge_class` shortlist from the trigger (e.g. `credibility_objection` → Salesperson Profile, Performance/Past Sales, Value Proposition).
   - `lead_type_tags` / `prospect_type_tags`: soft match for ranking, hard filter only when a chunk declares a constraint that excludes the current lead type.
   - `topic_tags` overlap.

---

### Step 4: Hard pre-filters (permission, validity, status)

Apply **before** similarity ranking. These are filters, not score penalties.

| Filter | Rule |
|---|---|
| **Ownership** | `owner_scope = 'user' AND owner_user_id = :user_id`. No other scope matches. |
| **Deletion** | Exclude `deleted = true` and any chunk with a pending delete tombstone. |
| **Supersession** | Exclude chunks flagged as superseded by a newer approved version (keep them in audit storage only). |
| **Validity window** | Exclude chunks where `now < effective_from` or `now > effective_to`. |
| **Sensitivity** | `restricted` chunks are never returned to the LLM context. `confidential` chunks require an explicit allow in config. |
| **Lead-type constraint** | Exclude chunks whose `lead_type_tags` explicitly don't include the current lead type. |

**Enforcement requirement:** enforce the ownership filter **inside the vector query** (namespace or metadata filter), not in application code after retrieval. Post-filtering risks leakage through logs, caches and re-rankers.

---

### Step 5: Vector search and ranking (Doc 02 §7)

Retrieve top-K candidates (start K = 10–20, tune later), then score. Doc 02 lists ranking factors but gives no weights. **Do not invent fixed numbers as if they were spec.** Make the weights configurable and tune them against the acceptance tests.

| Factor (Doc 02 §7) | Implementation |
|---|---|
| Semantic relevance to the current need | Cosine similarity against the query embedding |
| Permission/scope match | Already enforced by the hard filter (Step 4), so no score needed |
| Lead/prospect type relevance | Boost for matching `lead_type_tags` / `prospect_type_tags` |
| Recency / effective date (only when time-sensitive) | Apply a recency boost only if `time_sensitive = true` |
| Verification/approval status | Boost: `company approved` / `system validated` > `user supplied` > unverified |
| Specificity to the current prospect/question | Boost for tag/topic overlap and narrower chunks |
| Conflict penalty for stale or contradictory chunks | Penalty applied after conflict detection (Step 6) |

**Relevance threshold (critical for C4):**
- Define a minimum semantic-relevance floor. Chunks below it are dropped **even if they are the top result**.
- If nothing clears the floor → `status = none_found`.
- If the best candidate clears the floor only marginally → `status = low_confidence`, `fallback_hint = use_generic_truthful | ask_clarifying_question`.
- Cap returned facts (suggest 1–3). The LLM needs the right fact, not a document dump.

---

### Step 6: Conflict resolution (Doc 02 §8)

Run on the ranked candidates before packaging.

| Situation | Required behavior |
|---|---|
| Two user documents disagree | Prefer current / approved / effective content. If unresolved, **do not assert the disputed fact**. Return `conflict_unresolved` and list it under `withheld`. |
| User upload conflicts with compliance/safety rule | **Compliance/safety wins.** Withhold the chunk. |
| User script conflicts with live conversation truth | Live state / Core strategy wins. Use the newest effective approved version and retain provenance. |
| Playbook method conflicts with company factual policy | Keep the Playbook strategy but use the company-approved fact/policy. |
| Historical performance claim lacks verification | Do not present as fact. Withhold. |
| Old and new source versions coexist | Newest effective approved version wins. Keep the older one in audit history. |

**Implementation notes**
- Conflict detection can start rule-based: same `knowledge_class` + overlapping `topic_tags` + different `source_version` or contradictory values.
- Where factual contradiction needs semantic judgment, use a small LLM/NLI check and **default to withholding** when the result is uncertain.
- Always record `source_title`, `source_version` and `uploaded_at` for provenance.
- The compliance check must hook into the existing Compliance Gate (Doc 01 stage 2). Do not reimplement compliance rules inside retrieval.

---

### Step 7: Hallucination guardrails (Doc 02 §9)

Hard rules, enforced both at retrieval time and in the generation instruction:

- Never invent: listings sold, transaction volume, clients, credentials, statistics, conversion rates, case studies, guarantees, integrations, certifications, or legal/compliance status.
- Marketing examples inside old PitchProX training files are **not** verified product facts. Treat them per Doc 06 Level C/D, and exclude them or mark them unverified. Do not let them rank as proof points.
- **Performance / past-sales chunks** are returned only when `verification_status` supports the claim.
- If the Core wants Social Proof and nothing verified exists → `fallback_hint = choose_alternate_move` (not a softened fabrication).
- If retrieval confidence is weak → generic truthful language or a clarifying question. Never manufacture precision (no invented numbers or specifics).

**Post-generation check (recommended):** scan the LLM draft for numbers, named clients, credentials or guarantees that do not appear in `RetrievalResult.facts`. Block or regenerate if found.

---

### Step 8: Package structured context for the LLM (Doc 02 §10)

Pass retrieved content as **separate, labeled blocks**, never a raw text dump.

| Context block | Source | Example |
|---|---|---|
| Strategic objective | Core decision | "De-risk credibility objection" |
| Conversation evidence | State / transcript | "Prospect questioned local experience" |
| Relevant retrieved facts | **This component** | "Verified sales/market experience from user profile" |
| Playbook constraints | Active Playbook | "Consultative, concise, no defensive credential dump" |
| Calibration constraints | User Calibration profile | "Short sentences, measured pace" |
| Generation instruction | System | "Produce one exact natural response; no invented claims" |

**Generation instruction must include:**
- Use only facts in the "Relevant retrieved facts" block.
- If the block is empty, produce a truthful response with no user-specific claims.
- Output is exact word-for-word language, not coaching labels (Doc 01 §11).

**Priority when blocks conflict (Doc 01 §10):**
Safety/compliance → Core integrity & live conversation truth → live state/lead type → Playbook → **AI Training (retrieved facts)** → Calibration → stylistic polish.
Retrieved facts sit at priority 5. They never outrank live state or the Playbook methodology, and they only supply the facts.

---

### Step 9: Latency, stale-prompt cancellation, observability (Doc 01 §13)

- **Instrument retrieval latency separately** (Gate, embedding, vector search, ranking/conflict, packaging). Doc 01 requires retrieval to be a distinct measured segment.
- **Time budget:** set a `deadline_ms` from the Core's remaining prompt-latency budget. Doc 01 prefers a shorter timely response over a longer late one. Exact numbers are not in the docs, so make them configurable and confirm with the client.
- **Timeout behavior:** return `status = timed_out`, `fallback_hint = use_generic_truthful`. Never block the prompt waiting on retrieval.
- **Stale cancellation:** compare `state_version` on result return. If the state has advanced (prospect interrupted, new utterance), **discard** the result and let the Core recompute. A late retrieval must never overwrite newer state.
- **Cheap short-circuits:** Gate "no" decisions and "no knowledge for this user" checks must skip embedding and vector calls entirely.
- **Log per request:** gate decision and trigger, filters applied, candidate count, scores, chosen/withheld chunks with reasons, status, per-stage latency. This supports the post-call requirement that analysis can reconstruct *why* a decision was made (Doc 01 §15).

---

### Step 10: Editing, deletion, versioning, permissions (Doc 02 §11)

| Requirement | Implementation |
|---|---|
| Users can replace or delete outdated knowledge | Replace = new `source_version` and the old one marked superseded. Delete = soft-delete flag plus vector removal. |
| Deleted knowledge stops being retrieved after propagation | Tombstone checked in the pre-filter (Step 4) so deletes take effect immediately, even before vector-index removal completes. Define and document the propagation window. |
| Versioned approved documents supersede older ones without losing audit history | Supersession flag in retrieval. Older versions stay in audit storage and are excluded from live search. |
| Permission changes apply immediately or within a defined window | Permissions are evaluated at query time from the source of truth, with no long-lived caching of permission decisions. Document the maximum cache TTL, if any. |

Cache rule: if any retrieval cache is used, key it by `user_id` + `state_version` + need, and invalidate on any edit/delete/permission event for that user.

---

### Step 11: Keep AI Training separate from Prospect Memory (Doc 02 §12)

- AI Training = **durable knowledge about the user/company** (this component's data).
- Prospect Memory = **information about a specific lead and their prior conversations** (different store, different lifecycle).
- **Do not** put prospect facts in the AI Training index, and do not merge result sets.
- Prospect-memory context, when needed, enters the LLM context through its own block. This component never reads from it.
- Per C6, Prospect Memory stays **individual-user scoped**. Nothing in this build adds shared Prospect Memory.

> **Flag for client:** the client's workflow box says *"Relevant verified AI Training / company / prospect facts."* Doc 02 §12 says not to combine AI Training and Prospect Memory. Recommended resolution: keep this component limited to AI Training (user-owned knowledge). Prospect facts are supplied by Prospect Memory via its own path. They are combined only at the prompt-assembly stage, as separate labeled blocks. Confirm this reading.

---

## 4. Data model (from Doc 02 §5, adapted for individual-user scope)

| Field | Purpose | This build |
|---|---|---|
| `knowledge_id` | Stable identifier | Required |
| `owner_scope` | user / team / brokerage / company | **Only `user` accepted.** Others reserved, rejected on write. |
| `owner_user_id` | Ownership key (add if not already present) | Required, indexed, used in vector pre-filter |
| `knowledge_class` | Type of knowledge | Required (the 9 classes in Doc 02 §3) |
| `source_title` | Human-readable source | Required |
| `source_version` | Version where known | Required for supersession |
| `uploaded_at` | Recency signal | Required |
| `effective_from` / `effective_to` | Validity window | Optional, enforced in pre-filter |
| `lead_type_tags` | Relevance constraints | Optional |
| `prospect_type_tags` | Relevance constraints | Optional |
| `topic_tags` | Pricing, service, marketing, compliance… | Required for query matching |
| `verification_status` | user supplied / company approved / system validated | Required, drives ranking and guardrails |
| `sensitivity` | Normal / confidential / restricted | Required, enforced in pre-filter |
| `text` | Original source chunk | Required |
| *(add)* `deleted`, `superseded_by` | Soft-delete and supersession flags | Needed for Step 10 |

> `company approved` in `verification_status` is a **label on user-owned data** in this build. It does not enable company-level sharing.

Ingestion (Doc 02 §4) is the dependency that populates these fields. If ingestion is built separately, confirm it writes this metadata before retrieval testing begins.

---

## 5. Implementation phases

**Phase 1: Contracts and gate**
- [x] `RetrievalRequest` / `RetrievalResult` schemas
- [x] Retrieval Gate with the six trigger rules, plus "no trigger → none_needed"
- [x] Gate-decision logging
- [x] Stub returning `none_needed` / `none_found` wired into LLM assembly (confirms the pipeline works with **no** retrieval)

**Phase 2: Safe retrieval**
- [x] Query builder and pinned embedding model
- [x] In-query ownership filter (`owner_user_id`), with team/company scopes rejected
- [x] Pre-filters: deletion, supersession, validity window, sensitivity, lead-type constraint
- [x] Vector search, top-K candidates

**Phase 3: Quality and truthfulness**
- [x] Configurable ranking (all Doc 02 §7 factors)
- [x] Relevance floor, `none_found` and `low_confidence` paths
- [x] Conflict resolution table (Step 6), with default-withhold on uncertainty
- [x] Guardrails and post-generation unsupported-claim check
- [x] Fallback hints wired to the Core (`choose_alternate_move`, etc.)

**Phase 4: Prompt assembly integration**
- [x] Structured context blocks (Step 8)
- [x] Generation instruction with the no-invention rule
- [x] Priority ordering per Doc 01 §10 verified in assembly code

**Phase 5: Runtime hardening**
- [x] Per-stage latency instrumentation
- [x] Deadline and timeout fallback
- [x] `state_version` stale-result discard
- [x] Delete/replace/permission propagation tests and cache invalidation

**Phase 6: Acceptance and regression**
- [ ] Full acceptance suite (below) automated in CI

---

## 6. Acceptance tests (Doc 02 §13, plus Doc 01 cross-checks)

| # | Test | Pass criteria |
|---|---|---|
| 1 | **No upload** | Gate/search yields `none_found`. Core still produces a strong prompt with no user-specific claims. |
| 2 | **Relevant upload** | Prospect asks about the rep's local experience, and the prompt accurately uses the verified fact. No extra claims. |
| 3 | **Irrelevant upload** | Upload about an unrelated topic is **not** retrieved. The prompt is unchanged versus the no-upload case. |
| 4 | **Contradictory upload** | Two conflicting docs → the disputed fact is not asserted. Newest approved wins where resolvable, otherwise withheld. |
| 5 | **Deleted upload** | After deletion propagates, the chunk never appears in retrieval or the prompt. |
| 6 | **Cross-user leakage (individual-user scope)** | User A's knowledge never retrieved for User B. This holds even in the same company, because team sharing is **not enabled**. |
| 7 | **Historical marketing claim** | A claim from an old training doc is not used as a verified fact. |
| 8 | **No-trigger turn** | Turn with no trigger → no retrieval call is made (verify via logs/metrics). |
| 9 | **Script doesn't override Core** | A retrieved script cannot change the strategic action (Doc 01 §15). |
| 10 | **Social Proof with no proof** | Core picks a different move or generic truthful language. No fabricated proof. |
| 11 | **Stale result** | Interruption mid-retrieval → late result discarded, no state overwrite. |
| 12 | **Timeout** | Slow retrieval → prompt still delivered in time with a truthful fallback. |
| 13 | **Live truth over stored fact** | Stored fact conflicts with what the prospect just said → live state wins. |
| 14 | **Team scope rejection** | Writing `owner_scope = team/brokerage/company` is rejected, and no query ever matches such records. |

---

## 7. Out of scope (do not build)

- Team, brokerage or company knowledge sharing, or any shared index
- Shared or cross-user Prospect Memory
- Retrieval on every turn, or "always attach top-K"
- Retrieval influencing the strategic decision (the Core decides first)
- Fixed trust deltas, verbatim-adherence metrics, or other Level D items from Doc 06
- Treating legacy marketing/sales claims (SOC 2, integrations, rollout times, stats) as retrievable facts

---

## 8. Open questions to confirm with the client

1. **Pipeline order:** confirm that Doc 01's "Context Assembly loads AI Training" is deferred until after Strategic Decision (per the client workflow diagram).
2. **"Prospect facts" in the retrieval box:** confirm this component serves AI Training only, with Prospect Memory supplied separately.
3. **Latency budget:** the docs require instrumentation but give no millisecond target for retrieval. Provide a number or approve a default.
4. **Deletion/permission propagation window:** Doc 02 §11 says "defined window" without defining it. Confirm the value.
5. **`verification_status` handling for user-supplied content:** confirm whether `user supplied` facts may be asserted in prompts (as "the user's own claim") or need additional validation for high-risk claim types (credentials, transaction results, guarantees).
6. **Embedding model / vector store:** the docs are vendor-neutral (Doc 06 says vendor stacks are not mandatory), so confirm the choice.

---

## 9. Source map

| Topic | Source |
|---|---|
| Retrieval triggers | Doc 02 §6 |
| Ranking | Doc 02 §7 |
| Conflict resolution | Doc 02 §8 |
| Hallucination guardrails | Doc 02 §9 |
| Structured context to the LLM | Doc 02 §10 |
| Edit/delete/versioning | Doc 02 §11 |
| Separation from Prospect Memory | Doc 02 §12 |
| Acceptance tests | Doc 02 §13 |
| Metadata schema | Doc 02 §5 (and §4 as ingestion dependency) |
| Context priority | Doc 01 §10 |
| Latency / stale-prompt cancellation | Doc 01 §13 (also §11 prompt contract, §3 pipeline) |
| Authority and supersession | Doc 06 §1, §4–§6, §8 |