# Behavioral Signal Engine — Implementation Plan
**PitchProX | Spec 07 Compliant | Acoustic Layer Deferred | Agile-Ready**

---

## 0. Scope Statement

**In scope for this build:** transcript-derived timing signals, LLM-based semantic signals, per-speaker baselines, multi-window evidence storage, confidence-gated downstream inference.

**Explicitly out of scope for now (per client confirmation):** acoustic/prosodic analysis (Spec 07 Section 2 "Acoustic features" row, and Section 4 in full). This is architected as a future plug-in point, not built or called.

**Non-negotiable rule carried through every phase:** no component may claim acoustic/tone evidence it did not actually compute. Confidence scores must reflect only the evidence sources actually used.

---

## 1. Architecture at a Glance

```
Ingestion & Normalization
        ↓
 ┌──────────────┬───────────────┐
 Deterministic         Semantic
 Timing Engine         Feature Engine
 (arithmetic)          (LLM/embeddings)
 ┌──────────────┴───────────────┐
        ↓
 Baseline & Change-Point Engine
        ↓
 Multi-Window Aggregation + Evidence Log
        ↓
 Downstream Inference (Trust/Emotion/Pacing/Engagement)
        ↓
 [ Acoustic extension slot — defined, not implemented ]
```

---

## 2. Phase-by-Phase Plan

### PHASE 0 — Input Contract Validation
**Epic:** Confirm what the transcription pipeline actually delivers before building anything on top of assumptions.

| | |
|---|---|
| **What it does** | Produces a written gap report against Spec 07 Section 2 |
| **How it works after done** | Every downstream phase reads from a documented, verified schema — no surprises mid-sprint |
| **Tools needed** | None — this is a vendor documentation + sample-payload review |
| **Inputs** | ASR vendor API docs, sample call payloads |
| **Outputs** | Gap report: word-level vs utterance-level timestamps, diarization availability, disfluency/filler flags, overlap detection availability |
| **Definition of Done** | Every row in Spec 07's Input Contract table marked Available / Partial / Missing, with a specific note for each |
| **Sprint** | Sprint 0 (Discovery — before Sprint 1 starts) |

---

### PHASE 1 — Ingestion & Normalization Layer
**Epic:** One clean internal data shape, regardless of ASR vendor quirks.

| | |
|---|---|
| **What it does** | Converts raw ASR output into a single internal schema (`NormalizedUtterance`) that every later phase consumes |
| **How it works after done** | If the ASR vendor ever changes, only this layer is touched — nothing downstream breaks |
| **Tools needed** | Standard backend service (Node/Python), a schema/validation library (e.g., Pydantic, Zod) |
| **Inputs** | Raw ASR + diarization + word-timestamp payloads |
| **Outputs** | `NormalizedUtterance { speaker_id, text, start_ms, end_ms, asr_confidence, words[] }` |
| **Definition of Done** | 100% of sample calls from Phase 0 parse into the normalized schema without data loss |
| **Sprint** | Sprint 1 |

---

### PHASE 2 — Deterministic Timing Feature Engine (pure arithmetic)
**Epic:** Measurable signals computed directly from timestamps — no model calls, always-on.

| Feature | Computation |
|---|---|
| Speech rate (WPM) | word count ÷ voiced duration, per speaker, rolling window |
| Pause duration | gap between turn-end and next turn-start; intra-turn if word-level data allows |
| Response latency | time from end of one speaker's turn to start of the other's |
| Interruption rate | overlapping starts / cut-off turns per 60s |
| Turn length | words + seconds per turn, trend over call |
| Question rate | raw count of interrogative utterances per prospect-minute |

| | |
|---|---|
| **How it works after done** | Updates continuously and cheaply on every new word/turn event — no latency cost, no LLM dependency |
| **Tools needed** | Plain backend logic — no ML libraries required. A stream-processing pattern (event-driven functions) is enough |
| **Inputs** | `NormalizedUtterance` stream from Phase 1 |
| **Outputs** | Timing feature values, tagged with window + timestamp |
| **Definition of Done** | All six features validated against 3-5 real calls with manually spot-checked values |
| **Sprint** | Sprint 1-2 |

---

### PHASE 3 — Semantic Feature Engine (LLM/embedding-based, turn-triggered)
**Epic:** Meaning-dependent signals that arithmetic can't capture — triggered at turn boundaries, not per word.

| Feature | What it needs |
|---|---|
| Objection/semantic recurrence | Embedding similarity across prospect utterances; cluster into recurring concerns |
| **Recurrence type classification** | Not just match/no-match — classify as: same objection repeated / concern after failed reframe / positive echo of rep wording / boundary repeated / scheduling-detail repeated |
| Question type | LLM classification using question + prior 1-2 turns: evaluation / clarifying / rhetorical / hostile |
| Specificity | LLM or NER extraction of named facts, dates, amounts, stakeholders |
| Future language | LLM-scored confidence: genuine commitment vs. hypothetical framing |
| Boundary markers | Classifier flags stop-contact/privacy/representation language — **highest priority, can override persuasion strategy** |
| **Agreement markers** | Semantic (not keyword) agreement detection — flagged as weaker evidence than behavioral commitment |
| **Filler/repair** | Self-correction/restart detection — only if Phase 0 confirms ASR exposes disfluencies; otherwise marked unavailable, not guessed |

| | |
|---|---|
| **How it works after done** | Fires once per completed turn, not per word — keeps LLM cost and latency controlled |
| **Tools needed** | LLM API (for classification prompts), an embedding model/service (for recurrence clustering), a small vector store or in-memory similarity index scoped per call |
| **Inputs** | Completed utterances + short rolling context window |
| **Outputs** | `recurrence_id`, `recurrence_type`, `question_type`, `specificity_score`, `future_language_score`, `boundary_score`, `agreement_score`, `filler_score (or "unavailable")` |
| **Definition of Done** | Each feature tested against labeled example calls; boundary detection tested against explicit stop-contact phrasing with zero false negatives |
| **Sprint** | Sprint 2-3 |

---

### PHASE 4 — Baseline & Change-Point Engine
**Epic:** Compare every value against "normal for this person," not a population average.

| | |
|---|---|
| **What it does** | Builds an intra-call baseline in the first 60-90s of clean speech; blends in a cross-call baseline if the prospect has prior calls |
| **How it works after done** | A naturally slow talker isn't flagged as disengaged; only real deviation from their own norm counts as a signal |
| **Tools needed** | A small stats layer (rolling mean/stddev, z-score calc) — no ML needed; a persistence store keyed by `prospect_id` for cross-call data |
| **Inputs** | Phase 2 + Phase 3 outputs, historical baseline record (if any) |
| **Outputs** | Normalized deviation (z-score) per feature; change-point events with timestamp, feature delta, context, confidence |
| **Definition of Done** | Acceptance test passes: a consistently slow speaker's normal pace does not trigger a "disengaged" deviation flag |
| **Sprint** | Sprint 3 |

---

### PHASE 5 — Multi-Window Aggregation & Evidence Log
**Epic:** Store signals at every required time horizon, not just the latest value.

| Window | Purpose |
|---|---|
| Current utterance | Immediate meaning, intent, question type, boundary language |
| Last 5-10s | Immediate pace, response timing, interruption/overlap |
| Last 20-30s | Local emotional-state movement, tension, objection response |
| Last 60-90s | Trust trajectory, momentum, disclosure trend, strategy effectiveness |
| Full call | Long-range recurrence pattern strength |

| | |
|---|---|
| **How it works after done** | The same raw signal (e.g., a pause) is queryable at multiple horizons simultaneously — a single pause reads as local noise, five pauses at the same objection over 10 minutes reads as a strong pattern |
| **Tools needed** | A time-series-friendly data store (e.g., a windowed table in Postgres/Timescale, or an in-memory ring buffer per call for hot data) |
| **Inputs** | All feature outputs from Phases 2-4 |
| **Outputs** | The Section 7 output feature object, per window, written to an **append-only** evidence log |
| **Definition of Done** | Any high-level score can be queried backward to its exact contributing timestamped features (Spec 07 acceptance test) |
| **Sprint** | Sprint 3-4 |

---

### PHASE 6 — Downstream Inference (Trust / Emotion / Pacing / Engagement)
**Epic:** Turn evidence into scores — honestly.

| | |
|---|---|
| **What it does** | Computes expressed-state scores from behavioral + semantic evidence only, each with an attached confidence value |
| **How it works after done** | Confidence is structurally incapable of claiming acoustic support, because the acoustic term simply isn't part of the calculation — not zeroed, absent |
| **Tools needed** | Scoring logic layer, ideally rule-weighted + tunable rather than a black-box model at v1 |
| **Inputs** | Aggregated evidence object from Phase 5 |
| **Outputs** | Trust/Emotion/Pacing/Engagement/Momentum/Readiness scores + confidence, feeding the Core Runtime's ConversationState |
| **Definition of Done** | Text-only test calls never produce a high-confidence "tone" claim (Spec 07 acceptance test #1) |
| **Sprint** | Sprint 4 |

---

### PHASE 7 — Acoustic Extension Slot (defined now, not implemented)
**Epic:** Leave a clean seam so audio evidence can be added later without a rewrite.

| | |
|---|---|
| **What it does** | Defines an interface (`AcousticEvidenceProvider`) and optional schema fields — implements nothing |
| **How it works after done** | When audio access + a provider (e.g., Hume) are approved later, it plugs into this interface as a parallel, non-blocking signal — Core logic doesn't change |
| **Tools needed** | Just interface/type definitions — no vendor SDK yet |
| **Inputs/Outputs** | N/A at this stage |
| **Definition of Done** | Code review confirms Phase 6's confidence function accepts an optional acoustic parameter without needing modification later |
| **Sprint** | Sprint 4 (small task, bundled with Phase 6) |

---

### PHASE 8 — Acceptance Testing Against Spec 07
**Epic:** Prove the build against the client's own written criteria, not just "it runs."

| Spec 07 Test | Validation Method |
|---|---|
| Text-only calls → no high-confidence acoustic claims | Automated test asserting field absence |
| Slowdown after pricing objection → stored as evidence, not auto-labeled fear | Manual review of change-point log entries |
| Repeated objection after failed reframe → unresolved confidence rises, strategy changes | Scripted test call with repeated objection |
| Naturally slow speaker not mislabeled disengaged | Test call with a slow-baseline speaker |
| Every high-level score traceable to timestamped evidence | Automated backward-trace test from score → log entry |

**Sprint:** Sprint 5 (hardening/QA sprint)

---

## 3. Suggested Sprint Mapping (2-week sprints)

| Sprint | Focus |
|---|---|
| Sprint 0 | Phase 0 — input contract validation, gap report to client |
| Sprint 1 | Phase 1 + start of Phase 2 |
| Sprint 2 | Finish Phase 2, start Phase 3 |
| Sprint 3 | Finish Phase 3, Phase 4, start Phase 5 |
| Sprint 4 | Finish Phase 5, Phase 6, Phase 7 |
| Sprint 5 | Phase 8 — acceptance testing, hardening, client demo |

---

## 4. Tech Stack Summary

| Layer | Typical tooling |
|---|---|
| Ingestion/Normalization | Backend service (Node/Python) + schema validation (Zod/Pydantic) |
| Deterministic timing | Plain functions, event-driven, no ML dependency |
| Semantic layer | LLM API (classification prompts) + embedding model/vector similarity |
| Baseline engine | Lightweight stats (rolling mean/stddev/z-score) |
| Storage | Time-series-friendly DB (Postgres/Timescale) + append-only event log |
| Downstream scoring | Rule-weighted scoring logic, versioned and tunable |
| Acoustic slot | Interface/type definitions only — no vendor SDK yet |

---

## 5. Cross-Cutting Rules (apply to every phase)

- **Compliance priority:** boundary detection (Phase 3) can override any in-progress strategy at any point — treat this as a hard interrupt, not just another score.
- **State versioning:** every write to ConversationState should be versioned so stale computations can never overwrite newer ones out of order.
- **Latency monitoring:** every phase should emit timing metrics so a slow component is visible immediately, not discovered later in production.
- **Evidence traceability:** nothing gets a confidence score without a logged source — this is what Phase 8's final test enforces.
