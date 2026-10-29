PITCHPROX | AI INTELLIGENCE SPECIFICATION
Core Developer Implementation Contract & Test 
Harness
The machine-level contract connecting Specs 01-10 to production code
Version 1.0 | Developer Handoff | September 2026
AUTHORITATIVE IMPLEMENTATION RULE: The LLM is the language generator, not the PitchProX decision 
engine. No implementation is acceptable in which the complete transcript and retrieved documents are simply 
passed to an LLM with instructions to decide what the salesperson should say next.
1. Purpose and Scope
Specs 01-10 define the PitchProX intelligence architecture. This document defines the production contract developers 
must implement between streaming call input, feature extraction, state mutation, strategic decisioning, retrieval, LLM 
generation, teleprompter delivery, observation, persistence, and testing. It does not replace Specs 01-10; it 
operationalizes them.

Production baseline input: speaker-separated transcript + word timestamps + utterance timestamps + call 
metadata/history.
Optional acoustic/prosodic features may enrich inference later but are not required for Core v1.
All numerical intelligence must retain confidence, evidence, scoring version, and timestamps.
All asynchronous results must be state-version aware so stale analysis cannot overwrite newer conversational truth.
Lead-type intelligence, AI Training, Playbooks, Calibration, memory, and compliance are context/services around 
one persistent Core.
2. Runtime Service Responsibilities
Component
Owns
Must Not Own
Transcript Intake
Normalize provider events; 
speaker/word/utterance timestamps; ASR 
confidence
Sales strategy
Feature Engine
Rate, pauses, latency, turn structure, 
repetition, semantic recurrence, rolling
window features
Final emotional truth or strategy
State Engine
Authoritative ConversationState and version; 
state history
Alter live-call record retrospectively without 
audit
Free-form LLM generation
Core Decision Engine
Interpret state; choose objective, strategic 
action, intervention timing
Invent facts or bypass compliance
Retrieval Service
Relevant verified AI 
Training/prospect/company knowledge
Decide strategy
Playbook Layer
Methodology weights, constraints, style
Replace Core
Calibration Layer
Prompt construction/delivery compatibility
Determine sales truth/strategy
LLM Gateway
Convert structured decision/context into 
exact words
Choose unconstrained strategy
Teleprompter
Present current valid prompt; 
karaoke/progression; cancel stale prompts
Reinterpret strategy
Observation Engine
Compare prompt vs rep speech; capture 
prospect reaction
Rewrite history
Persistence/Post-Call
Store state/event history and create 
downstream analysis inputs
3. Event-Driven Lifecycle
For every finalized or sufficiently stable utterance, the production loop should operate as follows:
1. Receive provider transcript event.
2. Normalize into TranscriptEvent and assign event_id.
3. Verify call/session is active and AI processing is permitted.
4. Update rolling behavioral features.
5. Run semantic interpretation for intent, meaning, objection, boundary, stage and other applicable signals.
6. Mutate ConversationState atomically and increment state_version.
7. Evaluate whether the new state requires strategic recomputation. Not every token requires a new prompt.
8. If relevant personalized knowledge is needed, retrieve only the required scoped facts.
9. Core creates StrategicDecision against a specific state_version.
10. LLM Gateway generates exact language from the structured decision contract.
11. Before display, verify decision.state_version is still current enough to be valid.
12. If stale, cancel/discard and recompute if the current moment still needs intervention.
13. Teleprompter displays Prompt with prompt_id and source_state_version.
14. Observation Engine compares rep speech with prompt strategic intent.
15. Prospect reaction becomes new evidence; repeat until call termination.
16. On call end, freeze the live state timeline and create the PostCallHandoff object.
4. Canonical Machine Objects
The exact field names may be adapted to the backend naming convention, but these objects and relationships are 
required.
4.1 TranscriptEvent
{
}
  "event_id": "evt_...",
  "call_id": "call_...",
  "provider": "twilio_or_equivalent",
  "speaker": "prospect|rep|unknown",
  "utterance_text": "...",
  "utterance_start_ms": 42120,
  "utterance_end_ms": 44780,
  "words": [{"text":"...","start_ms":42120,"end_ms":42380,"confidence":0.96}],
  "asr_confidence": 0.94,
  "is_final": true,
  "received_at": "ISO-8601"
4.2 ConversationState
{
  "call_id": "call_...",
  "state_version": 37,
  "updated_at": "ISO-8601",
  "conversation_stage": {"value":"objection","confidence":0.91},
  "primary_intent": {"value":"resistance","confidence":0.88},
  "underlying_meaning": {"hypothesis":"commission/value concern","confidence":0.76},
  "objection_state": {
    "objection_id":"commission_value_01",
    "root_categories":["financial"],
    "depth":"underlying",
    "recurrence_count":3,
    "resolution":"unresolved",
    "strategies_attempted":["validate","quantify"]
  },
  "trust_state": {"score":61,"trend":"rising","confidence":0.82},
  "emotion_state": {
    "valence":-18,"arousal":48,"social_safety":57,
    "top_states":[{"state":"guarded","probability":0.48},{"state":"curious","probability":0.31}],
    "confidence":0.74
  },
  "pacing_state": {
    "prospect_wpm":119,"prospect_wpm_delta_pct":-24.7,
    "rep_alignment_score":83,"conversation_tempo":"slowing","confidence":0.89
  },
  "engagement_state":{"score":67,"trend":"rising","confidence":0.85},
  "momentum_state":{"score":58,"trend":"advancing","confidence":0.78},
  "readiness_state":{
    "emotional":61,"logical":70,"logistical":42,"decision_confidence":55,"overall":57
  },
  "decision_structure":{"type":"spouse_partner","status":"participant_not_present"},
  "strategy_history": [],
  "prompt_state": {},
  "micro_commitments": [],
  "unresolved_items": [],
  "compliance_state": {"ai_processing_allowed":true}
}
4.3 StrategicDecision
{
  "decision_id":"dec_...",
  "call_id":"call_...",
  "source_state_version":37,
  "should_prompt":true,
  "strategic_objective":"resolve_value_concern_then_earn_walkthrough",
  "strategic_action":"reframe",
  "secondary_action":"question",
  "reason_codes":["OBJECTION_RECURRED","TRUST_RISING","LOGISTICS_NOT_READY"],
  "do_not_do":["repeat_quantification","premature_close","invent_social_proof"],
  "required_facts":[],
  "retrieval_needed":false,
  "urgency":"immediate",
  "max_prompt_words":24,
  "expires_on_state_change":true
}
4.4 Prompt
{
  "prompt_id":"prm_...",
  "decision_id":"dec_...",
  "source_state_version":37,
  "text":"...",
  "strategic_action":"reframe",
  "strategic_objective":"resolve_value_concern_then_earn_walkthrough",
  "generated_at":"ISO-8601",
  "displayed_at":"ISO-8601",
  "status":"generated|displayed|consumed|skipped|cancelled|stale",
  "expires_at":null
}
4.5 PromptOutcome
{
  "prompt_id":"prm_...",
  "rep_behavior":"exact|strategic_paraphrase|useful_deviation|harmful_deviation|skipped|interrupted",
  "adherence_score":0.84,
  "rep_turn_ids":["turn_..."],
  "prospect_reaction_turn_ids":["turn_..."],
  "observed_state_delta":{"trust":5,"engagement":7,"momentum":4},
  "confidence":0.79
}
4.6 ConversionEvent and ComplianceEvent
{
}
  "conversion_event": {
    "type":"walkthrough|tour|strategy_meeting|professional_meeting",
    "status":"offered|tentative|confirmed|declined",
    "start_at":"ISO-8601|null",
    "participants":[],
    "confirmation_confidence":0.96,
    "source_turn_ids":[]
  },
  "compliance_event": {
    "event_type":"disclosure_detected|consent_issue|stop_ai_prompted|ai_stopped|continued_after_disclosure_issue",
    "occurred_at":"ISO-8601",
    "source_turn_ids":[],
    "details":{}
  }
5. State Ownership, Mutation and Versioning

ConversationState is the single authoritative live state for a call.
Every accepted mutation increments state_version monotonically.
Every StrategicDecision, retrieval result, LLM request, and Prompt must reference the state_version that produced 
it.
A late result from version N must not overwrite or display against version N+k if the relevant conversational 
condition has changed.
Use idempotency keys for provider events so duplicate transcript delivery cannot double-count repetition, trust 
evidence, or conversion events.
Store state deltas/events in addition to the current snapshot so post-call intelligence can reconstruct why scores 
changed.
Compliance state is write-protected from ordinary strategy logic; compliance events may suppress or terminate AI 
processing.
6. Rolling Time-Horizon Contract
Horizon
Primary Use
Current utterance
Meaning, intent, question/objection/boundary detection
5-10 seconds
Objection recurrence, strategy history, decision structure, 
conversion, outcome
Immediate response latency, interruption, local pace
20-30 seconds
Expressed emotional movement, tension, short-term behavioral 
change
60-90 seconds
Trust, engagement and momentum trajectory
Full call
Developers should maintain rolling aggregates rather than reprocess the full transcript on every word. Full-history 
retrieval is used only when the current interpretation requires it.
7. LLM Input Contract
The LLM receives a bounded structured context assembled after Core strategy selection. Minimum logical sections:
Block
Content
Current strategic decision
Objective, action, reason codes, do-not-do rules
Block
Content
Current conversation truth
Lead/prospect context
Verified knowledge
Playbook constraints
Calibration constraints
Compliance constraints
Generation instruction
Recent relevant turns + state summary, not uncontrolled full-history 
dumping
Applicable lead type and prospect history
Only relevant AI Training/retrieval facts with provenance
Methodology and language rules
Prompt length, complexity, cadence, timing profile
Required wording/stop conditions
Return one exact natural response; do not add coaching commentary
7.1 LLM Output Contract
{
}
  "response_text":"Exact words the salesperson can say next.",
  "decision_id":"dec_...",
  "source_state_version":37

The LLM may improve wording but may not silently change the strategic action.
If required verified facts are missing, it must not fabricate them.
No multiple-choice response list in normal live mode unless a separately approved UX explicitly requests it.
No explanation, labels, or coaching text inside response_text.
Generation must obey the max prompt length and Calibration constraints.
8. Retrieval Contract

Retrieval is conditional, not mandatory on every turn.
Core declares required_facts/topic before retrieval.
Retrieval results include source_id, scope, verification_status, effective dates/version where applicable, and 
relevance score.
A retrieval result cannot change the Core objective by itself.
If two sources conflict and authority cannot resolve the fact, the prompt must avoid asserting the disputed fact.
Deleted or permission-revoked knowledge must not be returned.
9. Latency and Degradation Contract
The exact production SLOs should be measured against the current stack, but the following initial engineering budget is 
recommended as a tuning target rather than a guaranteed product claim:
Stage
Initial Target
Degradation Behavior
Transcript stabilization/event delivery
Measure provider-dependent latency
Do not wait for unnecessary finality if 
provider supplies stable partials safely
Feature/state update
<=100 ms target
Short safe prompt from current state beats a 
perfect late prompt
Use cached rolling features
Semantic/Core decision
<=150 ms target
Prefer current-state heuristic/compact model 
path
Conditional retrieval
<=150 ms target when invoked
Skip nonessential retrieval; never fabricate
LLM generation
<=350 ms target where stack permits
Request shorter response; use fast 
generation route
Teleprompter dispatch
<=50 ms target
Cancel stale prompt rather than display late
Decision-to-display aggregate
Aim for conversationally usable sub-second 
path; validate empirically
All timings must be logged separately. These are engineering starting targets, not external performance promises.
10. Stale Prompt Rules

Cancel a pending prompt if the prospect begins a materially new utterance that changes intent, objection, boundary, 
stage, or decision state.
Cancel if compliance_state changes to disallow AI processing.
Cancel if the rep has already begun a materially different response and the old prompt would no longer be usable.
Do not cancel merely because state_version incremented for an irrelevant telemetry update; use semantic 
invalidation reason codes.
Persist cancellation reason for latency and quality analysis.
11. Failure Modes and Safe Fallbacks
Failure
Required Behavior
Transcript provider degraded
Lower confidence; avoid aggressive inference; keep audit event
Speaker separation uncertain
Do not attribute interpersonal metrics confidently until resolved
Retrieval timeout
Generate truthful strategy without personalized fact if possible
LLM timeout
Do not show stale text; optionally use separately approved 
deterministic micro-prompt fallback
State service conflict
Reject stale write and recompute from latest version
Compliance uncertainty
Compliance/safety path wins
Low emotion/trust confidence
Use neutral strategy; do not overreact to weak inference
Provider duplicate event
Idempotently ignore duplicate
Call disconnect
12. Observability and Audit Requirements

Finalize available state and mark incomplete/abrupt end
Log event_id -> state_version -> decision_id -> prompt_id lineage.

Log per-stage latency and total decision-to-display latency.
Log score changes with evidence references and scoring version.
Log retrieval source IDs used in generation.
Log stale prompt cancellations and reason.
Log compliance events separately from sales outcome.
Log model/config versions for semantic analysis, scoring, Core rules, Playbook, Calibration, retrieval and LLM 
generation.
Post-call analysis must be able to reconstruct the major state/strategy timeline without guessing.
13. Deterministic Test Harness
Engineering must create a replay harness that accepts a timestamped, speaker-separated scripted call and replays events 
in sequence. The harness must capture state snapshots, strategic decisions, prompts, latency, cancellations, and final 
outcomes. Tests should evaluate strategic equivalence rather than exact generated wording unless wording is 
compliance-mandated.
13.1 Required Test Scenarios
#
Scenario
Must Demonstrate
1
FSBO commission objection repeats three 
times
Processing/uncertainty hypothesis; not 
deterministic fear label
Semantic recurrence; root=Financial; 
strategy changes; no canned repetition
2
Expired/withdrawn prospect starts guarded 
then discloses prior failure
Trust trend rises with evidence; prompt 
becomes more direct only after state change
3
Prospect says 'I need to think about it' after 
price discussion
Contextual diagnosis, not keyword lookup
4
Same phrase 'I need to think about it' with 
spouse absent
Decision-structure/logistics interpretation 
differs from Test 3
5
Prospect slows after sensitive disclosure
# Scenario Must Demonstrate
6 Prospect accelerates, asks detailed future 
questions
Engagement/momentum/readiness rise 
without assuming excitement from speed 
alone
7 Positive friendly call with no operational 
movement
Positive valence but limited conversion 
readiness
8 High engagement but unresolved major 
objection
Do not close blindly
9 Meeting logistics become specific Conversion event moves offered/tentative -> 
confirmed only with sufficient evidence
10 Prospect says 'send me something' Not counted as conversion success by itself
11 Rep paraphrases prompt effectively Strategic adherence recognized despite non
verbatim wording
12 Rep ignores prompt and damages trust Useful/harmful deviation logic and state 
reaction
13 New prospect utterance arrives while LLM 
response pending
Old prompt cancelled/recomputed
14 AI Training contains relevant verified proof 
point
Conditional retrieval used accurately
15 AI Training contains irrelevant similar text Retrieval does not contaminate prompt
16 Two uploaded documents conflict Avoid false certainty / authority rules 
applied
17 Premium Playbook active Method changes but Core truth/compliance 
preserved
18 Different Calibration profiles on same 
strategic state
Same strategic intent, different usable 
construction
19 Disclosure/consent issue occurs Compliance overrides conversion strategy 
and creates audit trail
20 Explicit do-not-contact/boundary Persuasion stops; boundary state recorded
21 Second call to same prospect Prior unresolved objection/commitment 
recalled
22 Novel objection appears once Stored as learning candidate, not immediate 
Core doctrine
23 Low ASR confidence / poor diarization Metric confidence falls; no false precision
24 Naturally slow speaker Baseline normalization prevents 
disengagement misclassification
25 Call ends after excellent conversation but no 
fit
Conversation quality can score well while 
conversion outcome remains no 
appointment
26 Confirmed property walkthrough Primary user-side conversion success 
recorded
27 Buyer call confirms showing/tour Buyer conversion path recorded correctly
28 Sensitive divorce/deceased context Lead-type restraint overrides generic push 
strength
29 Rep interrupts during prospect disclosure Pacing alignment declines; strategy may 
recommend space/silence
30 Retrieval/LLM exceeds useful moment System degrades/cancels rather than 
displaying a brilliant stale prompt
13.2 Test Assertion Structure
{
  "at_ms": 183000,
  "expected": {
    "objection.root_categories": ["financial"],
    "objection.resolution": "unresolved",
    "trust.trend": "stable|rising",
    "momentum.trend": "stalling",
    "strategic_action": ["clarify","reframe"],
    "should_prompt": true,
    "prohibited_actions": ["repeat_prior_strategy","premature_close"]
  },
  "tolerance": {
    "trust_score": 8,
    "engagement_score": 10
  }
}
Tests should assert state class, direction, strategy, prohibited behavior, and approximate score ranges/tolerances. Exact 
natural-language output should not be required except for mandated compliance wording.
14. Definition of Done for the Core

The full event loop operates continuously on a live or replayed timestamped call.
ConversationState persists and updates with versioned evidence.
Trust, emotion, pacing, engagement, momentum, readiness and objection state are computed from the approved 
scoring architecture with confidence.
Core selects strategic action before LLM language generation.
Lead type, prospect history, Playbook, AI Training and Calibration influence the correct layer without replacing the 
Core.
Repeated objections branch strategically rather than returning canned paraphrases.
The system can intentionally wait/silence instead of always prompting.
Stale prompts are cancelled reliably.
Rep behavior/adherence and prospect reaction feed back into state.
Confirmed walkthrough/tour/meeting conversion is distinguishable from generic follow-up.
Compliance can stop AI processing and remains auditable.
Post-call analysis can reconstruct major trust/emotion/strategy changes from stored evidence.
The 30-scenario replay suite passes agreed state/strategy assertions.
Latency is instrumented by stage and meets an empirically approved conversational usability threshold.
15. Handoff Order
Developers should receive the production requirements in this order:
1. 01 AI Core Runtime & State Engine
2. 02 AI Training & Retrieval
3. 03 Playbook Integration
4. 04 Calibration Score & Prompting Profile
5. 05 Memory & Learning
6. 06 Intelligence Source Hierarchy & Supersession Index
7. 07 Behavioral Signal Extraction & Feature Engineering
8. 08 Trust, Emotion, Pacing & Engagement Scoring
9. 09 Momentum, Readiness, Objection Resolution & Conversion State
10. 10 Confidence, Evaluation & Brain Completion
11. 11 Core Developer Implementation Contract & Test Harness
12. Supporting Intelligence: lead-type frameworks, Master Intelligence/Playbook materials, and internal prospect-type 
intelligence
16. Final Engineering Acceptance Rule
PitchProX is not complete merely because a prospect utterance can produce an LLM response. It is complete 
only when the system can demonstrate, through replayable evidence, that it maintained the conversation 
state, interpreted the prospect in context, selected the appropriate strategic action, generated a usable exact 
response, observed what the rep actually did, evaluated the prospect reaction, updated the state, protected 
compliance, and repeated the loop at live-call speed.
Source and Authority Basis
This implementation contract operationalizes the current PitchProX AI specifications 01-10 and the founder-approved 
Complete Concept. Where this document conflicts with a later formally approved Level A specification, the later 
approved specification controls. Historical prototype JSON, fixed trust deltas, pass/fail calibration logic, autonomous AI
caller language, Silent-7 hard caps, and unsupported claims remain superseded as defined by the Intelligence Source 
Hierarchy