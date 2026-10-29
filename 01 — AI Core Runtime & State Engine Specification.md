PITCHPROX  |  AI INTELLIGENCE SPECIFICATION
AI Core Runtime & State Engine 
Specification
The authoritative real-time decision architecture for PitchProX
Version 1.0 | Developer Handoff | September 2026
Authoritative rule: PitchProX is one persistent real-time sales decision engine. Lead type, AI 
Training, Playbooks, Calibration, and live conversation signals are context layers feeding 
one strategic decision—not separate AIs.
1. Purpose
This document defines how the PitchProX AI Core operates during a live human-to-human sales 
call. It converts the sales psychology, objection logic, trust sequencing, emotional calibration, lead
type frameworks, pacing rules, and response-construction doctrine already developed for 
PitchProX into one runtime model.
The Core must continuously determine: what is happening, what it means, what changed, what the 
prospect is likely thinking or feeling, what the salesperson should strategically accomplish next, 
whether PitchProX should intervene now, and the exact word-for-word response the salesperson 
should deliver.
2. Non-Negotiable Architecture
 The human salesperson speaks to the prospect. PitchProX does not autonomously conduct the 
outbound sales call.
 The Core is not the vector database, the LLM, a Playbook, Calibration, sentiment analysis, a 
script library, or the teleprompter.
 The LLM generates language after the Core has determined the strategic action and assembled 
relevant constraints/context.
 The teleprompter is the output interface for the exact next words to say.
 Every prospect reaction becomes new evidence; the state is updated and the decision loop 
repeats.
 PitchProX Standard must remain a strong sales intelligence system even when no user uploads 
or premium Playbooks exist.
3. Runtime Pipeline
Stage
Required Behavior
1. Audio / Transcript Intake
2. Compliance Gate
3. Conversation Understanding
4. State Update
5. Context Assembly
6. Strategic Decision
7. Response Construction
8. Delivery Timing
9. Observation
Capture live call audio/transcript with speaker 
separation and timestamps.
Apply disclosure/consent/STOP AI rules before or 
alongside intelligence processing.
Interpret utterance meaning in context, not as 
isolated keywords.
Update stage, intent, objection, trust, emotion, 
momentum, readiness, unresolved issues, and 
history.
Load relevant lead type, prospect history, AI 
Training knowledge, active Playbook, and user 
Calibration profile.
Select the best next conversational action and 
whether to prompt now.
Generate concise, natural, exact word-for-word 
teleprompter language.
Present prompt only when usable within the live 
conversational moment.
Compare what the rep actually said with the 
strategic intent and observe the prospect reaction.
Update state and repeat until call end.
10. Feedback Loop
4. Canonical ConversationState Object
Engineering should maintain one authoritative call-state object that is updated throughout the 
conversation.
State Component
Contents
call_context
call_id, user_id, prospect_id, lead_type, 
active_playbook, timestamps
speaker_state
prospect speaking / rep speaking / silence / 
interruption / processing
conversation_stage
opening, engagement, discovery, problem 
identification, value, objection, resolution, 
commitment, closing, follow-up
primary_intent
question, concern, objection, resistance, curiosity, 
buying signal, rejection, negotiation, commitment, 
request
underlying_meaning
best current hypothesis about what the prospect 
actually needs or fears
objection_state
category, depth, severity, recurrence, 
resolved/partial/unresolved, strategies already 
attempted
emotion_state
trust_state
momentum_state
readiness_state
identity_state
decision_structure
strategy_history
prompt_state
micro_commitments
current emotional tags, confidence, direction/trend, 
evidence
current trust estimate, direction, confidence, major 
positive/negative events
advancing, neutral, stalling, regressing; supporting 
signals
not ready, exploratory, receptive, evaluating, 
commitment-ready
relevant identity narrative, role, status/relationship 
dynamics
single decision maker, spouse/partner, 
assistant/gatekeeper, manager/broker hierarchy, 
group
strategic moves already used and their observed 
result
last prompt, current prompt, shown_at, 
consumed/skipped/modified
questions accepted, information shared, 
meetings/trials/follow-up accepted, other 
commitments
questions, concerns, logistical blockers, promised 
follow-up
unresolved_items
5. Core Interpretation Rules
 No statement is interpreted from keywords alone. Meaning is contextual.
 The same words may imply different needs depending on lead type, prior dialogue, tone, trust, 
stage, and decision structure.
 Positive sentiment is not automatically conversion readiness. Curiosity, excitement, optimism, 
trust, playfulness, and eagerness require different handling.
 An objection is a stateful event, not a lookup key. The Core must track whether it is surface
level, underlying, repeated, partially resolved, or symptomatic of another concern.
 A good strategic action at one stage can be harmful at another. The Core must use stage and 
readiness before closing or escalating.
 Silence, hesitation, interruptions, pace changes, and contradiction between words and 
emotional signals are meaningful evidence when technically available with sufficient 
confidence.
6. Objection Depth Engine
The intended '25-level deep objection logic' means progressive depth and branching—not 25 
canned answers. Each repeated resistance should be interpreted against prior attempts, emotional 
movement, unresolved concerns, and new information.
Objection State
Core Question
Example Strategic Direction
Surface objection
What concern is being stated?
Validate / clarify
Underlying concern
What is the objection protecting?
Mirror value / reframe
First pushback
Did the first strategy miss the 
concern?
Change angle
Repeated resistance
Is trust falling, is identity 
threatened, or is a new blocker 
present?
De-risk / quantify / challenge 
selectively
Partial resolution
What remains unresolved?
Narrow the issue
Resolved
7. Trust Engine
Has the prospect behaviorally 
moved on?
Advance without re-answering
Trust is a continuous inferred state used to guide strategy. Do not implement fixed rules such as 
'empathy = +40 trust.' Historical fixed deltas are prototype examples only.
 Inputs may include openness, disclosure depth, responsiveness, interruption patterns, 
contradiction, defensiveness, acknowledgment, question quality, tone/sentiment trends, 
successful objection resolution, transparency, and rep behavior.
 Store trust estimate, confidence, trend, and the evidence that caused meaningful movement.
 Trust should influence prompt intensity, questioning depth, challenge level, closing readiness, 
and pacing.
 Trust Gain is a downstream metric derived from the assessed start-to-end change, not a 
manually awarded point total.
8. Emotion & Tension Engine
The Core should support both negative and positive tension. Negative states may include Guarded, 
Overwhelmed, Burned, Detached, Annoyed, Doubtful, or anxious patterns. Positive states may 
include Curious, Engaged, Hopeful, Confident, Excited, Trusting, Playful, or Eager.
 Emotion labels should affect strategy; they are not decorative analytics.
 Use confidence thresholds. If confidence is low, avoid overreacting to an uncertain emotional 
inference.
 Positive energy should be stabilized before advancing if logistical or decision blockers remain.
 Sensitive lead types (e.g., deceased-property/probate contexts) require additional restraint 
regardless of generic positive momentum.
9. Strategic Action Taxonomy
Action
When the Core Selects It
Wait / Silence
Prospect is still processing, speaking, or a pause is 
strategically useful.
Acknowledge
Recognition is needed before movement.
Clarify
Surface statement is ambiguous or underlying 
concern is unknown.
Validate
Concern/identity needs recognition without 
conceding the strategic objective.
Mirror
Readiness, trust, value, and decision structure 
support a next step.
Reflect logic, emotion, or identity to establish 
understanding.
Reframe
Change the interpretation of the concern or value 
equation.
Educate
A knowledge gap is blocking progress.
Quantify
Financial/value concern benefits from concrete 
comparison.
Differentiate
Prospect is comparing alternatives or sees 
PitchProX/agent as interchangeable.
De-Risk
Fear, implementation, commitment, reputation, or 
downside blocks action.
Social Proof
Credible evidence is relevant and available; never 
fabricate.
Challenge
Trust/readiness support a respectful pattern break.
Future Pace
Help prospect imagine a credible future state 
without manipulation.
Question
Discovery is more valuable than presenting.
Commitment / Close
10. Context Layer Priority
When instructions conflict, use this priority order:
1. Safety, legal/compliance stop conditions, and system-level prohibited behavior.
2. PitchProX Core strategic integrity and current conversation truth.
3. Live conversation state, prospect history, and applicable lead/prospect type.
4. Active Playbook methodology.
5. Relevant AI Training / user-company knowledge.
6. Calibration and user delivery preferences.
7. Language generation preferences and stylistic polish.
11. Prompt Generation Contract
 Output is the exact word-for-word language the rep can say next.
 Do not output coaching labels such as 'build trust' or 'ask a discovery question' in place of the 
actual line.
 Prompt length must respect live usability, user calibration, conversation pace, and UI limits.
 Do not repeat an answered objection or strategy simply because a keyword reappears.
 Do not invent product facts, statistics, credentials, results, market facts, or user-specific claims.
 If a necessary fact is unavailable, choose a truthful strategic response that does not require 
fabrication.
 If the prospect interrupts or the state changes before the prompt is delivered, stale prompts 
must be canceled or recomputed.
12. Rep Behavior & Prompt Adherence
Prompt Adherence should measure whether the rep preserved the strategic intent of the prompt—
not merely verbatim repetition. Track exact use, strategic paraphrase, useful deviation, harmful 
deviation, skip, and interruption where possible.
13. Latency Requirements
 Instrument latency separately for audio capture, transcription, state analysis, retrieval, strategic 
decision, LLM generation, and teleprompter delivery.
 Stale prompt cancellation is required. A late response to an old conversational moment must 
not overwrite a newer state.
 The Core should prefer a shorter timely response over a longer response that arrives too late.
14. End-of-Call Handoff
At hang-up, persist the full state history needed by Post-Call Intelligence: transcript, timestamps, 
prompt history, actual rep responses, trust/emotion trajectory, objections and resolution state, 
strategic actions, micro-commitments, outcome, follow-up obligations, and compliance events.
15. Acceptance Tests
 A new user with no AI Training or premium Playbook receives high-quality PitchProX Standard 
prompts.
 The same objection spoken in two different contexts can produce different strategic responses.
 A repeated objection does not trigger the same canned answer.
 The system can wait instead of always generating text.
 Positive sentiment does not automatically cause a close.
 A Playbook can change methodology without erasing Core safety/strategic reasoning.
 An uploaded script does not automatically override the Core simply because retrieval found it.
 Calibration changes prompt usability/style but does not decide the sales strategy.
 Prompt Adherence recognizes strategic paraphrase.
 Post-call analysis can reconstruct why a major trust or strategy shift occurred.
Source Basis
This specification consolidates and supersedes relevant concepts from the following PitchProX 
source materials. Historical examples are treated as examples unless explicitly restated as current 
requirements in this document.







AI Masterplaybook Module.pdf
PitchProX AI – Core Operating Manual.pdf
AI Masterplaybook – Volumes II, III, VI, VII, IX, XI
PitchProX AI – Onboarding Framework (lead-type editions)
Sales Team prospect-type training modules
Json for Sample PitchProX Call Report.pdf
Json.pdf (Master Cache prototype)