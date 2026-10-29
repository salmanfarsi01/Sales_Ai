For Core Intelligence + Strategic Decision, please use the following files from the Authoritative Developer Specifications PDFs as the primary implementation references:
Primary specifications
01 — AI Core Runtime & State Engine Specification
 This is the main specification for the Core Intelligence Engine. Follow this for the runtime architecture, Core responsibilities, strategic action taxonomy, source precedence, state-to-decision flow, response construction flow, and continuous feedback loop.
09 — Momentum, Readiness, Objection Resolution & Conversion State Specification
 Use this for determining whether the conversation should advance, hold, clarify, recover, or move toward commitment based on Momentum, Readiness, objection resolution, blockers and conversion state.
10 — Confidence, Evaluation & Brain Completion Specification
 Use this for confidence handling, evidence requirements, explainability, cross-metric consistency and preventing uncertain interpretations from being treated as certain facts.
11 — Core Developer Implementation Contract & Test Harness
 This is extremely important. Use this as the implementation/acceptance contract connecting ConversationState → Strategic Decision → Retrieval → LLM → Teleprompter → Observation → State Update. It also defines the machine objects, state-version requirements, stale-result handling and required testing.
Please also reference these supporting specifications when their respective layer is involved:
02 — AI Training & Retrieval — when the Core determines external/user/company knowledge is required.
03 — Playbook Integration — methodology preference/strategic weighting. A Playbook influences the Core but does not replace its judgment.
04 — Calibration Score & Prompting Profile v3 — affects how the final strategy is presented to the salesperson, not the strategic truth itself.
05 — Memory & Learning — previous calls/prospect history and learned information.
06 — Intelligence Source Hierarchy & Supersession Index — use this whenever specifications or information appear to conflict. This determines authority and supersession.
07 — Behavioral Signal Extraction and 08 — Trust, Emotion, Pacing & Engagement are already upstream evidence sources feeding ConversationState. The Strategic Decision layer should consume their resulting state/evidence rather than rebuilding those engines.

How this connects to the ConversationState you just completed
Think of the separation this way:
ConversationState = What is true right now?
 Core Intelligence = What does that truth mean strategically?
 Strategic Decision = What should PitchProX do next?
 LLM = What exact words should the salesperson say to execute that decision?
The current ConversationState should therefore become the authoritative runtime input into the Core rather than being replaced by another analysis object.
The runtime path should be:
New prospect/rep utterance
 ↓
 Behavioral/semantic evidence
 ↓
 ConversationState updates + state_version increments
 ↓
 Core Intelligence reads the latest valid ConversationState
 ↓
 Determines whether the new evidence requires strategic action
 ↓
 Strategic Decision Engine selects the next strategic action
 ↓
 Conditional knowledge retrieval only if required
 ↓
 Playbook methodology constraints/preferences applied
 ↓
 Calibration delivery constraints applied
 ↓
 LLM converts the StrategicDecision into the exact word-for-word response
 ↓
 Teleprompter
 ↓
 Observe what the salesperson actually says
 ↓
 Observe prospect reaction
 ↓
 ConversationState updates again
 ↓
 Core reassesses.
This is a continuous loop.

Very important implementation rule
Please do not send the full transcript + ConversationState + retrieved documents to the LLM and ask the LLM “what should the salesperson say?”
That would bypass the PitchProX Core.
The Core/Strategic Decision layer must make the strategic decision before wording generation.
For example, ConversationState might contain:
Lead Type: Expired
Stage: Decision Resolution
Active Objection: Financial / partially resolved
Wife: decision-maker
Trust: improving
Engagement: high
Momentum: advancing
Appointment: confirmed Thursday at 3
Open Concerns: 1
Contact preference: no frequent texting
The Core should interpret that state and produce a machine-readable StrategicDecision, conceptually something like:
Primary Action: Acknowledge
 Secondary Action: De-Risk
 Objective: Protect confirmed appointment while reducing remaining financial uncertainty
 Push Strength: Low
 Question: Only if necessary
 Avoid: reopening resolved issues, unnecessary closing pressure, violating contact preference
 Knowledge Retrieval Required: False
 Source State Version: 184
Only after that decision exists should the LLM generate the actual sentence the salesperson sees.
The LLM might then produce:
“Perfect. Thursday at three works. I’ll keep everything simple between now and then, and when your wife is there we can walk through the numbers together so both of you can decide whether the approach actually makes sense.”
The wording can vary. The strategy should not be invented by the wording model.

StrategicDecision should be inspectable
Similar to the ConversationState replay UI you built, I want us to be able to test:
ConversationState → Evidence considered → Strategic interpretation → StrategicDecision → Why → Final generated prompt
At minimum, each StrategicDecision should be traceable to:
decision_id
source_state_version
primary strategic action
optional secondary action
strategic objective
push strength
what should be protected
what should be avoided
whether a question is appropriate
whether knowledge retrieval is required
relevant Playbook influence, if active
relevant Calibration influence on delivery
confidence
evidence/reasoning reference
You do not need to create another independent “truth” object. ConversationState remains the single authoritative truth for the call.
Also preserve the state_version relationship. If the Core makes a decision from State Version 184 but new material evidence creates Version 185 before the prompt is delivered, the old decision/prompt must be eligible for cancellation rather than being delivered against stale conversational reality.

Strategic actions
Please follow the strategic action taxonomy in Specification 01, including:
Wait/Silence, Acknowledge, Clarify, Validate, Mirror, Reframe, Educate, Quantify, Differentiate, De-Risk, Social Proof, Challenge, Future Pace, Question, Commitment/Close.
These are strategic actions, not canned scripts.
The Core may combine actions where appropriate, but it should determine the primary strategic objective first rather than stacking techniques indiscriminately.

One final architecture distinction
ConversationState should not decide what to say.
It maintains current truth.
Strategic Decision should not write the final teleprompter sentence.
It decides what needs to happen strategically.
The LLM should not independently determine the sales strategy.
It verbalizes the approved StrategicDecision using the permitted context, knowledge, Playbook and Calibration constraints.
That separation is critical to the PitchProX architecture.
Please implement against 01 + 09 + 10 + 11 as the primary set, using 02–08 as supporting specifications where applicable.
Once the first implementation is ready, I would like a replay/debug test similar to ConversationState so we can test whether PitchProX chooses the correct strategy across difficult conversations before we focus on final prompt wording.
Integration note: Please build Core Intelligence and Strategic Decision as downstream consumers of the existing ConversationState service rather than duplicating any of its logic. When ConversationState produces a new material state_version, that should be the primary trigger for strategic reevaluation. Not every transcript fragment needs to generate a new StrategicDecision or teleprompter prompt. If the state change is non-material, the current strategy may remain valid. If it is material, Core reevaluates. Every resulting StrategicDecision must reference the exact source_state_version it was created from. Before sending any generated prompt to the teleprompter, confirm that its source_state_version is still valid against the latest ConversationState. If newer material state has superseded it, cancel the stale decision/prompt and reevaluate from the newest state.



