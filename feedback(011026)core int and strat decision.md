Oct 1 Core Intelligence + Strategic Decision

Thanks for the detailed update. The current 18-turn replay supports the main temporal-state correction: turns 3–6 no longer show the final appointment, turn 10 remains CLOSED with DE_RISK, and only turn 18 records the Thursday-at-3 confirmation, OPEN gate, commitment 1.00, and confirm_and_protect. The saved Test A/Test B reports also support cross-call isolation.
Before final Core approval, please close the remaining verification points and confirm the structural behavior below. These are checks on the current architecture, not a request to redesign the Core.
Please confirm the following state and retrieval protections with implementation references and test results:
Complete decision binding: The replay frontend currently checks source_state_version. Please also validate the originating call_sid, turn, and source event. Matching versions alone are insufficient because multiple turns—and different calls—can share the same version number.
Runtime isolation: Confirm the cache scope, removal of the shared default_call fallback, explicit Prospect Memory loading, and use of one immutable snapshot for decision input and Evidence Considered.
Missing decisions: Preserve the Pending / No Decision Recorded behavior. A missing decision must never silently become a later or final-call decision.
Remaining state consistency: Include the wife in the confirmed event’s participants and reconcile the earlier sole_decision_maker assumption with her required involvement.
Evidence and wording accuracy: Unmeasured trust must not be presented as established healthy trust. Prompts must not claim an appointment is already on the agent’s calendar unless a successful calendar write is confirmed.
The structural checks are:
Separate Strategic Action, Strategic Posture, and Push Strength.
confirm_and_protect describes the strategy or posture. Push Strength should independently describe pressure level, such as None / Low / Moderate / High. Protecting an appointment should not imply additional sales pressure.
Do not automatically map decision-maker changes to DE_RISK.
Distinguish logistical coordination, relationship conflict, trust concerns, and actual decision risk before selecting an action. A request to include a spouse may simply require coordination.
Support WAIT / HOLD / NO NEW PROMPT.
A valid StrategicDecision may intentionally produce no teleprompter sentence. A state update must not automatically trigger another spoken prompt.
Keep Primary and Secondary Strategy disciplined.
Prefer one primary action and, when useful, one secondary action. Avoid stacking techniques without a clear purpose.
Make confidence influence the action.
Lower-confidence interpretations should favor clarification, holding, or lower-risk actions. Confidence must affect behavior, not merely appear as a displayed score. Distinguish measured signals from defaults or unavailable evidence.
Keep Evidence Considered relevant to the decision.
Include the facts, signals, constraints, and history that materially explain the selected action. Do not automatically list every available metric. Keep the evidence traceable to the decision’s source snapshot.
Preserve object ownership.
ConversationState stores current truth; StrategicDecision stores what to do next. Retain necessary source references and decision rationale without unnecessarily duplicating canonical state between the objects.
Preserve strategy when nothing material changes.
Small acknowledgements should not generate a new strategy or repeat the same prompt. When a strategy remains active, preserve its original provenance and explicitly record that it is being carried forward. Do not disguise it as a freshly generated decision for another turn.
Use strategy history in future decisions.
If an approach was attempted and the objection persists or reactivates, Core must consider the attempt and its outcome. Avoid blindly repeating the same approach without a reason supported by new evidence.
Keep Prospect Memory isolated to the individual user.
The current version has no team/shared-memory functionality. Please do not implement team or workspace memory sharing at this stage.
Prospect Memory must conceptually follow:
user_id → prospect_id → prospect memory
A matching prospect identity or phone number must never create universal PitchProX memory or bypass user ownership.
Situation
Required behavior
Same user + same prospect
Eligible prior memory may be retrieved under the configured memory policy.
New call + same user + same prospect
Start a fresh ConversationState; eligible prior memory may be loaded separately with its original provenance.
Different user + same prospect or phone number
Do not retrieve or expose the other user’s memory.
New simulation/test
Keep Prospect Memory off unless explicitly enabled for that test.
Any current call
Isolate ConversationState to its own call_sid.

For example, User A previously calls John Smith, who says he will not move until next year. Later, User B calls John, who says, “I already told you we’re not moving until next year.”
For User B, that statement is new evidence in the current conversation. PitchProX must not retrieve User A’s history or generate wording that implies User B remembers that earlier interaction.
Every Prospect Memory record must include, at minimum:
user_id
prospect_id
Source call_sid
Timestamp
Memory/evidence type
Source event/turn where applicable
Retrieval must validate the requesting user’s ownership before returning memory to ConversationState or Core Intelligence.
Please provide a concise Pass / Needs Correction response for each point, supported by implementation references and relevant tests. Include:
The canonical 18-turn replay and corresponding runtime decision inputs.
Back-to-back calls demonstrating clean state and cache isolation.
Two different users calling the same phone number, plus a same-user retrieval test.
Missing/mismatched decision handling, intentional no-prompt behavior, low-confidence decisions, and persistent/reactivated objections.
Please also identify any changes to Core decision rules separately from state binding, retrieval, cache isolation, and rendering changes. Final approval should be based on the demonstrated behavior and implementation.
