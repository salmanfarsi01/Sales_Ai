I found one important issue that needs to be corrected before I can validate the Core decisions.
There appears to be future-state leakage into Column 4 when reviewing earlier turns.
A few examples:
1. Turn 3
The selected turn is shown as a point-in-time transition around V4, but Column 4 displays:
RAW BEHAVIORAL DIMENSIONS (V38)
and Commitment is already shown as 1.00.
V38 is the final call state, so that information should not be available when evaluating Turn 3.
2. Turn 4
At this point the prospect has only said:
“We might look at moving sometime next year, nothing urgent yet.”
However, Column 4 shows language based on the future appointment:
“Perfect, I have Thursday at three confirmed on my calendar. I will see you both then.”
That information does not exist yet at Turn 4.
3. Turn 5
The selected-turn header correctly shows:
Stage: OBJECTION_HANDLING
Turn Gate: CLOSED
STRAT: VALIDATE
But the State After / Core area shows:
Meeting / Conversion Gate: OPEN
Explicit Commitment: Thursday At 3
Push Strength: confirm_and_protect
Those are final-call facts and should not exist at Turn 5.
I also saw the same issue around Turn 6.
Required correction
When a turn is selected, every field inside the selected-turn State After / Core Intelligence / Strategic Decision area must be calculated or rendered exclusively from the state snapshot available at that turn.
The final cumulative call summary at the top can continue showing the final V38 outcome. That is fine.
But the selected-turn inspection area must remain temporally isolated:
State Before V(t-1)
→ current turn evidence
→ mutations
→ State After V(t)
→ Core Intelligence based on that V(t) only
→ StrategicDecision based on that V(t) only
No facts, scores, objections, commitments, decision makers, conversion state, or generated language from future turns should appear in an earlier-turn decision.
Please also confirm that this is only a replay/UI snapshot issue and not happening in the runtime decision logic.
Each StrategicDecision should reference its actual:
source_state_version
and the decision/prompt should only have access to information that existed in that version.
For example:
- Turn 5 should still see the gate as CLOSED and choose the appropriate objection strategy.
- Turn 10 should respond to the wife/decision-structure revelation based only on information known through Turn 10.
- Only Turn 18 should know that Thursday at 3 is confirmed and select confirm_and_protect.

Once this temporal-state issue is corrected, please rerun the same replay. I can then evaluate
whether the actual strategic decisions across the earlier turns are correct.
Please do not redesign the Core architecture. This appears to be a state snapshot/version binding
issue in the replay or decision input path.

I reviewed the additional test and noticed something more fundamental that I want you to investigate before we continue evaluating the strategic decisions.
This was a different test conversation, but the Core Intelligence Evidence Considered section contains information from the previous 18-turn test.
For example, it shows:
Turn utterance (client):
"Thursday at 3 works, and my wife will be there."
That statement did not occur in this new test. It came from the previous test conversation.
The same block also shows:
- Readiness: 80%
- Momentum: advancing
- Active objections: general_hesitation
- Gate: OPEN
- Contact preferences: 1 active
These also appear inconsistent with the current test conversation.
Therefore, please investigate cross-call / stale-state contamination, not only turn replay rendering.
I need to make sure Core Intelligence is not combining evidence from:
- a previous simulated call,
- the latest global ConversationState,
- a cached CoreDecision,
- a hard-coded test fixture,
- or different state versions.
For every StrategicDecision, all Evidence Considered must come from one coherent ConversationState snapshot belonging to the same call_sid and exact source_state_version.
Conceptually:
call_sid + source_state_version + current_event
must identify the complete evidence package used by the Core.
Please verify specifically:
1. New-call initialization
Starting a new simulated call must create a clean ConversationState. Objections, commitment, contact preferences, gate state, decision structure, momentum/readiness history, etc. must not carry over unless they are intentionally loaded from Prospect Memory.
2. Cache isolation
Any cached Core Intelligence / StrategicDecision result must be keyed by the correct call and state version. A decision from Call A must never be returned for Call B.
3. No “latest state” fallback
If a selected turn does not have a valid CoreDecision, do not display the latest/final decision as a fallback. Show no decision / pending instead.
4. Evidence consistency
Every value inside Evidence Considered must originate from the same source_state_version. Please do not combine the current utterance with metrics or objections from another version.
5. Traceability
In the debug UI, please expose:
call_sid
decision_id
source_state_version
source_event_id / turn_id
for the StrategicDecision so we can immediately verify which exact state created it.

6. Please confirm whether the Evidence Considered content is currently live-generated or whether
any portion is still using hard-coded/mock test data.
Before changing strategy logic, please first identify why the previous test's "Thursday at 3 works, and
my wife will be there" statement appeared inside this completely different test.
Once that is isolated and fixed, please run two different test calls back-to-back. When Test B starts,
we should be able to confirm that none of Test A's objections, facts, preferences, commitment state,
metrics, or evidence appears in Test B. Can you also tell me what caused this? UI-only stale data or
actual Core input contamination?
