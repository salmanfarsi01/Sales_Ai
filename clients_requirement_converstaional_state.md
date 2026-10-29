And I would keep that test console permanently, even after launch (in an offline state if possible). It
could become incredibly useful when something strange happens on a real call: engineering can
replay the conversation and see exactly what Behavioral Signal believed happened turn by turn.
Unless, we can do that regardless of keeping this or not.
Hi, as you begin building ConversationState, I want to clarify a few important implementation
principles so the state becomes the reliable source of truth for the rest of the Al architecture.

ConversationState should answer one fundamental question at any point in the call: "Given
everything that has happened so far, what is currently true about this conversation?"

A few important requirements:

1. Current truth should supersede previous truth.
If a prospect says, "We've decided to stay." but later clarifies, "I would still move if I believed there
was a better strategy." the newer evidence should update the current state. Preserve the historical
statement, but do not treat contradictory old and new information as equally active.

2. Keep evidence separate from interpretation.
Behavioral Signal may provide evidence such as pace change, response latency, repetition, questions,
boundaries, or increased specificity. ConversationState can then contain the interpreted meaning
such as an unresolved concern, changing engagement, reduced readiness, or increasing
commitment. Please preserve the evidence supporting important interpretations.

3. Do not blindly update every state field after every utterance.
Only update fields materially affected by new evidence. A statement might change the decision-
maker structure without changing trust, or resolve an objection without materially changing emotion.

4. Important state changes must be explainable.
If Trust changes from 61->54, Readiness from 42->78, or an objection changes from active->resolved,
we should be able to inspect what evidence caused that change and when.

5. Objections need lifecycle/history.
Please distinguish active, partially resolved, resolved, reactivated, and superseded objections. If the
same underlying concern returns in different language, it should retain its history and strategies
already attempted rather than becoming an entirely new objection. This is also important to our deep
objection progression/branching logic.

6. Persistent facts should remain available until superseded.
If we learn that the spouse is a decision maker, the seller cannot meet mornings, they prefer email, or
they do not want frequent texts, those facts should not disappear simply because subsequent turns
discuss something else.

7. Keep Engagement, Momentum, Trust, and Readiness distinct.
Someone can be highly engaged because they are arguing with the agent while having very low
readiness. Someone can also speak very little while providing strong commitment signals. These
dimensions should not automatically move together.

8. Preserve uncertainty.
Where an interpretation is uncertain, retain an appropriate confidence/evidence value rather than
forcing certainty.

9. State versioning is important.
Every meaningful ConversationState mutation should increment state_version. Downstream retrieval,
strategic decisions, and generated teleprompts should know which state version they were
generated from. A stale asynchronous result should not overwrite or act on a materially newer state.

10. Keep the live state useful rather than enormous.
ConversationState should contain what the Core needs to understand and decide what happens
next. Full transcripts, detailed historical analytics, and other large records can remain elsewhere and
be referenced when needed rather than turning the live state into a giant conversation summary.

11. Please make ConversationState inspectable during development.
Ideally, for testing we should eventually be able to select a turn/event and see:

State Before -> New Evidence -> State Changes + Reasons -> State After

This will make it much easier for us to validate that the intelligence is interpreting the conversation
correctly before we connect everything to the Strategic Decision Engine.

Most importantly. ConversationState should represent the current interpreted truth of the
conversation, supported by evidence and history, rather than simply summarizing everything that has
been said.

Behavioral Signal tells us what happened. ConversationState should tell us what is currently true
because of everything that has happened so far. The Strategic Decision Engine can then determine
what PitchProX should do next.

Please use the existing specifications as the implementation contract and these points as
clarification of the intended behavior rather than a replacement for those specs.
