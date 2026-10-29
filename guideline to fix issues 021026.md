the client explicitly approved everything from the last round (Turns 3-6/10/18, Test A/B). What follows now is two different kinds of asks: (1) harder proof of what you already built (they want deeper verification, not new fixes), and (2) genuinely new behavioral refinements to the Core decision model and a full Prospect Memory architecture (new work, but scoped — they explicitly say "not a redesign").

Here's the translation into an ordered, agile plan — each point restated simply, why it matters, what to build/check, and a done-condition.

Group 1: Deeper proof of what's already built (fast — mostly verification, little new code)

Point 1 — Decision binding needs more than version matching
Client's concern, in plain terms: "You check that the version number matches. But version numbers aren't unique across turns or calls — two different turns, or two different calls, could both be 'V4'. Matching version alone isn't proof you're looking at the right decision."
Fix: Every decision lookup/display must validate all four keys together: call_sid + turn_id + source_state_version + source_event_id — not version alone.
Done when: A test constructs two decisions with the same source_state_version but different call_sid/turn_id, and confirms the UI/lookup never confuses them.

Point 2 — Runtime isolation, re-confirmed as a checklist
Client's concern: "Prove, with references, that: cache scope is correct, the default_call fallback is really gone, Prospect Memory only loads when asked, and evidence comes from one frozen snapshot."
Fix: This is mostly already done (Points 2/3/4/5 from the prior round) — just needs to be re-stated as a Pass/Needs-Correction table with file + line + test name for each sub-item, not prose.
Done when: You can produce a 4-row table: Cache scope → Pass (core_decision_manager.py, StrategicDecisionCache, test X) / Singleton fallback removed → Pass (prompt_delivery_service.py, test Y) / etc.

Point 3 — Missing decisions stay missing
Client's concern: Same as before — already built (Pending / No Decision Recorded). Just needs explicit re-confirmation in this round's report.
Done when: Cite the existing test (test_point4_pending_state_rendered_when_no_decision_recorded) again in this round's response.

Point 4 — State consistency: the wife is missing as a participant, and the decision-maker fact is now contradictory
Client's concern, in plain terms: Turn 2 said "I'm the sole decision maker." Turn 10 revealed "my wife needs to be involved." Turn 18 confirms she'll attend. But does the final confirmed event's participants list actually include her? And does the system still claim "sole_decision_maker" as true, even though the client later said someone else must be involved? That's a logical contradiction sitting in the data.
Fix:

Add the spouse to conversion_event.participants when she's confirmed attending (Turn 18).
Reconcile decision_structure.primary_decision_maker — either supersede "sole_decision_maker" with something like "sole decision maker, spouse required for final approval," or add a co_decision_required: true flag, rather than leaving two contradictory facts both marked active.
Done when: Turn 18's final state shows the wife in participants, and the decision-maker field no longer flatly contradicts the "wife must be part of this" fact from Turn 10.

Point 5 — Evidence/wording must not overstate certainty
Client's concern, in plain terms: Two specific wording problems: (a) "Trust: 50%" with no real signal yet is a default, not a measured fact — don't phrase it as if trust is "established" or "healthy." (b) The fallback prompt says "I have that confirmed on my calendar" — but has anything actually been written to a real calendar? If not, that's a false claim.
Fix:

Tag default/neutral values distinctly from measured ones in Evidence Considered (e.g., "Trust: 50% (default, unmeasured)" vs "Trust: 65% (measured, 3 signals)").
Change calendar-confirmation language to something that doesn't assert a real calendar write unless one actually happened — e.g., "I've noted Thursday at 3" instead of "I have that confirmed on my calendar," unless/until there's a real calendar integration.
Done when: Grep confirms no prompt text claims a calendar action that didn't happen; Evidence Considered visibly distinguishes default vs. measured values.
Group 2: Decision-model structural refinements (new work, Core logic — but client explicitly says this is refinement, not redesign)

Point 6 — Split Action / Posture / Push Strength into three independent fields
Client's concern: confirm_and_protect is currently doing double duty as both "what to do" and "how hard to push." Those are different questions — protecting a confirmed appointment doesn't mean you should also be selling harder.
Fix: Three separate fields on StrategicDecision: primary_action (what), strategic_posture (e.g., protect, advance, defend), push_strength (none/low/moderate/high, independent of posture).
Done when: A test confirms confirm_and_protect-style decisions can have push_strength: none or low, not hardcoded high.

Point 7 — Stop auto-mapping "decision-maker changed" to DE_RISK
Client's concern: Turn 10 (wife needs to be involved) got mapped straight to DE_RISK. But "I need to loop someone in" isn't inherently risky — it might just be logistics. Treating every decision-structure change as risk is too blunt.
Fix: Before selecting an action, classify why the decision-maker info changed: pure logistics/coordination vs. relationship conflict vs. trust concern vs. genuine deal risk. Only the last should route toward something risk-flavored; logistics should route toward a coordination action.
Done when: Turn 10's own scenario (clearly logistics — "happy to loop her in") selects a coordination-flavored action, and you have a second constructed test case that is genuine risk (e.g., "my wife thinks we're being scammed") and confirms that one still gets flagged appropriately.

Point 8 — Allow WAIT / HOLD / no prompt at all
Client's concern: Right now, does every state change force a new spoken line? It shouldn't — sometimes the correct move is to say nothing new.
Fix: Add a legitimate "no new prompt" outcome to the decision model — a StrategicDecision can validly carry no teleprompter text.
Done when: Construct a turn where nothing material changed (e.g., a filler acknowledgment) and confirm the system can output "no new prompt" rather than always generating one.

Point 9 — Discipline on primary/secondary strategy stacking
Client's concern: Don't pile on techniques. One primary action, at most one secondary, with a clear reason for the secondary.
Fix: Audit current decisions for cases where more than 2 actions stack without justification; constrain the output model/logic accordingly.
Done when: Spot-check the 18-turn benchmark — no turn should show 3+ stacked strategies without an explicit reason field justifying each.

Point 10 — Confidence must change behavior, not just be a displayed number
Client's concern: If the system isn't sure, it should act more cautiously (clarify, hold) — not confidently push forward while just showing a low confidence score for show.
Fix: Add a rule: below some confidence threshold, the action selection logic is constrained to lower-risk options (CLARIFY, QUESTION, HOLD) rather than commitment/close-style actions.
Done when: A test that artificially lowers confidence on an otherwise "confirm" scenario shows the action downgrade to something more cautious.

Point 11 — Evidence Considered should be relevant, not exhaustive
Client's concern: Don't dump every available metric into the evidence block — only the facts that actually explain why this specific action was chosen.
Fix: Filter the evidence-assembly step to include only fields materially tied to the decision's reason codes.
Done when: Compare evidence blocks before/after — irrelevant always-included metrics (e.g., unrelated contact preferences) drop out when they didn't influence the decision.

Point 12 — Don't duplicate canonical state between ConversationState and StrategicDecision
Client's concern: ConversationState is the one source of truth for "what is true right now." StrategicDecision should reference it, not copy/duplicate the same facts into its own fields.
Fix: Audit StrategicDecision for fields that just mirror ConversationState data verbatim; replace with references (IDs/pointers) plus the decision's own rationale, rather than copied values.
Done when: A change to ConversationState after a decision was made doesn't leave two different copies of the same fact silently diverging.

Point 13 — Preserve strategy across turns when nothing material changed
Client's concern: A small "okay" shouldn't spawn a brand-new strategy or repeat the same prompt as if freshly decided. If the active strategy is still valid, say so explicitly — "carried forward from Turn N" — don't disguise it as new.
Fix: Add a carried_forward_from_decision_id field (or similar) so a continued strategy is marked as continuation, not regeneration.
Done when: A filler-acknowledgment turn shows carried_forward_from: dec_xxx rather than a fresh decision with duplicate reasoning.

Point 14 — Use strategy history to avoid repeating failed approaches
Client's concern: If a reframe was already tried and the objection came back, don't blindly try the exact same reframe again without new evidence justifying it.
Fix: Before selecting a strategy for a reactivated objection, check attempted_strategies/strategy_outcomes and avoid re-selecting one already marked ineffective, unless something new justifies retrying it.
Done when: Construct a reactivated-objection scenario where the first strategy failed; confirm the second attempt selects a different approach, with the earlier attempt and its outcome referenced in the rationale.

Group 3: Prospect Memory architecture (biggest new piece — do this last, it's genuinely new scope)

Point 15 — Memory hierarchy and ownership
Client's concern, in plain terms: Right now, Prospect Memory seems keyed just by prospect_id. That's not enough — if two different salespeople (users) happen to call someone with the same phone number, User B should never see User A's notes about that prospect. Memory must be scoped user_id → prospect_id → memory, never shared across users.
Fix: Add user_id as a mandatory, first-class key on every Prospect Memory record (not optional). Retrieval must always filter by the requesting user's user_id first.
Done when: Test: User A calls +1-555-0100, records "not moving until next year." User B calls the same number. Confirm User B's call retrieves zero memory from User A — the client's own named example (two users, "John Smith," same statement) should produce no cross-user memory bleed.

Point 16 — The required record schema
Fix: Every memory record must store: user_id, prospect_id, source_call_sid, timestamp, memory/evidence type, source_event_id/turn (where applicable).
Done when: Inspect a saved memory record and confirm all six fields are present.

Point 17 — The five required situations, each with its own test
Build and test each row of the client's table explicitly:

Same user + same prospect → eligible memory can load (under policy).
New call, same user, same prospect → fresh ConversationState, but eligible memory loads separately, tagged with its original provenance (not merged silently into current-call facts).
Different user, same prospect/phone → zero retrieval, zero exposure.
New simulation/test → memory off by default unless explicitly enabled.
Any call → ConversationState stays isolated to its own call_sid regardless of memory policy.
Done when: Five passing tests, one per row, named to match.

Point 18 — No team/shared memory (explicitly, don't build this)
Fix: Confirm (and if necessary, guard against) there being no code path that shares memory across users/teams. This is a "don't build it" instruction — just make sure nothing currently does it, even accidentally via a shared prospect_id lookup with no user_id filter.
Done when: Grep/audit confirms no memory retrieval path skips the user_id filter.

Final deliverable (Point 19 — the consolidated report)

The client asked for a very specific report shape at the end — build it last, once everything above passes:

Canonical 18-turn replay plus the runtime decision inputs behind each turn (not just outputs — they want to see what went into each decision too).
Back-to-back call isolation demo (you have this — re-include it).
New: two different users calling the same phone number test, plus a same-user-returning-prospect retrieval test.
Missing/mismatched decision handling, intentional no-prompt behavior, low-confidence downgrade, and persistent/reactivated-objection handling — each with its own test.
A Pass / Needs Correction table, one row per point above, each row citing file + test name.
Explicitly separate: which changes were "Core decision rule" changes (Group 2) versus "state binding / retrieval / cache / rendering" changes (Group 1) — the client wants these categorized separately, not blended into one list.