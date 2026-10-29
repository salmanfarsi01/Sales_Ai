PITCHPROX | AI INTELLIGENCE SPECIFICATION
Momentum, Readiness, Objection Resolution & 
Conversion State Specification
How PitchProX determines whether the conversation is advancing and when to push for the meeting
Proposed Production Baseline | Developer Handoff | September 2026
For user-side real-estate calls, the primary success event is a confirmed live next step: property walkthrough, 
showing/tour, or appropriate in-person strategy meeting. Generic follow-up activity is not conversion success.
1. Purpose
This specification defines how the Core converts trust/emotion/behavior evidence into momentum, readiness, 
objection-resolution state, and meeting-close decisions.
2. Momentum Score
Evidence Family
Default Weight
Positive Example
Negative Example
Problem/goal clarity
15%
prospect defines what they 
want/what failed
conversation remains vague
Value recognition
15%
acknowledges relevance
dismisses value
Objection movement
20%
concern narrows/resolves
same concern repeats
Trust/engagement trend
15%
openness/questions rise
withdrawal/defensiveness 
rises
Decision structure clarity
10%
decision-makers/timing 
known
authority hidden/blocked
Future/operational behavior
15%
discusses 
dates/access/tour/participants
stays abstract
Commitment behavior
10%
accepts concrete next step
backs away
Store Momentum 0-100 plus trend: advancing, stable, stalling, regressing. Momentum measures directional 
progress, not friendliness.
3. Readiness Model
Dimension
0-100 Meaning
Evidence
Emotional readiness
Can engage without threat/overload 
dominating
trust, safety, tension trend
Logical readiness
Understands enough value/problem to 
consider next step
recurrence, contradiction, commitment 
language
questions answered, value recognition
Logistical readiness
Timing/access/participants can support 
next step
availability, access, location, decision 
maker
Decision confidence
Remaining objections sufficiently 
resolved
Overall readiness is a weighted mean with blocker caps. Example: if logistical readiness <30 because the 
decision-maker is absent, overall readiness cannot exceed a configurable cap such as 60 even when enthusiasm 
is high.
4. Objection Resolution State
State
Operational Definition
Unresolved
Concern remains active; repeats or behavior does not move
Clarified
Meaning/root understood but not solved
Partially Resolved
Not an objection to overcome: stop/representation/legal/privacy 
boundary
Accepts some reframe/evidence but same issue remains
Resolved
Concern stops recurring and behavior advances
Reactivated
Previously resolved concern returns
Boundary
5. Repetition Rule

Same semantic objection within a configurable window increases recurrence_count and 
unresolved_confidence.
If it reappears after the same strategy style, prohibit automatic reuse unless context materially changes.
Repeated boundary escalates boundary confidence and reduces/stops persuasive behavior.
Repeated meeting logistics positively increases commitment confidence.
6. Meeting / Walkthrough / Tour Close Gate
Gate
Default Requirement
Trust
Not necessarily high, but not actively collapsing
Engagement
Prospect responds to actual decision topic
Objection state
Major blocker resolved/partial enough to move
Value
Prospect has a reason the live meeting adds value
Decision structure
Correct participant/path identified
Logistics
Plausible appointment format exists
Boundary
7. Push Strength
No active stop/compliance/representation condition
State
Prompting Behavior
Low trust / high threat
Protect control, shorten, clarify, do not stack closes
Moderate trust + unresolved concern
Resolve one root objection, then ask
Rising trust + clear value
Use direct meeting ask
Directionally agreeable but vague
Use two-window choice and calendarize
Soft hesitation after agreement
Reduce one friction point and re-ask
Hard boundary / repeated clear no
8. Conversion Event Object
Respect, record, exit
Field
Example
conversion_type
walkthrough | tour | strategy_meeting | professional_meeting
Field
Example
status
start_at
location_or_format
participants
confirmation_confidence
source_turn_ids
blocking_items
followup_is_conversion
offered | tentative | confirmed | declined
timestamp
property address | office | video
prospect + spouse + agent
0.96
turn_88, turn_90
none
false unless a specific future appointment is calendarized
9. Conversation Score Alignment
Conversation Score evaluates quality, not merely whether an appointment happened. Conversion Momentum 
and call outcome separately record whether a walkthrough/tour/meeting was confirmed. This keeps business 
success and conversation quality distinct.
10. Acceptance Tests

A friendly prospect with no future behavior can have high engagement but only moderate readiness.
A confirmed walkthrough is conversion success even if the call was short.
“Send me something” does not count as success by itself.
Repeated commission objection remains unresolved until behavior changes; polite “okay” alone is 
insufficient.
High enthusiasm with absent decision-maker does not bypass decision-structure logic