Thanks for the updates. The reviewed scenarios now demonstrate spouse coordination, wife inclusion in the confirmed appointment, disciplined primary/secondary actions, and prompt suppression during agent speech.
Before final Core approval, please close the following essential items. These are corrections and verification of the current architecture.
The turn references below refer to the fee replay, sim_muy04bh0.
Item
Observed issue
Required behavior
1. Binding and snapshot consistency
The frontend accepts a source state belonging to another call when the report and decision match. At Turn 4, the V10-bound decision also reports an open gate and 82.9% readiness, while V10 says closed and 56.7%.
Validate call ownership across the report, source state, and decision. Current interpretation and Evidence Considered must use the same immutable source snapshot. Preserve historical strategy provenance separately.
2. Non-material turns and prompt repetition
Turn 5 is classified as non-material, but creates a fresh decision, repeats the confirmation sentence, and leaves carry-forward references empty.
Preserve the active strategy and its original provenance. Allow HOLD / no new prompt when nothing warrants another sentence.
3. Unresolved concerns after appointment agreement
Turns 6–7 record a persistent fee objection and a rejected explanation, but the decision keeps recommending appointment confirmation and does not reference the objection.
Protect the appointment while responding to the current concern. Use the attempted strategy and outcome when selecting the next action. Appointment agreement must not imply objection resolution. When the client explicitly defers discussion to the meeting, retain that concern for follow-up.
4. Appointment lifecycle
The confirmed event disappears at Turns 4–5, is recreated during a fee complaint at Turn 6, and “Thursday at 4 pm” is classified as a reschedule.
Preserve the appointment’s identity and original commitment evidence. Change its status only on relevant evidence. Clarifying an unspecified AM/PM should update the existing appointment rather than automatically create a reschedule.
5. Unmeasured signals
Trust is labeled unmeasured in the trace but described as “healthy” in the gate. Defaults also contribute to affirmative interpretations and confidence.
Distinguish defaults from measured evidence throughout gate evaluation and confidence handling. An appointment agreement can support the appointment without establishing trust, fee acceptance, or decision authority.
6. Pressure-field consistency
StrategicDecision separates pressure correctly, but ConversationState and summaries still use confirm_and_protect / resolve_then_ask as push-strength values.
Apply the existing separation consistently: action/posture describes strategy; push strength describes pressure.

For the remaining verification, please provide existing implementation references and relevant test results for:
Runtime isolation: cache scope, removal of the shared default_call fallback, explicit Prospect Memory loading, and immutable decision inputs. The A/B outputs show clean separation; provide the supporting backend references.
Memory ownership: same-user retrieval succeeds under policy; a different user calling the same prospect/phone cannot retrieve that history; simulations leave memory off unless explicitly enabled. Show ownership validation before retrieval returns data, plus the required user, prospect, source-call, timestamp, evidence-type, and event/turn provenance. Keep memory scoped to user_id → prospect_id.
Behavioral regressions: client-turn no-prompt/carry-forward, confidence changing the selected action, persistent/reactivated objections, and missing/mismatched decisions. Identify the 18-turn fixture used and include its actual runtime decision inputs.
Also confirm that the earlier sole-decision-maker statement cannot override the later requirement for spouse involvement.
Please return a concise status with references and test results, separating Core decision-rule changes from binding, state, retrieval, cache, and rendering changes. Reuse existing passing tests; add regression coverage for the failures above.
