PITCHPROX | AI INTELLIGENCE SPECIFICATION
Confidence, Evaluation & Brain Completion 
Specification
How developers validate numerical intelligence and prevent false precision
Proposed Production Baseline | Developer Handoff | September 2026
The biggest remaining risk is false certainty. This document makes every inference testable, versioned, 
explainable, and tunable against real PitchProX call data.
1. Missing-Brain Completion Map
Area
Status After Specs 07-10
Raw transcript/audio to measurable features
Completed by Spec 07
Trust/emotion/pacing/engagement numerical outputs
Completed by Spec 08
Momentum/readiness/objection/meeting conversion
Completed by Spec 09
Confidence and explainability
Defined here
Score calibration against real calls
Defined here
Ground-truth labeling process
Defined here
Threshold/version governance
Defined here
Drift monitoring / rollback
Defined here
Transcript-only fallback
Defined here
Cross-metric consistency tests
2. No False Precision Rule

Defined here
No emotional or trust score is a direct measurement of inner mental state. It is a model inference from 
observable evidence.



Every score must store confidence and evidence.
When evidence is insufficient, return unknown/low-confidence instead of a fabricated number.
UI may display a rounded score while backend retains confidence and feature provenance.
3. Confirmed Production Input Validation
Before scoring, validate that each stream provides stable speaker IDs, utterance start/end timestamps, word 
timestamps where available, ASR confidence, and monotonically ordered events. Missing or degraded 
timestamps must lower confidence for pacing/latency features but should not disable semantic analysis. 
Provider-specific fields from Twilio or another vendor must be normalized into a provider-neutral internal 
event schema.
4. Ground-Truth Labeling Program
Label Family
Reviewer Task
Trust events
Mark timestamps where openness/defensiveness changes and 
Label Family Reviewer Task
why
Emotion Mark expressed state(s) + confidence; allow 
multi-label/unknown
Pacing Mark rushed, matched, slow, interruptive, appropriate silence
Objection Mark root category, recurrence, resolution state
Momentum Mark advancing/stalling/regressing transitions
Readiness Rate four readiness dimensions
Conversion Mark offered/tentative/confirmed meeting/tour/walkthrough 
and timestamp
Each selected call should be independently reviewed by at least two trained reviewers using the same rubric; 
disagreements become measurable annotation uncertainty.
5. Calibration Method
 Start with configurable heuristic weights from Specs 08-09.
 Compare outputs with reviewer labels using correlation, classification accuracy, calibration curves, and 
error analysis.
 Tune thresholds on a training set; lock them before evaluating a holdout set.
 Do not optimize solely for appointment conversion; include trust preservation and boundary compliance.
 Version every formula/threshold set, e.g. behavioral_scoring_v1.0.
6. Confidence Formula Inputs
Confidence Input Effect
ASR confidence Clean transcription raises confidence
Speaker diarization quality Poor separation lowers interpersonal metrics
Sample duration Too little speech lowers confidence
Independent cue agreement Text + timing + acoustic agreement raises confidence
Cue contradiction Lowers confidence
Baseline availability Personal baseline raises change-detection confidence
Context clarity Known objection/topic/lead type raises interpretability
Model agreement Optional ensemble agreement can raise confidence
7. Explainability Record
Every meaningful score shift should store: metric, prior score, new score, delta, timestamp, source turns, feature 
deltas, semantic reason, confidence, and scoring version.
8. Cross-Metric Consistency Rules
Contradiction Required Handling
Trust high but repeated boundary language Boundary wins; lower trust confidence
Positive valence but momentum falling Do not treat friendliness as progress
Engagement high but objection unresolved Continue strategy; do not close blindly
Pacing alignment high but prospect annoyed Good pacing does not erase negative content
Readiness high but authority/logistics blocked Apply blocker cap
Acoustic calm but negative transcript Mark mixed evidence and reduce emotion confidence
9. Drift Monitoring




Monitor score distributions by lead type, language, calibration band, device/audio quality, and call stage.
Alert if vendor/model changes shift average trust/emotion/readiness distributions.
Keep rollback capability for scoring configuration.
Do not let learned patterns silently change numerical scoring without versioning/evaluation.
10. Minimum Developer Dashboards / Logs









Feature extraction latency and missing-feature rate
Score confidence distribution
Top reasons for trust increases/decreases
Emotion unknown/confusion rate
Pacing mismatch frequency
Objection recurrence and strategy-miss rate
Meeting close offered to confirmed funnel
Boundary/compliance override count
Scoring version adoption and drift alerts
11. Final Integration Rule
PitchProX Core decides strategy from state. Specs 07-10 define how raw conversational evidence becomes 
trustworthy state. No score becomes an independent brain or overrides current conversation truth, 
compliance, lead-type safeguards, Playbook constraints, AI Training facts, or Calibration delivery rules.
11. Acceptance Tests

Timestamped-transcript-only calls can produce production Trust, Engagement, Pacing, Momentum, 
Readiness, objection-state, and expressed emotional-state outputs with confidence based on available 
evidence. If optional acoustic/prosodic evidence is later present, it may raise or lower confidence for vocal
tone-dependent hypotheses, but its absence is not an implementation failure.
A score change can be traced to timestamped evidence.
A model/vendor update cannot silently change score behavior without a new scoring version.
Two human reviewers can label a sample call with the same rubric and disagreements are measurable.
Low-confidence scores do not trigger aggressive strategy changes.
Confirmed meeting success is stored separately from overall conversation quality