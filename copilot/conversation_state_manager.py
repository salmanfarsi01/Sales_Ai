from datetime import datetime, timezone, timedelta
import logging
import re
import uuid
from typing import Any, Dict, List, Optional


def resolve_relative_hold_date(
    hold_str: str,
    base_dt: Optional[datetime] = None,
    tz_name: str = "UTC",
) -> str:
    """Resolves relative weekday/timing strings ('Thursday', 'tomorrow', 'next week')
    into an absolute ISO-8601 date string (YYYY-MM-DD) based on call time and prospect timezone.
    """
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    now = base_dt.astimezone(tz) if (base_dt and base_dt.tzinfo) else (base_dt or datetime.now(tz))
    if not now.tzinfo:
        now = now.replace(tzinfo=tz)

    clean = hold_str.lower().strip()
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    if clean == "tomorrow":
        target = now + timedelta(days=1)
    elif clean in ("tonight", "today"):
        target = now
    elif clean in ("next week", "later this week"):
        target = now + timedelta(days=7)
    elif clean in weekdays:
        target_wd = weekdays[clean]
        current_wd = now.weekday()
        days_ahead = (target_wd - current_wd) % 7
        if days_ahead == 0:
            days_ahead = 7
        target = now + timedelta(days=days_ahead)
    else:
        target = now + timedelta(days=2)

    return target.strftime("%Y-%m-%d")

from .conversation_state_contract import BehavioralSignalInputBundle
from .conversation_state_models import (
    ConversationStateSnapshot,
    DecisionStructure,
    DecisionStakeholder,
    DimensionScores,
    ContactComplianceState,
    ContactPreference,
    ComplianceEvent,
    PersistentFactRecord,
    PushStrengthRecommendation,
    StateChangeRecord,
    ConversionEventStatus,
    ConversionEventObject,
    DealDispositionType,
    DealDispositionRecord,
    ConversationStage,
    StageHistoryRecord,
    ObjectionLifecycleState,
    ObjectionRecord,
    DormancyEvidence,
)
from .behavioral_semantic import (
    extract_structured_contact_preference,
    extract_structured_contact_preferences,
    HARD_BOUNDARY_PATTERNS,
    SOFT_CONTACT_PREFERENCE_PATTERNS,
)
from .conversation_facts import PersistentFactsManager
from .conversation_objections import ObjectionLifecycleEngine, DECISION_TO_STAY_PATTERNS
from .conversation_supersession import TruthSupersessionDetector
from .conversation_materiality import (
    MaterialityFilter,
    MaterialityClassification,
    ABSENT_DECISION_MAKER_PATTERNS,
    PRESENCE_CONFIRMATION_PATTERNS,
    _get_dormancy_evidence,
)
from .conversation_scoring import ConversationScoringEngine
from .conversation_scoring_config import ConversationScoringConfig
from .conversation_conversion_config import ConversionBlockingConfig, DEFAULT_CONVERSION_BLOCKING_CONFIG
from .conversation_conversion import MeetingConversionGateEngine, is_ampm_clarification, format_time_slot
from .conversation_stage import ConversationStageEngine

LOGGER = logging.getLogger("copilot.conversation_state_manager")


class StaleStateUpdateError(ValueError):
    """Raised when an asynchronous or delayed update attempts to mutate state from an older turn."""
    pass


class ConversationStateManager:
    """Core state engine coordinating snapshots, explainability tracking, and monotonic versioning."""

    def __init__(
        self,
        call_sid: str,
        initial_snapshot: Optional[ConversationStateSnapshot] = None,
        scoring_config: Optional[ConversationScoringConfig] = None,
        blocking_config: Optional[ConversionBlockingConfig] = None,
        conversion_target: str = "appointment",
        load_prospect_memory: bool = False,
        prospect_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_store: Optional[Any] = None,
    ):
        self.call_sid = str(call_sid)
        self.conversion_target = conversion_target
        self.prospect_id = prospect_id
        self.user_id = user_id
        self.memory_store = memory_store
        
        # Point 2: Guarantee clean-slate state isolation. If an initial snapshot is passed,
        # deep-copy it so caller mutations never cross into this manager instance.
        if initial_snapshot is not None:
            self.current_state = initial_snapshot.model_copy(deep=True)
            self.current_state.call_sid = self.call_sid
        else:
            self.current_state = ConversationStateSnapshot(call_sid=self.call_sid)

        # Explicit Prospect Memory Loading (separate, intentional path per Group 3)
        if load_prospect_memory:
            if not self.user_id or not str(self.user_id).strip():
                LOGGER.warning("load_prospect_memory=True passed without mandatory user_id. Failing closed: zero memory loaded (Point 18).")
            elif prospect_id:
                self._load_prospect_memory(prospect_id, user_id=self.user_id)

        self.facts_manager = PersistentFactsManager(initial_facts=[f.model_copy(deep=True) for f in self.current_state.facts])
        dormancy_thresh = scoring_config.dormancy_turn_threshold if scoring_config else 3
        self.objections_engine = ObjectionLifecycleEngine(
            initial_objections=[o.model_copy(deep=True) for o in self.current_state.objections],
            dormancy_turn_threshold=dormancy_thresh,
        )
        self.supersession_detector = TruthSupersessionDetector()
        self.materiality_filter = MaterialityFilter()
        self.scoring_engine = ConversationScoringEngine(config=scoring_config)
        self.blocking_config = blocking_config or DEFAULT_CONVERSION_BLOCKING_CONFIG
        self.conversion_engine = MeetingConversionGateEngine(
            config=scoring_config,
            blocking_config=self.blocking_config,
        )
        self.stage_engine = ConversationStageEngine()
        if not self.current_state.stage_history:
            self.current_state.stage_history = [
                StageHistoryRecord(
                    stage=self.current_state.conversation_stage,
                    entered_turn_id=0,
                    trigger_reason="Initial call setup",
                    confidence=1.0,
                )
            ]
        self.prior_bundle: Optional[BehavioralSignalInputBundle] = None
        self._bundle_history: List[BehavioralSignalInputBundle] = []
        self.has_prospect_spoken: bool = False
        self._conversion_events: List[ConversionEventObject] = [e.model_copy(deep=True) for e in self.current_state.conversion_events]
        if self.current_state.conversion_event and not any(e.event_id == self.current_state.conversion_event.event_id for e in self._conversion_events):
            self._conversion_events.append(self.current_state.conversion_event.model_copy(deep=True))
        self.deal_disposition: Optional[DealDispositionRecord] = (
            self.current_state.deal_disposition.model_copy(deep=True)
            if self.current_state.deal_disposition
            else DealDispositionRecord(disposition=DealDispositionType.ACTIVELY_SELLING)
        )
        self.current_state.deal_disposition = self.deal_disposition
        self._deal_dispositions: List[DealDispositionRecord] = [d.model_copy(deep=True) for d in self.current_state.deal_dispositions]
        if self.deal_disposition and not any(d.disposition_id == self.deal_disposition.disposition_id for d in self._deal_dispositions):
            self._deal_dispositions.append(self.deal_disposition)
        self.current_state.deal_dispositions = list(self._deal_dispositions)
        self.last_materiality: Optional[MaterialityClassification] = None

    def _load_prospect_memory(self, prospect_id: str, user_id: Optional[str] = None) -> None:
        """Explicitly loads historical prospect memory (e.g. contact preferences, persistent boundary constraints)
        for returning prospects. This path is NEVER executed automatically on new calls unless explicitly requested.

        Point 15 & Point 18 Security Invariants:
        1. Retrieval is strictly scoped: user_id -> prospect_id.
        2. If user_id is missing or empty, execution fails closed (zero memory loaded).
           Universal or unscoped retrieval across users is structurally prohibited (Point 18).
        3. Memories are loaded into self.current_state.loaded_prospect_memory with complete provenance
           and kept separate from current-call facts to prevent silent fact conflation (Point 17 Situation 2).
        """
        if not user_id or not str(user_id).strip():
            LOGGER.warning("Refusing to load prospect memory without mandatory user_id (Point 18 fail-closed).")
            return

        try:
            from .prospect_memory import ProspectMemoryStore
            store = self.memory_store or ProspectMemoryStore()
            records = store.retrieve_memories(user_id=user_id, prospect_id=prospect_id)
            for rec in records:
                loaded_item = rec.to_loaded_memory()
                self.current_state.loaded_prospect_memory.append(loaded_item)
                
                # If this memory item represents contact compliance/preference, also register in compliance state
                if rec.memory_type == "contact_preference":
                    pref_data = rec.metadata or {}
                    chan = pref_data.get("channel", "sms" if "text" in rec.content.lower() else "call")

                    # Rule (c): A historical preference cannot override a newer in-call preference
                    has_newer_incall_pref = any(
                        p.channel == chan and not getattr(p, "is_historical", False)
                        for p in self.current_state.contact_compliance.contact_preferences
                    )
                    if has_newer_incall_pref:
                        LOGGER.info("Skipping historical contact preference for %s: superseded by newer in-call preference", chan)
                        continue

                    pref_val = rec.value if isinstance(rec.value, str) else pref_data.get("preference", "reduced_frequency")
                    self.current_state.contact_compliance.contact_preference = pref_val
                    self.current_state.contact_compliance.contact_preference_details = rec.content
                    self.current_state.contact_compliance.contact_preference_confidence = rec.confidence
                    self.current_state.contact_compliance.contact_preferences.append(
                        ContactPreference(
                            channel=chan,
                            allowed=pref_data.get("allowed", True),
                            cadence=pref_data.get("cadence", "reduced"),
                            prohibited_behavior=pref_data.get("prohibited_behavior", rec.content),
                            source_turn_id=rec.source_turn_id or 0,
                            confidence=rec.confidence,
                            is_historical=True,
                            source_call_sid=rec.source_call_sid,
                        )
                    )
        except Exception as exc:
            LOGGER.warning("Could not load prospect memory for user %s, prospect %s: %s", user_id, prospect_id, exc)

    def save_prospect_memory(
        self,
        memory_type: str,
        key: str,
        value: Any,
        content: str,
        source_turn_id: Optional[int] = None,
        source_event_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Saves a memory record strictly scoped to this manager's user_id and prospect_id (Point 15)."""
        if not self.user_id or not self.prospect_id:
            LOGGER.warning("Cannot save prospect memory: user_id and prospect_id are required (Point 15)")
            return None
        from .prospect_memory import ProspectMemoryStore
        store = self.memory_store or ProspectMemoryStore()
        return store.save_record(
            user_id=self.user_id,
            prospect_id=self.prospect_id,
            source_call_sid=self.call_sid,
            memory_type=memory_type,
            key=key,
            value=value,
            content=content,
            source_turn_id=source_turn_id,
            source_event_id=source_event_id,
            confidence=confidence,
            metadata=metadata,
        )


    def process_turn_bundle(
        self,
        bundle: BehavioralSignalInputBundle,
        decision_updates: Optional[Dict[str, Any]] = None,
        fact_updates: Optional[List[Dict[str, Any]]] = None,
        conversion_target: Optional[str] = None,
    ) -> ConversationStateSnapshot:
        """Applies a verified Behavioral Signal Engine turn bundle to update the conversation truth."""
        if conversion_target is not None:
            self.conversion_target = conversion_target
        if bundle.speaker_id == "client":
            self.has_prospect_spoken = True
            if bundle.turn_id not in self.current_state.prospect_turn_ids:
                self.current_state.prospect_turn_ids.append(bundle.turn_id)
        self._bundle_history.append(bundle)
        # Stale-write protection (Client Principle #9)
        # 1. Strictly reject turns with turn_id older than latest processed turn
        if bundle.turn_id < self.current_state.last_updated_turn_id:
            raise StaleStateUpdateError(
                f"Stale state rejection: bundle turn {bundle.turn_id} is older than current turn "
                f"{self.current_state.last_updated_turn_id}."
            )
        # 2. If turn_id equals latest turn, allow intra-turn refinement (e.g. late LLM enrichment or
        # continuation fragment) ONLY IF its timestamp is >= current state timestamp.
        # Reject if timestamp is strictly earlier (stale out-of-order packet).
        if (
            bundle.turn_id == self.current_state.last_updated_turn_id
            and bundle.timestamp_ms < self.current_state.last_updated_timestamp_ms
        ):
            raise StaleStateUpdateError(
                f"Stale state rejection: bundle turn {bundle.turn_id} has earlier timestamp "
                f"({bundle.timestamp_ms}ms) than current processed turn ({self.current_state.last_updated_timestamp_ms}ms)."
            )

        # Tier-1 Turn-Level Gating: Materiality Filter (Phase 5 / Sprint 4)
        materiality = self.materiality_filter.classify_turn(
            bundle=bundle,
            current_state=self.current_state,
            has_explicit_fact_updates=bool(fact_updates),
            prior_bundle=self.prior_bundle,
        )
        self.last_materiality = materiality

        changes: List[StateChangeRecord] = []
        old_version = self.current_state.state_version
        next_version = old_version

        dormancy_thresh = getattr(self.objections_engine, "dormancy_turn_threshold", 3)
        dormant_due = any(
            o.lifecycle_state in ("active", "partially_addressed", "unresolved", "reactivated", ObjectionLifecycleState.ACTIVE, ObjectionLifecycleState.PARTIALLY_ADDRESSED)
            and (
                (bundle.turn_id - o.last_updated_turn_id) >= dormancy_thresh
                or getattr(o, "superseded_by_objection_id", None) is not None
            )
            and _get_dormancy_evidence(o, self.current_state, bundle.turn_id, self._bundle_history) is not None
            for o in self.current_state.objections
        )

        # Check if conversation_stage would transition
        stage_check, _ = self.stage_engine.evaluate_stage(
            bundle=bundle,
            state=self.current_state,
            materiality=materiality,
            prior_bundle=self.prior_bundle,
        )
        stage_transition_due = (stage_check != self.current_state.conversation_stage)

        # If turn is non-material (no targets affected, no manual decision/fact updates, no dormancy aging, no stage transition due,
        # no pending suspected boundary, and no active hard boundary that prospect could retract), preserve state version without mutating.
        has_pending_suspected = self.current_state.contact_compliance.boundary_suspected
        has_active_hard_boundary = self.current_state.contact_compliance.hard_boundary_active and bundle.speaker_id == "client"
        has_pending_reframe = bool(
            getattr(self.objections_engine, "pending_reframe_strategy", None)
            and bundle.speaker_id == "client"
        )
        if (
            not materiality.is_material
            and not decision_updates
            and not fact_updates
            and not dormant_due
            and not stage_transition_due
            and not has_pending_suspected
            and not has_active_hard_boundary
            and not has_pending_reframe
        ):
            self.current_state.last_updated_turn_id = bundle.turn_id
            self.current_state.last_updated_timestamp_ms = bundle.timestamp_ms
            self.prior_bundle = bundle
            return self.current_state

        # 1. Update Dimensions (Gated by Materiality)
        if "dimensions" in materiality.affected_targets:
            commit_val, commit_conf = self._compute_commitment(bundle)
            trust_drivers = getattr(bundle.trust, "drivers", [])
            trust_is_measured = (
                bundle.trust.score != 0.50
                or getattr(bundle.trust, "is_measured", False)
                or any("confidence" not in str(d).lower() for d in trust_drivers)
            )
            new_dims = DimensionScores(
                trust=bundle.trust.score,
                trust_confidence=bundle.trust.confidence,
                trust_measured=trust_is_measured,
                emotion_valence=bundle.emotion.expressed_valence,
                emotion_tension=bundle.emotion.tension_level,
                emotion_confidence=bundle.emotion.confidence,
                engagement=bundle.engagement.score,
                engagement_confidence=bundle.engagement.confidence,
                momentum=bundle.momentum.score,
                momentum_confidence=bundle.momentum.confidence,
                readiness=bundle.readiness.score,
                readiness_confidence=bundle.readiness.confidence,
                commitment=commit_val,
                commitment_confidence=commit_conf,
                pacing=bundle.pacing.score,
                pacing_confidence=bundle.pacing.confidence,
            )
            if new_dims != self.current_state.dimensions:
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=old_version,
                        state_version_after=next_version,
                        field_path="dimensions",
                        old_value=self.current_state.dimensions.model_dump(),
                        new_value=new_dims.model_dump(),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason="Updated downstream inference dimension scores from Behavioral Signal Engine.",
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )
                self.current_state.dimensions = new_dims

        # 2. Update Contact/Compliance State (Gated by Materiality)
        if "contact_compliance" in materiality.affected_targets or bundle.speaker_id == "client":
            text_lower = bundle.utterance_text.lower().strip()

            # 1. Bounded Scheduling alternative exemption (Spec 09 / Client Audit Fix Item 4):
            # "Don't call me tomorrow, call Thursday" -> genuine scheduling alternative.
            # "Don't call me again, call my lawyer" -> cutoff / legal referral, MUST NOT pass as scheduling.
            has_cutoff_terms = bool(
                re.search(r"\b(?:again|anymore|never|stop|lawyer|attorney|counsel)\b", text_lower)
            )
            time_expr_pattern = (
                r"\b(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                r"tomorrow|tonight|next\s+week|this\s+weekend|next\s+month|later\s+this\s+week|"
                r"in\s+the\s+morning|in\s+the\s+afternoon|in\s+the\s+evening|after\s+\d+|at\s+\d+|\d+\s*(?:am|pm))\b"
            )
            neg_match = re.search(
                r"\bdon['’]?t\s+(?:call|reach\s+out|text)\s+(?:me\s+)?(?:today|tomorrow|tonight|right\s+now|this\s+morning|this\s+afternoon)\b",
                text_lower,
            )
            has_negative_timing_prefix = bool(neg_match)

            # Time-bounded hold check (Client Audit Fix):
            # "Don't call me again until Thursday" -> temporal hold, NOT boundary_suspected.
            hold_match = re.search(
                r"\bdon['’]?t\s+(?:call|reach\s+out|text|contact)\s+(?:me\s+)?(?:again\s+)?until\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|next\s+week|[a-z]+\s+\d{1,2}(?:th|st|rd|nd)?)\b",
                text_lower,
            )
            is_time_bounded_hold = bool(hold_match)
            hold_target_raw = hold_match.group(1).strip().title() if hold_match else None
            hold_target = resolve_relative_hold_date(hold_target_raw) if hold_target_raw else None

            # Daily timing preference check:
            # "Don't call after 4pm" -> daily time preference, NOT a hold and NOT a boundary.
            daily_timing_pref_match = re.search(
                r"\bdon['’]?t\s+(?:call|reach\s+out|text|contact)\s+(?:me\s+)?(?:after|before)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon)\b",
                text_lower,
            )
            is_daily_timing_pref = bool(daily_timing_pref_match)

            is_scheduling_alternative = False
            if not has_cutoff_terms and neg_match and not is_time_bounded_hold and not is_daily_timing_pref:
                subsequent_text = text_lower[neg_match.end():]
                if re.search(time_expr_pattern, subsequent_text):
                    is_scheduling_alternative = True

            # Ambiguous contact timing friction falls back to boundary_suspected (Client Audit Fix Item 4)
            is_ambiguous_contact_timing = has_negative_timing_prefix and not is_scheduling_alternative and not has_cutoff_terms and not is_time_bounded_hold and not is_daily_timing_pref

            # 2. Retraction patterns: explicit prospect invitation to contact after a prior boundary
            retraction_patterns = [
                r"\b(?:actually\b\s*,?\s*)?(?:you\s+can|feel\s+free\s+to|go\s+ahead\s+and)\s+(?:call|reach\s+out|contact|text)\b",
                r"\b(?:i\s+changed\s+my\s+mind|never\s+mind|changed\s+mind)\b.*?\b(?:call|reach\s+out|contact|talk)\b",
                r"\b(?:go\s+ahead\s+and\s+call|call\s+me\s+back|call\s+me\s+tomorrow|call\s+me\s+later)\b",
                r"\b(?:it['’]?s\s+okay\s+to|fine\s+to|you\s+may)\s+(?:call|contact|reach\s+out)\b",
            ]
            is_retraction = (
                bundle.speaker_id == "client"
                and self.current_state.contact_compliance.hard_boundary_active
                and any(re.search(p, text_lower) for p in retraction_patterns)
            )

            # 3. Soft preference check: if utterance is purely a soft preference ("don't text me every day", "don't call after 4pm"), keep out of boundary path
            is_pure_soft_preference = is_daily_timing_pref
            if not is_pure_soft_preference:
                for cat, pats in SOFT_CONTACT_PREFERENCE_PATTERNS.items():
                    if any(re.search(p, text_lower) for p in pats):
                        is_pure_soft_preference = True
                        break

            # 4. Clause-level Boundary and Negation Parsing (Client feedback Item 4):
            # Split utterance by contrastive conjunctions or punctuation to evaluate each clause independently.
            # Prevents "I don't mind you calling, but don't call me again" from being false-negatived by the first clause.
            negation_patterns = [
                r"\b(?:don['’]?t|do\s+not|doesn['’]?t)\s+mind\s+(?:if\s+you\s+|you\s+)?(?:call|reach\s+out|text|contact)\b",
                r"\b(?:fine|okay|ok|welcome|happy)\s+(?:if\s+you|for\s+you\s+to)?\s*(?:call|reach\s+out|text|contact)\b",
                r"\b(?:you\s+can|feel\s+free\s+to|go\s+ahead\s+and)\s+(?:call|reach\s+out|text|contact)\b",
                r"\bnot\s+saying\s+(?:you\s+can['’]?t|never)\s+(?:call|reach\s+out|contact)\b",
                r"\b(?:i['’]?m\s+)?not\s+against\s+(?:you\s+)?(?:calling|contacting)\b",
            ]

            raw_clauses = re.split(r"(?<=[.?!;,])\s+|\s+(?:but|however|except|although|though)\s+", text_lower)
            clauses = [c.strip() for c in raw_clauses if c.strip()]
            if not clauses:
                clauses = [text_lower]

            matched_boundary_pattern = None
            if bundle.speaker_id == "client" and not is_retraction and not is_scheduling_alternative:
                for clause in clauses:
                    clause_negated = any(re.search(p, clause) for p in negation_patterns)
                    if clause_negated:
                        continue  # Permissive clause, do not match boundary in this clause

                    candidates = [p for p in HARD_BOUNDARY_PATTERNS if re.search(p, clause)]
                    if is_pure_soft_preference:
                        candidates = [p for p in candidates if any(w in p for w in ["loss_my_number", "lose_my_number", "dnc", "harass", "leave_me_alone"])]
                    if candidates:
                        matched_boundary_pattern = candidates[0]
                        break

            is_hard_turn = (
                bundle.speaker_id == "client"
                and not is_retraction
                and not is_scheduling_alternative
                and not is_ambiguous_contact_timing
                and not is_time_bounded_hold
                and not is_daily_timing_pref
                and (
                    (bundle.boundary_score >= 0.85 and not any(re.search(p, text_lower) for p in negation_patterns if len(clauses) == 1))
                    or bundle.recurrence_type == "boundary_repeated"
                    or bool(matched_boundary_pattern)
                )
            )

            # 5. Narrowly Scoped boundary_suspected with Clear Lifecycle Exits (Client Audit Fix Items 2 & 3):
            # Scope: ONLY triggers on contact-specific friction ("too many calls", "you keep calling", "stop calling so much", "leave me alone").
            # Cold outreach first reactions ("why are you calling me", "who gave you this number") are objections / who-and-why clarify, NOT boundary_suspected.
            contact_specific_friction_patterns = [
                r"\b(?:too\s+many\s+calls|you\s+keep\s+calling|calling\s+(?:too\s+much|all\s+the\s+time)|stop\s+calling\s+so\s+much)\b",
                r"\b(?:not\s+sure|don['’]?t\s+know)\s+(?:if\s+)?(?:i\s+want\s+to|you\s+should)\s+(?:talk|call|chat)\b",
                r"\b(?:maybe\s+)?don['’]?t\s+call\s+(?:me\s+)?(?:right\s+now|today|for\s+now)\b",
                r"\b(?:leave\s+me\s+alone\s+(?:for\s+now)?|uncomfortable\s+with\s+you\s+calling)\b",
            ]
            has_contact_friction = (
                (any(re.search(p, text_lower) for p in contact_specific_friction_patterns) or is_ambiguous_contact_timing)
                and not is_time_bounded_hold
                and not is_daily_timing_pref
            )

            # Clearing boundary_suspected (Client Audit Fix Item 3 & Smaller Items):
            # Naked backchannels ("okay", "fine", "sure") do NOT clear suspected boundaries.
            # Must be an explicit contact permission statement requiring the object ('me' or 'us' or 'to you').
            # E.g. "you can call my lawyer" and "happy to talk to your manager" must NOT clear.
            has_contact_permission_statement = bool(
                re.search(
                    r"\b(?:it['’]?s\s+(?:fine|okay)\s+to\s+(?:call|reach\s+out\s+to|talk\s+to)\s+(?:me|us)\b|"
                    r"(?:you\s+can|feel\s+free\s+to|go\s+ahead\s+and)\s+(?:definitely\s+|certainly\s+)?(?:call|reach\s+out\s+to|talk\s+to|contact)\s+(?:me|us)\b|"
                    r"happy\s+to\s+talk\s+(?:to|with)\s+you\b|"
                    r"(?:fine|okay)\s+to\s+call\s+(?:me|us)\b|"
                    r"feel\s+free\s+to\s+(?:call|contact)\s+(?:me|us)\b)",
                    text_lower,
                )
            ) and not bool(re.search(r"\b(?:don['’]?t|do\s+not|never|can['’]?t|shouldn['’]?t)\s+(?:ever\s+)?(?:call|reach\s+out|contact)\b", text_lower))
            if re.search(r"\b(?:call|talk\s+to|reach\s+out\s+to)\s+(?:my\s+(?:lawyer|attorney)|your\s+manager|someone\s+else)\b", text_lower):
                has_contact_permission_statement = False

            has_substantive_engagement = (
                bundle.speaker_id == "client"
                and (
                    bundle.agreement_score >= 0.70
                    or bundle.specificity_score >= 0.70
                    or any(cat in text_lower for cat in ["price", "commission", "timeline", "property", "house", "appointment", "walkthrough"])
                )
                and not any(re.search(p, text_lower) for p in contact_specific_friction_patterns)
            )
            cleared_by_prospect = bool(
                self.current_state.contact_compliance.boundary_suspected
                and (has_contact_permission_statement or has_substantive_engagement)
            )

            # Compliance-First Boundary Catch: Negative imperative plus a contact word
            # ("no contact", "never call", "don't text", "stop calling", "take me off", etc.)
            # Always sets boundary_suspected = True whether or not extraction succeeded.
            compliance_negative_imperative_patterns = [
                r"\bno\s+contact\b",
                r"\bnever\s+call\b",
                r"\bnever\s+contact\b",
                r"\bdon['’]?t\s+text\b",
                r"\bdo\s+not\s+text\b",
                r"\bstop\s+calling\b",
                r"\bstop\s+texting\b",
                r"\bstop\s+contacting\b",
                r"\btake\s+me\s+off\b",
                r"\bno\s+calls\b",
                r"\bnever\s+reach\s+out\b",
                r"\bdon['’]?t\s+reach\s+out\b",
                r"\bdo\s+not\s+reach\s+out\b",
            ]
            has_compliance_negative_imperative = (
                bundle.speaker_id == "client"
                and any(re.search(p, text_lower) for p in compliance_negative_imperative_patterns)
                and not is_time_bounded_hold
                and not is_daily_timing_pref
            )

            # Check if prior boundary_suspected has expired (N >= 2 turns elapsed)
            prior_suspected_turn = self.current_state.contact_compliance.boundary_suspected_turn_id
            has_expired = (bundle.turn_id - prior_suspected_turn) >= 2 if prior_suspected_turn is not None else False

            is_suspected = (
                bundle.speaker_id == "client"
                and not is_hard_turn
                and not is_retraction
                and not is_scheduling_alternative
                and (
                    has_compliance_negative_imperative
                    or (
                        not is_pure_soft_preference
                        and not is_time_bounded_hold
                        and not is_daily_timing_pref
                        and has_contact_friction
                    )
                )
            )

            suspected_turn_id = self.current_state.contact_compliance.boundary_suspected_turn_id

            if is_retraction:
                hard_active = False
                hard_reason = f"Boundary retracted by explicit prospect invitation ('{bundle.utterance_text.strip()}')"
                hard_retracted = True
                retract_turn = bundle.turn_id
                suspected_active = False
                suspected_reason = None
                suspected_turn_id = None
                self.current_state.compliance_events.append(
                    ComplianceEvent(
                        event_type="boundary_retracted",
                        occurred_at=datetime.now(timezone.utc).isoformat(),
                        source_turn_ids=[bundle.turn_id],
                        details={
                            "reason": hard_reason,
                            "utterance": bundle.utterance_text,
                            "prior_boundary_reason": self.current_state.contact_compliance.hard_boundary_reason,
                            "is_inferred_extension": True,
                            "requires_human_compliance_review": True,
                            "automatic_suppression_removal": False,
                        },
                        is_inferred_extension=True,
                    )
                )
            elif is_hard_turn:
                # 1. Hard Boundary Confirmation takes priority (Evaluated before expiry)
                hard_active = True
                hard_retracted = False
                retract_turn = None
                suspected_active = False
                suspected_reason = None
                suspected_turn_id = None
                if bundle.boundary_score >= 0.85:
                    hard_reason = f"Triggered by upstream compliance boundary detection (score={bundle.boundary_score:.2f})"
                elif bundle.recurrence_type == "boundary_repeated":
                    hard_reason = "Triggered by repeated compliance boundary escalation"
                elif matched_boundary_pattern:
                    hard_reason = f"Triggered by explicit compliance cutoff phrase ('{bundle.utterance_text.strip()}')"
                else:
                    hard_reason = "Triggered by compliance boundary detection"
            elif is_suspected:
                # 2. Suspected boundary active or refreshed
                hard_active = self.current_state.contact_compliance.hard_boundary_active
                hard_reason = self.current_state.contact_compliance.hard_boundary_reason
                hard_retracted = self.current_state.contact_compliance.hard_boundary_retracted
                retract_turn = self.current_state.contact_compliance.retraction_turn_id
                suspected_active = True
                if has_compliance_negative_imperative:
                    suspected_reason = f"Compliance negative imperative detected ('{bundle.utterance_text.strip()}')"
                else:
                    suspected_reason = f"Contact-specific friction detected ('{bundle.utterance_text.strip()}')"
                suspected_turn_id = bundle.turn_id
            elif cleared_by_prospect:
                # 3. Explicit Clearing by Prospect takes priority (Evaluated before expiry)
                hard_active = self.current_state.contact_compliance.hard_boundary_active
                hard_reason = self.current_state.contact_compliance.hard_boundary_reason
                hard_retracted = self.current_state.contact_compliance.hard_boundary_retracted
                retract_turn = self.current_state.contact_compliance.retraction_turn_id
                suspected_active = False
                suspected_reason = None
                suspected_turn_id = None
            elif has_expired:
                # 4. Suspected boundary expires after 2 turns without confirmation:
                # Resume normal persuasion, but store friction in prospect memory (facts)
                hard_active = self.current_state.contact_compliance.hard_boundary_active
                hard_reason = self.current_state.contact_compliance.hard_boundary_reason
                hard_retracted = self.current_state.contact_compliance.hard_boundary_retracted
                retract_turn = self.current_state.contact_compliance.retraction_turn_id
                suspected_active = False
                suspected_reason = None
                suspected_turn_id = None

                # Persist friction in prospect memory so it's not forgotten (Client Audit Item 3)
                if not any(f.fact_key == "past_contact_friction" and f.status == "active" for f in self.facts_manager.get_active_facts()):
                    self.facts_manager.record_fact(
                        category="preference",
                        fact_key="past_contact_friction",
                        fact_value="Prospect previously exhibited contact hesitation/friction that expired without confirmation.",
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        confidence=0.85,
                        notes="Stored from expired boundary_suspected lifecycle",
                    )
            else:
                hard_active = self.current_state.contact_compliance.hard_boundary_active
                hard_reason = self.current_state.contact_compliance.hard_boundary_reason
                hard_retracted = self.current_state.contact_compliance.hard_boundary_retracted
                retract_turn = self.current_state.contact_compliance.retraction_turn_id
                suspected_active = self.current_state.contact_compliance.boundary_suspected
                suspected_reason = self.current_state.contact_compliance.boundary_suspected_reason

            # Track prohibited channels
            channels = list(self.current_state.contact_compliance.hard_boundary_channels)
            if is_hard_turn:
                if any(w in text_lower for w in ["call", "calling", "phone", "number"]):
                    if "call" not in channels:
                        channels.append("call")
                if any(w in text_lower for w in ["text", "sms", "message"]):
                    if "sms" not in channels:
                        channels.append("sms")
                if any(w in text_lower for w in ["email"]):
                    if "email" not in channels:
                        channels.append("email")
                if not channels or any(w in text_lower for w in ["contact", "reach out", "harass", "leave me alone", "dnc", "list"]):
                    for ch in ["call", "sms", "email"]:
                        if ch not in channels:
                            channels.append(ch)

                self.current_state.compliance_events.append(
                    ComplianceEvent(
                        event_type="hard_boundary_confirmed",
                        occurred_at=datetime.now(timezone.utc).isoformat(),
                        source_turn_ids=[bundle.turn_id],
                        details={
                            "reason": hard_reason,
                            "channels": channels,
                            "spec_literal_fallback": "consent_issue",
                            "is_inferred_extension": True,
                            "requires_human_compliance_review": True,
                            "automatic_suppression_removal": False,
                            "utterance": bundle.utterance_text,
                        },
                        is_inferred_extension=True,
                    )
                )

            # Update contact preferences list (distinct from hard boundary!)
            prefs = list(self.current_state.contact_compliance.contact_preferences)
            new_prefs = extract_structured_contact_preferences(
                text=bundle.utterance_text,
                source_turn_id=bundle.turn_id,
                confidence=bundle.contact_preference_confidence or 0.90,
            )
            for new_pref in new_prefs:
                new_pref.source_call_sid = self.call_sid
                new_pref.is_historical = False
                existing_idx = next((i for i, p in enumerate(prefs) if p.channel == new_pref.channel), None)
                if existing_idx is not None:
                    prefs[existing_idx] = new_pref
                else:
                    prefs.append(new_pref)

            # Record temporal contact hold in prospect memory
            if is_time_bounded_hold and hold_target:
                if not any(f.fact_key == "contact_not_before" and f.status == "active" for f in self.facts_manager.get_active_facts()):
                    self.facts_manager.record_fact(
                        category="preference",
                        fact_key="contact_not_before",
                        fact_value=f"Do not contact before {hold_target}",
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        confidence=0.90,
                        notes=f"Temporal contact hold declared until {hold_target}",
                    )

            contact_not_before_val = (
                hold_target if is_time_bounded_hold
                else self.current_state.contact_compliance.contact_not_before
            )
            contact_not_before_turn = (
                bundle.turn_id if is_time_bounded_hold
                else self.current_state.contact_compliance.contact_not_before_turn_id
            )

            new_compliance = ContactComplianceState(
                hard_boundary_active=hard_active,
                hard_boundary_reason=hard_reason,
                hard_boundary_channels=channels,
                contact_preferences=prefs,
                contact_not_before=contact_not_before_val,
                contact_not_before_turn_id=contact_not_before_turn,
                boundary_suspected=suspected_active,
                boundary_suspected_reason=suspected_reason,
                boundary_suspected_turn_id=suspected_turn_id,
                hard_boundary_retracted=hard_retracted,
                retraction_turn_id=retract_turn,
                contact_preference=bundle.contact_preference if bundle.contact_preference != "none" else self.current_state.contact_compliance.contact_preference,
                contact_preference_details=bundle.contact_preference_details or self.current_state.contact_compliance.contact_preference_details,
                contact_preference_confidence=bundle.contact_preference_confidence or self.current_state.contact_compliance.contact_preference_confidence,
            )
            if new_compliance != self.current_state.contact_compliance:
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path="contact_compliance",
                        old_value=self.current_state.contact_compliance.model_dump(),
                        new_value=new_compliance.model_dump(),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Updated compliance state (hard_boundary={hard_active}, pref={bundle.contact_preference}, prefs_count={len(prefs)}).",
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )
                self.current_state.contact_compliance = new_compliance

        # 3. Apply Decision Structure updates (Explicit or Autonomous from bundle)
        auto_decision_updates = None
        if not decision_updates and bundle.speaker_id == "client":
            auto_decision_updates = self._extract_autonomous_decision_updates(bundle, materiality)

        effective_dec_updates = decision_updates or auto_decision_updates
        if effective_dec_updates:
            next_version += 1
            old_dec = self.current_state.decision_structure.model_dump()
            for k, v in effective_dec_updates.items():
                if hasattr(self.current_state.decision_structure, k):
                    setattr(self.current_state.decision_structure, k, v)
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="decision_structure",
                    old_value=old_dec,
                    new_value=self.current_state.decision_structure.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Updated decision structure ({', '.join(effective_dec_updates.keys())}).",
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        # 4. Evaluate Objection Lifecycle (Gated by Materiality or Pending Reframe Response)
        if "objections" in materiality.affected_targets or (getattr(self.objections_engine, "pending_reframe_strategy", None) and bundle.speaker_id == "client"):
            # Sync any out-of-band added objections into the lifecycle engine
            engine_ids = {o.objection_id for o in self.objections_engine._objections}
            for o in self.current_state.objections:
                if o.objection_id not in engine_ids:
                    self.objections_engine._objections.append(o)

            updated_objs, obj_changes, next_version = self.objections_engine.evaluate_turn(
                bundle=bundle,
                current_version=next_version,
            )
            self.current_state.objections = updated_objs
            changes.extend(obj_changes)

        # 4b. Evaluate Dormancy Aging for unaddressed concerns across turns
        updated_objs, dormant_changes, next_version = self.objections_engine.check_dormancy_aging(
            current_turn_id=bundle.turn_id,
            timestamp_ms=bundle.timestamp_ms,
            current_version=next_version,
            contributing_evidence_ids=bundle.contributing_evidence_ids,
            state=self.current_state,
            recent_bundles=self._bundle_history,
        )
        self.current_state.objections = updated_objs
        changes.extend(dormant_changes)

        # 5. Process Fact Updates if provided or autonomously extracted from client disclosures
        autonomous_fact_updates = None
        if not fact_updates and bundle.speaker_id == "client":
            autonomous_fact_updates = self._extract_autonomous_fact_updates(bundle, materiality)
        all_fact_updates = (fact_updates or []) + (autonomous_fact_updates or [])
        if all_fact_updates:
            for fu in all_fact_updates:
                self.facts_manager.record_fact(
                    category=fu.get("category", "general"),
                    fact_key=fu["fact_key"],
                    fact_value=fu["fact_value"],
                    source_turn_id=bundle.turn_id,
                    timestamp_ms=bundle.timestamp_ms,
                    confidence=fu.get("confidence", 1.0),
                    notes=fu.get("notes"),
                )
            next_version += 1
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="facts",
                    old_value=[f.model_dump() for f in self.current_state.facts],
                    new_value=[f.model_dump() for f in self.facts_manager.get_all_facts()],
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Recorded {len(all_fact_updates)} persistent fact update(s).",
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        # 6. Evaluate Truth Supersession (Gated by Materiality)
        if "facts" in materiality.affected_targets and bundle.speaker_id == "client":
            supersession_decisions = self.supersession_detector.evaluate_turn(
                candidate_text=bundle.utterance_text,
                active_facts=self.facts_manager.get_active_facts(),
                decision_structure=self.current_state.decision_structure,
            )
            for decision in supersession_decisions:
                if decision.has_supersession and decision.target_fact_id:
                    effective_conf = round(min(decision.confidence, bundle.semantic_confidence, bundle.inference_confidence), 3)
                    old_f, new_f = self.facts_manager.supersede_fact(
                        old_fact_id=decision.target_fact_id,
                        new_fact_value=decision.new_truth_value or bundle.utterance_text,
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        confidence=effective_conf,
                        notes=decision.reasoning,
                    )
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path=f"facts.{decision.target_fact_key}",
                            old_value=old_f.fact_value,
                            new_value=new_f.fact_value,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Truth supersession ({decision.relation}): {decision.reasoning}",
                            confidence=effective_conf,
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )

        # 7. Sync facts snapshot
        self.current_state.facts = self.facts_manager.get_all_facts()

        # 7b. Evaluate Deal Disposition (Phase 8 / Fix 2)
        if bundle.speaker_id == "client":
            new_disp = self.supersession_detector.evaluate_deal_disposition(
                candidate_text=bundle.utterance_text,
                current_disposition=self.deal_disposition,
                turn_id=bundle.turn_id,
                timestamp_ms=bundle.timestamp_ms,
            )
            if new_disp is not None and (self.deal_disposition is None or self.deal_disposition.disposition != new_disp.disposition or self.deal_disposition.disposition_id != new_disp.disposition_id):
                if self.deal_disposition is not None and self.deal_disposition.disposition_id != new_disp.disposition_id:
                    self.deal_disposition.superseded_by_id = new_disp.disposition_id
                    self.deal_disposition.superseded_at_turn_id = bundle.turn_id

                if not any(d.disposition_id == new_disp.disposition_id for d in self._deal_dispositions):
                    self._deal_dispositions.append(new_disp)
                prev_disp = self.deal_disposition
                self.deal_disposition = new_disp
                next_version += 1
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path="deal_disposition",
                        old_value=prev_disp.model_dump() if prev_disp else None,
                        new_value=new_disp.model_dump(),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Deal disposition transitioned to {new_disp.disposition}: {new_disp.rationale or ''}".strip(),
                        confidence=new_disp.confidence,
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )

        self.current_state.deal_disposition = self.deal_disposition
        self.current_state.deal_dispositions = list(self._deal_dispositions)

        # 8. Compute Momentum & Readiness Scoring (Phase 6)
        momentum_res = self.scoring_engine.compute_momentum(bundle, self.current_state)
        readiness_res = self.scoring_engine.compute_readiness(
            bundle,
            self.current_state,
            conversion_target=self.conversion_target,
            blocking_config=self.blocking_config,
        )
        self.current_state.momentum = momentum_res
        self.current_state.readiness = readiness_res

        # 9. Evaluate Meeting/Conversion Gate & Push Strength (Phase 7 / Sprint 6)
        prev_gate = self.current_state.conversion_gate
        prev_conv = self.current_state.conversion_event

        # Speaker Blindspot Protection: Gate evaluation is held pending until the prospect speaks,
        # preventing evaluating agent questions against prospect conversion criteria.
        if not self.has_prospect_spoken and bundle.speaker_id == "salesperson":
            gate_res = None
            push_res = PushStrengthRecommendation(
                state="resolve_then_ask",
                rationale="Awaiting prospect opening response; meeting gate criteria unknown on clean slate.",
                recommended_action="Engage prospect and discover core needs before proposing appointment windows.",
                confidence=0.50,
            )
            conv_res = None
        else:
            gate_res = self.conversion_engine.evaluate_gate(
                bundle,
                self.current_state,
                conversion_target=self.conversion_target,
                prior_bundle=self.prior_bundle,
            )
            conv_res = self.conversion_engine.evaluate_conversion_event(
                bundle, self.current_state, gate_res, previous_event=prev_conv
            )
            # Update state conversion event so push strength evaluation operates on fresh event state
            self.current_state.conversion_event = conv_res
            push_res = self.conversion_engine.evaluate_push_strength(
                bundle,
                self.current_state,
                gate_res,
                conversion_target=self.conversion_target,
            )

        # Explainability tracking for conversion state changes
        if gate_res is not None and (
            prev_gate is None
            or prev_gate.is_open != gate_res.is_open
            or prev_gate.conversion_target != gate_res.conversion_target
            or (gate_res.explicit_commitment_detected and not prev_gate.explicit_commitment_detected)
        ):
            next_version += 1
            override_note = " (explicit commitment override applied)" if gate_res.explicit_commitment_detected else ""
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="conversion_gate",
                    old_value=prev_gate.model_dump() if prev_gate else None,
                    new_value=gate_res.model_dump(),
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Meeting gate [{gate_res.conversion_target}] transitioned to {'OPEN' if gate_res.is_open else 'CLOSED'}{override_note}. Failed conditions: {gate_res.failed_conditions or 'None'}.",
                    confidence=gate_res.confidence,
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        if conv_res is not None:
            # Sync into _conversion_events
            existing_idx = next((i for i, e in enumerate(self._conversion_events) if e.event_id == conv_res.event_id), None)
            if existing_idx is not None:
                self._conversion_events[existing_idx] = conv_res
            else:
                self._conversion_events.append(conv_res)

            # If it supersedes an event, ensure that event in self._conversion_events is marked superseded
            if conv_res.supersedes_event_id:
                for ev in self._conversion_events:
                    if ev.event_id == conv_res.supersedes_event_id:
                        ev.superseded_by_event_id = conv_res.event_id
                        ev.superseded_at_turn_id = bundle.turn_id

            if prev_conv is None or prev_conv.status != conv_res.status or prev_conv.event_id != conv_res.event_id:
                next_version += 1
                supersede_info = f" (supersedes {conv_res.supersedes_event_id}, reversal_reason: {conv_res.reversal_reason})" if conv_res.supersedes_event_id else ""
                changes.append(
                    StateChangeRecord(
                        state_version_before=next_version - 1,
                        state_version_after=next_version,
                        field_path="conversion_event",
                        old_value=prev_conv.model_dump() if prev_conv else None,
                        new_value=conv_res.model_dump(),
                        triggering_turn_id=bundle.turn_id,
                        evidence_ids=bundle.contributing_evidence_ids,
                        reason=f"Conversion event updated: {conv_res.conversion_type} status={conv_res.status}{supersede_info}.",
                        confidence=conv_res.confirmation_confidence,
                        timestamp_ms=bundle.timestamp_ms,
                    )
                )

            # Deterministic Cascade: Conversion Event status directly governs confirmed_meeting_time fact
            active_meeting_fact = next(
                (f for f in self.facts_manager.get_active_facts() if f.fact_key == "confirmed_meeting_time"),
                None,
            )
            if conv_res.status == ConversionEventStatus.CANCELLED and active_meeting_fact:
                if not active_meeting_fact.fact_value.lower().startswith("cancel"):
                    old_f, new_f = self.facts_manager.supersede_fact(
                        old_fact_id=active_meeting_fact.fact_id,
                        new_fact_value="Cancelled",
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        notes=conv_res.reversal_reason or "Conversion event cancelled",
                    )
                    self.current_state.facts = self.facts_manager.get_all_facts()
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path="facts.confirmed_meeting_time",
                            old_value=old_f.fact_value,
                            new_value=new_f.fact_value,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Deterministic cascade from conversion event cancellation: {conv_res.reversal_reason or 'Cancelled'}",
                            confidence=conv_res.confirmation_confidence,
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )
            elif conv_res.status == ConversionEventStatus.CONFIRMED and conv_res.start_at:
                if active_meeting_fact and active_meeting_fact.fact_value.strip().lower() != conv_res.start_at.strip().lower():
                    is_clarification = is_ampm_clarification(active_meeting_fact.fact_value, conv_res.start_at)
                    note_text = f"Meeting time clarified to '{conv_res.start_at}'" if is_clarification else (conv_res.reversal_reason or f"Meeting rescheduled to '{conv_res.start_at}'")
                    old_f, new_f = self.facts_manager.supersede_fact(
                        old_fact_id=active_meeting_fact.fact_id,
                        new_fact_value=conv_res.start_at,
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        notes=note_text,
                    )
                    self.current_state.facts = self.facts_manager.get_all_facts()
                    next_version += 1
                    changes.append(
                        StateChangeRecord(
                            state_version_before=next_version - 1,
                            state_version_after=next_version,
                            field_path="facts.confirmed_meeting_time",
                            old_value=old_f.fact_value,
                            new_value=new_f.fact_value,
                            triggering_turn_id=bundle.turn_id,
                            evidence_ids=bundle.contributing_evidence_ids,
                            reason=f"Deterministic cascade from conversion event confirmation: '{conv_res.start_at}'",
                            confidence=conv_res.confirmation_confidence,
                            timestamp_ms=bundle.timestamp_ms,
                        )
                    )

        self.current_state.conversion_gate = gate_res
        self.current_state.push_strength = push_res
        self.current_state.conversion_event = conv_res
        self.current_state.conversion_events = list(self._conversion_events)

        # Re-sync commitment on dimensions if conversion event or gate updated commitment
        post_commit_val, post_commit_conf = self._compute_commitment_from_lifecycle(bundle, conv_res, gate_res)
        if post_commit_val != self.current_state.dimensions.commitment:
            old_commit = self.current_state.dimensions.commitment
            self.current_state.dimensions.commitment = post_commit_val
            self.current_state.dimensions.commitment_confidence = post_commit_conf
            next_version += 1
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="dimensions.commitment",
                    old_value=old_commit,
                    new_value=post_commit_val,
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=(
                        f"Commitment dimension updated from {old_commit:.2f} to {post_commit_val:.2f} "
                        f"reflecting conversion event status '{conv_res.status if conv_res else 'cancelled'}'."
                    ),
                    confidence=post_commit_conf,
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        # Refresh Momentum & Readiness so same-turn conversion events, commitment updates,
        # and deterministic fact cascades are immediately reflected without a 1-turn lag.
        if self.scoring_engine._momentum_history:
            self.scoring_engine._momentum_history.pop()
        self.current_state.momentum = self.scoring_engine.compute_momentum(bundle, self.current_state)
        self.current_state.readiness = self.scoring_engine.compute_readiness(
            bundle,
            self.current_state,
            conversion_target=self.conversion_target,
            blocking_config=self.blocking_config,
        )

        # Inconsistency Guard: Prune dimensions from materiality affected_targets if no dimension change materialized
        has_dim_change = any(c.field_path.startswith("dimensions") for c in changes)
        if "dimensions" in materiality.affected_targets and not has_dim_change:
            materiality.affected_targets.discard("dimensions")

        # 10. Evaluate Conversation Stage & Track Stage History (Client Principle #1)
        new_stage, stage_reason = self.stage_engine.evaluate_stage(
            bundle=bundle,
            state=self.current_state,
            materiality=materiality,
            prior_bundle=self.prior_bundle,
        )
        old_stage = self.current_state.conversation_stage
        if new_stage != old_stage:
            if self.current_state.stage_history and self.current_state.stage_history[-1].exited_turn_id is None:
                self.current_state.stage_history[-1].exited_turn_id = bundle.turn_id
            self.current_state.stage_history.append(
                StageHistoryRecord(
                    stage=new_stage,
                    entered_turn_id=bundle.turn_id,
                    trigger_reason=stage_reason,
                    confidence=1.0,
                )
            )
            self.current_state.conversation_stage = new_stage
            next_version += 1
            old_val_str = old_stage.value if hasattr(old_stage, "value") else str(old_stage)
            new_val_str = new_stage.value if hasattr(new_stage, "value") else str(new_stage)
            changes.append(
                StateChangeRecord(
                    state_version_before=next_version - 1,
                    state_version_after=next_version,
                    field_path="conversation_stage",
                    old_value=old_val_str,
                    new_value=new_val_str,
                    triggering_turn_id=bundle.turn_id,
                    evidence_ids=bundle.contributing_evidence_ids,
                    reason=f"Conversation stage transitioned from '{old_val_str}' to '{new_val_str}': {stage_reason}",
                    confidence=1.0,
                    timestamp_ms=bundle.timestamp_ms,
                )
            )

        # Update metadata
        self.current_state.last_updated_turn_id = bundle.turn_id
        self.current_state.last_updated_timestamp_ms = bundle.timestamp_ms
        self.current_state.state_version = next_version
        self.current_state.overall_confidence = min(
            bundle.inference_confidence,
            bundle.semantic_confidence,
            self.current_state.decision_structure.confidence,
            readiness_res.confidence,
            gate_res.confidence if gate_res else 0.50,
        )
        # Client Feedback Point 5: Detect material client turns where no structured extractor fired
        has_substantive_state = any(
            c.field_path.startswith(("objections", "contact_compliance", "decision_structure", "conversion_gate", "facts"))
            for c in changes
        )
        is_unclassified = (
            bundle.speaker_id == "client"
            and getattr(materiality, "is_material", False) is True
            and not has_substantive_state
        )
        self.current_state.unclassified_material = is_unclassified

        self.current_state.change_history.extend(changes)
        self.prior_bundle = bundle

        return self.current_state

    def _maybe_transition_dormant(
        self,
        objection: ObjectionRecord,
        current_turn_id: int,
        timestamp_ms: int = 0,
        contributing_evidence_ids: Optional[List[str]] = None,
        next_version: int = 1,
    ) -> Optional[StateChangeRecord]:
        """Evaluates whether an objection qualifies for dormancy transition based on Δturns and corroborating evidence.
        Pre-filter: Δturns >= dormancy_turn_threshold (unless superseded immediately).
        Requires corroborating evidence from _get_dormancy_evidence.
        If no evidence exists, returns None (objection stays in its current lifecycle state).
        """
        delta = current_turn_id - objection.last_updated_turn_id
        thresh = getattr(self.objections_engine, "dormancy_turn_threshold", 3)
        if delta < thresh and not getattr(objection, "superseded_by_objection_id", None):
            return None

        evidence = _get_dormancy_evidence(
            objection=objection,
            state=self.current_state,
            current_turn_id=current_turn_id,
            recent_bundles=self._bundle_history,
        )
        if evidence is None:
            return None  # stays in current lifecycle_state — no transition

        old_s = getattr(objection.lifecycle_state, "value", objection.lifecycle_state)
        objection.lifecycle_state = ObjectionLifecycleState.DORMANT
        objection.dormancy_evidence = evidence

        reason = (
            f"Concern '{objection.canonical_category}' transitioned to DORMANT: "
            f"{evidence.description} ({delta} turns since last mention)."
        )
        return StateChangeRecord(
            state_version_before=next_version,
            state_version_after=next_version + 1,
            field_path=f"objections.{objection.objection_id}.lifecycle_state",
            old_value=old_s,
            new_value=ObjectionLifecycleState.DORMANT.value,
            triggering_turn_id=current_turn_id,
            evidence_ids=contributing_evidence_ids or [],
            reason=reason,
            timestamp_ms=timestamp_ms,
        )

    def _compute_commitment(self, bundle: BehavioralSignalInputBundle) -> tuple[float, float]:
        """Client Feedback Issue #8: Computes explicit commitment score (0.0 to 1.0)
        separate from general readiness and independent of trust.
        Readiness = 'how prepared are they, generally'
        Commitment = 'what have they concretely said yes to, specifically'
        """
        # 1. Check conversion event status (Ground-truth lifecycle state strictly takes precedence)
        if self.current_state.conversion_event:
            ev = self.current_state.conversion_event
            if ev.status == ConversionEventStatus.CONFIRMED:
                return 1.0, ev.confirmation_confidence or 0.95
            elif ev.status == ConversionEventStatus.TENTATIVE:
                return 0.70, ev.confirmation_confidence or 0.85
            elif ev.status == ConversionEventStatus.CANCELLED:
                return 0.0, 1.0
            elif ev.status == ConversionEventStatus.PROPOSED:
                return 0.40, 0.75

        # 2. Check conversion gate explicit commitment flag
        if (
            self.current_state.conversion_gate
            and self.current_state.conversion_gate.explicit_commitment_detected
        ):
            return 0.85, 0.85

        # 3. If bundle already carries an explicit commitment dimension, use it as behavioral prior
        if bundle.commitment is not None and bundle.commitment.score > 0.0:
            return bundle.commitment.score, bundle.commitment.confidence

        # 4. Check client affirmative scheduling agreement in current turn
        if bundle.speaker_id == "client":
            if bundle.recurrence_type in ("scheduling_detail_repeated", "scheduling_repeated") and bundle.agreement_score >= 0.70:
                return 0.85, 0.85

        # 5. Fallback to existing state commitment if available
        cur_commit = getattr(self.current_state.dimensions, "commitment", 0.0)
        cur_conf = getattr(self.current_state.dimensions, "commitment_confidence", 0.7)
        return cur_commit, cur_conf

    def _compute_commitment_from_lifecycle(
        self,
        bundle: BehavioralSignalInputBundle,
        conv_res: Optional[ConversionEventObject],
        gate_res: Optional[Any],
    ) -> tuple[float, float]:
        """Re-syncs commitment from the newly evaluated conversion event and conversion gate."""
        if conv_res:
            if conv_res.status == ConversionEventStatus.CONFIRMED:
                return 1.0, conv_res.confirmation_confidence or 0.95
            elif conv_res.status == ConversionEventStatus.TENTATIVE:
                return 0.70, conv_res.confirmation_confidence or 0.85
            elif conv_res.status == ConversionEventStatus.CANCELLED:
                return 0.0, 1.0
            elif conv_res.status == ConversionEventStatus.PROPOSED:
                return 0.40, 0.75

        if gate_res and getattr(gate_res, "explicit_commitment_detected", False):
            return 0.85, 0.85

        if bundle.commitment is not None and bundle.commitment.score > 0.0:
            return bundle.commitment.score, bundle.commitment.confidence

        cur_commit = getattr(self.current_state.dimensions, "commitment", 0.0)
        cur_conf = getattr(self.current_state.dimensions, "commitment_confidence", 0.7)
        return cur_commit, cur_conf

    def _extract_autonomous_decision_updates(
        self,
        bundle: BehavioralSignalInputBundle,
        materiality: Any,
    ) -> Optional[Dict[str, Any]]:
        """Autonomously extracts decision structure updates from client disclosures when explicit updates are not provided."""
        if bundle.speaker_id != "client":
            return None

        text = bundle.utterance_text.lower().strip()
        result: Dict[str, Any] = {}

        # 1. Direct presence confirmation of co-decision maker / spouse (Client Principle #1)
        if any(re.search(pat, text) for pat in PRESENCE_CONFIRMATION_PATTERNS):
            existing = list(self.current_state.decision_structure.stakeholders)
            target_role = "wife" if "wife" in text else ("husband" if "husband" in text else ("spouse" if "spouse" in text else ("partner" if "partner" in text else None)))

            updated = False
            new_stakeholders = []
            for s in existing:
                match_role = (
                    (target_role and s.role in (target_role, "spouse", "partner", "co_decision_maker", "female_decision_maker" if target_role == "wife" else "male_decision_maker"))
                    or (not target_role and s.is_decision_maker)
                )
                if match_role:
                    s_up = s.model_copy(deep=True)
                    s_up.presence = "confirmed_attending"
                    s_up.notes = f"Confirmed attending by client: '{bundle.utterance_text[:60]}'"
                    new_stakeholders.append(s_up)
                    updated = True
                else:
                    new_stakeholders.append(s.model_copy(deep=True))

            if not updated and target_role:
                new_stakeholders.append(
                    DecisionStakeholder(
                        stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
                        role=target_role,
                        is_decision_maker=True,
                        presence="confirmed_attending",
                        notes=f"Confirmed attending by client: '{bundle.utterance_text[:60]}'",
                        confidence=0.90,
                    )
                )

            result["stakeholders"] = new_stakeholders
            # If no stakeholders remain absent, decision authority is present/aligned
            has_absent = any(s.presence == "absent" for s in new_stakeholders)
            if not has_absent:
                result["decision_maker_present"] = True
                result["co_decision_required"] = True
                result["primary_decision_maker"] = "sole decision maker, spouse confirmed attending"

        # 2. Direct absent decision maker patterns
        elif any(re.search(pat, text) for pat in ABSENT_DECISION_MAKER_PATTERNS):
            role = "co_decision_maker"
            if "husband" in text:
                role = "husband"
            elif "wife" in text:
                role = "wife"
            elif "partner" in text or "spouse" in text:
                role = "partner"
            elif "attorney" in text or "lawyer" in text:
                role = "attorney"
            elif "he" in text:
                role = "male_decision_maker"
            elif "she" in text:
                role = "female_decision_maker"

            stakeholder = DecisionStakeholder(
                stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
                role=role,
                is_decision_maker=True,
                presence="absent",
                notes=f"Identified as absent decision maker from client disclosure: '{bundle.utterance_text[:60]}'",
                confidence=0.85,
            )

            existing = list(self.current_state.decision_structure.stakeholders)
            if not any(s.role == role and s.presence == "absent" for s in existing):
                existing.append(stakeholder)

            result["decision_maker_present"] = False
            result["stakeholders"] = existing
            result["co_decision_required"] = True
            result["primary_decision_maker"] = "sole decision maker, spouse required for final approval"

        # 2. Affirmative confirmation of prior salesperson inquiry about third-party stakeholder
        elif (
            self.prior_bundle
            and self.prior_bundle.speaker_id == "salesperson"
            and any(w in self.prior_bundle.utterance_text.lower() for w in ["him involved", "her involved", "them involved", "husband", "wife", "partner", "spouse", "decision maker", "sign off"])
            and (
                bundle.agreement_score >= 0.60
                or any(re.search(rf"\b{aff}\b", text) for aff in ["yeah", "yes", "definitely", "sure", "absolutely", "correct", "of course"])
            )
        ):
            prior_text_l = self.prior_bundle.utterance_text.lower()
            role = "male_decision_maker" if "him" in prior_text_l else ("female_decision_maker" if "her" in prior_text_l else "co_decision_maker")
            stakeholder = DecisionStakeholder(
                stakeholder_id=f"stk_{uuid.uuid4().hex[:6]}",
                role=role,
                is_decision_maker=True,
                presence="absent",
                notes=f"Confirmed stakeholder involvement in response to agent inquiry: '{bundle.utterance_text[:60]}'",
                confidence=0.80,
            )
            existing = list(self.current_state.decision_structure.stakeholders)
            if not any(s.role == role and s.presence == "absent" for s in existing):
                existing.append(stakeholder)

            result["decision_maker_present"] = False
            result["stakeholders"] = existing
            result["co_decision_required"] = True
            result["primary_decision_maker"] = "sole decision maker, spouse required for final approval"

        # 3. Sole decision maker declaration (e.g. Turn 2)
        elif any(re.search(p, text) for p in [
            r"\b(?:i'm|i\s+am)\s+the\s+(?:one|sole\s+person)\s+(?:making|who\s+makes)\b",
            r"\bno\s+one\s+else\s+needs?\s+to\s+sign\b",
            r"\bi\s+make\s+the\s+decisions?\s+alone\b",
        ]):
            if not self.current_state.decision_structure.stakeholders:
                result["decision_maker_present"] = True
                result["primary_decision_maker"] = "sole_decision_maker"
                result["co_decision_required"] = False

        # 4. Moving timeline horizon & urgency level (e.g. Turn 4)
        if any(re.search(p, text) for p in [
            r"\b(?:sometime\s+next\s+year|moving\s+next\s+year|look\s+at\s+moving|next\s+year)\b",
        ]):
            result["timeline_horizon"] = "sometime next year"
        if any(re.search(p, text) for p in [
            r"\bnothing\s+urgent\b",
            r"\bnot\s+urgent\b",
            r"\bno\s+rush\b",
        ]):
            result["urgency_level"] = "low"

        # 5. Access and scheduling constraints (e.g. Turn 16)
        if any(re.search(p, text) for p in [
            r"\bmornings?\s+(?:don['’]?t|do\s+not)\s+(?:really\s+)?work\b",
            r"\bnot\s+available\s+in\s+the\s+mornings?\b",
        ]):
            cur_constraints = list(self.current_state.decision_structure.access_constraints)
            if "Mornings unavailable" not in cur_constraints:
                cur_constraints.append("Mornings unavailable")
                result["access_constraints"] = cur_constraints

        return result or None

    def _extract_autonomous_fact_updates(
        self,
        bundle: BehavioralSignalInputBundle,
        materiality: Any,
    ) -> List[Dict[str, Any]]:
        """Autonomously extracts persistent facts (meeting confirmations, contact preferences, etc.) from dialogue evidence."""
        if bundle.speaker_id != "client":
            return []

        updates: List[Dict[str, Any]] = []
        text_lower = bundle.utterance_text.lower().strip()
        existing_keys = {f.fact_key for f in self.current_state.facts if f.status == "active"}

        # 1. Contact preference (from bundle metadata or explicit client statement)
        if "contact_preference" not in existing_keys:
            if bundle.contact_preference != "none":
                updates.append({
                    "category": "preference",
                    "fact_key": "contact_preference",
                    "fact_value": bundle.contact_preference,
                    "confidence": bundle.contact_preference_confidence or 0.85,
                    "notes": f"Communication preference declared: {bundle.contact_preference_details or bundle.contact_preference}",
                })
            elif any(re.search(p, text_lower) for p in [
                r"\b(?:don't|do\s+not)\s+(?:start\s+)?texting\s+me\s+every\s+day\b",
                r"\bavoid\s+texting\s+me\s+daily\b",
            ]):
                updates.append({
                    "category": "preference",
                    "fact_key": "contact_preference",
                    "fact_value": "No daily texting",
                    "confidence": 0.85,
                    "notes": f"Communication preference declared: '{bundle.utterance_text[:60]}'",
                })

        # 2. Confirmed meeting / appointment time
        # Case A: Prospect explicitly mentions day and time (must not be negated or hypothetical)
        time_day_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b.*?\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            text_lower,
        )
        is_negated_or_hypothetical = any(re.search(pat, text_lower) for pat in [
            r"\b(?:doesn't|does\s+not|won't|will\s+not|can't|cannot|couldn't|could\s+not)\s+(?:work|make\s+it|do\s+it)\b",
            r"\bhypothetical(?:ly)?\b",
            r"\blet['’]?s\s+say\b",
            r"\bwhat\s+if\b",
            r"\bsuppose\b",
        ])
        if time_day_match and not is_negated_or_hypothetical:
            new_val = format_time_slot(time_day_match.group(0).strip())
            # If earlier tentative meeting/walkthrough fact exists, supersede it directly
            tentative_f = next((f for f in self.current_state.facts if f.fact_key in ("tentative_meeting_time", "walkthrough_timing") and f.status == "active"), None)
            if tentative_f:
                self.facts_manager.supersede_fact(
                    old_fact_id=tentative_f.fact_id,
                    new_fact_value=new_val,
                    source_turn_id=bundle.turn_id,
                    timestamp_ms=bundle.timestamp_ms,
                    notes=f"Tentative walkthrough timing superseded by confirmed meeting '{new_val}'",
                )

            if "confirmed_meeting_time" not in existing_keys:
                updates.append({
                    "category": "timeline",
                    "fact_key": "confirmed_meeting_time",
                    "fact_value": new_val,
                    "confidence": 0.85,
                    "notes": f"Prospect confirmed meeting time directly: '{bundle.utterance_text[:60]}'",
                })
            else:
                active_f = next((f for f in self.current_state.facts if f.fact_key == "confirmed_meeting_time" and f.status == "active"), None)
                if active_f and active_f.fact_value.strip().lower() != new_val.strip().lower():
                    is_clarification = is_ampm_clarification(active_f.fact_value, new_val)
                    note_text = f"Meeting time clarified to '{new_val}'" if is_clarification else f"Meeting rescheduled to '{new_val}'"
                    self.facts_manager.supersede_fact(
                        old_fact_id=active_f.fact_id,
                        new_fact_value=new_val,
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        notes=note_text,
                    )
        elif self.prior_bundle and self.prior_bundle.speaker_id == "salesperson":
            # Case B: Salesperson proposed a day/time and prospect confirmed affirmatively without hedging
            prior_text_lower = self.prior_bundle.utterance_text.lower()
            prop_match = re.search(
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b.*?\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
                prior_text_lower,
            )
            hedged_patterns = [
                r"\b(?:could|might)\s+work\b",
                r"\b(?:could|might)\s+(?:be\s+able\s+to|possibly)\b",
                r"\btentative(?:ly)?\b",
                r"\bpossibly\b",
                r"\blet\s+me\s+(?:check|see|think)\b",
                r"\bnot\s+(?:100%|sure|certain)\b",
                r"\bif\s+(?:that|it)\s+works\b",
            ]
            is_hedged_fact = any(re.search(pat, text_lower) for pat in hedged_patterns)
            is_affirmative = (
                bundle.agreement_score >= 0.60
                or any(re.search(rf"\b{aff}\b", text_lower) for aff in ["yeah", "yes", "definitely", "sure", "absolutely", "works", "perfect", "sounds good"])
            )
            if prop_match and is_affirmative and not is_hedged_fact:
                raw_time_str = format_time_slot(prop_match.group(0).strip())
                norm_time = raw_time_str
                for word, num in [("One", "1:00 PM"), ("Two", "2:00 PM"), ("Three", "3:00 PM"), ("Four", "4:00 PM"), ("Five", "5:00 PM")]:
                    if f"At {word}" in norm_time:
                        norm_time = norm_time.replace(f"At {word}", f"at {num}")

                tentative_f = next((f for f in self.current_state.facts if f.fact_key in ("tentative_meeting_time", "walkthrough_timing") and f.status == "active"), None)
                if tentative_f:
                    self.facts_manager.supersede_fact(
                        old_fact_id=tentative_f.fact_id,
                        new_fact_value=norm_time,
                        source_turn_id=bundle.turn_id,
                        timestamp_ms=bundle.timestamp_ms,
                        notes=f"Tentative walkthrough timing superseded by confirmed meeting '{norm_time}'",
                    )

                if "confirmed_meeting_time" not in existing_keys:
                    updates.append({
                        "category": "timeline",
                        "fact_key": "confirmed_meeting_time",
                        "fact_value": norm_time,
                        "confidence": 0.85,
                        "notes": f"Prospect confirmed proposed time '{raw_time_str}' from salesperson proposal.",
                    })

        # 3. Decision to Stay / Cancellation of sale
        if any(re.search(p, text_lower) for p in DECISION_TO_STAY_PATTERNS) and "decision_to_stay" not in existing_keys:
            updates.append({
                "category": "timeline",
                "fact_key": "decision_to_stay",
                "fact_value": "Prospect decided to stay in home / cancel sale",
                "confidence": 0.90,
                "notes": f"Decision to stay declared by prospect: '{bundle.utterance_text[:60]}'",
            })

        # 4. Spousal / Absent Decision Maker Involvement
        if any(re.search(p, text_lower) for p in ABSENT_DECISION_MAKER_PATTERNS) and not any(re.search(p, text_lower) for p in PRESENCE_CONFIRMATION_PATTERNS) and "spouse_involvement" not in existing_keys:
            updates.append({
                "category": "decision_maker",
                "fact_key": "spouse_involvement",
                "fact_value": "Spouse/wife involvement declared for decisions",
                "confidence": 0.85,
                "notes": f"Spouse involvement declared by prospect: '{bundle.utterance_text[:60]}'",
            })

        # 4b. Spouse / Decision Maker Presence Confirmation (Client Principle #1)
        if any(re.search(p, text_lower) for p in PRESENCE_CONFIRMATION_PATTERNS):
            active_spouse_f = next((f for f in self.current_state.facts if f.fact_key == "spouse_involvement" and f.status == "active"), None)
            if active_spouse_f:
                self.facts_manager.supersede_fact(
                    old_fact_id=active_spouse_f.fact_id,
                    new_fact_value="Spouse confirmed attending meeting",
                    source_turn_id=bundle.turn_id,
                    timestamp_ms=bundle.timestamp_ms,
                    notes=f"Spouse confirmed attending: '{bundle.utterance_text[:60]}'",
                )

        # 5. Tentative Timeline Horizon (e.g. Turn 4)
        if "timeline_horizon" not in existing_keys and any(re.search(p, text_lower) for p in [
            r"\b(?:sometime\s+next\s+year|moving\s+next\s+year|look\s+at\s+moving\s+sometime\s+next\s+year|next\s+year)\b",
        ]):
            updates.append({
                "category": "timeline",
                "fact_key": "timeline_horizon",
                "fact_value": "Sometime next year",
                "confidence": 0.85,
                "notes": f"Tentative moving timeline declared by prospect: '{bundle.utterance_text[:60]}'",
            })

        # 6. Tentative Meeting / Walkthrough Window (e.g. Turn 6, Turn 9)
        if "confirmed_meeting_time" not in existing_keys and "tentative_meeting_time" not in existing_keys and any(re.search(p, text_lower) for p in [
            r"\b(?:maybe\s+next\s+week|sometime\s+next\s+week|next\s+week\s+could\s+work|think\s+about\s+it)\b",
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\s+(?:could|might)\s+work\b",
        ]):
            day_match = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b", text_lower)
            day_str = f"{day_match.group(0).title()} (tentative)" if day_match else "Maybe next week (tentative)"
            updates.append({
                "category": "timeline",
                "fact_key": "tentative_meeting_time",
                "fact_value": day_str,
                "confidence": 0.70,
                "notes": f"Tentative meeting timing expressed by prospect: '{bundle.utterance_text[:60]}'",
            })

        # 7. Scheduling Constraints / Availability Restrictions (e.g. Turn 16)
        if "scheduling_constraint" not in existing_keys and any(re.search(p, text_lower) for p in [
            r"\bmornings?\s+(?:don['’]?t|do\s+not)\s+(?:really\s+)?work\b",
            r"\bnot\s+available\s+in\s+the\s+mornings?\b",
            r"\bafternoons?\s+(?:only|preferred|work\s+better)\b",
        ]):
            updates.append({
                "category": "logistical",
                "fact_key": "scheduling_constraint",
                "fact_value": "Mornings unavailable / afternoons preferred",
                "confidence": 0.85,
                "notes": f"Scheduling constraint declared by prospect: '{bundle.utterance_text[:60]}'",
            })

        return updates


    def get_conversion_event_history(self) -> List[ConversionEventObject]:
        """Returns the complete chronological history of conversion events (both superseded and active)."""
        return list(self._conversion_events)

    def get_active_conversion_event(self) -> Optional[ConversionEventObject]:
        """Returns the current active (unsuperseded) conversion event, if any."""
        for ev in reversed(self._conversion_events):
            if ev.superseded_by_event_id is None:
                return ev
        return self.current_state.conversion_event

    def get_deal_disposition_history(self) -> List[DealDispositionRecord]:
        """Returns the complete chronological history of deal disposition records."""
        return list(self._deal_dispositions)

    def get_active_deal_disposition(self) -> Optional[DealDispositionRecord]:
        """Returns the current active (unsuperseded) deal disposition record."""
        for d in reversed(self._deal_dispositions):
            if d.superseded_by_id is None:
                return d
        return self.deal_disposition

    def get_deal_milestone_status(self) -> str:
        """Returns the high-level deal milestone status (Issue #6)."""
        from .conversation_presentation import compute_deal_milestone_status
        return compute_deal_milestone_status(self.current_state)

    def get_open_concerns(self) -> List[Dict[str, Any]]:
        """Returns live objections requiring agent awareness in the pre-call dossier (Issue #6)."""
        from .conversation_presentation import compute_open_concerns
        return compute_open_concerns(self.current_state)

    def get_conversion_presentation(self) -> Dict[str, Any]:
        """Returns complete presentation dictionary for API serialization (Issue #6)."""
        from .conversation_presentation import build_conversion_presentation
        return build_conversion_presentation(self.current_state)
