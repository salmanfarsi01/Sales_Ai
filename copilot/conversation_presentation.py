"""Presentation Layer for ConversationState (Issue #6).

Decouples report-facing derived fields, disposition labels, and pre-call dossier summaries
from raw momentum and readiness scoring math.

Guarantees:
1. Milestone Priority: Confirmed / tentative conversion events take precedence over gate status.
   Once an appointment is booked, the header reads 'APPOINTMENT CONFIRMED — <time>', not 'OPEN (Ready to Close)'.
2. Open Concerns Dossier: Distinguishes between micro-conversion (appointment booked) and objection resolution.
   Live objections (active, partially_resolved, reactivated) are surfaced alongside the confirmed appointment,
   while dormant and resolved concerns are cleanly excluded.
"""

from typing import Any, Dict, List, Optional
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    ConversionEventObject,
    ConversionEventStatus,
    ObjectionLifecycleState,
)

CATEGORY_DISPLAY_MAPPING: Dict[str, str] = {
    "general_hesitation": "financial net proceeds / general hesitation",
    "commission_fee": "commission fee",
    "timing": "timeline / schedule",
    "price_reduction": "price reduction",
}

LIFECYCLE_DISPLAY_MAPPING: Dict[str, str] = {
    "partially_resolved": "partially resolved",
    "partially_addressed": "partially resolved",
    "active": "active",
    "unresolved": "active",
    "reactivated": "reactivated",
}


def compute_deal_milestone_status(state: ConversationStateSnapshot) -> str:
    """Computes high-level deal milestone status with strict priority ordering.

    Priority:
    1. conversion_event.status == CONFIRMED -> 'appointment_confirmed'
    2. conversion_event.status == TENTATIVE -> 'appointment_tentative'
    3. conversion_gate.is_open == True      -> 'ready_to_close'
    4. conversion_gate.is_open == False     -> 'blocked'
    5. Fallback                             -> 'pursuing'
    """
    ce = state.conversion_event
    if not ce and getattr(state, "conversion_events", None):
        confirmed_evs = [
            e for e in state.conversion_events
            if getattr(e, "status", None) in (ConversionEventStatus.CONFIRMED, "confirmed")
        ]
        ce = confirmed_evs[-1] if confirmed_evs else state.conversion_events[-1]

    gate = state.conversion_gate

    if ce and getattr(ce, "status", None) in (ConversionEventStatus.CONFIRMED, "confirmed"):
        return "appointment_confirmed"
    if ce and getattr(ce, "status", None) in (ConversionEventStatus.TENTATIVE, "tentative"):
        return "appointment_tentative"
    if gate and getattr(gate, "is_open", False):
        return "ready_to_close"
    if gate and not getattr(gate, "is_open", False):
        return "blocked"
    return "pursuing"


def format_milestone_label(milestone: str, ce: Optional[ConversionEventObject] = None) -> str:
    """Formats human-facing header badge string for the deal milestone."""
    start_at = getattr(ce, "start_at", None) if ce else None

    labels = {
        "appointment_confirmed": f"APPOINTMENT CONFIRMED — {start_at or 'confirmed'}",
        "appointment_tentative": f"TENTATIVE — {start_at or 'time pending'}",
        "ready_to_close": "OPEN (Ready to Close)",
        "blocked": "BLOCKED",
        "pursuing": "PURSUING",
    }
    return labels.get(milestone, milestone.upper())


def compute_open_concerns(state: ConversationStateSnapshot) -> List[Dict[str, Any]]:
    """Extracts live objections requiring agent awareness/preparation in the pre-call dossier.

    Includes: ACTIVE, PARTIALLY_RESOLVED, REACTIVATED.
    Excludes: DORMANT, RESOLVED, SUPERSEDED.
    """
    live_states = {
        ObjectionLifecycleState.ACTIVE,
        ObjectionLifecycleState.PARTIALLY_RESOLVED,
        ObjectionLifecycleState.REACTIVATED,
        "active",
        "unresolved",
        "partially_resolved",
        "partially_addressed",
        "reactivated",
    }

    concerns: List[Dict[str, Any]] = []
    for o in getattr(state, "objections", []):
        st = o.lifecycle_state
        if st in live_states:
            concerns.append({
                "category": o.canonical_category,
                "lifecycle_state": getattr(st, "value", str(st)),
                "latest_statement": o.latest_statement,
                "deferred_to_meeting": getattr(o, "deferred_to_meeting", False),
                "retained_for_followup": getattr(o, "retained_for_followup", False),
            })
    return concerns


def format_open_concerns_summary(concerns: List[Dict[str, Any]]) -> str:
    """Formats open concerns into a single-line summary string.

    Example:
    'Open Concerns: 1 (financial net proceeds / general hesitation — partially resolved)'
    Returns empty string if concerns list is empty.
    """
    if not concerns:
        return ""

    n = len(concerns)
    parts = []
    for c in concerns:
        cat = c.get("category", "concern")
        cat_disp = CATEGORY_DISPLAY_MAPPING.get(cat, cat.replace("_", " "))
        st = str(c.get("lifecycle_state", "active")).lower()
        st_disp = LIFECYCLE_DISPLAY_MAPPING.get(st, st.replace("_", " "))
        if c.get("deferred_to_meeting"):
            parts.append(f"{cat_disp} — deferred to meeting")
        else:
            parts.append(f"{cat_disp} — {st_disp}")

    return f"Open Concerns: {n} ({'; '.join(parts)})"


def build_conversion_presentation(state: ConversationStateSnapshot) -> Dict[str, Any]:
    """Builds a complete, self-contained presentation dictionary for API serialization."""
    ce = state.conversion_event
    if not ce and getattr(state, "conversion_events", None):
        confirmed_evs = [
            e for e in state.conversion_events
            if getattr(e, "status", None) in (ConversionEventStatus.CONFIRMED, "confirmed")
        ]
        ce = confirmed_evs[-1] if confirmed_evs else state.conversion_events[-1]

    milestone = compute_deal_milestone_status(state)
    label = format_milestone_label(milestone, ce)
    concerns = compute_open_concerns(state)
    summary = format_open_concerns_summary(concerns)

    return {
        "deal_milestone_status": milestone,
        "milestone_label": label,
        "open_concerns": concerns,
        "open_concerns_summary": summary,
    }
