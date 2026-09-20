from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field


class ConversionBlockingConfig(BaseModel):
    """Named, versioned configuration specifying which canonical objection categories
    actually block which conversion targets.

    Follows the same architectural pattern as InferenceScoringConfig and ConversationScoringConfig.
    Allows different conversion goals (e.g. initial meeting vs. signing an exclusive listing agreement)
    to enforce logically relevant objection gates rather than indiscriminate universal blocking.
    """
    version: str = "1.0.0"
    blocking_categories: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "appointment": ["boundary"],  # only a hard boundary blocks scheduling an initial meeting/walkthrough
            "signed_listing_agreement": ["commission_fee", "pricing_value", "boundary"],
            "permission_to_follow_up": ["boundary"],
        }
    )
    escalate_on_recurrence: bool = Field(
        default=True,
        description="If True, repeated occurrences of a non-blocking objection escalate to blocking",
    )
    max_non_blocking_recurrence: int = Field(
        default=2,
        description="Maximum recurrence count an objection can reach before escalating to blocking (3rd occurrence escalates)",
    )
    non_escalating_categories: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "appointment": ["commission_fee"],  # commission fee is discussed at the meeting and never blocks appointment
        },
        description="Categories exempt from recurrence escalation for a given conversion target",
    )


DEFAULT_CONVERSION_BLOCKING_CONFIG = ConversionBlockingConfig()
