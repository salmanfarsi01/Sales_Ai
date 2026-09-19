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


DEFAULT_CONVERSION_BLOCKING_CONFIG = ConversionBlockingConfig()
