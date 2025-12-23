from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True, slots=True)
class MVPProfile:
    """A lightweight policy profile (arm).

    MVP rule: profiles change only *threshold policy*, not the core score.
    """

    arm_id: str
    min_score_delta: float = 0.0
    min_strength_delta: float = 0.0


PROFILES: Dict[str, MVPProfile] = {
    # Balanced (operator baseline)
    "balanced": MVPProfile(arm_id="balanced", min_score_delta=0.0, min_strength_delta=0.0),
    # Conservative: be a bit pickier
    "conservative": MVPProfile(arm_id="conservative", min_score_delta=+2.0, min_strength_delta=+0.02),
    # Aggressive: slightly looser to keep trade flow
    "aggressive": MVPProfile(arm_id="aggressive", min_score_delta=-2.0, min_strength_delta=-0.02),
}
