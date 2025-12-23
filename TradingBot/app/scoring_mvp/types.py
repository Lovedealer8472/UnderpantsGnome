from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class MVPScoreResult:
    """Result from MVP scoring engine.

    - final_score: 0-100
    - components: explainable breakdown
    - arm_id: optional policy/profile id (for bandit)
    """

    final_score: float
    components: Dict[str, Any]
    arm_id: Optional[str] = None

    # Policy metadata (filled by the caller/policy layer)
    effective_min_score: Optional[float] = None
    effective_min_strength: Optional[float] = None
    would_enter: Optional[bool] = None
