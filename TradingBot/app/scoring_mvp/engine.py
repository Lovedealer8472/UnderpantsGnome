from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..freshness_scorer import FreshnessScorer
from .types import MVPScoreResult


class MVPScoringEngine:
    """MVP scoring engine.

    Today: wraps FreshnessScorer as the canonical score.
    Later: we can add small, bounded adjusters here without changing the call sites.
    """

    def __init__(self, freshness_scorer: Optional[FreshnessScorer] = None):
        self._freshness = freshness_scorer or FreshnessScorer()

    def score(
        self,
        *,
        symbol: str,
        side: str,
        pct_change_24h: float,
        volume_24h: float,
        spread_bps: float,
        orderbook: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
        recent_trades: Optional[list] = None,
    ) -> MVPScoreResult:
        # Adapt to FreshnessScorer signature (may not have recent_trades/side params in older versions)
        try:
            # Try with all params first
            final_score, components = self._freshness.score_signal(
                symbol=symbol,
                pct_change_24h=pct_change_24h,
                volume_24h=volume_24h,
                spread_bps=spread_bps,
                orderbook=orderbook,
                indicators=indicators,
                latency_ms=latency_ms,
                recent_trades=recent_trades,
                side=side,
            )
        except TypeError:
            # Fallback: call without optional params if signature doesn't support them
            final_score, components = self._freshness.score_signal(
                symbol=symbol,
                pct_change_24h=pct_change_24h,
                volume_24h=volume_24h,
                spread_bps=spread_bps,
                orderbook=orderbook,
                indicators=indicators,
                latency_ms=latency_ms,
            )
        return MVPScoreResult(final_score=float(final_score), components=dict(components or {}))


def apply_profile_thresholds(
    *,
    base_min_score: float,
    base_min_strength: float,
    profile_min_score_delta: float,
    profile_min_strength_delta: float,
) -> Tuple[float, float]:
    # Keep sane bounds here; hard bounds are enforced elsewhere too.
    min_score = float(base_min_score) + float(profile_min_score_delta)
    min_strength = float(base_min_strength) + float(profile_min_strength_delta)
    min_score = max(0.0, min(100.0, min_score))
    min_strength = max(0.0, min(1.0, min_strength))
    return min_score, min_strength
