"""Scoring Engine - Main entrypoint for multi-dimensional scoring."""

from typing import Tuple

from .model import ScoreComponents, clamp
from .context import SignalContext
from .modules import (
    score_liquidity,
    score_regime,
    score_structure,
    score_portfolio,
    score_time_of_day,
    score_symbol_rating,
)
from .config import SCORE_CAPS, SCORE_MIN, SCORE_MAX, SCORING_BASE_SCALE


def build_score_components(ctx: SignalContext) -> ScoreComponents:
    """
    Compute raw (uncapped) components from context.
    
    Args:
        ctx: SignalContext with all required data
    
    Returns:
        ScoreComponents with raw (uncapped) values
    """
    # Apply global calibration scale to base score (before bonuses are added)
    # This deflates scores to prevent "everything is 90+" fantasy while keeping ranking intact
    base = ctx.base_score * SCORING_BASE_SCALE
    # Clamp to sane range
    if base < 0.0:
        base = 0.0
    elif base > 100.0:
        base = 100.0
    
    return ScoreComponents(
        base=base,
        liquidity=score_liquidity(ctx),
        regime=score_regime(ctx),
        structure=score_structure(ctx),
        portfolio=score_portfolio(ctx),
        time_of_day=score_time_of_day(ctx),
        symbol_rating=score_symbol_rating(ctx),
    )


def compute_final_score(ctx: SignalContext) -> Tuple[float, ScoreComponents, ScoreComponents]:
    """
    Compute final score with proper caps.
    
    Args:
        ctx: SignalContext with all required data
    
    Returns:
        final_score: float in [0,100]
        raw_components: ScoreComponents (before per-module caps)
        capped_components: ScoreComponents (after per-module caps, before global clamp)
    """
    raw = build_score_components(ctx)
    capped = raw.capped(SCORE_CAPS)

    raw_total = capped.total_raw()
    final_score = clamp(raw_total, SCORE_MIN, SCORE_MAX)

    return final_score, raw, capped

