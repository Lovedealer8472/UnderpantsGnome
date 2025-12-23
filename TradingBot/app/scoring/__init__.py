"""Multi-Dimensional Scoring System - Clean, capped, modular."""

from .engine import compute_final_score
from .context import SignalContext
from .model import ScoreComponents
from .adapter import build_signal_context
from .config import (
    SCORE_CAPS, SCORE_MIN, SCORE_MAX,
    BASE_SCORE_MIN, BASE_SCORE_MAX,
    MIN_SIGNAL_SCORE, SIGNAL_PERCENTILE_THRESHOLD
)

__all__ = [
    'compute_final_score',
    'SignalContext',
    'ScoreComponents',
    'build_signal_context',
    'SCORE_CAPS',
    'SCORE_MIN',
    'SCORE_MAX',
    'BASE_SCORE_MIN',
    'BASE_SCORE_MAX',
    'MIN_SIGNAL_SCORE',
    'SIGNAL_PERCENTILE_THRESHOLD',
]

