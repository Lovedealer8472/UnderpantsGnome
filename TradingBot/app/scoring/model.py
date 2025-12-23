"""Score Components Model - Core dataclasses for multi-dimensional scoring."""

from dataclasses import dataclass, asdict
from typing import Dict


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return lo if value < lo else hi if value > hi else value


@dataclass
class ScoreComponents:
    """
    Individual score components (raw, before per-module caps).
    
    base: core signal score (0–100) from your existing primary scoring logic.
    Others: additive adjustments (+/- 5-20 points each).
    """
    base: float = 0.0
    liquidity: float = 0.0
    regime: float = 0.0
    structure: float = 0.0
    portfolio: float = 0.0
    time_of_day: float = 0.0
    symbol_rating: float = 0.0

    def capped(self, caps: Dict[str, tuple]) -> "ScoreComponents":
        """
        Apply per-module caps to components.
        
        Args:
            caps: Dict mapping component name to (min, max) tuple
        
        Returns:
            New ScoreComponents with capped values
        """
        data = asdict(self)
        out = {}

        for name, value in data.items():
            if name == "base":
                # base is already 0–100 from primary engine; do not hard-cap here
                out[name] = value
                continue
            lo, hi = caps[name]
            out[name] = clamp(value, lo, hi)

        return ScoreComponents(**out)

    def total_raw(self) -> float:
        """Calculate total raw score (before global clamp)."""
        return (
            self.base
            + self.liquidity
            + self.regime
            + self.structure
            + self.portfolio
            + self.time_of_day
            + self.symbol_rating
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return asdict(self)

