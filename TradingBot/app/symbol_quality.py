"""
Symbol Quality Analyzer - Simplified
Basic symbol filtering to remove obviously bad symbols.
"""

from typing import List
from dataclasses import dataclass

@dataclass
class QualityResult:
    passed: bool
    reason: str

class SymbolQualityAnalyzer:
    """
    Simple symbol quality check.
    Only filters out obviously problematic symbols.
    """

    def __init__(self):
        # Simple thresholds
        self.MAX_NOISE_RATIO = 0.8  # Max 80% noise (too choppy)
        self.MIN_VOLATILITY = 0.001  # Min 0.1% volatility (dead market)
        
    def analyze(self, candles: List[List[float]]) -> QualityResult:
        """
        Simple quality check for symbols.

        Args:
            candles: List of [timestamp, open, high, low, close, volume]

        Returns:
            QualityResult with pass/fail and reason
        """
        if not candles or len(candles) < 20:
            return QualityResult(False, "Not enough data")

        # Extract recent closes for simple checks
        closes = [c[4] for c in candles[-20:]]

        # 1. Check for dead market (no movement)
        price_range = max(closes) - min(closes)
        avg_price = sum(closes) / len(closes)

        if avg_price == 0:
            return QualityResult(False, "Invalid prices")

        volatility = price_range / avg_price

        if volatility < self.MIN_VOLATILITY:
            return QualityResult(False, "No volatility")

        # 2. Check for excessive noise (choppy movement)
        if price_range == 0:
            # All prices identical - completely dead market
            return QualityResult(False, "No price movement")

        noise_ratio = 1.0 - (abs(closes[-1] - closes[0]) / price_range)
        if noise_ratio > self.MAX_NOISE_RATIO:
            return QualityResult(False, "Too noisy")

        return QualityResult(True, "OK")


