"""
Regime Detector - Dynamically detects market regime (Bull/Bear/Crab)
"""
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    CRAB = "crab"  # Choppy/sideways

@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float  # 0-1
    btc_trend: float  # % change
    volatility: float  # ATR or similar
    
class RegimeDetector:
    """Detects current market regime based on BTC trend and volatility."""
    
    def __init__(self):
        self.history = []
        
    def detect(self, btc_trend: float, volatility: Optional[float] = None) -> RegimeState:
        """
        Detect market regime based on BTC trend.
        
        Args:
            btc_trend: BTC 24h % change
            volatility: Optional volatility measure
            
        Returns:
            RegimeState with regime classification
        """
        # Simple regime classification based on BTC trend
        if btc_trend > 2.0:
            regime = MarketRegime.BULL
            confidence = min(0.5 + (btc_trend - 2.0) / 10.0, 1.0)
        elif btc_trend < -2.0:
            regime = MarketRegime.BEAR
            confidence = min(0.5 + abs(btc_trend + 2.0) / 10.0, 1.0)
        else:
            regime = MarketRegime.CRAB
            confidence = 1.0 - abs(btc_trend) / 2.0
            
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            btc_trend=btc_trend,
            volatility=volatility or 0.0
        )
        
        self.history.append(state)
        if len(self.history) > 100:
            self.history.pop(0)
            
        return state
    
    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get parameter adjustments for current regime.
        
        Returns:
            Dict with adjustment multipliers
        """
        if regime == MarketRegime.BULL:
            return {
                'long_bias': 1.2,  # Favor longs
                'short_bias': 0.8,  # Reduce shorts
                'min_score_adj': -5,  # Lower threshold
            }
        elif regime == MarketRegime.BEAR:
            return {
                'long_bias': 0.8,  # Reduce longs
                'short_bias': 1.2,  # Favor shorts
                'min_score_adj': -5,  # Lower threshold
            }
        else:  # CRAB
            return {
                'long_bias': 1.0,
                'short_bias': 1.0,
                'min_score_adj': 0,  # Keep threshold
            }

