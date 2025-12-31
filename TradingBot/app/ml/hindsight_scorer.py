"""
Hindsight ML Scorer - Integration of Master Hindsight ML System
Replaces/enhances existing ML scoring with trained hindsight models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Import the master predictor
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from LIVE_PREDICTOR import MasterPredictor
    HAS_HINDSIGHT_ML = True
except Exception as e:
    HAS_HINDSIGHT_ML = False
    print(f"[WARNING] Hindsight ML not available: {e}")


class HindsightMLScorer:
    """
    Wrapper for Master Hindsight ML models.
    Provides scoring compatible with existing bot infrastructure.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize hindsight ML scorer.
        
        Args:
            model_dir: Path to model directory (default: latest)
        """
        self.logger = logging.getLogger(__name__)
        self.predictor = None
        self.enabled = False
        
        # Check config flag first
        from ..config import USE_HINDSIGHT_ML
        if not USE_HINDSIGHT_ML:
            self.logger.warning("Hindsight ML disabled in config (USE_HINDSIGHT_ML=0) - using fallback scoring")
            return
        
        if not HAS_HINDSIGHT_ML:
            self.logger.warning("Hindsight ML models not available - using fallback scoring")
            return
        
        try:
            self.predictor = MasterPredictor(model_dir=model_dir)
            self.enabled = True
            self.logger.info(f"[HINDSIGHT_ML] Loaded models from: {self.predictor.model_dir}")
        except Exception as e:
            self.logger.error(f"[HINDSIGHT_ML] Failed to load models: {e}")
            self.enabled = False
    
    def score_signal(
        self,
        symbol: str,
        pct_change_24h: float,
        volume_24h: float,
        spread_bps: float,
        orderbook: Optional[Dict] = None,
        indicators: Optional[Dict] = None,
        latency_ms: float = 0.0,
        btc_trend: Optional[float] = None,
        funding_rate: Optional[float] = None,
        side: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Score a signal using hindsight ML models.
        
        Returns:
            (score, components) where score is 0-100
        """
        
        if not self.enabled or not self.predictor:
            # Fallback to simple heuristic
            return self._fallback_score(
                pct_change_24h, volume_24h, spread_bps, 
                indicators, side
            )
        
        try:
            # Calculate preliminary score for ML input variance
            # This gives the model differentiation between signals
            preliminary_score = self._calculate_preliminary_score(
                pct_change_24h, spread_bps, indicators, side
            )
            
            # Prepare signal dict for predictor
            signal_dict = {
                'symbol': symbol,
                'rsi': indicators.get('rsi', 50) if indicators else 50,
                'adx': indicators.get('adx', 20) if indicators else 20,
                'side': side or 'long',
                'score': preliminary_score  # FIX: Calculated score, not hardcoded 50
            }
            
            # Prepare market state
            market_state = {
                'atr_pct': indicators.get('atr_pct', 0.015) if indicators else 0.015,
                'volume_24h': volume_24h,
                'spread_bps': spread_bps,
                'btc_trend': btc_trend or 0.0
            }
            
            # Get ML prediction
            decision = self.predictor.predict_entry(signal_dict, market_state)
            
            # If action is skip, return low score
            if decision['action'] == 'skip':
                return 10.0, {
                    'action': 'skip',
                    'reason': decision.get('reason', 'unknown'),
                    'regime': decision.get('regime', -1),
                    'model_scores': {}
                }
            
            # Extract numeric confidence (entry_probability)
            entry_prob = decision.get('entry_probability', 0.5)
            
            # Extract components
            components = {
                'action': decision['action'],
                'confidence': entry_prob,  # Use numeric probability
                'regime': decision['regime'],
                'regime_name': decision['regime_name'],
                'optimal_sl': decision.get('optimal_sl_pct', 0.015),
                'expected_hold_time': decision.get('expected_hold_time_min', 60),
                'expected_r': decision.get('regime_expected_r', 0),
                'symbol_profile': decision.get('symbol_profile', {})
            }
            
            # ============================================================
            # INTELLIGENT SCORE CALCULATION
            # ============================================================
            # PROBLEM: entry_prob is often 80-97% for Regime 0
            # This means all signals end up at score 100 after clamping
            # SOLUTION: Use preliminary_score (which VARIES 20-80) as primary differentiator
            # Boost with entry_prob and regime adjustments
            
            # Preliminary score is our primary differentiator (20-80 range)
            # Scale it to 30-70 range for base score
            base_score = 30 + (preliminary_score - 20) * 0.67  # Maps 20-80 to 30-70
            
            # Entry probability bonus (0-20 points)
            # High confidence adds bonus, low confidence adds penalty
            confidence_bonus = (entry_prob - 0.5) * 40  # 0.5 = 0, 1.0 = +20, 0.0 = -20
            
            # Regime adjustment
            regime_adjustment = {
                0: 10,    # HIGH_PERFORMANCE: bonus
                1: 5,     # PROFITABLE: slight bonus
                2: -15,   # DANGEROUS: penalty
                3: 0,     # NEUTRAL: neutral
                4: -25    # VERY_DANGEROUS: heavy penalty
            }.get(decision['regime'], 0)
            
            # Combine: base (30-70) + confidence (+-20) + regime (-25 to +10)
            final_score = base_score + confidence_bonus + regime_adjustment
            
            # Clamp to 10-95 (leave room at top for truly exceptional signals)
            final_score = max(10, min(95, final_score))
            
            # Add ML probabilities for confidence filter
            components['model_scores'] = {
                'hindsight_ml': entry_prob
            }
            
            self.logger.debug(
                f"[HINDSIGHT_ML] {symbol} {side}: "
                f"score={final_score:.1f}, "
                f"action={decision['action']}, "
                f"regime={decision['regime_name']}, "
                f"conf={entry_prob:.2f}"
            )
            
            return final_score, components
            
        except Exception as e:
            self.logger.error(f"[HINDSIGHT_ML] Prediction failed for {symbol}: {e}")
            return self._fallback_score(
                pct_change_24h, volume_24h, spread_bps, 
                indicators, side
            )
    
    def _fallback_score(
        self,
        pct_change_24h: float,
        volume_24h: float,
        spread_bps: float,
        indicators: Optional[Dict],
        side: Optional[str]
    ) -> Tuple[float, Dict]:
        """
        Simple heuristic fallback when ML unavailable.
        """
        score = 50.0
        
        # ATR bonus
        if indicators:
            atr_pct = indicators.get('atr_pct', 0) or 0  # Handle None
            if atr_pct >= 2.0:
                score += 20
            elif atr_pct >= 1.5:
                score += 10
            elif atr_pct >= 1.0:
                score += 5
            elif atr_pct < 0.5:
                score -= 10
            
            # RSI extremes
            rsi = indicators.get('rsi', 50)
            if rsi < 25 or rsi > 75:
                score += 5
        
        # Momentum bonus
        if abs(pct_change_24h) > 0.05:
            score += 10
        
        # Spread penalty
        if spread_bps > 50:
            score -= 5
        
        score = max(35, min(85, score))
        
        return score, {'fallback': True, 'model_scores': {}}
    
    def _calculate_preliminary_score(
        self, 
        pct_change_24h: float,
        spread_bps: float, 
        indicators: Optional[Dict],
        side: Optional[str]
    ) -> float:
        """
        Calculate a preliminary score for ML input variance.
        This gives the ML model differentiation between signals.
        NOT the final score - just provides input feature variance.
        """
        score = 50.0
        
        if indicators:
            rsi = indicators.get('rsi', 50) or 50
            atr_pct = indicators.get('atr_pct', 1.0) or 1.0
            adx = indicators.get('adx', 20) or 20
            
            # RSI position (mean reversion opportunity)
            if side == 'long':
                if rsi < 30:
                    score += 15  # Oversold = good for long
                elif rsi < 40:
                    score += 10
                elif rsi > 70:
                    score -= 10  # Overbought = bad for long
            else:  # short
                if rsi > 70:
                    score += 15  # Overbought = good for short
                elif rsi > 60:
                    score += 10
                elif rsi < 30:
                    score -= 10  # Oversold = bad for short
            
            # ATR - higher volatility = more opportunity
            if atr_pct >= 2.0:
                score += 10
            elif atr_pct >= 1.5:
                score += 5
            elif atr_pct < 0.8:
                score -= 10  # Low volatility penalty
            
            # ADX - trend strength
            if adx > 30:
                score += 5  # Strong trend
            elif adx < 15:
                score -= 5  # Weak/choppy
        
        # Momentum bonus
        if abs(pct_change_24h) > 0.03:
            score += 5
        
        # Spread penalty
        if spread_bps > 30:
            score -= 5
        elif spread_bps < 10:
            score += 3
        
        return max(20, min(80, score))  # Clamp to reasonable range

