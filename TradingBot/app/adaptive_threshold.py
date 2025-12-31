"""
Adaptive Threshold Manager
===========================

Automatically adjusts MIN_SIGNAL_SCORE based on real-time market conditions.

Philosophy:
- Maintains absolute quality floor (45 = 60% ML probability)
- Adapts to market conditions (slow/normal/hot)
- Smooth adjustments to prevent whiplash
- No percentile filtering (avoids "best garbage" problem)

Market States:
- Slow market (avg < 50, density < 50/h): threshold = 45
- Normal market (avg 50-60, density 50-150/h): threshold = 50
- Hot market (avg > 60, density > 150/h): threshold = 60
"""

import time
from typing import Dict, Optional, List
from .logger import get_logger


class AdaptiveThresholdManager:
    """
    Automatically adjusts MIN_SIGNAL_SCORE based on market conditions.
    
    Uses absolute thresholds (not percentiles) to maintain quality standards
    while adapting to market activity levels.
    """
    
    def __init__(
        self,
        min_threshold: int = 45,
        max_threshold: int = 75,
        adjustment_interval: int = 300,
        window_size: int = 100
    ):
        """
        Initialize adaptive threshold manager.
        
        Args:
            min_threshold: Minimum threshold (never go below - quality floor)
            max_threshold: Maximum threshold (never go above - prevent over-restriction)
            adjustment_interval: Seconds between adjustments (prevent thrashing)
            window_size: Number of recent signals to analyze
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adjustment_interval = adjustment_interval
        self.window_size = window_size
        
        # Current threshold (starts at minimum)
        self.current_threshold = float(min_threshold)
        
        # Last adjustment time
        self.last_adjustment = time.time()
        
        # Logger
        self.logger = get_logger("AdaptiveThreshold")
        
        self.logger.info(
            f"[ADAPTIVE_THRESHOLD] Initialized: floor={min_threshold}, "
            f"cap={max_threshold}, interval={adjustment_interval}s"
        )
    
    def calculate_market_state(self, signal_history: List[Dict]) -> Optional[Dict]:
        """
        Analyze recent signals to determine market state.
        
        Args:
            signal_history: List of signal records with 'final_score' and 'timestamp'
        
        Returns:
            Dict with market metrics or None if insufficient data
        """
        if len(signal_history) < 20:
            return None  # Not enough data
        
        # Get recent signals within window
        recent = signal_history[-self.window_size:]
        
        # Calculate average score (quality indicator)
        scores = [s.get('final_score', 0) for s in recent if s.get('final_score', 0) > 0]
        if not scores:
            return None
        
        avg_score = sum(scores) / len(scores)
        
        # Calculate signal density (opportunity indicator)
        if len(recent) < 2:
            return None
        
        time_span = recent[-1].get('timestamp', 0) - recent[0].get('timestamp', 0)
        if time_span <= 0:
            signals_per_hour = 0
        else:
            signals_per_hour = (len(recent) / time_span) * 3600
        
        # Calculate score volatility (market stability indicator)
        if len(scores) > 10:
            score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            score_std = score_variance ** 0.5
        else:
            score_std = 0
        
        return {
            'avg_score': avg_score,
            'signals_per_hour': signals_per_hour,
            'score_volatility': score_std,
            'sample_size': len(scores)
        }
    
    def determine_target_threshold(self, market_state: Dict) -> float:
        """
        Determine optimal threshold based on market conditions.
        
        Strategy:
        - Hot market (high avg, many signals): Raise threshold
        - Normal market: Moderate threshold
        - Slow market (low avg, few signals): Lower threshold (but maintain floor)
        
        Args:
            market_state: Dict with avg_score, signals_per_hour, etc.
        
        Returns:
            Target threshold (will be smoothed toward this value)
        """
        avg_score = market_state['avg_score']
        density = market_state['signals_per_hour']
        
        # Determine target threshold based on market conditions
        if avg_score > 60 and density > 150:
            # HOT MARKET: High quality + abundant opportunities
            # Raise bar significantly - only take cream of the crop
            target = avg_score - 5  # Take signals within 5 points of average
            target = min(target, 70)  # Cap at 70
            
        elif avg_score > 50 and density > 50:
            # NORMAL MARKET: Good quality + decent opportunities
            # Moderate bar - take good signals
            target = avg_score - 10  # Take signals within 10 points of average
            target = min(target, 55)  # Cap at 55
            
        elif avg_score > 40 and density > 20:
            # SLOW MARKET: Moderate quality + limited opportunities
            # Lower bar but maintain quality floor
            target = avg_score - 8  # Take signals within 8 points of average
            target = max(target, self.min_threshold)  # Floor at minimum
            
        else:
            # VERY SLOW MARKET: Low quality or very few signals
            # Use minimum viable threshold
            target = self.min_threshold
        
        # Apply safety bounds
        target = max(self.min_threshold, min(self.max_threshold, target))
        
        return target
    
    def get_adaptive_threshold(self, signal_history: List[Dict]) -> float:
        """
        Calculate optimal threshold based on current market conditions.
        
        This is the main method called by SignalGenerator.
        
        Args:
            signal_history: List of recent signal records
        
        Returns:
            Current adaptive threshold (45-75 range)
        """
        now = time.time()
        
        # Only adjust every N seconds (prevent thrashing)
        if now - self.last_adjustment < self.adjustment_interval:
            return self.current_threshold
        
        # Calculate market state
        state = self.calculate_market_state(signal_history)
        if not state:
            # Not enough data - use current threshold
            return self.current_threshold
        
        # Determine target threshold
        target = self.determine_target_threshold(state)
        
        # Smooth adjustment (prevent sudden jumps)
        # Move 2 points at a time toward target
        if abs(target - self.current_threshold) > 2:
            if target > self.current_threshold:
                self.current_threshold += 2
            else:
                self.current_threshold -= 2
        else:
            self.current_threshold = target
        
        self.last_adjustment = now
        
        # Log adjustment for transparency
        self.logger.info(
            f"[ADAPTIVE_THRESHOLD] Adjusted: {self.current_threshold:.0f} | "
            f"Market: avg={state['avg_score']:.1f} density={state['signals_per_hour']:.0f}/h | "
            f"Target: {target:.0f}"
        )
        
        return self.current_threshold

