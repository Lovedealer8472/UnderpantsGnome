"""
Adaptive Performance Monitor

Monitors open positions and adjusts bot behavior based on real-time performance.
Intelligent system that adapts to what's working and what's not.
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import deque
from .logger import get_logger

logger = get_logger("AdaptivePerformance")


class AdaptivePerformanceMonitor:
    """
    Monitors open positions and adapts bot behavior based on real-time performance.
    
    Features:
    - Track position performance (PnL, age, trend)
    - Adjust entry thresholds based on portfolio health
    - Detect winning/losing patterns
    - Suggest position management improvements
    """
    
    def __init__(self, lookback_window_seconds: float = 3600.0):
        """
        Initialize adaptive performance monitor.
        
        Args:
            lookback_window_seconds: How far back to analyze (default: 1 hour)
        """
        self.lookback_window = lookback_window_seconds
        self.position_history: deque = deque(maxlen=1000)  # Track last 1000 position updates
        self.entry_threshold_adjustment: float = 0.0  # Adjustment to MIN_SIGNAL_SCORE
        self.last_analysis_time: float = 0.0
        self.analysis_interval: float = 60.0  # Analyze every 60 seconds
        
    def record_position_update(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        entry_time: float,
        side: str,
        position_size: float,
        stop_loss: float,
        take_profit: Optional[float] = None
    ):
        """Record a position update for analysis."""
        pnl_pct = ((current_price - entry_price) / entry_price * 100) * (1 if side == 'long' else -1)
        age_seconds = time.time() - entry_time
        
        position_data = {
            'timestamp': time.time(),
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': current_price,
            'entry_time': entry_time,
            'age_seconds': age_seconds,
            'side': side,
            'position_size': position_size,
            'pnl_pct': pnl_pct,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'distance_to_sl_pct': abs((current_price - stop_loss) / entry_price * 100),
            'distance_to_tp_pct': abs((take_profit - current_price) / entry_price * 100) if take_profit else None
        }
        
        self.position_history.append(position_data)
    
    def analyze_performance(self, current_positions: Dict[str, Dict]) -> Dict:
        """
        Analyze current positions and return adaptive recommendations.
        
        Args:
            current_positions: Dict of current open positions
            
        Returns:
            Dict with analysis results and recommendations:
            - portfolio_pnl_pct: Average PnL across all positions
            - winning_positions: Count of positions in profit
            - losing_positions: Count of positions in loss
            - average_age_seconds: Average position age
            - recommended_threshold_adjustment: Suggested MIN_SIGNAL_SCORE adjustment
            - health_score: Overall portfolio health (0-100)
        """
        now = time.time()
        
        # Only analyze periodically
        if now - self.last_analysis_time < self.analysis_interval:
            return self._get_cached_analysis()
        
        self.last_analysis_time = now
        
        if not current_positions:
            return {
                'portfolio_pnl_pct': 0.0,
                'winning_positions': 0,
                'losing_positions': 0,
                'average_age_seconds': 0.0,
                'recommended_threshold_adjustment': 0.0,
                'health_score': 50.0,  # Neutral when no positions
                'recommendations': []
            }
        
        # Analyze current positions
        total_pnl_pct = 0.0
        winning_count = 0
        losing_count = 0
        total_age = 0.0
        positions_at_risk = 0  # Positions close to stop loss
        
        for symbol, position in current_positions.items():
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', entry_price)
            entry_time = position.get('entry_time', now)
            side = position.get('side', 'long')
            stop_loss = position.get('stop_loss', entry_price)
            
            if entry_price <= 0:
                continue
            
            # Calculate PnL
            pnl_pct = ((current_price - entry_price) / entry_price * 100) * (1 if side == 'long' else -1)
            total_pnl_pct += pnl_pct
            
            # Count winners/losers
            if pnl_pct > 0.1:  # >0.1% profit
                winning_count += 1
            elif pnl_pct < -0.1:  # <-0.1% loss
                losing_count += 1
            
            # Calculate age
            age_seconds = now - entry_time
            total_age += age_seconds
            
            # Check if position is at risk (within 0.5% of stop loss)
            distance_to_sl = abs((current_price - stop_loss) / entry_price * 100)
            if distance_to_sl < 0.5:
                positions_at_risk += 1
        
        position_count = len(current_positions)
        avg_pnl_pct = total_pnl_pct / position_count if position_count > 0 else 0.0
        avg_age_seconds = total_age / position_count if position_count > 0 else 0.0
        
        # Calculate health score (0-100)
        # Factors:
        # - Average PnL (weight: 40%)
        # - Win rate (weight: 30%)
        # - Position age (weight: 20%) - prefer younger positions
        # - Risk level (weight: 10%) - penalize positions close to SL
        
        win_rate = winning_count / position_count if position_count > 0 else 0.0
        risk_factor = 1.0 - (positions_at_risk / position_count if position_count > 0 else 0.0)
        
        # Normalize PnL to 0-100 scale (assuming -5% to +5% range)
        pnl_score = max(0, min(100, 50 + (avg_pnl_pct * 10)))
        
        # Age score (prefer positions < 30 minutes old)
        age_score = max(0, min(100, 100 - (avg_age_seconds / 1800 * 100)))  # Decay over 30 minutes
        
        health_score = (
            pnl_score * 0.4 +
            win_rate * 100 * 0.3 +
            age_score * 0.2 +
            risk_factor * 100 * 0.1
        )
        
        # Generate recommendations
        recommendations = []
        threshold_adjustment = 0.0
        
        # If portfolio is performing well, can be slightly more aggressive
        if health_score > 70:
            recommendations.append("Portfolio performing well - maintain current strategy")
            threshold_adjustment = -0.5  # Slightly lower threshold
        elif health_score > 50:
            recommendations.append("Portfolio performing OK - maintain conservative approach")
            threshold_adjustment = 0.0
        else:
            recommendations.append("Portfolio underperforming - tighten entry criteria")
            threshold_adjustment = +1.0  # Raise threshold
        
        # If many positions at risk, recommend tightening
        if positions_at_risk > position_count * 0.3:  # >30% at risk
            recommendations.append(f"{positions_at_risk} positions near stop loss - consider tightening stops")
        
        # If average age is high, positions may be stale
        if avg_age_seconds > 1800:  # >30 minutes
            recommendations.append(f"Average position age {avg_age_seconds/60:.1f}min - consider reviewing stale positions")
        
        analysis = {
            'portfolio_pnl_pct': avg_pnl_pct,
            'winning_positions': winning_count,
            'losing_positions': losing_count,
            'average_age_seconds': avg_age_seconds,
            'recommended_threshold_adjustment': threshold_adjustment,
            'health_score': health_score,
            'positions_at_risk': positions_at_risk,
            'recommendations': recommendations,
            'timestamp': now
        }
        
        # Cache for quick access
        self._cached_analysis = analysis
        self.entry_threshold_adjustment = threshold_adjustment
        
        # Log periodic summary
        logger.info(
            f"[ADAPTIVE] Portfolio Health: {health_score:.1f}/100 | "
            f"PnL: {avg_pnl_pct:.2f}% | "
            f"Win Rate: {win_rate*100:.1f}% ({winning_count}W/{losing_count}L) | "
            f"Threshold Adj: {threshold_adjustment:+.1f}"
        )
        
        return analysis
    
    def _get_cached_analysis(self) -> Dict:
        """Return cached analysis if available."""
        if hasattr(self, '_cached_analysis'):
            return self._cached_analysis
        return {
            'portfolio_pnl_pct': 0.0,
            'winning_positions': 0,
            'losing_positions': 0,
            'average_age_seconds': 0.0,
            'recommended_threshold_adjustment': 0.0,
            'health_score': 50.0,
            'recommendations': []
        }
    
    def get_threshold_adjustment(self) -> float:
        """Get recommended MIN_SIGNAL_SCORE adjustment."""
        return self.entry_threshold_adjustment

