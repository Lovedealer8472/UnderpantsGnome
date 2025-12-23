"""
Signal Confirmation Window (SCW) - Delays entry until signal is validated by price action.
"""

import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .config import (
    USE_SIGNAL_CONFIRMATION, SCW_UNICORN_CONFIRM, SCW_STANDARD_CONFIRM, SCW_SCALP_CONFIRM,
    SCW_MAX_SPREAD_PCT, SCW_MAX_BODY_ATR_MULT, SCW_VOLUME_STABLE_MULT,
    R_EXIT_SCALP_SCORE_MIN, R_EXIT_SCALP_SCORE_MAX,
    R_EXIT_STANDARD_SCORE_MIN, R_EXIT_STANDARD_SCORE_MAX,
    R_EXIT_RUNNER_SCORE_MIN
)


@dataclass
class WaitingSignal:
    """Signal waiting for confirmation."""
    symbol: str
    signal: Any  # Signal object
    confirmation_required: int
    confirmation_count: int = 0
    is_waiting: bool = True
    ready: bool = False
    created_time: float = 0.0
    key_level: float = 0.0  # Price level that invalidates signal
    last_bar_time: float = 0.0


class SignalConfirmationManager:
    """Manages signal confirmation windows."""
    
    def __init__(self):
        self.waiting_signals: Dict[str, WaitingSignal] = {}
    
    def get_confirmation_required(self, final_score: float) -> int:
        """
        Get required confirmation bars based on signal score.
        
        CRITICAL FIX: Use config values instead of hardcoded thresholds.
        
        Args:
            final_score: Signal final score (0-100)
        
        Returns:
            Number of bars required for confirmation
        """
        if not USE_SIGNAL_CONFIRMATION:
            return 0

        # CRITICAL FIX: Use config thresholds (not hardcoded)
        # Runner (90+): SCW_UNICORN_CONFIRM bars (config: 0)
        # Standard (70-89): SCW_STANDARD_CONFIRM bars (config: 2)
        # Scalp (60-69): SCW_SCALP_CONFIRM bars (config: 3)
        # Below 60: Use standard confirmation as fallback
        if final_score >= R_EXIT_RUNNER_SCORE_MIN:  # 90+
            return SCW_UNICORN_CONFIRM  # Config: 0 bars
        elif final_score >= R_EXIT_STANDARD_SCORE_MIN:  # 70-89
            return SCW_STANDARD_CONFIRM  # Config: 2 bars
        elif final_score >= R_EXIT_SCALP_SCORE_MIN:  # 60-69
            return SCW_SCALP_CONFIRM  # Config: 3 bars
        else:
            # Scores below 60: Use standard confirmation
            return SCW_STANDARD_CONFIRM  # Config: 2 bars
    
    def add_waiting_signal(
        self,
        symbol: str,
        signal: Any,
        key_level: float = None
    ) -> WaitingSignal:
        """
        Add a signal to the waiting queue.
        
        Args:
            symbol: Trading symbol
            signal: Signal object
            key_level: Price level that invalidates signal (optional)
        
        Returns:
            WaitingSignal object
        """
        confirmation_required = self.get_confirmation_required(signal.final_score)
        
        # Calculate key level if not provided
        if key_level is None:
            if signal.side.lower() == 'long':
                # For long, key level is stop_loss (invalidation if price goes below)
                key_level = signal.stop_loss
            else:  # short
                # For short, key level is stop_loss (invalidation if price goes above)
                key_level = signal.stop_loss
        
        waiting = WaitingSignal(
            symbol=symbol,
            signal=signal,
            confirmation_required=confirmation_required,
            confirmation_count=0,
            is_waiting=True,
            ready=(confirmation_required == 0),
            created_time=time.time(),
            key_level=key_level,
            last_bar_time=time.time()
        )
        
        self.waiting_signals[symbol] = waiting
        return waiting
    
    def check_confirmation_conditions(
        self,
        waiting: WaitingSignal,
        current_price: float,
        low: float = None,
        high: float = None,
        spread_bps: float = 0.0,
        atr_pct: float = None,
        volume: float = None,
        volume_ma: float = None,
        opposite_signal_score: float = None
    ) -> Tuple[bool, str]:
        """
        Check if confirmation conditions are met for current bar.
        
        Returns:
            (is_valid, reason)
        """
        signal = waiting.signal
        side = signal.side.lower()
        
        # 1. Price must stay on valid side
        if side == 'long':
            if low is not None and low < waiting.key_level:
                return False, "price_invalidation"
            # Also check current price
            if current_price < waiting.key_level:
                return False, "price_invalidation"
        else:  # short
            if high is not None and high > waiting.key_level:
                return False, "price_invalidation"
            # Also check current price
            if current_price > waiting.key_level:
                return False, "price_invalidation"
        
        # 2. No opposite signal with equal or higher score (cancels)
        if opposite_signal_score is not None and waiting.signal and opposite_signal_score >= getattr(waiting.signal, 'final_score', 0.0):
            return False, "opposite_signal"
        
        # 3. Spread does not exceed threshold
        spread_pct = spread_bps / 10000.0  # Convert bps to percentage
        if spread_pct > SCW_MAX_SPREAD_PCT:
            return False, "spread_too_wide"
        
        # 4. No volatility spike (if ATR available)
        if atr_pct is not None and atr_pct > 0:
            # Estimate candle body (simplified: use price change)
            if low is not None and high is not None:
                candle_body_pct = abs(high - low) / current_price if current_price > 0 else 0
                max_body_pct = atr_pct * SCW_MAX_BODY_ATR_MULT
                if candle_body_pct > max_body_pct:
                    return False, "volatility_spike"
        
        # 5. Volume stable (optional, if volume data available)
        if volume is not None and volume_ma is not None and volume_ma > 0:
            if volume > volume_ma * SCW_VOLUME_STABLE_MULT:
                return False, "volume_spike"
        
        return True, "valid"
    
    def update_waiting_signal(
        self,
        symbol: str,
        current_price: float,
        low: float = None,
        high: float = None,
        spread_bps: float = 0.0,
        atr_pct: float = None,
        volume: float = None,
        volume_ma: float = None,
        opposite_signal_score: float = None,
        bar_closed: bool = False
    ) -> Tuple[bool, Optional[WaitingSignal]]:
        """
        Update waiting signal with new bar data.
        
        Returns:
            (is_ready, waiting_signal) - is_ready=True means signal is confirmed and ready to enter
        """
        if symbol not in self.waiting_signals:
            return False, None
        
        waiting = self.waiting_signals[symbol]
        
        if waiting.ready:
            return True, waiting
        
        # Check if bar closed (increment counter)
        if bar_closed:
            # Check confirmation conditions
            is_valid, reason = self.check_confirmation_conditions(
                waiting, current_price, low, high, spread_bps, atr_pct,
                volume, volume_ma, opposite_signal_score
            )
            
            if not is_valid:
                # Signal invalidated - remove it
                del self.waiting_signals[symbol]
                return False, None
            
            # Conditions met - increment counter
            waiting.confirmation_count += 1
            waiting.last_bar_time = time.time()
            
            # Check if confirmation complete
            if waiting.confirmation_count >= waiting.confirmation_required:
                waiting.ready = True
                waiting.is_waiting = False
                return True, waiting
        
        return False, waiting
    
    def get_ready_signal(self, symbol: str) -> Optional[WaitingSignal]:
        """Get ready signal if available."""
        if symbol in self.waiting_signals:
            waiting = self.waiting_signals[symbol]
            if waiting.ready:
                return waiting
        return None
    
    def remove_signal(self, symbol: str):
        """Remove signal from waiting queue."""
        if symbol in self.waiting_signals:
            del self.waiting_signals[symbol]
    
    def cleanup_stale_signals(self, max_age_sec: float = 300.0):
        """Remove signals that have been waiting too long."""
        now = time.time()
        stale = []
        for symbol, waiting in self.waiting_signals.items():
            if now - waiting.created_time > max_age_sec:
                stale.append(symbol)
        
        for symbol in stale:
            del self.waiting_signals[symbol]

