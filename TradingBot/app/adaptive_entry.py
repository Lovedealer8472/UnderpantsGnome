"""
Adaptive Entry Controller

Dynamically adjusts the minimum entry score threshold to keep the bot active 24/7
while always taking the BEST available signals.

Philosophy:
- When idle (no positions): Be more aggressive to get into the market
- When holding positions: Be more selective (only replace with better signals)
- Always track what signals are available and set threshold relative to market quality
- Never trade garbage (hard floor protection)

The threshold adapts based on:
1. Recent signal quality (rolling percentile of last N signals)
2. Current position count (more positions = more selective)
3. Time since last entry (decay to encourage activity)
4. Market conditions (volatility, trend)
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime
import numpy as np


@dataclass
class SignalObservation:
    """Record of a signal that was evaluated."""
    timestamp: float
    symbol: str
    score: float
    side: str
    was_taken: bool  # Did we actually enter?
    rejection_reason: Optional[str] = None


@dataclass
class AdaptiveState:
    """Current state of the adaptive entry controller."""
    # Current dynamic threshold
    current_threshold: float
    
    # Components that influence threshold
    base_threshold: float      # From percentile of recent signals
    position_adjustment: float # Adjustment based on position count
    decay_adjustment: float    # Adjustment based on time since last entry
    
    # Market quality metrics
    avg_signal_score: float    # Average score of recent signals
    best_signal_score: float   # Best signal in window
    signal_count: int          # Number of signals in window
    
    # Position info
    open_positions: int
    target_positions: int
    
    # Activity metrics
    minutes_since_last_entry: float
    entries_last_hour: int
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AdaptiveEntryController:
    """
    Intelligent entry threshold controller that keeps the bot active
    while taking the best available signals.
    """
    
    def __init__(
        self,
        # Signal tracking
        window_size: int = 500,           # Track last N signals
        window_time_sec: float = 3600,    # Also limit by time (1 hour)
        min_signals_required: int = 50,   # Minimum signals needed before trading (blocks until sufficient data)
        
        # Threshold bounds
        hard_floor: float = 25.0,         # NEVER go below this
        ceiling: float = 70.0,            # Maximum threshold
        
        # Percentile targets (lower = more selective)
        idle_percentile: float = 40.0,    # Take top 40% when idle
        normal_percentile: float = 25.0,  # Take top 25% normally
        full_percentile: float = 15.0,    # Take top 15% when near full
        
        # Position targets
        target_positions: int = 3,        # Target number of positions
        max_positions: int = 5,           # Maximum positions
        
        # Activity decay (be more aggressive over time)
        decay_start_min: float = 5.0,     # Start decay after 5 min idle
        decay_rate_per_min: float = 0.5,  # Lower threshold 0.5 per minute
        max_decay: float = 10.0,          # Maximum decay adjustment
        
        # Logging
        logger = None,
    ):
        self.window_size = window_size
        self.window_time_sec = window_time_sec
        self.min_signals_required = min_signals_required
        self.hard_floor = hard_floor
        self.ceiling = ceiling
        
        self.idle_percentile = idle_percentile
        self.normal_percentile = normal_percentile
        self.full_percentile = full_percentile
        
        self.target_positions = target_positions
        self.max_positions = max_positions
        
        self.decay_start_min = decay_start_min
        self.decay_rate_per_min = decay_rate_per_min
        self.max_decay = max_decay
        
        self.logger = logger
        
        # Signal history
        self._signals: deque = deque(maxlen=window_size)
        self._lock = threading.RLock()
        
        # Tracking
        self._last_entry_time: float = time.time()
        self._entries_last_hour: int = 0
        self._entry_times: deque = deque(maxlen=100)
        
        # Current state
        self._current_threshold: float = 45.0  # Start at 45 to prevent trash at startup
        self._last_calculation_time: float = 0
        self._calculation_interval: float = 5.0  # Recalculate every 5s
        
    def record_signal(
        self,
        symbol: str,
        score: float,
        side: str,
        was_taken: bool = False,
        rejection_reason: Optional[str] = None
    ) -> None:
        """
        Record a signal observation.
        Call this for EVERY signal evaluated, whether taken or not.
        """
        obs = SignalObservation(
            timestamp=time.time(),
            symbol=symbol,
            score=score,
            side=side,
            was_taken=was_taken,
            rejection_reason=rejection_reason
        )
        
        with self._lock:
            self._signals.append(obs)
            
            if was_taken:
                self._last_entry_time = time.time()
                self._entry_times.append(time.time())
    
    def record_entry(self, symbol: str, score: float) -> None:
        """Record that we entered a position."""
        with self._lock:
            self._last_entry_time = time.time()
            self._entry_times.append(time.time())
    
    def has_sufficient_data(self) -> bool:
        """
        Check if we have collected enough signal data to make informed decisions.
        
        Returns:
            True if we have enough data to start trading, False otherwise
        """
        with self._lock:
            # Clean old signals
            now = time.time()
            self._clean_old_signals(now)
            return len(self._signals) >= self.min_signals_required
    
    def get_threshold(self, open_positions: int = 0) -> float:
        """
        Get the current dynamic entry threshold.
        
        Args:
            open_positions: Number of currently open positions
            
        Returns:
            The minimum score required for entry (returns 100.0 to block trading if insufficient data)
        """
        now = time.time()
        
        # Rate limit recalculation
        if now - self._last_calculation_time < self._calculation_interval:
            return self._current_threshold
        
        self._last_calculation_time = now
        
        with self._lock:
            # Clean old signals
            self._clean_old_signals(now)
            
            # Get recent scores
            scores = [s.score for s in self._signals]
            
            # BLOCK TRADING until sufficient data is collected
            if len(scores) < self.min_signals_required:
                # Return very high threshold to block all trades
                self._current_threshold = 100.0
                if self.logger and not hasattr(self, '_last_warmup_log') or (now - getattr(self, '_last_warmup_log', 0)) > 30.0:
                    self.logger.info(
                        f"[ADAPTIVE] Warming up: {len(scores)}/{self.min_signals_required} signals collected. "
                        f"Trading blocked until sufficient data available."
                    )
                    self._last_warmup_log = now
                return self._current_threshold
            
            # 1. BASE THRESHOLD: Percentile of recent signals
            # Choose percentile based on position count
            if open_positions == 0:
                target_percentile = self.idle_percentile  # More aggressive
            elif open_positions >= self.target_positions:
                target_percentile = self.full_percentile  # Very selective
            else:
                # Interpolate between idle and full
                ratio = open_positions / self.target_positions
                target_percentile = self.idle_percentile - (self.idle_percentile - self.full_percentile) * ratio
            
            # Calculate threshold as percentile of recent signals
            # e.g., if target_percentile=25, we want top 25%, so threshold = 75th percentile
            base_threshold = float(np.percentile(scores, 100 - target_percentile))
            
            # 2. POSITION ADJUSTMENT: Raise threshold if we have many positions
            position_adjustment = 0.0
            if open_positions >= self.target_positions:
                # For each position over target, raise threshold
                excess = open_positions - self.target_positions
                position_adjustment = excess * 2.0  # +2 per excess position
            
            # 3. DECAY ADJUSTMENT: Lower threshold if idle too long
            decay_adjustment = 0.0
            minutes_since_entry = (now - self._last_entry_time) / 60.0
            
            if minutes_since_entry > self.decay_start_min:
                decay_minutes = minutes_since_entry - self.decay_start_min
                decay_adjustment = min(decay_minutes * self.decay_rate_per_min, self.max_decay)
            
            # 4. COMBINE
            threshold = base_threshold + position_adjustment - decay_adjustment
            
            # 5. CLAMP to bounds
            threshold = max(self.hard_floor, min(threshold, self.ceiling))
            
            self._current_threshold = threshold
            
            # Log if significant change
            if self.logger and hasattr(self, '_last_logged_threshold'):
                if abs(threshold - self._last_logged_threshold) > 2.0:
                    self.logger.info(
                        f"[ADAPTIVE] Threshold adjusted: {threshold:.1f} "
                        f"(base={base_threshold:.1f}, pos_adj={position_adjustment:+.1f}, "
                        f"decay={-decay_adjustment:+.1f}, positions={open_positions})"
                    )
            self._last_logged_threshold = threshold
            
            return threshold
    
    def get_state(self, open_positions: int = 0) -> AdaptiveState:
        """Get full state for UI/debugging."""
        now = time.time()
        
        with self._lock:
            self._clean_old_signals(now)
            scores = [s.score for s in self._signals]
            
            # Calculate components
            if len(scores) >= 10:
                if open_positions == 0:
                    target_percentile = self.idle_percentile
                elif open_positions >= self.target_positions:
                    target_percentile = self.full_percentile
                else:
                    ratio = open_positions / self.target_positions
                    target_percentile = self.idle_percentile - (self.idle_percentile - self.full_percentile) * ratio
                
                base_threshold = float(np.percentile(scores, 100 - target_percentile))
                avg_score = float(np.mean(scores))
                best_score = float(max(scores))
            else:
                # Insufficient data - return blocking state
                base_threshold = 100.0
                avg_score = 0.0
                best_score = 0.0
            
            # Position adjustment
            position_adjustment = 0.0
            if open_positions >= self.target_positions:
                excess = open_positions - self.target_positions
                position_adjustment = excess * 2.0
            
            # Decay adjustment
            minutes_since_entry = (now - self._last_entry_time) / 60.0
            decay_adjustment = 0.0
            if minutes_since_entry > self.decay_start_min:
                decay_minutes = minutes_since_entry - self.decay_start_min
                decay_adjustment = min(decay_minutes * self.decay_rate_per_min, self.max_decay)
            
            # Count entries in last hour
            hour_ago = now - 3600
            entries_last_hour = sum(1 for t in self._entry_times if t > hour_ago)
            
            return AdaptiveState(
                current_threshold=self._current_threshold,
                base_threshold=base_threshold,
                position_adjustment=position_adjustment,
                decay_adjustment=decay_adjustment,
                avg_signal_score=avg_score,
                best_signal_score=best_score,
                signal_count=len(scores),
                open_positions=open_positions,
                target_positions=self.target_positions,
                minutes_since_last_entry=minutes_since_entry,
                entries_last_hour=entries_last_hour,
            )
    
    def _clean_old_signals(self, now: float) -> None:
        """Remove signals older than window_time_sec."""
        cutoff = now - self.window_time_sec
        while self._signals and self._signals[0].timestamp < cutoff:
            self._signals.popleft()
    
    def should_enter(self, score: float, open_positions: int = 0) -> Tuple[bool, str]:
        """
        Check if a signal should be entered based on adaptive threshold.
        
        Returns:
            Tuple of (should_enter, reason)
        """
        threshold = self.get_threshold(open_positions)
        
        if score >= threshold:
            return True, f"score {score:.1f} >= adaptive threshold {threshold:.1f}"
        else:
            return False, f"score {score:.1f} < adaptive threshold {threshold:.1f}"


# Global instance
_adaptive_controller: Optional[AdaptiveEntryController] = None
_init_lock = threading.Lock()


def get_adaptive_controller(logger=None) -> AdaptiveEntryController:
    """Get or create the global adaptive entry controller."""
    global _adaptive_controller
    
    if _adaptive_controller is None:
        with _init_lock:
            if _adaptive_controller is None:
                from . import config
                _adaptive_controller = AdaptiveEntryController(
                    min_signals_required=int(getattr(config, 'ADAPTIVE_MIN_SIGNALS_REQUIRED', 50)),
                    hard_floor=float(getattr(config, 'ADAPTIVE_HARD_FLOOR', 25.0)),
                    ceiling=float(getattr(config, 'ADAPTIVE_CEILING', 70.0)),
                    idle_percentile=float(getattr(config, 'ADAPTIVE_IDLE_PERCENTILE', 40.0)),
                    normal_percentile=float(getattr(config, 'ADAPTIVE_NORMAL_PERCENTILE', 25.0)),
                    full_percentile=float(getattr(config, 'ADAPTIVE_FULL_PERCENTILE', 15.0)),
                    target_positions=int(getattr(config, 'ADAPTIVE_TARGET_POSITIONS', 3)),
                    decay_start_min=float(getattr(config, 'ADAPTIVE_DECAY_START_MIN', 5.0)),
                    decay_rate_per_min=float(getattr(config, 'ADAPTIVE_DECAY_RATE', 0.5)),
                    max_decay=float(getattr(config, 'ADAPTIVE_MAX_DECAY', 10.0)),
                    logger=logger,
                )
    
    return _adaptive_controller


def reset_adaptive_controller() -> None:
    """Reset the global controller (for testing)."""
    global _adaptive_controller
    with _init_lock:
        _adaptive_controller = None
