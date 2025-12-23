"""
ADAPTIVE ENTRY FILTERS
======================
Automatically adjusts entry thresholds based on current market conditions.

Instead of fixed thresholds like "ATR >= 1%", uses percentile-based:
- "ATR in top 40% of current market"

This means:
- In calm markets: absolute thresholds drop, more trades
- In volatile markets: absolute thresholds rise, fewer (better) trades

The system continuously learns what "normal" looks like right now.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import threading


@dataclass
class MarketSnapshot:
    """A single observation of market conditions."""
    timestamp: datetime
    symbol: str
    atr_pct: float
    momentum_1h: float  # Absolute value of 1h change
    rsi: float
    volume_ratio: float


@dataclass 
class AdaptiveThresholds:
    """Current adaptive thresholds based on market conditions."""
    atr_threshold: float
    momentum_threshold: float
    rsi_long_max: float
    rsi_short_min: float
    sample_size: int
    last_updated: datetime
    
    # Market regime info
    market_volatility: str  # 'low', 'normal', 'high'
    percentile_atr: float   # What percentile is the threshold at
    percentile_mom: float


class AdaptiveFilterEngine:
    """
    Maintains a rolling window of market observations and computes
    adaptive thresholds based on percentiles.
    """
    
    def __init__(
        self,
        window_size: int = 1000,      # Number of observations to track
        min_samples: int = 100,        # Minimum samples before adaptive kicks in
        atr_percentile: float = 60,    # Top 40% volatility
        momentum_percentile: float = 70,  # Top 30% momentum
        rsi_extreme_pct: float = 20,   # Top/bottom 20% RSI
        update_interval_sec: float = 60,  # Recalculate every 60s
    ):
        self.window_size = window_size
        self.min_samples = min_samples
        self.atr_percentile = atr_percentile
        self.momentum_percentile = momentum_percentile
        self.rsi_extreme_pct = rsi_extreme_pct
        self.update_interval_sec = update_interval_sec
        
        # Rolling window of observations
        self._observations: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
        # Cached thresholds
        self._cached_thresholds: Optional[AdaptiveThresholds] = None
        self._last_calculation: datetime = datetime.min
        
        # Fallback thresholds (used when not enough data)
        self.fallback_atr = 1.0
        self.fallback_momentum = 3.0
        self.fallback_rsi_long = 25
        self.fallback_rsi_short = 75
        
        # Statistics for logging
        self.stats = {
            'observations_total': 0,
            'threshold_updates': 0,
        }
    
    def add_observation(
        self,
        symbol: str,
        atr_pct: float,
        momentum_1h: float,
        rsi: float,
        volume_ratio: float = 1.0
    ):
        """Add a new market observation from a scanned symbol."""
        
        if atr_pct is None or momentum_1h is None or rsi is None:
            return
        
        # Sanity checks
        if atr_pct <= 0 or atr_pct > 50:  # Invalid ATR
            return
        if rsi < 0 or rsi > 100:  # Invalid RSI
            return
        
        obs = MarketSnapshot(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            atr_pct=atr_pct,
            momentum_1h=abs(momentum_1h),  # Use absolute value
            rsi=rsi,
            volume_ratio=volume_ratio
        )
        
        with self._lock:
            self._observations.append(obs)
            self.stats['observations_total'] += 1
    
    def get_thresholds(self) -> AdaptiveThresholds:
        """
        Get current adaptive thresholds.
        Recalculates if enough time has passed.
        """
        
        now = datetime.utcnow()
        
        # Check if we need to recalculate
        if (self._cached_thresholds is None or 
            (now - self._last_calculation).total_seconds() > self.update_interval_sec):
            self._cached_thresholds = self._calculate_thresholds()
            self._last_calculation = now
            self.stats['threshold_updates'] += 1
        
        return self._cached_thresholds
    
    def _calculate_thresholds(self) -> AdaptiveThresholds:
        """Calculate adaptive thresholds from current observations."""
        
        with self._lock:
            observations = list(self._observations)
        
        n = len(observations)
        
        # Not enough data - use fallbacks
        if n < self.min_samples:
            return AdaptiveThresholds(
                atr_threshold=self.fallback_atr,
                momentum_threshold=self.fallback_momentum,
                rsi_long_max=self.fallback_rsi_long,
                rsi_short_min=self.fallback_rsi_short,
                sample_size=n,
                last_updated=datetime.utcnow(),
                market_volatility='unknown',
                percentile_atr=0,
                percentile_mom=0
            )
        
        # Extract arrays
        atrs = np.array([o.atr_pct for o in observations])
        moms = np.array([o.momentum_1h for o in observations])
        rsis = np.array([o.rsi for o in observations])
        
        # Calculate percentile thresholds
        atr_threshold = np.percentile(atrs, self.atr_percentile)
        mom_threshold = np.percentile(moms, self.momentum_percentile)
        
        # RSI thresholds: bottom X% for longs, top X% for shorts
        rsi_long_max = np.percentile(rsis, self.rsi_extreme_pct)
        rsi_short_min = np.percentile(rsis, 100 - self.rsi_extreme_pct)
        
        # Determine market regime
        median_atr = np.median(atrs)
        if median_atr < 0.8:
            volatility = 'low'
        elif median_atr > 2.0:
            volatility = 'high'
        else:
            volatility = 'normal'
        
        return AdaptiveThresholds(
            atr_threshold=round(atr_threshold, 3),
            momentum_threshold=round(mom_threshold, 3),
            rsi_long_max=round(rsi_long_max, 1),
            rsi_short_min=round(rsi_short_min, 1),
            sample_size=n,
            last_updated=datetime.utcnow(),
            market_volatility=volatility,
            percentile_atr=self.atr_percentile,
            percentile_mom=self.momentum_percentile
        )
    
    def check_entry(
        self,
        side: str,  # 'long' or 'short'
        atr_pct: float,
        momentum_1h: float,
        rsi: float,
    ) -> Tuple[bool, str]:
        """
        Check if a signal passes adaptive filters.
        
        Returns:
            (passed: bool, reason: str)
        """
        
        thresholds = self.get_thresholds()
        
        # Check volatility (ATR)
        if atr_pct < thresholds.atr_threshold:
            return False, f"ADAPTIVE:low_vol(ATR={atr_pct:.2f}%<{thresholds.atr_threshold:.2f}%)"
        
        # Check momentum
        abs_mom = abs(momentum_1h)
        if abs_mom < thresholds.momentum_threshold:
            return False, f"ADAPTIVE:low_mom(|1h|={abs_mom:.2f}%<{thresholds.momentum_threshold:.2f}%)"
        
        # Check RSI alignment
        if side == 'long' and rsi > thresholds.rsi_long_max:
            return False, f"ADAPTIVE:RSI_high(RSI={rsi:.0f}>{thresholds.rsi_long_max:.0f})"
        
        if side == 'short' and rsi < thresholds.rsi_short_min:
            return False, f"ADAPTIVE:RSI_low(RSI={rsi:.0f}<{thresholds.rsi_short_min:.0f})"
        
        return True, "ADAPTIVE:PASS"
    
    def get_status_string(self) -> str:
        """Get a status string for UI display."""
        
        t = self.get_thresholds()
        return (
            f"ADAPTIVE[{t.market_volatility.upper()}] "
            f"ATR>{t.atr_threshold:.2f}% "
            f"Mom>{t.momentum_threshold:.2f}% "
            f"RSI:{t.rsi_long_max:.0f}-{t.rsi_short_min:.0f} "
            f"(n={t.sample_size})"
        )


# Global singleton instance
_adaptive_engine: Optional[AdaptiveFilterEngine] = None


def get_adaptive_engine() -> AdaptiveFilterEngine:
    """Get or create the global adaptive filter engine."""
    global _adaptive_engine
    
    if _adaptive_engine is None:
        # Import config to get settings
        try:
            from . import config
            _adaptive_engine = AdaptiveFilterEngine(
                atr_percentile=getattr(config, 'ENTRY_ATR_PERCENTILE', 60),
                momentum_percentile=getattr(config, 'ENTRY_MOMENTUM_PERCENTILE', 70),
                rsi_extreme_pct=getattr(config, 'ENTRY_RSI_EXTREME_PCT', 20),
            )
        except:
            _adaptive_engine = AdaptiveFilterEngine()
    
    return _adaptive_engine


def feed_observation(symbol: str, atr_pct: float, momentum_1h: float, rsi: float, volume_ratio: float = 1.0):
    """Feed a market observation to the adaptive engine."""
    engine = get_adaptive_engine()
    engine.add_observation(symbol, atr_pct, momentum_1h, rsi, volume_ratio)


def check_adaptive_entry(side: str, atr_pct: float, momentum_1h: float, rsi: float) -> Tuple[bool, str]:
    """Check if entry passes adaptive filters."""
    engine = get_adaptive_engine()
    return engine.check_entry(side, atr_pct, momentum_1h, rsi)


def get_adaptive_status() -> str:
    """Get current adaptive filter status for UI."""
    engine = get_adaptive_engine()
    return engine.get_status_string()
