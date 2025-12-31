"""
ADAPTIVE ENTRY FILTERS
======================
Automatically adjusts entry thresholds based on current market conditions.

Uses hybrid approach:
1. ML-based ATR threshold prediction (optimal for market conditions)
2. Percentile-based fallback (when ML unavailable or insufficient data)

Instead of fixed thresholds like "ATR >= 1%", uses:
- ML prediction: "ATR >= optimal_threshold_for_current_market"
- Percentile fallback: "ATR in top 40% of current market"

This means:
- In calm markets: ATR threshold drops to 0.2-0.3%, more trades
- In volatile markets: ATR threshold rises to 1.5-2.0%, fewer (better) trades

The system continuously learns optimal thresholds for different market regimes.
"""

import numpy as np
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import threading
from pathlib import Path
import joblib


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
    adaptive thresholds using ML models + percentiles.
    """

    def __init__(
        self,
        window_size: int = 1000,      # Number of observations to track
        min_samples: int = 10,         # Minimum samples before ML kicks in (bootstrap activates immediately)
        atr_percentile: float = 60,    # Top 40% volatility (fallback)
        momentum_percentile: float = 70,  # Top 30% momentum
        rsi_extreme_pct: float = 20,   # Top/bottom 20% RSI
        update_interval_sec: float = 60,  # Recalculate every 60s
        adaptive_models_dir: Optional[Path] = None,  # Directory with ML models
    ):
        self.window_size = window_size
        self.min_samples = min_samples
        self.atr_percentile = atr_percentile
        self.momentum_percentile = momentum_percentile
        self.rsi_extreme_pct = rsi_extreme_pct
        self.update_interval_sec = update_interval_sec
        self.adaptive_models_dir = adaptive_models_dir or Path("adaptive_models")

        # Rolling window of observations
        self._observations: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

        # Cached thresholds
        self._cached_thresholds: Optional[AdaptiveThresholds] = None
        self._last_calculation: datetime = datetime.min

        # ML Models (loaded on demand)
        self._atr_model = None
        self._atr_features = None

        # Fallback thresholds (used when not enough data or ML fails)
        self.fallback_atr = 1.0
        self.fallback_momentum = 3.0
        self.fallback_rsi_long = 25
        self.fallback_rsi_short = 75

        # Statistics for logging
        self.stats = {
            'observations_total': 0,
            'threshold_updates': 0,
            'ml_predictions': 0,
            'percentile_fallbacks': 0,
        }

        # Bootstrap with minimal representative observations for immediate ATR adaptation
        self._bootstrap_minimal_observations()

    def _bootstrap_minimal_observations(self):
        """Add minimal bootstrap observations to enable ATR adaptation immediately."""
        # Current low-volatility market conditions (ATR ~0.5%, momentum ~0.1-0.3%)
        bootstrap_data = [
            ("BTC/USDT", 0.8, 0.15, 65, 1.0),   # High volume, low ATR, low momentum
            ("ETH/USDT", 0.6, 0.12, 55, 0.8),   # Medium volume, low ATR, low momentum
            ("BNB/USDT", 0.5, 0.18, 60, 0.6),   # Lower volume, low ATR, low momentum
            ("ADA/USDT", 0.4, 0.08, 70, 0.4),   # Low volume, very low ATR, very low momentum
            ("SOL/USDT", 0.7, 0.22, 45, 0.7),   # Medium volume, low ATR, low momentum
            ("DOGE/USDT", 0.3, 0.05, 75, 0.3),  # Low ATR, very low momentum, high RSI
            ("XRP/USDT", 0.5, 0.14, 50, 0.5),   # Balanced, low momentum
            ("LTC/USDT", 0.6, 0.16, 55, 0.4),   # Medium ATR, low momentum
            ("LINK/USDT", 0.4, 0.11, 65, 0.6),  # Low ATR, low momentum
            ("DOT/USDT", 0.5, 0.13, 60, 0.5),   # Current market typical, low momentum
        ]

        for symbol, atr_pct, momentum, rsi, volume_ratio in bootstrap_data:
            obs = MarketSnapshot(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                atr_pct=atr_pct,
                momentum_1h=momentum,
                rsi=rsi,
                volume_ratio=volume_ratio
            )
            self._observations.append(obs)
            self.stats['observations_total'] += 1

    def _load_ml_models(self):
        """Load ML models for adaptive thresholds."""
        try:
            # Find latest model directory
            if not self.adaptive_models_dir.exists():
                return False

            model_dirs = [d for d in self.adaptive_models_dir.iterdir() if d.is_dir()]
            if not model_dirs:
                return False

            latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)

            # Load ATR threshold model
            atr_model_path = latest_dir / "optimal_atr_threshold_xgboost.joblib"
            atr_features_path = latest_dir / "optimal_atr_threshold_features.json"

            if atr_model_path.exists() and atr_features_path.exists():
                self._atr_model = joblib.load(atr_model_path)
                with open(atr_features_path, 'r') as f:
                    self._atr_features = json.load(f)
                return True

        except Exception as e:
            print(f"Warning: Failed to load ML models: {e}")
            return False

        return False

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
        """Calculate adaptive thresholds with automatic ML transition."""

        with self._lock:
            observations = list(self._observations)

        n = len(observations)

        # PHASE 1: WARMUP - Use percentile-based thresholds while learning
        if n < self.min_samples:
            # Use percentile-based thresholds from recent observations
            # If no observations yet, use fallbacks
            if n >= 10:  # Minimum for basic percentile calculation
                atrs = np.array([o.atr_pct for o in observations])
                moms = np.array([o.momentum_1h for o in observations])
                rsis = np.array([o.rsi for o in observations])

                # Calculate current market percentiles
                atr_threshold = np.percentile(atrs, self.atr_percentile)
                mom_threshold = np.percentile(moms, self.momentum_percentile)
                rsi_long_max = np.percentile(rsis, self.rsi_extreme_pct)
                rsi_short_min = np.percentile(rsis, 100 - self.rsi_extreme_pct)

                median_atr = np.median(atrs)
                if median_atr < 0.8:
                    volatility = 'low'
                elif median_atr > 2.0:
                    volatility = 'high'
                else:
                    volatility = 'normal'

                print(f"[ADAPTIVE_WARMUP] Learning market patterns... {n}/{self.min_samples} observations")
                print(".3f")
                print(".3f")
            else:
                # Not enough data even for basic percentiles - use fallbacks
                atr_threshold = self.fallback_atr
                mom_threshold = self.fallback_momentum
                rsi_long_max = self.fallback_rsi_long
                rsi_short_min = self.fallback_rsi_short
                volatility = 'unknown'

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

        # PHASE 2: ML-ACTIVE - Use ML predictions once warmed up
        # Extract arrays
        atrs = np.array([o.atr_pct for o in observations])
        moms = np.array([o.momentum_1h for o in observations])
        rsis = np.array([o.rsi for o in observations])

        # ATR THRESHOLD: ML prediction with percentile fallback
        atr_threshold = self._predict_atr_threshold(atrs, moms, rsis)
        if atr_threshold is None:
            # Fallback to percentile-based
            atr_threshold = np.percentile(atrs, self.atr_percentile)
            self.stats['percentile_fallbacks'] += 1

        # Momentum and RSI still use percentiles (could be ML in future)
        mom_threshold = np.percentile(moms, self.momentum_percentile)
        rsi_long_max = np.percentile(rsis, self.rsi_extreme_pct)
        rsi_short_min = np.percentile(rsis, 100 - self.rsi_extreme_pct)

        # Determine market regime based on current median ATR
        median_atr = np.median(atrs)
        if median_atr < 0.8:
            volatility = 'low'
        elif median_atr > 2.0:
            volatility = 'high'
        else:
            volatility = 'normal'

        # Log transition to ML (only once)
        if not hasattr(self, '_ml_activated'):
            print(f"[ADAPTIVE_ML_ACTIVE] ML ATR adaptation now active! {n} observations")
            print(".3f")
            self._ml_activated = True

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

    def _predict_atr_threshold(self, atrs: np.ndarray, moms: np.ndarray, rsis: np.ndarray) -> Optional[float]:
        """
        Use ML model to predict optimal ATR threshold for current market conditions.
        Returns None if ML prediction fails.
        """
        try:
            # Load model if not loaded
            if self._atr_model is None:
                if not self._load_ml_models():
                    return None

            # Calculate market features for ML prediction
            market_volatility = np.median(atrs)
            market_trend_strength = np.std(moms)  # Trend strength = momentum variability
            market_avg_volume = np.median([o.volume_ratio for o in self._observations])
            market_avg_spread = 0.5  # Placeholder - would need spread data

            # Create feature vector (must match training features)
            if self._atr_features and 'market_volatility' in self._atr_features:
                features = [
                    market_volatility,
                    market_trend_strength,
                    market_avg_volume,
                    market_avg_spread
                ]

                # Predict optimal ATR threshold
                predicted_threshold = self._atr_model.predict([features])[0]
                self.stats['ml_predictions'] += 1

                # Ensure reasonable bounds
                predicted_threshold = np.clip(predicted_threshold, 0.001, 0.02)  # 0.1% to 2.0%

                return float(predicted_threshold)

        except Exception as e:
            print(f"Warning: ML ATR prediction failed: {e}")
            return None

        return None
    
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
