"""
Technical Indicators - RSI, EMA, ATR calculations for signal scoring.
OPTIMIZED: Added caching to reduce redundant calculations.
"""

import math
import time
from typing import List, Optional, Dict, Tuple
from functools import lru_cache


# OPTIMIZATION: Cache for indicator calculations (key: tuple of prices hash + period)
_indicator_cache = {}
_cache_ttl = 5.0  # 5 second TTL for indicator cache
_cache_max_size = 1000  # Max cache entries

def _get_cache_key(prices: List[float], period: int, indicator_type: str) -> str:
    """Generate cache key from prices and period."""
    # OPTIMIZATION: Use last few prices + period for cache key (faster than hashing full list)
    if len(prices) < 2:
        return None
    # Use last price, second-to-last price, and period as key
    key_data = (prices[-1], prices[-2] if len(prices) > 1 else 0, period, indicator_type)
    return str(key_data)

def _is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid."""
    return (time.time() - timestamp) < _cache_ttl

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).
    OPTIMIZED: Uses caching to avoid redundant calculations.
    
    Args:
        prices: List of closing prices (most recent last)
        period: RSI period (default 14)
    
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    # OPTIMIZATION: Check cache first
    cache_key = _get_cache_key(prices, period, 'rsi')
    if cache_key and cache_key in _indicator_cache:
        cached_value, cached_time = _indicator_cache[cache_key]
        if _is_cache_valid(cached_time):
            return cached_value
    
    # Calculate price changes
    # OPTIMIZATION: Use generator for deltas (more memory efficient)
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Separate gains and losses
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    
    # Calculate average gain and loss over period
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        rsi = 100.0  # All gains, no losses
    else:
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # OPTIMIZATION: Cache result
    if cache_key:
        # Clean cache if too large
        if len(_indicator_cache) >= _cache_max_size:
            # Remove oldest 20% of entries
            sorted_entries = sorted(_indicator_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_entries[:int(_cache_max_size * 0.2)]:
                del _indicator_cache[key]
        _indicator_cache[cache_key] = (rsi, time.time())
    
    return rsi


def calculate_ema(prices: List[float], period: int, previous_ema: Optional[float] = None) -> Optional[float]:
    """
    Calculate Exponential Moving Average (EMA).
    OPTIMIZED: Supports incremental updates with previous EMA value.
    
    Args:
        prices: List of closing prices (most recent last)
        period: EMA period
        previous_ema: Previous EMA value for incremental calculation (optional)
    
    Returns:
        EMA value or None if insufficient data
    """
    if len(prices) < period:
        return None
    
    # OPTIMIZATION: If previous EMA provided and we only have one new price, do incremental update
    if previous_ema is not None and len(prices) == period + 1:
        multiplier = 2.0 / (period + 1.0)
        new_price = prices[-1]
        ema = (new_price * multiplier) + (previous_ema * (1 - multiplier))
        return ema
    
    # Calculate smoothing factor
    multiplier = 2.0 / (period + 1.0)
    
    # Start with SMA
    ema = sum(prices[-period:]) / period
    
    # Calculate EMA for remaining prices
    # OPTIMIZATION: Only iterate over new prices if previous_ema provided
    start_idx = -period + 1 if previous_ema is None else -1
    for price in prices[start_idx:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR).
    
    Args:
        highs: List of high prices (most recent last)
        lows: List of low prices (most recent last)
        closes: List of closing prices (most recent last)
        period: ATR period (default 14)
    
    Returns:
        ATR value or None if insufficient data
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    
    # Calculate True Range for each period
    true_ranges = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]  # Current high - current low
        tr2 = abs(highs[i] - closes[i-1])  # Current high - previous close
        tr3 = abs(lows[i] - closes[i-1])  # Current low - previous close
        true_ranges.append(max(tr1, tr2, tr3))
    
    if len(true_ranges) < period:
        return None
    
    # Calculate ATR as SMA of True Ranges
    atr = sum(true_ranges[-period:]) / period
    
    return atr


def calculate_trend_alignment(ema20: Optional[float], ema50: Optional[float], ema100: Optional[float], side: str) -> float:
    """
    Calculate trend alignment score (0-1).
    
    For longs: EMA20 > EMA50 > EMA100 → 1.0
    For shorts: EMA20 < EMA50 < EMA100 → 1.0
    Mixed/flat → 0.5
    Opposite → 0.0-0.2
    
    Args:
        ema20: EMA(20) value
        ema50: EMA(50) value
        ema100: EMA(100) value
        side: 'long' or 'short'
    
    Returns:
        Trend alignment score (0-1)
    """
    # If EMAs not available, return neutral
    if ema20 is None or ema50 is None or ema100 is None:
        return 0.5
    
    if side == "long":
        # Perfect alignment: EMA20 > EMA50 > EMA100
        if ema20 > ema50 > ema100:
            return 1.0
        # Good alignment: EMA20 > EMA50
        elif ema20 > ema50:
            return 0.7
        # Mixed/flat
        elif ema20 > ema100 or ema50 > ema100:
            return 0.5
        # Opposite (bearish)
        else:
            return 0.2
    else:  # short
        # Perfect alignment: EMA20 < EMA50 < EMA100
        if ema20 < ema50 < ema100:
            return 1.0
        # Good alignment: EMA20 < EMA50
        elif ema20 < ema50:
            return 0.7
        # Mixed/flat
        elif ema20 < ema100 or ema50 < ema100:
            return 0.5
        # Opposite (bullish)
        else:
            return 0.2


def calculate_trend_direction_from_prices(prices: List[float], ema_short: int = 5, ema_long: int = 10) -> int:
    """
    Calculate trend direction from price data using EMA crossover.
    
    Args:
        prices: List of recent prices (most recent last)
        ema_short: Short EMA period (default 5)
        ema_long: Long EMA period (default 10)
    
    Returns:
        +1 for uptrend, -1 for downtrend, 0 for neutral
    """
    if not prices or len(prices) < max(ema_short, ema_long):
        return 0
    
    try:
        # Calculate short and long EMAs
        ema_s = calculate_ema(prices, ema_short)
        ema_l = calculate_ema(prices, ema_long)
        
        if ema_s is None or ema_l is None or len(ema_s) == 0 or len(ema_l) == 0:
            return 0
        
        # Compare most recent values
        current_short = ema_s[-1]
        current_long = ema_l[-1]
        
        # Trend strength threshold (1% difference)
        threshold = 0.01
        diff_pct = (current_short - current_long) / current_long if current_long > 0 else 0
        
        if diff_pct > threshold:
            return 1  # Uptrend
        elif diff_pct < -threshold:
            return -1  # Downtrend
        else:
            return 0  # Neutral
    
    except (ValueError, IndexError, ZeroDivisionError):
        return 0


def calculate_momentum_from_rsi(rsi: Optional[float], side: str) -> float:
    """
    Calculate momentum score from RSI (0-1).
    
    For longs: RSI 50-70 mapped linearly to 0-1
    RSI < 45 → 0
    RSI > 75 → taper down (overextended)
    
    For shorts: RSI 30-50 mapped linearly to 0-1
    RSI > 55 → 0
    RSI < 25 → taper down (oversold)
    
    Args:
        rsi: RSI value (0-100)
        side: 'long' or 'short'
    
    Returns:
        Momentum score (0-1)
    """
    if rsi is None:
        return 0.5  # Neutral if no RSI
    
    if side == "long":
        if rsi < 45:
            return 0.0
        elif rsi <= 70:
            # Linear mapping: 45 → 0, 70 → 1
            return (rsi - 45) / 25.0
        elif rsi <= 75:
            # Taper down: 70 → 1, 75 → 0.8
            return 1.0 - ((rsi - 70) / 5.0) * 0.2
        else:
            # Overextended: taper down more aggressively
            return max(0.0, 0.8 - ((rsi - 75) / 25.0) * 0.8)
    else:  # short
        if rsi > 55:
            return 0.0
        elif rsi >= 30:
            # Linear mapping: 55 → 0, 30 → 1
            return (55 - rsi) / 25.0
        elif rsi >= 25:
            # Taper down: 30 → 1, 25 → 0.8
            return 1.0 - ((30 - rsi) / 5.0) * 0.2
        else:
            # Oversold: taper down more aggressively
            return max(0.0, 0.8 - ((25 - rsi) / 25.0) * 0.8)

