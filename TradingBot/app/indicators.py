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


# ============================================================================
# NEW INDICATORS FOR BETTER PERFORMANCE
# ============================================================================

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of closing prices (most recent last)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        Dict with 'macd', 'signal', 'histogram' or None if insufficient data
    """
    if len(prices) < slow + signal:
        return None
    
    try:
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_values = [ema_fast - ema_slow for ema_fast, ema_slow in zip(
            [calculate_ema(prices[:i], fast) for i in range(slow, len(prices) + 1)],
            [calculate_ema(prices[:i], slow) for i in range(slow, len(prices) + 1)]
        )]
        
        if len(macd_values) < signal:
            return None
        
        signal_line = calculate_ema(macd_values, signal)
        histogram = macd_line - signal_line if signal_line else 0
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
        }
    except (ValueError, IndexError, TypeError):
        return None


def calculate_stochastic_rsi(prices: List[float], rsi_period: int = 14, k_period: int = 3, d_period: int = 3) -> Optional[Dict[str, float]]:
    """
    Calculate Stochastic RSI (RSI of RSI) for overbought/oversold extremes.
    
    Args:
        prices: List of closing prices (most recent last)
        rsi_period: RSI period (default 14)
        k_period: K smoothing period (default 3)
        d_period: D smoothing period (default 3)
    
    Returns:
        Dict with '%K', '%D', or None if insufficient data
    """
    if len(prices) < rsi_period + k_period + d_period:
        return None
    
    try:
        # Calculate RSI values for the last (rsi_period + k_period + d_period) candles
        rsi_values = []
        for i in range(len(prices) - rsi_period - k_period - d_period + 1, len(prices) + 1):
            rsi = calculate_rsi(prices[:i], rsi_period)
            if rsi is not None:
                rsi_values.append(rsi)
        
        if len(rsi_values) < k_period + d_period:
            return None
        
        # Calculate Stochastic: (RSI - min RSI) / (max RSI - min RSI)
        min_rsi = min(rsi_values[-k_period:]) if len(rsi_values) >= k_period else min(rsi_values)
        max_rsi = max(rsi_values[-k_period:]) if len(rsi_values) >= k_period else max(rsi_values)
        
        if max_rsi == min_rsi:
            stoch_rsi = 50.0
        else:
            stoch_rsi = ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100.0
        
        # Calculate K and D lines
        k_line = stoch_rsi  # Simplified: just current stoch RSI
        d_line = calculate_ema([stoch_rsi], d_period) if len(rsi_values) >= d_period else stoch_rsi
        
        return {
            'K': k_line,
            'D': d_line,
        }
    except (ValueError, IndexError, TypeError, ZeroDivisionError):
        return None


def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Optional[Dict[str, float]]:
    """
    Calculate Bollinger Bands for dynamic support/resistance.
    
    Args:
        prices: List of closing prices (most recent last)
        period: Moving average period (default 20)
        num_std: Number of standard deviations (default 2.0)
    
    Returns:
        Dict with 'upper', 'middle', 'lower', 'width' or None
    """
    if len(prices) < period:
        return None
    
    try:
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in recent_prices) / period
        std_dev = variance ** 0.5
        
        upper = middle + (num_std * std_dev)
        lower = middle - (num_std * std_dev)
        width = upper - lower
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'width_pct': (width / middle * 100) if middle > 0 else 0,
        }
    except (ValueError, IndexError, ZeroDivisionError):
        return None


def calculate_volume_confirmation(volume: List[float], prices: List[float], period: int = 20) -> Optional[float]:
    """
    Calculate volume confirmation strength (0-1).
    Price move with volume > average volume = strong
    
    Args:
        volume: List of volumes (most recent last)
        prices: List of prices (most recent last)
        period: Period for average volume (default 20)
    
    Returns:
        Confirmation score 0-1, or None if insufficient data
    """
    if len(volume) < period or len(prices) < 2:
        return None
    
    try:
        avg_volume = sum(volume[-period:]) / period
        current_volume = volume[-1]
        current_move = abs(prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
        
        # Strong confirmation: volume > 1.5x average AND price moved
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 1.5 and current_move > 0.002:  # 0.2% move
            return min(1.0, (volume_ratio - 1.0) * 0.5 + (current_move * 100))  # Normalized
        elif volume_ratio > 1.2:
            return 0.6
        else:
            return 0.3  # Below average volume = weak
    except (ValueError, IndexError, ZeroDivisionError):
        return None


def detect_rsi_divergence(prices: List[float], highs: List[float], lows: List[float], lookback: int = 10) -> Optional[str]:
    """
    Detect RSI divergence (price makes new high/low but RSI doesn't).
    Early exit signal for reversals.
    
    Args:
        prices: Closing prices
        highs: High prices
        lows: Low prices
        lookback: Lookback period to check (default 10)
    
    Returns:
        'bullish_div' (RSI higher, price lower - uptrend), 
        'bearish_div' (RSI lower, price higher - downtrend),
        or None if no divergence
    """
    if len(prices) < lookback + 5 or len(highs) < lookback + 5 or len(lows) < lookback + 5:
        return None
    
    try:
        # Get current and previous RSI
        current_rsi = calculate_rsi(prices, 14)
        prev_rsi = calculate_rsi(prices[:-1], 14) if len(prices) > 15 else None
        
        if current_rsi is None or prev_rsi is None:
            return None
        
        # Compare highs/lows
        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])
        prev_high = max(highs[-lookback-5:-5]) if len(highs) > lookback + 5 else recent_high
        prev_low = min(lows[-lookback-5:-5]) if len(lows) > lookback + 5 else recent_low
        
        # Bearish divergence: price makes new high, but RSI makes lower high
        if recent_high > prev_high and current_rsi < prev_rsi and current_rsi > 70:
            return 'bearish_div'
        
        # Bullish divergence: price makes new low, but RSI makes higher low
        if recent_low < prev_low and current_rsi > prev_rsi and current_rsi < 30:
            return 'bullish_div'
        
        return None
    except (ValueError, IndexError, TypeError):
        return None
