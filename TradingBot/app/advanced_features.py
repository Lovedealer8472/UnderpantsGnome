"""
Advanced Feature Engineering - Higher timeframe, market structure, reversals, divergences.
Pandas-based computations for detecting bigger-picture context and reversal signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class AdvancedFeatures:
    """Advanced feature set for reversal and structure detection."""
    # Higher timeframe
    htf_ema50: Optional[float] = None
    htf_ema200: Optional[float] = None
    htf_adx_14: Optional[float] = None
    htf_trend_dir: int = 0  # 1=up, -1=down, 0=neutral
    htf_trend_strength: float = 0.0  # normalized ADX
    htf_price_position: int = 0  # 1=above both, -1=below both, 0=between
    
    # Market structure
    swing_high_price: Optional[float] = None
    swing_low_price: Optional[float] = None
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    bos_up: bool = False
    bos_down: bool = False
    choch_up: bool = False
    choch_down: bool = False
    
    # Extension
    dist_from_ema20: float = 0.0
    dist_from_ema50: float = 0.0
    dist_from_htf_ema50: float = 0.0
    dist_from_vwap: float = 0.0
    atr_zscore: float = 0.0
    extreme_up: bool = False
    extreme_down: bool = False
    
    # Momentum/Divergence
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    momentum: float = 0.0
    bearish_divergence: bool = False
    bullish_divergence: bool = False
    
    # Bands/Squeeze
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    kc_upper: Optional[float] = None
    kc_lower: Optional[float] = None
    price_above_bb: bool = False
    price_below_bb: bool = False
    bb_snapback: bool = False
    kc_tightness: bool = False
    exhaustion_up: bool = False
    exhaustion_down: bool = False
    
    # Levels/SFP
    htf_range_high: Optional[float] = None
    htf_range_low: Optional[float] = None
    sfp_top: bool = False
    sfp_bottom: bool = False
    
    # Reversal scores
    reversal_score_long: float = 0.0
    reversal_score_short: float = 0.0


def compute_higher_timeframe_features(htf_df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Compute HTF features: EMA50, EMA200, ADX, trend direction."""
    if len(htf_df) < 200:
        return {}
    
    close = htf_df['close'].values
    high = htf_df['high'].values
    low = htf_df['low'].values
    
    # EMAs
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    if len(ema50) == 0 or len(ema200) == 0:
        return {}
    
    ema50_val = ema50[-1]
    ema200_val = ema200[-1]
    
    # ADX
    adx = _adx(high, low, close, 14)
    adx_val = adx[-1] if len(adx) > 0 else None
    
    # Trend direction
    if current_price > ema50_val > ema200_val:
        trend_dir = 1
    elif current_price < ema50_val < ema200_val:
        trend_dir = -1
    else:
        trend_dir = 0
    
    trend_strength = (adx_val / 50.0) if adx_val else 0.0  # normalize ADX (max ~50)
    
    # Price position
    if current_price > ema50_val and current_price > ema200_val:
        price_position = 1
    elif current_price < ema50_val and current_price < ema200_val:
        price_position = -1
    else:
        price_position = 0
    
    return {
        'htf_ema50': ema50_val,
        'htf_ema200': ema200_val,
        'htf_adx_14': adx_val,
        'htf_trend_dir': trend_dir,
        'htf_trend_strength': trend_strength,
        'htf_price_position': price_position
    }


def compute_market_structure_features(ltf_df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """Compute swing highs/lows, BOS, CHOCH."""
    if len(ltf_df) < lookback * 2 + 1:
        return {}
    
    high = ltf_df['high'].values
    low = ltf_df['low'].values
    close = ltf_df['close'].values
    
    # Pivot detection
    pivot_highs = []
    pivot_lows = []
    swing_high_prices = []
    swing_low_prices = []
    
    for i in range(lookback, len(high) - lookback):
        # Pivot high
        if high[i] > np.max(high[i-lookback:i]) and high[i] > np.max(high[i+1:i+lookback+1]):
            pivot_highs.append(i)
            swing_high_prices.append(high[i])
        # Pivot low
        if low[i] < np.min(low[i-lookback:i]) and low[i] < np.min(low[i+1:i+lookback+1]):
            pivot_lows.append(i)
            swing_low_prices.append(low[i])
    
    last_swing_high = swing_high_prices[-1] if swing_high_prices else None
    last_swing_low = swing_low_prices[-1] if swing_low_prices else None
    current_high = high[-1]
    current_low = low[-1]
    
    # BOS detection
    bos_up = current_high > last_swing_high if last_swing_high else False
    bos_down = current_low < last_swing_low if last_swing_low else False
    
    # CHOCH detection (BOS after series of lower/higher swings)
    choch_up = False
    choch_down = False
    if len(swing_high_prices) >= 3 and bos_up:
        recent_highs = swing_high_prices[-3:]
        choch_up = recent_highs[0] > recent_highs[1] > recent_highs[2] and current_high > recent_highs[0]
    
    if len(swing_low_prices) >= 3 and bos_down:
        recent_lows = swing_low_prices[-3:]
        choch_down = recent_lows[0] < recent_lows[1] < recent_lows[2] and current_low < recent_lows[0]
    
    return {
        'swing_high_price': swing_high_prices[-1] if swing_high_prices else None,
        'swing_low_price': swing_low_prices[-1] if swing_low_prices else None,
        'last_swing_high': last_swing_high,
        'last_swing_low': last_swing_low,
        'bos_up': bos_up,
        'bos_down': bos_down,
        'choch_up': choch_up,
        'choch_down': choch_down
    }


def compute_extension_features(ltf_df: pd.DataFrame, current_price: float, htf_ema50: Optional[float] = None) -> Dict[str, Any]:
    """Compute distance from EMAs/VWAP, ATR z-score, extreme conditions."""
    if len(ltf_df) < 50:
        return {}
    
    close = ltf_df['close'].values
    high = ltf_df['high'].values
    low = ltf_df['low'].values
    
    # EMAs
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    if len(ema20) == 0 or len(ema50) == 0:
        return {}
    
    ema20_val = ema20[-1]
    ema50_val = ema50[-1]
    
    # VWAP (typical price * volume, simplified to TP)
    typical_price = (high + low + close) / 3.0
    if 'volume' in ltf_df.columns:
        volume = ltf_df['volume'].values
        vwap = np.sum(typical_price * volume) / np.sum(volume)
    else:
        vwap = np.mean(typical_price)
    
    # ATR and z-score
    atr = _atr(high, low, close, 14)
    atr_val = atr[-1] if len(atr) > 0 else None
    
    sma_close = _sma(close, 20)
    sma_close_val = sma_close[-1] if len(sma_close) > 0 else None
    
    # Distances
    dist_from_ema20 = (current_price - ema20_val) / ema20_val if ema20_val > 0 else 0.0
    dist_from_ema50 = (current_price - ema50_val) / ema50_val if ema50_val > 0 else 0.0
    dist_from_htf_ema50 = (current_price - htf_ema50) / htf_ema50 if htf_ema50 and htf_ema50 > 0 else 0.0
    dist_from_vwap = (current_price - vwap) / vwap if vwap > 0 else 0.0
    
    # ATR z-score
    if atr_val and atr_val > 0 and sma_close_val:
        atr_zscore = (current_price - sma_close_val) / atr_val
    else:
        atr_zscore = 0.0
    
    # Extreme conditions
    extreme_up = dist_from_ema50 > 0.02 or atr_zscore > 2.0
    extreme_down = dist_from_ema50 < -0.02 or atr_zscore < -2.0
    
    return {
        'dist_from_ema20': dist_from_ema20,
        'dist_from_ema50': dist_from_ema50,
        'dist_from_htf_ema50': dist_from_htf_ema50,
        'dist_from_vwap': dist_from_vwap,
        'atr_zscore': atr_zscore,
        'extreme_up': extreme_up,
        'extreme_down': extreme_down
    }


def compute_momentum_features(ltf_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute RSI, MACD, momentum, divergences."""
    if len(ltf_df) < 50:
        return {}
    
    close = ltf_df['close'].values
    high = ltf_df['high'].values
    low = ltf_df['low'].values
    
    # RSI
    rsi = _rsi(close, 14)
    rsi_val = rsi[-1] if len(rsi) > 0 else None
    
    # MACD
    macd_hist = _macd_histogram(close, 12, 26, 9)
    macd_hist_val = macd_hist[-1] if len(macd_hist) > 0 else None
    
    # Momentum (ROC)
    momentum = ((close[-1] - close[-6]) / close[-6]) if len(close) >= 6 else 0.0
    
    # Divergence detection (simplified: check recent highs/lows vs RSI/MACD)
    bearish_divergence = False
    bullish_divergence = False
    
    if len(high) >= 20 and len(low) >= 20 and rsi_val and macd_hist_val:
        recent_highs = high[-20:]
        recent_lows = low[-20:]
        recent_rsi = rsi[-20:]
        recent_macd = macd_hist[-20:]
        
        # Bearish: price higher high, RSI/MACD lower high
        price_higher_high = np.argmax(recent_highs) > len(recent_highs) // 2
        if price_higher_high and len(recent_rsi) > 0:
            rsi_lower_high = np.max(recent_rsi[:len(recent_rsi)//2]) > np.max(recent_rsi[len(recent_rsi)//2:])
            macd_lower_high = np.max(recent_macd[:len(recent_macd)//2]) > np.max(recent_macd[len(recent_macd)//2:])
            bearish_divergence = rsi_lower_high or macd_lower_high
        
        # Bullish: price lower low, RSI/MACD higher low
        price_lower_low = np.argmin(recent_lows) > len(recent_lows) // 2
        if price_lower_low and len(recent_rsi) > 0:
            rsi_higher_low = np.min(recent_rsi[:len(recent_rsi)//2]) < np.min(recent_rsi[len(recent_rsi)//2:])
            macd_higher_low = np.min(recent_macd[:len(recent_macd)//2]) < np.min(recent_macd[len(recent_macd)//2:])
            bullish_divergence = rsi_higher_low or macd_higher_low
    
    return {
        'rsi': rsi_val,
        'macd_hist': macd_hist_val,
        'momentum': momentum,
        'bearish_divergence': bearish_divergence,
        'bullish_divergence': bullish_divergence
    }


def compute_band_features(ltf_df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Compute Bollinger Bands, Keltner Channel, exhaustion."""
    if len(ltf_df) < 20:
        return {}
    
    close = ltf_df['close'].values
    high = ltf_df['high'].values
    low = ltf_df['low'].values
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    sma = _sma(close, bb_period)
    if len(sma) == 0:
        return {}
    
    sma_val = sma[-1]
    std = np.std(close[-bb_period:])
    bb_upper = sma_val + (bb_std * std)
    bb_lower = sma_val - (bb_std * std)
    
    # Keltner Channel
    kc_period = 20
    kc_mult = 1.5
    atr = _atr(high, low, close, kc_period)
    if len(atr) == 0:
        return {}
    
    kc_upper = sma_val + (kc_mult * atr[-1])
    kc_lower = sma_val - (kc_mult * atr[-1])
    
    price_above_bb = current_price > bb_upper
    price_below_bb = current_price < bb_lower
    
    # BB snapback
    bb_snapback = False
    if len(close) >= 2:
        bb_snapback = close[-2] > bb_upper and close[-1] < bb_upper
    
    # KC tightness
    bb_width = bb_upper - bb_lower
    kc_width = kc_upper - kc_lower
    kc_tightness = bb_width < kc_width
    
    # Exhaustion (requires divergence from momentum features)
    # This will be combined with divergence in compute_reversal_scores
    
    return {
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'kc_upper': kc_upper,
        'kc_lower': kc_lower,
        'price_above_bb': price_above_bb,
        'price_below_bb': price_below_bb,
        'bb_snapback': bb_snapback,
        'kc_tightness': kc_tightness
    }


def compute_level_features(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, current_price: float, current_high: float, current_low: float) -> Dict[str, Any]:
    """Compute HTF range, Swing Failure Pattern (SFP)."""
    if len(htf_df) < 50:
        return {}
    
    htf_high = htf_df['high'].values
    htf_low = htf_df['low'].values
    htf_close = htf_df['close'].values
    
    # HTF range (last 50 periods)
    lookback = min(50, len(htf_high))
    htf_range_high = np.max(htf_high[-lookback:])
    htf_range_low = np.min(htf_low[-lookback:])
    
    # SFP detection (simplified: needs current and next candle)
    sfp_top = False
    sfp_bottom = False
    
    if len(ltf_df) >= 2:
        prev_high = ltf_df['high'].iloc[-2]
        prev_close = ltf_df['close'].iloc[-2]
        current_close = ltf_df['close'].iloc[-1]
        
        # SFP Top: breaks HTF range high then closes below
        if current_high > htf_range_high and current_close < htf_range_high:
            sfp_top = True
        
        # SFP Bottom: breaks HTF range low then closes above
        if current_low < htf_range_low and current_close > htf_range_low:
            sfp_bottom = True
    
    return {
        'htf_range_high': htf_range_high,
        'htf_range_low': htf_range_low,
        'sfp_top': sfp_top,
        'sfp_bottom': sfp_bottom
    }


def compute_reversal_scores(features: AdvancedFeatures) -> Tuple[float, float]:
    """Compute reversal scores for long and short based on all features."""
    long_score = 0.0
    short_score = 0.0
    
    # LONG SCORE BOOSTERS
    if features.htf_trend_dir == 1:
        long_score += 5
    if features.extreme_down:
        long_score += 10
    if features.bullish_divergence:
        long_score += 8
    if features.exhaustion_down:
        long_score += 12
    if features.sfp_bottom:
        long_score += 15
    if features.dist_from_vwap < -0.015:
        long_score += 6
    if features.last_swing_low and features.dist_from_vwap < -0.01:
        long_score += 5
    
    # SHORT SCORE BOOSTERS
    if features.htf_trend_dir == -1:
        short_score += 5
    if features.extreme_up:
        short_score += 10
    if features.bearish_divergence:
        short_score += 8
    if features.exhaustion_up:
        short_score += 12
    if features.sfp_top:
        short_score += 15
    if features.dist_from_vwap > 0.015:
        short_score += 6
    if features.last_swing_high and features.dist_from_vwap > 0.01:
        short_score += 5
    
    # NEUTRALIZATION: Low ADX reduces trend-following scores
    if features.htf_trend_strength < 15:
        if features.htf_trend_dir == 1:
            long_score *= 0.5
        if features.htf_trend_dir == -1:
            short_score *= 0.5
    
    # OVEREXTENSION PROTECTION: Mid-range reduces reversal scores
    if -0.01 < features.dist_from_htf_ema50 < 0.01:
        if features.extreme_down or features.bullish_divergence or features.sfp_bottom:
            long_score *= 0.7
        if features.extreme_up or features.bearish_divergence or features.sfp_top:
            short_score *= 0.7
    
    # Exhaustion needs divergence + band conditions
    if features.price_below_bb and features.bb_snapback and features.bullish_divergence:
        features.exhaustion_down = True
        long_score += 12
    
    if features.price_above_bb and features.bb_snapback and features.bearish_divergence:
        features.exhaustion_up = True
        short_score += 12
    
    return long_score, short_score


def compute_all_features(
    ltf_df: pd.DataFrame,
    htf_df: Optional[pd.DataFrame],
    current_price: float,
    current_high: float,
    current_low: float
) -> AdvancedFeatures:
    """Compute all advanced features and return AdvancedFeatures object."""
    features = AdvancedFeatures()
    
    # Higher timeframe features
    if htf_df is not None and len(htf_df) >= 200:
        htf_data = compute_higher_timeframe_features(htf_df, current_price)
        features.htf_ema50 = htf_data.get('htf_ema50')
        features.htf_ema200 = htf_data.get('htf_ema200')
        features.htf_adx_14 = htf_data.get('htf_adx_14')
        features.htf_trend_dir = htf_data.get('htf_trend_dir', 0)
        features.htf_trend_strength = htf_data.get('htf_trend_strength', 0.0)
        features.htf_price_position = htf_data.get('htf_price_position', 0)
    else:
        features.htf_ema50 = None
        features.htf_ema200 = None
        features.htf_adx_14 = None
    
    # Market structure
    structure_data = compute_market_structure_features(ltf_df)
    features.swing_high_price = structure_data.get('swing_high_price')
    features.swing_low_price = structure_data.get('swing_low_price')
    features.last_swing_high = structure_data.get('last_swing_high')
    features.last_swing_low = structure_data.get('last_swing_low')
    features.bos_up = structure_data.get('bos_up', False)
    features.bos_down = structure_data.get('bos_down', False)
    features.choch_up = structure_data.get('choch_up', False)
    features.choch_down = structure_data.get('choch_down', False)
    
    # Extension
    ext_data = compute_extension_features(ltf_df, current_price, features.htf_ema50)
    features.dist_from_ema20 = ext_data.get('dist_from_ema20', 0.0)
    features.dist_from_ema50 = ext_data.get('dist_from_ema50', 0.0)
    features.dist_from_htf_ema50 = ext_data.get('dist_from_htf_ema50', 0.0)
    features.dist_from_vwap = ext_data.get('dist_from_vwap', 0.0)
    features.atr_zscore = ext_data.get('atr_zscore', 0.0)
    features.extreme_up = ext_data.get('extreme_up', False)
    features.extreme_down = ext_data.get('extreme_down', False)
    
    # Momentum
    mom_data = compute_momentum_features(ltf_df)
    features.rsi = mom_data.get('rsi')
    features.macd_hist = mom_data.get('macd_hist')
    features.momentum = mom_data.get('momentum', 0.0)
    features.bearish_divergence = mom_data.get('bearish_divergence', False)
    features.bullish_divergence = mom_data.get('bullish_divergence', False)
    
    # Bands
    band_data = compute_band_features(ltf_df, current_price)
    features.bb_upper = band_data.get('bb_upper')
    features.bb_lower = band_data.get('bb_lower')
    features.kc_upper = band_data.get('kc_upper')
    features.kc_lower = band_data.get('kc_lower')
    features.price_above_bb = band_data.get('price_above_bb', False)
    features.price_below_bb = band_data.get('price_below_bb', False)
    features.bb_snapback = band_data.get('bb_snapback', False)
    features.kc_tightness = band_data.get('kc_tightness', False)
    
    # Levels
    if htf_df is not None:
        level_data = compute_level_features(htf_df, ltf_df, current_price, current_high, current_low)
        features.htf_range_high = level_data.get('htf_range_high')
        features.htf_range_low = level_data.get('htf_range_low')
        features.sfp_top = level_data.get('sfp_top', False)
        features.sfp_bottom = level_data.get('sfp_bottom', False)
    
    # Compute reversal scores
    features.reversal_score_long, features.reversal_score_short = compute_reversal_scores(features)
    
    return features


# Helper functions for technical indicators

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    if len(prices) < period:
        return np.array([])
    ema = np.zeros(len(prices))
    ema[period-1] = np.mean(prices[:period])
    multiplier = 2.0 / (period + 1.0)
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema[period-1:]


def _sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    if len(prices) < period:
        return np.array([])
    sma = np.zeros(len(prices) - period + 1)
    for i in range(len(sma)):
        sma[i] = np.mean(prices[i:i+period])
    return sma


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range."""
    if len(high) < period + 1:
        return np.array([])
    tr = np.zeros(len(high) - 1)
    for i in range(1, len(high)):
        tr[i-1] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = _sma(tr, period)
    return atr


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index."""
    if len(high) < period * 2:
        return np.array([])
    
    # Directional Movement
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    
    for i in range(1, len(high)):
        move_up = high[i] - high[i-1]
        move_down = low[i-1] - low[i]
        if move_up > move_down and move_up > 0:
            plus_dm[i] = move_up
        if move_down > move_up and move_down > 0:
            minus_dm[i] = move_down
    
    tr = _atr(high, low, close, period)
    if len(tr) == 0:
        return np.array([])
    
    # Smooth DM
    plus_di = np.zeros(len(tr))
    minus_di = np.zeros(len(tr))
    
    for i in range(len(tr)):
        if tr[i] > 0:
            plus_di[i] = 100.0 * (np.mean(plus_dm[i:i+period]) / tr[i]) if i+period <= len(plus_dm) else 0
            minus_di[i] = 100.0 * (np.mean(minus_dm[i:i+period]) / tr[i]) if i+period <= len(minus_dm) else 0
    
    # ADX
    dx = np.zeros(len(tr))
    for i in range(len(tr)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = _sma(dx, period)
    return adx


def _rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return np.array([])
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.zeros(len(gains) - period + 1)
    for i in range(len(rsi)):
        avg_gain = np.mean(gains[i:i+period])
        avg_loss = np.mean(losses[i:i+period])
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def _macd_histogram(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD Histogram."""
    if len(prices) < slow + signal:
        return np.array([])
    
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    
    if len(ema_fast) == 0 or len(ema_slow) == 0:
        return np.array([])
    
    # Align lengths
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]
    
    if len(macd_line) < signal:
        return np.array([])
    
    signal_line = _ema(macd_line, signal)
    if len(signal_line) == 0:
        return np.array([])
    
    # Align for histogram
    hist_len = min(len(macd_line), len(signal_line))
    histogram = macd_line[-hist_len:] - signal_line[-hist_len:]
    
    return histogram

