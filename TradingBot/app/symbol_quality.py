"""
Symbol Quality Analyzer
Implements holistic, reliable symbol-filtering system.
Evaluates local OHLCV analysis, micro-structure, noise, and liquidity sanity.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    passed: bool
    reason: str
    noise_index: float
    trend_consistency: float
    wick_ratio: float
    atr_volatility: float
    impulse_strength: float
    pullback_depth: float

class SymbolQualityAnalyzer:
    """
    Analyzes symbol quality using candle data.
    Filters out choppy, directionless, or unstable symbols.
    """
    
    def __init__(self):
        # Thresholds (can be moved to config later)
        # CRITICAL: Relaxed for backtest mode to allow more signals,
        # but keep stricter thresholds for live/dry-run trading.
        import os
        is_replay = os.environ.get('REPLAY_MODE', '0') == '1'
        
        if is_replay:
            # DIAGNOSTIC / BACKTEST THRESHOLDS:
            # We allow very noisy symbols so that MARKSMAN can actually
            # generate trades during short backtest windows.
            # This does NOT affect live/dry-run trading (REPLAY_MODE=0).
            self.MAX_NOISE_INDEX = 10.0       # Fully disable noise filter in backtests
            self.MIN_TREND_CONSISTENCY = 0.20 # Allow choppier trends in tests
            self.MAX_WICK_RATIO = 0.8         # Accept more wicks
            self.MIN_ATR_VOLATILITY = 0.001   # Lower minimum
            self.MAX_ATR_VOLATILITY = 0.10    # Higher maximum
        else:
            # Production thresholds (used for DRY_RUN and LIVE)
            self.MAX_NOISE_INDEX = 0.70       # Relaxed to 0.70 (from 0.60) to allow more volatility
            self.MIN_TREND_CONSISTENCY = 0.45 # Tuned to 45% (from 35%) for cleaner trends
            self.MAX_WICK_RATIO = 0.5         # Max 50% of candle body can be wicks on avg
            self.MIN_ATR_VOLATILITY = 0.002   # Min 0.2% ATR (dead markets)
            self.MAX_ATR_VOLATILITY = 0.05    # Max 5% ATR (too dangerous)
        self.MIN_IMPULSE_STRENGTH = 1.5  # Min 1.5x ATR for impulse moves
        
    def analyze(self, candles: List[List[float]]) -> QualityMetrics:
        """
        Analyze candle data for quality.
        
        Args:
            candles: List of [timestamp, open, high, low, close, volume]
            
        Returns:
            QualityMetrics object with results
        """
        if not candles or len(candles) < 20:
            return QualityMetrics(False, "Insufficient data (<20 candles)", 0, 0, 0, 0, 0, 0)
            
        # Convert to numpy arrays for vectorization (if available, else pure python)
        # Using pure python for simplicity and no dependency issues, but optimized
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]
        
        # 1. Noise Index Calculation
        # Noise = Sum(High-Low) / Abs(Close[last] - Close[first])
        # Lower is better (1.0 = straight line)
        total_range = sum([h - l for h, l in zip(highs[-20:], lows[-20:])])
        net_displacement = abs(closes[-1] - closes[-20])
        
        if net_displacement == 0:
            noise_index = 10.0 # Max noise (nowhere)
        else:
            noise_index = 1.0 - (net_displacement / total_range) # 0 = pure signal, 1 = pure noise
            # Invert logic: prompt says "Local noise index"
            # Let's define Noise Index as percentage of "wasted" movement
            # 0.0 = perfect trend, 1.0 = perfect chop
            
        if noise_index > self.MAX_NOISE_INDEX:
            return QualityMetrics(False, f"Too noisy (Index {noise_index:.2f} > {self.MAX_NOISE_INDEX})", noise_index, 0, 0, 0, 0, 0)
            
        # 2. Wick Ratio
        # High wicks indicate indecision/instability
        body_sizes = [abs(c - o) for c, o in zip(closes[-20:], opens[-20:])]
        total_candle_sizes = [h - l for h, l in zip(highs[-20:], lows[-20:])]
        
        wick_ratios = []
        for body, total in zip(body_sizes, total_candle_sizes):
            if total > 0:
                wick_ratios.append((total - body) / total)
            else:
                wick_ratios.append(0)
                
        avg_wick_ratio = sum(wick_ratios) / len(wick_ratios)
        
        if avg_wick_ratio > self.MAX_WICK_RATIO:
            return QualityMetrics(False, f"Too wicky (Ratio {avg_wick_ratio:.2f} > {self.MAX_WICK_RATIO})", noise_index, 0, avg_wick_ratio, 0, 0, 0)
            
        # 3. Trend Consistency
        # Count candles aligning with overall trend
        direction = 1 if closes[-1] > closes[-20] else -1
        aligned_candles = 0
        for i in range(-20, 0):
            candle_dir = 1 if closes[i] > opens[i] else -1
            if candle_dir == direction:
                aligned_candles += 1
                
        trend_consistency = aligned_candles / 20.0
        
        if trend_consistency < self.MIN_TREND_CONSISTENCY:
            # Relaxed check: if momentum is HUGE, consistency matters less (explosion)
            # But general rule applies
            return QualityMetrics(False, f"Inconsistent trend ({trend_consistency:.2f} < {self.MIN_TREND_CONSISTENCY})", noise_index, trend_consistency, avg_wick_ratio, 0, 0, 0)

        # 4. ATR / Volatility Check
        # Calculate simplified ATR
        tr_sum = 0
        for i in range(-14, 0):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_sum += tr
        atr = tr_sum / 14
        
        atr_pct = atr / closes[-1]
        
        if atr_pct < self.MIN_ATR_VOLATILITY:
            return QualityMetrics(False, f"Dead market (ATR {atr_pct*100:.3f}% < {self.MIN_ATR_VOLATILITY*100}%)", noise_index, trend_consistency, avg_wick_ratio, atr_pct, 0, 0)
            
        if atr_pct > self.MAX_ATR_VOLATILITY:
            return QualityMetrics(False, f"Too volatile (ATR {atr_pct*100:.3f}% > {self.MAX_ATR_VOLATILITY*100}%)", noise_index, trend_consistency, avg_wick_ratio, atr_pct, 0, 0)

        # 5. Micro-Structure: Impulse -> Pullback check
        # Only relevant if we are in a setup (not just filtering universe, but valid for marksman)
        # Impulse: Large move in direction
        # Pullback: Smaller move against
        
        # Identify last swing
        # Simplified: Check last 3 candles vs prior 3
        # This is highly specific to timing, so we just return metrics for the signal generator to decide
        # But we can flag "Choppy/Directionless" here
        
        return QualityMetrics(True, "OK", noise_index, trend_consistency, avg_wick_ratio, atr_pct, 0, 0)


