"""
Unified Entry Filter System
============================

Consolidates all entry filtering logic into a single system:
- Adaptive filters (percentile-based)
- Position manager filters (limits, risk)
- Signal generator filters (score thresholds)
- Data-driven filters (ATR, momentum, RSI)

This is the single source of truth for all entry filtering decisions.
"""

import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from ..config import (
    MIN_SIGNAL_STRENGTH, MIN_SIGNAL_SCORE, HARD_MIN_SCORE,
    REPLAY_MIN_SIGNAL_STRENGTH, REPLAY_MIN_SIGNAL_SCORE, REPLAY_HARD_MIN_SCORE,
    MAX_CONCURRENT_POS, MAX_OPEN_POSITIONS,
    MIN_SPREAD_BPS, MAX_SPREAD_BPS, MIN_VOLUME_24H, MAX_LATENCY_MS,
    ENTRY_FILTER_ENABLED, ENTRY_MIN_ATR_PCT, ENTRY_MIN_MOMENTUM_PCT,
    ENTRY_RSI_LONG_MAX, ENTRY_RSI_SHORT_MIN,
    ENTRY_FILTER_RELAXED, ENTRY_MIN_ATR_PCT_RELAXED, ENTRY_MIN_MOMENTUM_PCT_RELAXED,
    ENTRY_RSI_LONG_MAX_RELAXED, ENTRY_RSI_SHORT_MIN_RELAXED,
    USE_ADAPTIVE_FILTERS, ENTRY_ATR_PERCENTILE, ENTRY_MOMENTUM_PERCENTILE, ENTRY_RSI_EXTREME_PCT,
    REPLAY_MODE,
    MAX_ENTRIES_PER_MIN,
    TOTAL_RISK_BUDGET,
    SYMBOL_BLACKLIST,
)

# Import adaptive filters if available
try:
    from ..adaptive_filters import AdaptiveFilterEngine
    HAS_ADAPTIVE_FILTERS = True
except ImportError:
    HAS_ADAPTIVE_FILTERS = False
    AdaptiveFilterEngine = None


@dataclass
class FilterResult:
    """Result of unified filter evaluation."""
    can_enter: bool
    reason: str
    replacement_symbol: Optional[str] = None  # If can replace existing position


class UnifiedFilter:
    """
    Unified entry filter system.
    
    Consolidates:
    - Adaptive filters (percentile-based)
    - Position manager filters (limits, risk)
    - Signal generator filters (score thresholds)
    - Data-driven filters (ATR, momentum, RSI)
    """
    
    def __init__(self):
        """Initialize unified filter system."""
        # Adaptive filter engine (if available)
        self.adaptive_engine = None
        if HAS_ADAPTIVE_FILTERS and USE_ADAPTIVE_FILTERS:
            try:
                self.adaptive_engine = AdaptiveFilterEngine()
            except Exception:
                pass
        
        # Entry rate limiting
        self._entry_times = []
        self._max_entries_per_min = MAX_ENTRIES_PER_MIN
    
    def add_market_observation(
        self,
        symbol: str,
        atr_pct: float,
        momentum_1h: float,
        rsi: float,
        volume_ratio: float = 1.0
    ):
        """Add market observation for adaptive filtering."""
        if self.adaptive_engine:
            try:
                self.adaptive_engine.add_observation(
                    symbol=symbol,
                    atr_pct=atr_pct,
                    momentum_1h=momentum_1h,
                    rsi=rsi,
                    volume_ratio=volume_ratio
                )
            except Exception:
                pass
    
    def can_enter_position(
        self,
        symbol: str,
        signal_score: float,
        signal_strength: float,
        spread_bps: float,
        volume_24h: float,
        latency_ms: float,
        current_positions: int,
        open_positions: Dict[str, Dict],
        # Data-driven filter parameters
        rsi: Optional[float] = None,
        atr_pct: Optional[float] = None,
        pct_change_1h: Optional[float] = None,
        side: Optional[str] = None,
        # Additional context
        is_unicorn: bool = False,
        drawdown_pct: Optional[float] = None,
        equity: Optional[float] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        position_size: Optional[float] = None,
        cached_now: Optional[float] = None,
    ) -> FilterResult:
        """
        Unified entry filter - single source of truth for all entry decisions.
        
        Returns:
            FilterResult with can_enter, reason, and optional replacement_symbol
        """
        now = cached_now if cached_now is not None else time.time()
        
        # ============================================================
        # 1. SYMBOL BLACKLIST CHECK (Data-Driven Exclusion)
        # ============================================================
        # Extract base symbol (e.g., "BTC" from "BTC/USDT" or "BTCUSDT")
        base_symbol = symbol.replace("/USDT", "").replace("/USDT:USDT", "").replace("USDT", "").replace(":USDT", "").upper()
        
        if base_symbol in SYMBOL_BLACKLIST:
            return FilterResult(
                can_enter=False,
                reason=f"blacklist:{base_symbol}"
            )
        
        # ============================================================
        # 2. HARD SCORE GATE - DISABLED
        # ============================================================
        # NOTE: bot.py's adaptive threshold system (market_speed_adapter) handles
        # all score filtering dynamically (45-70 range based on market conditions).
        # This static check is redundant and interferes with adaptive logic.
        # Removed to allow adaptive system to work properly.
        
        # ============================================================
        # 3. SIGNAL SCORE THRESHOLD
        # ============================================================
        # NOTE: bot.py already filters signals using adaptive threshold
        # This check is redundant and should only apply HARD_MIN_SCORE
        # Commenting out to allow adaptive system to work properly
        # min_score = thresholds['min_score']
        # if signal_score < min_score:
        #     return FilterResult(
        #         can_enter=False,
        #         reason=f"score_threshold:{signal_score:.1f}<{min_score}"
        #     )
        
        # ============================================================
        # 4. POSITION LIMITS
        # ============================================================
        if current_positions >= MAX_OPEN_POSITIONS:
            # Check if we can replace a weaker position
            replacement_symbol = self._find_replacement_candidate(
                symbol, signal_score, open_positions
            )
            if replacement_symbol:
                return FilterResult(
                    can_enter=True,
                    reason="can_replace_weaker",
                    replacement_symbol=replacement_symbol
                )
            return FilterResult(
                can_enter=False,
                reason=f"max_positions:{current_positions}>={MAX_OPEN_POSITIONS}"
            )
        
        # ============================================================
        # 5. RATE LIMITING
        # ============================================================
        # Clean old entries (older than 1 minute)
        self._entry_times = [t for t in self._entry_times if (now - t) < 60.0]
        
        if len(self._entry_times) >= self._max_entries_per_min:
            return FilterResult(
                can_enter=False,
                reason=f"rate_limit:{len(self._entry_times)}>={self._max_entries_per_min}/min"
            )
        
        # ============================================================
        # 6. BASIC MARKET CONDITIONS
        # ============================================================
        if spread_bps < MIN_SPREAD_BPS:
            return FilterResult(
                can_enter=False,
                reason=f"spread_too_tight:{spread_bps:.1f}<{MIN_SPREAD_BPS}"
            )
        
        if spread_bps > MAX_SPREAD_BPS:
            return FilterResult(
                can_enter=False,
                reason=f"spread_too_wide:{spread_bps:.1f}>{MAX_SPREAD_BPS}"
            )
        
        if volume_24h < MIN_VOLUME_24H:
            return FilterResult(
                can_enter=False,
                reason=f"volume_too_low:{volume_24h/1e6:.1f}M<{MIN_VOLUME_24H/1e6:.1f}M"
            )
        
        if latency_ms > MAX_LATENCY_MS:
            return FilterResult(
                can_enter=False,
                reason=f"latency_too_high:{latency_ms:.1f}>{MAX_LATENCY_MS}"
            )
        
        # ============================================================
        # 7. DATA-DRIVEN FILTERS (if enabled and data available)
        # ============================================================
        if ENTRY_FILTER_ENABLED:
            # Use relaxed thresholds if enabled
            if ENTRY_FILTER_RELAXED:
                atr_threshold = ENTRY_MIN_ATR_PCT_RELAXED
                momentum_threshold = ENTRY_MIN_MOMENTUM_PCT_RELAXED
                rsi_long_max = ENTRY_RSI_LONG_MAX_RELAXED
                rsi_short_min = ENTRY_RSI_SHORT_MIN_RELAXED
            else:
                atr_threshold = ENTRY_MIN_ATR_PCT
                momentum_threshold = ENTRY_MIN_MOMENTUM_PCT
                rsi_long_max = ENTRY_RSI_LONG_MAX
                rsi_short_min = ENTRY_RSI_SHORT_MIN
            
            # ATR filter
            if atr_pct is not None and atr_pct < atr_threshold:
                return FilterResult(
                    can_enter=False,
                    reason=f"atr_too_low:{atr_pct:.2f}%<{atr_threshold:.2f}%"
                )
            
            # Momentum filter
            if pct_change_1h is not None and abs(pct_change_1h) < momentum_threshold:
                return FilterResult(
                    can_enter=False,
                    reason=f"momentum_too_low:{abs(pct_change_1h):.2f}%<{momentum_threshold:.2f}%"
                )
            
            # RSI alignment filter
            if rsi is not None and side:
                if side.lower() == 'long' and rsi > rsi_long_max:
                    return FilterResult(
                        can_enter=False,
                        reason=f"rsi_not_extreme_long:{rsi:.1f}>{rsi_long_max}"
                    )
                elif side.lower() == 'short' and rsi < rsi_short_min:
                    return FilterResult(
                        can_enter=False,
                        reason=f"rsi_not_extreme_short:{rsi:.1f}<{rsi_short_min}"
                    )
        
        # ============================================================
        # 8. ADAPTIVE FILTERS (if enabled)
        # ============================================================
        if USE_ADAPTIVE_FILTERS and self.adaptive_engine:
            try:
                thresholds = self.adaptive_engine.get_thresholds()
                if thresholds:
                    # Check ATR percentile
                    if atr_pct is not None and atr_pct < thresholds.atr_threshold:
                        return FilterResult(
                            can_enter=False,
                            reason=f"adaptive_atr:{atr_pct:.2f}%<{thresholds.atr_threshold:.2f}%"
                        )
                    
                    # Check momentum percentile
                    if pct_change_1h is not None and abs(pct_change_1h) < thresholds.momentum_threshold:
                        return FilterResult(
                            can_enter=False,
                            reason=f"adaptive_momentum:{abs(pct_change_1h):.2f}%<{thresholds.momentum_threshold:.2f}%"
                        )
            except Exception:
                pass  # Fall back to fixed thresholds if adaptive fails
        
        # ============================================================
        # 9. RISK BUDGET CHECK
        # ============================================================
        if TOTAL_RISK_BUDGET > 0 and position_size and equity:
            # Calculate risk for this position
            if entry_price and stop_loss:
                risk_per_unit = abs(entry_price - stop_loss)
                position_risk = (position_size * risk_per_unit) / equity * 100.0
                
                # Calculate current risk usage
                current_risk = sum(
                    (pos.get('size', 0) * abs(pos.get('entry_price', 0) - pos.get('stop_loss', 0))) / equity * 100.0
                    for pos in open_positions.values()
                    if pos.get('entry_price') and pos.get('stop_loss')
                )
                
                if current_risk + position_risk > TOTAL_RISK_BUDGET:
                    return FilterResult(
                        can_enter=False,
                        reason=f"risk_budget_exceeded:{current_risk + position_risk:.1f}%>{TOTAL_RISK_BUDGET:.1f}%"
                    )
        
        # ============================================================
        # ALL CHECKS PASSED
        # ============================================================
        return FilterResult(
            can_enter=True,
            reason="approved"
        )
    
    def record_entry(self, timestamp: Optional[float] = None):
        """Record an entry for rate limiting."""
        now = timestamp if timestamp is not None else time.time()
        self._entry_times.append(now)
        # Keep only last minute
        self._entry_times = [t for t in self._entry_times if (now - t) < 60.0]
    
    def _get_thresholds(self) -> Dict[str, float]:
        """Get signal thresholds based on mode - ALWAYS reads fresh from config."""
        # Import config module fresh each time to avoid caching issues
        import importlib
        import sys
        
        # Force reload config module to get latest values
        if 'TradingBot.app.config' in sys.modules:
            config_module = sys.modules['TradingBot.app.config']
            importlib.reload(config_module)
        else:
            from .. import config as config_module
        
        if config_module.REPLAY_MODE:
            return {
                'min_strength': config_module.REPLAY_MIN_SIGNAL_STRENGTH,
                'min_score': config_module.REPLAY_MIN_SIGNAL_SCORE,
                'hard_min_score': config_module.REPLAY_HARD_MIN_SCORE
            }
        else:
            return {
                'min_strength': config_module.MIN_SIGNAL_STRENGTH,
                'min_score': config_module.MIN_SIGNAL_SCORE,
                'hard_min_score': config_module.HARD_MIN_SCORE
            }
    
    def _find_replacement_candidate(
        self,
        new_symbol: str,
        new_score: float,
        open_positions: Dict[str, Dict]
    ) -> Optional[str]:
        """
        Find a weaker position that can be replaced.
        
        Returns:
            Symbol to replace, or None if no replacement possible
        """
        if not open_positions:
            return None
        
        # Find weakest position (lowest score)
        weakest_symbol = None
        weakest_score = float('inf')
        
        for symbol, position in open_positions.items():
            if symbol == new_symbol:
                continue  # Don't replace with itself
            
            pos_score = position.get('signal_score') or position.get('score') or 0.0
            if pos_score < weakest_score:
                weakest_score = pos_score
                weakest_symbol = symbol
        
        # Only replace if new signal is significantly better
        if weakest_symbol and new_score > weakest_score + 5.0:  # At least 5 points better
            return weakest_symbol
        
        return None

