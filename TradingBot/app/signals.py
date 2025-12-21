"""
Signal Generator - Multi-factor signal generation with real-world trading criteria.
"""

import time
import math
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field

from .config import (
    MIN_SIGNAL_STRENGTH, MIN_SIGNAL_SCORE, HARD_MIN_SCORE, 
    REPLAY_MIN_SIGNAL_STRENGTH, REPLAY_MIN_SIGNAL_SCORE, REPLAY_HARD_MIN_SCORE,
    SIGNAL_PERCENTILE_THRESHOLD, SIGNAL_HISTORY_SIZE,
    DYNAMIC_THRESHOLDS_ENABLED, THRESHOLD_ADJUSTMENT_WINDOW, THRESHOLD_ADJUSTMENT_STEP,
    WIN_RATE_RELAX_THRESHOLD, WIN_RATE_TIGHTEN_THRESHOLD, MIN_SCORE_RANGE, MIN_STRENGTH_RANGE,
    DRY_RUN, REPLAY_MODE, USE_SIGNAL_PERCENTILE_FILTER,
    MVP_SCORING_MODE, SCORING_ROLLBACK, MVP_BANDIT_ENABLED,
    # Data-driven filters
    MIN_ATR_PCT,
)
# LONG BIAS: Import bonus for MAJOR coin longs
try:
    from .config import LONG_MAJOR_SCORE_BONUS
except ImportError:
    LONG_MAJOR_SCORE_BONUS = 0.0  # Default: no bonus

# ML SCORING: Replaced FreshnessScorer (r=0.0044 correlation) with trained LightGBM model
# Trained on 4M+ trades, achieves AUC 0.5246 with +6% lift in top decile win rate
from .ml_scorer import MLScorer
# Keep FreshnessScorer as fallback import (MLScorer has internal fallback)
from .freshness_scorer import FreshnessScorer  # Legacy - used as fallback in MLScorer


@dataclass(slots=True)
class TradingSignal:
    """Trading signal with all relevant information.
    OPTIMIZATION: Using __slots__ for 30-40% memory reduction and faster attribute access.
    """
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    strength: float  # 0-1 (backward compatibility, derived from final_score)
    signal_type: str  # 'momentum', 'mean_reversion', 'trend', 'breakout'
    reason: str
    timestamp: float = None
    signal_score: Optional[Any] = None  # Legacy field (not used)
    final_score: float = 0.0  # 0-100 scale (Freshness-based score)
    score_v2: Optional[float] = None  # Legacy field (not used)
    score_components_raw: Optional[Dict] = None  # Freshness score components
    score_components_capped: Optional[Dict] = None  # Freshness score components

    # MVP scoring (shadow/live behind rollback switch)
    mvp_mode: Optional[str] = None
    mvp_score: Optional[float] = None
    mvp_components: Optional[Dict] = None
    mvp_arm: Optional[str] = None
    mvp_effective_min_score: Optional[float] = None
    mvp_effective_min_strength: Optional[float] = None
    mvp_would_enter: Optional[bool] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        # Backward compatibility: if final_score set but strength not, normalize
        if self.final_score > 0 and self.strength == 0:
            self.strength = self.final_score / 100.0


class SignalGenerator:
    """Generates trading signals using multiple factors."""
    
    def __init__(self):
        self.signal_history = []
        # ML SCORING: Trained LightGBM model replaces FreshnessScorer
        # Model trained on 4M+ trades with ~6% win rate lift in top decile
        self.ml_scorer = MLScorer()
        # Keep reference as freshness_scorer for backward compatibility
        self.freshness_scorer = self.ml_scorer
        # OPTIMIZATION: Cache percentile threshold to avoid recalculating
        self._cached_threshold = None
        self._cached_history_size = 0
        self._cached_percentile = SIGNAL_PERCENTILE_THRESHOLD
        # Dynamic threshold adjustment with caching
        self._cached_dynamic_score = None
        self._cached_dynamic_strength = None
        self._cached_dynamic_percentile = None
        self._last_threshold_update = 0.0
        self._threshold_cache_ttl = 2.0  # Cache for 2 seconds
        self._cached_trades_hash = None  # Hash of recent trades for cache invalidation
        self._cached_regime = None
        self._cached_btc_trend = None
        
        # MARKSMAN removed - no longer used
        # MVP scoring engine (lazy)
        self._mvp_engine = None
    
    @staticmethod
    def _get_thresholds():
        """
        Get signal thresholds based on current mode (LIVE vs REPLAY).
        CRITICAL: Prevents REPLAY settings from affecting LIVE trading.
        NOTE: Must read from config module directly to pick up runtime overrides (e.g., from GO_LIVE.py)
        """
        from . import config as cfg  # Dynamic import to get current values
        if cfg.REPLAY_MODE:
            return {
                'min_strength': cfg.REPLAY_MIN_SIGNAL_STRENGTH,
                'min_score': cfg.REPLAY_MIN_SIGNAL_SCORE,
                'hard_min_score': cfg.REPLAY_HARD_MIN_SCORE
            }
        else:
            return {
                'min_strength': cfg.MIN_SIGNAL_STRENGTH,
                'min_score': cfg.MIN_SIGNAL_SCORE,
                'hard_min_score': cfg.HARD_MIN_SCORE
            }
    
    def _calculate_dynamic_thresholds(
        self,
        recent_trades: list,
        volatility_regime: str = "Normal",
        btc_trend: float = 0.0
    ) -> Tuple[int, float, float]:
        """
        Calculate dynamic thresholds based on recent performance and market conditions.
        OPTIMIZED: Caches results with TTL to avoid redundant calculations.
        
        Returns:
            (adjusted_min_score, adjusted_min_strength, adjusted_percentile)
        """
        # OPTIMIZATION: Check cache first
        now = time.time()
        # OPTIMIZATION: Create cache key from inputs (only hash if we have enough trades)
        if recent_trades and len(recent_trades) >= THRESHOLD_ADJUSTMENT_WINDOW:
            # Only hash the relevant window to avoid hashing entire list
            start_idx = len(recent_trades) - THRESHOLD_ADJUSTMENT_WINDOW
            # OPTIMIZATION: Use a simpler hash - just count wins and use last timestamp
            wins_count = sum(1 for i in range(start_idx, len(recent_trades)) if recent_trades[i].get('was_win', False))
            last_timestamp = recent_trades[-1].get('timestamp', 0) if recent_trades else 0
            trades_hash = hash((wins_count, last_timestamp))
        else:
            trades_hash = 0
        cache_key = (trades_hash, volatility_regime, round(btc_trend, 2))
        
        # Check if cache is valid
        if (self._cached_dynamic_score is not None and 
            (now - self._last_threshold_update) < self._threshold_cache_ttl and
            self._cached_trades_hash == trades_hash and
            self._cached_regime == volatility_regime and
            abs(self._cached_btc_trend - btc_trend) < 0.01):
            return self._cached_dynamic_score, self._cached_dynamic_strength, self._cached_dynamic_percentile
        
        # Start with base thresholds (mode-aware)
        thresholds = self._get_thresholds()
        base_score = thresholds['min_score']
        base_strength = thresholds['min_strength']
        base_percentile = SIGNAL_PERCENTILE_THRESHOLD
        
        # LIVE/DRY_RUN MODE: Force percentile to 0.0 if threshold is disabled
        # This ensures percentile gating is completely disabled for LIVE/DRY_RUN (not REPLAY)
        if SIGNAL_PERCENTILE_THRESHOLD <= 0.0:
            base_percentile = 0.0
        
        if not DYNAMIC_THRESHOLDS_ENABLED:
            # Cache the result
            self._cached_dynamic_score = base_score
            self._cached_dynamic_strength = base_strength
            self._cached_dynamic_percentile = base_percentile
            self._last_threshold_update = now
            self._cached_trades_hash = trades_hash
            self._cached_regime = volatility_regime
            self._cached_btc_trend = btc_trend
            return base_score, base_strength, base_percentile
        
        # OPTIMIZATION: Calculate recent win rate more efficiently
        # Avoid list slicing by using negative indexing directly
        win_rate = 50.0  # Default
        if len(recent_trades) >= THRESHOLD_ADJUSTMENT_WINDOW:
            # OPTIMIZATION: Count wins without creating a new list slice
            # Use iterator to avoid list creation
            wins = 0
            start_idx = len(recent_trades) - THRESHOLD_ADJUSTMENT_WINDOW
            for i in range(start_idx, len(recent_trades)):
                if recent_trades[i].get('was_win', False):
                    wins += 1
            win_rate = (wins / THRESHOLD_ADJUSTMENT_WINDOW) * 100.0
            
            # Adjust based on win rate
            if win_rate > WIN_RATE_RELAX_THRESHOLD:
                # Performing well - relax thresholds (allow more trades)
                base_score -= THRESHOLD_ADJUSTMENT_STEP
                base_strength -= (THRESHOLD_ADJUSTMENT_STEP / 100.0)
                base_percentile += 0.02  # Increase percentile (toward top 50%)
            elif win_rate < WIN_RATE_TIGHTEN_THRESHOLD:
                # Underperforming - tighten thresholds (be more selective)
                base_score += THRESHOLD_ADJUSTMENT_STEP
                base_strength += (THRESHOLD_ADJUSTMENT_STEP / 100.0)
                base_percentile -= 0.02  # Decrease percentile (away from top 50%)
        
        # Regime-aware adjustments
        # In quiet/calm markets: tighten (be more selective)
        # In volatile/trending markets: loosen (more opportunities)
        if volatility_regime in ["Low", "Quiet"]:
            base_score += 1  # Slightly tighter
            base_strength += 0.01
        elif volatility_regime in ["High", "Volatile"] and abs(btc_trend) > 1.0:
            # High volatility + strong trend = more opportunities
            base_score -= 1  # Slightly looser
            base_strength -= 0.01
            base_percentile += 0.02  # Allow more signals (toward top 50%)
        
        # Enforce hard limits
        min_score, max_score = MIN_SCORE_RANGE
        min_strength, max_strength = MIN_STRENGTH_RANGE
        base_score = max(min_score, min(max_score, base_score))
        base_strength = max(min_strength, min(max_strength, base_strength))
        # LIVE/DRY_RUN MODE: If percentile threshold is disabled (0.0), keep it at 0.0
        # Otherwise, keep percentile reasonable (top 50% max) for REPLAY mode
        if SIGNAL_PERCENTILE_THRESHOLD <= 0.0:
            base_percentile = 0.0  # Force to 0.0 for LIVE/DRY_RUN
        else:
            base_percentile = max(0.10, min(0.50, base_percentile))  # Keep percentile reasonable (top 50% max) for REPLAY
        
        # OPTIMIZATION: Cache the result
        result = (int(base_score), base_strength, base_percentile)
        self._cached_dynamic_score = result[0]
        self._cached_dynamic_strength = result[1]
        self._cached_dynamic_percentile = result[2]
        self._last_threshold_update = now
        self._cached_trades_hash = trades_hash
        self._cached_regime = volatility_regime
        self._cached_btc_trend = btc_trend
        
        return result
    
    def _is_in_top_percentile(self, signal_score: float, percentile_threshold: float = None) -> bool:
        """
        Check if signal score is in top percentile of recent signals.
        OPTIMIZED: Caches percentile threshold calculation.
        
        Args:
            signal_score: Final score of the signal (0-100)
            percentile_threshold: Optional percentile threshold (0-1), uses dynamic if None
        
        Returns:
            True if signal is in top percentile, False otherwise
            If history is too small (< 20 signals), uses relaxed fallback (75+) instead of 80+
        """
        # Use provided percentile threshold or default
        pct_threshold = percentile_threshold if percentile_threshold is not None else SIGNAL_PERCENTILE_THRESHOLD
        
        # Need at least 20 signals for reliable percentile calculation
        # When history is insufficient, use MIN_SIGNAL_SCORE as fallback
        history_size = len(self.signal_history)
        if history_size < 20:
            from .config import MIN_SIGNAL_SCORE
            return signal_score >= MIN_SIGNAL_SCORE  # Use configured minimum score
        
        # OPTIMIZATION: Only recalculate threshold if history changed or percentile threshold changed
        cache_key = (history_size, pct_threshold)
        if self._cached_threshold is None or self._cached_history_size != history_size or \
           abs(self._cached_percentile - pct_threshold) > 0.001:
            # Extract final scores from recent history (limit to SIGNAL_HISTORY_SIZE)
            # CHERRY PICKING: Only consider signals that passed HARD_MIN_SCORE for percentile calculation
            # This ensures percentile filter works on the pool of viable signals (40+), making it truly selective
            from . import config as cfg
            hard_min = getattr(cfg, 'HARD_MIN_SCORE', 40)
            recent_scores = []
            for signal_record in self.signal_history[-SIGNAL_HISTORY_SIZE:]:
                final_score = signal_record.get('final_score', 0.0)
                # Only include signals that would have passed HARD_MIN_SCORE (viable pool for percentile filtering)
                if final_score >= hard_min:
                    recent_scores.append(final_score)
            
            if len(recent_scores) < 20:
                # Use MIN_SIGNAL_SCORE as fallback when history is insufficient
                from .config import MIN_SIGNAL_SCORE
                self._cached_threshold = float(MIN_SIGNAL_SCORE)
                self._cached_history_size = history_size
                self._cached_percentile = pct_threshold
                return signal_score >= MIN_SIGNAL_SCORE  # Use configured minimum score
            
            # Sort scores descending
            recent_scores_sorted = sorted(recent_scores, reverse=True)
            
            # Calculate percentile threshold
            # pct_threshold = 0.90 means "90th percentile" = top 10% of signals
            top_percent = 1.0 - pct_threshold  # Convert percentile to "top X%"
            target_count = max(1, int(len(recent_scores_sorted) * top_percent))  # How many signals we want (top 10%)
            
            # CRITICAL FIX: Use a more aggressive percentile to combat score clustering
            # If many signals cluster at the same score, using the exact percentile threshold
            # causes "free pass" - all signals at that score pass, not just top 10%
            # Solution: Use a tighter percentile (e.g., 0.85 for top 15%) then take top 10% of those
            # OR: Use the score that's strictly better than the threshold percentile score
            threshold_index = target_count - 1  # 0-indexed: 10th signal is index 9
            threshold_index = max(0, min(threshold_index, len(recent_scores_sorted) - 1))
            
            threshold_score = recent_scores_sorted[threshold_index]
            
            # If there are many signals at threshold_score, we'll accept too many
            # Count how many signals are strictly ABOVE threshold (these should definitely pass)
            signals_above = sum(1 for s in recent_scores_sorted if s > threshold_score)
            
            if signals_above >= target_count:
                # We have enough signals strictly above threshold - use strict comparison
                # This ensures we only accept signals better than threshold, preventing tie inflation
                self._cached_threshold = threshold_score + 0.0001  # Tiny epsilon for > comparison
                self._cached_history_size = history_size
                self._cached_percentile = pct_threshold
                return signal_score > threshold_score  # Strict: only accept strictly better signals
            else:
                # Not enough signals above threshold - need to accept some at threshold
                # Store threshold for >= comparison
                self._cached_threshold = threshold_score
                self._cached_history_size = history_size
                self._cached_percentile = pct_threshold
                return signal_score >= threshold_score  # Standard: accept at or above
        
        # Use cached threshold
        return signal_score >= self._cached_threshold
    
    def generate_signal(
        self,
        symbol: str,
        symbol_stats: Dict[str, Any],
        entry_price: Optional[float] = None,
        price_data: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, Any]] = None,
        orderbook: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
        order_size_usd: float = 0.0,
        position_manager_state: Optional[Dict[str, Any]] = None,
        btc_trend: Optional[float] = None,
        regime_config: Optional[Any] = None,
        recent_trades: Optional[list] = None,
        volatility_regime: str = "Normal",
        bot_positions: Optional[Dict] = None,  # SCORING V2: For portfolio scoring
        market_regime: str = "neutral",  # New parameter for Marksman regime filter
        min_score_override: Optional[float] = None,  # Dynamic override
        min_strength_override: Optional[float] = None  # Dynamic override
    ) -> Tuple[Optional[TradingSignal], Optional[str]]:
        """
        Generate trading signal for symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            symbol_stats: Symbol statistics from universe
            price_data: Optional OHLCV price data
            indicators: Optional technical indicators
            orderbook: Optional orderbook data
            latency_ms: Current latency in milliseconds
            order_size_usd: Order size in USD
            position_manager_state: Current position manager state
            btc_trend: BTC trend percentage
            regime_config: Optional regime configuration
            recent_trades: Optional list of recent trades
            volatility_regime: Current volatility regime
            min_score_override: Dynamic score threshold override
            min_strength_override: Dynamic strength threshold override
        
        Returns:
            TradingSignal if valid signal found, None otherwise
        """
        # Basic validation
        if not symbol_stats:
            return None, f"No symbol_stats for {symbol}"
        
        bid = symbol_stats.get('bid', 0)
        ask = symbol_stats.get('ask', 0)
        last = symbol_stats.get('last', 0)
        spread_bps = symbol_stats.get('spread_bps', 9999)
        volume_24h = symbol_stats.get('vol_quote', 0)
        
        if not (bid > 0 and ask > 0 and last > 0):
            # Detailed error logging for debugging
            return None, f"Invalid price data for {symbol}: bid={bid:.8f}, ask={ask:.8f}, last={last:.8f}"
        
        # ============================================================
        # DATA-DRIVEN FILTER: MINIMUM ATR (Skip Low Volatility)
        # ============================================================
        # Analysis of 4M+ trades shows 56.5% SL rate when ATR < 0.5% vs 30.2% when ATR > 2%
        # Low volatility = SL too tight relative to noise = more stop-outs
        if indicators and MIN_ATR_PCT > 0:
            atr_pct = indicators.get('atr_pct', 0)
            if atr_pct > 0 and atr_pct < MIN_ATR_PCT:
                self._filter_stats['rejected_low_atr'] = self._filter_stats.get('rejected_low_atr', 0) + 1
                return None, f"LOW_ATR:atr={atr_pct*100:.2f}%<{MIN_ATR_PCT*100:.1f}%"
        # ============================================================
        
        # ENTRY PRICE:
        # - If caller provides an explicit entry_price, trust it (used by some diagnostics/tests).
        # - Otherwise compute from mid-price.
        if entry_price is not None:
            try:
                ep = float(entry_price)
                if ep > 0:
                    entry_price = ep
                else:
                    entry_price = None
            except Exception:
                entry_price = None

        if entry_price is None:
            entry_price = (bid + ask) / 2  # Use mid price
        
        # Try multiple signal types
        signals = []
        
        # MARKSMAN removed - process all symbols using evolution-optimized parameters
        import os
        is_replay = os.environ.get('REPLAY_MODE', '0') == '1'
        
        # Generate momentum and mean reversion signals
        # 1. Momentum signal
        momentum_signal = self._generate_momentum_signal(
            symbol, entry_price, symbol_stats, price_data, indicators, regime_config
        )
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 2. Mean reversion signal (RSI-based)
        mean_reversion_signal = self._generate_mean_reversion_signal(
            symbol, entry_price, symbol_stats, price_data, indicators
        )
        if mean_reversion_signal:
            signals.append(mean_reversion_signal)
        
        # Select best signal
        if not signals:
            return None, "No signals generated (no momentum/mean_reversion signals)"
        
        # Sort by strength and take the best
        best_signal = max(signals, key=lambda s: s.strength)
        
        # MARKET BIAS: DISABLED (Holy Grail config - backtested without these filters)
        # These filters were causing trade blocking. Backtested results achieved
        # without them, so keeping disabled to match expected performance.
        # Re-enable if you want more conservative trading.
        #
        # if btc_trend is not None and btc_trend < -1.0:  # BTC down >1%
        #     if best_signal.side.lower() == "long":
        #         return None, f"BTC bearish ({btc_trend:.2f}%) - LONG signals filtered"
        # 
        # if best_signal.side.lower() == "short":
        #     rsi = indicators.get('rsi') if indicators else None
        #     if rsi is not None and rsi < 40:
        #         return None, f"RJ – Short on oversold (RSI {rsi:.1f} < 40)"
        #     pct_change = symbol_stats.get('pct_change_24h', 0)
        #     if pct_change < -10.0:
        #         return None, f"RJ – Short extension (Dumped {pct_change:.1f}%)"
        pass  # Filters intentionally disabled for Holy Grail strategy

        # ============================================================================
        # FRESHNESS-BASED SCORING SYSTEM
        # ============================================================================
        # Philosophy: "The best trades are fresh momentum moves with confirmation"
        # - Freshness (40 pts): Bell curve peaks at 5%, drops after 7% (exhaustion)
        # - Confirmation (40 pts): Volume, orderbook, acceleration, volatility
        # - Execution (20 pts): Spread, depth, latency, symbol reliability
        # ============================================================================
        
        # Sanitize inputs: Ensure they are floats (handle None explicitly)
        # Some exchanges/pairs might return None for these fields
        raw_pct = symbol_stats.get('pct_change_24h')
        pct_change_24h = float(raw_pct) if raw_pct is not None else 0.0
        
        raw_vol = symbol_stats.get('vol_quote')
        volume_24h = float(raw_vol) if raw_vol is not None else 0.0
        
        raw_spread = symbol_stats.get('spread_bps')
        spread_bps = float(raw_spread) if raw_spread is not None else 0.0
        
        # ================================================================
        # ML SCORING - XGBoost is now the PRIMARY scorer
        # ================================================================
        from .ml_scorer import get_ml_scorer
        ml_scorer = get_ml_scorer()
        
        final_score, score_components = ml_scorer.score_signal(
            symbol=symbol,
            pct_change_24h=pct_change_24h,
            volume_24h=volume_24h,
            spread_bps=spread_bps,
            orderbook=orderbook,
            indicators=indicators,
            latency_ms=latency_ms,
        )
        
        # Store ML info for debugging
        best_signal.mvp_mode = "ml_xgb"
        best_signal.mvp_score = final_score
        best_signal.mvp_components = score_components
        
        # Store score in signal
        best_signal.final_score = final_score
        best_signal.signal_score = None  # Not using old SignalScore object
        best_signal.score_v2 = None
        best_signal.score_components_raw = score_components
        best_signal.score_components_capped = score_components  # Same for freshness system
        
        # LONG BIAS: Apply bonus for LONG signals on MAJOR coins
        # This effectively lowers the threshold from 65 to 60 for quality longs
        if LONG_MAJOR_SCORE_BONUS > 0:
            MAJOR_COINS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC']
            symbol_base = symbol.split('/')[0].upper()
            is_major = any(m in symbol_base for m in MAJOR_COINS)
            is_long = best_signal.side.lower() == 'long'
            if is_major and is_long:
                best_signal.final_score = final_score + LONG_MAJOR_SCORE_BONUS
                score_components['long_major_bonus'] = LONG_MAJOR_SCORE_BONUS
        
        # Update signal strength to match score (for backward compatibility)
        # Map 0-100 score to 0-1 strength
        best_signal.strength = final_score / 100.0
        
        # Calculate dynamic thresholds (always called, but only used if DYNAMIC_THRESHOLDS_ENABLED is True)
        recent_trades_list = recent_trades if recent_trades else []
        btc_trend_value = btc_trend if btc_trend is not None else 0.0
        dynamic_score, dynamic_strength, dynamic_percentile = self._calculate_dynamic_thresholds(
            recent_trades_list, volatility_regime, btc_trend_value
        )
        
        # PURE_SCALPER: static thresholds by default.
        # If DYNAMIC_THRESHOLDS_ENABLED is set, we can re-enable adaptive behavior later.
        # Get mode-aware thresholds
        thresholds = self._get_thresholds()
        
        # DIAGNOSTIC: Track filter rejections
        if not hasattr(self, '_filter_stats'):
            self._filter_stats = {
                'rejected_hard_min': 0,
                'rejected_score': 0,
                'rejected_strength': 0,
                'rejected_percentile': 0,
                'accepted': 0
            }
        
        # Apply explicit overrides if provided (Dynamic Regime Switching)
        if min_score_override is not None:
            min_score = float(min_score_override)
        elif DYNAMIC_THRESHOLDS_ENABLED:
            min_score = dynamic_score
        else:
            min_score = float(thresholds['min_score'])
            

        if min_strength_override is not None:
            min_strength = float(min_strength_override)
        elif DYNAMIC_THRESHOLDS_ENABLED:
            min_strength = dynamic_strength
        else:
            min_strength = thresholds['min_strength']

        # MVP scoring policy observability: compute what the selected profile would do,
        # but do not change trading decisions unless MVP_SCORING_MODE == "live" later.
        if (not SCORING_ROLLBACK) and str(MVP_SCORING_MODE).lower() in ("shadow", "live"):
            try:
                import os
                from .scoring_mvp.profiles import PROFILES
                from .scoring_mvp.engine import apply_profile_thresholds

                # Manual arm override for now (bandit wiring comes next).
                arm_id = str(os.environ.get("MVP_ARM", "balanced")).strip().lower()
                prof = PROFILES.get(arm_id) or PROFILES.get("balanced")
                if prof:
                    eff_min_score, eff_min_strength = apply_profile_thresholds(
                        base_min_score=float(min_score),
                        base_min_strength=float(min_strength),
                        profile_min_score_delta=float(prof.min_score_delta),
                        profile_min_strength_delta=float(prof.min_strength_delta),
                    )
                    best_signal.mvp_arm = best_signal.mvp_arm or prof.arm_id
                    best_signal.mvp_effective_min_score = eff_min_score
                    best_signal.mvp_effective_min_strength = eff_min_strength
                    best_signal.mvp_would_enter = (
                        float(best_signal.final_score) >= float(eff_min_score)
                        and float(best_signal.strength) >= float(eff_min_strength)
                    )
            except Exception:
                pass
            
        # Hard Min Score: must be explicitly below min_score.
        hard_min_cfg = float(thresholds.get('hard_min_score', (float(min_score) - 5.0)))
        hard_min = min(hard_min_cfg, float(min_score) - 1.0)
            
        # Use dynamic percentile if enabled, otherwise use static SIGNAL_PERCENTILE_THRESHOLD if percentile filtering is enabled
        if DYNAMIC_THRESHOLDS_ENABLED:
            percentile_threshold = dynamic_percentile
        elif USE_SIGNAL_PERCENTILE_FILTER:
            percentile_threshold = SIGNAL_PERCENTILE_THRESHOLD  # Use configured percentile threshold
        else:
            percentile_threshold = 0.0  # Disabled
        
        # Use Scoring v2 final_score (or base if v2 failed)
        score_to_check = best_signal.final_score
        
        # ============================================================
        # ML SCORE THRESHOLD CHECK
        # ============================================================
        # XGBoost V4 outputs win probability (0-100)
        # Check against HARD_MIN_SCORE as safety floor here
        # Main threshold (MIN_SIGNAL_SCORE) checked in scanner
        #
        # IMPORTANT: Add to history BEFORE checking hard_min so percentile filter
        # has accurate distribution (includes low-scoring signals that get rejected)
        self._add_to_history(symbol, best_signal.final_score, best_signal.strength)
        
        from . import config as cfg
        hard_min = getattr(cfg, 'HARD_MIN_SCORE', 40)
        if score_to_check < hard_min:
            self._filter_stats['rejected_hard_min'] += 1
            return None, f"ML_floor:{score_to_check:.1f}<{hard_min}"
        # ============================================================
        
        # Check percentile threshold (only accept top X% of recent signals) - use dynamic percentile
        # ADAPTIVE FILTERING: Reject signals that don't meet percentile threshold
        # This automatically adapts - if many signals score 41+, only top X% pass
        # NOTE: Signal already added to history above (before hard_min check)
        if USE_SIGNAL_PERCENTILE_FILTER and percentile_threshold > 0.0 and len(self.signal_history) >= 20:
            if not self._is_in_top_percentile(best_signal.final_score, percentile_threshold):
                # Signal not in top percentile - reject it
                self._filter_stats['rejected_percentile'] += 1
                percentile_pct = percentile_threshold * 100
                return None, f"percentile:{best_signal.final_score:.1f}<{percentile_pct:.0f}th"
        # If USE_SIGNAL_PERCENTILE_FILTER is False or percentile_threshold is 0.0, percentile check is completely bypassed (PURE_SCALPER mode)
        # For initial signals (history < 20), percentile check uses MIN_SIGNAL_SCORE as fallback
        # This ensures early signals still meet minimum quality standards
        
        # ============================================================
        # FINAL SAFETY: Garbage check only (adaptive handles real filtering)
        # ============================================================
        # Removed - already checked above with GARBAGE_THRESHOLD
        # ============================================================
        
        # Signal already added to history above (before hard_min check)
        # Just mark as accepted
        self._filter_stats['accepted'] += 1
        
        # Log signal with score breakdown
        from .logger import get_logger
        logger = get_logger("SignalGenerator")
        score_breakdown = self.freshness_scorer.get_score_breakdown_str(score_components)
        logger.info(
            f"[SIGNAL] {symbol} {best_signal.side.upper()} | "
            f"Score={best_signal.final_score:.1f} ({score_breakdown}) | "
            f"Momentum={pct_change_24h:.2f}% Vol=${volume_24h/1e6:.1f}M"
        )
        
        return best_signal, None  # None means no rejection
    
    def _add_to_history(self, symbol: str, final_score: float, strength: float):
        """Add signal to history for percentile calculation.
        OPTIMIZED: Invalidates cache when history changes."""
        signal_record = {
            'timestamp': time.time(),
            'symbol': symbol,
            'final_score': final_score,
            'strength': strength
        }
        old_size = len(self.signal_history)
        self.signal_history.append(signal_record)
        # Keep only last SIGNAL_HISTORY_SIZE signals
        if len(self.signal_history) > SIGNAL_HISTORY_SIZE:
            self.signal_history = self.signal_history[-SIGNAL_HISTORY_SIZE:]
        
        # OPTIMIZATION: Invalidate cache if history size changed
        if len(self.signal_history) != old_size:
            self._cached_threshold = None
            self._cached_history_size = 0
    
    def _generate_momentum_signal(
        self,
        symbol: str,
        entry_price: float,
        symbol_stats: Dict,
        price_data: Optional[Dict],
        indicators: Optional[Dict],
        regime_config: Optional[Any] = None
    ) -> Optional[TradingSignal]:
        """Generate momentum-based signal."""
        # Get mode-aware thresholds
        thresholds = self._get_thresholds()
        
        # Use regime-specific parameters if available
        if regime_config:
            min_momentum_pct = regime_config.min_momentum_pct
            # Allow config to override regime if lower (for testing/flexibility)
            min_signal_strength = min(regime_config.min_signal_strength, thresholds['min_strength'])
            stop_loss_atr_mult = regime_config.stop_loss_atr_multiplier
            take_profit_atr_mult = regime_config.take_profit_atr_multiplier
        else:
            # Fallback to defaults (scalping) - mode-aware thresholds
            min_momentum_pct = 0.1  # Lower from 0.8% to 0.1% for Binance Futures (allows more signals)
            min_signal_strength = thresholds['min_strength']
            stop_loss_atr_mult = 1.5
            take_profit_atr_mult = 2.5
        
        # Simple momentum based on price change and volume
        pct_change = symbol_stats.get('pct_change_24h', 0)
        volume_24h = symbol_stats.get('vol_quote', 0)
        
        # Need significant momentum (regime-specific threshold)
        if abs(pct_change) < min_momentum_pct:
            return None
        
        # MOMENTUM SCORING: Use arctangent curve for smooth saturation
        # Arctan provides natural compression: sensitive in 0-5% range, saturates at extremes
        # Formula: arctan(x / center) * (2/π) maps to 0-1 range
        # Center point = 3.5% means 3.5% change = ~0.7 strength, 7% = ~0.9, 10%+ = ~1.0
        momentum_abs = abs(pct_change)
        momentum_center = 3.5  # 3.5% change is the inflection point
        momentum_strength = math.atan(momentum_abs / momentum_center) * (2.0 / math.pi)  # Maps to 0-1
        
        # VOLUME SCORING: Use logarithmic scale (volume follows power laws)
        # Similar approach to universe scoring: log10(volume + 1) / max_log
        # max_log = 8.0 means $100M volume = full score (matches universe approach)
        # This means: $1M = 6.0 log = 75%, $10M = 7.0 log = 87.5%, $100M = 8.0 log = 100%
        volume_log = math.log10(volume_24h + 1.0)  # log10 of volume
        volume_max_log = 8.0  # $100M volume = full score (matches universe normalization)
        volume_factor = min(volume_log / volume_max_log, 1.0)  # Normalize to 0-1
        
        # WEIGHTED COMBINATION: Momentum 75%, Volume 25%
        # Momentum is primary factor, volume provides liquidity boost
        strength = (momentum_strength * 0.75) + (volume_factor * 0.25)
        
        if strength < min_signal_strength:
            return None
        
        # Determine direction
        side = "long" if pct_change > 0 else "short"
        
        # Use actual ATR from indicators if available, otherwise estimate
        from .config import SL_ATR_MULTIPLIER
        if indicators and indicators.get('atr_pct'):
            atr_pct = indicators.get('atr_pct')
        else:
            # Estimate ATR from recent volatility (24h change is a rough proxy)
            atr_pct = max(abs(pct_change) / 10.0, 0.01)  # At least 1%
        
        # Calculate SL/TP based on ATR and configurable multiplier
        # SL = 1x ATR (configurable), TP = 3x SL for 3:1 R:R
        stop_loss_pct = atr_pct * SL_ATR_MULTIPLIER
        take_profit_pct = stop_loss_pct * 3.0  # 3:1 risk:reward ratio
        
        if side == "long":
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        return TradingSignal(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strength=strength,
            signal_type="momentum",
            reason=f"Momentum {'breakout' if side == 'long' else 'breakdown'} ({pct_change:.2f}%, vol: ${volume_24h/1e6:.1f}M)"
        )
    
    def _generate_mean_reversion_signal(
        self,
        symbol: str,
        entry_price: float,
        symbol_stats: Dict,
        price_data: Optional[Dict],
        indicators: Optional[Dict]
    ) -> Optional[TradingSignal]:
        """Generate mean reversion signal (RSI-based)."""
        # Skip if no RSI data
        if not indicators or 'rsi' not in indicators:
            return None
        
        rsi = indicators.get('rsi', 50)
        atr_pct = indicators.get('atr_pct', 0.015)  # Use actual ATR if available
        
        from .config import TAKER_FEE_RATE, SLIPPAGE_BPS, SL_ATR_MULTIPLIER
        total_fee_rate = (TAKER_FEE_RATE * 2) + (SLIPPAGE_BPS / 10000)
        
        # Use ATR-based stops (consistent with momentum signals)
        stop_loss_pct = atr_pct * SL_ATR_MULTIPLIER
        take_profit_pct = atr_pct * SL_ATR_MULTIPLIER * 3.0  # 3:1 R:R target
        
        # Oversold (RSI < 30) -> Long
        if rsi < 30:
            # Strength based on how oversold (RSI 0-30 maps to strength 1.0-0.5)
            strength = 0.5 + (30 - rsi) / 60.0  # RSI 0 = 1.0, RSI 30 = 0.5
            
            return TradingSignal(
                symbol=symbol,
                side="long",
                entry_price=entry_price,
                stop_loss=entry_price * (1 - stop_loss_pct),
                take_profit=entry_price * (1 + take_profit_pct),
                strength=min(strength, 0.9),
                signal_type="mean_reversion",
                reason=f"RSI oversold ({rsi:.1f})"
            )
        
        # Overbought (RSI > 70) -> Short
        if rsi > 70:
            # Strength based on how overbought (RSI 70-100 maps to strength 0.5-1.0)
            strength = 0.5 + (rsi - 70) / 60.0  # RSI 70 = 0.5, RSI 100 = 1.0
            
            return TradingSignal(
                symbol=symbol,
                side="short",
                entry_price=entry_price,
                stop_loss=entry_price * (1 + stop_loss_pct),
                take_profit=entry_price * (1 - take_profit_pct),
                strength=min(strength, 0.9),
                signal_type="mean_reversion",
                reason=f"RSI overbought ({rsi:.1f})"
            )
        
        return None
    
    # REMOVED: _generate_trend_signal() method - dead code, never called
    # Trend signals disabled for pure scalper mode

