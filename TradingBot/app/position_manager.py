"""
Position Manager - Handles position sizing, entry validation, and cooldown management.
Implements real-world position-taking strategies with Kelly-adjusted sizing.
"""

import time
import math
import threading
from typing import Optional, Dict, Tuple, Any
from collections import defaultdict

from .config import (
    RISK_PCT, LEVERAGE_BASE, MAX_CONCURRENT_POS, MAX_OPEN_POSITIONS, MAX_CONCURRENT_POS_MIN, MAX_CONCURRENT_POS_MAX,
    MAX_ACCOUNT_RISK_PCT, RISK_PER_TRADE_PCT, MAX_CONCURRENT_POS_HARD,
    MAX_CAPITAL_PER_POS, TOTAL_RISK_BUDGET, MIN_RISK_PER_TRADE, MAX_RISK_PER_TRADE,
    USE_RANK_BASED_ALLOCATION, RPA_MIN_SIZE_MULT, RPA_MAX_SIZE_MULT, RPA_MIN_SIZE_USD, RPA_MAX_RISK_BUDGET_PCT,
    COOLDOWN_SEC, COOLDOWN_SAME_SYMBOL, COOLDOWN_AFTER_EXIT, COOLDOWN_DIFF_SYMBOL,
    MAX_ENTRIES_PER_MIN, MAX_TRADES_PER_DAY, MAX_TRADES_PER_DAY_REPLAY, MIN_SPREAD_BPS, MAX_SPREAD_BPS,
    MIN_VOLUME_24H, MAX_LATENCY_MS, MIN_SIGNAL_STRENGTH, MIN_SIGNAL_SCORE,
    REPLAY_MIN_SIGNAL_STRENGTH, REPLAY_MODE,
    USE_KELLY_SIZING, MIN_POSITION_SIZE, MAX_POSITION_SIZE,
    TAKER_FEE_RATE, SLIPPAGE_BPS,
    MIN_STOP_DISTANCE_PCT, MIN_RISK_USD, KELLY_FRACTION, MAX_KELLY_PCT, WIN_LOSS_RATIO_ASSUMPTION,
    USE_DYNAMIC_LEVERAGE, MIN_LEVERAGE, MAX_LEVERAGE,
    USE_DYNAMIC_POSITION_SIZE, BASE_POSITION_PCT, MAX_POSITION_PCT, MIN_POSITION_PCT,
    MAX_RISK_PER_TRADE_PCT, CORRELATION_PENALTY_PCT, STREAK_PENALTY_PCT,
    CORRELATION_THRESHOLD, STREAK_THRESHOLD, DRY_RUN,
    UNICORN_PROTOCOL_ENABLED, UNICORN_POSITION_SIZE_MULTIPLIER, UNICORN_LEVERAGE_MULTIPLIER,
    UNICORN_BYPASS_COOLDOWN, UNICORN_BYPASS_MAX_POSITIONS, UNICORN_EXTRA_POSITION_SLOTS,
    UNICORN_BYPASS_RATE_LIMIT, UNICORN_BYPASS_LOSS_STREAK, UNICORN_BYPASS_CORRELATION,
    TIERED_LOSS_STREAK_ENABLED, LOSS_STREAK_TIER1, LOSS_STREAK_TIER2,
    LOSS_STREAK_TIER1_RISK_MULTIPLIER, LOSS_STREAK_TIER2_RISK_MULTIPLIER,
    LOSS_STREAK_TIER1_POSITION_MULTIPLIER, LOSS_STREAK_TIER2_POSITION_MULTIPLIER,
    LOSS_STREAK_TIER1_SCORE_BOOST, LOSS_STREAK_TIER2_SCORE_BOOST,
    DD_AWARE_LOSS_STREAK_ENABLED, DD_AWARE_STREAK_THRESHOLD, DD_AWARE_DD_THRESHOLD,
    AUTO_RESET_LOSS_STREAK_ENABLED, AUTO_RESET_TIME_SEC, AUTO_RESET_WINNER_PNL_THRESHOLD,
    AUTO_RESET_DD_IMPROVEMENT_THRESHOLD,
    MAX_LOSS_STREAK_HARD, LOSS_STREAK_CAUTION_LEVEL, LOSS_STREAK_AUTO_RESET_SEC,
    SCORE_AWARE_REPLACEMENT_ENABLED, SCORE_REPLACEMENT_MARGIN,
    LOSS_STREAK_DEFENSE_LEVEL, LOSS_STREAK_HARD_LEVEL, LOSS_STREAK_PAUSE_SEC, LOSS_STREAK_DECAY_SECONDS,
    HIGH_SCORE_BYPASS, SYMBOL_CHURN_COOLDOWN_SEC,
    COST_GATE_ENABLED, COST_GATE_MULTIPLIER, COST_GATE_MIN_EDGE_BPS, COST_GATE_SCORE_TO_EDGE_MULT,
    ENTRY_FEE_RATE,
    # Data-driven filters (from 4M+ trades analysis)
    SYMBOL_BLACKLIST, MIN_ATR_PCT,
    # Data-driven entry filters (from backtest on 4M+ trades)
    ENTRY_FILTER_ENABLED, ENTRY_MIN_ATR_PCT, ENTRY_MIN_MOMENTUM_PCT,
    ENTRY_RSI_LONG_MAX, ENTRY_RSI_SHORT_MIN,
    ENTRY_FILTER_RELAXED, ENTRY_MIN_ATR_PCT_RELAXED, ENTRY_MIN_MOMENTUM_PCT_RELAXED,
    ENTRY_RSI_LONG_MAX_RELAXED, ENTRY_RSI_SHORT_MIN_RELAXED
)


class PositionManager:
    """Manages position entry, sizing, and validation."""
    
    def __init__(self):
        # PERFORMANCE: Pre-bind logger instance
        from .logger import get_logger
        self.logger = get_logger("PositionManager")
        
        # UNIFIED FILTER: Single source of truth for entry filtering
        from .filters import UnifiedFilter
        self.unified_filter = UnifiedFilter()
        
        self.positions = {}
        self.cooldown_until = {}
        self.entry_times = []  # Track entry times for rate limiting
        self.daily_trades = []  # Track all trades today for daily limit
        self.last_daily_reset = time.time()  # Track when daily counter was last reset
        self.last_entry_by_symbol = {}  # Track last entry per symbol
        self.last_exit_time = 0.0  # Track last exit time for exit→entry cooldown
        self.loss_streak = 0  # Track consecutive losses
        self.last_trade_was_win = None  # Track last trade result
        self.last_loss_time = 0.0  # Track time of last loss (for auto-reset)
        self.last_dd_check = 0.0  # Track last drawdown check (for DD-aware protection)
        self.dynamic_max_positions = MAX_CONCURRENT_POS  # Dynamic limit adjusted by LLM
        self.last_limit_adjustment = 0.0  # Track when limit was last adjusted
        # Loss-streak state machine
        self.loss_streak_state = "normal"  # normal|caution|defense|pause
        self.loss_streak_unlock_ts = 0.0
        # Symbol churn tracking: track last micro time_exit per symbol
        self._symbol_churn = {}  # key: symbol, value: {"ts": float, "profit_atr": float}
        
        # CRITICAL: Lock for atomic position limit checks (prevents race conditions)
        self._position_limit_lock = threading.Lock()

        # ML-SYMBOL ADAPTATION: Intelligent symbol-specific trading
        # Instead of blacklisting, adapt strategy per symbol using ML-learned parameters
        self._symbol_adaptation_manager = SymbolAdaptationManager()


        # ML-SYMBOL ADAPTATION: Cache for symbol performance data
        self._symbol_performance_cache = {}
        self._last_symbol_cache_update = 0
    
    def calculate_position_size(
        self,
        symbol: str,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        signal_strength: float = 1.0,
        side: str = "long",
        is_unicorn: bool = False,
        open_positions: Dict[str, Dict] = None,
        leverage: int = None,
        signal_score: float = None
    ) -> Tuple[float, float, Optional[str]]:
        """
        Calculate position size using risk budget approach.

        Args:
            symbol: Trading symbol
            equity: Current account equity
            entry_price: Entry price
            stop_loss_price: Stop loss price
            signal_strength: Signal strength (0-1)
            side: 'long' or 'short'
            is_unicorn: Whether this is a unicorn signal
            open_positions: Dictionary of open positions (for risk calculation)

        Returns:
            (position_size, risk_fraction, reason)
            - position_size: Position size in base currency
            - risk_fraction: Risk as fraction of equity (0.0 to 1.0)
            - reason: None if OK, or string explaining rejection
        """
        # Validate inputs
        if entry_price <= 0 or stop_loss_price <= 0 or equity <= 0:
            return 0.0, 0.0, "invalid_inputs"
        
        # Calculate stop distance
        if side == "long":
            if entry_price <= stop_loss_price:
                return 0.0, 0.0, "invalid_stop_loss"
            price_risk = entry_price - stop_loss_price
        else:  # short
            if stop_loss_price <= entry_price:
                return 0.0, 0.0, "invalid_stop_loss"
            price_risk = stop_loss_price - entry_price
        
        if price_risk <= 0:
            return 0.0, 0.0, "invalid_stop_distance"
        
        stop_distance_pct = price_risk / entry_price
        
        # Ensure minimum stop distance
        # Both LIVE and DRY: reject entries with too-tight stops (honest sandbox)
        if stop_distance_pct < MIN_STOP_DISTANCE_PCT:
            # Reject entry - both LIVE and DRY behave the same
            self.logger.info(
                f"[RISK] stop_too_tight: stop_pct={stop_distance_pct*100:.4f}% < min={MIN_STOP_DISTANCE_PCT*100:.4f}% – entry rejected"
            )
            return 0.0, 0.0, "stop_too_tight"
        
        # 1) Calculate current portfolio risk
        total_risk_open = 0.0
        if open_positions:
            total_risk_open = self.calculate_total_risk_at_sl(open_positions, equity)
        
        # 2) Calculate remaining risk budget
        remaining_risk = TOTAL_RISK_BUDGET - total_risk_open
        
        if remaining_risk <= MIN_RISK_PER_TRADE:
            return 0.0, 0.0, "risk_budget_exhausted"
        
        # 3) Calculate available position slots
        # FIX: Use dynamic position limit instead of static MAX_OPEN_POSITIONS
        current_positions = len(open_positions) if open_positions else 0
        # Use dynamic position limit from LLM controller if available, else use static MAX_OPEN_POSITIONS
        effective_max = getattr(self, 'dynamic_max_positions', MAX_OPEN_POSITIONS)
        available_slots = max(1, effective_max - current_positions)
        
        # 4) Target per-trade risk: share remaining risk across available slots
        target_risk = remaining_risk / available_slots
        
        # Clamp target risk to [MIN_RISK_PER_TRADE, MAX_RISK_PER_TRADE]
        target_risk = max(MIN_RISK_PER_TRADE, min(target_risk, MAX_RISK_PER_TRADE))
        
        # Apply loss-streak state machine adjustments (DRY_RUN parity)
        try:
            score_for_bypass = signal_score if signal_score is not None else (signal_strength * 100.0)
            ok_adj, size_factor, score_bonus, _ = self.loss_streak_adjustments(score_for_bypass, time.time())
            # Sizing is independent from allow/deny; deny handled in can_enter_position
            target_risk *= size_factor
        except (TypeError, ValueError, AttributeError) as e:
            # REFACTOR: Handle loss streak adjustment errors gracefully
            self.logger.warning(f"Loss streak adjustment failed: {e}, using unadjusted risk")
        
        # RANK-BASED POSITION ALLOCATION (RPA): Apply score-based size multiplier
        # ENHANCED DYNAMIC SIZING: Clear tiers based on signal quality
        if USE_RANK_BASED_ALLOCATION:
            # Use provided signal_score or fallback to estimate from strength
            if signal_score is None:
                # Fallback: estimate score from strength (0-1 -> 0-100)
                signal_score = signal_strength * 100.0
            
            # ENHANCED DYNAMIC SIZING: Tiered approach for clarity
            # Score 45-54: 1.0x base size (standard)
            # Score 55-64: 1.5x base size (strong signals)
            # Score 65+: 2.0x base size (exceptional signals)
            if signal_score >= 65:
                size_multiplier = 2.0  # Exceptional signals - maximum size
            elif signal_score >= 55:
                size_multiplier = 1.5  # Strong signals - increased size
            else:
                size_multiplier = 1.0  # Standard signals - base size
            
            # Clamp to config limits (respect min/max from config)
            size_multiplier = max(RPA_MIN_SIZE_MULT, min(size_multiplier, RPA_MAX_SIZE_MULT))

            # ML-SYMBOL ADAPTATION: Use comprehensive symbol adaptation system
            # Adapts sizing, leverage, timing based on ML-learned symbol profiles
            if signal_score is not None and symbol:
                base_adapt_params = {
                    'size_multiplier': size_multiplier,
                    'sl_atr_multiplier': 4.0,  # Default 4x ATR
                    'leverage_multiplier': 1.0
                }

                adapted_params = self._symbol_adaptation_manager.get_adapted_parameters(symbol, base_adapt_params)

                # Apply adaptations
                size_multiplier = adapted_params.get('size_multiplier', size_multiplier)
                self.logger.debug(f"[SYMBOL_ADAPT] {symbol}: ML-adapted {size_multiplier:.2f}x size")

            target_risk *= size_multiplier
        else:
            # Legacy: Apply signal strength adjustment (scale risk based on signal quality)
            strength_multiplier = 0.7 + (signal_strength * 0.4)  # 0.7x to 1.1x
            target_risk *= strength_multiplier
        
        # Unicorn protocol: Increase risk for unicorn signals
        if is_unicorn and UNICORN_PROTOCOL_ENABLED:
            target_risk *= UNICORN_POSITION_SIZE_MULTIPLIER
        
        # Re-clamp after adjustments
        target_risk = max(MIN_RISK_PER_TRADE, min(target_risk, MAX_RISK_PER_TRADE))
        
        # 5) Convert risk fraction into position size
        # CRITICAL: Calculate BASE size first, then apply leverage
        # equity * target_risk = max notional loss at SL
        # base_size = notional_risk / price_risk
        # leveraged_size = base_size * leverage (this is what we store in position)
        notional_risk = equity * target_risk
        base_size = notional_risk / price_risk
        
        if base_size <= 0:
            return 0.0, 0.0, "size_too_small"
        
        # Apply leverage (use provided leverage or fall back to LEVERAGE_BASE)
        if leverage is None:
            leverage = LEVERAGE_BASE
        actual_leverage = leverage  # Store for later use
        
        # CRITICAL: Store LEVERAGED SIZE in position (for consistency with bot.py)
        # This is the contract size that exchange sees (base_size * leverage)
        position_size = base_size * leverage
        
        # Cap position size by capital limits
        max_capital = equity * MAX_CAPITAL_PER_POS
        max_position_by_capital = max_capital / entry_price
        
        # CRITICAL FIX: RPA max size guard - ensure it respects MAX_RISK_PER_TRADE
        # RPA allows up to RPA_MAX_RISK_BUDGET_PCT of total risk budget per position
        # but must not exceed MAX_RISK_PER_TRADE absolute limit
        if USE_RANK_BASED_ALLOCATION:
            # Calculate max risk from RPA budget allocation
            rpa_max_risk_usd = equity * TOTAL_RISK_BUDGET * RPA_MAX_RISK_BUDGET_PCT
            # Ensure it doesn't exceed MAX_RISK_PER_TRADE
            max_risk_usd = min(rpa_max_risk_usd, equity * MAX_RISK_PER_TRADE)
            max_size_by_risk = (max_risk_usd * actual_leverage) / (entry_price * stop_distance_pct) if stop_distance_pct > 0 else float('inf')
            max_position_by_capital = min(max_position_by_capital, max_size_by_risk)
        
        position_size = min(position_size, max_position_by_capital)
        position_size = min(position_size, MAX_POSITION_SIZE / entry_price)
        
        # RPA: Minimum size guard
        min_size = RPA_MIN_SIZE_USD / entry_price if USE_RANK_BASED_ALLOCATION else MIN_POSITION_SIZE / entry_price
        position_size = max(position_size, min_size)
        
        # Recalculate actual risk after capping (in case size was reduced)
        # CRITICAL: Risk is calculated on BASE size (position_size / leverage)
        # position_size here is LEVERAGED SIZE, so divide by leverage to get base
        actual_base_size = position_size / actual_leverage
        actual_notional_risk = actual_base_size * price_risk
        actual_risk_fraction = (actual_notional_risk / equity) if equity > 0 else 0.0
        
        # DRY_RUN only: Reject trades with risk_usd < MIN_RISK_USD (avoids $0.04 dust stops)
        if DRY_RUN and actual_notional_risk < MIN_RISK_USD:
            self.logger.info(
                f"[RISK] risk_too_small: risk_usd={actual_notional_risk:.4f} < min={MIN_RISK_USD:.2f} – entry rejected"
            )
            return 0.0, 0.0, "risk_too_small"
        
        # Return LEVERAGED SIZE (what exchange sees) and risk fraction
        return position_size, actual_risk_fraction, None
    
    def can_enter_position(
        self,
        symbol: str,
        spread_bps: float,
        volume_24h: float,
        latency_ms: float,
        signal_strength: float,
        current_positions: int,
        cached_now: float = None,
        is_unicorn: bool = False,
        signal_score: float = None,
        drawdown_pct: float = None,
        open_positions: Dict[str, Dict] = None,
        equity: float = None,
        entry_price: float = None,
        stop_loss: float = None,
        position_size: float = None,
        side: str = None,
        # Data-driven entry filter parameters
        rsi: float = None,
        atr_pct: float = None,
        pct_change_1h: float = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate if position can be entered.
        
        Args:
            symbol: Trading symbol
            spread_bps: Spread in basis points
            volume_24h: 24h volume
            latency_ms: Latency in milliseconds
            signal_strength: Signal strength (0-1)
            current_positions: Current number of positions
            cached_now: Cached timestamp (optional)
            is_unicorn: Whether this is a unicorn signal
            signal_score: Signal score (0-100) for tiered loss streak score boost
            drawdown_pct: Current drawdown percentage (for DD-aware protection and auto-reset)
            rsi: RSI value for entry filter
            atr_pct: ATR as percentage for volatility filter
            pct_change_1h: 1-hour price change percentage for momentum filter
        
        Returns:
            (can_enter, reason)
        """
        # Initialize replacement_symbol at the start (may be set later if replacement is possible)
        replacement_symbol = None
        
        # OPTIMIZATION: Cache time once per validation
        now = cached_now if cached_now is not None else time.time()
        
        # ============================================================
        # UNIFIED FILTER: Core entry filtering (score, limits, market conditions)
        # ============================================================
        # Use UnifiedFilter for all basic filtering logic
        filter_result = self.unified_filter.can_enter_position(
            symbol=symbol,
            signal_score=signal_score if signal_score is not None else (signal_strength * 100.0),
            signal_strength=signal_strength,
            spread_bps=spread_bps,
            volume_24h=volume_24h,
            latency_ms=latency_ms,
            current_positions=current_positions,
            open_positions=open_positions or {},
            rsi=rsi,
            atr_pct=atr_pct,
            pct_change_1h=pct_change_1h,
            side=side,
            is_unicorn=is_unicorn,
            drawdown_pct=drawdown_pct,
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size,
            cached_now=now,
        )
        
        # If unified filter rejects, return immediately (unless unicorn bypass)
        if not filter_result.can_enter and not is_unicorn:
            return False, filter_result.reason, filter_result.replacement_symbol
        
        # ============================================================
        # ML CONFIDENCE FILTER: Check model confidence before entry
        # ============================================================
        # NEW SYSTEM: Only enter when ML models are 80%+ sure it will win
        try:
            from .ml_confidence_integration import apply_ml_confidence_filter
            from .ml_confidence_config import USE_ML_CONFIDENCE_FILTER
            
            if USE_ML_CONFIDENCE_FILTER and not is_unicorn:
                # Build signal data for confidence check
                signal_data = {
                    'final_score': signal_score if signal_score is not None else (signal_strength * 100.0),
                    'ml_data': {
                        # These will be populated by bot.py before calling can_enter_position
                        'xgboost_prob': getattr(self, '_xgb_prob', None),
                        'lightgbm_prob': getattr(self, '_lgb_prob', None),
                        'random_forest_prob': getattr(self, '_rf_prob', None),
                    }
                }
                
                passed, reason = apply_ml_confidence_filter(
                    symbol=symbol,
                    signal_data=signal_data,
                    position_count=current_positions
                )
                
                if not passed:
                    # Log ML confidence rejection for visibility in UI
                    score_val = signal_score if signal_score is not None else 0
                    self.logger.warning(
                        f"[ML_CONF_REJECT] {symbol} | {reason} | "
                        f"Score={score_val:.1f} | Pos={current_positions}"
                    )
                    return False, reason, None
        except Exception as e:
            # Graceful fallback: Log error but continue with score-based filtering
            self.logger.warning(f"[ML_CONFIDENCE] Error checking confidence for {symbol}: {e}")
        
        # ============================================================
        # POSITION MANAGER SPECIFIC CHECKS (beyond basic filtering)
        # ============================================================
        
        # Check symbol churn cooldown (after basic global checks, before score filters)
        # REPLAY_MODE: Skip churn cooldown for backtesting
        if not REPLAY_MODE and self.is_symbol_churn_paused(symbol, now, SYMBOL_CHURN_COOLDOWN_SEC):
            return False, f"churn_pause_after_time_exit (symbol={symbol} cooloff={SYMBOL_CHURN_COOLDOWN_SEC}s)", None
        
        # ============================================================
        # COST GATE: DISABLED - ML SCORER HANDLES THIS INTERNALLY
        # ============================================================
        # The ML model already factors in spread, fees, and market conditions
        # when computing win probability. A separate cost gate is redundant
        # and was blocking all trades (old formula assumed scores 50-100).
        # 
        # If you want to re-enable, use ML-native formula:
        # estimated_edge_bps = max(0, (signal_score - 25) * multiplier)
        # where score 25 = 0 edge, score 50 = max edge
        pass  # ML SCORER IS IN CHARGE
        
        # AUTO-RESET: Check if loss streak should be auto-reset (time-based)
        if AUTO_RESET_LOSS_STREAK_ENABLED and self.loss_streak > 0 and self.last_loss_time > 0:
            time_since_last_loss = now - self.last_loss_time
            if time_since_last_loss >= LOSS_STREAK_AUTO_RESET_SEC:
                # Loss streak is stale; reset
                self.loss_streak = 0
                self.last_loss_time = 0.0
                # PERFORMANCE: Use pre-bound logger
                self.logger.info(
                    f"[RISK] Loss streak auto-reset after {int(time_since_last_loss)}s cooldown (no new losses)"
                )
        
        # AUTO-RESET: Also check DD-based reset
        if AUTO_RESET_LOSS_STREAK_ENABLED and drawdown_pct is not None:
            if self.check_auto_reset_loss_streak(drawdown_pct):
                if self.loss_streak > 0:
                    # PERFORMANCE: Use pre-bound logger
                    self.logger.info(
                        f"[RISK] Loss streak auto-reset due to DD improvement (DD={drawdown_pct:.2f}%)"
                    )
                self.loss_streak = 0
                self.last_loss_time = 0.0
        
        # DYNAMIC MAX POSITION CAP: Based on risk budget (floor(MAX_ACCOUNT_RISK_PCT / RISK_PER_TRADE_PCT), clamped by MAX_CONCURRENT_POS_HARD)
        # UNICORN BYPASS: Unicorns ALWAYS get in, no matter the position limit
        if equity is None or equity <= 0:
            # Fallback to legacy cap if equity not available
            max_pos = MAX_OPEN_POSITIONS
        else:
            max_pos = self.get_effective_max_positions(equity=equity)
        
        # Check position limit, but BYPASS for unicorns
        if current_positions >= max_pos:
            if is_unicorn and UNICORN_PROTOCOL_ENABLED:
                self.logger.info(f"[UNICORN_BYPASS] Allowing entry despite position limit ({current_positions}/{max_pos}) - Unicorn signal!")
                # Continue - don't return, let unicorn through
            else:
                return False, f"RJ – max positions reached ({current_positions}/{max_pos})", None
        
        # Loss-streak handled via state machine with score bypass
        # Normal validation path
        # OPTIMIZATION: Early exit for most common rejections first
        # Loss-streak state machine gate
        adj_ok, size_factor, score_bonus, pause_reason = self.loss_streak_adjustments(
            score=(signal_score if signal_score is not None else (signal_strength * 100.0)),
            now_ts=now
        )
        # loss-streak pause: LIVE still blocked, DRY_RUN only logs and continues
        if not adj_ok:
            # Re-use whatever DRY_RUN flag already exists in this module
            # DRY_RUN is imported from .config at module level
            is_dry_run = DRY_RUN
            
            if not is_dry_run:
                # LIVE: keep existing behavior and reason string (e.g. "loss_streak_pause_900s")
                return False, pause_reason or "loss_streak_pause", None
            
            # DRY_RUN: do NOT block; just keep the adjusted effective_min / score bonus from loss_streak_adjustments().
            # Keep any existing logging hook (UIAdapter) if available, but avoid changing signatures.
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.info(
                    "[RISK] Loss streak pause would trigger in LIVE (reason=%s) but is ignored in DRY_RUN for testing",
                    pause_reason or "loss_streak_pause"
                )
        # ML SCORE FLOOR CHECK - REMOVED
        # The bot.py already handles score filtering with proper idle mode awareness:
        # - IDLE mode: uses IDLE_MIN_SCORE (e.g., 42)
        # - Normal mode: uses MIN_SIGNAL_SCORE (e.g., 51)
        # Having a duplicate check here caused signals to be rejected after passing the bot's filter.
        # See bot.py lines 3125-3155 for the actual score filtering logic.
        
        # DD-AWARE LOSS STREAK PROTECTION: Check if DD-aware block should apply
        if drawdown_pct is not None:
            dd_block, dd_reason = self.check_dd_aware_loss_streak_block(drawdown_pct)
            if dd_block:
                return False, dd_reason, None
        
        # HARD RISK GATE: Check if we're already at risk budget
        # (Detailed risk check happens in calculate_position_size after sizing)
        if open_positions is not None and equity is not None and equity > 0:
            # Calculate current total risk
            current_total_risk = self.calculate_total_risk_at_sl(open_positions, equity)
            
            # If already at or over risk budget, check for replacement
            if current_total_risk >= TOTAL_RISK_BUDGET:
                # At risk budget - check for score-aware replacement
                if (SCORE_AWARE_REPLACEMENT_ENABLED and 
                    signal_score is not None and 
                    open_positions):
                    
                    weakest_symbol, weakest_score = self.find_weakest_position(open_positions)
                    
                    if (weakest_symbol and weakest_score is not None and 
                        signal_score >= weakest_score + SCORE_REPLACEMENT_MARGIN):
                        # Allow replacement
                        replacement_symbol = weakest_symbol
                        # PERFORMANCE: Use pre-bound logger
                        self.logger.info(
                            f"[RISK] Score-aware replacement: New signal (score={signal_score:.1f}) "
                            f"replaces weakest position {weakest_symbol} (score={weakest_score:.1f}) "
                            f"at risk budget ({current_total_risk*100:.2f}%)"
                        )
                    else:
                        # No replacement possible - reject
                        return False, (
                            f"Risk budget reached: {current_total_risk*100:.2f}% >= {TOTAL_RISK_BUDGET*100:.2f}%"
                        ), None
                else:
                    # No replacement logic - reject
                    return False, (
                        f"Risk budget reached: {current_total_risk*100:.2f}% >= {TOTAL_RISK_BUDGET*100:.2f}%"
                    ), None
        
        # CRITICAL: Atomic position limit check (prevents race conditions)
        # Lock ensures only one signal can check and reserve a position slot at a time
        with self._position_limit_lock:
            # Check position limit (use dynamic limit) - soft cap (risk budget is hard cap)
            # NOTE: Hard cap already checked above, this is for score-aware replacement
            if equity is None or equity <= 0:
                # Fallback to legacy cap if equity not available
                max_pos = self.get_dynamic_max_positions()
            else:
                max_pos = self.get_effective_max_positions(equity=equity)
            
            # Re-check current_positions inside lock (may have changed)
            current_positions = len(open_positions) if open_positions else 0
            
            if current_positions >= max_pos:
                # At max positions - check for score-aware replacement
                if (SCORE_AWARE_REPLACEMENT_ENABLED and 
                    signal_score is not None and 
                    open_positions is not None):
                    
                    weakest_symbol, weakest_score = self.find_weakest_position(open_positions)
                    
                    if (weakest_symbol and weakest_score is not None and 
                        signal_score >= weakest_score + SCORE_REPLACEMENT_MARGIN):
                        # Allow replacement
                        replacement_symbol = weakest_symbol
                        from .logger import get_logger
                        get_logger("PositionManager").info(
                            f"[RISK] Score-aware replacement: New signal (score={signal_score:.1f}) "
                            f"replaces weakest position {weakest_symbol} (score={weakest_score:.1f})"
                        )
                    else:
                        return False, f"Max positions reached ({max_pos})", None
                else:
                    return False, f"Max positions reached ({max_pos})", None
        
        # LEGACY SIGNAL STRENGTH CHECK - REMOVED
        # Adaptive filters handle this based on current market conditions
        # (see adaptive_filters.py for percentile-based filtering)
        
        # Spread and volume checks are now handled by UnifiedFilter
        # (removed duplicate checks here)
        
        # Check latency
        if latency_ms > MAX_LATENCY_MS:
            return False, f"Latency too high ({latency_ms:.0f}ms > {MAX_LATENCY_MS}ms)", None
        
        # Check cooldown - OPTIMIZATION: Pass cached time
        # UNICORN PROTOCOL: Bypass cooldown for unicorns
        if not (is_unicorn and UNICORN_PROTOCOL_ENABLED and UNICORN_BYPASS_COOLDOWN):
            cooldown_reason = self._check_cooldown(symbol, cached_now=now)
            if cooldown_reason:
                return False, cooldown_reason, None
        
        # Check rate limiting - OPTIMIZATION: Pass cached time
        # UNICORN PROTOCOL: Bypass rate limit for unicorns
        if not (is_unicorn and UNICORN_PROTOCOL_ENABLED and UNICORN_BYPASS_RATE_LIMIT):
            rate_limit_reason = self._check_rate_limit(cached_now=now)
            if rate_limit_reason:
                return False, rate_limit_reason, None
        
        return True, "OK", replacement_symbol
    
    def _check_cooldown(self, symbol: str, cached_now: float = None) -> Optional[str]:
        """Check if symbol is in cooldown period."""
        # ⚠️ Even in REPLAY_MODE, enforce cooldowns to prevent excessive trading
        # (Previously disabled cooldowns, but this caused 122k+ trades)
        
        # OPTIMIZATION: Accept cached time to avoid repeated time.time() calls
        now = cached_now if cached_now is not None else time.time()
        
        # Check same symbol cooldown
        if symbol in self.cooldown_until:
            cooldown_end = self.cooldown_until[symbol]
            if now < cooldown_end:
                remaining = cooldown_end - now
                return f"Cooldown active ({int(remaining)}s remaining)"
        
        # Check last entry time for same symbol
        if symbol in self.last_entry_by_symbol:
            last_entry = self.last_entry_by_symbol[symbol]
            time_since = now - last_entry
            if time_since < COOLDOWN_SAME_SYMBOL:
                remaining = COOLDOWN_SAME_SYMBOL - time_since
                return f"Same symbol cooldown ({int(remaining)}s remaining)"
        
        return None
    
    def _check_rate_limit(self, cached_now: float = None) -> Optional[str]:
        """Check if entry rate limit is exceeded."""
        # ⚠️ Even in REPLAY_MODE, enforce rate limits to prevent excessive trading
        # (Previously allowed unlimited entries, but this caused 122k+ trades)
        
        # OPTIMIZATION: Accept cached time to avoid repeated time.time() calls
        now = cached_now if cached_now is not None else time.time()
        
        # Reset daily counter if it's a new day (86400 seconds = 24 hours)
        if now - self.last_daily_reset >= 86400:
            self.daily_trades = []
            self.last_daily_reset = now
        
        # Daily trade limit: DISABLED per user request
        # User prefers to fix the system if broken rather than limit trades
        # daily_limit = MAX_TRADES_PER_DAY_REPLAY if REPLAY_MODE else MAX_TRADES_PER_DAY
        # if len(self.daily_trades) >= daily_limit:
        #     return f"Daily trade limit exceeded ({daily_limit} trades/day, {len(self.daily_trades)} today)"
        
        # OPTIMIZATION: Use generator expression instead of list comprehension
        # Remove entries older than 1 minute
        self.entry_times = [t for t in self.entry_times if now - t < 60]
        
        # Check if we've exceeded the per-minute limit
        if len(self.entry_times) >= MAX_ENTRIES_PER_MIN:
            return f"Rate limit exceeded ({MAX_ENTRIES_PER_MIN} entries/min)"
        
        return None
    
    def record_entry(self, symbol: str, adaptive_cooldown: bool = True, was_win: Optional[bool] = None):
        """Record position entry for cooldown and rate limiting."""
        now = time.time()
        
        # Record in UnifiedFilter for rate limiting
        self.unified_filter.record_entry(timestamp=now)
        
        # Track daily trades
        if now - self.last_daily_reset >= 86400:
            self.daily_trades = []
            self.last_daily_reset = now
        self.daily_trades.append(now)
        
        # Record entry time for rate limiting (legacy - kept for backward compatibility)
        self.entry_times.append(now)
        
        # Set cooldown
        if adaptive_cooldown and was_win is not None:
            # Adaptive cooldown: longer after losses, shorter after wins
            base_cooldown = COOLDOWN_SEC
            if was_win:
                cooldown = base_cooldown * 0.5  # Shorter after wins
            else:
                cooldown = base_cooldown * 2.0  # Longer after losses
        else:
            cooldown = COOLDOWN_SEC
        
        self.cooldown_until[symbol] = now + cooldown
        self.last_entry_by_symbol[symbol] = now
    
    def get_cooldown_remaining(self, symbol: str) -> float:
        """Get remaining cooldown time for symbol."""
        if symbol not in self.cooldown_until:
            return 0.0
        
        remaining = self.cooldown_until[symbol] - time.time()
        return max(0.0, remaining)
    
    def clear_cooldown(self, symbol: str):
        """Clear cooldown for symbol (e.g., after manual intervention)."""
        if symbol in self.cooldown_until:
            del self.cooldown_until[symbol]
        if symbol in self.last_entry_by_symbol:
            del self.last_entry_by_symbol[symbol]
        if symbol in self._symbol_churn:
            del self._symbol_churn[symbol]
    
    def is_symbol_churn_paused(self, symbol: str, now_ts: float, cooldown_sec: float) -> bool:
        """
        Check if symbol is in churn cooldown after a micro time_exit.
        
        Args:
            symbol: Trading symbol
            now_ts: Current timestamp
            cooldown_sec: Cooldown duration in seconds
        
        Returns:
            True if symbol is in churn cooldown, False otherwise
        """
        info = self._symbol_churn.get(symbol)
        if not info:
            return False
        return (now_ts - info["ts"]) < cooldown_sec
    
    def get_entry_delay_ms(self, signal_strength: float, volatility: float = 0.0) -> int:
        """
        Calculate adaptive entry delay based on signal strength and volatility.
        OPTIMIZED: Much lower delays for faster entry execution.
        
        Args:
            signal_strength: Signal strength (0-1)
            volatility: Current volatility (0-1)
        
        Returns:
            Entry delay in milliseconds
        """
        from .config import ENTRY_DELAY_MS
        
        # OPTIMIZATION: Start with lower base delay
        base_delay = ENTRY_DELAY_MS
        
        # OPTIMIZATION: Aggressively reduce delay for high-strength signals
        if signal_strength > 0.85:
            base_delay = 100  # Ultra-fast for very strong signals (100ms)
        elif signal_strength > 0.75:
            base_delay = 150  # Very fast for strong signals (150ms)
        elif signal_strength > 0.6:
            base_delay = int(base_delay * 0.5)  # 50% of base for good signals
        elif signal_strength > 0.5:
            base_delay = int(base_delay * 0.7)  # 70% of base for moderate signals
        
        # OPTIMIZATION: Only increase delay significantly for very high volatility
        if volatility > 0.8:
            base_delay = int(base_delay * 1.2)  # Only 20% increase for extreme volatility
        elif volatility > 0.6:
            base_delay = int(base_delay * 1.1)  # 10% increase for high volatility
        
        # OPTIMIZATION: Lower minimum delay for faster execution
        return max(50, min(base_delay, 500))  # Min 50ms, max 500ms
    
    def calculate_estimated_costs(self, position_size: float, entry_price: float) -> Dict[str, float]:
        """
        Calculate estimated trading costs.
        
        Returns:
            Dict with 'entry_fee', 'exit_fee', 'slippage', 'total'
        """
        notional = position_size * entry_price
        
        entry_fee = notional * TAKER_FEE_RATE
        exit_fee = notional * TAKER_FEE_RATE
        slippage = notional * (SLIPPAGE_BPS / 10000)
        
        return {
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'slippage': slippage,
            'total': entry_fee + exit_fee + slippage
        }
    
    def record_exit(self, symbol: str, was_win: bool, pnl_pct: float = None, exit_reason: Optional[str] = None, profit_atr: Optional[float] = None):
        """Record position exit and update loss streak. Set cooldown to prevent immediate re-entry.
        
        Args:
            symbol: Trading symbol
            was_win: Whether the trade was a winner
            pnl_pct: PnL percentage (optional, for auto-reset on decent winners)
            exit_reason: Exit reason string (optional, for churn tracking)
            profit_atr: Profit in ATR units (optional, for churn tracking)
        """
        now = time.time()
        self.last_exit_time = now
        self.last_trade_was_win = was_win
        
        from .logger import get_logger
        logger = get_logger("PositionManager")
        
        # Determine if this is DRY_RUN
        # DRY_RUN is imported from .config at module level
        is_dry_run = DRY_RUN
        
        # Compute realized R if not already present
        # Try to get from position if available, otherwise calculate from pnl_pct and stop distance
        realized_R = None
        if symbol in self.positions:
            position = self.positions[symbol]
            # Try to get max_r_reached or current_r from position
            realized_R = position.get('max_r_reached') or position.get('current_r')
            
            # If not available, calculate from pnl_pct and stop distance
            if realized_R is None and pnl_pct is not None:
                entry_price = position.get('entry_price', 0)
                stop_loss = position.get('stop_loss', 0)
                if entry_price > 0 and stop_loss > 0:
                    # Calculate stop distance as percentage
                    side = position.get('side', 'long').lower()
                    if side == 'long':
                        stop_distance_pct = abs((entry_price - stop_loss) / entry_price)
                    else:  # short
                        stop_distance_pct = abs((stop_loss - entry_price) / entry_price)
                    
                    if stop_distance_pct > 0:
                        # R = pnl_pct / stop_distance_pct (convert pnl_pct from % to decimal)
                        realized_R = (pnl_pct / 100.0) / stop_distance_pct
        
        # 6.1: Scratches (|R| < 0.1) do NOT affect loss streak at all
        if realized_R is not None and abs(realized_R) < 0.10:
            # Keep optional debug log so we know scratches happened
            logger.info(
                f"[RISK] Scratch exit (|R|={realized_R:.3f} < 0.10) – not counted towards loss streak"
            )
            # Early return: do NOT increment loss_streak, do NOT change state
            # Still set cooldown below
        # 6.2: In DRY_RUN, completely disable loss-streak accumulation for testing
        elif is_dry_run:
            logger.info(
                "[RISK] Loss streak update suppressed in DRY_RUN (would have %s).",
                "incremented" if not was_win else "reset"
            )
            # Again: do not modify self.loss_streak or state; just continue to cooldown
        # 6.3: LIVE mode – keep existing logic exactly as is
        else:
            if was_win:
                old_streak = self.loss_streak
                self.loss_streak = 0
                self.last_loss_time = 0.0  # Clear last loss time on win
                # Log streak reset on win
                if old_streak > 0:
                    logger.info(
                        f"[RISK] Loss streak reset to 0 after win (was {old_streak} losses, "
                        f"pnl_pct={pnl_pct:.2f}% if available)"
                    )
                # AUTO-RESET: Reset loss streak on decent winner (if enabled)
                if (AUTO_RESET_LOSS_STREAK_ENABLED and 
                    pnl_pct is not None and 
                    pnl_pct >= AUTO_RESET_WINNER_PNL_THRESHOLD):
                    logger.info(
                        f"[RISK] Loss streak reset due to decent winner (pnl_pct={pnl_pct:.2f}% >= "
                        f"{AUTO_RESET_WINNER_PNL_THRESHOLD}%)"
                    )
            else:
                self.loss_streak += 1
                self.last_loss_time = now  # Track time of last loss for auto-reset
                # Log loss streak increment
                logger.info(
                    f"[RISK] Loss streak incremented to {self.loss_streak} "
                    f"(last_loss_time={now:.0f})"
                )
                # Recompute state and set pause timer if needed
                self._recompute_loss_streak_state(now)
        
        # Track micro time_exit churn (flat-ish exits that indicate symbol is not working)
        if exit_reason and exit_reason.startswith("time_exit"):
            if profit_atr is not None and abs(profit_atr) <= 0.25:
                self._symbol_churn[symbol] = {"ts": now, "profit_atr": profit_atr}
        
        # Set cooldown after exit to prevent immediate re-entry (now 2 minutes, softened)
        # This helps prevent position accumulation without being too conservative
        self.cooldown_until[symbol] = now + COOLDOWN_AFTER_EXIT
        self.last_entry_by_symbol[symbol] = now  # Update last entry time to track cooldown

    def _recompute_loss_streak_state(self, now: float) -> None:
        """Recompute loss-streak state and manage pause timers."""
        ls = self.loss_streak
        prev_state = self.loss_streak_state
        if ls >= LOSS_STREAK_HARD_LEVEL:
            self.loss_streak_state = "pause"
            # Start/refresh pause window
            self.loss_streak_unlock_ts = max(self.loss_streak_unlock_ts, now + LOSS_STREAK_PAUSE_SEC)
        elif ls >= LOSS_STREAK_DEFENSE_LEVEL:
            self.loss_streak_state = "defense"
        elif ls >= LOSS_STREAK_CAUTION_LEVEL:
            self.loss_streak_state = "caution"
        else:
            self.loss_streak_state = "normal"
        if prev_state != self.loss_streak_state:
            from .logger import get_logger
            get_logger("PositionManager").warning(
                f"[RISK] Loss-streak state: {prev_state} -> {self.loss_streak_state} (ls={ls})"
            )

    def maybe_decay_loss_streak(self, now: float) -> None:
        """Time-based decay of loss streak to self-heal over time."""
        if self.loss_streak > 0 and self.last_loss_time > 0:
            if now - self.last_loss_time >= LOSS_STREAK_DECAY_SECONDS:
                self.loss_streak = max(0, self.loss_streak - 1)
                self.last_loss_time = now
                self._recompute_loss_streak_state(now)
                try:
                    from .logger import get_logger
                    # PERFORMANCE: Use pre-bound logger
                    self.logger.info(
                        f"[RISK] Loss streak decayed by 1 (ls={self.loss_streak})"
                    )
                except (TypeError, ValueError, KeyError):
                    # REFACTOR: Silently handle metrics update failures
                    pass  # Metrics are non-critical

    def loss_streak_adjustments(self, score: float, now_ts: float) -> Tuple[bool, float, float, Optional[str]]:
        """
        Return (ok, size_factor, min_score_bonus, reason) based on loss-streak state machine.
        Unicorn bypass: if score >= HIGH_SCORE_BYPASS, always ok with size_factor <= 0.5.
        """
        # Self-heal periodically
        self.maybe_decay_loss_streak(now_ts)
        # Ensure state consistent with current streak
        self._recompute_loss_streak_state(now_ts)
        ls_state = self.loss_streak_state
        ls = self.loss_streak

        # Unicorn bypass (score-based)
        if score is not None and score >= HIGH_SCORE_BYPASS:
            return True, 0.5, 0.0, None

        if ls_state == "pause":
            # DRY_RUN override: soften global pause to 45s
            if DRY_RUN:
                dry_remaining = self.loss_streak_unlock_ts - now_ts
                if dry_remaining > 45:
                    return False, 0.0, 0.0, f"loss_streak_pause_dryrun_{int(dry_remaining)}s"
                # If dry_remaining <= 45, fall through to end-of-pause logic
            else:
                # LIVE mode: keep full strict pause
                if now_ts < self.loss_streak_unlock_ts:
                    remaining = int(self.loss_streak_unlock_ts - now_ts)
                    return False, 0.0, 0.0, f"loss_streak_pause_{remaining}s"
                # If pause expired, fall through to end-of-pause logic
            
            # End of pause: soften by 1 and move to defense (applies to both DRY_RUN and LIVE)
            self.loss_streak = max(LOSS_STREAK_DEFENSE_LEVEL, self.loss_streak - 1)
            self._recompute_loss_streak_state(now_ts)
            # CONTINUOUS RISK UPDATE: No score penalty, only size reduction (0.5x)
            return True, 0.5, 0.0, None
        elif ls_state == "defense":
            # CONTINUOUS RISK UPDATE: No score penalty, only size reduction (0.5x)
            return True, 0.5, 0.0, None
        elif ls_state == "caution":
            return True, 0.7, 0.0, None
        else:
            return True, 1.0, 0.0, None
    
    def check_auto_reset_loss_streak(self, drawdown_pct: float = None) -> bool:
        """
        Check if loss streak should be auto-reset based on time, performance, or DD improvement.
        
        Args:
            drawdown_pct: Current drawdown percentage (optional, for DD-based reset)
        
        Returns:
            True if loss streak should be reset, False otherwise
        """
        # CRITICAL FIX: Removed DRY_RUN check for test/prod parity
        if not AUTO_RESET_LOSS_STREAK_ENABLED:
            return False
        
        if self.loss_streak == 0:
            return False  # No streak to reset
        
        now = time.time()
        
        # Time-based reset: Reset after N minutes without new loss (use LOSS_STREAK_AUTO_RESET_SEC)
        if self.last_loss_time > 0:
            time_since_last_loss = now - self.last_loss_time
            if time_since_last_loss >= LOSS_STREAK_AUTO_RESET_SEC:
                return True
        
        # CRITICAL FIX: DD-based reset - drawdown_pct is NEGATIVE (e.g., -5%)
        # Reset if DD improved (less negative) to threshold or better
        # E.g., if threshold is -2% and current DD is -1%, that's better (less negative = higher value)
        # The comparison drawdown_pct >= threshold is correct: -1 >= -2 is True (less negative is better)
        # NOTE: Config currently sets AUTO_RESET_DD_IMPROVEMENT_THRESHOLD to 0.0, meaning "reset when profitable"
        # To reset at improved DD (e.g., from -5% to -2%), set threshold to a negative value like -2.0
        if drawdown_pct is not None and drawdown_pct >= AUTO_RESET_DD_IMPROVEMENT_THRESHOLD:
            return True
        
        return False
    
    def get_tiered_loss_streak_adjustments(self) -> Tuple[float, float, int]:
        """
        Get tiered loss streak adjustments for risk, position limit, and score threshold.
        
        Returns:
            (risk_multiplier, position_multiplier, score_boost)
        """
        # CRITICAL FIX: Removed DRY_RUN check for test/prod parity
        if not TIERED_LOSS_STREAK_ENABLED or self.loss_streak == 0:
            return 1.0, 1.0, 0
        
        if self.loss_streak >= LOSS_STREAK_TIER2:
            # Tier 2: Most aggressive throttling
            return (
                LOSS_STREAK_TIER2_RISK_MULTIPLIER,
                LOSS_STREAK_TIER2_POSITION_MULTIPLIER,
                LOSS_STREAK_TIER2_SCORE_BOOST
            )
        elif self.loss_streak >= LOSS_STREAK_TIER1:
            # Tier 1: Moderate throttling
            return (
                LOSS_STREAK_TIER1_RISK_MULTIPLIER,
                LOSS_STREAK_TIER1_POSITION_MULTIPLIER,
                LOSS_STREAK_TIER1_SCORE_BOOST
            )
        else:
            # Below tier 1: No adjustments
            return 1.0, 1.0, 0
    
    def check_dd_aware_loss_streak_block(self, drawdown_pct: float) -> Tuple[bool, str]:
        """
        Check if loss streak should block entries based on DD-aware protection.
        
        Args:
            drawdown_pct: Current drawdown percentage
        
        Returns:
            (should_block, reason)
        """
        # CRITICAL FIX: Removed DRY_RUN check for test/prod parity
        if not DD_AWARE_LOSS_STREAK_ENABLED:
            return False, ""
        
        # DD-aware block: Only block if BOTH streak >= threshold AND DD >= threshold
        if (self.loss_streak >= DD_AWARE_STREAK_THRESHOLD and 
            drawdown_pct <= -DD_AWARE_DD_THRESHOLD):
            return True, f"DD-aware loss streak protection: {self.loss_streak} losses with {drawdown_pct:.2f}% DD"
        
        return False, ""
    
    def get_cooldown_status(self) -> Dict:
        """
        Get comprehensive cooldown status summary.
        
        Returns:
            Dict with cooldown information:
            - symbol_cooldowns: List of active symbol cooldowns
            - global_cooldown_remaining: Remaining global cooldown (0 if none)
            - exit_to_entry_remaining: Remaining exit→entry cooldown
            - loss_streak_cooldown: Loss streak cooldown status
            - active_count: Number of active cooldowns
        """
        now = time.time()
        symbol_cooldowns = []
        
        for symbol, cooldown_end in self.cooldown_until.items():
            if now < cooldown_end:
                remaining = cooldown_end - now
                symbol_cooldowns.append({
                    'symbol': symbol,
                    'remaining': remaining
                })
        
        # Global cooldown (based on last entry)
        global_cooldown_remaining = 0.0
        if self.entry_times:
            last_entry = max(self.entry_times)
            time_since = now - last_entry
            if time_since < COOLDOWN_DIFF_SYMBOL:
                global_cooldown_remaining = COOLDOWN_DIFF_SYMBOL - time_since
        
        # Exit→Entry cooldown
        exit_to_entry_remaining = 0.0
        if self.last_exit_time > 0:
            time_since_exit = now - self.last_exit_time
            if time_since_exit < COOLDOWN_DIFF_SYMBOL:
                exit_to_entry_remaining = COOLDOWN_DIFF_SYMBOL - time_since_exit
        
        # Loss streak cooldown (2x normal cooldown after 3+ losses)
        loss_streak_cooldown = None
        if self.loss_streak >= 3:
            loss_streak_cooldown = {
                'active': True,
                'streak': self.loss_streak,
                'multiplier': 2.0
            }
        
        return {
            'symbol_cooldowns': symbol_cooldowns,
            'global_cooldown_remaining': global_cooldown_remaining,
            'exit_to_entry_remaining': exit_to_entry_remaining,
            'loss_streak_cooldown': loss_streak_cooldown,
            'active_count': len(symbol_cooldowns) + (1 if global_cooldown_remaining > 0 else 0) + (1 if exit_to_entry_remaining > 0 else 0)
        }
    
    def get_effective_max_positions(self, equity: float) -> int:
        """
        Dynamic position cap based on risk budget:
        floor(MAX_ACCOUNT_RISK_PCT / RISK_PER_TRADE_PCT), clamped by MAX_CONCURRENT_POS_HARD.
        
        Args:
            equity: Current account equity (used for validation, not calculation)
        
        Returns:
            Effective maximum positions based on risk budget
        """
        total_risk = float(MAX_ACCOUNT_RISK_PCT)
        per_trade = float(RISK_PER_TRADE_PCT)
        hard_cap = int(MAX_CONCURRENT_POS_HARD)
        
        if per_trade <= 0:
            return 0
        
        dyn_cap = int(math.floor(total_risk / per_trade))
        if dyn_cap < 1:
            dyn_cap = 1
        
        return max(1, min(dyn_cap, hard_cap))
    
    def get_dynamic_max_positions(self) -> int:
        """
        Get current dynamic maximum positions.
        Can be adjusted by LLM based on market conditions.
        """
        return max(MAX_CONCURRENT_POS_MIN, min(self.dynamic_max_positions, MAX_CONCURRENT_POS_MAX))
    
    def adjust_max_positions(
        self,
        new_limit: int,
        reason: str = "LLM adjustment"
    ):
        """
        Adjust maximum concurrent positions dynamically.
        Called by LLM controller based on market conditions.
        
        Args:
            new_limit: New position limit (will be clamped to MIN/MAX)
            reason: Reason for adjustment
        """
        # Clamp to min/max bounds
        clamped_limit = max(MAX_CONCURRENT_POS_MIN, min(new_limit, MAX_CONCURRENT_POS_MAX))
        
        old_limit = self.dynamic_max_positions
        self.dynamic_max_positions = clamped_limit
        self.last_limit_adjustment = time.time()
        
        if old_limit != clamped_limit:
            # Note: This is informational, not an error - using logger if available
            try:
                from .logger import get_logger
                get_logger("PositionManager").info(
                    f"Position limit adjusted",
                    old_limit=old_limit,
                    new_limit=clamped_limit,
                    reason=reason
                )
            except ImportError:
                pass  # Logger not available, skip
        
        return clamped_limit
    
    def calculate_optimal_max_positions(
        self,
        volatility_regime: str,
        spread_regime: str,
        win_rate: float,
        profit_factor: float,
        drawdown_pct: float,
        signal_quality: float,
        btc_trend: float,
        regime_config=None
    ) -> int:
        """
        Calculate optimal max positions based on market conditions.
        This is used by LLM to make informed decisions.
        
        Args:
            regime_config: Optional regime configuration. If provided, uses regime-specific limits.
        
        Returns:
            Suggested max positions
        """
        # Use regime-specific base limit if available, otherwise use static config
        if regime_config and hasattr(regime_config, 'max_concurrent_positions'):
            base_limit = regime_config.max_concurrent_positions
            min_limit = regime_config.max_concurrent_positions_min
            max_limit = regime_config.max_concurrent_positions_max
        else:
            base_limit = MAX_CONCURRENT_POS
            min_limit = MAX_CONCURRENT_POS_MIN
            max_limit = MAX_CONCURRENT_POS_MAX
        
        # Volatility adjustments
        if volatility_regime == "Low":
            base_limit += 2  # More positions in low volatility
        elif volatility_regime == "High":
            base_limit -= 2  # Fewer positions in high volatility
        
        # Spread adjustments
        if spread_regime == "Tight":
            base_limit += 1  # More positions when spreads are tight
        elif spread_regime == "Wide":
            base_limit -= 1  # Fewer positions when spreads are wide
        
        # Performance adjustments
        if win_rate > 0.65 and profit_factor > 1.5:
            base_limit += 2  # Increase if performing well
        elif win_rate < 0.45 or profit_factor < 0.8:
            base_limit -= 2  # Decrease if performing poorly
        
        # Drawdown adjustments
        if drawdown_pct < -5.0:
            base_limit -= 3  # Significantly reduce in large drawdown
        elif drawdown_pct < -2.0:
            base_limit -= 1  # Slightly reduce in moderate drawdown
        elif drawdown_pct > 5.0:
            base_limit += 1  # Slightly increase in strong performance
        
        # Signal quality adjustments
        if signal_quality > 0.75:
            base_limit += 1  # More positions when signals are strong
        elif signal_quality < 0.5:
            base_limit -= 1  # Fewer positions when signals are weak
        
        # BTC trend adjustments
        if abs(btc_trend) > 2.0:
            base_limit += 1  # More positions in strong BTC trend
        elif abs(btc_trend) < 0.3:
            base_limit -= 1  # Fewer positions in weak/no BTC trend
        
        # Clamp to bounds (use regime-specific limits if available)
        return max(min_limit, min(int(base_limit), max_limit))
    
    def get_open_positions_count(self) -> int:
        """Return count of open positions."""
        return len(self.positions)
    
    def get_loss_streak(self) -> int:
        """Return current loss streak."""
        return self.loss_streak
    
    def compute_position_risk_fraction(
        self,
        position: Dict,
        equity: float
    ) -> float:
        """
        Compute risk fraction for a single position.
        
        CRITICAL CLARIFICATION: Position 'size' field stores LEVERAGED SIZE (contracts).
        Risk calculation must use BASE SIZE = leveraged_size / leverage.
        
        Example:
          - Entry: $50,000 BTCUSDT, leverage 5x, stop 2% away
          - Leveraged size stored: 250 contracts (5x multiplier)
          - Base size for risk: 50 contracts (250 / 5)
          - Risk at SL: 50 * 2% * price = actual capital at risk
        
        Args:
            position: Position dictionary with entry_price, stop_loss, size (leveraged), leverage, side
            equity: Total account equity
        
        Returns:
            Risk as fraction of equity (0.0 to 1.0)
        """
        if equity <= 0:
            return 0.0
        
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        position_size = position.get('size', 0)  # This is LEVERAGED size
        leverage = position.get('leverage', 1)  # Get leverage from position
        side = position.get('side', 'long').lower()
        
        if not entry_price or not stop_loss or not position_size:
            return 0.0
        
        # Calculate absolute price distance to SL
        if side == 'long':
            if entry_price <= stop_loss:
                return 0.0  # Invalid
            price_risk = entry_price - stop_loss
        else:  # short
            if stop_loss <= entry_price:
                return 0.0  # Invalid
            price_risk = stop_loss - entry_price
        
        # CRITICAL: Position size stored is LEVERAGED SIZE
        # Base size = leveraged size / leverage (actual capital exposure)
        # Handle None leverage gracefully (default to 1)
        if leverage is None:
            leverage = 1
            
        base_size = abs(position_size) / leverage if leverage > 0 else abs(position_size)
        
        # Notional risk (quote currency lost if SL hit)
        # Risk is independent of leverage - it's the price movement * base size
        notional_risk = price_risk * base_size
        
        # Risk as fraction of equity
        return notional_risk / equity
    
    def calculate_total_risk_at_sl(
        self,
        open_positions: Dict[str, Dict],
        equity: float
    ) -> float:
        """
        Calculate total risk at stop-loss across all open positions.
        
        Args:
            open_positions: Dictionary of open positions {symbol: position_dict}
            equity: Current account equity
        
        Returns:
            Total risk as fraction of equity (0.0 to 1.0)
        """
        if not open_positions or equity <= 0:
            return 0.0
        
        total_risk = 0.0
        for symbol, position in open_positions.items():
            risk_fraction = self.compute_position_risk_fraction(position, equity)
            total_risk += risk_fraction
        
        return total_risk
    
    def calculate_new_trade_risk_at_sl(
        self,
        entry_price: float,
        stop_loss: float,
        position_size: float,
        side: str,
        equity: float
    ) -> float:
        """
        Calculate risk at stop-loss for a new trade.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop-loss price
            position_size: Position size
            side: 'long' or 'short'
            equity: Current account equity
        
        Returns:
            Risk as fraction of equity (0.0 to 1.0)
        """
        if equity <= 0:
            return 0.0
        
        # Create temporary position dict for calculation
        temp_position = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'size': position_size,
            'side': side
        }
        
        return self.compute_position_risk_fraction(temp_position, equity)
    
    def find_weakest_position(
        self,
        open_positions: Dict[str, Dict]
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Find the weakest position (lowest signal score) for replacement.
        
        Args:
            open_positions: Dictionary of open positions {symbol: position_dict}
        
        Returns:
            (weakest_symbol, weakest_score) or (None, None) if no positions
        """
        if not open_positions:
            return None, None
        
        weakest_symbol = None
        weakest_score = None
        
        for symbol, position in open_positions.items():
            # Try to get signal score from position (stored as 'signal_score' or similar)
            # If not available, use signal_strength as proxy
            score = position.get('signal_score', None)
            if score is None:
                # Fall back to signal_strength * 100 as proxy
                strength = position.get('signal_strength', 0.0)
                score = strength * 100.0
            
            if weakest_score is None or (score is not None and score < weakest_score):
                weakest_score = score
                weakest_symbol = symbol
        
        return weakest_symbol, weakest_score
    
    def get_exposure_score_data(self) -> Dict:
        """
        Return exposure data for ExposureScore calculation.
        
        Returns:
            Dict with open_positions, max_positions, correlated_positions
        """
        # For now, we don't track correlation, so correlated_positions = 0
        # Could be enhanced later to track correlated positions (e.g., same sector)
        return {
            'open_positions': len(self.positions),
            'max_positions': self.dynamic_max_positions,
            'correlated_positions': 0  # Not tracking correlation yet
        }
    
    def calculate_dynamic_leverage(self, signal_strength: float, is_unicorn: bool = False) -> int:
        """
        Calculate dynamic leverage based on signal strength.
        
        Formula: leverage = MIN_LEVERAGE + (strength × (MAX_LEVERAGE - MIN_LEVERAGE))
        
        Args:
            signal_strength: Signal strength (0.0-1.0)
            is_unicorn: Whether this is a unicorn signal
        
        Returns:
            Leverage (2x to 10x)
        """
        if not USE_DYNAMIC_LEVERAGE:
            leverage = LEVERAGE_BASE
        else:
            # Clamp strength to valid range
            strength = max(0.0, min(1.0, signal_strength))
            
            # Calculate leverage: MIN + (strength × range)
            leverage = MIN_LEVERAGE + (strength * (MAX_LEVERAGE - MIN_LEVERAGE))
            
            # CRITICAL FIX: Apply streak penalty (removed DRY_RUN check for test/prod parity)
            if self.loss_streak >= STREAK_THRESHOLD:
                leverage *= (1.0 - STREAK_PENALTY_PCT)
        
        # UNICORN PROTOCOL: Increase leverage for unicorn signals
        if is_unicorn and UNICORN_PROTOCOL_ENABLED:
            leverage *= UNICORN_LEVERAGE_MULTIPLIER
        
        # Round to integer and clamp
        leverage = int(round(leverage))
        return max(MIN_LEVERAGE, min(MAX_LEVERAGE, leverage))
    
    def count_correlated_positions(self, side: str, btc_trend: float = 0.0) -> int:
        """
        Count positions that are correlated with the new position.
        
        Simple correlation: If BTC is trending and we have multiple positions in same direction,
        they're likely correlated.
        
        Args:
            side: 'long' or 'short'
            btc_trend: BTC trend percentage
        
        Returns:
            Number of correlated positions
        """
        # If BTC trend is weak, positions are less correlated
        if abs(btc_trend) < 0.3:
            return 0
        
        # Count positions in same direction when BTC is trending
        correlated = 0
        for pos in self.positions.values():
            pos_side = pos.get('side', '').lower()
            if pos_side == side.lower():
                correlated += 1
        
        return correlated
    
    def calculate_dynamic_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        signal_strength: float,
        side: str,
        leverage: int,
        btc_trend: float = 0.0,
        atr_pct: float = None,
        is_unicorn: bool = False
    ) -> Tuple[float, int]:
        """
        Calculate dynamic position size and leverage based on signal strength and risk factors.
        
        Args:
            equity: Current account equity
            entry_price: Entry price
            stop_loss_price: Stop loss price
            signal_strength: Signal strength (0.0-1.0)
            side: 'long' or 'short'
            leverage: Calculated leverage
            btc_trend: BTC trend percentage (for correlation check)
            atr_pct: ATR as percentage (optional, for volatility scaling)
        
        Returns:
            (position_size, final_leverage)
        """
        if not USE_DYNAMIC_POSITION_SIZE:
            # Fall back to risk-based method (always use risk budget approach)
            # Note: open_positions not available here, will be passed from bot.py
            size, risk_fraction, reason = self.calculate_position_size(
                "", equity, entry_price, stop_loss_price, signal_strength, side,
                is_unicorn=is_unicorn, open_positions=None
            )
            if reason:
                return 0.0, leverage
            return size, leverage
        
        # Validate inputs
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0, leverage
        
        # Calculate stop distance
        if side == "long":
            if entry_price <= stop_loss_price:
                return 0.0, leverage
            stop_distance_pct = (entry_price - stop_loss_price) / entry_price
        else:  # short
            if stop_loss_price <= entry_price:
                return 0.0, leverage
            stop_distance_pct = (stop_loss_price - entry_price) / entry_price
        
        if stop_distance_pct <= 0 or stop_distance_pct < MIN_STOP_DISTANCE_PCT:
            return 0.0, leverage
        
        # Calculate base position size as % of capital
        # Formula: BASE_POSITION_PCT + (strength - 0.75) * 6
        # This gives: strength 0.65 → 0.5%, 0.75 → 1.5%, 0.85 → 2.5%, 1.0 → 3.0%
        strength = max(0.0, min(1.0, signal_strength))
        size_pct = BASE_POSITION_PCT + (strength - 0.75) * 6.0
        size_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, size_pct))
        
        # Apply correlation penalty (unless unicorn bypasses it)
        if not (is_unicorn and UNICORN_PROTOCOL_ENABLED and UNICORN_BYPASS_CORRELATION):
            correlated_count = self.count_correlated_positions(side, btc_trend)
            if correlated_count >= CORRELATION_THRESHOLD:
                size_pct *= (1.0 - CORRELATION_PENALTY_PCT)
        
        # CRITICAL FIX: Apply streak penalty (removed DRY_RUN check for test/prod parity)
        if not (is_unicorn and UNICORN_PROTOCOL_ENABLED and UNICORN_BYPASS_LOSS_STREAK):
            if self.loss_streak >= STREAK_THRESHOLD:
                size_pct *= (1.0 - STREAK_PENALTY_PCT)
                # Also reduce leverage (already done in calculate_dynamic_leverage, but ensure consistency)
                leverage = int(round(leverage * (1.0 - STREAK_PENALTY_PCT)))
        
        # UNICORN PROTOCOL: Increase position size for unicorn signals
        if is_unicorn and UNICORN_PROTOCOL_ENABLED:
            size_pct *= UNICORN_POSITION_SIZE_MULTIPLIER
            leverage = max(MIN_LEVERAGE, min(MAX_LEVERAGE, leverage))
        
        # Calculate position size in base currency
        position_size_usd = equity * (size_pct / 100.0)
        position_size = position_size_usd / entry_price
        
        # Apply leverage
        position_size = position_size * leverage
        
        # Risk cap: Ensure max risk at stop loss doesn't exceed MAX_RISK_PER_TRADE_PCT
        max_risk_usd = equity * (MAX_RISK_PER_TRADE_PCT / 100.0)
        risk_at_stop = position_size * entry_price * stop_distance_pct / leverage
        if risk_at_stop > max_risk_usd:
            # Reduce position size to meet risk cap
            position_size = (max_risk_usd * leverage) / (entry_price * stop_distance_pct)
        
        # Apply final caps
        max_capital = equity * MAX_CAPITAL_PER_POS
        max_position_by_capital = (max_capital * leverage) / entry_price
        
        position_size = min(position_size, max_position_by_capital)
        position_size = min(position_size, MAX_POSITION_SIZE / entry_price)
        position_size = max(position_size, MIN_POSITION_SIZE / entry_price)
        
        return position_size, leverage


class SymbolAdaptationManager:
    """
    ML-Driven Symbol Adaptation System

    Instead of blacklisting bad symbols, this system:
    1. Learns optimal parameters for each symbol from ML training
    2. Adapts position sizing, stops, leverage, and timing per symbol
    3. Enables profitable trading of "difficult" symbols like BNB

    Key insight: BNB shows -95% avg loss in raw data, but ML finds +59% avg R patterns!
    """

    def __init__(self):
        self._symbol_profiles = {}
        self._last_cache_update = 0
        self._cache_ttl = 300  # 5 minutes

    def get_adapted_parameters(self, symbol: str, base_params: dict) -> dict:
        """
        Get ML-adapted trading parameters for a symbol.

        Args:
            symbol: Trading symbol
            base_params: Default/base parameters

        Returns:
            Adapted parameters with symbol-specific optimizations
        """
        profile = self._get_symbol_profile(symbol)
        if not profile:
            return base_params  # No adaptation data, use defaults

        adapted = base_params.copy()

        # 1. POSITION SIZING: Base on historical win rate
        win_rate = profile.get('win_rate', 0.4)
        if win_rate > 0.55:  # Above 55% win rate
            size_boost = 1.0 + (win_rate - 0.55) * 2.0  # Up to 2x size for 65%+ WR
            adapted['size_multiplier'] = adapted.get('size_multiplier', 1.0) * size_boost

        # 2. STOP LOSS: Use ML-learned optimal SL
        if 'optimal_sl' in profile:
            # ML suggests optimal SL as percentage, convert to ATR multiplier
            optimal_sl_pct = profile['optimal_sl']
            # ATR multiplier = optimal_SL / typical_ATR (assume 1% ATR)
            adapted['sl_atr_multiplier'] = optimal_sl_pct / 0.01

        # 3. LEVERAGE: Conservative for volatile symbols
        avg_r = profile.get('avg_r', 0)
        if avg_r < 0:  # Losing symbol
            adapted['leverage_multiplier'] = 0.7  # Reduce leverage
        elif avg_r > 50:  # Very profitable
            adapted['leverage_multiplier'] = 1.2  # Increase leverage

        # 4. HOLD TIME: Use ML-learned optimal duration
        if 'avg_hold_time' in profile:
            adapted['max_hold_minutes'] = profile['avg_hold_time'] * 1.5  # 50% buffer

        # 5. TIMING: Prefer historically good hours/days
        if 'best_hour' in profile:
            adapted['preferred_hour'] = profile['best_hour']
        if 'best_day' in profile:
            adapted['preferred_day'] = profile['best_day']

        return adapted

    def _get_symbol_profile(self, symbol: str) -> Optional[dict]:
        """Get ML-learned symbol profile."""
        import time
        from pathlib import Path
        import json

        # Cache profiles for performance
        now = time.time()
        if now - self._last_cache_update > self._cache_ttl:
            try:
                profiles_dir = Path("master_hindsight_models")
                if profiles_dir.exists():
                    model_dirs = sorted(profiles_dir.glob("*"), reverse=True)
                    if model_dirs:
                        profile_file = model_dirs[0] / "symbol_profiles.json"
                        if profile_file.exists():
                            with open(profile_file, 'r') as f:
                                self._symbol_profiles = json.load(f)
                            self._last_cache_update = now
            except Exception:
                pass  # Silent failure, use defaults

        return self._symbol_profiles.get(symbol.upper())

