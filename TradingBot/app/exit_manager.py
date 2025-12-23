"""
Exit Manager - Handles position exits with proper stop-loss, take-profit, and trailing stop logic.
Implements realistic exit strategies matching the tightened entry strategy.
"""

import time
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from .config import (
    TAKER_FEE_RATE, SLIPPAGE_BPS, DRY_RUN, DRY_SIMPLE_EXITS,
    TRAILING_STOP_ACTIVATION_PCT, TRAILING_STOP_PCT,
    WIDE_SPREAD_EXIT_THRESHOLD_BPS,
    USE_ATR_TRAILING_STOP, ATR_TRAILING_MULTIPLIER, ATR_TRAILING_MIN_DISTANCE_PCT,
    ATR_TRAILING_SCALPING_MULTIPLIER, ATR_TRAILING_DAY_MULTIPLIER, ATR_TRAILING_SWING_MULTIPLIER,
    MAX_POSITION_AGE_SEC, STALE_POSITION_PNL_THRESHOLD,
    BERP_ENABLED, BERP_TRIGGER_AGE_SEC, BERP_TRIGGER_PNL_THRESHOLD, BERP_RESCUE_DURATION_SEC,
    STALE_90MIN_AGE_SEC, STALE_90MIN_PNL_THRESHOLD, EXTENDED_LEASH_PNL_THRESHOLD,
    EXTENDED_LEASH_AGE_SEC, STALE_DRAWDOWN_RESUME_THRESHOLD,
    USE_R_BASED_EXITS, R_EXIT_SCALP_SCORE_MIN, R_EXIT_SCALP_SCORE_MAX,
    R_EXIT_STANDARD_SCORE_MIN, R_EXIT_STANDARD_SCORE_MAX, R_EXIT_RUNNER_SCORE_MIN,
    R_SCALP_TP_R, R_SCALP_TIME_STOP_BARS, R_SCALP_BOREDOM_RANGE,
    R_STANDARD_PARTIAL_TP_R, R_STANDARD_PARTIAL_PCT, R_STANDARD_TRAIL_START_R,
    R_STANDARD_TRAIL_ATR_MULT, R_STANDARD_MAX_R, R_STANDARD_TIME_STOP_BARS, R_STANDARD_BOREDOM_RANGE,
    R_RUNNER_PARTIAL_TP_R, R_RUNNER_PARTIAL_PCT, R_RUNNER_BE_MOVE_R,
    R_RUNNER_TRAIL_START_R, R_RUNNER_TRAIL_ATR_MULT_NORMAL, R_RUNNER_TRAIL_ATR_MULT_HIGH,
    R_RUNNER_MAX_R_NORMAL, R_RUNNER_MAX_R_HIGH, R_RUNNER_TIME_STOP_BARS, R_RUNNER_BOREDOM_RANGE,
    R_BAR_SCAN_CYCLE_SEC,
    USE_SWTA, SWTA_BASE_MULTIPLIER, SWTA_START_R,
    USE_MSX, MSX_STAGE1_VALIDATION_R, MSX_STAGE1_VALIDATION_BARS,
    MSX_PARTIAL_SCALP_PCT, MSX_PARTIAL_STANDARD_PCT, MSX_PARTIAL_RUNNER_PCT,
    MSX_UNICORN_BE_R, MSX_TIME_STOP_SCALP_BARS, MSX_TIME_STOP_STANDARD_BARS,
    MSX_TIME_STOP_RUNNER_BARS, MSX_TIME_STOP_MIN_R,
    # New Stage 1 controls
    MSX_STAGE1_ENABLED, MSX_EARLY_INVALIDATION_R, MSX_VOL_SPIKE_MULT, MSX_MAX_SPREAD_STAGE1,
    USE_TRAILING_ENGINE, USE_NEW_TRAILING_ENGINE,
)


@dataclass
class ExitResult:
    """Exit execution result."""
    success: bool
    exit_price: Optional[float] = None
    exit_size: Optional[float] = None
    reason: Optional[str] = None
    gross_pnl: Optional[float] = None
    total_costs: Optional[float] = None
    net_pnl: Optional[float] = None
    entry_fee: Optional[float] = None
    exit_fee: Optional[float] = None
    slippage: Optional[float] = None
    funding_cost: Optional[float] = None
    error: Optional[str] = None


class ExitManager:
    """Manages position exits with proper risk management."""
    
    def __init__(self, exchange=None, order_manager=None):
        self.exchange = exchange
        self.order_manager = order_manager
        # Early stop logic removed - relying on proper stop-loss protection instead
        
        # Initialize new trailing stop engine if enabled
        if USE_NEW_TRAILING_ENGINE:
            try:
                from .engine.trailing_engine import TrailingStopEngine
                self.trailing_engine = TrailingStopEngine()
            except ImportError:
                self.trailing_engine = None
        else:
            self.trailing_engine = None
        
        # PERFORMANCE: Pre-bind frequently used config constants
        self._scalp_tp_r = R_SCALP_TP_R
        self._scalp_time_stop_bars = R_SCALP_TIME_STOP_BARS
        self._scalp_boredom_range = R_SCALP_BOREDOM_RANGE
        self._standard_partial_tp_r = R_STANDARD_PARTIAL_TP_R
        self._standard_partial_pct = R_STANDARD_PARTIAL_PCT
        self._standard_trail_start_r = R_STANDARD_TRAIL_START_R
        self._standard_time_stop_bars = R_STANDARD_TIME_STOP_BARS
        self._standard_boredom_range = R_STANDARD_BOREDOM_RANGE
        self._runner_time_stop_bars = R_RUNNER_TIME_STOP_BARS
    
    def compute_recovery_score(
        self,
        position: Dict[str, Any],
        current_price: float,
        trend5: int = 0,
        trend15: int = 0,
        vol_regime: str = "normal"
    ) -> float:
        """
        Compute Position Recovery Score (PRS) - 0-100 scale.
        Higher = more likely to recover / worth keeping.
        Lower = stale, against trend, should be closed.
        
        Args:
            position: Position dictionary with entry_price, side, entry_time, etc.
            current_price: Current market price
            trend5: 5m trend direction (+1 uptrend, -1 downtrend, 0 neutral)
            trend15: 15m trend direction (+1 uptrend, -1 downtrend, 0 neutral)
            vol_regime: "high", "normal", or "low"
        
        Returns:
            Recovery score (0-100)
        """
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        side = position.get('side', '').lower()
        entry_time = position.get('entry_time', 0)
        peak_pnl = position.get('peak_pnl', 0.0)  # Max favorable excursion in %
        
        if not entry_price or entry_price <= 0:
            return 0.0
        
        # Calculate position age in minutes (ensure non-negative)
        age_minutes = max(0.0, (time.time() - entry_time) / 60.0) if entry_time > 0 else 0.0
        
        # Calculate current PnL percentage
        if side == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            side_mult = 1
        else:  # short
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            side_mult = -1
        
        # Calculate MFE (max favorable excursion) - use peak_pnl if available, otherwise current pnl
        mfe_pct = max(peak_pnl, pnl_pct) if peak_pnl > 0 else pnl_pct
        
        # Calculate distance to TP and SL in percentage
        if take_profit > 0:
            if side == 'long':
                dist_to_tp_pct = ((take_profit - current_price) / current_price) * 100
            else:
                dist_to_tp_pct = ((current_price - take_profit) / current_price) * 100
        else:
            dist_to_tp_pct = 10.0  # Default if no TP set
        
        if stop_loss > 0:
            if side == 'long':
                dist_to_sl_pct = ((current_price - stop_loss) / current_price) * 100
            else:
                dist_to_sl_pct = ((stop_loss - current_price) / current_price) * 100
        else:
            dist_to_sl_pct = 2.0  # Default if no SL set
        
        # Start with neutral baseline
        score = 50.0
        
        # --- Trend alignment ---
        # With-trend positions are more likely to recover, against-trend are less likely
        score += 15.0 * (side_mult * trend5)      # -15 to +15
        score += 10.0 * (side_mult * trend15)     # -10 to +10
        
        # --- Volatility regime ---
        # High vol means moves can extend; low vol means dead markets, less patience
        if vol_regime == "high":
            score += 10.0
        elif vol_regime == "low":
            score -= 15.0
        
        # --- TP vs SL asymmetry ---
        total_risk = dist_to_tp_pct + dist_to_sl_pct
        if total_risk > 0:
            tp_bias = dist_to_sl_pct / total_risk      # >0.5 → closer to TP than SL
            score += int((tp_bias - 0.5) * 50.0)       # -25 to +25
        
        # --- Nonlinear age penalty ---
        # 0–30 min: mild decay
        # 30–120 min: stronger decay
        # >120 min: aggressive decay
        if age_minutes <= 30:
            score -= age_minutes * 0.2
        elif age_minutes <= 120:
            score -= (30 * 0.2) + (age_minutes - 30) * 0.5
        else:
            score -= (30 * 0.2) + (90 * 0.5) + (age_minutes - 120) * 1.5
        
        # --- Penalize giveback: had good profit, now negative ---
        if mfe_pct > 0.8 and pnl_pct < 0:
            score -= 25.0
        
        # --- Clamp result ---
        score = max(0.0, min(100.0, score))
        return score
    
    def get_trend_direction(
        self,
        symbol: str,
        timeframe_minutes: int = 5,
        indicators: Optional[Dict] = None
    ) -> int:
        """
        Get trend direction for a symbol at a given timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe_minutes: Timeframe in minutes (5 or 15)
            indicators: Optional indicators dict with EMA values
        
        Returns:
            +1 for uptrend, -1 for downtrend, 0 for neutral
        """
        if not indicators:
            return 0
        
        # Use EMA20 and EMA50 to determine trend
        ema20 = indicators.get('ema20')
        ema50 = indicators.get('ema50')
        current_price = indicators.get('close') or indicators.get('last')
        
        if not ema20 or not ema50 or not current_price:
            return 0
        
        # Uptrend: price > EMA20 > EMA50
        # Downtrend: price < EMA20 < EMA50
        if current_price > ema20 > ema50:
            return 1
        elif current_price < ema20 < ema50:
            return -1
        else:
            return 0
    
    def get_volatility_regime(
        self,
        atr_pct: Optional[float] = None,
        indicators: Optional[Dict] = None
    ) -> str:
        """
        Determine volatility regime from ATR or indicators.
        
        Args:
            atr_pct: ATR as percentage of price
            indicators: Optional indicators dict
        
        Returns:
            "high", "normal", or "low"
        """
        if atr_pct is None:
            atr_pct = indicators.get('atr_pct') if indicators else None
        
        if atr_pct is None:
            return "normal"
        
        # High vol: > 2%, Low vol: < 0.5%
        if atr_pct > 0.02:
            return "high"
        elif atr_pct < 0.005:
            return "low"
        else:
            return "normal"
    
    def tighten_stop(
        self,
        position: Dict[str, Any],
        current_price: float,
        atr_pct: Optional[float] = None
    ) -> bool:
        """
        Tighten stop-loss for a position (move closer to current price).
        
        Args:
            position: Position dictionary
            current_price: Current market price
            atr_pct: ATR as percentage (for calculating stop distance)
        
        Returns:
            True if stop was tightened, False otherwise
        """
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        side = position.get('side', '').lower()
        
        if not entry_price:
            return False
        # For longs, we need a stop_loss to tighten. For shorts, stop_loss == 0 is handled below.
        if side == 'long' and not stop_loss:
            return False
        
        # Calculate current profit
        if side == 'long':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            # Move stop to break-even or slightly above if in profit
            if profit_pct > 0.1:  # Only tighten if in profit
                new_stop = entry_price * 1.001  # 0.1% above entry
                if new_stop > stop_loss:  # Only move stop up, never down
                    position['stop_loss'] = new_stop
                    return True
        else:  # short
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            if profit_pct > 0.1:
                new_stop = entry_price * 0.999  # 0.1% below entry
                if new_stop < stop_loss or stop_loss == 0:  # Only move stop down, never up
                    position['stop_loss'] = new_stop
                    return True
        
        return False
    
    def should_exit_position(
        self,
        position: Dict[str, Any],
        current_price: float,
        spread_bps: float = 0.0,
        regime_config: Optional[Any] = None,
        symbol: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if position should be exited.
        TIGHTENED: Uses realistic exit conditions matching entry strategy.
        Includes stale position kill-switch to prevent dead weight.
        
        Returns:
            (should_exit, reason, target_price)
        """
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        side = position.get('side', '').lower()
        entry_time = position.get('entry_time', 0)
        
        if not entry_price or entry_price <= 0:
            return False, "Invalid entry price", 0.0
        
        # Calculate current profit/loss percentage
        if side == 'long':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # short
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        # BREAK-EVEN RESCUE PROTOCOL (BERP) and EXTENDED STALE POSITION RULES
        if entry_time > 0:
            position_age_sec = time.time() - entry_time
            rescue_flag = position.get('rescue_flag', False)
            rescue_start_time = position.get('rescue_start_time', 0)
            peak_pnl = position.get('peak_pnl', profit_pct)  # Track peak PnL for drawdown detection
            
            # Update peak PnL (always track, not just in rescue mode)
            if profit_pct > peak_pnl:
                position['peak_pnl'] = profit_pct
                peak_pnl = profit_pct
            
            # BREAK-EVEN RESCUE PROTOCOL: Trigger rescue mode if conditions met
            if BERP_ENABLED and not rescue_flag:
                if position_age_sec >= BERP_TRIGGER_AGE_SEC and profit_pct < BERP_TRIGGER_PNL_THRESHOLD:
                    # Position enters rescue mode
                    position['rescue_flag'] = True
                    position['rescue_start_time'] = time.time()
                    if 'peak_pnl' not in position or profit_pct > position.get('peak_pnl', profit_pct):
                        position['peak_pnl'] = profit_pct  # Initialize peak PnL
                    rescue_flag = True
                    rescue_start_time = time.time()
                    peak_pnl = position.get('peak_pnl', profit_pct)
            
            # Evaluate rescue mode positions (BERP)
            if BERP_ENABLED and rescue_flag and rescue_start_time > 0:
                time_in_rescue = time.time() - rescue_start_time
                
                # ✅ Break-even reached - exit immediately
                if profit_pct >= 0.0:
                    return True, "rescued_at_breakeven", current_price
                
                # ❌ Rescue timeout expired - exit as failed rescue
                if time_in_rescue >= BERP_RESCUE_DURATION_SEC:
                    return True, "failed_rescue_after_60m", current_price
            
            # OLD HARD TIME-BASED EXIT RULES REMOVED
            # These have been replaced by dynamic Position Recovery Score (PRS) logic
            # which is evaluated in the monitoring loop (monitor_and_exit_positions)
            # The PRS system considers trend alignment, volatility, age, and PnL
            # to make market-aware exit decisions instead of blind time limits.
            
            # Log warning if position is very old (for debugging, but don't force exit)
            if position_age_sec >= 7200:  # 2 hours
                from .logger import get_logger
                logger = get_logger("ExitManager")
                logger.warning(
                    f"Position {symbol or 'unknown'} is very old ({position_age_sec/60:.1f} min), "
                    f"but exit decision will be made by PRS system"
                )
        
        # Get regime-specific parameters
        if regime_config:
            trailing_activation = regime_config.trailing_stop_activation_pct
            trailing_pct = regime_config.trailing_stop_pct
            wide_spread_threshold = regime_config.wide_spread_exit_threshold_bps
        else:
            trailing_activation = TRAILING_STOP_ACTIVATION_PCT
            trailing_pct = TRAILING_STOP_PCT
            wide_spread_threshold = WIDE_SPREAD_EXIT_THRESHOLD_BPS
        
        # Update trailing stop if profit exceeds activation threshold
        updated_stop_loss = stop_loss
        if profit_pct > trailing_activation:
            should_trail, new_stop = self.should_use_trailing_stop(
                position, current_price, profit_pct, regime_config
            )
            if should_trail:
                updated_stop_loss = new_stop
                # Update position's stop_loss for next check
                position['stop_loss'] = new_stop
        
        # Check stop-loss (priority - exit immediately on stop-loss)
        if updated_stop_loss > 0:
            if side == 'long':
                if current_price <= updated_stop_loss:
                    return True, "stop_loss", updated_stop_loss
            else:  # short
                if current_price >= updated_stop_loss:
                    return True, "stop_loss", updated_stop_loss
        
        # Check take-profit (exit when target reached)
        if take_profit > 0:
            if side == 'long':
                if current_price >= take_profit:
                    return True, "take_profit", take_profit
            else:  # short
                if current_price <= take_profit:
                    return True, "take_profit", take_profit
        
        # Additional exit conditions (regime-specific):
        # 1. Exit if spread widens significantly (slippage risk)
        # Only exit if spread exceeds threshold AND profit > activation threshold (protect larger profits)
        if spread_bps > wide_spread_threshold:
            # Only exit if in significant profit (protect profits from slippage)
            if profit_pct > trailing_activation:
                return True, "wide_spread", current_price
        
        # Early stop logic DISABLED - positions will run to their intended stop-loss
        # This prevents premature exits that cut winners short
        # Stop-loss protection (checked above) provides proper risk management
        
        return False, "hold", 0.0
    
    def calculate_exit_costs(
        self,
        exit_size: float,
        exit_price: float,
        entry_price: float = None,
        entry_size: float = None
    ) -> Dict[str, float]:
        """
        Calculate exit costs (entry fees + exit fees + slippage).
        
        Args:
            exit_size: Exit position size
            exit_price: Exit price
            entry_price: Entry price (optional, for entry fee calculation)
            entry_size: Entry position size (optional, for entry fee calculation)
        
        Returns:
            Dict with 'entry_fee', 'exit_fee', 'slippage', 'total'
        """
        exit_notional = exit_size * exit_price
        exit_fee = exit_notional * TAKER_FEE_RATE
        slippage = exit_notional * (SLIPPAGE_BPS / 10000)
        
        # Include entry fees if provided
        entry_fee = 0.0
        if entry_price and entry_size:
            entry_notional = entry_size * entry_price
            entry_fee = entry_notional * TAKER_FEE_RATE
        
        return {
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'slippage': slippage,
            'total': entry_fee + exit_fee + slippage
        }
    
    def calculate_funding_costs(
        self,
        position: Dict,
        exit_time: float,
        funding_rate: float = None
    ) -> float:
        """
        Calculate funding costs for a position.
        
        Binance Futures funding is paid every 8 hours.
        Formula: funding_cost = position_size * entry_price * funding_rate * (hours_held / 8)
        
        Args:
            position: Position dictionary with entry_time, size, entry_price
            exit_time: Exit timestamp
            funding_rate: Funding rate per 8 hours (optional, will use position funding_rate if not provided)
        
        Returns:
            Total funding cost
        """
        entry_time = position.get('entry_time', 0)
        position_size = position.get('size', 0)
        entry_price = position.get('entry_price', 0)
        
        if not entry_time or not position_size or not entry_price:
            return 0.0
        
        # Use provided funding_rate or get from position
        if funding_rate is None:
            funding_rate = position.get('funding_rate', 0.0)
        
        if not funding_rate:
            return 0.0
        
        # Calculate hours held
        hours_held = (exit_time - entry_time) / 3600.0
        
        if hours_held <= 0:
            return 0.0
        
        # Funding is paid every 8 hours
        funding_periods = hours_held / 8.0
        
        # Calculate funding cost
        notional = position_size * entry_price
        funding_cost = notional * funding_rate * funding_periods
        
        return funding_cost
    
    def calculate_pnl(
        self,
        position: Dict,
        exit_price: float,
        exit_size: float,
        total_costs: float
    ) -> Tuple[float, float, float]:
        """
        Calculate PnL for position exit.
        
        Returns:
            (gross_pnl, total_costs, net_pnl)
        """
        entry_price = position.get('entry_price', 0)
        side = position.get('side', '').lower()
        
        if not entry_price or entry_price <= 0:
            return 0.0, total_costs, -total_costs
        
        # Calculate gross PnL
        if side == 'long':
            price_diff = exit_price - entry_price
        else:  # short
            price_diff = entry_price - exit_price
        
        gross_pnl = price_diff * exit_size
        
        # Net PnL = Gross PnL - Total Costs
        net_pnl = gross_pnl - total_costs
        
        return gross_pnl, total_costs, net_pnl
    
    async def exit_position(
        self,
        symbol: str,
        position: Dict,
        reason: str,
        target_price: float = None,
        use_limit: bool = False,
        funding_rate: float = None
    ) -> ExitResult:
        """
        Exit a position.
        
        Args:
            symbol: Trading symbol
            position: Position dictionary
            reason: Exit reason ('stop_loss', 'take_profit', 'manual', etc.)
            target_price: Target exit price (if None, uses market price)
            use_limit: Use limit order instead of market
        
        Returns:
            ExitResult
        """
        if not self.order_manager:
            return ExitResult(
                success=False,
                error="Order manager not initialized"
            )
        
        side = position.get('side', '').lower()
        position_size = position.get('size', 0)
        
        # EDGE CASE HARDENING: Validate position state before exit
        if position_size <= 0:
            return ExitResult(
                success=False,
                error="Invalid position size (zero or negative)"
            )
        
        # Check if position is already closed (defensive check)
        if symbol and hasattr(self, 'exchange') and self.exchange:
            # In live mode, could verify with exchange, but in DRY_RUN we trust internal state
            pass
        
        # Validate entry price exists
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0:
            return ExitResult(
                success=False,
                error="Invalid entry price (zero or negative)"
            )
        
        # Determine exit side (opposite of entry)
        exit_side = "sell" if side == "long" else "buy"
        
        # For LIMIT orders: target_price is REQUIRED and must be valid
        # For MARKET orders: target_price can be None (we'll fetch market price for PnL calculation only)
        ticker_data = None  # Cache ticker data to avoid duplicate fetches
        
        if use_limit:
            # LIMIT ORDER: Validate target_price is required and valid
            if target_price is None or target_price <= 0:
                # Invalid limit price - skip with clear message
                return ExitResult(
                    success=False,
                    error="invalid_limit_price"
                )
            
            # Get current market price for validation (defensive clamp)
            current_market_price = None
            try:
                if hasattr(self.exchange, 'fetch_ticker') and self.exchange:
                    ticker_data = await self.exchange.fetch_ticker(symbol)
                    if side == "long":
                        current_market_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                    else:
                        current_market_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
            except Exception:
                # If we can't get market price, use entry price as fallback for validation
                current_market_price = entry_price
            
            # Defensive clamp: Validate target_price is on correct side of market
            if current_market_price and current_market_price > 0:
                if side == "long":
                    # Long exit: target_price should be >= current_market_price (selling higher)
                    if target_price < current_market_price * 0.99:  # Allow 1% tolerance for slippage
                        # Price is on wrong side - downgrade to market exit
                        use_limit = False
                        # Reuse ticker data if available
                        if ticker_data:
                            if side == "long":
                                target_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                            else:
                                target_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
                        else:
                            target_price = None  # Will fetch market price below
                else:
                    # Short exit: target_price should be <= current_market_price (buying lower)
                    if target_price > current_market_price * 1.01:  # Allow 1% tolerance for slippage
                        # Price is on wrong side - downgrade to market exit
                        use_limit = False
                        # Reuse ticker data if available
                        if ticker_data:
                            if side == "long":
                                target_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                            else:
                                target_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
                        else:
                            target_price = None  # Will fetch market price below
        
        # For MARKET orders or after downgrade: Fetch market price for PnL calculation
        # Note: Market orders don't need target_price for execution, but we need it for PnL
        if not use_limit and (target_price is None or target_price <= 0):
            # Fetch current market price for PnL calculation (not needed for order execution)
            # Reuse ticker_data if already fetched above
            if ticker_data:
                if side == "long":
                    target_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                else:
                    target_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
            else:
                try:
                    if hasattr(self.exchange, 'fetch_ticker') and self.exchange:
                        ticker_data = await self.exchange.fetch_ticker(symbol)
                        if side == "long":
                            target_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                        else:
                            target_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
                    else:
                        # Fallback: use entry price (for DRY_RUN or when exchange unavailable)
                        target_price = entry_price
                except Exception as e:
                    # Last resort: use entry price fallback to allow exit to proceed
                    # (Better to exit with estimated PnL than to fail the exit entirely)
                    target_price = entry_price
                    from .logger import get_logger
                    get_logger("ExitManager").warning(
                        f"Market price unavailable for {symbol} exit. Using entry price {entry_price} as fallback to ensure execution."
                    )
        
        # Final validation: target_price must be valid for PnL calculation
        if target_price is None or target_price <= 0:
            # If we still don't have a valid price, try one more fallback
            # This can happen if entry_price is also invalid (shouldn't happen, but defensive)
            if entry_price and entry_price > 0:
                target_price = entry_price
                from .logger import get_logger
                logger = get_logger("ExitManager")
                logger.warning(
                    f"Using entry_price as fallback for {symbol} PnL calculation "
                    f"(market price unavailable)"
                )
            else:
                return ExitResult(
                    success=False,
                    error="invalid_price"
                )
        
        # Execute exit order
        try:
            if DRY_RUN:
                # Simulate exit
                exit_result = ExitResult(
                    success=True,
                    exit_price=target_price,
                    exit_size=position_size,
                    reason=reason
                )
            else:
                # CANCEL BINANCE TRAILING STOP (if any)
                # When the bot closes a position, we need to cancel any trailing stop
                # to avoid a double-close attempt by Binance
                try:
                    if hasattr(self.exchange, 'cancel_trailing_stop'):
                        await self.exchange.cancel_trailing_stop(symbol)
                except Exception:
                    pass  # Non-critical - continue with exit
                
                # Real exit order
                # CRITICAL: Format quantity to exchange precision before placing order
                # This prevents precision errors like "872.7468749999999" which Binance rejects
                # The exchange wrapper (binance_futures.create_order) handles precision, but we ensure it here too
                try:
                    # Get the inner exchange if we have a wrapper (for precision methods)
                    inner_exchange = None
                    if hasattr(self.exchange, '_inner'):
                        inner_exchange = self.exchange._inner
                    elif hasattr(self.exchange, 'exchange'):
                        inner_exchange = getattr(self.exchange, 'exchange', None)
                    
                    # Format quantity using exchange's precision settings
                    if inner_exchange and hasattr(inner_exchange, 'amount_to_precision'):
                        formatted_quantity = inner_exchange.amount_to_precision(symbol, abs(position_size))
                        final_quantity = float(formatted_quantity)
                    else:
                        # Fallback: round to reasonable precision (8 decimals)
                        final_quantity = round(abs(position_size), 8)
                    
                    # Ensure quantity is not zero after formatting
                    if final_quantity <= 0:
                        return ExitResult(
                            success=False,
                            error=f"Exit quantity {position_size} formatted to zero (too small)"
                        )
                except Exception as e:
                    # If formatting fails, use absolute value and round
                    final_quantity = round(abs(position_size), 8)
                    if final_quantity <= 0:
                        return ExitResult(
                            success=False,
                            error=f"Could not format exit quantity: {e}"
                        )
                
                # OPTIMIZATION: Binance Futures USDT-M doesn't use positionSide
                # Use reduceOnly=True for exits (Binance-specific safety parameter)
                # This ensures the order only reduces position, never increases it
                exit_params = {"reduceOnly": True}  # Binance Futures safety parameter
                
                if use_limit:
                    # Format price to exchange precision
                    try:
                        if inner_exchange and hasattr(inner_exchange, 'price_to_precision'):
                            formatted_price = inner_exchange.price_to_precision(symbol, target_price)
                            final_price = float(formatted_price)
                        else:
                            final_price = target_price
                    except Exception:
                        final_price = target_price
                    
                    order = await self.exchange.create_order(
                        symbol,
                        "limit",
                        exit_side,
                        final_quantity,
                        final_price,
                        params=exit_params
                    )
                else:
                    # OPTIMIZATION: Market orders for exits are faster
                    # Note: create_order in binance_futures.py will also format the quantity
                    # This double-formatting is safe and ensures precision
                    order = await self.exchange.create_order(
                        symbol,
                        "market",
                        exit_side,
                        final_quantity,
                        None,
                        params=exit_params
                    )
                
                # Get filled price
                filled_price = order.get('price') or order.get('average') or target_price
                filled_size = order.get('filled', position_size)
                
                exit_result = ExitResult(
                    success=True,
                    exit_price=filled_price,
                    exit_size=filled_size,
                    reason=reason
                )
            
            # Calculate costs and PnL (including entry fees)
            entry_price = position.get('entry_price', 0)
            entry_size = position.get('size', 0)  # Use original entry size
            costs = self.calculate_exit_costs(
                exit_result.exit_size,
                exit_result.exit_price,
                entry_price=entry_price,
                entry_size=entry_size
            )
            
            # Calculate funding costs (use provided funding_rate or from position)
            funding_cost = self.calculate_funding_costs(position, time.time(), funding_rate=funding_rate)
            
            # Total costs = entry fee + exit fee + slippage + funding
            total_costs = costs['total'] + funding_cost
            
            gross_pnl, _, net_pnl = self.calculate_pnl(
                position,
                exit_result.exit_price,
                exit_result.exit_size,
                total_costs
            )
            
            exit_result.gross_pnl = gross_pnl
            exit_result.total_costs = total_costs
            exit_result.net_pnl = net_pnl
            exit_result.entry_fee = costs['entry_fee']
            exit_result.exit_fee = costs['exit_fee']
            exit_result.slippage = costs['slippage']
            exit_result.funding_cost = funding_cost
            
            return exit_result
            
        except Exception as e:
            return ExitResult(
                success=False,
                error=str(e)
            )
    
    def calculate_atr_trailing_stop(
        self,
        position: Dict,
        current_price: float,
        regime_config: Optional[Any] = None
    ) -> Optional[float]:
        """
        Calculate ATR-based trailing stop.
        
        Returns:
            New stop price based on ATR, or None if ATR not available
        """
        entry_price = position.get('entry_price', 0)
        side = position.get('side', '').lower()
        atr_pct = position.get('atr_pct')  # ATR as percentage of price
        
        if not entry_price or atr_pct is None or atr_pct <= 0:
            return None
        
        # Get regime-specific ATR multiplier
        if regime_config:
            from .regime import TradingRegime
            regime_type = regime_config.regime_type
            if regime_type == TradingRegime.SCALPING:
                atr_multiplier = ATR_TRAILING_SCALPING_MULTIPLIER
            elif regime_type == TradingRegime.SWING_TRADING:
                atr_multiplier = ATR_TRAILING_SWING_MULTIPLIER
            else:  # DAY_TRADING
                atr_multiplier = ATR_TRAILING_DAY_MULTIPLIER
        else:
            atr_multiplier = ATR_TRAILING_MULTIPLIER
        
        # Get current peak (long) or trough (short) price
        # Peak/trough should be updated in bot.py before calling this method
        if side == 'long':
            peak_price = position.get('peak_price', max(entry_price, current_price))
            # Calculate ATR-based stop: peak - (ATR × multiplier)
            atr_distance = atr_pct * atr_multiplier
            atr_stop = peak_price * (1.0 - atr_distance)
            # Ensure minimum distance from entry
            min_stop = entry_price * (1.0 - ATR_TRAILING_MIN_DISTANCE_PCT)
            return max(atr_stop, min_stop)
        else:  # short
            trough_price = position.get('trough_price', min(entry_price, current_price))
            # Calculate ATR-based stop: trough + (ATR × multiplier)
            atr_distance = atr_pct * atr_multiplier
            atr_stop = trough_price * (1.0 + atr_distance)
            # Ensure minimum distance from entry
            max_stop = entry_price * (1.0 + ATR_TRAILING_MIN_DISTANCE_PCT)
            return min(atr_stop, max_stop)
    
    def should_use_trailing_stop(
        self,
        position: Dict,
        current_price: float,
        profit_pct: float,
        regime_config: Optional[Any] = None
    ) -> Tuple[bool, float]:
        """
        Determine if trailing stop should be activated.
        Uses ATR-based trailing if enabled and ATR available, otherwise falls back to percentage-based.
        
        Returns:
            (should_activate, new_stop_price)
        """
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        side = position.get('side', '').lower()
        
        if not entry_price or not stop_loss:
            return False, stop_loss
        
        # Get regime-specific parameters
        if regime_config:
            trailing_activation = regime_config.trailing_stop_activation_pct
            trailing_pct = regime_config.trailing_stop_pct
        else:
            trailing_activation = TRAILING_STOP_ACTIVATION_PCT
            trailing_pct = TRAILING_STOP_PCT
        
        # Activate trailing stop when profit exceeds activation threshold
        if profit_pct < trailing_activation:
            return False, stop_loss
        
        # Try ATR-based trailing stop first (if enabled and ATR available)
        if USE_ATR_TRAILING_STOP:
            atr_stop = self.calculate_atr_trailing_stop(position, current_price, regime_config)
            if atr_stop is not None:
                # Update peak/trough in position for next check
                if side == 'long':
                    position['peak_price'] = max(position.get('peak_price', entry_price), current_price)
                    # Use the better (higher) stop for long
                    if atr_stop > stop_loss:
                        return True, atr_stop
                else:  # short
                    position['trough_price'] = min(position.get('trough_price', entry_price), current_price)
                    # Use the better (lower) stop for short
                    if atr_stop < stop_loss:
                        return True, atr_stop
        
        # Fallback to percentage-based trailing stop
        if side == 'long':
            # Trail stop up to configured percentage of profit
            profit_amount = current_price - entry_price
            trail_amount = profit_amount * trailing_pct
            new_stop = entry_price + trail_amount
            # Don't move stop down
            if new_stop > stop_loss:
                return True, new_stop
        else:  # short
            # Trail stop down to configured percentage of profit
            profit_amount = entry_price - current_price
            trail_amount = profit_amount * trailing_pct
            new_stop = entry_price - trail_amount
            # Don't move stop up
            if new_stop < stop_loss:
                return True, new_stop
        
        return False, stop_loss
    
    def calculate_r_multiple(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> float:
        """
        Calculate current R-multiple for a position.
        
        R = distance from entry to SL in price
        R_multiple = (current_price - entry_price) / R * direction_sign
        
        Args:
            position: Position dictionary
            current_price: Current market price
        
        Returns:
            R-multiple (positive = profit, negative = loss)
        """
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        side = position.get('side', '').lower()
        initial_r = position.get('initial_r', 0)
        
        if not entry_price or not stop_loss or initial_r <= 0:
            # Fallback: calculate R from entry and stop_loss
            if side == 'long':
                initial_r = abs(entry_price - stop_loss)
            else:  # short
                initial_r = abs(stop_loss - entry_price)
        
        if initial_r <= 0:
            return 0.0
        
        # Calculate R-multiple
        if side == 'long':
            price_diff = current_price - entry_price
        else:  # short
            price_diff = entry_price - current_price
        
        r_multiple = price_diff / initial_r
        return r_multiple
    
    def select_exit_profile(self, signal_score: float) -> str:
        """
        Select exit profile based on signal score.
        
        CRITICAL FIX: Use scalp profile for low-quality signals (< 60) to exit quickly.
        
        Args:
            signal_score: Signal score (0-100)
        
        Returns:
            Exit profile: "scalp", "standard", or "runner"
        """
        if signal_score >= R_EXIT_RUNNER_SCORE_MIN:
            # 90+: Runner profile (let winners run)
            return "runner"
        elif signal_score >= R_EXIT_STANDARD_SCORE_MIN:
            # 70-89: Standard profile (balanced)
            return "standard"
        elif signal_score >= R_EXIT_SCALP_SCORE_MIN:
            # 60-69: Scalp profile (quick exits)
            return "scalp"
        else:
            # CRITICAL FIX: Scores below 60 use scalp (not standard) for faster exits
            # Low-quality signals should exit quickly, not wait for standard targets
            return "scalp"
    
    def update_position_r_metadata(
        self,
        position: Dict[str, Any],
        current_price: float,
        bar_closed: bool = False
    ) -> None:
        """
        Update R-based position metadata (max_r_reached, bars_in_trade).
        
        Args:
            position: Position dictionary (modified in place)
            current_price: Current market price
            bar_closed: Whether a bar/scan cycle has closed
        """
        r_multiple = self.calculate_r_multiple(position, current_price)
        
        # Update max R reached
        max_r_reached = position.get('max_r_reached', 0.0)
        position['max_r_reached'] = max(max_r_reached, r_multiple)
        
        # Update bars in trade (increment on bar close)
        if bar_closed:
            bars_in_trade = position.get('bars_in_trade', 0)
            bars_in_trade += 1
            position['bars_in_trade'] = bars_in_trade
            if bars_in_trade >= 1:
                position['survived_msx1'] = True
    
    def handle_scalp_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        r_multiple: float
    ) -> Tuple[bool, str, float, Optional[float]]:
        """
        Handle scalp profile exit (60-69 score).
        
        Returns:
            (should_exit, reason, target_price, exit_size_pct)
        """
        # PERFORMANCE: Use pre-bound config constants
        # Check TP at 1.0R
        if r_multiple >= self._scalp_tp_r:
            return True, "scalp_tp_1r", current_price, 1.0
        
        # Check stop-loss (1.0R loss)
        if r_multiple <= -1.0:
            return True, "scalp_sl_1r", current_price, 1.0
        
        # Check time-stop / boredom exit
        bars_in_trade = position.get('bars_in_trade', 0)
        if bars_in_trade >= self._scalp_time_stop_bars:
            # Check if trade stayed within boredom range
            max_r_reached = position.get('max_r_reached', 0.0)
            if abs(max_r_reached) <= self._scalp_boredom_range:
                return True, "scalp_time_stop", current_price, 1.0
        
        return False, "hold", 0.0, None
    
    def handle_standard_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        r_multiple: float,
        atr_pct: Optional[float] = None
    ) -> Tuple[bool, str, float, Optional[float]]:
        """
        Handle standard profile exit (70-85 score).
        
        Returns:
            (should_exit, reason, target_price, exit_size_pct)
        """
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        side = position.get('side', '').lower()
        partial_exit_done = position.get('partial_exit_done', False)
        sl_moved_to_be = position.get('sl_moved_to_be', False)
        initial_r = position.get('initial_r', 0)
        
        # Calculate initial R if not set
        if initial_r <= 0:
            if side == 'long':
                initial_r = abs(entry_price - stop_loss)
            else:
                initial_r = abs(stop_loss - entry_price)
            position['initial_r'] = initial_r
        
        # Check stop-loss (1.0R loss)
        if r_multiple <= -1.0:
            return True, "standard_sl_1r", current_price, 1.0
        
        # PERFORMANCE: Use pre-bound config constants
        # Partial TP at 1.0R (close 50%)
        if not partial_exit_done and r_multiple >= self._standard_partial_tp_r:
            # Move SL to breakeven when partial TP hit
            if not sl_moved_to_be:
                position['sl_moved_to_be'] = True
                position['stop_loss'] = entry_price  # Move to breakeven
            position['partial_exit_done'] = True
            return True, "standard_partial_1r", current_price, self._standard_partial_pct
        
        # Trailing stop (start at 1.5R) - legacy fallback
        if (not USE_TRAILING_ENGINE) and r_multiple >= R_STANDARD_TRAIL_START_R:
            if atr_pct and atr_pct > 0:
                if side == 'long':
                    peak_price = position.get('peak_price', max(entry_price, current_price))
                    atr_distance = peak_price * atr_pct * R_STANDARD_TRAIL_ATR_MULT
                    new_stop = peak_price - atr_distance
                    if new_stop > position.get('stop_loss', stop_loss):
                        position['stop_loss'] = new_stop
                else:  # short
                    trough_price = position.get('trough_price', min(entry_price, current_price))
                    atr_distance = trough_price * atr_pct * R_STANDARD_TRAIL_ATR_MULT
                    new_stop = trough_price + atr_distance
                    if new_stop < position.get('stop_loss', stop_loss):
                        position['stop_loss'] = new_stop
            
            updated_stop = position.get('stop_loss', stop_loss)
            if side == 'long' and current_price <= updated_stop:
                return True, "standard_trailing_stop", updated_stop, 1.0
            elif side == 'short' and current_price >= updated_stop:
                return True, "standard_trailing_stop", updated_stop, 1.0
            
            if r_multiple >= R_STANDARD_MAX_R:
                return True, "standard_max_r", current_price, 1.0
        elif r_multiple >= R_STANDARD_MAX_R:
            return True, "standard_max_r", current_price, 1.0
        
        # PERFORMANCE: Use pre-bound config constants
        # Time-stop / boredom exit
        bars_in_trade = position.get('bars_in_trade', 0)
        if bars_in_trade >= self._standard_time_stop_bars:
            max_r_reached = position.get('max_r_reached', 0.0)
            if abs(max_r_reached) <= self._standard_boredom_range:
                return True, "standard_time_stop", current_price, 1.0
        
        return False, "hold", 0.0, None
    
    def handle_runner_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        r_multiple: float,
        atr_pct: Optional[float] = None,
        is_high_volatility: bool = False
    ) -> Tuple[bool, str, float, Optional[float]]:
        """
        Handle runner profile exit (90+ score).
        
        Returns:
            (should_exit, reason, target_price, exit_size_pct)
        """
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        side = position.get('side', '').lower()
        partial_exit_done = position.get('partial_exit_done', False)
        sl_moved_to_be = position.get('sl_moved_to_be', False)
        initial_r = position.get('initial_r', 0)
        
        # Calculate initial R if not set
        if initial_r <= 0:
            if side == 'long':
                initial_r = abs(entry_price - stop_loss)
            else:
                initial_r = abs(stop_loss - entry_price)
            position['initial_r'] = initial_r
        
        # Check stop-loss (1.0R loss)
        if r_multiple <= -1.0:
            return True, "runner_sl_1r", current_price, 1.0
        
        # Small partial at 1.0R (close 25%)
        if not partial_exit_done and r_multiple >= R_RUNNER_PARTIAL_TP_R:
            # Move SL to breakeven or +0.25R
            if not sl_moved_to_be:
                position['sl_moved_to_be'] = True
                if side == 'long':
                    be_price = entry_price + (initial_r * R_RUNNER_BE_MOVE_R)
                else:
                    be_price = entry_price - (initial_r * R_RUNNER_BE_MOVE_R)
                position['stop_loss'] = be_price
            position['partial_exit_done'] = True
            return True, "runner_partial_1r", current_price, R_RUNNER_PARTIAL_PCT
        
        # Trailing stop (start at 1.5R) - legacy fallback
        if (not USE_TRAILING_ENGINE) and r_multiple >= R_RUNNER_TRAIL_START_R:
            atr_mult = R_RUNNER_TRAIL_ATR_MULT_HIGH if is_high_volatility else R_RUNNER_TRAIL_ATR_MULT_NORMAL
            
            if atr_pct and atr_pct > 0:
                if side == 'long':
                    peak_price = position.get('peak_price', max(entry_price, current_price))
                    atr_distance = peak_price * atr_pct * atr_mult
                    new_stop = peak_price - atr_distance
                    if new_stop > position.get('stop_loss', stop_loss):
                        position['stop_loss'] = new_stop
                else:  # short
                    trough_price = position.get('trough_price', min(entry_price, current_price))
                    atr_distance = trough_price * atr_pct * atr_mult
                    new_stop = trough_price + atr_distance
                    if new_stop < position.get('stop_loss', stop_loss):
                        position['stop_loss'] = new_stop
            
            updated_stop = position.get('stop_loss', stop_loss)
            if side == 'long' and current_price <= updated_stop:
                return True, "runner_trailing_stop", updated_stop, 1.0
            elif side == 'short' and current_price >= updated_stop:
                return True, "runner_trailing_stop", updated_stop, 1.0
        
        max_r = R_RUNNER_MAX_R_HIGH if is_high_volatility else R_RUNNER_MAX_R_NORMAL
        if r_multiple >= max_r:
            return True, "runner_max_r", current_price, 1.0
        
        # PERFORMANCE: Use pre-bound config constants
        # Time-stop (much laxer for runners)
        bars_in_trade = position.get('bars_in_trade', 0)
        if bars_in_trade >= self._runner_time_stop_bars:
            # Only exit if trade never exceeded +0.3R after entry phase
            max_r_reached = position.get('max_r_reached', 0.0)
            if max_r_reached < R_RUNNER_BOREDOM_RANGE:
                return True, "runner_time_stop", current_price, 1.0
        
        return False, "hold", 0.0, None
    
    def should_exit_position_r_based(
        self,
        position: Dict[str, Any],
        current_price: float,
        atr_pct: Optional[float] = None,
        is_high_volatility: bool = False,
        bar_closed: bool = False,
        now_ts: Optional[float] = None
    ) -> Tuple[bool, str, float, Optional[float]]:
        """
        R-based exit decision engine.
        
        Args:
            position: Position dictionary
            current_price: Current market price
            atr_pct: ATR as percentage of price (optional)
            is_high_volatility: Whether market is in high volatility regime
            bar_closed: Whether a bar/scan cycle has closed
        
        Returns:
            (should_exit, reason, target_price, exit_size_pct)
            exit_size_pct: Percentage of position to exit (1.0 = full exit, 0.5 = 50%, etc.)
        """
        # Update R metadata
        self.update_position_r_metadata(position, current_price, bar_closed)
        
        # DRY_RUN simple exits: bypass trailing/partial logic
        if DRY_RUN and DRY_SIMPLE_EXITS:
            # In DRY simple mode, exits are handled by evaluate_scalper_trailing
            # Just check hard SL here as fallback
            stop_loss = position.get('stop_loss')
            if stop_loss:
                side = position.get('side', '').lower()
                if side == 'long' and current_price <= stop_loss:
                    return True, "stop_loss_hit", current_price, 1.0
                elif side == 'short' and current_price >= stop_loss:
                    return True, "stop_loss_hit", current_price, 1.0
            # Otherwise let scalper_exits handle simple DRY exits
            return False, None, None, None
        
        # NEW: Check new trailing stop engine first (gets priority)
        if USE_NEW_TRAILING_ENGINE and self.trailing_engine:
            side = position.get('side', '').lower()
            atr_value = (atr_pct * current_price) if atr_pct else None
            
            market_info = {
                'current_price': current_price,
                'atr': atr_value,
                'side': side
            }
            
            trailing_result = self.trailing_engine.update(position, market_info)
            
            if trailing_result:
                if trailing_result.action == "update_sl":
                    # Update stop loss in position
                    position['stop_loss'] = trailing_result.new_stop
                    # Log SL update (minimal, no spam)
                    from .logger import get_logger
                    logger = get_logger("ExitManager")
                    logger.info(
                        f"TRAIL_SL_UPDATE {position.get('symbol', '?')} "
                        f"SL->{trailing_result.new_stop:.4f} "
                        f"R={trailing_result.current_r:.2f} "
                        f"peak_R={trailing_result.peak_r:.2f}"
                    )
                    # Don't exit, just update SL
                    return False, None, None, None
                
                elif trailing_result.action == "exit":
                    # Trailing stop triggered exit
                    return True, trailing_result.reason, trailing_result.price, trailing_result.exit_pct
        
        # Calculate current R-multiple
        r_multiple = self.calculate_r_multiple(position, current_price)
        
        # Get exit profile
        exit_profile = position.get('exit_profile', 'standard')
        
        if now_ts is None:
            now_ts = time.time()
        # Use MSX framework if enabled, otherwise use legacy handlers
        if USE_MSX:
            atr_value = (atr_pct * current_price) if atr_pct else None
            ohlc = {
                "close": current_price,
                "atr": atr_value,
            }
            return self.handle_msx_exit(position, ohlc, now_ts)
        else:
            # Legacy exit handlers
            if exit_profile == 'scalp':
                return self.handle_scalp_exit(position, current_price, r_multiple)
            elif exit_profile == 'runner':
                return self.handle_runner_exit(position, current_price, r_multiple, atr_pct, is_high_volatility)
            else:  # standard
                return self.handle_standard_exit(position, current_price, r_multiple, atr_pct)
    
    def calculate_swta_trail_distance(
        self,
        position: Dict[str, Any],
        atr_pct: float
    ) -> float:
        """
        Calculate SWTA (Score-Weighted Trailing Algorithm) trail distance.
        
        Formula: trail_multiplier = 2.0 - (final_score / 100)
        trail_distance = ATR × trail_multiplier
        
        Args:
            position: Position dictionary
            atr_pct: ATR as percentage of price
        
        Returns:
            Trail distance as percentage of price
        """
        if not USE_SWTA or not atr_pct or atr_pct <= 0:
            return 0.0
        
        # Get signal score from position
        signal_score = position.get('signal_score', 0.0)
        if signal_score <= 0:
            # Fallback: estimate from exit_profile (ML SCORER: 25-50 range)
            exit_profile = position.get('exit_profile', 'standard')
            if exit_profile == 'scalp':
                signal_score = 35.0  # ML: mid-range for scalp
            elif exit_profile == 'runner':
                signal_score = 50.0  # ML: high for runner
            else:
                signal_score = 40.0  # ML: mid-range for standard
        
        # Calculate trail multiplier: 2.0 - (score / 100)
        trail_multiplier = SWTA_BASE_MULTIPLIER - (signal_score / 100.0)
        
        # Calculate trail distance
        trail_distance = atr_pct * trail_multiplier
        
        return trail_distance
    
    def handle_msx_exit(
        self,
        position: Dict[str, Any],
        ohlc: Dict[str, Any],
        now_ts: float
    ) -> Tuple[bool, str, float, Optional[float]]:
        """
        Multi-Stage Exit Framework (MSX) handler (state machine).
        Returns (should_exit, reason, price, partial_pct)
        """
        entry_price = position.get("entry_price", 0.0)
        stop_loss = position.get("stop_loss", 0.0)
        side = position.get("side", "long").lower()
        signal_score = position.get("signal_score", 0.0)
        exit_profile = position.get("exit_profile", "standard")
        is_unicorn = position.get("is_unicorn", False)
        take_profit = position.get("take_profit", 0.0)

        current_price = ohlc.get("close", entry_price)
        atr_value = ohlc.get("atr")

        initial_r = position.get("initial_r", abs(entry_price - stop_loss))
        if initial_r <= 0:
            initial_r = max(1e-9, abs(entry_price - stop_loss))
            position["initial_r"] = initial_r

        if side == "long":
            r_multiple = (current_price - entry_price) / initial_r
        else:
            r_multiple = (entry_price - current_price) / initial_r

        prev_max_r = position.get("max_r_reached", 0.0)
        position["max_r_reached"] = max(prev_max_r, r_multiple)

        if side == "long":
            position["peak_price"] = max(position.get("peak_price", entry_price), current_price)
        else:
            position["trough_price"] = min(position.get("trough_price", entry_price), current_price)

        stage = position.setdefault("stage", 1)
        bars_in_trade = position.get("bars_in_trade", 0)

        # Stage 1 - survive initial validation
        if stage == 1:
            if position.get("survived_msx1") or bars_in_trade >= 1:
                position["stage"] = 2
                return False, "msx_stage1_complete", 0.0, None
            return False, "msx_stage1_wait", 0.0, None

        # Stage 2 - partial profit
        if stage == 2:
            partial_done = position.get("partial_exit_done", False)
            # CRITICAL FIX: Use config thresholds instead of hardcoded values
            if exit_profile == "scalp" or (R_EXIT_SCALP_SCORE_MIN <= signal_score <= R_EXIT_SCALP_SCORE_MAX):
                partial_trigger = 0.8
                partial_pct = MSX_PARTIAL_SCALP_PCT
            elif is_unicorn or signal_score >= R_EXIT_RUNNER_SCORE_MIN:
                partial_trigger = 1.2
                partial_pct = MSX_PARTIAL_RUNNER_PCT
            else:
                partial_trigger = 1.0
                partial_pct = MSX_PARTIAL_STANDARD_PCT

            if not partial_done and r_multiple >= partial_trigger:
                position["partial_exit_done"] = True
                position["stage"] = 3
                return True, f"msx_stage2_partial_{partial_trigger:.1f}r", current_price, partial_pct
            return False, "hold", 0.0, None

        # Stage 3 - move stop to break-even
        if stage == 3:
            if not position.get("sl_moved_to_be", False) and r_multiple >= 1.0:
                position["sl_moved_to_be"] = True
                position["stop_loss"] = entry_price
                position["stage"] = 4
                return False, "msx_stage3_be_moved", entry_price, None
            return False, "hold", 0.0, None

        # Stage 4 - SWTA trailing
        if stage == 4:
            if atr_value is not None and atr_value > 0:
                base_mult = SWTA_BASE_MULTIPLIER
                trail_mult = max(0.3, base_mult - (signal_score / 120.0))
                trail_distance = trail_mult * atr_value

                updated = False
                if side == "long":
                    peak_price = position.get("peak_price", max(entry_price, current_price))
                    new_stop = peak_price - trail_distance
                    if new_stop > position.get("stop_loss", stop_loss):
                        position["stop_loss"] = new_stop
                        updated = True
                else:
                    trough_price = position.get("trough_price", min(entry_price, current_price))
                    new_stop = trough_price + trail_distance
                    if new_stop < position.get("stop_loss", stop_loss):
                        position["stop_loss"] = new_stop
                        updated = True

                if updated:
                    return False, "msx_stage4_swta_trail_update", position["stop_loss"], None

            if r_multiple >= 2.0:
                position["stage"] = 5
            return False, "hold", 0.0, None

        # Stage 5 - extended trailing / time stop
        if stage == 5:
            if bars_in_trade >= MSX_TIME_STOP_RUNNER_BARS:
                return True, "msx_stage5_time_stop", current_price, 1.0

            if atr_value is not None and atr_value > 0:
                base_mult = SWTA_BASE_MULTIPLIER
                trail_mult = max(0.3, base_mult - (signal_score / 120.0))
                trail_distance = trail_mult * atr_value
                updated = False
                if side == "long":
                    peak_price = position.get("peak_price", max(entry_price, current_price))
                    new_stop = peak_price - trail_distance
                    if new_stop > position.get("stop_loss", stop_loss):
                        position["stop_loss"] = new_stop
                        updated = True
                else:
                    trough_price = position.get("trough_price", min(entry_price, current_price))
                    new_stop = trough_price + trail_distance
                    if new_stop < position.get("stop_loss", stop_loss):
                        position["stop_loss"] = new_stop
                        updated = True
                if updated:
                    return False, "msx_stage5_swta_trail_update", position["stop_loss"], None
            return False, "hold", 0.0, None

        return False, "hold", 0.0, None

