"""
Unified Exit Pipeline - CANONICAL exit path for all position exits.
This is the ONLY way exits should happen in the system.

All exit types route through here:
- R-based exits (scalp/standard/runner)
- PRS full exits
- PRS scale-outs
- Timeout exits
- ATR-based trails
- Stop-loss/take-profit
- Stale position rules
- BERP exits
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

from .. import config as cfg
from ..logger import get_logger
from ..order_manager import OrderManager
from ..exit_manager import ExitResult
from ..models import Position, Trade

# Import config constants needed for exit logic
from ..config import (
    TAKER_FEE_RATE, SLIPPAGE_BPS,
    TRAILING_STOP_ACTIVATION_PCT, TRAILING_STOP_PCT,
    WIDE_SPREAD_EXIT_THRESHOLD_BPS,
    USE_ATR_TRAILING_STOP, ATR_TRAILING_MULTIPLIER, ATR_TRAILING_MIN_DISTANCE_PCT,
    ATR_TRAILING_SCALPING_MULTIPLIER, ATR_TRAILING_DAY_MULTIPLIER, ATR_TRAILING_SWING_MULTIPLIER,
    MAX_POSITION_AGE_SEC, STALE_POSITION_PNL_THRESHOLD,
    BERP_ENABLED, BERP_TRIGGER_AGE_SEC, BERP_TRIGGER_PNL_THRESHOLD, BERP_RESCUE_DURATION_SEC, BERP_PROFIT_OVERRIDE_PCT,
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
    MSX_STAGE1_ENABLED, MSX_EARLY_INVALIDATION_R, MSX_VOL_SPIKE_MULT, MSX_MAX_SPREAD_STAGE1,
    USE_TRAILING_ENGINE, USE_NEW_TRAILING_ENGINE,
    MAX_POSITION_LOSS_PCT, DISABLE_PARTIAL_EXITS,
)

logger = get_logger("ExitPipeline")


@dataclass
class ExitRequest:
    """
    Exit request structure.
    All exit requests must be validated before execution.
    """
    symbol: str
    position: Dict[str, Any]
    reason: str
    target_price: Optional[float] = None
    exit_size_ratio: float = 1.0  # 1.0 = full exit, 0.5 = 50% exit, etc.
    use_limit: bool = False
    priority: int = 0  # Higher priority exits first
    
    def __post_init__(self):
        """Validate exit request."""
        if not 0 < self.exit_size_ratio <= 1.0:
            raise ValueError(f"Invalid exit_size_ratio: {self.exit_size_ratio} (must be 0 < ratio <= 1.0)")


@dataclass
class TrailingConfig:
    start_buffer_r: float = cfg.TRAIL_ENGINE_START_BUFFER_R
    partial_1_r: float = cfg.TRAIL_ENGINE_PARTIAL_1_R
    partial_1_size: float = cfg.TRAIL_ENGINE_PARTIAL_1_SIZE
    partial_1_sl_offset_r: float = cfg.TRAIL_ENGINE_PARTIAL_1_SL_OFFSET_R
    break_even_r: float = cfg.TRAIL_ENGINE_BREAK_EVEN_R
    break_even_buffer_r: float = cfg.TRAIL_ENGINE_BE_BUFFER_R
    partial_2_r: float = cfg.TRAIL_ENGINE_PARTIAL_2_R
    partial_2_size: float = cfg.TRAIL_ENGINE_PARTIAL_2_SIZE
    lock_r_level: float = cfg.TRAIL_ENGINE_LOCK_R_LEVEL
    lock_amount_r: float = cfg.TRAIL_ENGINE_LOCK_AMOUNT_R
    runner_start_r: float = cfg.TRAIL_ENGINE_RUNNER_START_R
    runner_trail_distance_r: float = cfg.TRAIL_ENGINE_RUNNER_TRAIL_DISTANCE_R
    min_r_increment: float = cfg.TRAIL_ENGINE_MIN_R_INCREMENT
    min_update_seconds: float = cfg.TRAIL_ENGINE_MIN_UPDATE_SECONDS


@dataclass
class TrailingAction:
    stop_updated: bool = False
    new_stop: Optional[float] = None
    stop_reason: Optional[str] = None
    locked_r: Optional[float] = None
    partial_actions: List[Tuple[float, str]] = field(default_factory=list)


class TrailingStopEngine:
    """
    Dynamic R-based trailing stop engine.
    Determines when to move stops or request partial exits.
    """
    
    def __init__(self, config: TrailingConfig):
        self.config = config
        self.logger = get_logger("TrailingStopEngine")
    
    def evaluate(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        now: Optional[float] = None
    ) -> Optional[TrailingAction]:
        if not cfg.USE_TRAILING_ENGINE:
            return None
        
        entry_price = position.get('entry_price')
        stop_loss = position.get('stop_loss')
        side = (position.get('side') or '').lower()
        
        if not entry_price or not stop_loss or side not in ('long', 'short') or current_price <= 0:
            return None
        
        now_ts = now or time.time()
        initial_stop = position.get('initial_stop_price', stop_loss)
        position.setdefault('initial_stop_price', initial_stop)
        
        risk_per_unit = abs(entry_price - initial_stop)
        if risk_per_unit <= 0:
            return None
        
        multiplier = 1 if side == 'long' else -1
        current_r = ((current_price - entry_price) * multiplier) / risk_per_unit
        position['current_r'] = current_r
        
        if side == 'long':
            peak_price = max(position.get('peak_price', entry_price), current_price)
            position['peak_price'] = peak_price
            peak_r = (peak_price - entry_price) / risk_per_unit
        else:
            trough_price = min(position.get('trough_price', entry_price), current_price)
            position['trough_price'] = trough_price
            peak_r = (entry_price - trough_price) / risk_per_unit
        
        position['max_r'] = max(position.get('max_r', 0.0) or 0.0, peak_r)
        position['max_r_reached'] = max(position.get('max_r_reached', 0.0) or 0.0, peak_r)
        
        if peak_r < self.config.start_buffer_r:
            return None
        
        state = position.setdefault('trailing_state', {
            'partial_1_taken': False,
            'partial_2_taken': False,
            'runner_mode': False,
            'last_update_r': 0.0,
            'last_update_ts': 0.0,
            'last_locked_r': 0.0
        })
        
        action = TrailingAction()
        current_stop = position.get('stop_loss', initial_stop)
        
        def clamp_stop(candidate: float) -> Optional[float]:
            buffer_pct = 0.0005  # ~5 bps
            if side == 'long':
                candidate = min(candidate, current_price * (1 - buffer_pct))
                if candidate <= current_stop + 1e-9:
                    return None
            else:
                candidate = max(candidate, current_price * (1 + buffer_pct))
                if candidate >= current_stop - 1e-9:
                    return None
            return candidate
        
        def update_stop(candidate: float, reason: str, locked_r_value: Optional[float] = None):
            nonlocal current_stop
            clamped = clamp_stop(candidate)
            if clamped is None:
                return
            action.new_stop = clamped
            action.stop_updated = True
            action.stop_reason = reason
            if locked_r_value is not None:
                action.locked_r = locked_r_value
            state['last_update_r'] = peak_r
            state['last_update_ts'] = now_ts
            position['stop_loss'] = clamped
            position['trailing_stop_price'] = clamped
            current_stop = clamped
            if reason in ("breakeven_lock", "locked_r", "runner_trail"):
                position['sl_moved_to_be'] = True
            # Logging handled at ExitPipeline level to avoid spam
        
        def should_progress() -> bool:
            if peak_r - state.get('last_update_r', 0.0) >= self.config.min_r_increment:
                return True
            if now_ts - state.get('last_update_ts', 0.0) >= self.config.min_update_seconds:
                return True
            return False
        
        # Partial at +1R
        if current_r >= self.config.partial_1_r and not state.get('partial_1_taken', False):
            state['partial_1_taken'] = True
            # Force 100% exit if partials disabled (prevents dust positions)
            if DISABLE_PARTIAL_EXITS:
                action.partial_actions.append((1.0, "trailing_tp_1r_full"))
            else:
                action.partial_actions.append((self.config.partial_1_size, "trailing_partial_1r"))
            position['partial_exit_done'] = True
            # Move stop closer but still below entry for LONG (above for SHORT)
            desired = entry_price - (multiplier * self.config.partial_1_sl_offset_r * risk_per_unit)
            update_stop(desired, "partial_1_guard")
        
        # Break-even protection at 1.5R
        if peak_r >= self.config.break_even_r and state.get('last_locked_r', 0.0) < self.config.break_even_r and should_progress():
            desired = entry_price + (multiplier * self.config.break_even_buffer_r * risk_per_unit)
            update_stop(desired, "breakeven_lock", locked_r_value=self.config.break_even_buffer_r)
            state['last_locked_r'] = self.config.break_even_r
        
        # Partial at 2R / lock-in
        if peak_r >= self.config.partial_2_r and not state.get('partial_2_taken', False):
            state['partial_2_taken'] = True
            # Force 100% exit if partials disabled (prevents dust positions)
            if DISABLE_PARTIAL_EXITS:
                action.partial_actions.append((1.0, "trailing_tp_2r_full"))
            else:
                action.partial_actions.append((self.config.partial_2_size, "trailing_partial_2r"))
            position['partial_exit_done'] = True
        
        if peak_r >= self.config.lock_r_level and state.get('last_locked_r', 0.0) < self.config.lock_r_level and should_progress():
            lock_price = entry_price + (multiplier * self.config.lock_amount_r * risk_per_unit)
            update_stop(lock_price, "locked_r", locked_r_value=self.config.lock_amount_r)
            state['last_locked_r'] = self.config.lock_r_level
        
        # Runner mode
        if peak_r >= self.config.runner_start_r:
            state['runner_mode'] = True
        
        if state.get('runner_mode', False) and should_progress():
            if side == 'long':
                peak_reference = position.get('peak_price', current_price)
                trailing_distance = self.config.runner_trail_distance_r * risk_per_unit
                desired = peak_reference - trailing_distance
            else:
                trough_reference = position.get('trough_price', current_price)
                trailing_distance = self.config.runner_trail_distance_r * risk_per_unit
                desired = trough_reference + trailing_distance
            update_stop(desired, "runner_trail", locked_r_value=max(0.0, peak_r - self.config.runner_trail_distance_r))
        
        if action.stop_updated or action.partial_actions:
            return action
        
        return None
class ExitPipeline:
    """
    Unified exit pipeline - single entry point for all exits.
    
    Responsibilities:
    1. Validate exit request
    2. Compute new position size
    3. Apply partial/full reduction
    4. Update PnL
    5. Update portfolio state
    6. Generate ONE log event
    7. Communicate with UI (single-line update)
    
    NO duplicates, NO scattered logic.
    """
    
    def __init__(
        self,
        order_manager: OrderManager,
        exit_manager: Any = None,  # Optional - for backward compatibility
        position_registry: Any = None,  # PositionRegistry from core
        exchange: Any = None  # Exchange for price fetching
    ):
        """
        Initialize exit pipeline.
        
        Args:
            order_manager: Order execution manager
            exit_manager: Exit logic manager (optional - for backward compatibility, will extract exchange from it)
            position_registry: Position registry for state updates
            exchange: Exchange instance for price fetching (takes precedence over exit_manager.exchange)
        """
        self.order_manager = order_manager
        self.position_registry = position_registry
        self.logger = get_logger("ExitPipeline")
        
        # Get exchange from exit_manager if provided, or use direct exchange parameter
        if exchange:
            self.exchange = exchange
        elif exit_manager and hasattr(exit_manager, 'exchange'):
            self.exchange = exit_manager.exchange
        else:
            self.exchange = None
        
        # Backward compatibility: store exit_manager if provided (for methods that still need it)
        self._exit_manager = exit_manager
        
        # Exit queue (priority-based)
        self._exit_queue: List[ExitRequest] = []
        self._processing = False
        
        # Statistics
        self.exits_processed = 0
        self.exits_failed = 0
        self.exits_by_reason: Dict[str, int] = {}

        # Trailing stop engine - match ExitManager logic
        if cfg.USE_NEW_TRAILING_ENGINE:
            try:
                from .trailing_engine import TrailingStopEngine as NewTrailingStopEngine
                self.trailing_engine = NewTrailingStopEngine()
            except ImportError:
                self.trailing_engine = None
        elif cfg.USE_TRAILING_ENGINE:
            # Use local TrailingStopEngine (has evaluate() method)
            self.trailing_engine = TrailingStopEngine(TrailingConfig())
        else:
            self.trailing_engine = None
        
        # PERFORMANCE: Pre-bind frequently used config constants (from ExitManager)
        from ..config import (
            R_SCALP_TP_R, R_SCALP_TIME_STOP_BARS, R_SCALP_BOREDOM_RANGE,
            R_STANDARD_PARTIAL_TP_R, R_STANDARD_PARTIAL_PCT, R_STANDARD_TRAIL_START_R,
            R_STANDARD_TIME_STOP_BARS, R_STANDARD_BOREDOM_RANGE,
            R_RUNNER_TIME_STOP_BARS
        )
        self._scalp_tp_r = R_SCALP_TP_R
        self._scalp_time_stop_bars = R_SCALP_TIME_STOP_BARS
        self._scalp_boredom_range = R_SCALP_BOREDOM_RANGE
        self._standard_partial_tp_r = R_STANDARD_PARTIAL_TP_R
        self._standard_partial_pct = R_STANDARD_PARTIAL_PCT
        self._standard_trail_start_r = R_STANDARD_TRAIL_START_R
        self._standard_time_stop_bars = R_STANDARD_TIME_STOP_BARS
        self._standard_boredom_range = R_STANDARD_BOREDOM_RANGE
        self._runner_time_stop_bars = R_RUNNER_TIME_STOP_BARS
    
    def queue_exit(self, request: ExitRequest) -> bool:
        """
        Queue an exit request for processing.
        
        Args:
            request: Exit request
            
        Returns:
            True if queued successfully, False if duplicate/invalid
        """
        # CRITICAL: Force 100% exit if partials disabled (prevents dust positions)
        # This is the primary enforcement point - all exits go through here
        if DISABLE_PARTIAL_EXITS and request.exit_size_ratio < 1.0:
            self.logger.debug(
                f"[PARTIAL_BLOCKED] {request.symbol}: exit_size_ratio={request.exit_size_ratio} "
                f"forced to 1.0 (DISABLE_PARTIAL_EXITS=True prevents dust positions)"
            )
            request.exit_size_ratio = 1.0
        
        # Check for duplicates (same symbol, same reason)
        for existing in self._exit_queue:
            if existing.symbol == request.symbol and existing.reason == request.reason:
                self.logger.debug(f"Duplicate exit request ignored: {request.symbol} {request.reason}")
                return False
        
        # Validate position exists
        if request.symbol not in self.position_registry.positions:
            self.logger.warning(f"Exit request for non-existent position: {request.symbol}")
            return False
        
        # Add to queue (sorted by priority)
        self._exit_queue.append(request)
        self._exit_queue.sort(key=lambda x: x.priority, reverse=True)
        
        return True
    
    async def process_exits(self, bot_instance: Any = None) -> int:
        """
        Process all queued exit requests.
        
        Args:
            bot_instance: Bot instance for PnL/statistics updates (optional)
        
        Returns:
            Number of exits processed
        """
        if self._processing or not self._exit_queue:
            return 0
        
        self._processing = True
        processed = 0
        
        try:
            # Process queue (highest priority first)
            # Track symbols being processed to prevent duplicate exits
            symbols_processed_this_batch = set()
            
            while self._exit_queue:
                request = self._exit_queue.pop(0)
                
                # Skip if we've already processed an exit for this symbol in this batch
                # (prevents duplicate exits when multiple requests are queued)
                if request.symbol in symbols_processed_this_batch:
                    self.logger.debug(
                        f"Skipping duplicate exit request: {request.symbol} reason={request.reason}"
                    )
                    continue
                
                try:
                    success = await self._execute_exit(request, bot_instance=bot_instance)
                    if success:
                        processed += 1
                        self.exits_processed += 1
                        self.exits_by_reason[request.reason] = self.exits_by_reason.get(request.reason, 0) + 1
                        # Mark symbol as processed (only if full exit)
                        if request.exit_size_ratio >= 1.0:
                            symbols_processed_this_batch.add(request.symbol)
                    # Note: Expected failures (position already closed) are logged as debug
                    # Only unexpected failures are counted and logged as errors
                    # The _execute_exit method handles this appropriately
                except Exception as e:
                    self.logger.error(
                        f"Exit execution failed: {request.symbol}",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        reason=request.reason
                    )
                    self.exits_failed += 1
        finally:
            self._processing = False
        
        return processed
    
    async def _execute_exit(
        self, 
        request: ExitRequest,
        bot_instance: Any = None  # Bot instance for PnL updates
    ) -> bool:
        """
        Execute a single exit request.
        
        Args:
            request: Exit request
            bot_instance: Bot instance for PnL/statistics updates (optional)
            
        Returns:
            True if exit successful, False otherwise
        """
        symbol = request.symbol
        position = request.position
        
        # Validate position state - always use current position from registry
        if symbol not in self.position_registry.positions:
            # Position already closed - this is expected when multiple exits are queued
            # Don't log as warning, just debug
            self.logger.debug(f"Exit skipped: position already closed: {symbol}")
            return False
        
        current_position = self.position_registry.positions[symbol]
        current_size = current_position.get('size', 0)
        
        if current_size <= 0:
            # Position size invalid - likely already closed
            # Don't log as warning, just debug
            self.logger.debug(f"Exit skipped: position size invalid: {symbol} size={current_size}")
            return False
        
        # Calculate exit size
        exit_size = current_size * request.exit_size_ratio
        new_size = current_size - exit_size
        
        # SMART DUST PREVENTION (The "Overshoot" Strategy)
        # When performing a 100% exit, our internal size tracking might be slightly lower
        # than the real exchange size due to rounding errors or fee deductions.
        # If we send exactly 'current_size', precision truncation might leave dust (e.g. 1.234 -> 1.23).
        # FIX: For full exits, we intentionally overshoot the size by 0.1%.
        # Since exit orders utilize 'reduceOnly=True' (enforced in ExitManager/Binance), 
        # the exchange will automatically clamp the order size to the exact open position.
        # This ensures we wipe out the position completely, including dust.
        if request.exit_size_ratio >= 0.999:  # Treat >= 99.9% as full exit
            exit_size *= 1.001  # +0.1% overshoot bias
            # new_size remains 0.0 conceptually
            new_size = 0.0
        
        # Validate exit size
        if exit_size <= 0:
            self.logger.warning(f"Invalid exit size: {symbol} exit_size={exit_size}")
            return False
        
        # Create temporary position dict for exit calculation
        exit_position_dict = current_position.copy()
        exit_position_dict['size'] = exit_size
        
        # Determine order type
        # Trailing stops always use market orders to avoid price validation issues
        use_limit = (request.use_limit or (request.reason in [
            "take_profit", "scalp_tp_1r", "standard_partial_1r", "runner_partial_1r"
        ])) and "trailing" not in request.reason.lower()
        
        # Execute exit (consolidated - no longer depends on exit_manager)
        exit_result = await self._execute_exit_order(
            symbol=symbol,
            position=exit_position_dict,
            reason=request.reason,
            target_price=request.target_price,
            use_limit=use_limit
        )
        
        if not exit_result.success:
            error_msg = exit_result.error or "Unknown error"
            
            # Check if this is an expected error (position already closed/invalid)
            expected_errors = [
                "Invalid position size (zero or negative)",
                "Invalid entry price (zero or negative)",
                "Position already closed",
                "invalid_limit_price",
                "invalid_price"
            ]
            
            # Special handling for trailing stop exits - fallback to market if limit fails
            is_trailing_exit = "trailing" in request.reason.lower()
            
            if is_trailing_exit and ("invalid_price" in error_msg or "invalid_limit_price" in error_msg):
                # For trailing stops, if price validation fails, try market order
                # Mark position to prevent retry spam
                if symbol in self.position_registry.positions:
                    position = self.position_registry.positions[symbol]
                    position['trailing_stop_exit_triggered'] = True
                    # If position size is tiny, just close it
                    if position.get('size', 0) <= 0.001:
                        self.logger.debug(f"EXIT_SKIPPED {symbol} trailing_stop (tiny size)")
                        # Remove position to prevent retries
                        self.position_registry.remove(symbol)
                        return False
                
                # Try market order instead
                self.logger.debug(
                    f"Trailing exit {symbol} falling back to market order (price validation failed)"
                )
                # Re-queue as market order
                request.use_limit = False
                request.target_price = None
                # Don't retry immediately - let it be processed in next cycle
                return False
            
            is_expected = any(expected in error_msg for expected in expected_errors)
            
            if is_expected:
                # Expected error - log as debug only, don't count as failure
                self.logger.debug(
                    f"Exit skipped (expected): {symbol} reason={request.reason} error={error_msg}"
                )
            else:
                # Unexpected error - log as error and count as failure
                self.logger.error(
                    f"Exit failed: {symbol}",
                    reason=request.reason,
                    error=error_msg
                )
                # Track unexpected failures (will be counted in process_exits)
                self.exits_failed += 1
            return False
        
        # Update bot statistics if bot_instance provided
        if bot_instance:
            was_win = exit_result.net_pnl > 0 if exit_result.net_pnl else False
            
            # Update statistics
            if was_win:
                bot_instance.win_count += 1
                bot_instance.gross_win += exit_result.gross_pnl or 0
            else:
                bot_instance.loss_count += 1
                bot_instance.gross_loss += abs(exit_result.gross_pnl or 0)
            
            # Update PnL totals
            bot_instance.realized_pnl_total += exit_result.net_pnl or 0
            bot_instance.realized_fees_total += exit_result.total_costs or 0
            
            # Track fee breakdown
            if exit_result.entry_fee is not None:
                bot_instance.realized_entry_fees_total += exit_result.entry_fee
            if exit_result.exit_fee is not None:
                bot_instance.realized_exit_fees_total += exit_result.exit_fee
            if exit_result.slippage is not None:
                bot_instance.realized_slippage_total += exit_result.slippage
            if exit_result.funding_cost is not None:
                bot_instance.realized_funding_total += exit_result.funding_cost
        
        # Update position state using canonical apply_exit_and_log
        from ..position_utils import apply_exit_and_log
        
        entry_price = current_position.get('entry_price', 0)
        entry_time = current_position.get('entry_time', time.time())
        exit_time = time.time()
        position_side = current_position.get('side', '')
        
        # Calculate PnL percentage
        pnl_pct = 0.0
        if entry_price > 0 and exit_result.exit_price and exit_result.exit_price > 0:
            if position_side.lower() == 'long':
                pnl_pct = ((exit_result.exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - exit_result.exit_price) / entry_price) * 100
        
        # Determine action type
        if request.exit_size_ratio < 1.0:
            action = "PARTIAL_EXIT" if "prs_" in request.reason or "scale" in request.reason.lower() else "SCALE_OUT"
        else:
            action = "EXIT"
        
        size_before = current_size
        size_after = new_size if request.exit_size_ratio < 1.0 else 0.0
        prs = current_position.get('recovery_score')
        
        # CANONICAL: Apply exit and log atomically
        success, event_created = apply_exit_and_log(
            positions=self.position_registry._positions,
            positions_set=self.position_registry._positions_set,
            symbol=symbol,
            new_size=new_size,
            action=action,
            exit_price=exit_result.exit_price,
            entry_price=entry_price,
            entry_time=entry_time,
            exit_time=exit_time,
            exit_size=exit_result.exit_size,
            size_before=size_before,
            size_after=size_after,
            side=position_side,
            pnl_value=exit_result.net_pnl,
            pnl_pct=pnl_pct,
            gross_pnl=exit_result.gross_pnl,
            net_pnl=exit_result.net_pnl,
            total_costs=exit_result.total_costs,
            reason=request.reason,
            prs=prs,
            was_win=was_win if bot_instance else None,
            is_unicorn=current_position.get('is_unicorn', False),
            bot_instance=bot_instance
        )
        
        if not success:
            self.logger.warning(f"Position update failed: {symbol}")
            return False
        
        # Record exit in position manager (only for full exits)
        if bot_instance and request.exit_size_ratio >= 1.0:
            equity = bot_instance.equity_now() if hasattr(bot_instance, 'equity_now') else 0
            pnl_pct_for_manager = (exit_result.net_pnl / equity * 100) if exit_result.net_pnl and equity > 0 else None
            # Calculate profit_atr for churn tracking (if time_exit)
            profit_atr = None
            if request.reason and request.reason.startswith("time_exit"):
                # Try to extract from reason string: "time_exit: 4 bars, profit=0.00ATR"
                import re
                match = re.search(r'profit=([\d\.-]+)ATR', request.reason)
                if match:
                    try:
                        profit_atr = float(match.group(1))
                    except (ValueError, AttributeError):
                        pass
                # Fallback: calculate from position data if available
                if profit_atr is None and entry_price > 0 and exit_result.exit_price > 0:
                    atr_pct = current_position.get('atr_pct', None)
                    if atr_pct and atr_pct > 0:
                        if position_side.lower() == 'long':
                            profit_pct = ((exit_result.exit_price - entry_price) / entry_price)
                        else:
                            profit_pct = ((entry_price - exit_result.exit_price) / entry_price)
                        profit_atr = profit_pct / atr_pct
            bot_instance.position_manager.record_exit(symbol, was_win, pnl_pct=pnl_pct_for_manager, exit_reason=request.reason, profit_atr=profit_atr)
        
        # Update metrics
        if bot_instance and hasattr(bot_instance, 'metrics'):
            try:
                bot_instance.metrics['exits_by_reason'][request.reason] = \
                    bot_instance.metrics['exits_by_reason'].get(request.reason, 0) + 1
            except Exception:
                pass
        
        # Log exit (ONE log event per exit - compact format)
        self.logger.info(
            f"EXIT: {symbol} | {request.reason} | "
            f"size={exit_size:.4f} | price={exit_result.exit_price:.2f} | "
            f"pnl={exit_result.net_pnl:.2f}"
        )
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get exit pipeline statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'exits_processed': self.exits_processed,
            'exits_failed': self.exits_failed,
            'exits_by_reason': self.exits_by_reason.copy(),
            'queue_length': len(self._exit_queue)
        }

    def evaluate_trailing(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        now: Optional[float] = None
    ) -> Optional[TrailingAction]:
        """
        Evaluate trailing stop logic for a position.
        
        Args:
            symbol: Trading symbol
            position: Position dictionary
            current_price: Latest mark price
            now: Optional timestamp
        
        Returns:
            TrailingAction if any updates/partials were triggered, or if trailing stop was hit
        """
        if not self.trailing_engine:
            return None
        
        # Check if trailing stop exit is already queued/processing for this position
        # This prevents duplicate trailing stop hit detections
        for queued_request in self._exit_queue:
            if queued_request.symbol == symbol and "trailing_stop" in queued_request.reason:
                # Already queued - skip detection
                return None
        
        side = (position.get('side') or '').lower()
        current_stop = position.get('stop_loss')
        
        # Check if we've already triggered a trailing stop exit for this position
        # (prevents repeated detections in subsequent loops)
        if position.get('trailing_stop_exit_triggered', False):
            return None
        
        # First, check if current stop has been hit (before any updates)
        if current_stop and current_stop > 0:
            stop_hit = False
            if side == 'long' and current_price <= current_stop:
                stop_hit = True
            elif side == 'short' and current_price >= current_stop:
                stop_hit = True
            
            if stop_hit:
                # Stop was hit - create action to trigger exit
                # Mark position to prevent duplicate detections
                position['trailing_stop_exit_triggered'] = True
                action = TrailingAction()
                action.partial_actions.append((1.0, "trailing_stop_hit"))
                self.logger.info(
                    f"[TRAIL] {symbol} TRAILING STOP HIT | "
                    f"price={current_price:.4f} stop={current_stop:.4f}"
                )
                return action
        
        # Stop not hit, evaluate trailing logic (may update stop, return partial actions)
        # Only use old trailing engine (has evaluate() method)
        # New trailing engine (has update() method) is handled in should_exit_position_r_based()
        if cfg.USE_NEW_TRAILING_ENGINE:
            # New trailing engine is handled elsewhere, skip here
            return None
        
        # EARLY PROTECTION: No trailing stops during first 15 minutes
        # This prevents premature exits from micro-volatility
        # SKIP for adopted positions (they may already be old)
        is_adopted = position.get('is_adopted', False)
        entry_time = position.get('entry_time', 0)
        if entry_time > 0 and not is_adopted:
            from ..config import EARLY_PROTECTION_MIN, EARLY_PROTECTION_ALLOW_EMERGENCY
            age_minutes = (now - entry_time) / 60.0
            
            if age_minutes < EARLY_PROTECTION_MIN:
                # Check for emergency exit (catastrophic loss)
                if EARLY_PROTECTION_ALLOW_EMERGENCY:
                    entry_price = position.get('entry_price', 0)
                    if entry_price > 0:
                        if side == 'long':
                            loss_pct = ((current_price - entry_price) / entry_price) * 100
                        else:
                            loss_pct = ((entry_price - current_price) / entry_price) * 100
                        
                        # Emergency exit if loss > 5%
                        if loss_pct < -5.0:
                            action = TrailingAction()
                            action.partial_actions.append((1.0, "emergency_exit_catastrophic_loss"))
                            self.logger.warning(
                                f"[EMERGENCY] {symbol} CATASTROPHIC LOSS | "
                                f"age={age_minutes:.1f}min loss={loss_pct:.2f}% - Emergency exit"
                            )
                            return action
                
                # Still in protection period - no trailing
                self.logger.debug(
                    f"[EARLY_PROTECTION] {symbol} age={age_minutes:.1f}min < {EARLY_PROTECTION_MIN}min - "
                    f"Blocking trailing stops"
                )
                return None
        
        # ADOPTED POSITIONS: Only use trailing if position is in profit
        # This prevents immediate exits on adopted manual positions
        if position.get('force_adopted') or position.get('emergency_adopted'):
            # Calculate profit percentage
            entry_price = position.get('entry_price', 0)
            if entry_price > 0:
                if side == 'long':
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # short
                    profit_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Only use trailing if in profit by at least 1.5%
                if profit_pct < 1.5:
                    return None  # Skip trailing, use initial stop-loss only
        
        # Old trailing engine has evaluate() method
        action = self.trailing_engine.evaluate(symbol, position, current_price, now=now)
        
        # After trailing evaluation, check if updated stop was hit
        if action and action.stop_updated:
            updated_stop = position.get('stop_loss')
            
            if updated_stop and updated_stop > 0:
                # Check if price crossed the newly updated stop
                stop_hit = False
                if side == 'long' and current_price <= updated_stop:
                    stop_hit = True
                elif side == 'short' and current_price >= updated_stop:
                    stop_hit = True
                
                if stop_hit:
                    # Newly updated stop was immediately hit - trigger full exit
                    # Mark position to prevent duplicate detections
                    position['trailing_stop_exit_triggered'] = True
                    action.partial_actions.append((1.0, "trailing_stop_hit"))
                    self.logger.info(
                        f"[TRAIL] {symbol} TRAILING STOP HIT (after update) | "
                        f"price={current_price:.4f} stop={updated_stop:.4f} "
                        f"reason={action.stop_reason}"
                    )
                else:
                    # Stop was updated but not hit - log the update
                    locked_r = action.locked_r or 0.0
                    self.logger.info(
                        f"[TRAIL] {symbol} TRAIL_SL_UPDATE | "
                        f"side={side.upper()} new_SL={updated_stop:.4f} "
                        f"locked_R={locked_r:.2f} reason={action.stop_reason}"
                    )
        
        # Persist stop updates to position registry
        if action and action.stop_updated:
            self.position_registry.update(symbol, position)
        
        return action
    
    # ============================================================================
    # Methods migrated from ExitManager for consolidation
    # ============================================================================
    
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
        
        # ============================================================
        # EMERGENCY CIRCUIT BREAKER: Force exit on excessive losses
        # ============================================================
        # This prevents catastrophic losses like PIPPIN (2 days of losses)
        # CRITICAL: Check this BEFORE any other exit logic
        if profit_pct <= MAX_POSITION_LOSS_PCT:
            self.logger.error(
                f"[EMERGENCY_EXIT] {symbol or 'UNKNOWN'} {side.upper()} | "
                f"LOSS LIMIT HIT: {profit_pct:.2f}% <= {MAX_POSITION_LOSS_PCT}% | "
                f"entry={entry_price:.6f} current={current_price:.6f} | "
                f"FORCING IMMEDIATE EXIT"
            )
            return True, f"emergency_loss_limit_{MAX_POSITION_LOSS_PCT}%", current_price
        
        # Additional safety: If stop-loss is missing or invalid, force exit on any loss > 5%
        if (stop_loss <= 0 or stop_loss == entry_price) and profit_pct <= -5.0:
            self.logger.error(
                f"[EMERGENCY_EXIT] {symbol or 'UNKNOWN'} {side.upper()} | "
                f"INVALID STOP-LOSS with {profit_pct:.2f}% loss | "
                f"stop_loss={stop_loss:.6f} entry={entry_price:.6f} | "
                f"FORCING IMMEDIATE EXIT"
            )
            return True, "emergency_invalid_stop_loss", current_price
        
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
                
                # 🎯 PROFIT OVERRIDE: If position in profit after 60min, extend hold with tight trailing
                if time_in_rescue >= BERP_RESCUE_DURATION_SEC:
                    if profit_pct >= BERP_PROFIT_OVERRIDE_PCT:
                        # Position is profitable - extend hold and tighten trailing stop
                        logger.info(
                            f"[BERP_PROFIT_EXTENSION] {symbol} LONG R={profit_pct:.2f}% "
                            f"at 60min - extending hold with tight trailing (profit_override={BERP_PROFIT_OVERRIDE_PCT}%)"
                        )
                        # Don't exit - let trailing stop handle it
                        # Tighten the stop loss to lock in profits
                        if stop_loss > 0:
                            if side == 'long':
                                # Tighter trailing for profitable positions: move stop to lock in 50% of profit
                                profit_range = current_price - entry_price
                                tight_stop = entry_price + (profit_range * 0.5)  # Lock in at 50% of profit
                                if tight_stop > stop_loss:
                                    position['stop_loss'] = tight_stop
                                    logger.info(f"[BERP_TIGHT_TRAILING] {symbol} tightened SL: {stop_loss:.6f} -> {tight_stop:.6f}")
                            else:  # short
                                profit_range = entry_price - current_price
                                tight_stop = entry_price - (profit_range * 0.5)  # Lock in at 50% of profit
                                if tight_stop < stop_loss:
                                    position['stop_loss'] = tight_stop
                                    logger.info(f"[BERP_TIGHT_TRAILING] {symbol} tightened SL: {stop_loss:.6f} -> {tight_stop:.6f}")
                        return False, "berp_extended_hold_profitable", current_price
                    else:
                        # ❌ Rescue timeout expired - exit as failed rescue (not profitable enough)
                        return True, "failed_rescue_after_60m", current_price
        
        # Get regime-specific parameters
        if regime_config:
            wide_spread_threshold = regime_config.wide_spread_exit_threshold_bps
        else:
            wide_spread_threshold = WIDE_SPREAD_EXIT_THRESHOLD_BPS
        
        # NOTE: Trailing stop updates are now handled by SimpleStopLossManager in bot.py
        # This function ONLY checks if SL is hit, it does NOT update trailing stops
        
        # Check stop-loss (priority - exit immediately on stop-loss)
        if stop_loss > 0:
            stop_hit = False
            if side == 'long':
                if current_price <= stop_loss:
                    stop_hit = True
            else:  # short
                if current_price >= stop_loss:
                    stop_hit = True
            
            if stop_hit:
                # Log stop-loss hit for debugging
                if symbol and ('PIPPIN' in symbol.upper() or self.logger.isEnabledFor(logging.DEBUG)):
                    self.logger.warning(
                        f"[STOP_LOSS_HIT] {symbol} {side.upper()} | "
                        f"price={current_price:.6f} stop={stop_loss:.6f} entry={entry_price:.6f}"
                    )
                return True, "stop_loss", stop_loss
        
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
        
        return False, "hold", 0.0
    
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
        entry_price = position.get('entry_price', 0)
        side = position.get('side', '').lower()
        
        if not entry_price or entry_price <= 0:
            return False, None, None, None
        
        # Calculate current profit/loss percentage
        if side == 'long':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # short
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        # ============================================================
        # EMERGENCY CIRCUIT BREAKER: Force exit on excessive losses
        # ============================================================
        # CRITICAL: Check this BEFORE any other exit logic
        if profit_pct <= MAX_POSITION_LOSS_PCT:
            self.logger.error(
                f"[EMERGENCY_EXIT_R] {position.get('symbol', 'UNKNOWN')} {side.upper()} | "
                f"LOSS LIMIT HIT: {profit_pct:.2f}% <= {MAX_POSITION_LOSS_PCT}% | "
                f"entry={entry_price:.6f} current={current_price:.6f} | "
                f"FORCING IMMEDIATE EXIT"
            )
            return True, f"emergency_loss_limit_{MAX_POSITION_LOSS_PCT}%", current_price, 1.0
        
        # Additional safety: If stop-loss is missing or invalid, force exit on any loss > 5%
        stop_loss = position.get('stop_loss', 0)
        if (stop_loss <= 0 or stop_loss == entry_price) and profit_pct <= -5.0:
            self.logger.error(
                f"[EMERGENCY_EXIT_R] {position.get('symbol', 'UNKNOWN')} {side.upper()} | "
                f"INVALID STOP-LOSS with {profit_pct:.2f}% loss | "
                f"stop_loss={stop_loss:.6f} entry={entry_price:.6f} | "
                f"FORCING IMMEDIATE EXIT"
            )
            return True, "emergency_invalid_stop_loss", current_price, 1.0
        
        # Update R metadata
        self.update_position_r_metadata(position, current_price, bar_closed)
        
        # Always live - use full exit logic with trailing/partial
        
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
                    self.logger.info(
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
        # Partial TP at 1.0R (close 50%) - DISABLED if DISABLE_PARTIAL_EXITS
        if not partial_exit_done and r_multiple >= self._standard_partial_tp_r:
            # Move SL to breakeven when TP hit
            if not sl_moved_to_be:
                position['sl_moved_to_be'] = True
                position['stop_loss'] = entry_price  # Move to breakeven
            position['partial_exit_done'] = True
            # Force 100% exit if partials disabled (prevents dust positions)
            if DISABLE_PARTIAL_EXITS:
                return True, "standard_tp_1r_full", current_price, 1.0
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
        
        # Small partial at 1.0R (close 25%) - DISABLED if DISABLE_PARTIAL_EXITS
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
            # Force 100% exit if partials disabled (prevents dust positions)
            if DISABLE_PARTIAL_EXITS:
                return True, "runner_tp_1r_full", current_price, 1.0
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
                # Force 100% exit if partials disabled (prevents dust positions)
                if DISABLE_PARTIAL_EXITS:
                    return True, f"msx_stage2_tp_{partial_trigger:.1f}r_full", current_price, 1.0
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
            from ..regime import TradingRegime
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
    
    async def _execute_exit_order(
        self,
        symbol: str,
        position: Dict[str, Any],
        reason: str,
        target_price: float = None,
        use_limit: bool = False,
        funding_rate: float = None
    ) -> ExitResult:
        """
        Execute an exit order (consolidated from ExitManager.exit_position).
        
        Args:
            symbol: Trading symbol
            position: Position dictionary
            reason: Exit reason ('stop_loss', 'take_profit', 'manual', etc.)
            target_price: Target exit price (if None, uses market price)
            use_limit: Use limit order instead of market
            funding_rate: Funding rate for cost calculation (optional)
        
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
        
        # Validate entry price exists
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0:
            return ExitResult(
                success=False,
                error="Invalid entry price (zero or negative)"
            )
        
        # Determine exit side (opposite of entry)
        exit_side = "sell" if side == "long" else "buy"
        
        # Get exchange (use order_manager.exchange as fallback)
        exchange = self.exchange
        if not exchange and self.order_manager and hasattr(self.order_manager, 'exchange'):
            exchange = self.order_manager.exchange
        
        # Validate exchange is available for live trading
        if not exchange:
            return ExitResult(
                success=False,
                error="Exchange not initialized"
            )
        
        # For LIMIT orders: target_price is REQUIRED and must be valid
        # For MARKET orders: target_price can be None (we'll fetch market price for PnL calculation only)
        ticker_data = None  # Cache ticker data to avoid duplicate fetches
        
        if use_limit:
            # LIMIT ORDER: Validate target_price is required and valid
            if target_price is None or target_price <= 0:
                return ExitResult(
                    success=False,
                    error="invalid_limit_price"
                )
            
            # Get current market price for validation (defensive clamp)
            current_market_price = None
            try:
                if exchange and hasattr(exchange, 'fetch_ticker'):
                    ticker_data = await exchange.fetch_ticker(symbol)
                    if side == "long":
                        current_market_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                    else:
                        current_market_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
            except Exception:
                current_market_price = entry_price
            
            # Defensive clamp: Validate target_price is on correct side of market
            if current_market_price and current_market_price > 0:
                if side == "long":
                    if target_price < current_market_price * 0.99:  # Allow 1% tolerance
                        use_limit = False
                        if ticker_data:
                            target_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                        else:
                            target_price = None
                else:
                    if target_price > current_market_price * 1.01:  # Allow 1% tolerance
                        use_limit = False
                        if ticker_data:
                            target_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
                        else:
                            target_price = None
        
        # For MARKET orders: Fetch market price for PnL calculation
        if not use_limit and (target_price is None or target_price <= 0):
            if ticker_data:
                if side == "long":
                    target_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                else:
                    target_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
            else:
                try:
                    if exchange and hasattr(exchange, 'fetch_ticker'):
                        ticker_data = await exchange.fetch_ticker(symbol)
                        if side == "long":
                            target_price = ticker_data.get('bid', ticker_data.get('last', entry_price))
                        else:
                            target_price = ticker_data.get('ask', ticker_data.get('last', entry_price))
                    else:
                        target_price = entry_price
                except Exception:
                    target_price = entry_price
        
        # Final validation: target_price must be valid for PnL calculation
        if target_price is None or target_price <= 0:
            if entry_price and entry_price > 0:
                target_price = entry_price
            else:
                return ExitResult(
                    success=False,
                    error="invalid_price"
                )
        
        # Execute exit order
        try:
            # Live mode - always execute on exchange
            # CANCEL BINANCE TRAILING STOP (if any)
            try:
                if exchange and hasattr(exchange, 'cancel_trailing_stop'):
                    await exchange.cancel_trailing_stop(symbol)
            except Exception:
                pass  # Non-critical
            
            # Format quantity to exchange precision
            try:
                inner_exchange = None
                if exchange:
                    if hasattr(exchange, '_inner'):
                        inner_exchange = exchange._inner
                    elif hasattr(exchange, 'exchange'):
                        inner_exchange = getattr(exchange, 'exchange', None)
                
                if inner_exchange and hasattr(inner_exchange, 'amount_to_precision'):
                    formatted_quantity = inner_exchange.amount_to_precision(symbol, abs(position_size))
                    final_quantity = float(formatted_quantity)
                else:
                    final_quantity = round(abs(position_size), 8)
                
                if final_quantity <= 0:
                    return ExitResult(
                        success=False,
                        error=f"Exit quantity {position_size} formatted to zero (too small)"
                    )
            except Exception:
                final_quantity = round(abs(position_size), 8)
                if final_quantity <= 0:
                    return ExitResult(
                        success=False,
                        error="Could not format exit quantity"
                    )
            
            # Use reduceOnly=True for exits (Binance-specific safety parameter)
            exit_params = {"reduceOnly": True}
            
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
                
                order = await exchange.create_order(
                    symbol,
                    "limit",
                    exit_side,
                    final_quantity,
                    final_price,
                    params=exit_params
                )
            else:
                order = await exchange.create_order(
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
            
            # Calculate costs and PnL
            entry_price = position.get('entry_price', 0)
            entry_size = position.get('size', 0)
            costs = self.calculate_exit_costs(
                exit_result.exit_size,
                exit_result.exit_price,
                entry_price=entry_price,
                entry_size=entry_size
            )
            
            # Calculate funding costs
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

