"""
Scalper Exit Logic - Tight, protective trailing exits for scalper positions.

Implements:
1. Initial SL at 0.35 * ATR
2. Micro trailing activation at +0.3 ATR
3. Main trailing at +0.6 ATR
4. Structural exit overrides
5. Time-based exits (3-4 candles max)
"""

import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from ..config import (
    DRY_RUN, TIME_EXIT_BARS, DRY_SIMPLE_EXITS, DRY_SIMPLE_SL_R, DRY_SIMPLE_TP_R,
    # Data-driven hold time (from 4M+ trades analysis)
    MIN_HOLD_TIME_SEC, MIN_HOLD_PROFIT_EXCEPTION_PCT,
    # Diamond Hands strategy
    DISABLE_STOP_DURING_HOLD, TIME_EXIT_AFTER_HOLD
)


@dataclass
class ScalperExitAction:
    """Action to take for a scalper position."""
    action: str  # "update_sl", "exit", or None
    new_stop: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_size_ratio: float = 1.0  # 1.0 = full exit
    exit_price: Optional[float] = None  # Trigger price for execution

def _is_dry_simple_exits() -> bool:
    """
    Check if DRY_RUN simple exits mode is enabled.
    DRY sandbox mode: only hard SL and hard TP exits, no partials/trailing.
    """
    return bool(DRY_RUN and DRY_SIMPLE_EXITS)


def _evaluate_simple_dry_exit(
    position: Dict[str, Any],
    current_price: float,
    atr_pct: Optional[float],
    side: str,
    entry_price: float,
    initial_stop: float
) -> Optional[ScalperExitAction]:
    """
    DRY_RUN sandbox: only hard SL and hard TP exits based on R.
    No partials, no trailing, no time-based micro scalps.
    
    Args:
        position: Position dictionary
        current_price: Current market price
        atr_pct: ATR as percentage of price
        side: "long" or "short"
        entry_price: Entry price
        initial_stop: Initial stop loss price
    
    Returns:
        ScalperExitAction if exit needed, None otherwise
    """
    if not atr_pct or atr_pct <= 0:
        atr_pct = 0.01
    
    # Calculate R-multiple based on initial stop distance
    if initial_stop <= 0 or entry_price <= 0:
        return None  # Can't calculate R without valid stop
    
    risk_per_unit = abs(entry_price - initial_stop)
    if risk_per_unit <= 0:
        return None
    
    # Calculate current R
    if side == "long":
        current_r = (current_price - entry_price) / risk_per_unit
    else:  # short
        current_r = (entry_price - current_price) / risk_per_unit
    
    # Hard stop loss at DRY_SIMPLE_SL_R or worse
    sl_r = float(DRY_SIMPLE_SL_R)
    if current_r <= sl_r:
        return ScalperExitAction(
            action="exit",
            exit_reason=f"hard_sl_dry_simple R={current_r:.2f}",
            exit_size_ratio=1.0,
            exit_price=current_price
        )
    
    # Hard take-profit at DRY_SIMPLE_TP_R or better
    tp_r = float(DRY_SIMPLE_TP_R)
    if current_r >= tp_r:
        return ScalperExitAction(
            action="exit",
            exit_reason=f"hard_tp_dry_simple R={current_r:.2f}",
            exit_size_ratio=1.0,
            exit_price=current_price
        )
    
    # Otherwise: hold (let hard SL protect if price moves against us)
    return None


def evaluate_scalper_trailing(
    position: Dict[str, Any],
    current_price: float,
    atr_pct: Optional[float],
    advanced_features: Optional[Any],
    indicators: Optional[Dict],
    side: str,
    entry_price: float,
    entry_time: float,
    bars_in_trade: int
) -> Optional[ScalperExitAction]:
    """
    Evaluate scalper-specific trailing exit logic.
    
    Args:
        position: Position dictionary
        current_price: Current market price
        atr_pct: ATR as percentage of price
        advanced_features: AdvancedFeatures object
        indicators: Indicators dict
        side: "long" or "short"
        entry_price: Entry price
        entry_time: Entry timestamp
        bars_in_trade: Number of candles/bars since entry
    
    Returns:
        ScalperExitAction if action needed, None otherwise
    """
    if not atr_pct or atr_pct <= 0:
        # Fallback to 1% if ATR not available
        atr_pct = 0.01
    
    current_stop = position.get('stop_loss', 0)
    initial_stop = position.get('initial_stop_price', current_stop)
    if not initial_stop:
        # Set initial stop if not set (use configurable ATR multiplier)
        from ..config import SL_ATR_MULTIPLIER
        if side == "long":
            initial_stop = entry_price * (1.0 - SL_ATR_MULTIPLIER * atr_pct)
        else:
            initial_stop = entry_price * (1.0 + SL_ATR_MULTIPLIER * atr_pct)
        position['initial_stop_price'] = initial_stop
        position['stop_loss'] = initial_stop
    
    # DRY_RUN simple exits: bypass all trailing/partial logic, use simple SL/TP only
    # BUT respect Diamond Hands strategy if enabled
    if _is_dry_simple_exits():
        import time as _time
        hold_time_sec = _time.time() - entry_time
        min_hold_active = MIN_HOLD_TIME_SEC > 0 and hold_time_sec < MIN_HOLD_TIME_SEC
        hold_expired = MIN_HOLD_TIME_SEC > 0 and hold_time_sec >= MIN_HOLD_TIME_SEC
        
        # Diamond Hands: After hold period expires, exit immediately
        if TIME_EXIT_AFTER_HOLD and hold_expired:
            # Calculate current R for logging
            risk_per_unit = abs(entry_price - initial_stop) if initial_stop > 0 else entry_price * 0.01
            if side == "long":
                current_r = (current_price - entry_price) / risk_per_unit if risk_per_unit > 0 else 0
            else:
                current_r = (entry_price - current_price) / risk_per_unit if risk_per_unit > 0 else 0
            return ScalperExitAction(
                action="exit",
                exit_reason=f"time_exit_diamond_hands ({hold_time_sec/60:.0f}min, R={current_r:.2f})",
                exit_size_ratio=1.0,
                exit_price=current_price
            )
        
        # Diamond Hands during hold: NO stops allowed
        if DISABLE_STOP_DURING_HOLD and min_hold_active:
            # Check for early profit lock
            profit_pct = ((current_price - entry_price) / entry_price) if side == "long" else ((entry_price - current_price) / entry_price)
            if profit_pct >= MIN_HOLD_PROFIT_EXCEPTION_PCT:
                return ScalperExitAction(
                    action="exit",
                    exit_reason=f"early_profit_lock ({profit_pct*100:.1f}%, {hold_time_sec/60:.0f}min)",
                    exit_size_ratio=1.0,
                    exit_price=current_price
                )
            # Otherwise, HODL - no exit
            return None
        
        # Legacy mode: Check hard SL first (price hit stop)
        if side == "long" and current_price <= current_stop:
            return ScalperExitAction(
                action="exit",
                exit_reason="stop_loss_hit",
                exit_size_ratio=1.0,
                exit_price=current_price
            )
        elif side == "short" and current_price >= current_stop:
            return ScalperExitAction(
                action="exit",
                exit_reason="stop_loss_hit",
                exit_size_ratio=1.0,
                exit_price=current_price
            )
        
        # Use simple DRY exit logic (hard SL/TP in R-space)
        return _evaluate_simple_dry_exit(
            position=position,
            current_price=current_price,
            atr_pct=atr_pct,
            side=side,
            entry_price=entry_price,
            initial_stop=initial_stop
        )
    
    # LIVE/NORMAL path: full exit engine with trailing/partials
    
    # Calculate profit in ATR units
    if side == "long":
        profit_pct = ((current_price - entry_price) / entry_price)
        profit_atr = profit_pct / atr_pct
    else:  # short
        profit_pct = ((entry_price - current_price) / entry_price)
        profit_atr = profit_pct / atr_pct
    
    # ============================================================
    # DIAMOND HANDS STRATEGY (Data-driven from 4M+ trades)
    # ============================================================
    # KEY INSIGHT: Trades that survive 1 hour are profitable!
    #   - Time exits (1hr): +0.27R avg, 72% WR
    #   - Stop losses: -0.43R avg, killing trades on noise
    #   - Total R with stops: -451,044 (LOSING)
    #   - Total R without stops: +505,635 (WINNING!)
    #
    # Strategy: DISABLE stop losses during hold period
    # ============================================================
    import time as _time
    hold_time_sec = _time.time() - entry_time
    min_hold_active = MIN_HOLD_TIME_SEC > 0 and hold_time_sec < MIN_HOLD_TIME_SEC
    hold_expired = MIN_HOLD_TIME_SEC > 0 and hold_time_sec >= MIN_HOLD_TIME_SEC
    
    # Diamond Hands: After hold period expires, exit immediately (time exit)
    if TIME_EXIT_AFTER_HOLD and hold_expired:
        return ScalperExitAction(
            action="exit",
            exit_reason=f"time_exit_diamond_hands ({hold_time_sec/60:.0f}min, R={profit_atr:.2f})",
            exit_size_ratio=1.0,
            exit_price=current_price
        )
    
    if min_hold_active:
        # Exception: Allow exit if profit exceeds threshold (lock in big wins early)
        exceeds_profit_threshold = profit_pct >= MIN_HOLD_PROFIT_EXCEPTION_PCT
        
        if exceeds_profit_threshold:
            # Big profit - allow early exit to lock in gains
            return ScalperExitAction(
                action="exit",
                exit_reason=f"early_profit_lock ({profit_pct*100:.1f}%, {hold_time_sec/60:.0f}min)",
                exit_size_ratio=1.0,
                exit_price=current_price
            )
        
        if DISABLE_STOP_DURING_HOLD:
            # DIAMOND HANDS MODE: NO stop losses during hold period
            # Just hold. Let it cook. The data says this works.
            return None  # No exit action - stay in trade
        else:
            # Legacy mode: still allow hard SL hit during hold
            if side == "long" and current_price <= current_stop:
                pass  # Allow stop exit
            elif side == "short" and current_price >= current_stop:
                pass  # Allow stop exit
            else:
                return None  # Block other exits
    # ============================================================
    
    # 1. Structural Exit Overrides (hard exits regardless of trailing)
    # DATA-DRIVEN: Block structural exits during minimum hold period unless in big profit
    structural_exit = _check_structural_exit(
        side, advanced_features, current_price, entry_price, atr_pct
    )
    if structural_exit:
        # Respect minimum hold time (unless profit exceeds threshold)
        if min_hold_active and profit_pct < MIN_HOLD_PROFIT_EXCEPTION_PCT:
            pass  # Block structural exit during hold period
        else:
            return ScalperExitAction(
                action="exit",
                exit_reason=structural_exit,
                exit_size_ratio=1.0
            )
    
    # 2. Time-based exit (more patient, only kill losers/zombies)
    # DATA-DRIVEN: Use MIN_HOLD_TIME_SEC as minimum before considering time exits
    # Analysis: 60+ min holds = 55.9% WR vs 11.7% for 5-15 min
    
    # Only consider time-exit after minimum hold period
    if min_hold_active:
        # Still in minimum hold period - no time exits
        pass
    elif bars_in_trade < TIME_EXIT_BARS:
        # Too early to time-exit (legacy bar count check)
        pass
    else:
        # Calculate current R (profit in R units)
        # profit_atr is already in ATR units, which approximates R for scalping
        # For scalping, 1 ATR ≈ 1R (stop is typically ~0.35 ATR, so profit_atr ≈ R)
        current_R = profit_atr  # Approximate: profit_atr ≈ R for scalping
        
        # Get peak R (best R seen so far) from position metadata
        peak_R = position.get('max_r_reached', 0.0)
        if peak_R is None:
            peak_R = 0.0
        
        # Fallback: calculate peak_R from peak_price if available
        if peak_R == 0.0:
            peak_price = position.get('peak_price') if side == 'long' else position.get('trough_price')
            if peak_price and entry_price > 0:
                if side == "long":
                    peak_profit_pct = ((peak_price - entry_price) / entry_price)
                else:  # short
                    peak_profit_pct = ((entry_price - peak_price) / entry_price)
                peak_R = peak_profit_pct / atr_pct if atr_pct > 0 else 0.0
        
        # Only nuke if it's clearly a loser or a dead/zombie trade:
        # LOSER: cut it if we're worse than -0.3R
        is_clear_loser = current_R <= -0.30
        
        # ZOMBIE: if it never exceeded +0.3R and is still ≤ +0.1R, we give up
        is_dead_trade = (peak_R < 0.30 and current_R <= 0.10)
        
        if is_clear_loser or is_dead_trade:
            return ScalperExitAction(
                action="exit",
                exit_reason=f"time_exit: {bars_in_trade} bars, profit={profit_atr:.2f}ATR",
                exit_size_ratio=1.0
            )
        # Everything else stays open for trailing / normal SL
    
    # 3. Micro Trailing Activation (+0.5 ATR) - Break Even Defense
    if profit_atr >= 0.5:
        # Move SL to +0.05 ATR (Break Even + Fees). Free Ride.
        if side == "long":
            new_stop = entry_price * (1.0 + 0.05 * atr_pct)
            if new_stop > current_stop:  # Only move stop up
                return ScalperExitAction(
                    action="update_sl",
                    new_stop=new_stop,
                    exit_reason="break_even_lock"
                )
        else:  # short
            new_stop = entry_price * (1.0 - 0.05 * atr_pct)
            if new_stop < current_stop or current_stop == 0:  # Only move stop down
                return ScalperExitAction(
                    action="update_sl",
                    new_stop=new_stop,
                    exit_reason="break_even_lock"
                )
    
    # 4. Main Trailing (+0.6 ATR)
    if profit_atr >= 0.6:
        # DYNAMIC TRAILING DISTANCE ("The Moonbag Logic")
        # 1. Base Trail (0.6 - 2.0 ATR Profit): Tight (0.35 ATR) to secure the bag.
        trail_mult = 0.35
        
        # 2. Runner Mode (2.0 - 5.0 ATR Profit): Loosen (0.6 ATR) to survive pullbacks and catch the moonshot.
        if profit_atr > 2.0:
            trail_mult = 0.6
            
        # 3. Victory Lap (5.0+ ATR Profit): Tighten (0.2 ATR) to lock in massive gains.
        if profit_atr > 5.0:
            trail_mult = 0.2
            
        # Use trailing stop tied to current price
        if side == "long":
            new_stop = current_price * (1.0 - trail_mult * atr_pct)
            if new_stop > current_stop:  # Only move stop up
                return ScalperExitAction(
                    action="update_sl",
                    new_stop=new_stop,
                    exit_reason=None
                )
        else:  # short
            new_stop = current_price * (1.0 + trail_mult * atr_pct)
            if new_stop < current_stop or current_stop == 0:  # Only move stop down
                return ScalperExitAction(
                    action="update_sl",
                    new_stop=new_stop,
                    exit_reason=None
                )
    
    # 5. Check if stop-loss hit (ONLY if not in Diamond Hands mode during hold)
    # Diamond Hands: NO stop losses during hold period - this is the key to profitability
    if DISABLE_STOP_DURING_HOLD and min_hold_active:
        # During hold period with Diamond Hands, completely ignore stop loss
        return None
    
    if side == "long" and current_price <= current_stop:
        return ScalperExitAction(
            action="exit",
            exit_reason="stop_loss_hit",
            exit_size_ratio=1.0
        )
    elif side == "short" and current_price >= current_stop:
        return ScalperExitAction(
            action="exit",
            exit_reason="stop_loss_hit",
            exit_size_ratio=1.0
        )
    
    return None


def _check_structural_exit(
    side: str,
    advanced_features: Optional[Any],
    current_price: float,
    entry_price: float,
    atr_pct: float
) -> Optional[str]:
    """
    Check for structural exit conditions (opposite exhaustion, divergence, SFP, VWAP reversal).
    
    Returns:
        Exit reason string if structural exit needed, None otherwise
    """
    if not advanced_features:
        return None
    
    # Check opposite exhaustion
    if side == "long":
        if hasattr(advanced_features, 'exhaustion_up') and advanced_features.exhaustion_up:
            return "structural_exit: opposite_exhaustion"
    else:  # short
        if hasattr(advanced_features, 'exhaustion_down') and advanced_features.exhaustion_down:
            return "structural_exit: opposite_exhaustion"
    
    # Check opposite divergence
    if side == "long":
        if hasattr(advanced_features, 'bearish_divergence') and advanced_features.bearish_divergence:
            return "structural_exit: opposite_divergence"
    else:  # short
        if hasattr(advanced_features, 'bullish_divergence') and advanced_features.bullish_divergence:
            return "structural_exit: opposite_divergence"
    
    # Check opposite SFP
    if side == "long":
        if hasattr(advanced_features, 'sfp_top') and advanced_features.sfp_top:
            return "structural_exit: opposite_sfp"
    else:  # short
        if hasattr(advanced_features, 'sfp_bottom') and advanced_features.sfp_bottom:
            return "structural_exit: opposite_sfp"
    
    # Check VWAP reversal (price returns sharply through VWAP against position)
    if hasattr(advanced_features, 'dist_from_vwap'):
        dist_from_vwap = advanced_features.dist_from_vwap
        # If we've moved significantly and now reversed through VWAP
        move_pct = abs((current_price - entry_price) / entry_price)
        if move_pct > 0.01:  # At least 1% move
            if side == "long" and dist_from_vwap > 0.005:  # Price now above VWAP after long move
                return "structural_exit: vwap_reversal"
            elif side == "short" and dist_from_vwap < -0.005:  # Price now below VWAP after short move
                return "structural_exit: vwap_reversal"
    
    return None

