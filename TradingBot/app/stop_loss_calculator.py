"""
Stop Loss Calculator - CENTRALIZED

Two distinct modes:
1. INITIAL SL: Protect against initial loss (1.5%+ from entry)
2. TRAILING SL: Lock in profits as price moves in our favor

Key Rules:
- LONG: SL always BELOW entry (initial) or BELOW current price (trailing)
- SHORT: SL always ABOVE entry (initial) or ABOVE current price (trailing)
- Never move SL in the wrong direction (only tighten, never loosen)
"""

from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Configuration
MIN_INITIAL_STOP_PCT = 0.015   # 1.5% minimum for initial SL
TRAIL_DISTANCE_PCT = 0.003     # 0.3% trailing distance when in profit
BREAKEVEN_THRESHOLD_PCT = 0.005  # Move to breakeven at 0.5% profit


def calculate_initial_stop_loss(
    entry_price: float,
    atr_pct: Optional[float],
    side: str,
    min_stop_pct: float = MIN_INITIAL_STOP_PCT,
    atr_multiplier: float = 2.5,
) -> float:
    """
    Calculate INITIAL stop loss for a new entry.
    
    This is the protective stop - should be at least 1.5% from entry
    to avoid getting stopped out on noise.
    
    Args:
        entry_price: Entry price
        atr_pct: ATR as percentage (e.g., 0.02 = 2%)
        side: 'long' or 'short'
        min_stop_pct: Minimum stop distance (default 1.5%)
        atr_multiplier: ATR multiplier (default 2.5x)
    
    Returns:
        Initial stop loss price
    """
    if entry_price <= 0:
        logger.warning(f"[SL_CALC] Invalid entry price: {entry_price}")
        return 0
    
    # Calculate ATR-based stop distance
    atr_distance = atr_multiplier * atr_pct if atr_pct and atr_pct > 0 else 0.0
    
    # Use MAXIMUM of ATR-based or minimum distance
    # This prevents stops too tight on low-volatility coins
    stop_distance = max(atr_distance, min_stop_pct)
    
    # Clamp to reasonable range (1.5% - 10%)
    stop_distance = max(min_stop_pct, min(stop_distance, 0.10))
    
    if side.lower() == 'long':
        return entry_price * (1.0 - stop_distance)
    elif side.lower() == 'short':
        return entry_price * (1.0 + stop_distance)
    else:
        logger.error(f"[SL_CALC] Invalid side: {side}")
        return entry_price


def calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    current_sl: float,
    side: str,
    trail_pct: float = TRAIL_DISTANCE_PCT,
) -> Tuple[float, bool, str]:
    """
    Calculate trailing stop loss for a position in profit.
    
    Rules:
    - Only trail when in profit
    - Only move SL in the profitable direction (never loosen)
    - Trail at fixed distance from current price
    
    Args:
        entry_price: Original entry price
        current_price: Current market price
        current_sl: Current stop loss price
        side: 'long' or 'short'
        trail_pct: Distance to trail (default 0.3%)
    
    Returns:
        (new_sl, should_update, reason)
    """
    if entry_price <= 0 or current_price <= 0:
        return current_sl, False, "invalid_prices"
    
    side_lower = side.lower()
    
    # Calculate profit
    if side_lower == 'long':
        profit_pct = (current_price - entry_price) / entry_price
        in_profit = current_price > entry_price
    else:  # short
        profit_pct = (entry_price - current_price) / entry_price
        in_profit = current_price < entry_price
    
    # Not in profit - don't trail
    if not in_profit:
        return current_sl, False, "not_in_profit"
    
    # Calculate new trailing SL
    if side_lower == 'long':
        # LONG: Trail below current price
        new_sl = current_price * (1.0 - trail_pct)
        
        # Only update if new SL is HIGHER (tighter) than current
        # Never loosen the stop!
        if current_sl <= 0 or new_sl > current_sl:
            return new_sl, True, f"trailing_up_{profit_pct*100:.2f}%_profit"
        else:
            return current_sl, False, "sl_already_tighter"
    
    else:  # SHORT
        # SHORT: Trail above current price
        new_sl = current_price * (1.0 + trail_pct)
        
        # Only update if new SL is LOWER (tighter) than current
        if current_sl <= 0 or new_sl < current_sl:
            return new_sl, True, f"trailing_down_{profit_pct*100:.2f}%_profit"
        else:
            return current_sl, False, "sl_already_tighter"


def validate_stop_loss(
    entry_price: float,
    stop_loss: float,
    side: str,
    current_price: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Validate that stop loss is correct for current position state.
    
    Rules:
    - LONG: SL must be below BOTH entry and current price (or in profit zone)
    - SHORT: SL must be above BOTH entry and current price (or in profit zone)
    
    Returns:
        (is_valid, reason)
    """
    if entry_price <= 0:
        return False, "entry_price_invalid"
    
    if stop_loss <= 0:
        return False, "stop_loss_zero_or_negative"
    
    side_lower = side.lower()
    
    # Default current_price to entry_price if not provided
    if current_price is None:
        current_price = entry_price
    
    if side_lower == 'long':
        # LONG: SL must be below entry (loss protection)
        if stop_loss >= entry_price:
            # Exception: If we're in profit and SL is above entry, that's a trailing stop
            if current_price > entry_price and stop_loss < current_price:
                return True, "valid_trailing_stop_above_entry"
            return False, f"long_sl_above_entry"
        return True, "valid"
    
    elif side_lower == 'short':
        # SHORT: SL must be above entry (loss protection)
        if stop_loss <= entry_price:
            # Exception: If we're in profit and SL is below entry, that's a trailing stop
            if current_price < entry_price and stop_loss > current_price:
                return True, "valid_trailing_stop_below_entry"
            return False, f"short_sl_below_entry"
        return True, "valid"
    
    return False, "invalid_side"


def fix_position_stop_losses(positions: dict, logger_instance=None) -> dict:
    """
    Validate and fix stop losses for all positions.
    
    IMPORTANT: This only fixes BROKEN stops (wrong direction).
    It does NOT reset valid trailing stops!
    
    Args:
        positions: Dict of positions {symbol: position_data}
        logger_instance: Optional logger for output
    
    Returns:
        Dict with corrected positions
    """
    log = logger_instance or logger
    fixed_count = 0
    
    for symbol, pos in positions.items():
        entry_price = pos.get('entry_price', 0)
        current_sl = pos.get('stop_loss', 0)
        current_price = pos.get('current_price', entry_price)  # Use entry if no current
        side = pos.get('side', 'long')
        atr_pct = pos.get('atr_pct')
        
        if entry_price <= 0:
            continue
        
        side_lower = side.lower()
        needs_fix = False
        reason = ""
        
        # Check for BROKEN stops (would immediately trigger OR no stop at all)
        # NOTE: Trailing stops above entry (LONG) or below entry (SHORT) are VALID!
        if side_lower == 'long':
            # LONG with no SL = BROKEN
            if current_sl <= 0:
                needs_fix = True
                reason = "no_stop_loss"
            # LONG with SL >= current price = would immediately trigger (broken)
            elif current_sl >= current_price:
                needs_fix = True
                reason = "long_sl_at_or_above_price"
        
        elif side_lower == 'short':
            # SHORT with no SL = BROKEN
            if current_sl <= 0:
                needs_fix = True
                reason = "no_stop_loss"
            # SHORT with SL <= current price = would immediately trigger (broken)
            elif current_sl <= current_price:
                needs_fix = True
                reason = "short_sl_at_or_below_price"
        
        if needs_fix:
            # Calculate correct initial stop loss
            new_sl = calculate_initial_stop_loss(
                entry_price=entry_price,
                atr_pct=atr_pct,
                side=side
            )
            
            old_distance = abs(current_sl - entry_price) / entry_price * 100 if current_sl > 0 else 0
            new_distance = abs(new_sl - entry_price) / entry_price * 100
            
            log.warning(
                f"[SL_FIX] {symbol} {side.upper()}: "
                f"BROKEN SL ${current_sl:.8f} -> ${new_sl:.8f} ({new_distance:.2f}% from entry) | "
                f"Reason: {reason}"
            )
            
            pos['stop_loss'] = new_sl
            if 'initial_stop_price' not in pos or pos.get('initial_stop_price', 0) <= 0:
                pos['initial_stop_price'] = new_sl
            fixed_count += 1
    
    if fixed_count > 0:
        log.warning(f"[SL_FIX] Fixed {fixed_count} positions with broken stop losses")
    
    return positions


def update_trailing_stops(positions: dict, logger_instance=None) -> dict:
    """
    Update trailing stops for all positions in profit.
    
    This should be called on every price update to trail stops
    and lock in profits.
    
    Args:
        positions: Dict of positions {symbol: position_data}
        logger_instance: Optional logger for output
    
    Returns:
        Dict with updated positions
    """
    log = logger_instance or logger
    trailed_count = 0
    
    for symbol, pos in positions.items():
        entry_price = pos.get('entry_price', 0)
        current_price = pos.get('current_price', 0)
        current_sl = pos.get('stop_loss', 0)
        side = pos.get('side', 'long')
        
        if entry_price <= 0 or current_price <= 0:
            continue
        
        # Calculate trailing stop
        new_sl, should_update, reason = calculate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            current_sl=current_sl,
            side=side
        )
        
        if should_update:
            old_sl = current_sl
            pos['stop_loss'] = new_sl
            
            # Calculate profit locked
            if side.lower() == 'long':
                profit_locked_pct = (new_sl - entry_price) / entry_price * 100
            else:
                profit_locked_pct = (entry_price - new_sl) / entry_price * 100
            
            log.info(
                f"[TRAIL] {symbol} {side.upper()}: "
                f"SL ${old_sl:.6f} -> ${new_sl:.6f} | "
                f"Profit locked: {profit_locked_pct:.2f}% | {reason}"
            )
            trailed_count += 1
    
    return positions


# Backwards compatibility - map old function name to new
calculate_stop_loss = calculate_initial_stop_loss


# Example usage:
if __name__ == "__main__":
    print("=" * 60)
    print("STOP LOSS CALCULATOR - TESTS")
    print("=" * 60)
    
    # Test 1: Initial SL for LONG
    print("\n1. LONG Initial Stop Loss:")
    sl = calculate_initial_stop_loss(entry_price=100, atr_pct=0.006, side='long')
    print(f"   Entry: $100, ATR: 0.6%, SL: ${sl:.2f} ({(100-sl)/100*100:.2f}% below entry)")
    
    # Test 2: Initial SL for SHORT
    print("\n2. SHORT Initial Stop Loss:")
    sl = calculate_initial_stop_loss(entry_price=50, atr_pct=0.01, side='short')
    print(f"   Entry: $50, ATR: 1.0%, SL: ${sl:.2f} ({(sl-50)/50*100:.2f}% above entry)")
    
    # Test 3: Trailing stop for LONG in profit
    print("\n3. LONG Trailing Stop (in profit):")
    new_sl, should_update, reason = calculate_trailing_stop(
        entry_price=100, current_price=105, current_sl=98.5, side='long'
    )
    print(f"   Entry: $100, Current: $105 (+5%), Old SL: $98.5")
    print(f"   New SL: ${new_sl:.2f}, Update: {should_update}, Reason: {reason}")
    
    # Test 4: Trailing stop for SHORT in profit
    print("\n4. SHORT Trailing Stop (in profit):")
    new_sl, should_update, reason = calculate_trailing_stop(
        entry_price=100, current_price=95, current_sl=101.5, side='short'
    )
    print(f"   Entry: $100, Current: $95 (-5%), Old SL: $101.5")
    print(f"   New SL: ${new_sl:.2f}, Update: {should_update}, Reason: {reason}")
    
    # Test 5: Fix broken stops
    print("\n5. Fix Broken Stops:")
    import logging
    logging.basicConfig(level=logging.WARNING)
    test_positions = {
        'BROKEN_LONG': {
            'entry_price': 100, 
            'current_price': 102,
            'stop_loss': 105,  # WRONG: Above entry for LONG
            'side': 'long'
        },
        'GOOD_TRAIL': {
            'entry_price': 100,
            'current_price': 110,
            'stop_loss': 108,  # GOOD: Trailing stop above entry (in profit)
            'side': 'long'
        }
    }
    fix_position_stop_losses(test_positions, logging.getLogger())
    print(f"   BROKEN_LONG SL: ${test_positions['BROKEN_LONG']['stop_loss']:.2f}")
    print(f"   GOOD_TRAIL SL: ${test_positions['GOOD_TRAIL']['stop_loss']:.2f} (unchanged)")
