"""
Position management utilities for canonical position updates.

Provides safe, atomic position update functions that prevent double-deletes,
missing position errors, and race conditions.
"""

from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime


def update_position_after_exit(
    positions: Dict[str, Dict[str, Any]],
    positions_set: set,
    symbol: str,
    new_size: float,
    pnl_value: float = 0.0,
    pnl_pct: float = 0.0,
    reason: str = "closed"
) -> Tuple[bool, str]:
    """
    Canonical function to update position after exit.
    
    This is the ONLY function that should modify positions dict after exits.
    Prevents double-deletes, missing position errors, and race conditions.
    
    Args:
        positions: Positions dictionary (modified in-place)
        positions_set: Positions set for O(1) lookups (modified in-place)
        symbol: Position symbol
        new_size: New position size after exit
            - If 0, position is CLOSED → remove from dict
            - If > 0, position is PARTIAL → update size only
        pnl_value: PnL value (for logging)
        pnl_pct: PnL percentage (for logging)
        reason: Exit reason (for logging)
        
    Returns:
        Tuple of (success: bool, message: str)
        - success: True if position was updated/deleted successfully
        - message: Human-readable status message
    """
    # CRITICAL: Check if position exists
    if symbol not in positions:
        return False, f"Position {symbol} not found (already closed?)"
    
    position = positions[symbol]
    old_size = position.get('size', 0)
    
    # CRITICAL: Validate old size
    if old_size <= 0:
        # Position already has zero size (invalid state)
        # Clean up invalid position
        del positions[symbol]
        if symbol in positions_set:
            positions_set.discard(symbol)
        return False, f"Position {symbol} had zero size (cleaned up)"
    
    # Update position based on new_size
    if new_size <= 0:
        # FULL EXIT: Remove position
        del positions[symbol]
        if symbol in positions_set:
            positions_set.discard(symbol)
        
        pnl_str = f"${pnl_value:+.2f}" if pnl_value != 0 else "$0.00"
        pnl_pct_str = f"({pnl_pct:+.2f}%)" if pnl_pct != 0 else ""
        
        return True, f"EXIT {symbol}  {pnl_str} {pnl_pct_str}"
    else:
        # PARTIAL EXIT: Update size only
        position['size'] = new_size
        
        exit_size_pct = ((old_size - new_size) / old_size * 100) if old_size > 0 else 0
        pnl_str = f"${pnl_value:+.2f}" if pnl_value != 0 else "$0.00"
        pnl_pct_str = f"({pnl_pct:+.2f}%)" if pnl_pct != 0 else ""
        
        return True, f"PARTIAL_EXIT {symbol} {exit_size_pct:.0f}%  {pnl_str} {pnl_pct_str}"


def validate_position_exists(
    positions: Dict[str, Dict[str, Any]],
    symbol: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a position exists and is valid.
    
    Args:
        positions: Positions dictionary
        symbol: Position symbol
        
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    if symbol not in positions:
        return False, f"Position {symbol} not found"
    
    position = positions[symbol]
    position_size = position.get('size', 0)
    
    if position_size <= 0:
        return False, f"Position {symbol} has zero size"
    
    return True, None


# CANONICAL EXIT TRACKING: Per-loop tracker for self-check
# Tracks which exits actually succeeded and created DecisionEvents in the current loop
_loop_exit_events: List[Tuple[str, str]] = []  # List of (symbol, action) tuples for this loop


def reset_loop_exit_tracker():
    """Reset the per-loop exit event tracker. Call at start of each loop."""
    global _loop_exit_events
    _loop_exit_events.clear()


def record_loop_exit(symbol: str, action: str):
    """
    Record that an exit/partial-exit succeeded and created a DecisionEvent in this loop.
    
    Args:
        symbol: Position symbol
        action: Exit action ("EXIT", "PARTIAL_EXIT", "SCALE_OUT")
    """
    global _loop_exit_events
    _loop_exit_events.append((symbol, action))


def get_loop_exit_events() -> List[Tuple[str, str]]:
    """
    Get all exit events recorded in the current loop.
    
    Returns:
        List of (symbol, action) tuples
    """
    return list(_loop_exit_events)


def apply_exit_and_log(
    positions: Dict[str, Dict[str, Any]],
    positions_set: set,
    symbol: str,
    new_size: float,
    action: str,
    exit_price: float,
    entry_price: float,
    entry_time: float,
    exit_time: float,
    exit_size: float,
    size_before: float,
    size_after: float,
    side: str,
    pnl_value: float,
    pnl_pct: float,
    gross_pnl: float,
    net_pnl: float,
    total_costs: float,
    reason: str,
    prs: Optional[float],
    was_win: bool,
    is_unicorn: bool,
    bot_instance: Any
) -> Tuple[bool, bool]:
    """
    CANONICAL EXIT PATH: Atomically update position AND log DecisionEvent.
    
    This is the ONLY function that should both update positions dict after exits
    AND create DecisionEvents. Ensures every successful exit/partial-exit produces
    exactly one DecisionEvent.
    
    Args:
        positions: Positions dictionary (modified in-place)
        positions_set: Positions set for O(1) lookups (modified in-place)
        symbol: Position symbol
        new_size: New position size after exit
            - If 0, position is CLOSED → remove from dict
            - If > 0, position is PARTIAL → update size only
        action: Exit action ("EXIT", "PARTIAL_EXIT", "SCALE_OUT")
        exit_price: Exit price
        entry_price: Entry price
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        exit_size: Size of exit (base currency)
        size_before: Position size before exit
        size_after: Position size after exit
        side: Position side ("LONG" or "SHORT")
        pnl_value: Net PnL value
        pnl_pct: PnL percentage
        gross_pnl: Gross PnL
        net_pnl: Net PnL
        total_costs: Total trading costs
        reason: Exit reason
        prs: Position Recovery Score (if available)
        was_win: Whether exit was profitable
        is_unicorn: Whether position was a unicorn signal
        bot_instance: Bot instance (for logging)
        
    Returns:
        Tuple of (success: bool, event_created: bool)
        - success: True if position was updated/deleted successfully
        - event_created: True if DecisionEvent was created and logged
    """
    from .decision_event import DecisionEvent, log_trade_decision
    
    # TURBO: Record outcome for streak tracking
    try:
        from .learner.integration import get_learner, is_learner_enabled
        if is_learner_enabled():
            learner = get_learner()
            learner.record_outcome(symbol, was_win)
    except Exception:
        pass
    
    # STEP 0: Get stop_loss from position BEFORE it's deleted (for R calculation)
    stop_loss = None
    if symbol in positions:
        # CRITICAL FIX: Use initial_stop_price for R calculation if available (Standard R definition)
        # Fallback to current stop_loss if initial not found
        stop_loss = positions[symbol].get('initial_stop_price', positions[symbol].get('stop_loss'))
    
    # STEP 1: Update position atomically
    success, status_msg = update_position_after_exit(
        positions=positions,
        positions_set=positions_set,
        symbol=symbol,
        new_size=new_size,
        pnl_value=pnl_value,
        pnl_pct=pnl_pct,
        reason=reason
    )
    
    if not success:
        # Position update failed (already deleted or invalid)
        # Do not create DecisionEvent if position update failed
        return False, False
    
    # STEP 2: Create and log DecisionEvent (only if position update succeeded)
    try:
        duration = exit_time - entry_time
        
        decision = DecisionEvent(
            timestamp=exit_time,
            action=action,
            symbol=symbol,
            side=side.upper(),
            price=exit_price,
            entry_price=entry_price,
            exit_price=exit_price,
            size=exit_size,
            size_before=size_before,
            size_after=size_after,
            pnl_value=pnl_value,
            pnl_pct=pnl_pct,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_costs=total_costs,
            reason=reason,
            prs=prs,
            duration_sec=duration,
            was_win=was_win,
            is_unicorn=is_unicorn,
            stop_loss=stop_loss  # For R calculation in exit log
        )
        
        # Log using unified system (writes to decisions.jsonl, recent_trades, and Recent Activity buffer)
        log_trade_decision(decision, bot_instance=bot_instance)
        
        # Success: position updated and DecisionEvent logged
        return True, True
        
    except Exception as e:
        # Logging failed, but position was already updated
        # This is a problem - position updated but no DecisionEvent
        # Log error but don't fail (position update already succeeded)
        from .logger import get_logger
        logger = get_logger("PositionUtils")
        logger.error(f"Failed to create DecisionEvent for {symbol}: {e}")
        return True, False  # Position updated, but event not created
