"""
Canonical Accounting Module

Single source of truth for all PnL, equity, and balance calculations.
Used by both trading logic and UI to ensure consistency.
"""

from typing import Dict, Any, Tuple, Optional


def calculate_position_pnl(
    position: Dict[str, Any],
    current_price: float
) -> Tuple[float, float]:
    """
    Calculate unrealized PnL for a single position.
    
    Args:
        position: Position dictionary with entry_price, size, side
        current_price: Current market price
        
    Returns:
        (pnl_absolute, pnl_percent)
        - pnl_absolute: PnL in quote currency (e.g. USDT)
        - pnl_percent: PnL as percentage of entry value
    """
    entry_price = position.get('entry_price', 0)
    size = position.get('size', 0)
    side = position.get('side', '').lower()
    
    if not all([entry_price, size, current_price]) or entry_price <= 0 or current_price <= 0:
        return 0.0, 0.0
    
    # Calculate PnL based on side
    if side == 'long':
        # Long: profit when price goes up
        price_diff = current_price - entry_price
        pnl_abs = price_diff * size
        pnl_pct = (price_diff / entry_price) * 100.0
    elif side == 'short':
        # Short: profit when price goes down
        price_diff = entry_price - current_price
        pnl_abs = price_diff * size
        pnl_pct = (price_diff / entry_price) * 100.0
    else:
        return 0.0, 0.0
    
    return pnl_abs, pnl_pct


def calculate_total_unrealized_pnl(
    positions: Dict[str, Dict[str, Any]],
    price_getter: callable
) -> float:
    """
    Calculate total unrealized PnL across all positions.
    
    Args:
        positions: Dictionary of position dicts (keyed by symbol)
        price_getter: Function that takes symbol and returns current price
        
    Returns:
        Total unrealized PnL in quote currency
    """
    total = 0.0
    for symbol, position in positions.items():
        try:
            current_price = price_getter(symbol)
            if current_price:
                pnl_abs, _ = calculate_position_pnl(position, current_price)
                total += pnl_abs
        except Exception:
            # Skip positions with pricing errors
            continue
    
    return total


def calculate_equity(
    start_balance: float,
    realized_pnl: float,
    realized_funding: float,
    unrealized_pnl: float
) -> float:
    """
    Calculate current total equity.
    
    Formula: Equity = Starting Balance + Realized PnL + Realized Funding + Unrealized PnL
    
    Args:
        start_balance: Initial account balance
        realized_pnl: Cumulative realized PnL (net of fees)
        realized_funding: Cumulative funding payments
        unrealized_pnl: Current unrealized PnL from open positions
        
    Returns:
        Total current equity
    """
    return start_balance + realized_pnl + realized_funding + unrealized_pnl


def calculate_balance(
    start_balance: float,
    realized_pnl: float,
    realized_funding: float
) -> float:
    """
    Calculate current balance (excluding unrealized PnL).
    
    Formula: Balance = Starting Balance + Realized PnL + Realized Funding
    
    Args:
        start_balance: Initial account balance
        realized_pnl: Cumulative realized PnL (net of fees)
        realized_funding: Cumulative funding payments
        
    Returns:
        Current balance (realized only)
    """
    return start_balance + realized_pnl + realized_funding


def get_current_price_from_bot(bot, symbol: str) -> Optional[float]:
    """
    Helper to get current market price for a symbol from bot's data sources.
    
    Tries multiple sources in order:
    1. REPLAY MODE: Replay feed
    2. Universe stats (mark price)
    3. Ticker cache
    4. Fast storage
    5. Falls back to entry price from position
    
    Args:
        bot: ScalperBot instance
        symbol: Trading symbol
        
    Returns:
        Current market price or None
    """
    # REPLAY MODE: Get price from replay feed first
    if getattr(bot, 'replay_mode', False) and getattr(bot, 'replay_feed', None):
        try:
            price = bot.replay_feed.get_current_price(symbol)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
    
    # Try universe stats first
    try:
        universe = getattr(bot, 'universe', None)
        if universe:
            stats = getattr(universe, 'stats', {})
            symbol_stats = stats.get(symbol)
            if symbol_stats:
                mark_price = getattr(symbol_stats, 'mark', None)
                if mark_price and mark_price > 0:
                    return float(mark_price)
                last_price = getattr(symbol_stats, 'last', None)
                if last_price and last_price > 0:
                    return float(last_price)
    except Exception:
        pass
    
    # Try ticker cache
    try:
        ticker_cache = getattr(bot, 'ticker_cache', None)
        if ticker_cache:
            cached = ticker_cache.get(symbol, max_age=5.0)
            if cached:
                if cached.mark and cached.mark > 0:
                    return float(cached.mark)
                if cached.last and cached.last > 0:
                    return float(cached.last)
    except Exception:
        pass
    
    # Try fast storage
    try:
        fast_storage = getattr(bot, 'fast_storage', None)
        if fast_storage:
            storage_data = fast_storage.get(symbol, max_age=5.0)
            if storage_data:
                if hasattr(storage_data, 'mark') and storage_data.mark > 0:
                    return float(storage_data.mark)
                if hasattr(storage_data, 'last') and storage_data.last > 0:
                    return float(storage_data.last)
    except Exception:
        pass
    
    # Fallback to position entry price
    try:
        positions = getattr(bot, 'positions', {})
        position = positions.get(symbol)
        if position:
            entry_price = position.get('entry_price', 0)
            if entry_price > 0:
                return float(entry_price)
    except Exception:
        pass
    
    return None


def validate_accounting_consistency(
    positions_table_pnl: float,
    performance_unrealized: float,
    tolerance: float = 0.01
) -> Tuple[bool, float, str]:
    """
    Validate that position table PnL matches performance panel unrealized PnL.
    
    Args:
        positions_table_pnl: Sum of PnL from positions table
        performance_unrealized: Unrealized PnL shown in performance panel
        tolerance: Maximum acceptable difference (default 0.01 = 1 cent)
        
    Returns:
        (is_consistent, difference, message)
    """
    diff = abs(positions_table_pnl - performance_unrealized)
    is_consistent = diff <= tolerance
    
    if is_consistent:
        message = f"[OK] Accounting consistent (diff: ${diff:.4f})"
    else:
        message = f"[!] Accounting MISMATCH: Table=${positions_table_pnl:.2f}, Performance=${performance_unrealized:.2f}, Diff=${diff:.2f}"
    
    return is_consistent, diff, message

