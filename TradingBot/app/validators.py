"""
Input validation utilities for the trading bot.
Provides consistent validation across all modules.
"""

from typing import Optional, Tuple, Any, Dict
from decimal import Decimal, InvalidOperation


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_symbol(symbol: str) -> str:
    """
    Validate trading symbol.
    
    Args:
        symbol: Trading symbol to validate
        
    Returns:
        Validated symbol (normalized)
        
    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol:
        raise ValidationError("Symbol cannot be empty")
    
    if not isinstance(symbol, str):
        raise ValidationError(f"Symbol must be string, got {type(symbol).__name__}")
    
    symbol = symbol.strip().upper()
    
    if len(symbol) < 3:
        raise ValidationError(f"Symbol too short: {symbol}")
    
    if len(symbol) > 20:
        raise ValidationError(f"Symbol too long: {symbol}")
    
    # Basic format check (should contain / or be alphanumeric)
    if not (symbol.replace('/', '').replace('_', '').replace('-', '').isalnum()):
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")
    
    return symbol


def validate_price(price: Any, name: str = "price") -> float:
    """
    Validate price value.
    
    Args:
        price: Price to validate
        name: Name of the price field (for error messages)
        
    Returns:
        Validated price as float
        
    Raises:
        ValidationError: If price is invalid
    """
    if price is None:
        raise ValidationError(f"{name} cannot be None")
    
    try:
        price_float = float(price)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be numeric, got {type(price).__name__}")
    
    if price_float <= 0:
        raise ValidationError(f"{name} must be positive, got {price_float}")
    
    if price_float > 1e10:  # Sanity check
        raise ValidationError(f"{name} is unreasonably large: {price_float}")
    
    return price_float


def validate_size(size: Any, name: str = "size") -> float:
    """
    Validate position size.
    
    Args:
        size: Size to validate
        name: Name of the size field (for error messages)
        
    Returns:
        Validated size as float
        
    Raises:
        ValidationError: If size is invalid
    """
    if size is None:
        raise ValidationError(f"{name} cannot be None")
    
    try:
        size_float = float(size)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be numeric, got {type(size).__name__}")
    
    if size_float <= 0:
        raise ValidationError(f"{name} must be positive, got {size_float}")
    
    # Import here to avoid circular dependency
    try:
        from .config import MAX_SIZE_SANITY_CHECK
        max_size = MAX_SIZE_SANITY_CHECK
    except ImportError:
        max_size = 1e6  # Fallback
    
    if size_float > max_size:
        raise ValidationError(f"{name} is unreasonably large: {size_float}")
    
    return size_float


def validate_percentage(value: Any, name: str = "percentage", min_val: float = -100.0, max_val: float = 100.0) -> float:
    """
    Validate percentage value.
    
    Args:
        value: Percentage to validate
        name: Name of the field (for error messages)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated percentage as float
        
    Raises:
        ValidationError: If percentage is invalid
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    
    try:
        pct_float = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be numeric, got {type(value).__name__}")
    
    if pct_float < min_val or pct_float > max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {pct_float}")
    
    return pct_float


def validate_side(side: str) -> str:
    """
    Validate position side.
    
    Args:
        side: Side to validate (long/short)
        
    Returns:
        Validated side (lowercase)
        
    Raises:
        ValidationError: If side is invalid
    """
    if not side:
        raise ValidationError("Side cannot be empty")
    
    if not isinstance(side, str):
        raise ValidationError(f"Side must be string, got {type(side).__name__}")
    
    side_lower = side.lower().strip()
    
    if side_lower not in ('long', 'short'):
        raise ValidationError(f"Side must be 'long' or 'short', got '{side}'")
    
    return side_lower


def validate_stop_loss_take_profit(
    entry_price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    side: str
) -> Tuple[float, float]:
    """
    Validate stop loss and take profit relative to entry price.

    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        side: Position side (long/short)

    Returns:
        Tuple of (validated_stop_loss, validated_take_profit)

    Raises:
        ValidationError: If stop loss or take profit is invalid
    """
    entry_price = validate_price(entry_price, "entry_price")
    side = validate_side(side)

    # Validate stop loss
    if stop_loss is None:
        raise ValidationError("Stop loss is required")

    stop_loss = validate_price(stop_loss, "stop_loss")

    if side == "long" and stop_loss >= entry_price:
        raise ValidationError(f"Stop loss ({stop_loss}) must be below entry price ({entry_price}) for long position")
    elif side == "short" and stop_loss <= entry_price:
        raise ValidationError(f"Stop loss ({stop_loss}) must be above entry price ({entry_price}) for short position")

    # Validate take profit
    if take_profit is None:
        raise ValidationError("Take profit is required")

    take_profit = validate_price(take_profit, "take_profit")

    if side == "long" and take_profit <= entry_price:
        raise ValidationError(f"Take profit ({take_profit}) must be above entry price ({entry_price}) for long position")
    elif side == "short" and take_profit >= entry_price:
        raise ValidationError(f"Take profit ({take_profit}) must be below entry price ({entry_price}) for short position")

    return stop_loss, take_profit


def validate_position_dict(position: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate position dictionary structure.
    
    Args:
        position: Position dictionary to validate
        
    Returns:
        Validated position dictionary
        
    Raises:
        ValidationError: If position dict is invalid
    """
    if not isinstance(position, dict):
        raise ValidationError(f"Position must be dict, got {type(position).__name__}")
    
    required_fields = ['symbol', 'side', 'size', 'entry_price', 'stop_loss', 'take_profit']
    
    for field in required_fields:
        if field not in position:
            raise ValidationError(f"Position missing required field: {field}")
    
    # Validate individual fields
    symbol = validate_symbol(position['symbol'])
    side = validate_side(position['side'])
    size = validate_size(position['size'], "size")
    entry_price = validate_price(position['entry_price'], "entry_price")
    
    # Validate stop loss and take profit relative to entry
    stop_loss, take_profit = validate_stop_loss_take_profit(
        entry_price,
        position.get('stop_loss'),
        position.get('take_profit'),
        side
    )
    
    return {
        'symbol': symbol,
        'side': side,
        'size': size,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        **{k: v for k, v in position.items() if k not in required_fields}
    }


def safe_validate(func, *args, default=None, **kwargs):
    """
    Safely validate input, returning default on error.
    
    Args:
        func: Validation function to call
        *args: Arguments for validation function
        default: Default value to return on error
        **kwargs: Keyword arguments for validation function
        
    Returns:
        Validated value or default
    """
    try:
        return func(*args, **kwargs)
    except ValidationError:
        return default

