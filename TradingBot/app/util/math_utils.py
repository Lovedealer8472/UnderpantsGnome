"""
Math utilities - Common mathematical helper functions.
"""

from typing import Union


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that returns default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def round_to_precision(value: float, precision: int = 2) -> float:
    """
    Round value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    return round(value, precision)

