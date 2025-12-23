"""
Utility Module - Time, math, and helper functions.
Common utilities used throughout the bot.
"""

from .time_utils import now_ms, format_duration, parse_duration
from .math_utils import safe_div, clamp, round_to_precision

__all__ = [
    "now_ms",
    "format_duration",
    "parse_duration",
    "safe_div",
    "clamp",
    "round_to_precision",
]

