"""
Time utilities - Common time-related helper functions.
"""

import time
from typing import Optional


def now_ms() -> int:
    """
    Get current time in milliseconds.
    
    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def parse_duration(duration_str: str) -> Optional[float]:
    """
    Parse duration string to seconds.
    
    Args:
        duration_str: Duration string (e.g., "1h 23m 45s", "30m", "120s")
        
    Returns:
        Duration in seconds or None if invalid
    """
    try:
        parts = duration_str.split()
        total_seconds = 0.0
        
        for part in parts:
            if part.endswith('h'):
                total_seconds += float(part[:-1]) * 3600
            elif part.endswith('m'):
                total_seconds += float(part[:-1]) * 60
            elif part.endswith('s'):
                total_seconds += float(part[:-1])
            else:
                # Assume seconds if no suffix
                total_seconds += float(part)
        
        return total_seconds
    except Exception:
        return None

