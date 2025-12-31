"""
Log Filter for Human-Readable Output
====================================

Filters and cleans logs to show only important information.
"""

import logging
import re
from typing import Set
from datetime import datetime, timedelta
from collections import defaultdict


class LogFilter(logging.Filter):
    """
    Filters logs to make them human-readable.
    
    Removes: Debug spam, repeated messages, noise
    Keeps: Important events, errors, trades, decisions
    """
    
    def __init__(self):
        super().__init__()
        
        # Patterns to suppress completely
        self.suppress_patterns = [
            r'\[ML_SCORE_DEBUG\]',
            r'\[POST_SCORE\]',
            r'\[POSITION_SYNC_SUCCESS\]',
            r'Spread: \d+',
            r'ml_confidence:no_model_probabilities.*Pos=3',  # Expected when full
        ]
        
        # Track recent messages to avoid duplicates
        self.recent_messages = defaultdict(list)
        self.condense_interval = 30  # seconds
        
        # Patterns that are IMPORTANT
        self.important_patterns = [
            r'\[ENTRY\]',
            r'\[EXIT\]',
            r'\[WIN\]',
            r'\[LOSS\]',
            r'\[SL_HIT\]',
            r'\[ERROR\]',
            r'\[CRITICAL\]',
            r'\[ML_SL\]',
            r'entered.*position',
            r'closed.*position',
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record.
        
        Returns:
            True if record should be logged, False otherwise
        """
        msg = record.getMessage()
        
        # Always show critical and errors
        if record.levelno >= logging.ERROR:
            return True
        
        # Always show important patterns
        for pattern in self.important_patterns:
            if re.search(pattern, msg, re.IGNORECASE):
                return True
        
        # Suppress specific noise patterns
        for pattern in self.suppress_patterns:
            if re.search(pattern, msg):
                return False
        
        # Suppress repeated "REJECTED" messages (too many)
        if 'REJECTED' in msg and 'max positions reached' in msg:
            # Allow first one, suppress duplicates within condense_interval
            key = 'REJECTED_MAX_POS'
            now = datetime.now()
            
            if key in self.recent_messages:
                last_time = self.recent_messages[key][0]
                if (now - last_time).seconds < self.condense_interval:
                    return False  # Suppress duplicate
            
            self.recent_messages[key] = [now]
            return True
        
        # Suppress repeated position sync messages
        if '[POSITION_SYNC_SUCCESS]' in msg:
            return False
        
        # Suppress repeated score pass messages  (only show significant ones)
        if '[SCORE_PASS]' in msg:
            # Extract score
            score_match = re.search(r'score=(\d+\.?\d*)', msg)
            if score_match:
                score = float(score_match.group(1))
                # Only show high-scoring signals (>75)
                if score < 75:
                    return False
        
        # Default: show the message
        return True


class LogFormatter(logging.Formatter):
    """
    Custom formatter for clean, readable logs.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for readability."""
        
        # Get base message
        msg = record.getMessage()
        
        # Color/highlight important messages
        if record.levelno >= logging.ERROR:
            # Red for errors
            prefix = 'ERROR'
        elif '[ENTRY]' in msg:
            # Green for entries
            prefix = '[+] ENTRY'
        elif '[EXIT]' in msg or '[WIN]' in msg or '[LOSS]' in msg:
            # Important exits
            prefix = '[*] EXIT'
        elif '[ERROR]' in msg or '[CRITICAL]' in msg:
            # Critical
            prefix = '[!] CRITICAL'
        else:
            # Normal
            time_str = record.created
            ts = datetime.fromtimestamp(time_str).strftime('%H:%M:%S')
            
            # Keep timestamp format but make it shorter
            return f"{ts}|{record.levelname[0]}|{msg}"
        
        # For special messages, show more detail
        time_str = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        return f"{time_str} {prefix} {msg}"


def setup_clean_logging():
    """
    Setup logging with filters for clean output.
    
    Usage:
        from log_cleaner import setup_clean_logging
        setup_clean_logging()
    """
    
    # Get root logger
    logger = logging.getLogger()
    
    # Add filter to all handlers
    log_filter = LogFilter()
    
    for handler in logger.handlers:
        handler.addFilter(log_filter)
        # Don't change formatter - keep existing format
    
    logger.info("[LOG_CLEANUP] Logging filters enabled - showing only important events")


# For use in specific loggers
def add_filter_to_logger(logger_name: str, enable: bool = True):
    """Add filter to a specific logger."""
    logger = logging.getLogger(logger_name)
    
    if enable:
        log_filter = LogFilter()
        logger.addFilter(log_filter)
        logger.debug(f"[LOG_CLEANUP] Filter enabled for {logger_name}")
    else:
        # Remove all LogFilter instances
        logger.filters[:] = [f for f in logger.filters if not isinstance(f, LogFilter)]

