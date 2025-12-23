"""
Structured logging infrastructure for the trading bot.
Provides consistent logging with levels, context, and file output.

CRITICAL: When UI v2 is active, NO console output is allowed.
All logs go to file + in-memory buffer for UI display.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from logging.handlers import RotatingFileHandler
from collections import deque
from threading import Lock
import time


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Custom RotatingFileHandler that handles Windows file locking gracefully.
    
    Windows can lock log files, preventing rotation. This handler:
    - Retries rotation with exponential backoff
    - Gracefully continues if rotation fails
    - Prevents crashes from PermissionError
    """
    
    def doRollover(self):
        """
        Override doRollover to handle Windows file locking gracefully.
        """
        try:
            # Try standard rotation
            super().doRollover()
        except PermissionError as e:
            # Windows file locking issue - retry with backoff
            max_retries = 3
            retry_delay = 0.1  # Start with 100ms
            
            for attempt in range(max_retries):
                try:
                    time.sleep(retry_delay)
                    # Close and reopen the stream to release the lock
                    if self.stream:
                        self.stream.close()
                        self.stream = None
                    
                    # Try rotation again
                    super().doRollover()
                    return  # Success
                except PermissionError:
                    if attempt < max_retries - 1:
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed - log error but continue
                        # Don't crash the bot due to log rotation failure
                        import sys
                        print(f"[LOGGER] WARNING: Could not rotate log file after {max_retries} attempts: {e}", file=sys.stderr)
                        # Continue without rotation - logs will just append to current file
                        return
        except Exception as e:
            # Any other error - log but don't crash
            import sys
            print(f"[LOGGER] WARNING: Log rotation error (non-fatal): {e}", file=sys.stderr)
            # Continue without rotation
            return


class InMemoryLogBuffer:
    """
    Thread-safe in-memory log buffer for UI display.
    Stores recent log messages that can be shown in the UI log panel.
    """
    
    def __init__(self, maxlen: int = 50):
        """
        Initialize log buffer.
        
        Args:
            maxlen: Maximum number of log entries to keep
        """
        self._buffer = deque(maxlen=maxlen)
        self._lock = Lock()
    
    def append(self, timestamp: datetime, level: str, message: str):
        """
        Add log entry to buffer.
        
        Args:
            timestamp: Log timestamp
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
        """
        with self._lock:
            self._buffer.append({
                'timestamp': timestamp,
                'level': level,
                'message': message
            })
    
    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries (newest first)
        """
        with self._lock:
            # Return a copy to prevent concurrent modification
            entries = list(self._buffer)
            return entries[-limit:] if entries else []
    
    def clear(self):
        """Clear all log entries."""
        with self._lock:
            self._buffer.clear()


class UILogHandler(logging.Handler):
    """
    Custom logging handler that filters and routes ONLY high-signal messages to UI LOG panel.
    
    Only shows:
    - Loop summaries (contains "SIG=" or "SIG:")
    - Risk warnings (max positions, correlation, drawdown)
    - System warnings/errors (API timeouts, exchange errors)
    - Major regime/mode changes (BTC trend, volatility, strategy)
    
    Excludes:
    - DEBUG messages (always)
    - Per-signal "Unicorn check" spam
    - Per-signal "SIGNAL GENERATED" spam
    - PRS debug per position
    - Any messages that fire dozens of times per minute
    """
    
    # Patterns that should appear in LOG panel
    APPROVED_PATTERNS = [
        r'SIG[=:]',  # Loop summaries (SIG=3 or SIG:3)
        r'max positions|position limit|positions reached',
        r'correlation.*block|RJ.*correlation',
        r'drawdown.*threshold|circuit breaker',
        r'API.*timeout|API.*retry|exchange.*error|exchange.*outage',
        r'BTC.*trend.*flip|BTC.*trend.*UP|BTC.*trend.*DOWN',
        r'volatility.*regime|VOLATILITY.*LOW|VOLATILITY.*HIGH',
        r'strategy.*switch|mode.*switch|DRY.*LIVE|LIVE.*DRY',
        r'STARTUP|SHUTDOWN',
        r'FATAL|CRITICAL',
        r'ERROR.*Exit failed',  # Exit failures (important)
    ]
    
    # Patterns that should be EXCLUDED from LOG panel (even if INFO/WARNING)
    EXCLUDED_PATTERNS = [
        r'Unicorn check',
        r'SIGNAL GENERATED',
        r'score.*dump|final_score.*=',
        r'PRS.*debug|\[PRS\].*position',
        r'Symbol.*processed|symbol.*scanned',
        r'Signal.*confirmed.*ready',  # Too frequent
        r'Signal.*waiting.*confirmation',  # Too frequent
        r'Signal.*blocked.*position_manager',  # Too frequent
        r'Signal.*REJECTED.*generator',  # Too frequent
        r'⚠️ NO STATS',  # Debug only
    ]
    
    def __init__(self, buffer: InMemoryLogBuffer):
        """
        Initialize UI log handler.
        
        Args:
            buffer: InMemoryLogBuffer instance
        """
        super().__init__()
        self.buffer = buffer
        import re
        self.approved_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.APPROVED_PATTERNS]
        self.excluded_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.EXCLUDED_PATTERNS]
        self.setLevel(logging.INFO)  # Only INFO/WARNING/ERROR, never DEBUG
    
    def emit(self, record: logging.LogRecord):
        """
        Emit log record to in-memory buffer ONLY if it matches approved patterns.
        
        Args:
            record: Log record
        """
        # Never show DEBUG messages
        if record.levelno < logging.INFO:
            return
        
        try:
            message = self.format(record)
            
            # Check if message matches excluded patterns (explicitly block spam)
            for pattern in self.excluded_regex:
                if pattern.search(message):
                    return  # Excluded - don't show in UI
            
            # Check if message matches approved patterns (high-signal only)
            is_approved = False
            for pattern in self.approved_regex:
                if pattern.search(message):
                    is_approved = True
                    break
            
            # Also allow all ERROR/CRITICAL messages (always important)
            if record.levelno >= logging.ERROR:
                is_approved = True
            
            # Only add to buffer if approved
            if is_approved:
                timestamp = datetime.fromtimestamp(record.created)
                level = record.levelname
                self.buffer.append(timestamp, level, message)
        except Exception:
            # Silently fail to prevent logging loops
            pass


class InMemoryHandler(logging.Handler):
    """
    DEPRECATED: Legacy in-memory handler (replaced by UILogHandler).
    Kept for backward compatibility.
    """
    
    def __init__(self, buffer: InMemoryLogBuffer):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record: logging.LogRecord):
        try:
            timestamp = datetime.fromtimestamp(record.created)
            level = record.levelname
            message = self.format(record)
            self.buffer.append(timestamp, level, message)
        except Exception:
            pass


class StructuredLogger:
    """
    Structured logger with file and optional in-memory output.
    NO console output to prevent interference with Rich UI.
    """
    
    def __init__(
        self,
        name: str = "ScalperBot",
        log_dir: str = "logs",
        log_file: str = "bot.log",
        level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        enable_console: bool = True,  # ENABLED by default for Static UI visibility
        in_memory_buffer: Optional[InMemoryLogBuffer] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_file: Log file name
            level: Overall log level
            file_level: File log level
            enable_console: Enable console output (Safe for Static UI)
            in_memory_buffer: Optional in-memory buffer for UI display
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Remove any existing handlers
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
        # ROTATING FILE HANDLER (primary output) - LOGGING V2: Improved rotation
        log_path = self.log_dir / log_file
        max_bytes = int(os.getenv("LOG_MAX_BYTES", "20971520"))  # 20MB default (was 10MB)
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "20"))  # Keep 20 backups (was 5)
        
        # Use SafeRotatingFileHandler to handle Windows file locking gracefully
        file_handler = SafeRotatingFileHandler(
            log_path,
            encoding='utf-8',
            maxBytes=max_bytes,
            backupCount=backup_count,
            mode='a'  # Append mode for rotation
        )
        file_handler.setLevel(logging.INFO)  # File handler at INFO level (DEBUG goes to file only if file_level is DEBUG, but we want compact INFO+)
        # Very compact formatter for file handler
        file_formatter = logging.Formatter(
            fmt='%(asctime)s|%(levelname).1s|%(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # UI LOG HANDLER (for LOG panel display - FILTERED, high-signal only)
        if in_memory_buffer:
            ui_handler = UILogHandler(in_memory_buffer)
            ui_handler.setLevel(logging.INFO)  # Only INFO/WARNING/ERROR (no DEBUG)
            ui_formatter = logging.Formatter(
                '%(message)s',  # Just the message (no level prefix, already filtered)
                datefmt='%H:%M:%S'
            )
            ui_handler.setFormatter(ui_formatter)
            self.logger.addHandler(ui_handler)
        
        # CONSOLE HANDLER (DISABLED by default to prevent UI interference)
        # Only enable if explicitly requested (e.g., for debugging without UI)
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Store log path for reference
        self.log_file_path = log_path
        self.in_memory_buffer = in_memory_buffer
    
    def debug(self, message, *args, **kwargs):
        """Log debug message with optional context."""
        # Support both standard logging format (msg % args) and context kwargs
        if args:
            # Standard logging format: logger.info("Exit: %s", reason)
            self.logger.debug(message, *args, **kwargs)
        elif kwargs:
            # Context format: logger.info("Exit", reason=reason)
            self._log(logging.DEBUG, message, **kwargs)
        else:
            # Plain message: logger.info("Exit")
            self.logger.debug(message)
    
    def info(self, message, *args, **kwargs):
        """Log info message with optional context."""
        # Support both standard logging format (msg % args) and context kwargs
        if args:
            # Standard logging format: logger.info("Exit: %s", reason)
            self.logger.info(message, *args, **kwargs)
        elif kwargs:
            # Context format: logger.info("Exit", reason=reason)
            self._log(logging.INFO, message, **kwargs)
        else:
            # Plain message: logger.info("Exit")
            self.logger.info(message)
    
    def warning(self, message, *args, **kwargs):
        """Log warning message with optional context."""
        # Support both standard logging format (msg % args) and context kwargs
        if args:
            # Standard logging format: logger.warning("Exit: %s", reason)
            self.logger.warning(message, *args, **kwargs)
        elif kwargs:
            # Context format: logger.warning("Exit", reason=reason)
            self._log(logging.WARNING, message, **kwargs)
        else:
            # Plain message: logger.warning("Exit")
            self.logger.warning(message)
    
    def error(self, message, *args, **kwargs):
        """Log error message with optional context."""
        # Support both standard logging format (msg % args) and context kwargs
        if args:
            # Standard logging format: logger.error("Exit: %s", reason)
            self.logger.error(message, *args, **kwargs)
        elif kwargs:
            # Context format: logger.error("Exit", reason=reason)
            self._log(logging.ERROR, message, **kwargs)
        else:
            # Plain message: logger.error("Exit")
            self.logger.error(message)
    
    def critical(self, message, *args, **kwargs):
        """Log critical message with optional context."""
        # Support both standard logging format (msg % args) and context kwargs
        if args:
            # Standard logging format: logger.critical("Exit: %s", reason)
            self.logger.critical(message, *args, **kwargs)
        elif kwargs:
            # Context format: logger.critical("Exit", reason=reason)
            self._log(logging.CRITICAL, message, **kwargs)
        else:
            # Plain message: logger.critical("Exit")
            self.logger.critical(message)
    
    def exception(self, message, *args, **kwargs):
        """Log exception with traceback."""
        # Support both standard logging format (msg % args) and context kwargs
        exc_info = kwargs.pop('exc_info', True)
        if args:
            # Standard logging format: logger.exception("Exit: %s", reason)
            self.logger.exception(message, *args, exc_info=exc_info, **kwargs)
        elif kwargs:
            # Context format: logger.exception("Exit", reason=reason)
            self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
        else:
            # Plain message: logger.exception("Exit")
            self.logger.exception(message, exc_info=exc_info)
    
    def isEnabledFor(self, level: int) -> bool:
        """
        Check if logging is enabled for the given level.
        OPTIMIZATION: Use this to avoid expensive message formatting when logging is disabled.
        
        Example:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Expensive computation: {expensive_function()}")
        
        Args:
            level: Log level to check (logging.DEBUG, logging.INFO, etc.)
            
        Returns:
            True if logging is enabled for this level
        """
        return self.logger.isEnabledFor(level)
    
    def _log(self, level: int, message: str, exc_info: bool = False, **context: Any):
        """
        Internal logging method with context support.
        
        Args:
            level: Log level
            message: Log message
            exc_info: Include exception info
            **context: Additional context fields
        """
        if context:
            # Format context as key=value pairs
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            full_message = f"{message} | {context_str}"
        else:
            full_message = message
        
        self.logger.log(level, full_message, exc_info=exc_info)
    
    def log_trade(
        self,
        action: str,
        symbol: str,
        side: Optional[str] = None,
        price: Optional[float] = None,
        size: Optional[float] = None,
        pnl: Optional[float] = None,
        reason: Optional[str] = None,
        **extra: Any
    ):
        """
        Log trade-related event with structured data.
        
        Args:
            action: Trade action (entry, exit, partial_tp, etc.)
            symbol: Trading symbol
            side: Position side (long/short)
            price: Trade price
            size: Trade size
            pnl: Profit/loss
            reason: Reason for action
            **extra: Additional fields
        """
        context = {
            'action': action,
            'symbol': symbol,
            **extra
        }
        
        if side:
            context['side'] = side
        if price is not None:
            context['price'] = price
        if size is not None:
            context['size'] = size
        if pnl is not None:
            context['pnl'] = pnl
        if reason:
            context['reason'] = reason
        
        self.info(f"TRADE: {action.upper()}", **context)
    
    def log_signal(
        self,
        symbol: str,
        side: str,
        score: float,
        action: str,
        reason: Optional[str] = None,
        **extra: Any
    ):
        """
        Log signal decision with structured data.
        
        Args:
            symbol: Trading symbol
            side: Signal side (long/short)
            score: Signal score
            action: Action taken (approved/rejected)
            reason: Rejection reason if rejected
            **extra: Additional fields
        """
        context = {
            'symbol': symbol,
            'side': side,
            'score': score,
            'action': action,
            **extra
        }
        
        if reason:
            context['reason'] = reason
        
        self.info(f"SIGNAL: {action.upper()}", **context)
    
    def log_error_with_context(
        self,
        operation: str,
        error: Exception,
        symbol: Optional[str] = None,
        **context: Any
    ):
        """
        Log error with full context for debugging.
        
        Args:
            operation: Operation that failed
            error: Exception object
            symbol: Trading symbol if applicable
            **context: Additional context
        """
        error_context = {
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            **context
        }
        
        if symbol:
            error_context['symbol'] = symbol
        
        self.exception(f"ERROR in {operation}", **error_context)


class DiaryLogger:
    """
    Minimalist summary logger (diary-style) that writes at low frequency.
    Writes to logs/diary.log with format:
    "YYYY-MM-DD HH:MM:SS | eq=$... | pnl_day=$... | open_pos=N | used_risk=..% | WR=..% | PF=.."
    
    This file must be small, never spam, and never crash the bot.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize diary logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.diary_path = self.log_dir / "diary.log"
        self._last_write_time = 0.0
        self._min_interval = 600.0  # Minimum 10 minutes between writes
        self._lock = Lock()
    
    def write_summary(
        self,
        equity: float,
        pnl_day: float,
        open_positions: int,
        used_risk_pct: float,
        win_rate_pct: float = None,
        profit_factor: float = None,
        **extra: Any
    ):
        """
        Write a summary line to diary.log.
        
        Args:
            equity: Current equity
            pnl_day: PnL for the day
            open_positions: Number of open positions
            used_risk_pct: Risk budget used (percentage)
            win_rate_pct: Win rate percentage (optional)
            profit_factor: Profit factor (optional)
            **extra: Additional fields to include
        """
        import time
        current_time = time.time()
        
        # Rate limit: only write every 10 minutes minimum
        with self._lock:
            if current_time - self._last_write_time < self._min_interval:
                return
            
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Build summary line
                parts = [f"{timestamp}"]
                parts.append(f"eq=${equity:.2f}")
                parts.append(f"pnl_day=${pnl_day:+.2f}")
                parts.append(f"open_pos={open_positions}")
                parts.append(f"used_risk={used_risk_pct:.1f}%")
                
                if win_rate_pct is not None:
                    parts.append(f"WR={win_rate_pct:.1f}%")
                
                if profit_factor is not None:
                    parts.append(f"PF={profit_factor:.2f}")
                
                # Add extra fields
                for key, value in extra.items():
                    if isinstance(value, (int, float)):
                        parts.append(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}")
                    else:
                        parts.append(f"{key}={value}")
                
                summary_line = " | ".join(parts)
                
                # Write to diary.log (append mode)
                with open(self.diary_path, 'a', encoding='utf-8') as f:
                    f.write(summary_line + '\n')
                
                self._last_write_time = current_time
                
            except Exception:
                # Silently fail to prevent diary logger from crashing the bot
                pass


# Global instances cache
_loggers: Dict[str, StructuredLogger] = {}
_log_buffer: Optional[InMemoryLogBuffer] = None
_diary_logger: Optional[DiaryLogger] = None


def get_log_buffer() -> InMemoryLogBuffer:
    """
    Get or create global in-memory log buffer.
    Used by UI to display recent logs.
    
    Returns:
        InMemoryLogBuffer instance
    """
    global _log_buffer
    
    if _log_buffer is None:
        _log_buffer = InMemoryLogBuffer(maxlen=100)
    
    return _log_buffer


def get_logger(name: str = "ScalperBot", enable_console: bool = False) -> StructuredLogger:
    """
    Get or create global logger instance for the given name.
    Supports multiple named loggers sharing the same file configuration.
    
    Args:
        name: Logger name
        enable_console: Enable console output (default: False for UI safety)
        
    Returns:
        StructuredLogger instance
    """
    global _loggers
    
    # Return existing logger if already created
    if name in _loggers:
        logger = _loggers[name]
        # Update console handler if requested and not present
        if enable_console:
            has_console = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.logger.handlers)
            if not has_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                logger.logger.addHandler(console_handler)
        return logger
    
    # Determine log level from environment
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)

    # Generate timestamped log filename: bot_YYYY-MM-DD_HH-MM-SS.log
    # Use the SAME log file for all loggers in the same run if possible
    # For simplicity, we regenerate name if not cached, but that's fine as they append.
    # Ideally we'd cache the filename.
    if not hasattr(get_logger, "_log_file"):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        get_logger._log_file = f"bot_{timestamp}.log"

    log_file = get_logger._log_file

    # Get global log buffer
    log_buffer = get_log_buffer()

    logger_instance = StructuredLogger(
        name=name,
        log_file=log_file,
        level=log_level,
        file_level=logging.DEBUG,  # Always log everything to file
        enable_console=enable_console,  # Use passed parameter
        in_memory_buffer=log_buffer
    )

    _loggers[name] = logger_instance

    return logger_instance


def get_diary_logger(log_dir: str = "logs") -> DiaryLogger:
    """
    Get or create global diary logger instance.
    
    Args:
        log_dir: Directory for diary log file
        
    Returns:
        DiaryLogger instance
    """
    global _diary_logger
    
    if _diary_logger is None:
        _diary_logger = DiaryLogger(log_dir=log_dir)
    
    return _diary_logger


def disable_external_loggers():
    """
    Disable console output from external libraries (ccxt, httpx, etc.).
    Routes their logs to file only.
    """
    # Disable ccxt verbose logging
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('ccxt').propagate = False
    
    # Disable httpx/httpcore verbose logging
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # Disable urllib3 verbose logging
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Remove all StreamHandlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)
