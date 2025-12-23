"""
UI Adapter - Intercepts all prints and routes them appropriately.
Ensures no text escapes beneath UI, prevents flickering, and manages output routing.
"""

import sys
import io
from typing import Optional, TextIO
from contextlib import contextmanager
from ..logger import get_logger

logger = get_logger("UIAdapter")


class PrintInterceptor:
    """
    Intercepts print() calls and routes them to logger instead of stdout.
    Prevents print statements from interfering with Rich UI.
    """
    
    def __init__(self, log_level: str = "debug"):
        """
        Initialize print interceptor.
        
        Args:
            log_level: Log level for intercepted prints ("debug", "info", "warning", "error")
        """
        self.log_level = log_level
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.intercepting = False
        self.buffer = io.StringIO()
    
    def start(self):
        """Start intercepting prints."""
        if self.intercepting:
            return
        
        self.intercepting = True
        sys.stdout = self
        sys.stderr = self
    
    def stop(self):
        """Stop intercepting prints."""
        if not self.intercepting:
            return
        
        self.intercepting = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def write(self, text: str):
        """
        Intercept write calls (from print statements).
        
        Args:
            text: Text to write
        """
        if not text.strip():
            return
        
        # Route to logger instead of stdout
        log_method = getattr(logger, self.log_level, logger.debug)
        log_method(f"[STDOUT] {text.strip()}")
    
    def flush(self):
        """Flush buffer (no-op for interceptor)."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class UIAdapter:
    """
    UI Adapter - Manages UI output and print interception.
    Ensures clean UI rendering without interference from print statements.
    """
    
    def __init__(self, enable_print_interception: bool = True):
        """
        Initialize UI adapter.
        
        Args:
            enable_print_interception: Enable print interception (default: True)
        """
        self.enable_print_interception = enable_print_interception
        self.print_interceptor: Optional[PrintInterceptor] = None
        self.ui_active = False
    
    def start(self):
        """Start UI adapter (enable print interception)."""
        if self.enable_print_interception and not self.ui_active:
            self.print_interceptor = PrintInterceptor(log_level="debug")
            self.print_interceptor.start()
            self.ui_active = True
            logger.debug("[UIAdapter] Print interception enabled")
    
    def stop(self):
        """Stop UI adapter (disable print interception)."""
        if self.print_interceptor:
            self.print_interceptor.stop()
            self.print_interceptor = None
            self.ui_active = False
            logger.debug("[UIAdapter] Print interception disabled")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

