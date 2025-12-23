"""
Signal Registry - Centralized signal management.
Provides signal queue and tracking functionality.
"""

from typing import Dict, Any, Optional, List
from collections import deque
from ..logger import get_logger

logger = get_logger("SignalRegistry")


class SignalRegistry:
    """
    Centralized signal registry.
    Provides signal queue and tracking functionality.
    """
    
    def __init__(self, max_signals: int = 1000):
        """
        Initialize signal registry.
        
        Args:
            max_signals: Maximum number of signals to keep in registry
        """
        self.logger = get_logger("SignalRegistry")
        self.max_signals = max_signals
        
        # Signal queue (FIFO)
        self._signal_queue: deque = deque(maxlen=max_signals)
        
        # Signal tracking
        self.signals_registered = 0
        self.signals_processed = 0
    
    def register_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Register a new signal.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if registered successfully
        """
        if not signal or not isinstance(signal, dict):
            self.logger.warning("Invalid signal: not a dictionary")
            return False
        
        # Validate required fields
        required_fields = ['symbol', 'side', 'entry_price']
        for field in required_fields:
            if field not in signal:
                self.logger.warning(f"Invalid signal: missing field '{field}'")
                return False
        
        # Add to queue
        self._signal_queue.append(signal)
        self.signals_registered += 1
        
        return True
    
    def fetch_next_signal(self) -> Optional[Dict[str, Any]]:
        """
        Fetch next signal from queue (FIFO).
        
        Returns:
            Signal dictionary or None if queue is empty
        """
        if not self._signal_queue:
            return None
        
        signal = self._signal_queue.popleft()
        self.signals_processed += 1
        
        return signal
    
    def peek_next_signal(self) -> Optional[Dict[str, Any]]:
        """
        Peek at next signal without removing it.
        
        Returns:
            Signal dictionary or None if queue is empty
        """
        if not self._signal_queue:
            return None
        
        return self._signal_queue[0]
    
    def queue_size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of signals in queue
        """
        return len(self._signal_queue)
    
    def clear(self):
        """Clear all signals from queue."""
        self._signal_queue.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get signal registry statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'queue_size': self.queue_size(),
            'signals_registered': self.signals_registered,
            'signals_processed': self.signals_processed,
            'max_signals': self.max_signals
        }

