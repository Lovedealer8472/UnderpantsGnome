"""
Position Registry - Centralized position management.
Provides thread-safe access to positions with validation.
"""

from typing import Dict, Any, Optional, Set
from threading import Lock
from ..logger import get_logger

logger = get_logger("PositionRegistry")


class PositionRegistry:
    """
    Centralized position registry.
    Provides safe access to positions with validation.
    """
    
    def __init__(self):
        """Initialize position registry."""
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._positions_set: Set[str] = set()  # O(1) lookups
        self._lock = Lock()
    
    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get positions dictionary (read-only view).
        
        Returns:
            Positions dictionary
        """
        with self._lock:
            return self._positions.copy()
    
    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position dictionary or None
        """
        with self._lock:
            return self._positions.get(symbol)
    
    def add(self, symbol: str, position: Dict[str, Any]) -> bool:
        """
        Add position to registry.
        
        Args:
            symbol: Trading symbol
            position: Position dictionary
            
        Returns:
            True if added, False if already exists
        """
        with self._lock:
            if symbol in self._positions:
                logger.warning(f"Position already exists: {symbol}")
                return False
            
            # Initialize trailing stop tracking fields
            entry_price = position.get('entry_price', 0)
            if entry_price > 0:
                # Initialize peak_favorable_price to entry (will be updated by trailing engine)
                position.setdefault('peak_favorable_price', entry_price)
                position.setdefault('trailing_state', {
                    'partial_1_taken': False,
                    'partial_2_taken': False,
                    'runner_mode': False,
                    'last_update_r': 0.0,
                    'last_update_ts': 0.0,
                    'last_locked_r': 0.0
                })
                position.setdefault('initial_stop_price', position.get('stop_loss', entry_price))
            
            self._positions[symbol] = position
            self._positions_set.add(symbol)
            return True
    
    def update(self, symbol: str, position: Dict[str, Any]) -> bool:
        """
        Update position in registry.
        
        Args:
            symbol: Trading symbol
            position: Updated position dictionary
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            if symbol not in self._positions:
                logger.warning(f"Position not found for update: {symbol}")
                return False
            
            self._positions[symbol] = position
            return True
    
    def remove(self, symbol: str) -> bool:
        """
        Remove position from registry.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if symbol not in self._positions:
                return False
            
            del self._positions[symbol]
            self._positions_set.discard(symbol)
            return True
    
    def contains(self, symbol: str) -> bool:
        """
        Check if position exists.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position exists
        """
        with self._lock:
            return symbol in self._positions
    
    def count(self) -> int:
        """
        Get position count.
        
        Returns:
            Number of positions
        """
        with self._lock:
            return len(self._positions)
    
    def clear(self):
        """Clear all positions."""
        with self._lock:
            self._positions.clear()
            self._positions_set.clear()

