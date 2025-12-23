"""
Risk Manager - Centralized risk management and position sizing.
Provides risk calculations and position limit enforcement.
"""

from typing import Dict, Any, Optional
from ..logger import get_logger
from ..config import (
    TOTAL_RISK_BUDGET,
    MAX_OPEN_POSITIONS,
    MIN_RISK_PER_TRADE,
    MAX_RISK_PER_TRADE,
    BASE_POSITION_PCT
)

logger = get_logger("RiskManager")


class RiskManager:
    """
    Centralized risk management.
    Provides risk calculations and position limit enforcement.
    """
    
    def __init__(self):
        """Initialize risk manager."""
        self.logger = get_logger("RiskManager")
        
        # Risk tracking
        self.total_risk_used = 0.0
        self.positions_count = 0
    
    def can_open_new_position(
        self,
        current_equity: float,
        current_positions: int,
        proposed_risk_pct: Optional[float] = None
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Args:
            current_equity: Current account equity
            current_positions: Current number of open positions
            proposed_risk_pct: Proposed risk percentage for this trade (optional)
            
        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        # Check position limit
        if current_positions >= MAX_OPEN_POSITIONS:
            return False, f"Max positions reached: {current_positions}/{MAX_OPEN_POSITIONS}"
        
        # Check risk budget if proposed risk provided
        if proposed_risk_pct is not None:
            if proposed_risk_pct < MIN_RISK_PER_TRADE:
                return False, f"Risk too low: {proposed_risk_pct:.2f}% < {MIN_RISK_PER_TRADE:.2f}%"
            
            if proposed_risk_pct > MAX_RISK_PER_TRADE:
                return False, f"Risk too high: {proposed_risk_pct:.2f}% > {MAX_RISK_PER_TRADE:.2f}%"
            
            # Check total risk budget
            if self.total_risk_used + proposed_risk_pct > TOTAL_RISK_BUDGET:
                return False, f"Risk budget exceeded: {self.total_risk_used + proposed_risk_pct:.2f}% > {TOTAL_RISK_BUDGET:.2f}%"
        
        return True, "OK"
    
    def current_risk_usage(self) -> float:
        """
        Get current risk usage percentage.
        
        Returns:
            Current risk usage as percentage
        """
        return self.total_risk_used
    
    def in_budget(self, additional_risk_pct: float) -> bool:
        """
        Check if additional risk fits within budget.
        
        Args:
            additional_risk_pct: Additional risk percentage to check
            
        Returns:
            True if risk fits within budget
        """
        return (self.total_risk_used + additional_risk_pct) <= TOTAL_RISK_BUDGET
    
    def remaining_risk_capacity(self) -> float:
        """
        Get remaining risk capacity.
        
        Returns:
            Remaining risk capacity as percentage
        """
        return max(0.0, TOTAL_RISK_BUDGET - self.total_risk_used)
    
    def record_position_opened(self, risk_pct: float):
        """
        Record that a position was opened.
        
        Args:
            risk_pct: Risk percentage for this position
        """
        self.total_risk_used += risk_pct
        self.positions_count += 1
    
    def record_position_closed(self, risk_pct: float):
        """
        Record that a position was closed.
        
        Args:
            risk_pct: Risk percentage that was released
        """
        self.total_risk_used = max(0.0, self.total_risk_used - risk_pct)
        self.positions_count = max(0, self.positions_count - 1)
    
    def reset(self):
        """Reset risk tracking."""
        self.total_risk_used = 0.0
        self.positions_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get risk manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_risk_used': self.total_risk_used,
            'remaining_capacity': self.remaining_risk_capacity(),
            'positions_count': self.positions_count,
            'max_positions': MAX_OPEN_POSITIONS,
            'risk_budget': TOTAL_RISK_BUDGET
        }

