"""
Bot State - Centralized bot state management.
Provides a clean interface for querying high-level bot state.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from ..logger import get_logger
from .positions import PositionRegistry
from .risk import RiskManager
from .recovery import RecoveryModule

logger = get_logger("BotState")


@dataclass
class BotState:
    """
    Centralized bot state.
    Holds all high-level state information.
    """
    
    # Mode and configuration
    mode: str = "DRY_RUN"  # DRY_RUN or TESTNET
    dry_run: bool = True
    
    # Account information
    balance: float = 0.0
    equity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Position information
    open_positions_count: int = 0
    max_positions: int = 12
    
    # Runtime statistics
    loop_count: int = 0
    last_loop_duration: float = 0.0
    average_loop_duration: float = 0.0
    uptime_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)
    
    # Component references
    position_registry: Optional[PositionRegistry] = None
    risk_manager: Optional[RiskManager] = None
    recovery_module: Optional[RecoveryModule] = None
    
    def update_equity(self, balance: float, unrealized_pnl: float, realized_pnl: float):
        """
        Update equity information.
        
        Args:
            balance: Account balance
            unrealized_pnl: Unrealized PnL
            realized_pnl: Realized PnL
        """
        self.balance = balance
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.equity = balance + unrealized_pnl
    
    def update_positions(self, count: int):
        """
        Update position count.
        
        Args:
            count: Number of open positions
        """
        self.open_positions_count = count
    
    def update_loop_stats(self, duration: float):
        """
        Update loop statistics.
        
        Args:
            duration: Last loop duration in seconds
        """
        self.loop_count += 1
        self.last_loop_duration = duration
        
        # Simple moving average
        if self.loop_count == 1:
            self.average_loop_duration = duration
        else:
            alpha = 0.1  # Smoothing factor
            self.average_loop_duration = (alpha * duration) + ((1 - alpha) * self.average_loop_duration)
        
        self.uptime_seconds = time.time() - self.started_at
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert bot state to dictionary.
        
        Returns:
            State dictionary
        """
        return {
            'mode': self.mode,
            'dry_run': self.dry_run,
            'balance': self.balance,
            'equity': self.equity,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'open_positions_count': self.open_positions_count,
            'max_positions': self.max_positions,
            'loop_count': self.loop_count,
            'last_loop_duration': self.last_loop_duration,
            'average_loop_duration': self.average_loop_duration,
            'uptime_seconds': self.uptime_seconds
        }
    
    def get_summary(self) -> str:
        """
        Get human-readable summary.
        
        Returns:
            Summary string
        """
        return (
            f"Mode: {self.mode} | "
            f"Equity: ${self.equity:.2f} | "
            f"Positions: {self.open_positions_count}/{self.max_positions} | "
            f"PnL: ${self.realized_pnl:.2f} (real) + ${self.unrealized_pnl:.2f} (unreal) | "
            f"Uptime: {self.uptime_seconds:.0f}s | "
            f"Loop: {self.loop_count} (avg: {self.average_loop_duration*1000:.1f}ms)"
        )

