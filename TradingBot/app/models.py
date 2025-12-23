"""
Data Models - Structured data classes for positions, trades, and signals.
Provides type safety and better code organization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


@dataclass
class Position:
    """
    Position data model.
    Represents an open trading position with all relevant information.
    """
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float
    entry_time: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: Optional[int] = None
    entry_order_id: Optional[str] = None
    
    # Tracking fields
    peak_price: Optional[float] = None  # Highest price reached (for trailing stops)
    trough_price: Optional[float] = None  # Lowest price reached (for trailing stops)
    max_r: Optional[float] = None  # Maximum R achieved
    current_r: Optional[float] = None  # Current R value
    
    # Exit tracking
    partial_exit_size: Optional[float] = None  # Size already partially exited
    trailing_stop_price: Optional[float] = None  # Current trailing stop price
    
    # Metadata
    signal_score: Optional[float] = None  # Original signal score
    signal_strength: Optional[float] = None  # Original signal strength
    entry_reason: Optional[str] = None  # Reason for entry
    
    # Additional fields (for backward compatibility)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values."""
        if self.peak_price is None:
            self.peak_price = self.entry_price
        if self.trough_price is None:
            self.trough_price = self.entry_price
        if self.entry_time is None:
            self.entry_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'size': self.size,
            'entry_time': self.entry_time,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'leverage': self.leverage,
            'entry_order_id': self.entry_order_id,
            'peak_price': self.peak_price,
            'trough_price': self.trough_price,
            'max_r': self.max_r,
            'current_r': self.current_r,
            'partial_exit_size': self.partial_exit_size,
            'trailing_stop_price': self.trailing_stop_price,
            'signal_score': self.signal_score,
            'signal_strength': self.signal_strength,
            'entry_reason': self.entry_reason,
            **self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary."""
        # Extract known fields
        known_fields = {
            'symbol', 'side', 'entry_price', 'size', 'entry_time',
            'stop_loss', 'take_profit', 'leverage', 'entry_order_id',
            'peak_price', 'trough_price', 'max_r', 'current_r',
            'partial_exit_size', 'trailing_stop_price',
            'signal_score', 'signal_strength', 'entry_reason'
        }
        
        # Separate known and extra fields
        known_data = {k: v for k, v in data.items() if k in known_fields}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        # Create position with extra fields
        position = cls(**known_data, extra=extra_data)
        return position


@dataclass
class Trade:
    """
    Trade data model.
    Represents a completed trade (entry + exit) with all relevant information.
    """
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    entry_time: float
    exit_time: float
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    
    # PnL calculations
    gross_pnl: Optional[float] = None
    entry_fee: Optional[float] = None
    exit_fee: Optional[float] = None
    funding_cost: Optional[float] = None
    slippage_cost: Optional[float] = None
    total_costs: Optional[float] = None
    net_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    # Trade metadata
    exit_reason: Optional[str] = None
    was_win: Optional[bool] = None
    duration_sec: Optional[float] = None
    
    # Signal information
    signal_score: Optional[float] = None
    signal_strength: Optional[float] = None
    
    # Additional fields
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.duration_sec is None and self.entry_time and self.exit_time:
            self.duration_sec = self.exit_time - self.entry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'entry_order_id': self.entry_order_id,
            'exit_order_id': self.exit_order_id,
            'gross_pnl': self.gross_pnl,
            'entry_fee': self.entry_fee,
            'exit_fee': self.exit_fee,
            'funding_cost': self.funding_cost,
            'slippage_cost': self.slippage_cost,
            'total_costs': self.total_costs,
            'net_pnl': self.net_pnl,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason,
            'was_win': self.was_win,
            'duration_sec': self.duration_sec,
            'signal_score': self.signal_score,
            'signal_strength': self.signal_strength,
            **self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create trade from dictionary."""
        known_fields = {
            'symbol', 'side', 'entry_price', 'exit_price', 'size',
            'entry_time', 'exit_time', 'entry_order_id', 'exit_order_id',
            'gross_pnl', 'entry_fee', 'exit_fee', 'funding_cost',
            'slippage_cost', 'total_costs', 'net_pnl', 'pnl_pct',
            'exit_reason', 'was_win', 'duration_sec',
            'signal_score', 'signal_strength'
        }
        
        known_data = {k: v for k, v in data.items() if k in known_fields}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(**known_data, extra=extra_data)


