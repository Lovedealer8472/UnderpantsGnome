"""
Unified Decision Event System

Single canonical source for all trade decisions (entries, exits, partial exits).
All decisions flow through log_trade_decision() which ensures:
- Recent Activity panel is updated
- decisions.jsonl is written
- Consistency between UI and logs
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import deque

from .decision_logger import get_decision_logger
from .logger import get_logger


@dataclass
class DecisionEvent:
    """
    Canonical decision event representing any trade action.
    
    This is the single source of truth for all trade decisions.
    """
    # Core identification
    timestamp: float  # Unix timestamp
    action: str  # "ENTRY", "EXIT", "PARTIAL_EXIT", "SCALE_OUT", "REJECT"
    symbol: str
    side: Optional[str] = None  # "LONG", "SHORT"
    
    # Pricing
    price: Optional[float] = None  # Execution price
    entry_price: Optional[float] = None  # Original entry price (for exits)
    exit_price: Optional[float] = None  # Exit price (for exits)
    
    # Position sizing
    size: Optional[float] = None  # Trade size
    size_before: Optional[float] = None  # Size before this action (for partials)
    size_after: Optional[float] = None  # Size after this action (for partials)
    
    # PnL (if available)
    pnl_value: Optional[float] = None  # Absolute PnL
    pnl_pct: Optional[float] = None  # PnL percentage
    gross_pnl: Optional[float] = None  # Gross PnL before costs
    net_pnl: Optional[float] = None  # Net PnL after costs
    total_costs: Optional[float] = None  # Total costs (fees, slippage, funding)
    
    # Decision context
    reason: Optional[str] = None  # e.g. "prs_weak_score_44.0", "hit_tp", "hit_sl", "max_positions"
    score: Optional[float] = None  # Signal score (for entries)
    prs: Optional[float] = None  # Position Recovery Score (for exits)
    
    # Metadata
    duration_sec: Optional[float] = None  # Trade duration (for exits)
    was_win: Optional[bool] = None  # Win/loss flag (for exits)
    is_unicorn: bool = False  # Unicorn signal flag
    
    # Signal details (for entries/rejections)
    signal_type: Optional[str] = None
    signal_strength: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # SCORING V2: Score components for detailed logging
    score_components_capped: Optional[Dict] = None
    # SCALPER UPGRADE: Three-stage filter details
    filter_details: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_trade_record(self) -> Dict[str, Any]:
        """Convert to legacy trade_record format for backward compatibility."""
        return {
            'type': self.action.lower(),
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'side': self.side,
            'price': self.price,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'pnl': self.net_pnl or self.pnl_value,
            'gross_pnl': self.gross_pnl,
            'total_costs': self.total_costs,
            'reason': self.reason,
            'duration_sec': self.duration_sec,
            'was_win': self.was_win,
            'is_partial': self.action in ("PARTIAL_EXIT", "SCALE_OUT"),
            'exit_size_pct': (self.size / self.size_before) if (self.size_before and self.size_before > 0 and self.size) else None,
            'is_unicorn': self.is_unicorn,
            'signal_type': self.signal_type,
            'final_score': self.score,
        }


# Global buffer for Recent Activity panel (canonical source)
_decision_events_buffer: deque = deque(maxlen=20)  # Last 20 events


def log_trade_decision(decision: DecisionEvent, bot_instance=None) -> None:
    """
    Unified logging function for all trade decisions.
    
    This is the SINGLE POINT OF ENTRY for all trade decisions.
    It ensures:
    1. Event is added to Recent Activity buffer
    2. Event is logged to decisions.jsonl
    3. Event is added to bot's recent_trades (for backward compatibility)
    4. Optional console/log message
    
    Args:
        decision: DecisionEvent object
        bot_instance: Optional bot instance (for adding to recent_trades)
    """
    logger = get_logger("DecisionEvent")
    
    # 1. Add to canonical Recent Activity buffer
    _decision_events_buffer.append(decision)
    
    # 2. Log to decisions.jsonl (asynchronous, non-blocking)
    try:
        decision_logger = get_decision_logger()
        log_entry = decision.to_dict()
        decision_logger.log(log_entry)
    except Exception as e:
        logger.debug(f"Failed to log decision to JSONL: {e}")
    
    # 3. Add to bot's recent_trades (for backward compatibility with UI)
    if bot_instance and hasattr(bot_instance, 'recent_trades'):
        try:
            trade_record = decision.to_trade_record()
            bot_instance.recent_trades.append(trade_record)
        except Exception as e:
            logger.debug(f"Failed to add to recent_trades: {e}")
    
    # 4. Standardized single-line entry/exit logs (LOGGING V2: Compact format)
    from datetime import datetime
    time_str = datetime.utcfromtimestamp(decision.timestamp).strftime("%H:%M:%S")
    symbol_short = (decision.symbol or "?").replace("/USDT", "")
    
    if decision.action == "ENTRY":
        # ENTRY format: "[ENTRY] symbol side qty=%.4f px=%.6f Scr=%.1f base=%.1f"
        side_str = (decision.side or "?").upper()
        qty = decision.size if decision.size else 0.0
        px = decision.entry_price if decision.entry_price else 0.0
        score = decision.score if decision.score is not None else 0.0
        
        # Extract base score from components if available
        base_score = score  # Default to full score if components not available
        if hasattr(decision, 'score_components_capped') and decision.score_components_capped:
            try:
                comp = decision.score_components_capped
                base_score = comp.get('base', score)
            except Exception:
                pass

        logger.info(
            f"[ENTRY] {decision.symbol} {side_str} qty={qty:.4f} px={px:.6f} Scr={score:.1f} base={base_score:.1f}"
        )
    
    elif decision.action in ("EXIT", "PARTIAL_EXIT", "SCALE_OUT"):
        # EXIT format: "[EXIT] symbol side R=%.2f pnl=%.4f hold=%ds reason=%s"
        side_str = (decision.side or "?").upper()
        
        # Calculate R (risk multiple) from entry/exit prices and stop loss
        r_value = 0.0
        if decision.entry_price and decision.exit_price and decision.side:
            try:
                stop_loss = getattr(decision, 'stop_loss', None)
                if stop_loss and decision.entry_price > 0:
                    if decision.side.upper() == "LONG":
                        stop_distance = abs(decision.entry_price - stop_loss)
                        profit = decision.exit_price - decision.entry_price
                    else:  # SHORT
                        stop_distance = abs(stop_loss - decision.entry_price)
                        profit = decision.entry_price - decision.exit_price
                    if stop_distance > 0:
                        r_value = profit / stop_distance
            except (ZeroDivisionError, TypeError, AttributeError):
                pass
        
        # PnL in quote currency (use net_pnl if available, else pnl_value)
        pnl_value = decision.net_pnl if decision.net_pnl is not None else (decision.pnl_value if decision.pnl_value is not None else 0.0)
        
        # Hold time in seconds
        hold_sec = int(decision.duration_sec) if decision.duration_sec is not None else 0
        
        # Exit reason (concise)
        reason_str = decision.reason if decision.reason else "unknown"
        
        logger.info(
            f"[EXIT] {decision.symbol} {side_str} R={r_value:.2f} pnl={pnl_value:.4f} hold={hold_sec}s reason={reason_str}"
        )


def get_recent_decision_events(limit: int = 20) -> List[DecisionEvent]:
    """
    Get recent decision events for Recent Activity panel.
    
    This is the canonical source for the Recent Activity panel.
    
    Args:
        limit: Maximum number of events to return
        
    Returns:
        List of DecisionEvent objects (newest first)
    """
    events = list(_decision_events_buffer)
    # Return newest first (deque is FIFO, so reverse)
    return list(reversed(events[-limit:]))


def get_decision_events_count() -> int:
    """Get the current count of decision events in buffer."""
    return len(_decision_events_buffer)


def clear_decision_events() -> None:
    """Clear the decision events buffer (for testing/debugging)."""
    _decision_events_buffer.clear()

