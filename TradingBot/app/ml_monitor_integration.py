"""
ML Monitor Integration
======================

Feeds trade data from the bot to the Live ML Monitor for real-time learning.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

MONITOR_DB = Path(__file__).parent.parent.parent / "live_ml_monitor.db"


class MLMonitorFeed:
    """
    Feeds trade outcomes to the Live ML Monitor.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.db_path = MONITOR_DB
        
        if self.enabled:
            self._ensure_db_exists()
            logger.info("[ML_MONITOR_FEED] Enabled - feeding data to Live ML Monitor")
        else:
            logger.info("[ML_MONITOR_FEED] Disabled")
    
    def _ensure_db_exists(self):
        """Ensure monitor database exists."""
        if not self.db_path.exists():
            logger.warning(f"[ML_MONITOR_FEED] Monitor DB not found: {self.db_path}")
            logger.warning("[ML_MONITOR_FEED] Run LIVE_ML_MONITOR.py first to initialize")
    
    def feed_trade_outcome(self, trade_data: Dict):
        """
        Feed a completed trade to the monitor.
        
        Args:
            trade_data: Dictionary with trade information:
                - symbol: str
                - side: str ('long' or 'short')
                - entry_time: float (timestamp)
                - exit_time: float (timestamp)
                - entry_price: float
                - exit_price: float
                - pnl: float
                - pnl_pct: float
                - exit_reason: str
                - signal_score: float (optional)
                - ml_confidence: float (optional)
                - regime: str (optional)
        """
        if not self.enabled:
            return
        
        if not self.db_path.exists():
            return  # Silently skip if monitor not running
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Extract data
            symbol = trade_data.get('symbol', 'UNKNOWN')
            side = trade_data.get('side', 'long')
            entry_time = trade_data.get('entry_time', 0)
            exit_time = trade_data.get('exit_time', time.time())
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('exit_price', 0)
            pnl = trade_data.get('pnl', 0)
            pnl_pct = trade_data.get('pnl_pct', 0)
            exit_reason = trade_data.get('exit_reason', 'unknown')
            signal_score = trade_data.get('signal_score', 0)
            ml_confidence = trade_data.get('ml_confidence', 0)
            regime = trade_data.get('regime', 'unknown')
            won = 1 if pnl > 0 else 0
            duration_min = (exit_time - entry_time) / 60 if exit_time > entry_time else 0
            
            # Insert into monitor database
            cursor.execute("""
                INSERT INTO monitored_trades 
                (symbol, side, entry_time, exit_time, entry_price, exit_price, pnl, pnl_pct,
                 exit_reason, signal_score, ml_confidence, regime, won, duration_min, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, side, entry_time, exit_time, entry_price, exit_price,
                pnl, pnl_pct, exit_reason, signal_score, ml_confidence, regime,
                won, duration_min, time.time()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(
                f"[ML_MONITOR_FEED] Fed trade: {symbol} {side.upper()} | "
                f"{'WIN' if won else 'LOSS'} | PnL: {pnl_pct*100:+.2f}%"
            )
            
        except Exception as e:
            logger.error(f"[ML_MONITOR_FEED] Failed to feed trade: {e}")
    
    def feed_signal_decision(self, signal_data: Dict):
        """
        Feed a signal decision (entry/skip) to the monitor.
        
        This can be used for tracking signal quality even before trade completion.
        """
        # TODO: Implement signal decision tracking
        pass


# Global instance
_ml_monitor_feed: Optional[MLMonitorFeed] = None


def get_ml_monitor_feed(enabled: bool = True) -> MLMonitorFeed:
    """Get or create global ML monitor feed instance."""
    global _ml_monitor_feed
    if _ml_monitor_feed is None:
        _ml_monitor_feed = MLMonitorFeed(enabled=enabled)
    return _ml_monitor_feed

