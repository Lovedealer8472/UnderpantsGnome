"""
Statistics Writer for DRY_RUN_MONITOR
======================================
Writes bot statistics to JSON files that the monitor can read.
This ensures the monitor has accurate, real-time data.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional
import threading


class StatsWriter:
    """Thread-safe statistics writer for monitoring."""
    
    SCHEMA_VERSION = 2
    
    def __init__(self, output_dir: Path = Path(".")):
        self.output_dir = Path(output_dir)
        self.positions_file = self.output_dir / "bot_positions.json"
        self.stats_file = self.output_dir / "bot_stats.json"
        self.lock = threading.Lock()
        self.positions_last_write = 0.0
        self.stats_last_write = 0.0
        self.positions_write_interval = 0.5  # up to 2 writes/sec
        self.stats_write_interval = 1.0      # stats slower cadence
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default
    
    def _atomic_write(self, target: Path, data: Dict):
        """Write JSON atomically with a temp file swap."""
        temp_file = target.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(target)
    
    def write_positions(self, bot):
        """Write current positions snapshot."""
        now = time.time()
        if now - self.positions_last_write < self.positions_write_interval:
            # Throttled - skip this write
            return
        
        with self.lock:
            try:
                positions = getattr(bot, 'positions', {}) or {}
                mode = "DRY_RUN" if getattr(bot, 'dry_run', False) else "LIVE"
                cfg = getattr(bot, 'cfg', None)
                exchange = getattr(cfg, 'EXCHANGE', 'unknown')
                metrics = getattr(bot, 'metrics', {})
                
                serializable_positions = {}
                total_unrealized = 0.0
                for symbol, pos in positions.items():
                    if not isinstance(pos, dict):
                        continue
                    entry_price = self._safe_float(pos.get('entry_price'))
                    current_price = self._safe_float(pos.get('current_price', entry_price), entry_price)
                    size = self._safe_float(pos.get('size'))
                    unrealized = self._safe_float(pos.get('unrealized_pnl'))
                    total_unrealized += unrealized
                    
                    serializable_positions[symbol] = {
                        'symbol': symbol,
                        'side': pos.get('side', 'long'),
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized,
                        'r_multiple': self._safe_float(pos.get('r_multiple')),
                        'entry_time': self._safe_float(pos.get('entry_time'), now),
                        'stop_loss': self._safe_float(pos.get('stop_loss')),
                        'take_profit': self._safe_float(pos.get('take_profit')),
                        # NEW: Binance market data (first-hand exchange data)
                        'bid_price': self._safe_float(pos.get('bid_price')),
                        'ask_price': self._safe_float(pos.get('ask_price')),
                        'spread_bps': self._safe_float(pos.get('spread_bps')),
                        'vol_24h_quote': self._safe_float(pos.get('vol_24h_quote')),
                        'pct_change_24h': self._safe_float(pos.get('pct_change_24h')),
                        'leverage': int(pos.get('leverage', 1) or 1),
                    }
                
                payload = {
                    'version': self.SCHEMA_VERSION,
                    'mode': mode,
                    'exchange': exchange,
                    'timestamp': now,
                    'status': 'ok',
                    'position_count': len(serializable_positions),
                    'positions': serializable_positions,
                    'totals': {
                        'unrealized_pnl': total_unrealized,
                        'entries_attempted': metrics.get('entries_attempted', 0),
                        'entries_opened': metrics.get('entries_opened', 0),
                    },
                    'rejections': metrics.get('rejections_by_reason', {}),
                }
                
                self._atomic_write(self.positions_file, payload)
                self.positions_last_write = now
                
                # DEBUG: Log successful write
                import sys
                print(f"[STATS_WRITER_DEBUG] Wrote {len(serializable_positions)} positions to file", file=sys.stderr)
            
            except Exception as e:
                # Never blow up the bot due to monitoring writes
                import sys
                print(f"[STATS_WRITER_ERROR] write_positions failed: {e}", file=sys.stderr)
                pass
    
    def write_stats(self, bot):
        """Write aggregate statistics snapshot."""
        now = time.time()
        if now - self.stats_last_write < self.stats_write_interval:
            return
        
        with self.lock:
            try:
                mode = "DRY_RUN" if getattr(bot, 'dry_run', False) else "LIVE"
                exchange = getattr(getattr(bot, 'cfg', None), 'EXCHANGE', 'unknown')
                positions = getattr(bot, 'positions', {}) or {}
                metrics = getattr(bot, 'metrics', {})
                cfg = getattr(bot, 'cfg', None)
                exchange = getattr(cfg, 'EXCHANGE', 'unknown')
                
                equity_now = bot.equity_now() if hasattr(bot, 'equity_now') else (
                    self._safe_float(getattr(bot, 'start_equity', 0.0))
                    + self._safe_float(getattr(bot, 'realized_pnl_total', 0.0))
                    + sum(self._safe_float(pos.get('unrealized_pnl')) for pos in positions.values())
                )
                
                signal_stats = getattr(bot, 'signal_stats', None)
                last_signal_time = signal_stats.get('last_signal_time') if isinstance(signal_stats, dict) else None
                
                stats_payload = {
                    'version': self.SCHEMA_VERSION,
                    'mode': mode,
                    'exchange': exchange,
                    'timestamp': now,
                    'start_equity': self._safe_float(getattr(bot, 'start_equity', 1000.0)),
                    'current_equity': self._safe_float(equity_now),
                    'realized_pnl': self._safe_float(getattr(bot, 'realized_pnl_total', 0.0)),
                    'unrealized_pnl': self._safe_float(sum(
                        self._safe_float(pos.get('unrealized_pnl')) for pos in positions.values()
                    )),
                    'total_fees': self._safe_float(getattr(bot, 'realized_fees_total', 0.0)),
                    'win_count': int(getattr(bot, 'win_count', 0)),
                    'loss_count': int(getattr(bot, 'loss_count', 0)),
                    'total_trades': int(getattr(bot, 'win_count', 0) + getattr(bot, 'loss_count', 0)),
                    'position_count': len(positions),
                    'max_positions': int(getattr(cfg, 'MAX_CONCURRENT_POS', 3)),
                    'entries': {
                        'attempted': metrics.get('entries_attempted', 0),
                        'opened': metrics.get('entries_opened', 0),
                        'rejections': metrics.get('rejections_by_reason', {}),
                    },
                    'last_signal_time': last_signal_time,
                    'last_update_ts': now,
                    'status': 'ok',
                    'writer': {
                        'name': 'StatsWriter',
                        'version': self.SCHEMA_VERSION
                    }
                }
                
                self._atomic_write(self.stats_file, stats_payload)
                self.stats_last_write = now
            
            except Exception:
                pass
    
    def update(self, bot):
        """Update both positions and stats."""
        try:
            self.write_positions(bot)
            self.write_stats(bot)
        except Exception:
            pass


# Global instance
_stats_writer: Optional[StatsWriter] = None


def get_stats_writer() -> StatsWriter:
    """Get global stats writer instance."""
    global _stats_writer
    if _stats_writer is None:
        _stats_writer = StatsWriter()
    return _stats_writer


def update_stats(bot):
    """Convenience function to update stats from bot."""
    writer = get_stats_writer()
    writer.update(bot)

