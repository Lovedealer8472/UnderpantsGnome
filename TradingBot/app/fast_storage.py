"""
Fast local storage using SQLite for persistent caching.
Reduces API calls and provides instant data access.
"""

import sqlite3
import time
import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class MarketData:
    """Market data structure for storage."""
    symbol: str
    bid: float
    ask: float
    last: float
    mark: float
    volume: float
    spread_bps: float
    pct_change_24h: float
    timestamp: float


class FastStorage:
    """
    High-performance SQLite storage for market data.
    Provides sub-millisecond read access and efficient writes.
    """
    
    def __init__(self, db_path: str = "data/market_cache.db"):
        """
        Initialize fast storage.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT PRIMARY KEY,
                bid REAL,
                ask REAL,
                last REAL,
                mark REAL,
                volume REAL,
                spread_bps REAL,
                pct_change_24h REAL,
                timestamp REAL,
                updated_at REAL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp)
        """)
        self.conn.commit()
    
    def save_ticker(self, symbol: str, ticker_data: dict):
        """
        Save ticker data to storage.
        
        Args:
            symbol: Trading symbol
            ticker_data: Ticker data dict
        """
        bid = ticker_data.get("bid") or 0.0
        ask = ticker_data.get("ask") or 0.0
        last = ticker_data.get("last") or 0.0
        mark = (
            ticker_data.get("info", {}).get("fairPx")
            or ticker_data.get("info", {}).get("markPrice")
            or last
        )
        volume = (
            ticker_data.get("quoteVolume")
            or ticker_data.get("info", {}).get("amount24h")
            or 0.0
        )
        
        spread_bps = 9999.0
        if bid and ask and ask > 0:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                spread_bps = abs(ask - bid) / mid * 1e4
        
        pct_change_24h = ticker_data.get("percentage") or ticker_data.get("info", {}).get("priceChangePercent") or 0.0
        timestamp = time.time()
        
        self.conn.execute("""
            INSERT OR REPLACE INTO market_data
            (symbol, bid, ask, last, mark, volume, spread_bps, pct_change_24h, timestamp, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, bid, ask, last, mark, volume, spread_bps, pct_change_24h, timestamp, timestamp))
        self.conn.commit()
    
    def batch_save(self, tickers: Dict[str, dict]):
        """
        Batch save tickers (much faster than individual saves).
        
        Args:
            tickers: Dict of {symbol: ticker_data}
        """
        data = []
        now = time.time()
        
        for symbol, ticker_data in tickers.items():
            bid = ticker_data.get("bid") or 0.0
            ask = ticker_data.get("ask") or 0.0
            last = ticker_data.get("last") or 0.0
            mark = (
                ticker_data.get("info", {}).get("fairPx")
                or ticker_data.get("info", {}).get("markPrice")
                or last
            )
            volume = (
                ticker_data.get("quoteVolume")
                or ticker_data.get("info", {}).get("amount24h")
                or 0.0
            )
            
            spread_bps = 9999.0
            if bid and ask and ask > 0:
                mid = 0.5 * (bid + ask)
                if mid > 0:
                    spread_bps = abs(ask - bid) / mid * 1e4
            
            pct_change_24h = ticker_data.get("percentage") or ticker_data.get("info", {}).get("priceChangePercent") or 0.0
            
            data.append((symbol, bid, ask, last, mark, volume, spread_bps, pct_change_24h, now, now))
        
        self.conn.executemany("""
            INSERT OR REPLACE INTO market_data
            (symbol, bid, ask, last, mark, volume, spread_bps, pct_change_24h, timestamp, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()
    
    def get_ticker(self, symbol: str, max_age: float = 5.0) -> Optional[MarketData]:
        """
        Get ticker from storage if fresh.
        
        Args:
            symbol: Trading symbol
            max_age: Maximum age in seconds
        
        Returns:
            MarketData if fresh, None if stale/missing
        """
        cursor = self.conn.execute("""
            SELECT symbol, bid, ask, last, mark, volume, spread_bps, pct_change_24h, timestamp
            FROM market_data
            WHERE symbol = ? AND (?) - timestamp < ?
        """, (symbol, time.time(), max_age))
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        return MarketData(
            symbol=row[0],
            bid=row[1],
            ask=row[2],
            last=row[3],
            mark=row[4],
            volume=row[5],
            spread_bps=row[6],
            pct_change_24h=row[7],
            timestamp=row[8]
        )
    
    def get_multi(self, symbols: List[str], max_age: float = 5.0) -> Dict[str, MarketData]:
        """
        Get multiple tickers at once.
        
        Args:
            symbols: List of symbols
            max_age: Maximum age in seconds
        
        Returns:
            Dict of {symbol: MarketData} for fresh data only
        """
        # Validate symbols before use (defensive validation)
        for symbol in symbols:
            if not isinstance(symbol, str):
                raise ValueError(f"Invalid symbol type: {type(symbol).__name__}")
            # Basic format check
            if not symbol.replace('/', '').replace('_', '').replace('-', '').isalnum():
                raise ValueError(f"Invalid symbol format: {symbol}")
        
        placeholders = ",".join("?" * len(symbols))
        now = time.time()
        
        cursor = self.conn.execute(f"""
            SELECT symbol, bid, ask, last, mark, volume, spread_bps, pct_change_24h, timestamp
            FROM market_data
            WHERE symbol IN ({placeholders}) AND (?) - timestamp < ?
        """, (*symbols, now, max_age))
        
        result = {}
        for row in cursor.fetchall():
            result[row[0]] = MarketData(
                symbol=row[0],
                bid=row[1],
                ask=row[2],
                last=row[3],
                mark=row[4],
                volume=row[5],
                spread_bps=row[6],
                pct_change_24h=row[7],
                timestamp=row[8]
            )
        
        return result
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# Global storage instance
_global_storage: Optional[FastStorage] = None


def get_fast_storage(db_path: str = "data/market_cache.db") -> FastStorage:
    """Get or create global storage instance."""
    global _global_storage
    if _global_storage is None:
        _global_storage = FastStorage(db_path)
    return _global_storage

