"""
High-performance in-memory ticker cache with TTL for ultra-low latency.
Uses CPU/RAM to minimize API calls and provide instant data access.
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CachedTicker:
    """Cached ticker data with timestamp."""
    symbol: str
    bid: float
    ask: float
    last: float
    mark: float
    volume: float
    spread_bps: float
    timestamp: float = field(default_factory=time.time)
    pct_change_24h: float = 0.0


class TickerCache:
    """
    Ultra-fast in-memory ticker cache with TTL.
    Reduces API calls by 90%+ and provides sub-millisecond data access.
    """
    
    def __init__(self, default_ttl: float = 1.0):
        """
        Initialize ticker cache.
        
        Args:
            default_ttl: Default time-to-live in seconds (1.0 = 1 second)
        """
        self.cache: Dict[str, CachedTicker] = {}
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0
        self.last_update = 0.0
        self.update_times = []
        
    def get(self, symbol: str, max_age: float = None) -> Optional[CachedTicker]:
        """
        Get ticker from cache if fresh.
        
        Args:
            symbol: Trading symbol
            max_age: Maximum age in seconds (uses default_ttl if None)
        
        Returns:
            CachedTicker if fresh, None if stale/missing
        """
        if max_age is None:
            max_age = self.default_ttl
        
        cached = self.cache.get(symbol)
        if cached is None:
            self.miss_count += 1
            return None
        
        age = time.time() - cached.timestamp
        if age > max_age:
            self.miss_count += 1
            return None
        
        self.hit_count += 1
        return cached
    
    def set(self, symbol: str, ticker_data: dict):
        """
        Cache ticker data.
        
        Args:
            symbol: Trading symbol
            ticker_data: Ticker data dict from exchange
        """
        bid = ticker_data.get("bid") or 0.0
        ask = ticker_data.get("ask") or 0.0
        last = ticker_data.get("last") or 0.0
        
        # Get mark price
        mark = (
            ticker_data.get("info", {}).get("fairPx")
            or ticker_data.get("info", {}).get("markPrice")
            or last
        )
        
        # Get volume
        volume = (
            ticker_data.get("quoteVolume")
            or ticker_data.get("info", {}).get("amount24h")
            or ticker_data.get("info", {}).get("quote_volume")
            or 0.0
        )
        
        # Calculate spread
        spread_bps = 9999.0
        if bid and ask and ask > 0:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                spread_bps = abs(ask - bid) / mid * 1e4
        
        # Get 24h change
        pct_change_24h = ticker_data.get("percentage") or ticker_data.get("info", {}).get("priceChangePercent") or 0.0
        
        self.cache[symbol] = CachedTicker(
            symbol=symbol,
            bid=float(bid),
            ask=float(ask),
            last=float(last),
            mark=float(mark),
            volume=float(volume),
            spread_bps=float(spread_bps),
            pct_change_24h=float(pct_change_24h),
            timestamp=time.time()
        )
    
    def batch_set(self, tickers: Dict[str, dict]):
        """
        Batch update cache (faster than individual sets).
        
        Args:
            tickers: Dict of {symbol: ticker_data}
        """
        update_start = time.time()
        for symbol, ticker_data in tickers.items():
            self.set(symbol, ticker_data)
        self.last_update = time.time()
        self.update_times.append(self.last_update - update_start)
        if len(self.update_times) > 100:
            self.update_times.pop(0)
    
    def get_multi(self, symbols: list, max_age: float = None) -> Dict[str, CachedTicker]:
        """
        Get multiple tickers at once (optimized).
        
        Args:
            symbols: List of symbols
            max_age: Maximum age in seconds
        
        Returns:
            Dict of {symbol: CachedTicker} for fresh data only
        """
        result = {}
        for symbol in symbols:
            cached = self.get(symbol, max_age)
            if cached:
                result[symbol] = cached
        return result
    
    def clear_stale(self, max_age: float = None):
        """
        Remove stale entries to free memory.
        
        Args:
            max_age: Maximum age before considering stale (default: 5x TTL)
        """
        if max_age is None:
            max_age = self.default_ttl * 5
        
        now = time.time()
        stale = [
            symbol for symbol, cached in self.cache.items()
            if (now - cached.timestamp) > max_age
        ]
        for symbol in stale:
            del self.cache[symbol]
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0.0
        avg_update_time = sum(self.update_times) / len(self.update_times) if self.update_times else 0.0
        
        return {
            "size": len(self.cache),
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
            "avg_update_time_ms": avg_update_time * 1000,
            "last_update": self.last_update
        }


# Global cache instance
_global_cache: Optional[TickerCache] = None


def get_ticker_cache(ttl: float = 1.0) -> TickerCache:
    """Get or create global ticker cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = TickerCache(default_ttl=ttl)
    return _global_cache

