"""
Live Indicator Calculator - Fetches candles and calculates indicators for LIVE mode.
Efficient caching to minimize API calls.
"""

import time
import asyncio
from typing import Dict, Optional, List
from .indicators import calculate_rsi, calculate_ema, calculate_atr


class LiveIndicatorCalculator:
    """Calculates indicators from live candle data with caching."""
    
    def __init__(self, exchange_wrapper, logger):
        self.exchange = exchange_wrapper
        self.logger = logger
        self.cache = {}  # {symbol: (indicators, timestamp)}
        self.cache_ttl = 60.0  # Cache for 60 seconds
        self.last_batch_fetch = 0
        self.batch_interval = 30.0  # Fetch batch every 30s
    
    async def get_indicators(self, symbol: str) -> Optional[Dict]:
        """
        Get indicators for a symbol (cached).
        
        Returns dict with: rsi, adx (estimated), atr, atr_pct, ema20, ema50
        """
        now = time.time()
        
        # Check cache
        if symbol in self.cache:
            indicators, cached_time = self.cache[symbol]
            if (now - cached_time) < self.cache_ttl:
                return indicators
        
        # Fetch and calculate
        try:
            indicators = await self._calculate_indicators(symbol)
            if indicators:
                self.cache[symbol] = (indicators, now)
            return indicators
        except Exception as e:
            self.logger.debug(f"[INDICATORS] Failed to calculate for {symbol}: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str) -> Optional[Dict]:
        """Fetch candles and calculate indicators."""
        try:
            # Fetch 1m candles (need ~100 for reliable indicators)
            denormalized = self.exchange.denormalize_symbol(symbol)
            candles = await asyncio.wait_for(
                self.exchange.fetch_ohlcv(denormalized, timeframe='1m', limit=100),
                timeout=2.0
            )
            
            if not candles or len(candles) < 50:
                return None
            
            # Extract OHLCV data
            closes = [c[4] for c in candles]
            highs = [c[2] for c in candles]
            lows = [c[3] for c in candles]
            
            # Calculate indicators
            rsi = calculate_rsi(closes, period=14)
            atr = calculate_atr(highs, lows, closes, period=14)
            ema20 = calculate_ema(closes, period=20)
            ema50 = calculate_ema(closes, period=50)
            
            # Calculate ATR as percentage
            atr_pct = None
            if atr and closes[-1] > 0:
                atr_pct = (atr / closes[-1]) * 100.0
            
            # Estimate ADX from price trend (simple approximation)
            # ADX measures trend strength - we'll use EMA slope as proxy
            adx = 20.0  # Default neutral
            if ema20 and ema50 and len(closes) >= 20:
                # Calculate trend strength from EMA divergence
                ema_diff_pct = abs(ema20 - ema50) / ema50 * 100 if ema50 > 0 else 0
                # Map to ADX scale (0-100, typically 20-40 range)
                adx = min(50.0, 15.0 + (ema_diff_pct * 10))  # Scale: weak trend ~20, strong trend ~40+
            
            return {
                'rsi': rsi,
                'adx': adx,
                'atr': atr,
                'atr_pct': atr_pct,
                'ema20': ema20,
                'ema50': ema50,
            }
        except Exception as e:
            self.logger.debug(f"[INDICATORS] Error calculating for {symbol}: {e}")
            return None
    
    async def batch_update(self, symbols: List[str], max_concurrent: int = 5):
        """
        Update indicators for multiple symbols in parallel.
        
        Args:
            symbols: List of symbols to update
            max_concurrent: Max concurrent API calls
        """
        now = time.time()
        
        # Rate limit: Only batch update every 30s
        if (now - self.last_batch_fetch) < self.batch_interval:
            return
        
        self.last_batch_fetch = now
        
        # Filter to symbols that need update
        symbols_to_update = []
        for symbol in symbols:
            if symbol not in self.cache:
                symbols_to_update.append(symbol)
            else:
                _, cached_time = self.cache[symbol]
                if (now - cached_time) >= self.cache_ttl:
                    symbols_to_update.append(symbol)
        
        if not symbols_to_update:
            return
        
        # Limit batch size to avoid rate limits
        symbols_to_update = symbols_to_update[:20]  # Max 20 at once
        
        self.logger.debug(f"[INDICATORS] Batch updating {len(symbols_to_update)} symbols")
        
        # Fetch in batches
        for i in range(0, len(symbols_to_update), max_concurrent):
            batch = symbols_to_update[i:i+max_concurrent]
            tasks = [self.get_indicators(symbol) for symbol in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Small delay between batches
            if i + max_concurrent < len(symbols_to_update):
                await asyncio.sleep(0.5)
    
    def get_cached(self, symbol: str) -> Optional[Dict]:
        """Get cached indicators without fetching (synchronous)."""
        if symbol in self.cache:
            indicators, _ = self.cache[symbol]
            return indicators
        return None
    
    def clear_cache(self):
        """Clear all cached indicators."""
        self.cache.clear()

