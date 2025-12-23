"""
Feature Engine - Integration layer for fetching OHLCV and computing advanced features.
Handles data fetching, caching, and feature computation for signal scoring.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

from .advanced_features import compute_all_features, AdvancedFeatures
from .exchanges.base import ExchangeBase


@dataclass
class FeatureCache:
    """Cache for OHLCV data and computed features."""
    ltf_data: Optional[pd.DataFrame] = None
    htf_data: Optional[pd.DataFrame] = None
    features: Optional[AdvancedFeatures] = None
    timestamp: float = 0.0
    ttl: float = 60.0  # 60 second cache for computed features
    htf_last_fetch: float = 0.0  # Last HTF data fetch timestamp


class FeatureEngine:
    """Feature engine for computing advanced trading features."""
    
    def __init__(self, exchange: Optional[ExchangeBase] = None):
        self.exchange = exchange
        self.cache: Dict[str, FeatureCache] = {}
        self.cache_ttl = 60.0  # 60 second cache for computed features
        self.htf_refresh_interval = 3600.0  # 1 hour cache for HTF OHLCV data (CRITICAL: prevents API explosion)
    
    async def fetch_ohlcv_to_dataframe(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data and convert to pandas DataFrame."""
        # REPLAY MODE: Get OHLCV from replay feed
        if hasattr(self.bot, 'replay_mode') and self.bot.replay_mode and hasattr(self.bot, 'replay_feed') and self.bot.replay_feed:
            try:
                ohlcv = self.bot.replay_feed.get_ohlcv(symbol, timeframe, limit)
                if not ohlcv or len(ohlcv) == 0:
                    return None
            except Exception:
                return None
        elif not self.exchange:
            return None
        else:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv or len(ohlcv) == 0:
                    return None
            except Exception:
                return None
        
        # Convert to DataFrame: [timestamp, open, high, low, close, volume]
        try:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception:
            return None
    
    async def get_features(
        self,
        symbol: str,
        current_price: float,
        current_high: float,
        current_low: float,
        ltf_timeframe: str = "5m",
        htf_timeframe: str = "1h",
        ltf_limit: int = 500,
        htf_limit: int = 500,
        ltf_df: Optional[pd.DataFrame] = None  # Optional: use existing LTF data to avoid duplicate fetch
    ) -> Optional[AdvancedFeatures]:
        """
        Get advanced features for a symbol.
        
        CRITICAL: HTF data is cached for 1 hour to minimize API calls.
        LTF features compute from existing LTF data (no extra API calls).
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            current_high: Current high (for current candle)
            current_low: Current low (for current candle)
            ltf_timeframe: Low timeframe (5m or 15m)
            htf_timeframe: High timeframe (1h)
            ltf_limit: Number of LTF bars to fetch (500-1500)
            htf_limit: Number of HTF bars to fetch (200-500)
            ltf_df: Optional existing LTF DataFrame to reuse (avoids duplicate fetch)
        
        Returns:
            AdvancedFeatures object or None if insufficient data
        """
        now = time.time()
        
        # Check computed features cache (60s TTL)
        if symbol in self.cache:
            cached = self.cache[symbol]
            if now - cached.timestamp < self.cache_ttl and cached.features is not None:
                return cached.features
        
        # Get or fetch LTF data
        if ltf_df is not None:
            # Use provided LTF data (no API call)
            ltf_df = ltf_df.copy()
        else:
            # Fetch LTF data (only if not provided)
            ltf_df = await self.fetch_ohlcv_to_dataframe(symbol, ltf_timeframe, ltf_limit)
        
        if ltf_df is None or len(ltf_df) < 50:
            return None
        
        # Append current price as latest candle if not already there
        last_timestamp = ltf_df.index[-1]
        if abs(current_price - ltf_df['close'].iloc[-1]) > 0.0001:  # Price changed
            # Update last candle or append new
            new_row = pd.DataFrame({
                'open': [ltf_df['close'].iloc[-1]],
                'high': [max(ltf_df['high'].iloc[-1], current_high)],
                'low': [min(ltf_df['low'].iloc[-1], current_low)],
                'close': [current_price],
                'volume': [0]  # Unknown volume for current candle
            }, index=[last_timestamp])
            ltf_df = pd.concat([ltf_df.iloc[:-1], new_row])
        
        # Get or fetch HTF data with 1-hour cache (CRITICAL: prevents API explosion)
        htf_df = None
        if symbol in self.cache:
            cached = self.cache[symbol]
            # Check if HTF data exists and is fresh enough (< 1 hour old)
            if cached.htf_data is not None and (now - cached.htf_last_fetch) < self.htf_refresh_interval:
                # Use cached HTF data (NO API CALL)
                htf_df = cached.htf_data
            else:
                # HTF cache expired - fetch new data (1 call per symbol per hour)
                htf_df = await self.fetch_ohlcv_to_dataframe(symbol, htf_timeframe, htf_limit)
                if htf_df is not None:
                    # Update cache with new HTF data
                    cached.htf_data = htf_df
                    cached.htf_last_fetch = now
        else:
            # No cache entry - fetch HTF data (1 call per symbol per hour)
            htf_df = await self.fetch_ohlcv_to_dataframe(symbol, htf_timeframe, htf_limit)
            if htf_df is not None:
                self.cache[symbol] = FeatureCache(htf_data=htf_df, htf_last_fetch=now)
        
        # Compute features (all LTF features compute from existing data - NO API CALLS)
        try:
            features = compute_all_features(ltf_df, htf_df, current_price, current_high, current_low)
            
            # Cache computed features (60s TTL)
            if symbol not in self.cache:
                self.cache[symbol] = FeatureCache()
            cached = self.cache[symbol]
            cached.ltf_data = ltf_df
            cached.features = features
            cached.timestamp = now
            if htf_df is not None:
                cached.htf_data = htf_df
                if cached.htf_last_fetch == 0.0:  # Only set if not already set
                    cached.htf_last_fetch = now
            
            return features
        except Exception as e:
            return None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache for a symbol or all symbols."""
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear()
    
    def get_reversal_scores(
        self,
        symbol: str,
        features: Optional[AdvancedFeatures] = None
    ) -> Tuple[float, float]:
        """
        Get reversal scores for long and short.
        
        Returns:
            (reversal_score_long, reversal_score_short)
        """
        if features is None:
            cached = self.cache.get(symbol)
            if cached and cached.features:
                features = cached.features
            else:
                return 0.0, 0.0
        
        return features.reversal_score_long, features.reversal_score_short


def integrate_reversal_scores(
    base_score: float,
    reversal_score_long: float,
    reversal_score_short: float,
    side: str
) -> float:
    """
    Integrate reversal scores into base signal score.
    
    Args:
        base_score: Base signal score (0-100)
        reversal_score_long: Reversal score for long direction
        reversal_score_short: Reversal score for short direction
        side: 'long' or 'short'
    
    Returns:
        Final signal score with reversal boost
    """
    if side == "long":
        reversal_score = reversal_score_long
        # Only boost if long reversal score is higher
        if reversal_score_long > reversal_score_short:
            final_score = base_score + reversal_score
        else:
            # Penalize if opposite direction has stronger reversal signal
            final_score = base_score * 0.7
    else:  # short
        reversal_score = reversal_score_short
        # Only boost if short reversal score is higher
        if reversal_score_short > reversal_score_long:
            final_score = base_score + reversal_score
        else:
            # Penalize if opposite direction has stronger reversal signal
            final_score = base_score * 0.7
    
    # Cap at 100
    return min(100.0, max(0.0, final_score))

