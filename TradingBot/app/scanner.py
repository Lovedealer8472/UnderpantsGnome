"""
Symbol Scanner Module

RECOMMENDATION #7: Extracted scanner logic from bot.py for better modularity.

This module handles:
- Processing symbols for signal generation
- Fetching market data (tickers, orderbooks)
- Generating trading signals
- Scoring and filtering signals
"""

import asyncio
import time
from typing import Optional, Tuple, Dict, Any

from .logger import get_logger
from .config import (
    MIN_SIGNAL_SCORE, MIN_SIGNAL_STRENGTH, REPLAY_MODE,
    REPLAY_MIN_SIGNAL_SCORE, REPLAY_MIN_SIGNAL_STRENGTH,
    SIGNAL_PERCENTILE_THRESHOLD, DRY_RUN
)


class SymbolScanner:
    """Handles scanning symbols and generating signals."""
    
    def __init__(self, bot):
        """
        Initialize scanner with reference to bot.
        
        Args:
            bot: ScalperBot instance (for access to exchange, caches, etc.)
        """
        self.bot = bot
        self.logger = get_logger("SymbolScanner")
    
    async def process_symbol(
        self,
        symbol: str,
        batch_now: float,
        scan_regime_config: Dict,
        scan_recent_trades: list,
        scan_volatility_regime: str,
        scan_btc_trend: Optional[str]
    ) -> Tuple[str, ...]:
        """
        Process a single symbol: fetch data, generate signal, score it.
        
        PERFORMANCE OPTIMIZED:
        - Uses pre-bound invariants (regime_config, etc.)
        - Reuses batch timestamp
        - Cache-first for tickers and orderbooks
        
        Args:
            symbol: Symbol to process
            batch_now: Cached timestamp for this batch
            scan_regime_config: Pre-bound regime config
            scan_recent_trades: Pre-bound recent trades list
            scan_volatility_regime: Pre-bound volatility regime
            scan_btc_trend: Pre-bound BTC trend
        
        Returns:
            Tuple of (status, symbol, stats, signal, latency_ms, was_cache_hit, orderbook, orderbook_fetched)
        """
        try:
            symbol_now = batch_now  # Reuse batch timestamp
            
            # EARLY SKIP: Check invalid symbols cache BEFORE signal generation
            # This prevents unnecessary signal processing for symbols in "Reduce Only" mode
            if hasattr(self.bot, 'invalid_symbols') and symbol in self.bot.invalid_symbols:
                invalid_since = self.bot.invalid_symbols[symbol]
                time_since_invalid = symbol_now - invalid_since
                if time_since_invalid < getattr(self.bot, 'invalid_symbol_cooldown', 3600.0):
                    # Still in cooldown - skip silently (DEBUG log only)
                    return ('skipped', 'invalid_symbol', None, None, False, False)
                else:
                    # Cooldown expired - remove from cache and allow processing
                    del self.bot.invalid_symbols[symbol]
            
            # 1) Get denormalized symbol for exchange API (only needed in LIVE mode)
            denormalized_symbol = None
            if self.bot.exchange_wrapper:
                denormalized_symbol = self.bot.exchange_wrapper.denormalize_symbol(symbol)
            
            # 2) Fetch ticker (cache-first, or from replay feed)
            was_cache_hit = False
            if self.bot.replay_mode and self.bot.replay_feed:
                # REPLAY MODE: Get ticker from replay feed
                ticker_data = self.bot.replay_feed.get_ticker_data(symbol)
                if ticker_data:
                    # CRITICAL FIX: Validate replay ticker price data
                    bid_raw = ticker_data.get('bid', 0.0)
                    ask_raw = ticker_data.get('ask', 0.0)
                    last_raw = ticker_data.get('last', 0.0)
                    last = float(last_raw) if last_raw else 0.0
                    
                    # Skip if last price is invalid
                    if last <= 0:
                        return ('skipped', 'invalid_price_data', None, None, was_cache_hit, False)
                    
                    # Apply fallback logic for bid/ask
                    bid = float(bid_raw) if bid_raw and float(bid_raw) > 0 else last
                    ask = float(ask_raw) if ask_raw and float(ask_raw) > 0 else last
                    
                    # Ensure valid spread
                    if bid > 0 and ask <= bid:
                        ask = bid * 1.001  # Create 10 bps spread
                    
                    # Convert ticker data to SymbolStats object (for compatibility)
                    from .universe import SymbolStats
                    stats = SymbolStats(symbol)
                    stats.bid = bid
                    stats.ask = ask
                    stats.last = last
                    stats.mark = ticker_data.get('mark', last)
                    stats.vol_quote = ticker_data.get('quoteVolume', 0.0)
                    # Calculate spread_bps with validated prices
                    if bid > 0 and ask > 0:
                        mid = 0.5 * (bid + ask)
                        if mid > 0:
                            stats.spread_bps = abs(ask - bid) / mid * 1e4
                        else:
                            stats.spread_bps = 9999.0
                    else:
                        stats.spread_bps = 9999.0
                    
                    # CRITICAL FIX: Get pct_change_24h from universe (pre-calculated in replay_runner)
                    # The replay feed doesn't have this data, but replay_runner calculates it from OHLCV history
                    if self.bot.universe and symbol in self.bot.universe.stats:
                        stats.pct_change_24h = self.bot.universe.stats[symbol].pct_change_24h
                    else:
                        stats.pct_change_24h = 0.0  # Fallback if universe not updated yet
                    
                    was_cache_hit = True
                else:
                    return ('skipped', 'no_replay_ticker', None, None, was_cache_hit, False)
            else:
                # LIVE MODE: Fetch from exchange/cache
                try:
                    # Check cache first
                    cached_ticker = self.bot.ticker_cache.get(symbol, max_age=2.0)
                    if cached_ticker:
                        # CRITICAL FIX: Validate cached ticker price data before using
                        bid = cached_ticker.bid or 0.0
                        ask = cached_ticker.ask or 0.0
                        last = cached_ticker.last or 0.0
                        
                        # Skip if last price is invalid
                        if last <= 0:
                            return ('skipped', 'invalid_price_data', None, None, was_cache_hit, False)
                        
                        # Apply fallback logic for bid/ask (use last if invalid)
                        if bid <= 0:
                            bid = last
                        if ask <= 0:
                            ask = last
                        
                        # Ensure valid spread
                        if bid > 0 and ask <= bid:
                            ask = bid * 1.001  # Create 10 bps spread
                        
                        # Convert CachedTicker to SymbolStats
                        from .universe import SymbolStats
                        stats = SymbolStats(symbol)
                        stats.bid = bid
                        stats.ask = ask
                        stats.last = last
                        stats.mark = cached_ticker.mark or last
                        stats.vol_quote = cached_ticker.volume or 0.0
                        # Recalculate spread with validated prices
                        if bid > 0 and ask > 0:
                            mid = 0.5 * (bid + ask)
                            if mid > 0:
                                stats.spread_bps = abs(ask - bid) / mid * 1e4
                            else:
                                stats.spread_bps = 9999.0
                        else:
                            stats.spread_bps = cached_ticker.spread_bps or 9999.0
                        stats.pct_change_24h = cached_ticker.pct_change_24h or 0.0
                        was_cache_hit = True
                    else:
                        # Fetch from exchange
                        ticker = await asyncio.wait_for(
                            self.bot.exchange_wrapper.fetch_ticker(denormalized_symbol),
                            timeout=1.5
                        )
                        # Cache it
                        self.bot.ticker_cache.set(symbol, ticker)
                        # Convert to SymbolStats
                        from .universe import SymbolStats
                        stats = SymbolStats(symbol)
                        
                        # CRITICAL FIX: Apply same price validation logic as refresh_universe
                        # Extract raw values first, then apply fallback logic
                        bid_raw = ticker.get('bid')
                        ask_raw = ticker.get('ask')
                        last_raw = ticker.get('last') or 0.0
                        last = float(last_raw)
                        
                        # CRITICAL: Skip tickers with zero price (invalid/inactive)
                        if last <= 0:
                            return ('skipped', 'invalid_price_data', None, None, was_cache_hit, False)
                        
                        # If bid/ask are None or <= 0, use last price as fallback (common for futures)
                        if bid_raw is None or (isinstance(bid_raw, (int, float)) and float(bid_raw) <= 0):
                            bid = last
                        else:
                            bid = float(bid_raw)
                        
                        if ask_raw is None or (isinstance(ask_raw, (int, float)) and float(ask_raw) <= 0):
                            ask = last
                        else:
                            ask = float(ask_raw)
                        
                        # CRITICAL: Ensure valid spread (bid < ask)
                        if bid > 0 and ask <= bid:
                            ask = bid * 1.001  # Create 10 bps spread
                        
                        stats.bid = bid
                        stats.ask = ask
                        stats.last = last
                        stats.mark = ticker.get('mark', last)
                        stats.vol_quote = ticker.get('quoteVolume', 0.0)
                        stats.pct_change_24h = ticker.get('percentage', 0.0)
                        # Calculate spread
                        if stats.bid and stats.ask and stats.ask > 0:
                            mid = 0.5 * (stats.bid + stats.ask)
                            if mid > 0:
                                stats.spread_bps = abs(stats.ask - stats.bid) / mid * 1e4
                        else:
                            stats.spread_bps = 9999.0
                        was_cache_hit = False
                except asyncio.TimeoutError:
                    return ('skipped', 'ticker_timeout', None, None, was_cache_hit, False)
                except Exception as e:
                    self.logger.debug(f"Ticker fetch failed for {symbol}: {e}")
                    return ('skipped', 'ticker_error', None, None, was_cache_hit, False)
            
            if stats is None:
                return ('skipped', 'no_stats', None, None, was_cache_hit, False)
            
            # PERFORMANCE: Pre-extract stats attributes to avoid repeated getattr() calls
            stats_pct_change = getattr(stats, 'pct_change_24h', None)
            stats_spread_bps = getattr(stats, 'spread_bps', None)
            stats_vol_quote = getattr(stats, 'vol_quote', None)
            stats_last = getattr(stats, 'last', None)
            stats_bid = getattr(stats, 'bid', None)
            stats_ask = getattr(stats, 'ask', None)
            
            # 3) Calculate latency
            latency_ms = (time.time() - symbol_now) * 1000.0
            
            # 4) Fetch orderbook (cache-first or fresh, or from replay feed)
            orderbook = None
            orderbook_fetched = False
            
            # REPLAY MODE: Get orderbook from replay feed
            if self.bot.replay_mode and self.bot.replay_feed:
                snapshot = self.bot.replay_feed.get_market_snapshot(symbol)
                if snapshot and 'orderbook' in snapshot:
                    orderbook = snapshot['orderbook']
                    orderbook_fetched = True
                    # Also update cache
                    self.bot.orderbook_cache[symbol] = {
                        'data': orderbook,
                        'timestamp': symbol_now
                    }
            else:
                # LIVE MODE: Try cache first
                cached_ob = self.bot.orderbook_cache.get(symbol)
                if cached_ob and (symbol_now - cached_ob.get('timestamp', 0)) < 5.0:
                    orderbook = cached_ob['data']
                    orderbook_fetched = True
                else:
                    # Fetch fresh orderbook if not in cache or stale
                    if not DRY_RUN and self.bot.budget.remaining_capacity() > 5:
                        try:
                            # Strict token bucket: wait for token
                            await self.bot.budget.wait_for_token(1, "orderbook")
                            orderbook = await asyncio.wait_for(
                                self.bot.exchange_wrapper.fetch_order_book(denormalized_symbol, limit=20),
                                timeout=0.3
                            )
                            orderbook_fetched = True
                            
                            self.bot.orderbook_cache[symbol] = {
                                'data': orderbook,
                                'timestamp': symbol_now
                            }
                        except Exception:
                            orderbook = None
                            orderbook_fetched = False
            
            # 5) Convert SymbolStats to dict for signal generator (which expects Dict[str, Any])
            # Signal generator uses .get() method which only works on dicts
            # SymbolStats uses __slots__, so we need to manually extract fields
            if isinstance(stats, dict):
                stats_dict = stats
            else:
                # Convert SymbolStats (dataclass with slots) to dict
                from dataclasses import fields
                stats_dict = {}
                for field in fields(stats):
                    stats_dict[field.name] = getattr(stats, field.name, field.default if hasattr(field, 'default') else None)
                # Ensure all required fields are present with defaults
                stats_dict.setdefault('bid', 0.0)
                stats_dict.setdefault('ask', 0.0)
                stats_dict.setdefault('last', 0.0)
                stats_dict.setdefault('mark', stats_dict.get('last', 0.0))
                stats_dict.setdefault('spread_bps', 9999.0)
                stats_dict.setdefault('vol_quote', 0.0)
                stats_dict.setdefault('pct_change_24h', 0.0)
            
            # 6) Get OHLCV data and calculate indicators (needed for signal generation)
            price_data = None
            indicators = None
            if self.bot.replay_mode and self.bot.replay_feed:
                # REPLAY MODE: Get OHLCV from replay feed and calculate indicators
                try:
                    # Get 1m candles for indicators (RSI, EMA, etc.)
                    ohlcv_1m = self.bot.replay_feed.get_ohlcv(symbol, timeframe='1m', limit=100)
                    
                    if ohlcv_1m and len(ohlcv_1m) >= 20:  # Need at least 20 candles for RSI
                        # Store OHLCV data for signal generator
                        # Use 1m candles (MARKSMAN removed)
                        if True:  # Always use 1m candles
                            # MARKSMAN disabled or no 5m available: use 1m as primary
                            price_data = {'candles': ohlcv_1m}
                            if ohlcv_5m:
                                price_data['candles_5m'] = ohlcv_5m
                        
                        from .indicators import calculate_rsi, calculate_ema, calculate_atr
                        # Extract close prices
                        closes = [candle[4] for candle in ohlcv_1m]  # close is index 4
                        highs = [candle[2] for candle in ohlcv_1m]  # high is index 2
                        lows = [candle[3] for candle in ohlcv_1m]  # low is index 3
                        
                        # Calculate indicators
                        rsi = calculate_rsi(closes, period=14)
                        ema20 = calculate_ema(closes, period=20)
                        ema50 = calculate_ema(closes, period=50)
                        ema100 = calculate_ema(closes, period=100) if len(closes) >= 100 else None
                        atr = calculate_atr(highs, lows, closes, period=14)
                        
                        # Calculate ATR as percentage of price
                        atr_pct = None
                        if atr and closes and closes[-1] > 0:
                            atr_pct = (atr / closes[-1]) * 100.0
                        
                        # Calculate pct_change_24h from OHLCV (last 24 hours = 1440 1m candles, use last 100)
                        pct_change_24h = 0.0
                        if len(closes) >= 2:
                            # Use first and last close in the window
                            first_close = closes[0]
                            last_close = closes[-1]
                            if first_close > 0:
                                pct_change_24h = ((last_close - first_close) / first_close) * 100.0
                        
                        # Update stats_dict with calculated pct_change_24h
                        stats_dict['pct_change_24h'] = pct_change_24h
                        
                        indicators = {
                            'rsi': rsi,
                            'ema20': ema20,
                            'ema50': ema50,
                            'ema100': ema100,
                            'atr': atr,
                            'atr_pct': atr_pct,
                        }
                except Exception as e:
                    self.logger.debug(f"Failed to get OHLCV/indicators for {symbol} in replay: {e}")
                    price_data = None
                    indicators = None
            else:
                # LIVE MODE: Get indicators from live calculator or cache
                indicators = None
                if hasattr(self.bot, 'live_indicator_calculator') and self.bot.live_indicator_calculator:
                    # Try to get from live calculator (may fetch if needed)
                    try:
                        indicators = await self.bot.live_indicator_calculator.get_indicators(symbol)
                        # Store in cache for other code that reads it
                        if indicators:
                            self.bot.indicators_cache[symbol] = indicators
                    except Exception as e:
                        self.logger.debug(f"Failed to get live indicators for {symbol}: {e}")
                
                # Fallback to cache if live calculator failed
                if not indicators and hasattr(self.bot, 'indicators_cache'):
                    indicators = self.bot.indicators_cache.get(symbol)
            
            # CRITICAL FIX: Final validation check before signal generation
            # Ensure bid, ask, and last are all > 0 (catch any edge cases)
            bid_final = stats_dict.get('bid', 0.0) or 0.0
            ask_final = stats_dict.get('ask', 0.0) or 0.0
            last_final = stats_dict.get('last', 0.0) or 0.0
            
            if not (bid_final > 0 and ask_final > 0 and last_final > 0):
                return ('skipped', 'invalid_price_data', None, None, was_cache_hit, False)
            
            # 7) Generate signal - CRITICAL: Pass price_data with candles for MARKSMAN
            signal, rejection_reason = self.bot.signal_generator.generate_signal(
                symbol=symbol,
                symbol_stats=stats_dict,  # Pass dict, not SymbolStats object
                price_data=price_data,  # CRITICAL: Pass OHLCV candles (needed for MARKSMAN and other strategies)
                indicators=indicators,  # Pass calculated indicators
                orderbook=orderbook,
                regime_config=scan_regime_config,
                recent_trades=scan_recent_trades,
                volatility_regime=scan_volatility_regime,
                btc_trend=scan_btc_trend
            )
            
            # 8) Early filter: no signal
            if not signal:
                return ('skipped', 'no_signal', None, None, was_cache_hit, orderbook_fetched)
            
            # 9) ML SCORE THRESHOLD - REMOVED (redundant check)
            # This check is redundant - bot.py already checks MIN_SIGNAL_SCORE with idle/adaptive logic
            # Removing to avoid confusion and duplicate filtering
            # The bot.py check (line 3174) handles:
            # - Normal mode: MIN_SIGNAL_SCORE (51)
            # - Idle mode: Gradually lowered threshold (51 -> 42)
            # - Adaptive adjustments based on portfolio health
            
            return ('processed', symbol, stats, signal, latency_ms, was_cache_hit, orderbook, orderbook_fetched)
            
        except Exception as e:
            self.logger.debug(f"Error processing {symbol}: {e}")
            return ('error', symbol, None, None, False, False)

