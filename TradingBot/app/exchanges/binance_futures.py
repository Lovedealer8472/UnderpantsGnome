"""
Binance Futures Exchange Wrapper - Binance USDT-M Futures trading.
Supports both REST API and WebSocket for real-time ticker data.
"""
import sys
import logging
import asyncio
import time
import ccxt.async_support as ccxt_async
from typing import Dict, Any, Optional, List
from .base import ExchangeBase
from .websocket_manager import BinanceWebSocketManager
from ..logger import get_logger
from ..config import (
    BINANCE_API_KEY, BINANCE_SECRET, DRY_RUN, EXCHANGE_TIMEOUT_MS, 
    EXCHANGE_RETRIES, RETRY_DELAY_MULTIPLIER, MARGIN_MODE
)


class BinanceFuturesExchange(ExchangeBase):
    """Wrapper for Binance USDT-M Futures trading."""
    
    def __init__(self, config=None):
        """
        Initialize Binance Futures exchange wrapper.
        
        Args:
            config: Configuration object (optional, uses module config if not provided)
        """
        self.logger = get_logger("BinanceFutures")
        self.config = config
        self.exchange = None
        self.markets = {}
        
        # OPTIMIZATION: Cache symbol normalization/denormalization (hot path)
        self._normalize_cache = {}
        self._denormalize_cache = {}
        self._cache_max_size = 1000  # Limit cache size
        
        # WebSocket manager for real-time ticker data
        self.ws_manager: Optional[BinanceWebSocketManager] = None
        self.use_websocket = True  # Enable WebSocket by default
        self.ws_ticker_cache: Dict[str, Dict[str, Any]] = {}  # Cache tickers from WebSocket
        
        # Get credentials from config or module
        if config:
            self.api_key = getattr(config, 'BINANCE_API_KEY', BINANCE_API_KEY)
            self.api_secret = getattr(config, 'BINANCE_API_SECRET', BINANCE_SECRET)
            self.dry_run = getattr(config, 'DRY_RUN', DRY_RUN)
            self.testnet = getattr(config, 'BINANCE_TESTNET', False)
            self.margin_mode = getattr(config, 'MARGIN_MODE', MARGIN_MODE)
        else:
            self.api_key = BINANCE_API_KEY
            self.api_secret = BINANCE_SECRET
            self.dry_run = DRY_RUN
            self.testnet = False
            self.margin_mode = MARGIN_MODE
    
    async def initialize(self):
        """Initialize Binance Futures exchange connection."""
        # Check API keys
        if not self.api_key or not self.api_secret:
            error_msg = "Binance API credentials not configured (BINANCE_API_KEY or BINANCE_SECRET missing)"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Configure for futures trading
        options = {
            "enableRateLimit": True,
            "timeout": 5000,  # 5s timeout to fail fast and prevent queue buildup
            "options": {
                "defaultType": "future",  # Binance futures
                "fetchCurrencies": False,
                "adjustForTimeDifference": True,
            }
        }
        
        # API keys are required for authenticated endpoints
        options["apiKey"] = self.api_key
        options["secret"] = self.api_secret
        
        # CRITICAL: Use binanceusdm for USDT-M futures support (includes fetch_tickers)
        # binanceusdm has proper futures support, while binance() with defaultType='future' doesn't
        self.exchange = ccxt_async.binanceusdm(options)
        
        # Enable testnet if configured
        if self.testnet:
            try:
                self.exchange.set_sandbox_mode(True)
                self.logger.info("Binance Futures testnet mode enabled")
            except Exception as e:
                self.logger.warning(f"Could not enable testnet mode: {e}")
        
        # Load markets with retries
        attempts = EXCHANGE_RETRIES
        last_err = None
        for i in range(1, attempts + 1):
            try:
                await self.load_markets(reload=True)
                self.logger.info(f"Binance Futures initialized: {len(self.markets)} markets loaded")
                return
            except Exception as e:
                last_err = e
                error_type = type(e).__name__
                error_msg = str(e)
                self.logger.warning(
                    f"Binance Futures market load attempt {i}/{attempts} failed: {error_type}: {error_msg[:150]}"
                )
                
                # Log specific error details for debugging
                if "DDoSProtection" in error_type or "-1003" in error_msg or "too many requests" in error_msg.lower() or "banned" in error_msg.lower():
                    # Extract ban timestamp if available
                    import re
                    ban_match = re.search(r'banned until (\d+)', error_msg)
                    if ban_match:
                        ban_until = int(ban_match.group(1))
                        import time
                        time_left = max(0, (ban_until / 1000) - time.time())
                        if time_left > 0:
                            hours_left = time_left / 3600
                            self.logger.warning(f"[!] Binance IP rate limit ban active - wait {hours_left:.1f} hours or use websockets")
                        else:
                            self.logger.warning("[!] Binance rate limit ban expired - retrying...")
                    else:
                        self.logger.warning("[!] Binance rate limit protection active - wait before retrying")
                elif "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                    self.logger.warning("[!] Authentication error - check API keys and permissions")
                elif "timeout" in error_msg.lower():
                    self.logger.warning("[!] Connection timeout - check network connection")
                
                if i < attempts:
                    await asyncio.sleep(RETRY_DELAY_MULTIPLIER * i)
        
        # If all retries failed, raise error with detailed message
        if last_err:
            error_type = type(last_err).__name__
            error_msg = str(last_err)
            self.logger.error(
                f"[X] Binance Futures initialization failed after {attempts} attempts",
                error_type=error_type,
                error_message=error_msg[:200],
                api_key_configured=bool(self.api_key),
                api_secret_configured=bool(self.api_secret)
            )
            raise last_err
    
    async def load_markets(self, reload: bool = False, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Load markets from Binance Futures."""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized: Cannot load_markets - exchange connection not established")
        
        # Binance futures uses defaultType="future" in options
        mkts = await self.exchange.load_markets(reload=reload)
        
        # Filter for USDT-M perpetual futures only
        futures_markets = {}
        for sym, m in mkts.items():
            # Only include USDT-margined futures
            if "USDT" not in sym:
                continue
            # Binance futures use type="future" in ccxt
            market_type = m.get("type", "")
            # Binance futures can be "future" or "swap" in ccxt
            if market_type not in ("future", "swap"):
                continue
            # Only include perpetual contracts (no delivery date)
            if m.get("expiry"):
                continue  # Skip delivery contracts
            # Only USDT-margined
            if m.get("settle") and m.get("settle") != "USDT":
                continue
            
            futures_markets[sym] = m
        
        self.markets = futures_markets
        return futures_markets
    
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker data for a symbol."""
        if not self.exchange:
            return None
        try:
            # Denormalize symbol for exchange API (CRITICAL: must be "BTCUSDT" format)
            exchange_symbol = self.denormalize_symbol(symbol)
            return await self.exchange.fetch_ticker(exchange_symbol)
        except Exception as e:
            self.logger.warning(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    async def start_websocket(self, symbols: Optional[List[str]] = None):
        """
        Start WebSocket connection and subscribe to ticker streams.
        
        Args:
            symbols: List of symbols to subscribe to (None = all USDT futures)
        """
        # Initialize WebSocket manager if not already done
        if self.ws_manager is None:
            try:
                self.ws_manager = BinanceWebSocketManager(logger=self.logger)
            except Exception as e:
                self.logger.warning(f"Failed to create WebSocket manager: {e}")
                self.use_websocket = False
                return False
        
        if not self.use_websocket:
            return False
        
        try:
            # Normalize symbols if provided
            normalized_symbols = None
            if symbols:
                normalized_symbols = [self.normalize_symbol(s) for s in symbols]
            
            # Register callback to update cache
            async def ticker_update_callback(ticker: Dict[str, Any]):
                symbol = ticker.get("symbol")
                if symbol:
                    self.ws_ticker_cache[symbol] = ticker
            
            # Connect and subscribe
            await self.ws_manager.connect(normalized_symbols)
            
            # Register callbacks for all symbols we care about
            if normalized_symbols:
                for symbol in normalized_symbols:
                    self.ws_manager.register_ticker_callback(symbol, ticker_update_callback)
            else:
                # For all tickers stream, we'll handle updates in the callback
                # Register a generic callback that handles any symbol
                async def generic_callback(ticker: Dict[str, Any]):
                    symbol = ticker.get("symbol")
                    if symbol:
                        self.ws_ticker_cache[symbol] = ticker
                
                # We'll register this for a dummy symbol, but it will catch all updates
                self.ws_manager.register_ticker_callback("ALL", generic_callback)
            
            self.logger.info("[OK] WebSocket connected and subscribed to ticker streams")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket: {e}")
            self.use_websocket = False
            return False
    
    async def stop_websocket(self):
        """Stop WebSocket connection."""
        if self.ws_manager:
            await self.ws_manager.disconnect()
    
    async def fetch_tickers(
        self,
        symbols: Optional[List[str]] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Fetch ticker data for multiple symbols (USDT-M futures).

        - Prefers unified snake_case fetch_tickers() when available.
        - Falls back to per-symbol fetch_ticker() if batch is unsupported or fails.
        - Uses normalized symbols internally ("BTC/USDT"), denormalized only for API calls ("BTCUSDT").
        """
        if not self.exchange:
            return {}

        # Ensure markets are loaded so self.markets is usable
        if not self.markets:
            await self.load_markets(reload=False)

        params = params or {}

        # Diagnostics: confirm exchange + capabilities
        exchange_class = type(self.exchange).__name__
        exchange_id = getattr(self.exchange, "id", "unknown")
        # CRITICAL: Check both snake_case and camelCase for binanceusdm
        # binanceusdm might have has['fetchTickers']=True but has['fetch_tickers']=None
        has_fetch_tickers = (
            self.exchange.has.get("fetch_tickers") is True or
            self.exchange.has.get("fetchTickers") is True or
            hasattr(self.exchange, "fetch_tickers")  # Fallback: if method exists, try it
        )

        self.logger.debug(
            f"fetch_tickers(): exchange_class={exchange_class}, "
            f"exchange_id={exchange_id}, "
            f"has.fetch_tickers={self.exchange.has.get('fetch_tickers')}, "
            f"has.fetchTickers={self.exchange.has.get('fetchTickers')}, "
            f"can_call={has_fetch_tickers}, "
            f"symbols_requested={len(symbols) if symbols else 'all'}"
        )

        results: Dict[str, Any] = {}
        
        # ---------------------------------------------------------
        # 0) WebSocket path: Use real-time data if available (NO REST API CALL!)
        # ---------------------------------------------------------
        if self.use_websocket:
            # Initialize WebSocket manager if not already done
            if self.ws_manager is None:
                try:
                    self.ws_manager = BinanceWebSocketManager(logger=self.logger)
                except Exception as e:
                    self.logger.warning(f"Failed to create WebSocket manager: {e}. Using REST API only.")
                    self.use_websocket = False
            
            # Try to start WebSocket if not connected
            if self.ws_manager and not self.ws_manager.connected:
                try:
                    await self.start_websocket(symbols)
                    # Wait a moment for initial data
                    await asyncio.sleep(1.0)
                except Exception as e:
                    self.logger.warning(f"WebSocket start failed: {e}. Falling back to REST API.")
                    self.use_websocket = False
            
            # Use WebSocket data if available
            if self.ws_manager and self.ws_manager.connected and self.ws_ticker_cache:
                # Convert WebSocket cache to ticker format
                for symbol, ticker_data in self.ws_ticker_cache.items():
                    # Filter invalid tickers (zero price)
                    last_price = ticker_data.get("last", 0.0)
                    if last_price <= 0:
                        continue

                    # Filter by requested symbols if provided
                    if symbols is None:
                        # All symbols requested
                        results[symbol] = {
                            "symbol": symbol,
                            "bid": ticker_data.get("bid", 0.0),
                            "ask": ticker_data.get("ask", 0.0),
                            "last": ticker_data.get("last", 0.0),
                            "mark": ticker_data.get("mark", ticker_data.get("last", 0.0)),
                            "quoteVolume": ticker_data.get("quoteVolume", 0.0),
                            "percentage": ticker_data.get("percentage", 0.0),
                            "info": ticker_data.get("info", {})
                        }
                    else:
                        # Filter by requested symbols
                        normalized_requested = set(self.normalize_symbol(s) for s in symbols)
                        if symbol in normalized_requested:
                            results[symbol] = {
                                "symbol": symbol,
                                "bid": ticker_data.get("bid", 0.0),
                                "ask": ticker_data.get("ask", 0.0),
                                "last": ticker_data.get("last", 0.0),
                                "mark": ticker_data.get("mark", ticker_data.get("last", 0.0)),
                                "quoteVolume": ticker_data.get("quoteVolume", 0.0),
                                "percentage": ticker_data.get("percentage", 0.0),
                                "info": ticker_data.get("info", {})
                            }
                
                if results:
                    self.logger.debug(f"fetch_tickers(): Using WebSocket data - {len(results)} tickers (NO REST API CALL!)")
                    return results

        # ---------------------------------------------------------
        # 1) Preferred path: unified snake_case fetch_tickers() (REST API fallback)
        # ---------------------------------------------------------
        if has_fetch_tickers:
            try:
                # CRITICAL: binanceusdm.fetch_tickers() works best when called without arguments
                # It returns all tickers for the futures exchange
                # For specific symbols, fetch all and filter after (more reliable)
                if symbols is None:
                    # Fetch all tickers (no symbols parameter) - this is the most reliable path
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("fetch_tickers(): calling fetch_tickers() without symbols (fetch all)")
                    raw = await self.exchange.fetch_tickers(params=params)
                else:
                    # For specific symbols, fetch all tickers first, then filter
                    # This is more reliable than passing symbols list (which may not be supported)
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"fetch_tickers(): fetching all tickers, will filter to {len(symbols)} requested symbols"
                        )
                    raw = await self.exchange.fetch_tickers(params=params)
                
                # CRITICAL DEBUG: Log raw response
                self.logger.debug(
                    f"fetch_tickers(): raw response type={type(raw)}, "
                    f"is_dict={isinstance(raw, dict)}, len={len(raw) if isinstance(raw, dict) else 'N/A'}"
                )

                if raw and isinstance(raw, dict):
                    # Process all tickers from raw response
                    # If specific symbols were requested, filter after processing
                    symbol_set = None
                    if symbols is not None:
                        symbol_set = set(symbols)
                    
                    for ex_sym, ticker in raw.items():
                        if not ticker or not isinstance(ticker, dict):
                            continue
                        
                        # CRITICAL: Binance Futures tickers may have bid/ask as None
                        # Use last price as fallback for bid/ask if needed
                        bid = ticker.get("bid")
                        ask = ticker.get("ask")
                        last = ticker.get("last")
                        
                        # Convert None to 0 for validation, but preserve None in ticker data
                        bid_val = bid if bid is not None else 0
                        ask_val = ask if ask is not None else 0
                        last_val = last if last is not None else 0
                        
                        # CRITICAL: Only require last price (required for pricing)
                        # bid/ask can be None for futures tickers (will use last or orderbook)
                        if last_val <= 0:
                            continue  # Must have last price
                        
                        # Update ticker with fallback values for consistency
                        ticker["bid"] = bid_val
                        ticker["ask"] = ask_val
                        
                        # If bid/ask are None, use last price as fallback (common for futures)
                        if bid is None or bid <= 0:
                            bid = last
                            ticker["bid"] = bid
                        if ask is None or ask <= 0:
                            ask = last
                            ticker["ask"] = ask
                        
                        # Normalize exchange symbol to internal format
                        norm = self.normalize_symbol(ex_sym)
                        
                        # If specific symbols were requested, filter
                        if symbol_set is not None and norm not in symbol_set:
                            continue  # Skip symbols not in request list
                        
                        results[norm] = ticker

                self.logger.info(
                    f"fetch_tickers(): batch fetched {len(results)} valid tickers from {len(raw) if isinstance(raw, dict) else 0} raw tickers"
                )
                # If we got anything useful, return early
                if results:
                    return results

                self.logger.warning(
                    f"fetch_tickers(): batch call returned {len(raw) if isinstance(raw, dict) else 0} raw tickers but 0 valid tickers, "
                    f"falling back to per-symbol fetch_ticker()"
                )
            except Exception as e:
                # Log as warning instead of error - fallback will handle it
                # Only log error for critical issues (rate limits, auth errors)
                error_type = type(e).__name__
                error_str = str(e)
                
                # Check if this is a rate limit error - if so, DON'T fall back to per-symbol (would make it worse)
                # Check both error type name and error message
                # IMPORTANT: Treat CCXT "throttle queue is over maxCapacity" as rate-limit pressure too.
                # If we fall back to per-symbol in this state, we can self-DDoS the CCXT request queue.
                is_rate_limit = (
                    'ddos' in error_type.lower() or
                    'ratelimit' in error_type.lower() or
                    any(keyword in error_str.lower() for keyword in [
                        'too many requests', 'rate limit', 'banned', '-1003', 'ddos', '418', 'teapot',
                        'throttle queue', 'maxcapacity', 'maximum-requests-capacity'
                    ])
                )
                is_critical = any(keyword in error_str.lower() for keyword in [
                    'authentication', 'unauthorized', 'invalid key', 'api key',
                    'permission denied', 'forbidden', '403', '401'
                ])
                
                if is_rate_limit:
                    # Rate limit hit - DO NOT fall back to per-symbol fetching (would make it worse)
                    self.logger.error(
                        f"fetch_tickers(): Rate limit hit - batch fetch failed ({error_type}). "
                        f"NOT using per-symbol fallback to avoid making situation worse. "
                        f"Error: {error_str[:150]}"
                    )
                    # Return empty results instead of falling back
                    return results
                elif is_critical:
                    self.logger.error(
                        f"fetch_tickers(): batch fetch_tickers() failed with {error_type}: {error_str[:100]}, "
                        f"falling back to per-symbol"
                    )
                else:
                    # Non-critical errors (timeouts, network issues) - log as warning
                    self.logger.warning(
                        f"fetch_tickers(): batch fetch failed ({error_type}), using per-symbol fallback"
                    )

        # ---------------------------------------------------------
        # 2) Fallback: per-symbol fetch_ticker() in small batches
        # ---------------------------------------------------------
        self.logger.debug("fetch_tickers(): using per-symbol fetch_ticker() fallback")
        
        # If no symbols provided, derive from markets for fallback
        if symbols is None:
            if not self.markets:
                self.logger.warning("fetch_tickers(): fallback - no markets loaded and no symbols provided")
                return results
            market_symbols = list(self.markets.keys())
            # Limit for fallback (smaller batches for individual calls)
            market_symbols = market_symbols[:100]
            symbols = [self.normalize_symbol(s) for s in market_symbols]
        
        if not symbols:
            return results

        # Guardrail: per-symbol fallback is expensive. Don't run it too frequently.
        # This prevents repeated fallback loops from building up CCXT request backlog.
        now_ts = time.time()
        last_fallback = getattr(self, "_last_ticker_fallback_ts", 0.0)
        if now_ts - last_fallback < 60.0:
            return results
        self._last_ticker_fallback_ts = now_ts

        # VERY IMPORTANT:
        # Do NOT use asyncio.gather() here. Even with enableRateLimit=True, gather can enqueue
        # too many requests into CCXT's internal throttle queue and hit maxCapacity(1000).
        batch_size = 10
        success = 0
        failed = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            for sym in batch:
                # IMPORTANT: pass normalized symbols ("BTC/USDT") into self.fetch_ticker()
                # It will denormalize internally.
                try:
                    ticker = await self.fetch_ticker(sym)
                except Exception as e:
                    failed += 1
                    err_s = str(e).lower()
                    # If CCXT queue is under pressure, stop immediately (do not keep enqueueing).
                    if "throttle queue" in err_s or "maxcapacity" in err_s:
                        return results
                    # Only log a few failures to avoid spam
                    if failed <= 3:
                        self.logger.debug(f"fetch_ticker() exception for {sym}: {e}")
                    continue

                if not ticker or not isinstance(ticker, dict):
                    failed += 1
                    continue

                # CRITICAL: Binance Futures tickers may have bid/ask as None
                bid = ticker.get("bid")
                ask = ticker.get("ask")
                last = ticker.get("last")
                
                # Convert None to 0 for validation
                bid_val = bid if bid is not None else 0
                ask_val = ask if ask is not None else 0
                last_val = last if last is not None else 0
                
                # CRITICAL: Only require last price (required for pricing)
                # bid/ask can be None for futures tickers (will use last or orderbook)
                if last_val <= 0:
                    continue  # Must have last price
                
                # Update ticker with fallback values for consistency
                ticker["bid"] = bid_val
                ticker["ask"] = ask_val
                
                # If bid/ask are None, use last price as fallback (common for futures)
                if bid is None or bid <= 0:
                    bid = last
                    ticker["bid"] = bid
                if ask is None or ask <= 0:
                    ask = last
                    ticker["ask"] = ask

                results[sym] = ticker
                success += 1

            # Small pause between batches to respect rate limits
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.2)  # Increased from 0.05s to 0.2s to reduce rate limit pressure

        self.logger.debug(
            f"fetch_tickers(): fallback fetched {success} tickers, {failed} failures "
            f"out of {len(symbols)} symbols"
        )
        return results
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100, params: Optional[Dict] = None) -> List[List]:
        """Fetch OHLCV data for a symbol."""
        if not self.exchange:
            return []
        try:
            # Ensure params is always a dict (CCXT requires it)
            if params is None:
                params = {}
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, params=params)
        except Exception as e:
            self.logger.warning(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return []
    
    async def close(self):
        """Close exchange connection and WebSocket."""
        if self.ws_manager:
            await self.stop_websocket()
        
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                self.logger.warning(f"Error closing exchange: {e}")
    
    async def fetch_order_book(self, symbol: str, limit: int = 50, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Fetch orderbook for a symbol."""
        if not self.exchange:
            return None
        try:
            # Ensure params is always a dict (CCXT requires it)
            if params is None:
                params = {}
            return await self.exchange.fetch_order_book(symbol, limit=limit, params=params)
        except Exception as e:
            self.logger.warning(f"Error fetching orderbook for {symbol}: {e}")
            return None
    
    async def fetch_balance(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Fetch account balance."""
        if not self.exchange:
            return {}
        try:
            # Ensure params is always a dict (CCXT requires it)
            if params is None:
                params = {}
            # Binance futures uses defaultType="future" in options
            balance = await self.exchange.fetch_balance(params=params)
            
            # CRITICAL: Handle BNFCR (Binance Credits) and USDC for EEA/Iceland users
            # EEA users cannot hold USDT, so they use USDC or BNFCR as collateral in Multi-Asset Mode
            # We need to aggregate these into the "USDT" balance for the bot's logic
            
            # 1. Get raw balances
            usdt_total = float(balance.get('USDT', {}).get('total', 0.0))
            usdt_free = float(balance.get('USDT', {}).get('free', 0.0))
            usdt_used = float(balance.get('USDT', {}).get('used', 0.0))
            
            bnfcr_total = float(balance.get('BNFCR', {}).get('total', 0.0))
            bnfcr_free = float(balance.get('BNFCR', {}).get('free', 0.0))
            bnfcr_used = float(balance.get('BNFCR', {}).get('used', 0.0))
            
            usdc_total = float(balance.get('USDC', {}).get('total', 0.0))
            usdc_free = float(balance.get('USDC', {}).get('free', 0.0))
            usdc_used = float(balance.get('USDC', {}).get('used', 0.0))
            
            # 2. Check if we need to synthesize USDT balance
            # If USDT is effectively zero but we have USDC or positive BNFCR, use them
            has_alternative_collateral = (usdc_total > 5.0) or (bnfcr_total > 5.0)
            
            if usdt_total < 5.0 and has_alternative_collateral:
                self.logger.info(
                    f"EEA Mode: Synthesizing USDT balance from collateral (USDC: {usdc_total:.2f}, BNFCR: {bnfcr_total:.2f})"
                )
                
                # Create USDT entry if missing
                if 'USDT' not in balance:
                    balance['USDT'] = {}
                
                # Combine balances (BNFCR + USDC)
                # Note: This is an approximation. In Multi-Asset mode, Binance applies haircuts (e.g. 5% on USDC)
                # But for bot logic, raw sum is sufficient to allow trading
                total_synth = bnfcr_total + usdc_total
                free_synth = bnfcr_free + usdc_free
                used_synth = bnfcr_used + usdc_used
                
                # Map synthesized values to USDT
                balance['USDT']['free'] = free_synth
                balance['USDT']['used'] = used_synth
                balance['USDT']['total'] = total_synth
                
                # Also update total/free top-level keys
                balance['free']['USDT'] = free_synth
                balance['total']['USDT'] = total_synth
                balance['used']['USDT'] = used_synth
            
            return balance
        except Exception as e:
            print(f"DEBUG: fetch_balance failed with error: {e}", flush=True)  # TEMPORARY DEBUG
            self.logger.warning(f"Error fetching balance: {e}")
            return {}
    
    async def fetch_positions(self, symbols: Optional[List[str]] = None, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Fetch open futures positions."""
        if not self.exchange:
            return []
        try:
            # Ensure params is always a dict (CCXT requires it)
            if params is None:
                params = {}
            # Binance futures uses defaultType="future" in options
            positions = await self.exchange.fetch_positions(symbols, params=params)
            # Filter for open positions only (contracts != 0)
            return [p for p in positions if p.get("contracts", 0) != 0]
        except Exception as e:
            self.logger.warning(f"Error fetching positions: {e}")
            return []
    
    async def create_order(self, symbol: str, order_type: str, side: str,
                          amount: float, price: Optional[float] = None,
                          params: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a futures order."""
        # CRITICAL FIX: Pass normalized symbol to CCXT so it can look up market metadata (precision, etc.)
        # CCXT handles denormalization internally for the API call
        # Previous manual denormalization (BEATUSDT) caused failures because 'BEATUSDT' key doesn't exist in markets dict
        
        if self.dry_run:
            import time
            exchange_symbol = self.denormalize_symbol(symbol) # Only needed for dry run ID
            return {
                "id": f"DRY_{int(time.time() * 1000)}",
                "symbol": exchange_symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price or 0.0,
                "status": "closed",
                "filled": amount,
                "average": price or 0.0,
            }
        
        if not self.exchange:
            raise RuntimeError("Exchange not initialized: Cannot create_order - exchange connection not established")
        
        try:
            # Binance futures doesn't use positionSide for USDT-M
            # Filter out positionSide if present (it's not needed for Binance)
            if params is None:
                params = {}
            # Remove positionSide if present (Binance doesn't use it for USDT-M)
            if "positionSide" in params:
                params = {k: v for k, v in params.items() if k != "positionSide"}
            
            # Binance futures uses "buy" for long, "sell" for short
            # Side is already "buy" or "sell" from our code
            # Pass normalized symbol (e.g. BEAT/USDT:USDT)
            
            # CRITICAL FIX: Ensure amount respects precision and min quantity requirements
            try:
                # Check min quantity and min notional first
                market = self.exchange.market(symbol) if self.exchange.markets else None
                if market:
                    min_qty = market.get('limits', {}).get('amount', {}).get('min')
                    min_cost = market.get('limits', {}).get('cost', {}).get('min')
                    
                    # 1. Min Quantity Check
                    if min_qty and amount < min_qty:
                        self.logger.warning(f"Order amount {amount} < min_qty {min_qty} for {symbol}. Skipping.")
                        return {
                            "id": None,
                            "status": "rejected", 
                            "info": {"msg": f"Amount {amount} < min_qty {min_qty}"}
                        }
                        
                    # 2. Min Notional Check (BINANCE BEST PRACTICE: Validate before order placement)
                    # For market orders, price is None - must fetch current price for notional calculation
                    if min_cost:
                        notional = None
                        if price:
                            # Limit order: use provided price
                            notional = amount * price
                        else:
                            # Market order: fetch current price for notional calculation
                            try:
                                ticker = await self.exchange.fetch_ticker(symbol)
                                current_price = ticker.get('last') or ticker.get('close') or ticker.get('bid')
                                if current_price:
                                    notional = amount * current_price
                                    self.logger.debug(f"[NOTIONAL_CHECK] Market order: using current price {current_price} for {symbol}")
                                else:
                                    self.logger.warning(f"[NOTIONAL_CHECK] Could not get current price for {symbol}, skipping notional check")
                            except Exception as e:
                                self.logger.debug(f"[NOTIONAL_CHECK] Failed to fetch price for notional check: {e}")
                                # Continue without notional check - exchange will reject if too small
                        
                        if notional and notional < min_cost:
                            self.logger.warning(f"[NOTIONAL_CHECK] Order value ${notional:.2f} < min_cost ${min_cost} for {symbol}. Skipping.")
                            return {
                                "id": None,
                                "status": "rejected",
                                "info": {"msg": f"Value ${notional:.2f} < min_cost ${min_cost}"}
                            }

                # 3. Precision Formatting
                formatted_amount = self.exchange.amount_to_precision(symbol, amount)
                
                # Check if amount is effectively zero after truncation
                if float(formatted_amount) == 0:
                    self.logger.warning(f"Order amount {amount} for {symbol} truncated to 0 by precision settings. Skipping.")
                    return {
                        "id": None,
                        "status": "rejected", 
                        "info": {"msg": "Amount too small for precision"}
                    }
                
                # Use the formatted amount
                final_amount = float(formatted_amount)
                
                # Debug log for precision handling
                if final_amount != amount:
                    self.logger.debug(f"Precision: Adjusted {symbol} amount {amount} -> {final_amount}")
                    
            except Exception as e:
                # Fallback if precision/market lookup fails
                self.logger.debug(f"Precision formatting/check failed for {symbol}: {e}. Using raw amount.")
                final_amount = amount

            # CRITICAL: Log before calling CCXT create_order
            self.logger.warning(
                f"[BINANCE_ORDER_CALL] {symbol} {side} {final_amount:.6f}: "
                f"Calling CCXT create_order (type={order_type}, price={price})"
            )
            
            try:
                order_result = await self.exchange.create_order(
                    symbol, order_type, side, final_amount, price, params
                )
                
                # CRITICAL: Log the result from CCXT
                self.logger.warning(
                    f"[BINANCE_ORDER_RESULT] {symbol} {side}: "
                    f"CCXT returned id={order_result.get('id')}, status={order_result.get('status')}, "
                    f"filled={order_result.get('filled')}, amount={order_result.get('amount')}"
                )
                
                return order_result
            except Exception as e:
                # GUARD: Log -4140 (Invalid symbol status) as warning, not error
                # This is expected behavior for symbols in "Reduce Only" mode or temporarily unavailable
                error_str = str(e)
                exchange_symbol = self.denormalize_symbol(symbol) # For logging
                if "-4140" in error_str or "Invalid symbol status" in error_str:
                    self.logger.warning(f"Symbol unavailable for new positions: {exchange_symbol} ({error_str[:60]})")
                else:
                    self.logger.error(
                        f"[BINANCE_ORDER_ERROR] {exchange_symbol} {side} {amount}: "
                        f"{type(e).__name__}: {error_str}"
                    )
                    import traceback
                    self.logger.error(f"[BINANCE_ORDER_ERROR_TRACEBACK] {traceback.format_exc()}")
                raise
        except Exception as e:
            # Outer exception handler for the entire create_order function
            error_str = str(e)
            exchange_symbol = self.denormalize_symbol(symbol) if hasattr(self, 'denormalize_symbol') else symbol
            
            # Check for Binance -4140 error (Invalid symbol status) - log at DEBUG level
            is_invalid_symbol = "-4140" in error_str or "Invalid symbol status" in error_str
            if is_invalid_symbol:
                self.logger.debug(
                    f"[BINANCE_ORDER_INVALID_SYMBOL] {exchange_symbol} {side} {amount}: "
                    f"Symbol in Reduce Only mode or delisted (-4140). Skipping order."
                )
            else:
                # Other errors - log at ERROR level
                self.logger.error(
                    f"[BINANCE_ORDER_OUTER_ERROR] {exchange_symbol} {side} {amount}: "
                    f"{type(e).__name__}: {error_str}"
                )
            raise
    
    async def set_leverage(self, leverage: int, symbol: str, params: Optional[Dict] = None):
        """Set leverage for a symbol."""
        # Pass normalized symbol to CCXT
        
        if self.dry_run:
            return
        if not self.exchange:
            raise RuntimeError("Exchange not initialized: Cannot cancel_order - exchange connection not established")
        
        # GUARD: Skip delivery contracts (those with dates like 260327) or symbols not in markets
        # Use normalized symbol for checks
        if "USDTUSDT" in symbol:
             self.logger.debug(f"Skipping leverage set for invalid contract: {symbol}")
             return

        # GUARD: Verify symbol exists in markets (if markets loaded)
        if self.markets:
            if symbol not in self.markets:
                self.logger.debug(f"Skipping leverage set for unknown symbol: {symbol}")
                return
        
        try:
            # Binance futures uses set_leverage method
            # Binance requires marginMode (ISOLATED or CROSS) - use configured mode
            if params is None:
                params = {}
            if "marginMode" not in params:
                params["marginMode"] = self.margin_mode.upper()  # Use configured margin mode
            
            await self.exchange.set_leverage(leverage, symbol, params=params)
        except Exception as e:
            self.logger.warning(f"Error setting leverage for {symbol}: {e}")
    
    async def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Fetch order status by order ID."""
        # Pass normalized symbol to CCXT
        
        if not self.exchange:
            return None
        try:
            # Ensure params is always a dict (CCXT requires it)
            if params is None:
                params = {}
            return await self.exchange.fetch_order(order_id, symbol, params=params)
        except Exception as e:
            self.logger.warning(f"Error fetching order {order_id} for {symbol}: {e}")
            return None
    
    async def place_trailing_stop(
        self,
        symbol: str,
        side: str,
        amount: float,
        callback_rate: float = 1.0,
        activation_price: Optional[float] = None,
        current_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place a TRUE trailing stop order on Binance Futures (server-side).
        
        This uses Binance's TRAILING_STOP_MARKET order type which automatically
        adjusts the stop price as the market moves in your favor.
        
        Args:
            symbol: Trading pair (normalized, e.g., 'BTC/USDT:USDT')
            side: 'sell' to close long, 'buy' to close short
            amount: Position size to close
            callback_rate: Trailing distance in % (e.g., 1.0 = 1%, max 5%)
            activation_price: Optional price to activate trailing (default: immediate)
            current_price: Current market price (used for activation price if not provided)
        
        Returns:
            Order response dict or None on failure
        """
        if self.dry_run:
            import time
            self.logger.info(f"[DRY] Trailing stop: {symbol} {side} {amount} @ {callback_rate}% callback")
            return {
                "id": f"DRY_TRAIL_{int(time.time() * 1000)}",
                "symbol": symbol,
                "type": "TRAILING_STOP_MARKET",
                "side": side,
                "amount": amount,
                "callbackRate": callback_rate,
                "status": "open",
            }
        
        if not self.exchange:
            self.logger.error("Exchange not initialized: Cannot place trailing stop")
            return None
        
        try:
            # Format amount to exchange precision
            try:
                formatted_amount = self.exchange.amount_to_precision(symbol, amount)
                final_amount = float(formatted_amount)
            except Exception:
                final_amount = amount
            
            # Get current price if not provided (needed for activation price)
            if not current_price:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = ticker.get('last', ticker.get('close', 0))
                except Exception:
                    self.logger.warning(f"[TRAIL] Could not get price for {symbol}")
                    return None
            
            if not current_price or current_price <= 0:
                self.logger.warning(f"[TRAIL] Invalid price for {symbol}: {current_price}")
                return None
            
            # Clamp callback rate to Binance limits (0.1% - 5%)
            callback_rate_pct = max(0.1, min(5.0, callback_rate))
            
            # CRITICAL: Binance requires callbackRate in basis points (BIPS), not percentage!
            # 1 basis point = 0.01%, so 2.5% = 250 basis points
            # Convert percentage to basis points
            callback_rate_bips = callback_rate_pct * 100  # 2.5% -> 250 basis points
            
            # Build params for TRAILING_STOP_MARKET order
            # Binance requires: callbackRate in basis points (mandatory), activationPrice (optional)
            params = {
                'callbackRate': callback_rate_bips,  # Trailing distance in basis points (e.g., 250 = 2.5%)
                'reduceOnly': True,
            }
            
            # Set activation price if provided, otherwise Binance activates immediately
            if activation_price and activation_price > 0:
                try:
                    params['activationPrice'] = float(self.exchange.price_to_precision(symbol, activation_price))
                except Exception:
                    params['activationPrice'] = activation_price
            
            self.logger.info(
                f"[TRAIL] Placing TRAILING_STOP_MARKET: {symbol} {side} {final_amount} "
                f"@ {callback_rate_pct}% callback ({callback_rate_bips} bips) (current: ${current_price:.4f}) - Will follow price up!"
            )
            
            # Place the REAL trailing stop order - Binance will automatically:
            # 1. Follow price upward as it moves favorably
            # 2. Keep stop price trailing by callback_rate% behind peak
            # 3. Execute when price reverses and hits trailing stop
            order = await self.exchange.create_order(
                symbol,
                'TRAILING_STOP_MARKET',  # TRUE trailing stop order type
                side,
                final_amount,
                None,  # No limit price for market trailing stop
                params
            )
            
            order_id = order.get('id') or order.get('orderId') or order.get('clientOrderId', 'UNKNOWN')
            # Extract actual callback rate from response (might be in bips)
            response_callback = order.get('callbackRate', callback_rate_bips)
            # If response is in bips, convert to percentage for logging
            if response_callback > 10:  # Likely in bips (250) not percentage (2.5)
                response_callback_pct = response_callback / 100.0
            else:
                response_callback_pct = response_callback
            self.logger.info(
                f"[TRAIL] ✅ SUCCESS: {symbol} trailing stop placed @ {response_callback_pct}% callback "
                f"(Order ID: {order_id}) - Following price upward, will catch reversals!"
            )
            
            # CRITICAL: Verify the trailing stop was actually placed
            try:
                await asyncio.sleep(0.5)  # Brief delay to allow order to appear
                verify_orders = await self.fetch_stop_orders(symbol)
                found = any(
                    (o.get('id') == str(order_id) or o.get('id') == order_id or 
                     o.get('orderId') == str(order_id) or o.get('clientOrderId') == str(order_id))
                    for o in verify_orders
                )
                if found:
                    self.logger.info(f"[TRAIL] ✅ VERIFIED: Trailing stop {order_id} confirmed active on Binance - following price!")
                else:
                    self.logger.warning(
                        f"[TRAIL] ⚠️ WARNING: Trailing stop {order_id} not found in open orders. "
                        f"It may still be active but not yet visible. Check Binance UI."
                    )
            except Exception as verify_err:
                self.logger.warning(f"[TRAIL] Could not verify trailing stop {order_id}: {verify_err}")
            
            return order
            
        except Exception as e:
            error_str = str(e)
            error_code = None
            if hasattr(e, 'code'):
                error_code = e.code
            elif "code" in str(e):
                # Try to extract error code from error string (e.g., "code":-4140)
                import re
                match = re.search(r'"code":(-?\d+)', error_str)
                if match:
                    error_code = int(match.group(1))
            
            # Log detailed error information
            self.logger.error(f"[TRAIL] FAILED for {symbol}: {e} (code: {error_code})")
            
            if "ReduceOnly" in error_str or "-4140" in error_str:
                # Position doesn't exist or symbol status invalid
                self.logger.warning(f"[TRAIL] Position not found or invalid for {symbol}, trying static stop")
                return await self._place_static_stop(symbol, side, final_amount, callback_rate, current_price)
            elif "notional" in error_str.lower() or "MIN_NOTIONAL" in error_str or "-4141" in error_str:
                # Position too small
                self.logger.warning(f"[TRAIL] Position too small for {symbol} (size: {final_amount}), trying static stop")
                return await self._place_static_stop(symbol, side, final_amount, callback_rate, current_price)
            elif "4046" in error_str or "not supported" in error_str.lower() or error_code == -4046:
                # Symbol doesn't support TRAILING_STOP_MARKET
                self.logger.warning(f"[TRAIL] Symbol {symbol} doesn't support trailing stop, using static stop")
                return await self._place_static_stop(symbol, side, final_amount, callback_rate, current_price)
            elif "-2010" in error_str or error_code == -2010:
                # NEW_ORDER_REJECTED - might be insufficient margin or other order rejection
                self.logger.warning(f"[TRAIL] Order rejected for {symbol}, trying static stop")
                return await self._place_static_stop(symbol, side, final_amount, callback_rate, current_price)
            else:
                # Unknown error - try static stop as fallback
                self.logger.warning(f"[TRAIL] Unknown error for {symbol}, attempting static stop as fallback")
                return await self._place_static_stop(symbol, side, final_amount, callback_rate, current_price)
    
    async def _place_static_stop(
        self,
        symbol: str,
        side: str,
        amount: float,
        callback_rate: float,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback: Place a static STOP_MARKET order for symbols that don't support trailing stops.
        """
        try:
            # Calculate stop price based on callback rate
            if side.lower() == 'sell':
                stop_price = current_price * (1 - callback_rate / 100)
            else:
                stop_price = current_price * (1 + callback_rate / 100)
            
            # Format stop price
            try:
                stop_price = float(self.exchange.price_to_precision(symbol, stop_price))
            except Exception:
                pass
            
            self.logger.info(f"[TRAIL-STATIC] Placing static SL: {symbol} {side} @ ${stop_price:.4f}")
            
            order = await self.exchange.create_order(
                symbol,
                'STOP_MARKET',
                side,
                amount,
                None,
                {
                    'stopPrice': stop_price,
                    'reduceOnly': True,
                }
            )
            
            self.logger.info(f"[TRAIL-STATIC] SUCCESS: {symbol} static SL @ ${stop_price:.4f}")
            return order
            
        except Exception as e:
            self.logger.warning(f"[TRAIL-STATIC] FAILED for {symbol}: {e}")
            return None
    
    async def place_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        current_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place a regular STOP_MARKET order on Binance (server-side stop-loss).
        
        This is simpler and more reliable than trailing stops - just a regular stop-loss
        that can be updated later by cancelling and replacing.
        
        Args:
            symbol: Trading pair (normalized, e.g., 'BTC/USDT:USDT')
            side: 'sell' to close long, 'buy' to close short
            quantity: Position size to close
            stop_price: Stop price (absolute price level)
            current_price: Current market price (for validation)
        
        Returns:
            Order response dict or None on failure
        """
        if self.dry_run:
            import time
            self.logger.info(f"[DRY] Stop-loss: {symbol} {side} {quantity} @ ${stop_price:.4f}")
            return {
                "id": f"DRY_SL_{int(time.time() * 1000)}",
                "symbol": symbol,
                "type": "STOP_MARKET",
                "side": side,
                "amount": quantity,
                "stopPrice": stop_price,
                "status": "open",
            }
        
        if not self.exchange:
            self.logger.error("Exchange not initialized: Cannot place stop-loss")
            return None
        
        try:
            # Format quantity to exchange precision
            try:
                formatted_quantity = self.exchange.amount_to_precision(symbol, quantity)
                final_quantity = float(formatted_quantity)
            except Exception:
                final_quantity = quantity
            
            # Format stop price to exchange precision
            try:
                formatted_stop_price = self.exchange.price_to_precision(symbol, stop_price)
                final_stop_price = float(formatted_stop_price)
            except Exception:
                final_stop_price = stop_price
            
            # Validate stop price is reasonable (must be below current for long, above for short)
            if current_price:
                if side.lower() == 'sell':  # Closing long position
                    if final_stop_price >= current_price:
                        self.logger.warning(f"[SL] Stop price {final_stop_price:.4f} >= current {current_price:.4f} for long. Adjusting.")
                        final_stop_price = current_price * 0.99  # 1% below current
                else:  # 'buy' - closing short position
                    if final_stop_price <= current_price:
                        self.logger.warning(f"[SL] Stop price {final_stop_price:.4f} <= current {current_price:.4f} for short. Adjusting.")
                        final_stop_price = current_price * 1.01  # 1% above current
            
            current_price_str = f"{current_price:.4f}" if current_price else "N/A"
            self.logger.info(f"[SL] Placing STOP_MARKET: {symbol} {side} {final_quantity:.4f} @ ${final_stop_price:.4f} (current: ${current_price_str})")
            
            # IMPORTANT: Binance Futures stop-loss orders are "Conditional Orders"
            # They do NOT appear in "Open Orders" until the stop price is reached
            # They are active and will trigger when price hits stop_price
            # You can verify them via API or check "Conditional Orders" section (if available)
            
            # Place STOP_MARKET order via CCXT (handles Binance-specific formatting)
            order = await self.exchange.create_order(
                symbol,
                'STOP_MARKET',
                side,
                final_quantity,
                None,  # No limit price for market stop
                {
                    'stopPrice': final_stop_price,
                    'reduceOnly': True,  # Only reduce position, don't open new one
                }
            )
            
            order_id = order.get('id') or order.get('orderId') or order.get('clientOrderId', 'UNKNOWN')
            self.logger.info(f"[SL] SUCCESS: {symbol} stop-loss placed @ ${final_stop_price:.4f} (Order ID: {order_id})")
            
            # Verify the order was actually placed by fetching it back
            try:
                await asyncio.sleep(0.5)  # Brief delay to allow order to appear
                verify_orders = await self.fetch_stop_orders(symbol)
                found = any(o.get('id') == str(order_id) or o.get('id') == order_id for o in verify_orders)
                if found:
                    self.logger.info(f"[SL] VERIFIED: Stop-loss order {order_id} confirmed active on Binance")
                else:
                    self.logger.warning(f"[SL] WARNING: Order {order_id} not found in open orders - may still be active but not yet visible")
            except Exception as verify_err:
                self.logger.debug(f"[SL] Could not verify order {order_id}: {verify_err}")
            
            return order
            
        except Exception as e:
            error_str = str(e)
            if "ReduceOnly" in error_str:
                self.logger.debug(f"[SL] Skipped (no position): {symbol}")
            elif "notional" in error_str.lower() or "MIN_NOTIONAL" in error_str:
                self.logger.debug(f"[SL] Skipped (too small): {symbol}")
            elif "-4140" in error_str or "Invalid symbol status" in error_str:
                self.logger.debug(f"[SL] Symbol unavailable: {symbol}")
            else:
                self.logger.warning(f"[SL] FAILED for {symbol}: {e}")
            return None
    
    async def fetch_stop_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch all stop-loss orders (conditional orders) for a symbol or all symbols.
        
        NOTE: Binance Futures stop-loss orders are conditional and appear in open orders
        when active. This method fetches them and filters for STOP orders.
        
        Args:
            symbol: Trading pair (normalized), or None for all symbols
        
        Returns:
            List of stop orders with their details
        """
        if self.dry_run or not self.exchange:
            return []
        
        try:
            # Fetch open orders - STOP_MARKET orders should appear here when active
            orders = await self.exchange.fetch_open_orders(symbol)
            
            # Filter for STOP orders (STOP_MARKET, TRAILING_STOP_MARKET, etc.)
            stop_orders = []
            for order in orders:
                order_type = order.get('type', '').upper()
                if 'STOP' in order_type or order.get('stopPrice'):
                    stop_orders.append({
                        'id': order.get('id'),
                        'symbol': order.get('symbol'),
                        'type': order_type,
                        'side': order.get('side'),
                        'amount': order.get('amount'),
                        'stopPrice': order.get('stopPrice'),
                        'price': order.get('price'),
                        'status': order.get('status'),
                        'info': order.get('info', {})
                    })
            
            if stop_orders:
                self.logger.info(f"[SL] Found {len(stop_orders)} active stop-loss orders")
                for so in stop_orders:
                    self.logger.info(f"[SL]   {so['symbol']} {so['type']} {so['side']} {so.get('amount', 'N/A')} @ ${so.get('stopPrice', 'N/A')}")
            
            return stop_orders
        except Exception as e:
            self.logger.warning(f"[SL] Error fetching stop orders for {symbol}: {e}")
            return []
    
    async def cancel_trailing_stop(self, symbol: str) -> bool:
        """
        Cancel any open stop orders for a symbol.
        
        Args:
            symbol: Trading pair (normalized)
        
        Returns:
            True if cancelled or no orders, False on error
        """
        if self.dry_run or not self.exchange:
            return True
        
        try:
            # Fetch open orders for symbol (includes conditional/stop orders)
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
            cancelled_count = 0
            for order in open_orders:
                order_type = order.get('type', '').upper()
                # Cancel any stop orders (STOP_MARKET, TRAILING_STOP_MARKET, STOP_LOSS, etc.)
                if 'STOP' in order_type:
                    try:
                        await self.exchange.cancel_order(order['id'], symbol)
                        self.logger.info(f"[SL] Cancelled stop order: {symbol} ({order_type}) Order ID: {order.get('id', 'N/A')}")
                        cancelled_count += 1
                    except Exception as cancel_error:
                        self.logger.warning(f"[SL] Failed to cancel order {order.get('id')}: {cancel_error}")
            
            if cancelled_count == 0:
                self.logger.debug(f"[SL] No stop orders found to cancel for {symbol}")
            return True
        except Exception as e:
            self.logger.warning(f"Error cancelling stop order for {symbol}: {e}")
            return False
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for internal use.
        OPTIMIZED: Uses caching and string interning to avoid repeated string operations.
        Converts exchange-specific format (e.g., "BTCUSDT") to normalized format (e.g., "BTC/USDT").
        """
        # OPTIMIZATION: Check cache first
        if symbol in self._normalize_cache:
            return self._normalize_cache[symbol]
        
        # GUARD: Reject delivery contracts (those with 6-digit dates like 260327)
        # Pattern: Symbol ends with 6 digits (delivery date)
        clean_for_check = symbol.replace("/", "").replace("-", "").replace(":", "")
        if len(clean_for_check) > 6 and clean_for_check[-6:].isdigit():
            # This is a delivery contract - return a normalized version but mark it as invalid
            # The caller should filter this out, but we'll normalize it anyway to prevent crashes
            self.logger.debug(f"Detected delivery contract in normalization: {symbol}")
        
        # Remove any separators
        clean = symbol.replace("/", "").replace("-", "").replace(":", "")
        
        # If it already ends with USDT (like "BTCUSDT"), extract base
        if clean.endswith("USDT"):
            # Check if it's double USDT (like "CHRUSDTUSDT" from bad normalization)
            if clean.endswith("USDTUSDT"):
                base = clean[:-8]  # Remove "USDTUSDT"
            else:
                base = clean[:-4]  # Remove "USDT"
            result = f"{base}/USDT"
        elif "/" in symbol:
            # If it's already in normalized format with slash, return as-is
            result = symbol.upper()
        elif len(clean) <= 10:  # Reasonable base currency length
            # If it's just base currency, add USDT
            result = f"{clean}/USDT"
        else:
            result = symbol.upper()
        
        # OPTIMIZATION: Intern string for faster comparisons and lower memory
        result = sys.intern(result)
        
        # OPTIMIZATION: Cache result (with size limit)
        if len(self._normalize_cache) < self._cache_max_size:
            self._normalize_cache[symbol] = result
        elif len(self._normalize_cache) >= self._cache_max_size:
            # Clear cache if too large (simple FIFO eviction)
            self._normalize_cache.clear()
            self._normalize_cache[symbol] = result
        
        return result
    
    def denormalize_symbol(self, symbol: str) -> str:
        """
        Denormalize symbol format for Binance Futures API.
        OPTIMIZED: Uses caching and string interning to avoid repeated string operations.
        CRITICAL: Binance Futures requires "BTCUSDT" format (no slashes, no :USDT suffix).
        """
        # OPTIMIZATION: Check cache first
        if symbol in self._denormalize_cache:
            return self._denormalize_cache[symbol]
        
        # Remove any separators
        clean = symbol.replace("/", "").replace("-", "").replace(":", "")
        
        # If it already ends with USDT (like "BTCUSDT"), return as-is
        if clean.endswith("USDT"):
            # Check if it's double USDT (like "CHRUSDTUSDT" from bad normalization)
            if clean.endswith("USDTUSDT"):
                result = clean[:-4].upper()  # Remove one "USDT"
            else:
                result = clean.upper()
        elif len(clean) <= 10:  # Reasonable base currency length
            # If it's just base currency, add USDT
            result = f"{clean}USDT".upper()
        else:
            result = clean.upper()
        
        # OPTIMIZATION: Intern string for faster comparisons and lower memory
        result = sys.intern(result)
        
        # OPTIMIZATION: Cache result (with size limit)
        if len(self._denormalize_cache) < self._cache_max_size:
            self._denormalize_cache[symbol] = result
        elif len(self._denormalize_cache) >= self._cache_max_size:
            # Clear cache if too large (simple FIFO eviction)
            self._denormalize_cache.clear()
            self._denormalize_cache[symbol] = result
        
        return result
    
    def is_futures_market(self, market: Dict[str, Any]) -> bool:
        """Check if a market is a futures market."""
        market_type = market.get("type", "")
        # Binance futures can be "future" or "swap" in ccxt
        return market_type in ("future", "swap")
    
    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            try:
                await self.exchange.close()
            except (ConnectionError, OSError, AttributeError, RuntimeError):
                # Ignore cleanup errors during shutdown - connection may already be closed
                pass

