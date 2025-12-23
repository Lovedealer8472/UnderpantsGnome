"""
OKX Exchange Wrapper - OKX USDT Perpetual Futures trading.
"""
import sys
import logging
import asyncio
import ccxt.async_support as ccxt_async
from typing import Dict, Any, Optional, List
from .base import ExchangeBase
from ..logger import get_logger
from ..config import (
    OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE, DRY_RUN, EXCHANGE_TIMEOUT_MS, 
    EXCHANGE_RETRIES, RETRY_DELAY_MULTIPLIER
)


class OKXExchange(ExchangeBase):
    """Wrapper for OKX USDT Perpetual Futures trading."""
    
    def __init__(self, config=None):
        """Initialize OKX exchange wrapper."""
        self.logger = get_logger("OKX")
        self.config = config
        self.exchange = None
        self.markets = {}
        
        self._normalize_cache = {}
        self._denormalize_cache = {}
        self._cache_max_size = 1000
        
        if config:
            self.api_key = getattr(config, 'OKX_API_KEY', OKX_API_KEY)
            self.api_secret = getattr(config, 'OKX_API_SECRET', OKX_API_SECRET)
            self.passphrase = getattr(config, 'OKX_PASSPHRASE', OKX_PASSPHRASE)
            self.dry_run = getattr(config, 'DRY_RUN', DRY_RUN)
            self.testnet = getattr(config, 'OKX_TESTNET', False)
        else:
            self.api_key = OKX_API_KEY
            self.api_secret = OKX_API_SECRET
            self.passphrase = OKX_PASSPHRASE
            self.dry_run = DRY_RUN
            self.testnet = False
    
    async def initialize(self):
        """Initialize OKX exchange connection."""
        options = {
            "enableRateLimit": True,
            "timeout": EXCHANGE_TIMEOUT_MS,
            "options": {
                "defaultType": "swap",  # OKX uses "swap" for perpetuals
            }
        }
        
        if self.api_key and self.api_secret and self.passphrase:
            options["apiKey"] = self.api_key
            options["secret"] = self.api_secret
            options["password"] = self.passphrase  # OKX uses "password" for passphrase
        
        self.exchange = ccxt_async.okx(options)
        
        if self.testnet:
            try:
                self.exchange.set_sandbox_mode(True)
                self.logger.info("OKX testnet mode enabled")
            except Exception as e:
                self.logger.warning(f"Could not enable testnet mode: {e}")
        
        attempts = EXCHANGE_RETRIES
        last_err = None
        for i in range(1, attempts + 1):
            try:
                await self.load_markets(reload=True)
                self.logger.info(f"OKX initialized: {len(self.markets)} markets loaded")
                return
            except Exception as e:
                last_err = e
                self.logger.warning(f"OKX market load attempt {i}/{attempts} failed: {type(e).__name__}: {str(e)}")
                if i < attempts:
                    await asyncio.sleep(RETRY_DELAY_MULTIPLIER * i)
        
        if last_err:
            self.logger.error(f"OKX initialization failed after {attempts} attempts: {last_err}")
            raise last_err
    
    async def load_markets(self, reload: bool = False, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Load markets from OKX."""
        if not self.exchange:
            raise Exception("Exchange not initialized")
        
        mkts = await self.exchange.load_markets(reload=reload)
        
        futures_markets = {}
        for sym, m in mkts.items():
            if "USDT" not in sym:
                continue
            market_type = m.get("type", "")
            if market_type != "swap":
                continue
            if m.get("expiry"):
                continue
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
            exchange_symbol = self.denormalize_symbol(symbol)
            return await self.exchange.fetch_ticker(exchange_symbol)
        except Exception as e:
            self.logger.warning(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    async def fetch_tickers(
        self,
        symbols: Optional[List[str]] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Fetch ticker data for multiple symbols."""
        if not self.exchange:
            return {}
        
        if not self.markets:
            await self.load_markets(reload=False)
        
        params = params or {}
        results: Dict[str, Any] = {}
        
        has_fetch_tickers = (
            self.exchange.has.get("fetch_tickers") is True or
            hasattr(self.exchange, "fetch_tickers")
        )
        
        if has_fetch_tickers:
            try:
                raw = await self.exchange.fetch_tickers(params=params)
                
                if raw and isinstance(raw, dict):
                    symbol_set = None
                    if symbols is not None:
                        symbol_set = set(symbols)
                    
                    for ex_sym, ticker in raw.items():
                        if not ticker or not isinstance(ticker, dict):
                            continue
                        
                        last = ticker.get("last")
                        if not last or last <= 0:
                            continue
                        
                        bid = ticker.get("bid") or last
                        ask = ticker.get("ask") or last
                        ticker["bid"] = bid
                        ticker["ask"] = ask
                        
                        norm = self.normalize_symbol(ex_sym)
                        
                        if symbol_set is not None and norm not in symbol_set:
                            continue
                        
                        results[norm] = ticker
                
                if results:
                    return results
            except Exception as e:
                self.logger.debug(f"fetch_tickers() batch failed: {e}, falling back")
        
        if symbols is None:
            if not self.markets:
                return results
            market_symbols = list(self.markets.keys())[:100]
            symbols = [self.normalize_symbol(s) for s in market_symbols]
        
        if not symbols:
            return results
        
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [self.fetch_ticker(sym) for sym in batch]
            results_batch = await asyncio.gather(*tasks, return_exceptions=True)
            
            for sym, ticker_or_exc in zip(batch, results_batch):
                if isinstance(ticker_or_exc, Exception):
                    continue
                ticker = ticker_or_exc
                if ticker and isinstance(ticker, dict):
                    results[sym] = ticker
            
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.05)
        
        return results
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100, params: Optional[Dict] = None) -> List[List]:
        """Fetch OHLCV data for a symbol."""
        if not self.exchange:
            return []
        try:
            if params is None:
                params = {}
            exchange_symbol = self.denormalize_symbol(symbol)
            return await self.exchange.fetch_ohlcv(exchange_symbol, timeframe, limit=limit, params=params)
        except Exception as e:
            self.logger.warning(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return []
    
    async def fetch_order_book(self, symbol: str, limit: int = 50, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Fetch orderbook for a symbol."""
        if not self.exchange:
            return None
        try:
            if params is None:
                params = {}
            exchange_symbol = self.denormalize_symbol(symbol)
            return await self.exchange.fetch_order_book(exchange_symbol, limit=limit, params=params)
        except Exception as e:
            self.logger.warning(f"Error fetching orderbook for {symbol}: {e}")
            return None
    
    async def fetch_balance(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Fetch account balance."""
        if not self.exchange:
            return {}
        try:
            if params is None:
                params = {}
            return await self.exchange.fetch_balance(params=params)
        except Exception as e:
            self.logger.warning(f"Error fetching balance: {e}")
            return {}
    
    async def fetch_positions(self, symbols: Optional[List[str]] = None, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Fetch open futures positions."""
        if not self.exchange:
            return []
        try:
            if params is None:
                params = {}
            
            exchange_symbols = None
            if symbols:
                exchange_symbols = [self.denormalize_symbol(s) for s in symbols]
            
            positions = await self.exchange.fetch_positions(exchange_symbols, params=params)
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
        
        if self.dry_run:
            import time
            exchange_symbol = self.denormalize_symbol(symbol)  # Only needed for dry run ID
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
            raise Exception("Exchange not initialized")
        
        try:
            if params is None:
                params = {}
            
            # Pass normalized symbol (e.g. BEAT/USDT:USDT)
            return await self.exchange.create_order(
                symbol, order_type, side, amount, price, params
            )
        except Exception as e:
            exchange_symbol = self.denormalize_symbol(symbol)  # For logging
            self.logger.error(f"Error creating order {exchange_symbol} {side} {amount}: {e}")
            raise
    
    async def set_leverage(self, leverage: int, symbol: str, params: Optional[Dict] = None):
        """Set leverage for a symbol."""
        # Pass normalized symbol to CCXT
        
        if self.dry_run:
            return
        if not self.exchange:
            raise Exception("Exchange not initialized")
        try:
            if params is None:
                params = {}
            
            # Pass normalized symbol (e.g. BEAT/USDT:USDT)
            await self.exchange.set_leverage(leverage, symbol, params=params)
        except Exception as e:
            exchange_symbol = self.denormalize_symbol(symbol)  # For logging
            self.logger.warning(f"Error setting leverage for {exchange_symbol}: {e}")
    
    async def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Fetch order status by order ID."""
        # Pass normalized symbol to CCXT
        
        if not self.exchange:
            return None
        try:
            if params is None:
                params = {}
            # Pass normalized symbol (e.g. BEAT/USDT:USDT)
            return await self.exchange.fetch_order(order_id, symbol, params=params)
        except Exception as e:
            exchange_symbol = self.denormalize_symbol(symbol)  # For logging
            self.logger.warning(f"Error fetching order {order_id} for {exchange_symbol}: {e}")
            return None
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for internal use."""
        if symbol in self._normalize_cache:
            return self._normalize_cache[symbol]
        
        clean = symbol.replace("/", "").replace("-", "").replace(":", "")
        
        if clean.endswith("USDT"):
            if clean.endswith("USDTUSDT"):
                base = clean[:-8]
            else:
                base = clean[:-4]
            result = f"{base}/USDT"
        elif "/" in symbol:
            result = symbol.upper()
        elif len(clean) <= 10:
            result = f"{clean}/USDT"
        else:
            result = symbol.upper()
        
        result = sys.intern(result)
        
        if len(self._normalize_cache) < self._cache_max_size:
            self._normalize_cache[symbol] = result
        elif len(self._normalize_cache) >= self._cache_max_size:
            self._normalize_cache.clear()
            self._normalize_cache[symbol] = result
        
        return result
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Denormalize symbol format for OKX API."""
        if symbol in self._denormalize_cache:
            return self._denormalize_cache[symbol]
        
        # OKX uses "BTC/USDT:USDT" format for perpetuals
        if "/" in symbol:
            parts = symbol.split("/")
            base = parts[0].upper()
            quote = parts[1].upper() if len(parts) > 1 else "USDT"
            if ":" in quote:
                quote = quote.split(":")[0]
            result = f"{base}/{quote}:{quote}"
        else:
            clean = symbol.replace("-", "").replace(":", "")
            if clean.endswith("USDT"):
                if clean.endswith("USDTUSDT"):
                    base = clean[:-8]
                else:
                    base = clean[:-4]
                result = f"{base}/USDT:USDT"
            else:
                result = f"{clean}/USDT:USDT"
        
        result = sys.intern(result)
        
        if len(self._denormalize_cache) < self._cache_max_size:
            self._denormalize_cache[symbol] = result
        elif len(self._denormalize_cache) >= self._cache_max_size:
            self._denormalize_cache.clear()
            self._denormalize_cache[symbol] = result
        
        return result
    
    def is_futures_market(self, market: Dict[str, Any]) -> bool:
        """Check if a market is a futures market."""
        market_type = market.get("type", "")
        return market_type == "swap"
    
    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            try:
                await self.exchange.close()
            except (ConnectionError, OSError, AttributeError, RuntimeError):
                # Ignore cleanup errors during shutdown - connection may already be closed
                pass

