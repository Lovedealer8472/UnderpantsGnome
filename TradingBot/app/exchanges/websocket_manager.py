"""
WebSocket Manager for Binance Futures - Real-time ticker data via WebSocket.
Eliminates rate limit issues by using persistent connections instead of REST API polling.
"""
import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, Callable, Set
from collections import defaultdict
import websockets
from ..logger import get_logger


class BinanceWebSocketManager:
    """
    Manages WebSocket connections to Binance Futures for real-time ticker data.
    
    Features:
    - Automatic reconnection on disconnect
    - Ticker stream subscription
    - Real-time cache updates
    - Error handling and recovery
    """
    
    # Binance Futures WebSocket endpoints
    WS_BASE_URL = "wss://fstream.binance.com/ws"  # Single stream
    WS_STREAM_URL = "wss://fstream.binance.com/stream"  # Combined streams
    
    def __init__(self, logger=None):
        """
        Initialize WebSocket manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("BinanceWebSocket")
        self.ws = None
        self.ws_task = None
        self.connected = False
        self.subscribed_symbols: Set[str] = []  # Symbols we're subscribed to
        self.ticker_callbacks: Dict[str, Callable] = {}  # Callbacks for ticker updates
        self.reconnect_delay = 5.0  # Seconds to wait before reconnecting
        self.max_reconnect_delay = 60.0  # Maximum reconnect delay
        self.reconnect_attempts = 0
        self.last_message_time = 0.0
        self.message_count = 0
        self.error_count = 0
        self._stop_event = asyncio.Event()
        
    async def connect(self, symbols: Optional[list] = None):
        """
        Connect to Binance WebSocket and subscribe to ticker streams.
        
        Args:
            symbols: List of symbols to subscribe to (None = all USDT futures)
        """
        if self.connected:
            self.logger.warning("WebSocket already connected")
            return
        
        if symbols:
            self.subscribed_symbols = set(symbols)
        else:
            # Subscribe to all USDT futures (will be filtered by Binance)
            self.subscribed_symbols = set()
        
        self._stop_event.clear()
        self.ws_task = asyncio.create_task(self._run_websocket())
        
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self.logger.info("Disconnecting WebSocket...")
        self._stop_event.set()
        self.connected = False
        
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                self.logger.debug(f"Error closing WebSocket: {e}")
            self.ws = None
        
        if self.ws_task:
            try:
                self.ws_task.cancel()
                await self.ws_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.debug(f"Error cancelling WebSocket task: {e}")
            self.ws_task = None
        
        self.logger.info("WebSocket disconnected")
    
    async def _run_websocket(self):
        """Main WebSocket connection loop with automatic reconnection."""
        while not self._stop_event.is_set():
            try:
                await self._connect_and_subscribe()
                await self._listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"WebSocket error: {type(e).__name__}: {e}")
                
                if not self._stop_event.is_set():
                    # Wait before reconnecting (exponential backoff)
                    delay = min(
                        self.reconnect_delay * (2 ** min(self.reconnect_attempts, 5)),
                        self.max_reconnect_delay
                    )
                    self.logger.warning(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempts + 1})...")
                    await asyncio.sleep(delay)
                    self.reconnect_attempts += 1
                else:
                    break
    
    async def _connect_and_subscribe(self):
        """Connect to WebSocket and subscribe to ticker streams."""
        # Build stream names for subscription
        if self.subscribed_symbols:
            # Subscribe to specific symbols
            # Convert normalized symbols (BTC/USDT:USDT) to Binance format (btcusdt)
            binance_symbols = []
            for symbol in self.subscribed_symbols:
                # Remove / and :USDT suffix
                clean = symbol.replace("/", "").replace(":USDT", "").lower()
                binance_symbols.append(f"{clean}@ticker")
            streams = binance_symbols
        else:
            # Subscribe to all ticker stream (Binance supports this)
            streams = ["!ticker@arr"]  # All tickers array stream
        
        # Use combined stream endpoint for multiple streams
        if len(streams) == 1 and streams[0] == "!ticker@arr":
            # Single stream for all tickers
            stream_name = streams[0]
            ws_url = f"{self.WS_BASE_URL}/{stream_name}"
        else:
            # Multiple streams - use stream endpoint
            stream_params = "/".join(streams)
            ws_url = f"{self.WS_STREAM_URL}?streams={stream_params}"
        
        self.logger.info(f"Connecting to Binance WebSocket: {len(streams)} stream(s)")
        
        try:
            self.ws = await websockets.connect(
                ws_url,
                ping_interval=30,  # Ping every 30 seconds to keep connection alive
                ping_timeout=30,   # Allow 30 seconds for pong response (more tolerant of network latency)
                close_timeout=10
            )
            self.connected = True
            self.reconnect_attempts = 0
            self.last_message_time = time.time()
            self.logger.info("[OK] WebSocket connected and subscribed")
        except Exception as e:
            self.connected = False
            raise Exception(f"Failed to connect WebSocket: {e}")
    
    async def _listen(self):
        """Listen for WebSocket messages and process them."""
        try:
            async for message in self.ws:
                if self._stop_event.is_set():
                    break
                
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                    self.last_message_time = time.time()
                    self.message_count += 1
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    self.logger.warning(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            self.logger.warning("WebSocket connection closed")
            raise
        except Exception as e:
            self.connected = False
            self.logger.error(f"WebSocket listen error: {type(e).__name__}: {e}")
            raise
    
    async def _process_message(self, data: Dict[str, Any]):
        """
        Process incoming WebSocket message.
        
        Handles both single ticker updates and array of tickers.
        """
        # Check if this is an array of tickers (all tickers stream)
        if isinstance(data, list):
            # Array of tickers from !ticker@arr stream
            for ticker_data in data:
                await self._process_ticker(ticker_data)
        elif isinstance(data, dict):
            # Single ticker update
            if "data" in data:
                # Wrapped in 'data' field (combined stream format)
                await self._process_ticker(data["data"])
            elif "e" in data and data.get("e") == "24hrTicker":
                # Direct ticker update
                await self._process_ticker(data)
            else:
                # Unknown message format, log for debugging
                self.logger.debug(f"Unknown WebSocket message format: {list(data.keys())}")
    
    async def _process_ticker(self, ticker_data: Dict[str, Any]):
        """
        Process a single ticker update.
        
        Args:
            ticker_data: Ticker data from Binance WebSocket
        """
        try:
            # Extract symbol from ticker data
            symbol = ticker_data.get("s")  # Symbol field
            if not symbol:
                return
            
            # Normalize symbol format (e.g., "BTCUSDT" -> "BTC/USDT:USDT")
            normalized = self._normalize_symbol(symbol)
            
            # Convert Binance ticker format to our format
            ticker = {
                "symbol": normalized,
                "bid": float(ticker_data.get("b", 0) or 0),  # Best bid price
                "ask": float(ticker_data.get("a", 0) or 0),  # Best ask price
                "last": float(ticker_data.get("c", 0) or 0),  # Last price
                "mark": float(ticker_data.get("p", 0) or 0),  # Mark price (if available)
                "volume": float(ticker_data.get("v", 0) or 0),  # 24h volume
                "quoteVolume": float(ticker_data.get("q", 0) or 0),  # 24h quote volume
                "percentage": float(ticker_data.get("P", 0) or 0),  # 24h price change %
                "timestamp": int(ticker_data.get("E", 0) or time.time() * 1000),  # Event time
                "info": ticker_data  # Keep raw data
            }
            
            # If mark price not in main fields, try to get from info
            if ticker["mark"] == 0:
                ticker["mark"] = ticker["last"]
            
            # Call registered callbacks
            # Check for specific symbol callback
            if normalized in self.ticker_callbacks:
                try:
                    await self.ticker_callbacks[normalized](ticker)
                except Exception as e:
                    self.logger.warning(f"Error in ticker callback for {normalized}: {e}")
            
            # Also call "ALL" callback if registered (for all tickers stream)
            if "ALL" in self.ticker_callbacks:
                try:
                    await self.ticker_callbacks["ALL"](ticker)
                except Exception as e:
                    self.logger.warning(f"Error in ALL ticker callback: {e}")
            
        except Exception as e:
            self.logger.warning(f"Error processing ticker: {e}")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize Binance symbol format to internal format.
        
        Args:
            symbol: Binance symbol (e.g., "BTCUSDT")
            
        Returns:
            Normalized symbol (e.g., "BTC/USDT:USDT")
        """
        # Binance futures symbols are like "BTCUSDT"
        # We need "BTC/USDT:USDT" format
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT:USDT"
        return symbol
    
    def register_ticker_callback(self, symbol: str, callback: Callable):
        """
        Register a callback for ticker updates.
        
        Args:
            symbol: Symbol to watch (normalized format)
            callback: Async function that receives ticker dict
        """
        self.ticker_callbacks[symbol] = callback
    
    def unregister_ticker_callback(self, symbol: str):
        """Unregister ticker callback for a symbol."""
        if symbol in self.ticker_callbacks:
            del self.ticker_callbacks[symbol]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        time_since_last = time.time() - self.last_message_time if self.last_message_time > 0 else 0
        return {
            "connected": self.connected,
            "subscribed_symbols": len(self.subscribed_symbols),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "reconnect_attempts": self.reconnect_attempts,
            "seconds_since_last_message": time_since_last,
            "callbacks_registered": len(self.ticker_callbacks)
        }

