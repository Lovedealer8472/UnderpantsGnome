"""
Exchange Base Interface - Abstract base class for all exchange implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class ExchangeBase(ABC):
    """Abstract base class for all exchange implementations."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize exchange connection. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def load_markets(self, reload: bool = False, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load markets from exchange.
        
        Args:
            reload: If True, reload markets from exchange
            params: Additional parameters for market loading
            
        Returns:
            Dictionary of markets (symbol -> market info)
        """
        pass
    
    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ticker data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker data dictionary or None if error
        """
        pass
    
    @abstractmethod
    async def fetch_tickers(self, symbols: Optional[List[str]] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fetch ticker data for multiple symbols.
        
        Args:
            symbols: List of symbols (optional, fetch all if None)
            params: Additional parameters
            
        Returns:
            Dictionary of tickers (symbol -> ticker data)
        """
        pass
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100, params: Optional[Dict] = None) -> List[List]:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "4h", "1d")
            limit: Number of candles to fetch
            params: Additional parameters
            
        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        pass
    
    @abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 50, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch orderbook for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of orders to fetch
            params: Additional parameters
            
        Returns:
            Orderbook data dictionary or None if error
        """
        pass
    
    @abstractmethod
    async def fetch_balance(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Args:
            params: Additional parameters
            
        Returns:
            Balance data dictionary
        """
        pass
    
    @abstractmethod
    async def fetch_positions(self, symbols: Optional[List[str]] = None, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Fetch open positions.
        
        Args:
            symbols: List of symbols (optional, fetch all if None)
            params: Additional parameters
            
        Returns:
            List of position dictionaries
        """
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, order_type: str, side: str,
                          amount: float, price: Optional[float] = None,
                          params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create an order.
        
        Args:
            symbol: Trading symbol
            order_type: "market" or "limit"
            side: "buy" or "sell"
            amount: Order amount
            price: Limit price (required for limit orders)
            params: Additional parameters (e.g., positionSide, leverage)
            
        Returns:
            Order result dictionary
        """
        pass
    
    @abstractmethod
    async def set_leverage(self, leverage: int, symbol: str, params: Optional[Dict] = None):
        """
        Set leverage for a symbol.
        
        Args:
            leverage: Leverage value
            symbol: Trading symbol
            params: Additional parameters
        """
        pass
    
    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch order status by order ID.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            params: Additional parameters
            
        Returns:
            Order data dictionary or None if error
        """
        pass
    
    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for internal use.
        
        Examples:
            "BTC/USDT:USDT" -> "BTC/USDT"
            "BTCUSDT" -> "BTC/USDT"
            
        Args:
            symbol: Exchange-specific symbol format
            
        Returns:
            Normalized symbol format
        """
        pass
    
    @abstractmethod
    def denormalize_symbol(self, symbol: str) -> str:
        """
        Denormalize symbol format for exchange API.
        
        Examples:
            "BTC/USDT" -> "BTC/USDT" (for Binance Futures)
            
        Args:
            symbol: Normalized symbol format
            
        Returns:
            Exchange-specific symbol format
        """
        pass
    
    @abstractmethod
    def is_futures_market(self, market: Dict[str, Any]) -> bool:
        """
        Check if a market is a futures market.
        
        Args:
            market: Market dictionary from exchange
            
        Returns:
            True if futures market, False otherwise
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Close exchange connection."""
        pass

