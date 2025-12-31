from typing import Dict, Any, Optional, List

from .base import ExchangeBase
from ..api_governor import ApiGovernor


class GovernedExchange(ExchangeBase):
    """
    Wrapper that enforces the "single gateway" rule:
      Every network call goes through ApiGovernor (rate + inflight).

    This prevents:
      - CCXT throttle queue maxCapacity overflow
      - rate bursts beyond our budget
      - hangs without timeouts
    """

    def __init__(self, inner: ExchangeBase, governor: ApiGovernor):
        self._inner = inner
        self._gov = governor

        # Expose commonly used attributes for compatibility
        for name in ("logger", "exchange", "markets", "use_websocket", "ws_manager", "ws_ticker_cache"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    async def initialize(self):
        return await self._gov.call(cost=10, lane="universe", fn=lambda: self._inner.initialize(), timeout_sec=30.0)

    async def load_markets(self, reload: bool = False, params: Optional[Dict] = None) -> Dict[str, Any]:
        return await self._gov.call(cost=10, lane="universe", fn=lambda: self._inner.load_markets(reload=reload, params=params), timeout_sec=30.0)

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return await self._gov.call(cost=1, lane="scan", fn=lambda: self._inner.fetch_ticker(symbol))

    async def fetch_tickers(self, symbols: Optional[List[str]] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        # Batch tickers are heavy (weight=40 for all tickers)
        return await self._gov.call(cost=40, lane="universe", fn=lambda: self._inner.fetch_tickers(symbols=symbols, params=params), timeout_sec=15.0)

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100, params: Optional[Dict] = None) -> List[List]:
        return await self._gov.call(cost=1, lane="scan", fn=lambda: self._inner.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params))

    async def fetch_order_book(self, symbol: str, limit: int = 50, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        return await self._gov.call(cost=2, lane="scan", fn=lambda: self._inner.fetch_order_book(symbol, limit=limit, params=params))

    async def fetch_balance(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        return await self._gov.call(cost=5, lane="pos", fn=lambda: self._inner.fetch_balance(params=params))

    async def fetch_positions(self, symbols: Optional[List[str]] = None, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return await self._gov.call(cost=40, lane="pos", fn=lambda: self._inner.fetch_positions(symbols=symbols, params=params))

    async def fetch_orders(self, symbol: Optional[str] = None, since: Optional[int] = None, 
                          limit: Optional[int] = None, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        # Order history is moderate weight (5 per request)
        return await self._gov.call(cost=5, lane="pos", fn=lambda: self._inner.fetch_orders(symbol=symbol, since=since, limit=limit, params=params))

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        p = dict(params or {})
        lane_override = p.pop("__lane", None)
        cost_override = p.pop("__cost", None)

        # Infer exits using reduceOnly (Binance), otherwise treat as entry
        reduce_only = bool(p.get("reduceOnly") or p.get("closePosition"))
        lane = str(lane_override) if lane_override else ("exit" if reduce_only else "entry")
        cost = int(cost_override) if cost_override is not None else 1

        return await self._gov.call(
            cost=cost,
            lane=lane,
            fn=lambda: self._inner.create_order(symbol, type, side, amount, price, p),
            timeout_sec=10.0,  # orders can take longer
        )

    async def set_leverage(self, leverage: int, symbol: str, params: Optional[Dict] = None):
        return await self._gov.call(cost=1, lane="entry", fn=lambda: self._inner.set_leverage(leverage, symbol, params=params))

    async def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        return await self._gov.call(cost=1, lane="pos", fn=lambda: self._inner.fetch_order(order_id, symbol, params=params))

    def normalize_symbol(self, symbol: str) -> str:
        return self._inner.normalize_symbol(symbol)

    def denormalize_symbol(self, symbol: str) -> str:
        return self._inner.denormalize_symbol(symbol)

    def is_futures_market(self, market: Dict[str, Any]) -> bool:
        return self._inner.is_futures_market(market)

    async def close(self):
        # Closing should be allowed even under pressure; treat as "pos" lane.
        return await self._gov.call(cost=1, lane="pos", fn=lambda: self._inner.close())

    async def place_trailing_stop(self, symbol: str, side: str, quantity: float, 
                                   callback_rate: float = None, activation_price: float = None,
                                   current_price: float = None, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Place a trailing stop order."""
        return await self._gov.call(
            cost=1, 
            lane="exit", 
            fn=lambda: self._inner.place_trailing_stop(
                symbol=symbol, side=side, amount=quantity, 
                callback_rate=callback_rate, activation_price=activation_price,
                current_price=current_price
            ),
            timeout_sec=10.0
        )
    
    async def place_stop_loss_order(self, symbol: str, side: str, quantity: float,
                                     stop_price: float, current_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Place a regular STOP_MARKET order (stop-loss)."""
        return await self._gov.call(
            cost=1,
            lane="exit",
            fn=lambda: self._inner.place_stop_loss_order(
                symbol=symbol, side=side, quantity=quantity,
                stop_price=stop_price, current_price=current_price
            ),
            timeout_sec=10.0
        )
    
    async def fetch_stop_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all active stop-loss orders."""
        return await self._gov.call(
            cost=1,
            lane="exit",
            fn=lambda: self._inner.fetch_stop_orders(symbol=symbol),
            timeout_sec=5.0
        )

    async def cancel_trailing_stop(self, symbol: str) -> bool:
        """Cancel a trailing stop order."""
        return await self._gov.call(
            cost=1,
            lane="exit",
            fn=lambda: self._inner.cancel_trailing_stop(symbol)
        )


