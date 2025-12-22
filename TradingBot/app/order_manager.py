"""
Order Manager - Handles order execution with smart order routing and timing.
"""

import time
import asyncio
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .config import (
    ENTRY_DELAY_MS, TAKER_FEE_RATE,
    LEVERAGE_BASE, MAX_LATENCY_MS
)
# CRITICAL: Don't import DRY_RUN at module level - it's set dynamically in GO_LIVE.py
# Import config module to check DRY_RUN at runtime
import app.config as config_module


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_size: Optional[float] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class OrderManager:
    """Manages order execution with smart routing."""
    
    def __init__(self, exchange=None):
        self.exchange = exchange
        self.order_history = []
        self.leverage_cache = {}  # Cache for leverage settings
    
    async def enter_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        delay_ms: Optional[int] = None,
        use_limit: bool = False,
        leverage: Optional[int] = None
    ) -> OrderResult:
        """
        Enter a position with smart execution.
        OPTIMIZED: Parallel operations and faster execution paths.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size in base currency
            entry_price: Target entry price
            delay_ms: Entry delay in milliseconds
            use_limit: Use limit order instead of market
            leverage: Leverage to use
        
        Returns:
            OrderResult
        """
        if delay_ms is None:
            delay_ms = ENTRY_DELAY_MS
        
        start_time = time.time()
        
        # OPTIMIZATION: Parallelize delay and leverage setting
        # Start leverage setting immediately (non-blocking if already set)
        leverage_task = None
        if leverage and self.exchange and not config_module.DRY_RUN:
            leverage_task = asyncio.create_task(
                self._set_leverage_async(leverage, symbol)
            )
        
        # OPTIMIZATION: Apply delay concurrently with leverage setting
        # This allows leverage to be set while we wait
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        
        # OPTIMIZATION: Wait for leverage to complete (should be done by now)
        if leverage_task:
            try:
                await asyncio.wait_for(leverage_task, timeout=0.1)  # 100ms timeout
            except asyncio.TimeoutError:
                pass  # Continue even if leverage setting is slow (may already be set)
            except (AttributeError, RuntimeError, KeyError) as e:
                # REFACTOR: Handle leverage setting errors (may already be set)
                self.logger.debug(f"Leverage setting skipped for {symbol}: {e}")
        
        try:
            # CRITICAL: Check DRY_RUN dynamically (not at import time)
            # DIAGNOSTIC: Log DRY_RUN status for debugging
            try:
                from .logger import get_logger
                logger = get_logger("OrderManager")
                logger.warning(f"[ORDER_DRY_RUN_CHECK] {symbol} {side}: DRY_RUN = {config_module.DRY_RUN} (should be False for live trading)")
            except Exception:
                pass
            
            if config_module.DRY_RUN:
                # Simulate order execution
                try:
                    from .logger import get_logger
                    logger = get_logger("OrderManager")
                    logger.warning(f"[ORDER_DRY_RUN_MODE] {symbol} {side}: Simulating order (DRY_RUN=True) - NO REAL ORDER PLACED")
                except Exception:
                    pass
                result = OrderResult(
                    success=True,
                    order_id=f"DRY_{int(time.time() * 1000)}",
                    filled_price=entry_price,
                    filled_size=size,
                    latency_ms=(time.time() - start_time) * 1000
                )
            else:
                # OPTIMIZATION: Use faster execution path for market orders
                if use_limit:
                    result = await self._place_limit_order(
                        symbol, side, size, entry_price
                    )
                else:
                    # OPTIMIZATION: Market orders are faster - use optimized path
                    result = await self._place_market_order_fast(
                        symbol, side, size
                    )
                
                result.latency_ms = (time.time() - start_time) * 1000
            
            # Record order (non-blocking)
            self.order_history.append({
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': result.filled_price or entry_price,
                'timestamp': time.time(),
                'success': result.success
            })
            
            return result
            
        except Exception as e:
            # Log error with context
            try:
                from .logger import get_logger
                get_logger().log_error_with_context(
                    operation="enter_position",
                    error=e,
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price
                )
            except ImportError:
                pass  # Logger not available
            
            return OrderResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _set_leverage_async(self, leverage: int, symbol: str):
        """
        Set leverage asynchronously (non-blocking).
        OPTIMIZATION: Cache leverage settings to avoid redundant API calls.
        """
        # OPTIMIZATION: Check cache first
        cache_key = f"{symbol}:{leverage}"
        if cache_key in self.leverage_cache:
            return  # Already set, skip API call
        
        try:
            await self.exchange.set_leverage(leverage, symbol)
            # Cache successful setting
            self.leverage_cache[cache_key] = time.time()
        except (AttributeError, RuntimeError, KeyError):
            # REFACTOR: Leverage may already be set or not supported
            pass  # Non-critical, continue
    
    def pre_set_leverage(self, symbols: list, leverage: int):
        """
        Pre-set leverage for multiple symbols (background task).
        OPTIMIZATION: Set leverage in advance to reduce entry latency.
        
        Args:
            symbols: List of symbols to set leverage for
            leverage: Leverage to set
        """
        if not self.exchange or config_module.DRY_RUN:
            return
        
        async def _pre_set():
            for symbol in symbols:
                cache_key = f"{symbol}:{leverage}"
                if cache_key not in self.leverage_cache:
                    try:
                        await self.exchange.set_leverage(leverage, symbol)
                        self.leverage_cache[cache_key] = time.time()
                    except (AttributeError, RuntimeError, KeyError):
                        # REFACTOR: Silently skip symbols where leverage setting fails
                        pass  # Non-critical background task
        
        # Run in background (fire and forget)
        asyncio.create_task(_pre_set())
    
    async def _place_market_order_fast(
        self,
        symbol: str,
        side: str,
        size: float
    ) -> OrderResult:
        """
        OPTIMIZED: Fast market order placement with minimal overhead.
        Uses direct API call without extra validation steps.
        """
        try:
            order_type = "market"
            order_side = "buy" if side == "long" else "sell"
            
            # OPTIMIZATION: Direct order creation without extra checks
            # OPTIMIZATION: Binance Futures USDT-M doesn't use positionSide (only COIN-M does)
            # The exchange wrapper will handle this correctly
            # CRITICAL: Log BEFORE attempting to place order
            try:
                from .logger import get_logger
                logger = get_logger("OrderManager")
                logger.warning(
                    f"[ORDER_ATTEMPT] {symbol} {order_side} size={size:.6f}: "
                    f"About to call exchange.create_order()"
                )
            except Exception:
                pass
            
            try:
                # CRITICAL: This is where the order is actually sent to Binance
                # Check if exchange is available (handle both direct exchange and wrapped exchanges)
                if not self.exchange:
                    error_msg = "Exchange not initialized - cannot place order"
                    try:
                        from .logger import get_logger
                        logger = get_logger("OrderManager")
                        logger.error(f"[ORDER_NO_EXCHANGE] {symbol}: {error_msg}")
                    except Exception:
                        pass
                    return OrderResult(
                        success=False,
                        error=error_msg
                    )
                
                # Check if wrapped exchange has inner exchange (GovernedExchange pattern)
                inner_exchange = None
                if hasattr(self.exchange, '_inner'):
                    inner_exchange = self.exchange._inner
                elif hasattr(self.exchange, 'exchange'):
                    inner_exchange = self.exchange.exchange
                
                if not inner_exchange and not hasattr(self.exchange, 'create_order'):
                    error_msg = "Exchange wrapper has no inner exchange and no create_order method"
                    try:
                        from .logger import get_logger
                        logger = get_logger("OrderManager")
                        logger.error(f"[ORDER_NO_INNER_EXCHANGE] {symbol}: {error_msg}")
                    except Exception:
                        pass
                    return OrderResult(
                        success=False,
                        error=error_msg
                    )
                
                order = await self.exchange.create_order(
                    symbol,
                    order_type,
                    order_side,
                    size,
                    None,  # price not needed for market orders
                    params={}  # Let exchange wrapper handle Binance-specific params
                )
                
                # CRITICAL: Log order response immediately after receiving it
                try:
                    from .logger import get_logger
                    logger = get_logger("OrderManager")
                    logger.warning(  # Use WARNING level so it's visible
                        f"[ORDER_RECEIVED] {symbol} {order_side}: "
                        f"id={order.get('id')}, status={order.get('status')}, "
                        f"filled={order.get('filled')}, amount={order.get('amount')}, "
                        f"remaining={order.get('remaining')}, price={order.get('price')}, "
                        f"average={order.get('average')}, info={order.get('info', {})}"
                    )
                except Exception:
                    pass  # Logging failure shouldn't break order placement
            except Exception as e:
                # CRITICAL: Check for Binance -4140 error (Invalid symbol status) BEFORE logging
                error_str = str(e)
                is_invalid_symbol = "-4140" in error_str or "Invalid symbol status" in error_str
                
                # For invalid symbol errors, return early with specific error code (don't log as ERROR)
                if is_invalid_symbol:
                    try:
                        from .logger import get_logger
                        logger = get_logger("OrderManager")
                        logger.debug(
                            f"[INVALID_SYMBOL] {symbol} {order_side}: Symbol in Reduce Only mode or delisted (-4140). "
                            f"Skipping order placement."
                        )
                    except Exception:
                        pass
                    return OrderResult(
                        success=False,
                        error=f"INVALID_SYMBOL_STATUS:-4140:{error_str}"
                    )
                
                # CRITICAL: Log the actual exception from create_order (for non-invalid-symbol errors)
                try:
                    from .logger import get_logger
                    logger = get_logger("OrderManager")
                    logger.error(
                        f"[ORDER_EXCEPTION] {symbol} {order_side} size={size:.6f}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    import traceback
                    logger.error(f"[ORDER_EXCEPTION_TRACEBACK] {traceback.format_exc()}")
                except Exception:
                    pass
                # Return error result instead of raising (allows bot.py to handle gracefully)
                return OrderResult(
                    success=False,
                    error=f"{type(e).__name__}: {error_str}"
                )
            
            # CRITICAL: Check if order was rejected by exchange
            order_status = order.get('status', '').lower()
            order_id = order.get('id')
            
            # DIAGNOSTIC: Log full order response for debugging
            try:
                from .logger import get_logger
                logger = get_logger("OrderManager")
                logger.warning(  # Changed to WARNING for visibility
                    f"[ORDER_RESPONSE_FULL] {symbol} {order_side}: "
                    f"id={order_id}, status={order_status}, "
                    f"filled={order.get('filled')}, amount={order.get('amount')}, "
                    f"remaining={order.get('remaining')}, price={order.get('price')}, "
                    f"average={order.get('average')}, info={order.get('info', {})}"
                )
            except Exception:
                pass
            
            if order_status == 'rejected' or order_id is None:
                error_msg = order.get('info', {}).get('msg', 'Order rejected by exchange')
                # Log rejection details
                try:
                    from .logger import get_logger
                    logger = get_logger("OrderManager")
                    logger.error(
                        f"[ORDER_REJECTED] {symbol} {order_side} size={size}: "
                        f"status={order_status}, id={order_id}, error={error_msg}, "
                        f"info={order.get('info', {})}"
                    )
                except Exception:
                    pass
                return OrderResult(
                    success=False,
                    error=error_msg
                )
            
            # CRITICAL FIX: Only check 'filled' field, NOT 'amount' (amount is order size, not filled size)
            # For Binance market orders, 'filled' will be 0 or None if not yet filled
            # 'amount' is the ORDER size, not the FILLED size - using it causes false positives
            filled_size = order.get('filled')  # Only use 'filled', not 'amount'
            filled_price = order.get('price') or order.get('average') or order.get('info', {}).get('price')
            
            # IMPROVED: Check multiple fields for filled size (Binance can return it in different places)
            # Some exchanges return filled in 'info' dict
            if not filled_size or float(filled_size or 0) == 0:
                # Try alternative fields
                filled_size = order.get('info', {}).get('executedQty') or order.get('info', {}).get('cumQty') or filled_size
            
            # Check order status - Binance market orders should be "closed" when filled
            # Status can be: "new", "open", "closed", "canceled", "rejected", "FILLED" (Binance sometimes uses uppercase)
            order_status_normalized = order_status.upper()
            if (order_status == 'closed' or order_status_normalized == 'FILLED') and filled_size and float(filled_size) > 0:
                # Order is closed/filled and has filled size - verify it's fully filled
                filled_size_float = float(filled_size)
                fill_ratio = filled_size_float / size if size > 0 else 0
                if fill_ratio >= 0.99:  # 99% or more = fully filled
                    # Order was fully filled - return success
                    return OrderResult(
                        success=True,
                        order_id=order.get('id', ''),
                        filled_price=float(filled_price) if filled_price else None,
                        filled_size=filled_size_float
                    )
                else:
                    # Partial fill - reject it (all-or-nothing policy)
                    return OrderResult(
                        success=False,
                        error=f"Partial fill rejected: {filled_size_float:.6f}/{size:.6f} ({fill_ratio*100:.1f}% filled). All-or-nothing policy."
                    )
            elif order_status in ('new', 'open'):
                # Order is placed but not yet filled - need to poll for status
                # This is normal for market orders that take a moment to fill
                # Continue to fetch_order check below
                pass
            elif filled_size and float(filled_size) > 0:
                # Order has filled size but status is not 'closed' - might be in transition
                # Check fill ratio anyway (some exchanges return filled before status updates)
                filled_size_float = float(filled_size)
                fill_ratio = filled_size_float / size if size > 0 else 0
                if fill_ratio >= 0.99:
                    try:
                        from .logger import get_logger
                        logger = get_logger("OrderManager")
                        logger.warning(
                            f"[ORDER_FILLED_BY_SIZE] {symbol}: Status={order_status} but filled={filled_size_float:.6f}, "
                            f"treating as filled (fill_ratio={fill_ratio*100:.1f}%)"
                        )
                    except Exception:
                        pass
                    return OrderResult(
                        success=True,
                        order_id=order.get('id', ''),
                        filled_price=float(filled_price) if filled_price else None,
                        filled_size=filled_size_float
                    )
            
            # CRITICAL: If order has no ID at this point, it was rejected
            # This check must come AFTER we've tried to use the order_id
            if not order_id:
                try:
                    from .logger import get_logger
                    logger = get_logger("OrderManager")
                    logger.error(
                        f"[ORDER_NO_ID] {symbol} {order_side} size={size}: "
                        f"Order response has no ID - likely rejected. "
                        f"status={order_status}, info={order.get('info', {})}"
                    )
                except Exception:
                    pass
                return OrderResult(
                    success=False,
                    error="Order rejected: No order ID returned from exchange"
                )
            
            # CRITICAL: For market orders, we MUST fetch order status to verify fill
            # Binance market orders can fill instantly, but the initial response might not show it
            # Always fetch order status to get accurate fill information
            try:
                filled_order = await asyncio.wait_for(
                    self.exchange.fetch_order(order_id, symbol),
                    timeout=2.0  # 2 second timeout (increased from 500ms)
                )
                filled_price = filled_order.get('average') or filled_order.get('price') or filled_price
                # CRITICAL FIX: Only use 'filled' field, NOT 'amount' (amount is order size, not filled size)
                filled_size = filled_order.get('filled') or 0  # Only check 'filled', not 'amount'
                order_status = filled_order.get('status', '').lower()
                
                # Check if order was rejected or cancelled
                if order_status in ('rejected', 'canceled', 'cancelled'):
                    return OrderResult(
                        success=False,
                        error=f"Order {order_status} by exchange"
                    )
                
                # CRITICAL: Check if order is filled - accept "closed" status OR if filled_size > 0
                # Binance market orders can fill instantly but status might not be "closed" immediately
                filled_size_float = float(filled_size) if filled_size else 0.0
                
                if filled_size_float > 0:
                    # Order has filled size - check if it's fully filled
                    fill_ratio = filled_size_float / size if size > 0 else 0
                    if fill_ratio >= 0.99:  # 99% or more = fully filled
                        # Order was fully filled - return success regardless of status
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            filled_price=float(filled_price) if filled_price else None,
                            filled_size=filled_size_float
                        )
                    elif fill_ratio > 0:
                        # Partial fill - reject it (all-or-nothing policy)
                        return OrderResult(
                            success=False,
                            error=f"Partial fill rejected: {filled_size_float:.6f}/{size:.6f} ({fill_ratio*100:.1f}% filled). All-or-nothing policy."
                        )
                
                # If order status is "closed" but no filled_size, it might be a different issue
                if order_status == 'closed' and filled_size_float == 0:
                    # Order is closed but has no filled size - might be rejected or cancelled
                    return OrderResult(
                        success=False,
                        error="Order closed but not filled (likely rejected or cancelled)"
                    )
                
                if order_status in ('new', 'open'):
                    # Order still open - might need more time or might be rejected
                    # Wait a bit more and check again (one more attempt)
                    await asyncio.sleep(0.3)  # Wait 300ms
                    try:
                        final_check = await asyncio.wait_for(
                            self.exchange.fetch_order(order_id, symbol),
                            timeout=1.0
                        )
                        final_status = final_check.get('status', '').lower()
                        final_filled = final_check.get('filled') or 0
                        if final_status == 'closed' and final_filled and float(final_filled) > 0:
                            final_filled_float = float(final_filled)
                            final_ratio = final_filled_float / size if size > 0 else 0
                            if final_ratio >= 0.99:
                                return OrderResult(
                                    success=True,
                                    order_id=order_id,
                                    filled_price=final_check.get('average') or final_check.get('price') or filled_price,
                                    filled_size=final_filled_float
                                )
                    except Exception:
                        pass  # If second check fails, continue to error below
            except asyncio.TimeoutError:
                    # Timeout fetching order - might still be filling, but we can't verify
                    # For market orders, try one more quick check after a short delay
                    try:
                        from .logger import get_logger
                        logger = get_logger("OrderManager")
                        logger.warning(f"[ORDER_TIMEOUT] {symbol} order_id={order_id}: First fetch timed out, retrying...")
                    except Exception:
                        pass
                    
                    # Give it one more chance with a shorter timeout
                    try:
                        await asyncio.sleep(0.2)  # Brief delay
                        quick_check = await asyncio.wait_for(
                            self.exchange.fetch_order(order_id, symbol),
                            timeout=0.5  # Quick 500ms check
                        )
                        quick_filled = quick_check.get('filled') or 0
                        quick_status = quick_check.get('status', '').lower()
                        if quick_filled and float(quick_filled) > 0:
                            quick_filled_float = float(quick_filled)
                            quick_ratio = quick_filled_float / size if size > 0 else 0
                            if quick_ratio >= 0.99:
                                return OrderResult(
                                    success=True,
                                    order_id=order_id,
                                    filled_price=quick_check.get('average') or quick_check.get('price'),
                                    filled_size=quick_filled_float
                                )
                    except Exception:
                        pass  # If retry also fails, try position check fallback
                    
                    # FALLBACK: If verification fails, check exchange positions directly
                    # Market orders on Binance fill instantly - if order_id exists, check positions
                    if order_id:
                        try:
                            from .logger import get_logger
                            logger = get_logger("OrderManager")
                            logger.warning(f"[ORDER_VERIFY_FALLBACK] {symbol} order_id={order_id}: Verification failed, checking exchange positions...")
                        except Exception:
                            pass
                        
                        try:
                            # Check if position exists on exchange (order might have filled despite verification failure)
                            await asyncio.sleep(0.5)  # Brief delay for position to appear
                            positions = await self.exchange.fetch_positions([symbol])
                            for pos in positions:
                                pos_symbol = pos.get('symbol', '')
                                # Normalize for comparison
                                if hasattr(self.exchange, 'normalize_symbol'):
                                    pos_symbol = self.exchange.normalize_symbol(pos_symbol)
                                
                                if pos_symbol == symbol or pos_symbol.replace('/USDT:USDT', '') == symbol.replace('/USDT:USDT', ''):
                                    pos_size = abs(float(pos.get('contracts', 0) or pos.get('positionAmt', 0) or 0))
                                    if pos_size > 0:
                                        # Position exists! Order actually filled
                                        entry_price = float(pos.get('entryPrice', 0) or pos.get('markPrice', 0) or 0)
                                        try:
                                            from .logger import get_logger
                                            logger = get_logger("OrderManager")
                                            logger.warning(
                                                f"[ORDER_RECOVERED] {symbol}: Position found on exchange despite verification failure! "
                                                f"Size={pos_size:.6f}, Entry=${entry_price:.4f}"
                                            )
                                        except Exception:
                                            pass
                                        return OrderResult(
                                            success=True,
                                            order_id=order_id or f"RECOVERED_{int(time.time())}",
                                            filled_price=entry_price,
                                            filled_size=pos_size
                                        )
                        except Exception as e:
                            try:
                                from .logger import get_logger
                                logger = get_logger("OrderManager")
                                logger.debug(f"[ORDER_FALLBACK_ERROR] {symbol}: Position check failed: {e}")
                            except Exception:
                                pass
                    
                    # If we still can't verify, return failure
                    try:
                        from .logger import get_logger
                        logger = get_logger("OrderManager")
                        logger.error(f"[ORDER_TIMEOUT_FINAL] {symbol} order_id={order_id}: Could not verify order fill after retry and position check")
                    except Exception:
                        pass
                    return OrderResult(
                        success=False,
                        error=f"Could not verify order fill: timeout fetching order status (order_id={order_id})"
                    )
            except Exception as e:
                    # Other error fetching order status
                    try:
                        from .logger import get_logger
                        logger = get_logger("OrderManager")
                        logger.error(f"[ORDER_FETCH_ERROR] {symbol} order_id={order_id}: {type(e).__name__}: {str(e)}")
                    except Exception:
                        pass
                    
                    # FALLBACK: If verification fails, check exchange positions directly
                    # Market orders on Binance fill instantly - if order_id exists, check positions
                    if order_id:
                        try:
                            from .logger import get_logger
                            logger = get_logger("OrderManager")
                            logger.warning(f"[ORDER_VERIFY_FALLBACK] {symbol} order_id={order_id}: Verification error, checking exchange positions...")
                        except Exception:
                            pass
                        
                        try:
                            # Check if position exists on exchange (order might have filled despite verification failure)
                            await asyncio.sleep(0.5)  # Brief delay for position to appear
                            positions = await self.exchange.fetch_positions([symbol])
                            for pos in positions:
                                pos_symbol = pos.get('symbol', '')
                                # Normalize for comparison
                                if hasattr(self.exchange, 'normalize_symbol'):
                                    pos_symbol = self.exchange.normalize_symbol(pos_symbol)
                                
                                if pos_symbol == symbol or pos_symbol.replace('/USDT:USDT', '') == symbol.replace('/USDT:USDT', ''):
                                    pos_size = abs(float(pos.get('contracts', 0) or pos.get('positionAmt', 0) or 0))
                                    if pos_size > 0:
                                        # Position exists! Order actually filled
                                        entry_price = float(pos.get('entryPrice', 0) or pos.get('markPrice', 0) or 0)
                                        try:
                                            from .logger import get_logger
                                            logger = get_logger("OrderManager")
                                            logger.warning(
                                                f"[ORDER_RECOVERED] {symbol}: Position found on exchange despite verification error! "
                                                f"Size={pos_size:.6f}, Entry=${entry_price:.4f}"
                                            )
                                        except Exception:
                                            pass
                                        return OrderResult(
                                            success=True,
                                            order_id=order_id or f"RECOVERED_{int(time.time())}",
                                            filled_price=entry_price,
                                            filled_size=pos_size
                                        )
                        except Exception as fallback_error:
                            try:
                                from .logger import get_logger
                                logger = get_logger("OrderManager")
                                logger.debug(f"[ORDER_FALLBACK_ERROR] {symbol}: Position check failed: {fallback_error}")
                            except Exception:
                                pass
                    
                    return OrderResult(
                        success=False,
                        error=f"Could not verify order fill: {str(e)}"
                    )
            
            # No fill detected - order likely rejected
            return OrderResult(
                success=False,
                error="Order placed but not filled (likely rejected by exchange)"
            )
        except Exception as e:
            return OrderResult(
                success=False,
                error=str(e)
            )
    
    async def _place_market_order(
        self,
        symbol: str,
        side: str,
        size: float
    ) -> OrderResult:
        """Place market order."""
        try:
            order_type = "market"
            order_side = "buy" if side == "long" else "sell"
            
            # OPTIMIZATION: Binance Futures USDT-M doesn't use positionSide
            order = await self.exchange.create_order(
                symbol,
                order_type,
                order_side,
                size,
                None,  # price not needed for market orders
                params={}  # Let exchange wrapper handle Binance-specific params
            )
            
            # Get filled price
            filled_price = order.get('price') or order.get('average')
            if not filled_price:
                # Try to fetch order status
                order_id = order.get('id')
                if order_id:
                    filled_order = await self.exchange.fetch_order(order_id, symbol)
                    filled_price = filled_order.get('average') or filled_order.get('price')
            
            return OrderResult(
                success=True,
                order_id=order.get('id'),
                filled_price=filled_price,
                filled_size=size
            )
        except Exception as e:
            return OrderResult(
                success=False,
                error=str(e)
            )
    
    async def _place_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float
    ) -> OrderResult:
        """Place limit order."""
        try:
            order_type = "limit"
            order_side = "buy" if side == "long" else "sell"
            
            # OPTIMIZATION: Binance Futures USDT-M doesn't use positionSide
            order = await self.exchange.create_order(
                symbol,
                order_type,
                order_side,
                size,
                price,
                params={}  # Let exchange wrapper handle Binance-specific params
            )
            
            # Wait for fill (with timeout)
            order_id = order.get('id')
            if order_id:
                filled_order = await self._wait_for_fill(order_id, symbol, timeout=5.0)
                if filled_order:
                    filled_size_from_order = filled_order.get('filled', size)
                    filled_size_float = float(filled_size_from_order) if filled_size_from_order else 0.0
                    # ALL-OR-NOTHING: Only accept orders if fully filled (within 1% tolerance for floating point)
                    fill_ratio = filled_size_float / size if size > 0 else 0
                    if fill_ratio >= 0.99:  # 99% or more = fully filled
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            filled_price=filled_order.get('average') or price,
                            filled_size=filled_size_float
                        )
                    else:
                        # Partial fill - reject it (all-or-nothing policy)
                        return OrderResult(
                            success=False,
                            error=f"Partial fill rejected: {filled_size_float:.6f}/{size:.6f} ({fill_ratio*100:.1f}% filled). All-or-nothing policy."
                        )
            
            # Order placed but not filled yet
            return OrderResult(
                success=True,
                order_id=order_id,
                filled_price=price,
                filled_size=0.0  # Not filled yet
            )
        except Exception as e:
            return OrderResult(
                success=False,
                error=str(e)
            )
    
    async def _wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: float = 5.0
    ) -> Optional[Dict]:
        """Wait for order to fill."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                order = await self.exchange.fetch_order(order_id, symbol)
                if order.get('status') == 'closed' or order.get('filled', 0) > 0:
                    return order
                await asyncio.sleep(0.1)
            except (AttributeError, KeyError, RuntimeError):
                # REFACTOR: Break on order fetch errors
                break  # Stop polling on error
        return None
    
    def should_use_limit_order(
        self,
        spread_bps: float,
        signal_strength: float,
        volatility: float = 0.0
    ) -> bool:
        """
        Determine if limit order should be used instead of market order.
        
        Returns:
            True if limit order is recommended
        """
        # Use limit orders for:
        # 1. Tight spreads (< 20 bps)
        # 2. High signal strength (> 0.7)
        # 3. Low volatility (< 0.5)
        
        if spread_bps < 20 and signal_strength > 0.7 and volatility < 0.5:
            return True
        
        # Use market orders for:
        # 1. Wide spreads (> 30 bps)
        # 2. High volatility (> 0.7)
        # 3. Fast-moving markets
        
        if spread_bps > 30 or volatility > 0.7:
            return False
        
        # Default: use limit for strong signals, market for others
        return signal_strength > 0.8

