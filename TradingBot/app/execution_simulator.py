"""
Realistic Exchange Execution Simulation
Models slippage, partial fills, stop triggers, and order queue behavior.
"""

import random
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of order execution simulation"""
    filled_price: float
    filled_quantity: float
    slippage_bps: float
    partial_fill: bool
    execution_time_ms: float


class ExchangeSimulator:
    """
    Realistic exchange execution simulator.
    Models Binance Futures execution characteristics.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize simulator with optional seed for determinism.
        
        Args:
            seed: Random seed for deterministic execution (None = non-deterministic)
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed
    
    def calculate_slippage(
        self,
        order_price: float,
        side: str,
        spread_bps: float,
        volatility_pct: float,
        order_size_usd: float,
        market_impact_factor: float = 0.1
    ) -> float:
        """
        Calculate expected slippage in basis points.
        
        Slippage components:
        1. Spread cost: Half the spread (crossing spread)
        2. Volatility slippage: Higher vol = worse fills
        3. Market impact: Larger orders = worse fills
        
        Args:
            order_price: Intended execution price
            side: 'long' or 'short'
            spread_bps: Current spread in basis points
            volatility_pct: Current volatility (ATR %)
            order_size_usd: Order size in USD
            market_impact_factor: Market impact multiplier (0.1 = 10% of spread per $10k)
            
        Returns:
            Slippage in basis points
        """
        # Base slippage: half the spread (crossing the spread)
        base_slippage = spread_bps / 2.0
        
        # Volatility slippage: higher vol = worse fills
        # Formula: vol_slippage = volatility_pct * 0.5 (scaled)
        vol_slippage = min(volatility_pct * 0.5, 20.0)  # Cap at 20 bps
        
        # Market impact: larger orders = worse fills
        # Impact = (order_size / 10000) * market_impact_factor * spread
        impact_multiplier = 1.0 + (order_size_usd / 10000.0) * market_impact_factor
        impact_slippage = base_slippage * (impact_multiplier - 1.0)
        
        # Total slippage
        total_slippage = base_slippage + vol_slippage + impact_slippage
        
        # Add small random component (execution timing variance)
        # In deterministic mode, this should be seeded
        random_component = random.uniform(-2.0, 2.0)  # ±2 bps variance
        
        return max(0.0, total_slippage + random_component)
    
    def simulate_market_order(
        self,
        side: str,
        intended_price: float,
        quantity: float,
        spread_bps: float,
        volatility_pct: float,
        orderbook: Optional[Dict] = None
    ) -> ExecutionResult:
        """
        Simulate market order execution.
        
        Args:
            side: 'long' or 'short'
            intended_price: Intended execution price
            quantity: Order quantity
            spread_bps: Current spread in basis points
            volatility_pct: Current volatility (ATR %)
            orderbook: Optional orderbook data for depth analysis
            
        Returns:
            ExecutionResult with filled price and slippage
        """
        order_size_usd = intended_price * quantity
        
        # Calculate slippage
        slippage_bps = self.calculate_slippage(
            intended_price, side, spread_bps, volatility_pct, order_size_usd
        )
        
        # Apply slippage to price
        slippage_pct = slippage_bps / 10000.0
        
        if side == 'long':
            # Long: pay more (slippage increases price)
            filled_price = intended_price * (1 + slippage_pct)
        else:
            # Short: receive less (slippage decreases price)
            filled_price = intended_price * (1 - slippage_pct)
        
        # Check for partial fills (large orders relative to depth)
        partial_fill = False
        filled_quantity = quantity
        
        if orderbook and orderbook.get('bids') and orderbook.get('asks'):
            # Estimate depth at 1% from mid price
            mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2.0
            depth_threshold = mid_price * 0.01
            
            if side == 'long':
                # Check ask depth
                ask_depth = sum(
                    price * qty for price, qty in orderbook['asks'][:5]
                    if price - mid_price <= depth_threshold
                )
                if order_size_usd > ask_depth * 0.5:  # Order > 50% of available depth
                    partial_fill = True
                    # Fill proportionally
                    fill_ratio = min(1.0, (ask_depth * 0.5) / order_size_usd)
                    filled_quantity = quantity * fill_ratio
            else:
                # Check bid depth
                bid_depth = sum(
                    price * qty for price, qty in orderbook['bids'][:5]
                    if mid_price - price <= depth_threshold
                )
                if order_size_usd > bid_depth * 0.5:
                    partial_fill = True
                    fill_ratio = min(1.0, (bid_depth * 0.5) / order_size_usd)
                    filled_quantity = quantity * fill_ratio
        
        # Execution time (latency + queue time)
        base_latency = 50.0  # ms
        queue_time = 0.0
        if partial_fill:
            queue_time = random.uniform(100.0, 500.0)  # Partial fills take longer
        execution_time = base_latency + queue_time
        
        return ExecutionResult(
            filled_price=filled_price,
            filled_quantity=filled_quantity,
            slippage_bps=slippage_bps,
            partial_fill=partial_fill,
            execution_time_ms=execution_time
        )
    
    def simulate_stop_trigger(
        self,
        side: str,
        stop_price: float,
        current_high: float,
        current_low: float,
        current_close: float,
        mark_price: Optional[float] = None,
        mark_price_timestamp: Optional[float] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Simulate stop loss/take profit trigger.
        
        Binance Futures uses mark price for stop triggers (not last price).
        If mark price not available, use last price with conservative assumption.
        
        Args:
            side: 'long' or 'short'
            stop_price: Stop loss or take profit price
            current_high: Current candle high
            current_low: Current candle low
            current_close: Current candle close
            mark_price: Optional mark price (more accurate for futures)
            mark_price_timestamp: Optional timestamp of mark price (Unix time in seconds)
            
        Returns:
            (triggered: bool, execution_price: float or None)
        """
        # CRITICAL: Check mark price staleness (if >5 seconds old, use last price instead)
        # Stale mark prices can cause premature exits or missed stops
        MAX_MARK_PRICE_AGE_SEC = 5.0
        
        if mark_price is not None and mark_price_timestamp is not None:
            mark_price_age_sec = time.time() - mark_price_timestamp
            if mark_price_age_sec > MAX_MARK_PRICE_AGE_SEC:
                # Mark price is stale, use last price instead
                from .logger import get_logger
                logger = get_logger("ExecutionSimulator")
                logger.warning(
                    f"Mark price is stale ({mark_price_age_sec:.1f}s old, max {MAX_MARK_PRICE_AGE_SEC}s), "
                    f"using last price ({current_close}) for stop trigger"
                )
                trigger_price = current_close
            else:
                trigger_price = mark_price
        else:
            # Use mark price if available, otherwise use last price (conservative)
            trigger_price = mark_price if mark_price is not None else current_close
        
        if side == 'long':
            # Long stop loss: triggered if price drops to/below stop
            if current_low <= stop_price:
                # Execution at stop price (or worse if gap)
                execution_price = min(stop_price, current_low)
                return True, execution_price
            # Long take profit: triggered if price rises to/above target
            elif current_high >= stop_price:
                execution_price = max(stop_price, current_high)
                return True, execution_price
        else:  # short
            # Short stop loss: triggered if price rises to/above stop
            if current_high >= stop_price:
                execution_price = max(stop_price, current_high)
                return True, execution_price
            # Short take profit: triggered if price drops to/below target
            elif current_low <= stop_price:
                execution_price = min(stop_price, current_low)
                return True, execution_price
        
        return False, None
    
    def calculate_fees(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,
        taker_fee_rate: float = 0.0004
    ) -> Tuple[float, float]:
        """
        Calculate entry and exit fees.
        
        Args:
            entry_price: Entry execution price
            exit_price: Exit execution price
            quantity: Trade quantity
            side: 'long' or 'short'
            taker_fee_rate: Taker fee rate (default 0.04% for Binance Futures)
            
        Returns:
            (entry_fee_usd, exit_fee_usd)
        """
        notional_entry = entry_price * quantity
        notional_exit = exit_price * quantity
        
        entry_fee = notional_entry * taker_fee_rate
        exit_fee = notional_exit * taker_fee_rate
        
        return entry_fee, exit_fee

