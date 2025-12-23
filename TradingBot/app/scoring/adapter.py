"""Adapter to convert existing data structures to SignalContext."""

import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any

from .context import SignalContext
from .config import BASE_SCORE_MIN, BASE_SCORE_MAX
from .model import clamp
from ..advanced_features import AdvancedFeatures


def build_signal_context(
    symbol: str,
    side: str,
    base_score: float,  # Base score 0-100 (trusted as true primary score)
    symbol_stats: Dict,
    orderbook: Optional[Dict] = None,
    indicators: Optional[Dict] = None,
    advanced_features: Optional[AdvancedFeatures] = None,
    position_manager_state: Optional[Dict] = None,
    bot_positions: Optional[Dict] = None  # Optional: actual positions dict from bot
) -> SignalContext:
    """
    Build SignalContext from existing data structures.
    
    Args:
        symbol: Trading symbol
        side: 'long' or 'short'
        base_score: Base signal score (0-100) from primary scoring engine
        symbol_stats: Symbol statistics dict
        orderbook: Orderbook dict
        indicators: Indicators dict
        advanced_features: AdvancedFeatures object (optional)
        position_manager_state: Position manager state dict
        bot_positions: Optional bot.positions dict for sector tracking
    
    Returns:
        SignalContext object
    
    Notes:
        - base_score is now the true 0-100 value from the primary engine
        - No scaling or artificial capping applied
        - Modules add small adjustments (+/- 5-20 points) around the base
        - Final score = base + modules, clamped to [0, 100]
    """
    # Base score is already 0-100 from primary engine - just clamp for safety
    base_score_clamped = clamp(base_score, BASE_SCORE_MIN, BASE_SCORE_MAX)
    
    # Time context
    now_ts = int(time.time())
    now_utc = datetime.now(timezone.utc)
    hour_utc = now_utc.hour
    weekday = now_utc.weekday()  # 0=Mon, 6=Sun
    
    # Determine session
    if weekday >= 5:  # Sat-Sun
        session = "WEEKEND"
    elif 0 <= hour_utc < 8:
        session = "ASIA"
    elif 8 <= hour_utc < 13:
        session = "EU"
    elif 13 <= hour_utc < 21:
        session = "US"
    else:
        session = "ASIA"  # Late night US / early Asia
    
    # Market microstructure
    spread_bps = symbol_stats.get('spread_bps', 9999)
    spread_pct = spread_bps / 10000.0  # Convert bps to percentage
    
    # Calculate depth from orderbook
    depth_top_usd = 0.0
    if orderbook:
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if bids and asks:
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0
            
            if mid_price > 0:
                # Depth within 10 bps (0.1%) of mid
                depth_window = mid_price * 0.001
                bid_depth = sum(
                    qty * price for price, qty in bids
                    if abs(best_bid - price) <= depth_window
                )
                ask_depth = sum(
                    qty * price for price, qty in asks
                    if abs(price - best_ask) <= depth_window
                )
                depth_top_usd = min(bid_depth, ask_depth)
    
    # ATR and HTF ADX
    atr_pct = indicators.get('atr_pct') if indicators else 0.01  # Default 1%
    htf_adx = advanced_features.htf_adx_14 if advanced_features else 20.0  # Default
    
    # Structure / pattern flags
    htf_trend_dir = advanced_features.htf_trend_dir if advanced_features else 0
    at_range_high = False
    at_range_low = False
    
    if advanced_features:
        if advanced_features.htf_range_high and advanced_features.dist_from_htf_ema50 > 0.01:
            at_range_high = True
        if advanced_features.htf_range_low and advanced_features.dist_from_htf_ema50 < -0.01:
            at_range_low = True
    
    extreme_up = advanced_features.extreme_up if advanced_features else False
    extreme_down = advanced_features.extreme_down if advanced_features else False
    bearish_divergence = advanced_features.bearish_divergence if advanced_features else False
    bullish_divergence = advanced_features.bullish_divergence if advanced_features else False
    exhaustion_up = advanced_features.exhaustion_up if advanced_features else False
    exhaustion_down = advanced_features.exhaustion_down if advanced_features else False
    sfp_top = advanced_features.sfp_top if advanced_features else False
    sfp_bottom = advanced_features.sfp_bottom if advanced_features else False
    dist_from_vwap = advanced_features.dist_from_vwap if advanced_features else 0.0
    
    # Portfolio stats
    open_positions_same_sector = 0  # TODO: Track sectors if available
    corr_to_btc_24h = symbol_stats.get('btc_correlation', 0.0)
    
    # Symbol rating (from historical or defaults)
    symbol_rating = 0.0
    if advanced_features:
        # Use advanced features as proxy for rating
        # Higher score = better rating
        if advanced_features.reversal_score_long > 20 or advanced_features.reversal_score_short > 20:
            symbol_rating = 5.0  # Good
        elif advanced_features.reversal_score_long > 10 or advanced_features.reversal_score_short > 10:
            symbol_rating = 2.0  # Decent
        # Default is 0.0
    
    return SignalContext(
        symbol=symbol,
        side=side,  # type: ignore
        base_score=base_score_clamped,
        now_ts=now_ts,
        hour_utc=hour_utc,
        session=session,
        spread_pct=spread_pct,
        depth_top_usd=depth_top_usd,
        atr_pct=atr_pct,
        htf_adx=htf_adx,
        htf_trend_dir=htf_trend_dir,
        at_range_high=at_range_high,
        at_range_low=at_range_low,
        extreme_up=extreme_up,
        extreme_down=extreme_down,
        bearish_divergence=bearish_divergence,
        bullish_divergence=bullish_divergence,
        exhaustion_up=exhaustion_up,
        exhaustion_down=exhaustion_down,
        sfp_top=sfp_top,
        sfp_bottom=sfp_bottom,
        dist_from_vwap=dist_from_vwap,
        open_positions_same_sector=open_positions_same_sector,
        corr_to_btc_24h=corr_to_btc_24h,
        symbol_rating=symbol_rating
    )

