"""
Extended Multi-Dimensional Scoring System
Modular score components with proper caps to ensure unicorn rarity (90+ = 1-2/hour).
"""

import math
import time
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timezone

from .advanced_features import AdvancedFeatures


@dataclass
class ScoreComponents:
    """Individual score components with caps."""
    base_score: float = 0.0  # 0-60 (capped)
    liq_score: float = 0.0  # -10 to +10
    regime_score: float = 0.0  # -10 to +10
    structure_score: float = 0.0  # -20 to +20
    portfolio_score: float = 0.0  # -15 to +5
    time_score: float = 0.0  # -8 to +8
    symbol_rating_score: float = 0.0  # -10 to +10
    
    def get_total(self) -> float:
        """Calculate total score before clipping."""
        return (
            self.base_score +
            self.liq_score +
            self.regime_score +
            self.structure_score +
            self.portfolio_score +
            self.time_score +
            self.symbol_rating_score
        )
    
    def get_final_score(self) -> float:
        """Get final score clipped to 0-100."""
        raw = self.get_total()
        return max(0.0, min(100.0, raw))


def compute_liquidity_score(
    spread_bps: float,
    orderbook: Optional[Dict] = None,
    order_size_usd: float = 0.0
) -> float:
    """
    Compute liquidity score (-10 to +10).
    
    Goal: Penalize symbols where getting in/out cleanly is hard or expensive.
    """
    liq_score = 0.0
    
    # Spread scoring (0.03% = 3 bps)
    spread_pct = spread_bps / 100.0  # Convert bps to percentage
    
    if spread_pct <= 0.0003:  # <= 0.03%
        liq_score += 8.0
    elif spread_pct <= 0.0005:  # <= 0.05%
        liq_score += 5.0
    elif spread_pct <= 0.001:  # <= 0.10%
        liq_score += 2.0
    else:
        liq_score -= 10.0  # Avoid illiquid symbols
    
    # Depth scoring (orderbook depth within X bps of mid)
    if orderbook and order_size_usd > 0:
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if bids and asks:
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0
            
            if mid_price > 0:
                # Calculate depth within 10 bps (0.1%) of mid
                depth_window = mid_price * 0.001
                
                # Sum bid depth within window
                bid_depth = sum(
                    qty * price for price, qty in bids
                    if (best_bid - price) <= depth_window
                )
                
                # Sum ask depth within window
                ask_depth = sum(
                    qty * price for price, qty in asks
                    if (price - best_ask) <= depth_window
                )
                
                depth_top = min(bid_depth, ask_depth)  # Use the weaker side
                
                if depth_top >= 20000:  # >= 20k USDT
                    liq_score += 5.0
                elif depth_top >= 10000:  # >= 10k USDT
                    liq_score += 3.0
                elif depth_top < 2000:  # < 2k USDT
                    liq_score -= 5.0
    
    # Cap liquidity score
    return max(-10.0, min(10.0, liq_score))


def compute_regime_score(
    atr_pct: Optional[float],
    htf_adx: Optional[float],
    regime_type: str = "scalping"
) -> float:
    """
    Compute volatility/regime score (-10 to +10).
    
    Goal: Stop using the same logic in chop vs trend vs panic.
    """
    regime_score = 0.0
    
    if atr_pct is None:
        atr_pct = 1.0  # Default to normal
    if htf_adx is None:
        htf_adx = 20.0  # Default to moderate
    
    # Determine regime
    if atr_pct < 0.005 and htf_adx < 15:  # 0.5% and ADX < 15
        regime = "chop_lowvol"
    elif atr_pct > 0.02 and htf_adx > 25:  # 2% and ADX > 25
        regime = "trend_highvol"
    elif atr_pct > 0.04:  # 4%
        regime = "panic"
    else:
        regime = "normal"
    
    # Score based on regime and strategy type
    if regime_type == "scalping":
        if regime == "chop_lowvol":
            regime_score += 5.0  # Good for scalping
        elif regime == "trend_highvol":
            regime_score += 3.0  # OK but more volatile
        elif regime == "panic":
            regime_score -= 5.0  # Too wild
        else:  # normal
            regime_score += 2.0
    elif regime_type == "day":
        if regime == "trend_highvol":
            regime_score += 5.0  # Good for day trading
        elif regime == "normal":
            regime_score += 3.0
        elif regime == "chop_lowvol":
            regime_score -= 3.0  # Too choppy
        elif regime == "panic":
            regime_score -= 5.0
    else:  # swing
        if regime == "trend_highvol":
            regime_score += 6.0  # Best for swing
        elif regime == "normal":
            regime_score += 4.0
        elif regime == "chop_lowvol":
            regime_score -= 5.0  # Bad for swing
        elif regime == "panic":
            regime_score -= 8.0  # Very bad
    
    # Cap regime score
    return max(-10.0, min(10.0, regime_score))


def compute_structure_score(
    advanced_features: Optional[AdvancedFeatures],
    side: str
) -> float:
    """
    Compute structure/pattern quality score (-20 to +20).
    
    Uses HTF trend, exhaustion, SFP, divergences, etc.
    """
    if advanced_features is None:
        return 0.0
    
    struct_score = 0.0
    
    if side == "long":
        # Long structure boosters
        if advanced_features.htf_trend_dir == 1:
            struct_score += 5.0
        
        # Check if at range low (using HTF range)
        if (advanced_features.htf_range_low and 
            advanced_features.dist_from_htf_ema50 < -0.01):  # Below HTF EMA50
            struct_score += 6.0
        
        if advanced_features.extreme_down:
            struct_score += 8.0
        
        if advanced_features.bullish_divergence:
            struct_score += 6.0
        
        if advanced_features.exhaustion_down:
            struct_score += 10.0
        
        if advanced_features.sfp_bottom:
            struct_score += 12.0
    else:  # short
        # Short structure boosters
        if advanced_features.htf_trend_dir == -1:
            struct_score += 5.0
        
        # Check if at range high
        if (advanced_features.htf_range_high and 
            advanced_features.dist_from_htf_ema50 > 0.01):  # Above HTF EMA50
            struct_score += 6.0
        
        if advanced_features.extreme_up:
            struct_score += 8.0
        
        if advanced_features.bearish_divergence:
            struct_score += 6.0
        
        if advanced_features.exhaustion_up:
            struct_score += 10.0
        
        if advanced_features.sfp_top:
            struct_score += 12.0
    
    # Cap structure score
    return max(-20.0, min(20.0, struct_score))


def compute_portfolio_score(
    symbol: str,
    side: str,
    open_positions: Union[Dict[str, Any], int],  # Can be Dict or int
    symbol_sector: Optional[str] = None,
    btc_correlation: Optional[float] = None,
    has_btc_position: bool = False
) -> float:
    """
    Compute portfolio/correlation score (-15 to +5).
    
    Goal: Avoid over-concentrating in one sector/narrative.
    """
    portfolio_score = 0.0
    
    # Handle case where open_positions is an int (count) or dict
    if isinstance(open_positions, int):
        # Just count - can't check sectors, but penalize if too many total
        if open_positions >= 8:
            portfolio_score -= 5.0
        elif open_positions >= 6:
            portfolio_score -= 2.0
        return max(-15.0, min(5.0, portfolio_score))
    
    if not isinstance(open_positions, dict):
        return 0.0
    
    # Count positions by sector and direction
    sector_long_count = 0
    sector_short_count = 0
    
    if symbol_sector:
        for pos_symbol, pos_data in open_positions.items():
            if isinstance(pos_data, dict):
                pos_side = pos_data.get('side', 'long').lower()
                pos_sector = pos_data.get('sector', None)
                
                if pos_sector == symbol_sector:
                    if pos_side == 'long':
                        sector_long_count += 1
                    else:
                        sector_short_count += 1
    
    # Penalize if too many positions in same sector
    if side == "long":
        if sector_long_count >= 3:
            portfolio_score -= 8.0
        elif sector_long_count >= 2:
            portfolio_score -= 4.0
    else:  # short
        if sector_short_count >= 3:
            portfolio_score -= 8.0
        elif sector_short_count >= 2:
            portfolio_score -= 4.0
    
    # Penalize high BTC correlation if already long BTC
    if btc_correlation is not None and btc_correlation > 0.9:
        if side == "long" and has_btc_position:
            portfolio_score -= 5.0
    
    # Cap portfolio score
    return max(-15.0, min(5.0, portfolio_score))


def compute_time_score(
    hour_utc: Optional[int] = None
) -> float:
    """
    Compute time/session score (-8 to +8).
    
    Goal: Treat signals differently during dead hours vs active hours.
    """
    time_score = 0.0
    
    if hour_utc is None:
        hour_utc = datetime.now(timezone.utc).hour
    
    # Determine session (UTC hours)
    # Asia: 00-08, Europe: 08-16, US: 13-21, Weekend: Sat-Sun
    weekday = datetime.now(timezone.utc).weekday()  # 0=Mon, 6=Sun
    is_weekend = weekday >= 5
    
    if is_weekend:
        session = "Weekend"
    elif 0 <= hour_utc < 8:
        session = "Asia"
    elif 8 <= hour_utc < 13:
        session = "EU"
    elif 13 <= hour_utc < 21:
        session = "US"
    else:
        session = "Asia"  # Late night US / early Asia
    
    # Score based on session
    if session in ["EU", "US"]:
        time_score += 4.0  # Active sessions
    elif session == "Asia":
        time_score += 1.0  # Moderate activity
    elif session == "Weekend":
        if 0 <= hour_utc < 6:  # Dead hours
            time_score -= 6.0
        else:
            time_score -= 2.0  # Weekend but not dead
    
    # Cap time score
    return max(-8.0, min(8.0, time_score))


def compute_symbol_rating_score(
    symbol: str,
    symbol_stats: Optional[Dict] = None,
    historical_performance: Optional[Dict] = None
) -> float:
    """
    Compute symbol rating score (-10 to +10).
    
    Goal: Auto-downrank symbols with poor historical performance.
    
    If historical data available, use it.
    Otherwise, use symbol stats as proxy.
    """
    rating_score = 0.0
    
    if historical_performance:
        # Use historical data
        avg_slippage = historical_performance.get('avg_slippage', 0.0)
        avg_spread = historical_performance.get('avg_spread', 0.0)
        realized_rr = historical_performance.get('realized_rr', 1.0)
        win_rate = historical_performance.get('win_rate', 0.5)
        
        # Penalize high slippage
        if avg_slippage > 0.001:  # > 0.1%
            rating_score -= 3.0
        elif avg_slippage > 0.0005:  # > 0.05%
            rating_score -= 1.0
        
        # Penalize wide spreads
        if avg_spread > 0.001:  # > 0.1%
            rating_score -= 2.0
        
        # Penalize poor R:R
        if realized_rr < 0.8:
            rating_score -= 3.0
        elif realized_rr < 1.0:
            rating_score -= 1.0
        elif realized_rr > 1.5:
            rating_score += 2.0  # Bonus for good R:R
        
        # Reward good win rate
        if win_rate > 0.6:
            rating_score += 2.0
        elif win_rate < 0.4:
            rating_score -= 2.0
    elif symbol_stats:
        # Use current stats as proxy
        volume_24h = symbol_stats.get('vol_quote', 0)
        spread_bps = symbol_stats.get('spread_bps', 9999)
        
        # Reward high volume (more reliable)
        if volume_24h > 50_000_000:  # > $50M
            rating_score += 2.0
        elif volume_24h < 5_000_000:  # < $5M
            rating_score -= 2.0
        
        # Penalize wide spread
        if spread_bps > 20:  # > 0.2%
            rating_score -= 3.0
    
    # Cap symbol rating score
    return max(-10.0, min(10.0, rating_score))


def compute_extended_score(
    base_score: float,  # Base score (will be capped to 0-60)
    symbol: str,
    side: str,
    symbol_stats: Dict,
    orderbook: Optional[Dict] = None,
    order_size_usd: float = 0.0,
    indicators: Optional[Dict] = None,
    advanced_features: Optional[AdvancedFeatures] = None,
    position_manager_state: Optional[Dict] = None,
    regime_type: str = "scalping",
    historical_performance: Optional[Dict] = None,
    hour_utc: Optional[int] = None
) -> Tuple[ScoreComponents, float]:
    """
    Compute extended multi-dimensional score.
    
    Args:
        base_score: Base signal score (will be capped to 0-60)
        ... (all other params for component scores)
    
    Returns:
        (ScoreComponents, final_score)
    """
    # Cap base score to 0-60 (base must dominate)
    base_score = max(0.0, min(60.0, base_score))
    
    # Compute component scores
    components = ScoreComponents(base_score=base_score)
    
    # 1. Liquidity score
    spread_bps = symbol_stats.get('spread_bps', 9999)
    components.liq_score = compute_liquidity_score(spread_bps, orderbook, order_size_usd)
    
    # 2. Regime score
    atr_pct = indicators.get('atr_pct') if indicators else None
    htf_adx = advanced_features.htf_adx_14 if advanced_features else None
    components.regime_score = compute_regime_score(atr_pct, htf_adx, regime_type)
    
    # 3. Structure score (uses advanced_features)
    components.structure_score = compute_structure_score(advanced_features, side)
    
    # 4. Portfolio score
    # position_manager_state.get('open_positions') returns int (count), not dict
    open_positions_raw = position_manager_state.get('open_positions', 0) if position_manager_state else 0
    symbol_sector = symbol_stats.get('sector', None)
    btc_correlation = symbol_stats.get('btc_correlation', None)
    
    # Check if we have actual positions dict (from bot.positions) or just count
    # For now, use count - sector tracking would need bot.positions dict
    has_btc_position = False  # TODO: Check from bot.positions if available
    
    components.portfolio_score = compute_portfolio_score(
        symbol, side, open_positions_raw, symbol_sector, btc_correlation, has_btc_position
    )
    
    # 5. Time score
    components.time_score = compute_time_score(hour_utc)
    
    # 6. Symbol rating score
    components.symbol_rating_score = compute_symbol_rating_score(
        symbol, symbol_stats, historical_performance
    )
    
    # Get final score (clipped to 0-100)
    final_score = components.get_final_score()
    
    return components, final_score

