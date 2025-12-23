"""
Unified Quality Gate - Single quality score instead of 3-stage filter pipeline.

PHASE 4: Enhanced with evolvable thresholds and exchange-aware calculations.

Returns a quality multiplier (0-1) that penalizes signals instead of rejecting them.
This allows evolution to optimize filter strictness.

Quality multiplier:
- 1.0 = perfect quality (no penalty)
- 0.5 = moderate quality (50% penalty)
- 0.0 = poor quality (100% penalty, effectively rejects)
"""

import math
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Quality gate result with multiplier and details."""
    multiplier: float  # 0-1, applied to final_score
    microstructure_score: float  # 0-1
    structure_score: float  # 0-1
    direction_score: float  # 0-1
    details: Dict[str, Any]


def compute_quality_gate_score(
    symbol: str,
    side: str,
    symbol_stats: Dict,
    orderbook: Optional[Dict],
    indicators: Optional[Dict],
    advanced_features: Optional[Any],
    entry_price: float,
    filter_strictness: float = 0.5,  # Evolvable: 0.0 = lenient, 1.0 = strict
    # PHASE 4: Evolvable microstructure thresholds
    max_spread_pct: float = 0.0025,  # Evolvable: max spread (0.25% default)
    min_depth_usd: float = 800.0,  # Evolvable: min depth (800 USD default)
    max_volatility_pct: float = 5.0,  # Evolvable: max ATR % (5% default)
    min_volatility_pct: float = 0.3,  # Evolvable: min ATR % (0.3% default)
    slippage_budget_bps: float = 10.0,  # Evolvable: max acceptable slippage (10 bps default)
    min_viable_size_usd: float = 10.0  # Evolvable: min order size (10 USD default)
) -> QualityScore:
    """
    Compute unified quality gate score.
    
    Args:
        symbol: Trading symbol
        side: Position side ('long' or 'short')
        symbol_stats: Symbol statistics
        orderbook: Orderbook data (optional)
        indicators: Technical indicators (optional)
        advanced_features: Advanced features object (optional)
        entry_price: Entry price
        filter_strictness: Evolvable parameter (0-1)
            - 0.0 = lenient (minimal penalties)
            - 1.0 = strict (maximum penalties)
    
    Returns:
        QualityScore with multiplier and component scores
    """
    details = {}
    
    # ========================================================================
    # Component 1: Microstructure Quality (spread, depth, liquidity, volatility)
    # PHASE 4: Enhanced with exchange-aware calculations
    # ========================================================================
    microstructure_score = 1.0  # Start at perfect
    
    spread_bps = symbol_stats.get('spread_bps', 9999)
    spread_pct = spread_bps / 10000.0
    details['spread_pct'] = spread_pct
    
    # PHASE 4: Exchange-aware spread calculation (spread in ticks)
    price_tick = symbol_stats.get('price_tick', 0.0)  # From SymbolStats
    spread_in_ticks = 0.0
    if price_tick > 0 and entry_price > 0:
        spread_in_ticks = (spread_pct * entry_price) / price_tick
        details['spread_in_ticks'] = spread_in_ticks
        details['price_tick'] = price_tick
    else:
        details['spread_in_ticks'] = None
        details['price_tick'] = None
    
    # Calculate depth if orderbook available
    depth_top_usd = 0.0
    if orderbook:
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if bids and asks:
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0
            
            if mid_price > 0:
                depth_window = mid_price * 0.001  # 0.1% window
                bid_depth = sum(
                    qty * price for price, qty in bids
                    if abs(best_bid - price) <= depth_window
                )
                ask_depth = sum(
                    qty * price for price, qty in asks
                    if abs(price - best_ask) <= depth_window
                )
                depth_top_usd = min(bid_depth, ask_depth)
    
    details['depth_top_usd'] = depth_top_usd
    
    # PHASE 4: Volatility gate (ATR-based)
    atr_pct = indicators.get('atr_pct') if indicators else None
    volatility_penalty = 0.0
    if atr_pct is not None:
        details['atr_pct'] = atr_pct
        # Penalize extreme volatility (too high or too low)
        if atr_pct > max_volatility_pct:
            # Too volatile - dangerous for execution
            vol_excess = (atr_pct - max_volatility_pct) / max_volatility_pct
            volatility_penalty = min(0.4, vol_excess * 0.2)  # Max 40% penalty
        elif atr_pct < min_volatility_pct:
            # Too low volatility - hard to profit
            vol_deficit = (min_volatility_pct - atr_pct) / min_volatility_pct
            volatility_penalty = min(0.2, vol_deficit * 0.1)  # Max 20% penalty
    else:
        details['atr_pct'] = None
    
    # PHASE 4: Slippage budget check
    # Estimate slippage based on spread, volatility, and order size
    estimated_slippage_bps = spread_bps / 2.0  # Base: half spread
    if atr_pct is not None:
        estimated_slippage_bps += min(atr_pct * 0.5, 20.0)  # Volatility component
    
    slippage_budget_exceeded = estimated_slippage_bps > slippage_budget_bps
    details['estimated_slippage_bps'] = estimated_slippage_bps
    details['slippage_budget_bps'] = slippage_budget_bps
    details['slippage_budget_exceeded'] = slippage_budget_exceeded
    
    if slippage_budget_exceeded:
        slippage_penalty = min(0.3, (estimated_slippage_bps - slippage_budget_bps) / slippage_budget_bps * 0.3)
        microstructure_score -= slippage_penalty
    
    # PHASE 4: Minimum viable order size check
    # Check if depth can support minimum order size
    min_viable_supported = depth_top_usd >= min_viable_size_usd if depth_top_usd > 0 else True
    details['min_viable_size_usd'] = min_viable_size_usd
    details['min_viable_supported'] = min_viable_supported
    
    if not min_viable_supported and depth_top_usd > 0:
        size_penalty = 0.2  # 20% penalty if can't support min size
        microstructure_score -= size_penalty
    
    # Penalize wide spreads (soft penalty, not hard reject)
    # PHASE 4: Uses evolvable max_spread_pct
    if spread_pct > max_spread_pct:
        # Penalty increases with spread width
        spread_penalty = min(1.0, (spread_pct - max_spread_pct) / max_spread_pct)
        microstructure_score -= spread_penalty * 0.5  # Max 50% penalty for wide spread
    
    # Penalize thin depth (only if depth data available)
    # PHASE 4: Uses evolvable min_depth_usd
    if orderbook and depth_top_usd > 0:
        if depth_top_usd < min_depth_usd:
            depth_penalty = 1.0 - (depth_top_usd / min_depth_usd)
            microstructure_score -= depth_penalty * 0.3  # Max 30% penalty for thin depth
    
    # Apply volatility penalty
    microstructure_score -= volatility_penalty
    
    microstructure_score = max(0.0, min(1.0, microstructure_score))
    details['microstructure_score'] = microstructure_score
    details['volatility_penalty'] = volatility_penalty
    
    # ========================================================================
    # Component 2: Structure Quality (trend, support/resistance, confluence)
    # ========================================================================
    structure_score = 1.0  # Start at perfect
    
    # Check for structure signals (if advanced_features available)
    if advanced_features:
        # Count structure signals (support, resistance, trend, etc.)
        structure_signals = []
        
        if hasattr(advanced_features, 'has_support') and advanced_features.has_support:
            structure_signals.append('support')
        if hasattr(advanced_features, 'has_resistance') and advanced_features.has_resistance:
            structure_signals.append('resistance')
        if hasattr(advanced_features, 'trend_strength') and advanced_features.trend_strength:
            structure_signals.append('trend')
        
        # Bonus for multiple structure signals (confluence)
        if len(structure_signals) >= 2:
            structure_score = 1.0  # Perfect with confluence
        elif len(structure_signals) == 1:
            structure_score = 0.8  # Good with single signal
        else:
            structure_score = 0.6  # Neutral without structure signals
        
        details['structure_signals'] = structure_signals
    else:
        # No advanced features available - neutral score
        structure_score = 0.7
        details['structure_signals'] = []
    
    details['structure_score'] = structure_score
    
    # ========================================================================
    # Component 3: Direction Quality (momentum, exhaustion, divergence)
    # ========================================================================
    direction_score = 0.5  # Start neutral
    
    if indicators:
        rsi = indicators.get('rsi', 50.0)
        adx = indicators.get('adx', 20.0)
        atr_pct = indicators.get('atr_pct', 0.01)
        
        # Direction score based on RSI and ADX
        if side == "long":
            # Long: prefer RSI 40-60 (not overbought), ADX > 20 (trending)
            if 40 <= rsi <= 60 and adx > 20:
                direction_score = 1.0
            elif 30 <= rsi <= 70 and adx > 15:
                direction_score = 0.8
            else:
                direction_score = 0.5
        else:  # short
            # Short: prefer RSI 40-60 (not oversold), ADX > 20 (trending)
            if 40 <= rsi <= 60 and adx > 20:
                direction_score = 1.0
            elif 30 <= rsi <= 70 and adx > 15:
                direction_score = 0.8
            else:
                direction_score = 0.5
        
        # Check for exhaustion (bad for direction)
        if advanced_features:
            if side == "long" and hasattr(advanced_features, 'exhaustion_down') and advanced_features.exhaustion_down:
                direction_score *= 0.7  # Penalize exhaustion
            elif side == "short" and hasattr(advanced_features, 'exhaustion_up') and advanced_features.exhaustion_up:
                direction_score *= 0.7
        
        details['rsi'] = rsi
        details['adx'] = adx
    else:
        # No indicators available - neutral score
        direction_score = 0.6
        details['rsi'] = None
        details['adx'] = None
    
    details['direction_score'] = direction_score
    
    # ========================================================================
    # Combine components with weights
    # ========================================================================
    # Weighted average of components
    # Microstructure: 40% (most important - execution quality)
    # Structure: 30% (setup quality)
    # Direction: 30% (momentum quality)
    base_multiplier = (
        0.4 * microstructure_score +
        0.3 * structure_score +
        0.3 * direction_score
    )
    
    # Apply filter_strictness: strict mode penalizes more
    # filter_strictness = 0.0 -> multiplier stays at base
    # filter_strictness = 1.0 -> multiplier reduced by up to 50%
    strictness_penalty = (1.0 - base_multiplier) * filter_strictness * 0.5
    final_multiplier = base_multiplier - strictness_penalty
    
    # Clamp to valid range
    final_multiplier = max(0.0, min(1.0, final_multiplier))
    
    details['filter_strictness'] = filter_strictness
    details['base_multiplier'] = base_multiplier
    details['strictness_penalty'] = strictness_penalty
    details['final_multiplier'] = final_multiplier
    
    return QualityScore(
        multiplier=final_multiplier,
        microstructure_score=microstructure_score,
        structure_score=structure_score,
        direction_score=direction_score,
        details=details
    )

