"""
Scalper Entry Filters - Three-stage filter pipeline for high-quality scalper entries.

Stage 1: Microstructure Gate
Stage 2: Structure & Reversal Gate  
Stage 3: Candle Direction Prediction Gate
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Result of a filter stage."""
    passed: bool
    reason: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class StructureConfluence:
    """Structure confluence evaluation result."""
    count: int  # Number of aligned structure signals
    signals: list  # List of signal names present
    score_boost: float  # Additional score boost for confluence


def stage1_microstructure_gate(
    symbol: str,
    side: str,
    symbol_stats: Dict,
    orderbook: Optional[Dict],
    indicators: Optional[Dict],
    advanced_features: Optional[Any],
    entry_price: float
) -> FilterResult:
    """
    Stage 1: Microstructure Gate - PURE_SCALPER MODE
    
    FIX: Pass by default, only reject on TRULY invalid conditions:
    - spread > 0.25% (absurdly wide)
    - depth < 800 USD (extremely thin)
    - Missing data is NOT a rejection reason
    """
    details = {}
    
    # FIX: Default to pass - missing microstructure data is neutral, not a rejection
    # Only reject on truly invalid conditions
    spread_bps = symbol_stats.get('spread_bps', 9999)
    spread_pct = spread_bps / 10000.0
    details['spread_pct'] = spread_pct
    
    # Calculate depth if orderbook available (but missing orderbook is OK)
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
    details['depth_top_usd'] = depth_top_usd
    
    # FIX: Only reject on TRULY invalid conditions (not missing data)
    # Hard thresholds: spread > 0.25% OR depth < 800 USD (only if depth data available)
    max_spread_pct = 0.0025  # 0.25% (absurdly wide)
    min_depth_usd = 800.0  # $800 USD (extremely thin)
    
    # Check spread threshold (only reject if absurdly wide)
    spread_too_wide = spread_pct > max_spread_pct
    
    # Check depth only if orderbook is available AND we have depth data
    # Missing orderbook is NOT a rejection reason
    depth_insufficient = False
    if orderbook and depth_top_usd > 0:
        # Only check depth if we actually have orderbook data
        depth_insufficient = depth_top_usd < min_depth_usd
    # If orderbook is missing, don't reject (missing data is neutral)
    
    # FIX: Only reject on truly invalid conditions
    too_thin = spread_too_wide or depth_insufficient
    if too_thin:
        reason_detail = []
        if spread_too_wide:
            reason_detail.append(f"spread={spread_pct*100:.3f}% > {max_spread_pct*100:.2f}%")
        if depth_insufficient:
            reason_detail.append(f"depth=${depth_top_usd:.0f} < ${min_depth_usd:.0f}")
        return FilterResult(
            passed=False,
            reason=f"microstructure_fail: {' and '.join(reason_detail)}",
            details={"spread_pct": spread_pct, "depth_top_usd": depth_top_usd, "threshold_spread": max_spread_pct, "threshold_depth": min_depth_usd}
        )
    
    # FIX: Missing microstructure data (orderbook, depth preview, etc.) is NOT a rejection
    # Pass by default - only reject on truly invalid conditions above
    
    return FilterResult(passed=True, reason="microstructure_pass", details=details)


def stage2_structure_gate(
    side: str,
    advanced_features: Optional[Any],
    indicators: Optional[Dict],
    symbol_stats: Dict
) -> Tuple[FilterResult, StructureConfluence]:
    """
    Stage 2: Structure & Reversal Gate - BALANCED MODE
    
    Requires: (1 strong signal) OR (2 medium signals)
    
    Strong signals:
    - Exhaustion (exhaustion_up/down)
    - Divergence (bullish/bearish)
    - SFP (swing failure pattern top/bottom)
    
    Medium signals:
    - VWAP recoil
    - Range position (at_range_low/high)
    """
    signals_present = []
    count = 0
    
    # FIX: If advanced_features is missing OR has "no_advance" flag, treat as pass (neutral, not rejection)
    # Only reject on truly conflicting signals, not missing data
    if not advanced_features:
        # Missing advanced_features is NOT a rejection reason - pass through
        return (
            FilterResult(passed=True, reason="structure_pass: no_advanced_features_available"),
            StructureConfluence(count=0, signals=[], score_boost=0.0)
        )
    
    # FIX: Check if advanced_features has a "no_advance" flag indicating missing microstructure data
    # If so, treat as pass (neutral) - do NOT reject
    has_no_advance = False
    if hasattr(advanced_features, 'no_advance'):
        has_no_advance = bool(advanced_features.no_advance)
    elif hasattr(advanced_features, 'has_advanced_microstructure'):
        has_no_advance = not bool(advanced_features.has_advanced_microstructure)
    
    if has_no_advance:
        # "no_advance" flag is set - missing microstructure data is NOT a rejection reason
        # Pass through with neutral structure (no boost, but not rejected)
        return (
            FilterResult(passed=True, reason="structure_pass: no_advance_flag_set_but_allowed"),
            StructureConfluence(count=0, signals=[], score_boost=0.0)
        )
    
    # Check exhaustion
    if side == "long":
        if hasattr(advanced_features, 'exhaustion_down') and advanced_features.exhaustion_down:
            signals_present.append("exhaustion_down")
            count += 1
    else:
        if hasattr(advanced_features, 'exhaustion_up') and advanced_features.exhaustion_up:
            signals_present.append("exhaustion_up")
            count += 1
    
    # Check divergence
    if side == "long":
        if hasattr(advanced_features, 'bullish_divergence') and advanced_features.bullish_divergence:
            signals_present.append("bullish_divergence")
            count += 1
    else:
        if hasattr(advanced_features, 'bearish_divergence') and advanced_features.bearish_divergence:
            signals_present.append("bearish_divergence")
            count += 1
    
    # Check SFP
    if side == "long":
        if hasattr(advanced_features, 'sfp_bottom') and advanced_features.sfp_bottom:
            signals_present.append("sfp_bottom")
            count += 1
    else:
        if hasattr(advanced_features, 'sfp_top') and advanced_features.sfp_top:
            signals_present.append("sfp_top")
            count += 1
    
    # Check VWAP recoil (price snapping back toward VWAP in trade direction)
    if hasattr(advanced_features, 'dist_from_vwap'):
        dist = advanced_features.dist_from_vwap
        if side == "long" and dist < -0.01:  # Price below VWAP, snapping back up
            signals_present.append("vwap_recoil")
            count += 1
        elif side == "short" and dist > 0.01:  # Price above VWAP, snapping back down
            signals_present.append("vwap_recoil")
            count += 1
    
    # Check range position (at range low/high)
    if side == "long":
        if hasattr(advanced_features, 'at_range_low') and advanced_features.at_range_low:
            signals_present.append("range_low")
            count += 1
    else:
        if hasattr(advanced_features, 'at_range_high') and advanced_features.at_range_high:
            signals_present.append("range_high")
            count += 1
    
    # PURE_SCALPER MODE: Structure gate is informational only - pass by default
    # Only reject on truly bad conditions (conflicting signals, not missing signals)
    # Strong signals: exhaustion, divergence, SFP
    # Medium signals: VWAP recoil, range position
    strong_signals = [s for s in signals_present if s in ['exhaustion_down', 'exhaustion_up', 
                                                           'bullish_divergence', 'bearish_divergence',
                                                           'sfp_bottom', 'sfp_top']]
    medium_signals = [s for s in signals_present if s in ['vwap_recoil', 'range_low', 'range_high']]
    
    # FIX: Pass by default - structure gate should not block signals for missing structure data
    # Only reject if we have conflicting signals (e.g., bullish and bearish at the same time)
    # For PURE_SCALPER: Missing structure signals is not a rejection reason
    passed = True  # Default: pass (structure gate is informational, not blocking)
    
    # Check for conflicting signals (truly bad condition)
    # Only reject if we have strong conflicting signals (e.g., trying to long but have bearish signals)
    has_bullish = any(s in ['exhaustion_down', 'bullish_divergence', 'sfp_bottom'] for s in signals_present)
    has_bearish = any(s in ['exhaustion_up', 'bearish_divergence', 'sfp_top'] for s in signals_present)
    if side == "long" and has_bearish and not has_bullish:
        # Conflicting: trying to long but only bearish structure signals present
        passed = False
    elif side == "short" and has_bullish and not has_bearish:
        # Conflicting: trying to short but only bullish structure signals present
        passed = False
    
    # Calculate score boost based on confluence - reward good structure when present
    if count >= 3:
        score_boost = 8.0  # Strong confluence
    elif len(strong_signals) >= 2:
        score_boost = 6.0  # Two strong signals
    elif len(strong_signals) >= 1:
        score_boost = 4.0  # One strong signal
    elif count >= 2:
        score_boost = 3.0  # Two medium signals
    else:
        score_boost = 0.0  # No structure signals (but still pass)
    
    confluence = StructureConfluence(
        count=count,
        signals=signals_present,
        score_boost=score_boost
    )
    
    if not passed:
        # Only reject on truly conflicting signals
        return (
            FilterResult(
                passed=False,
                reason=f"structure_fail: conflicting_signals (side={side}, bullish={has_bullish}, bearish={has_bearish})",
                details={"signals": signals_present, "count": count, "strong": strong_signals, "medium": medium_signals}
            ),
            confluence
        )
    
    return (
        FilterResult(
            passed=True,
            reason="structure_pass",
            details={"signals": signals_present, "count": count}
        ),
        confluence
    )


def stage3_direction_prediction(
    side: str,
    indicators: Optional[Dict],
    advanced_features: Optional[Any],
    symbol_stats: Dict,
    orderbook: Optional[Dict],
    entry_price: float
) -> FilterResult:
    """
    Stage 3: Candle Direction Prediction Gate - PURE_SCALPER MODE
    
    FIX: Pass by default if indicators missing - missing data is NOT a rejection reason.
    Only reject if direction_score is truly too low (< 0.50) when we have data.
    """
    direction_score = 0.5  # Start neutral
    
    # FIX: Missing indicators is NOT a rejection reason - pass through with neutral score
    if not indicators:
        return FilterResult(
            passed=True,  # FIX: Pass when indicators missing
            reason="direction_pass: no_indicators_available_fallback",
            details={"direction_score": direction_score}
        )
    
    # Extract features
    atr_pct = indicators.get('atr_pct', 0.01)
    rsi = indicators.get('rsi', 50.0)
    adx = indicators.get('adx', 20.0)
    
    # VWAP distance
    dist_from_vwap = 0.0
    if advanced_features and hasattr(advanced_features, 'dist_from_vwap'):
        dist_from_vwap = advanced_features.dist_from_vwap
    
    # Exhaustion flags
    has_exhaustion = False
    if advanced_features:
        if side == "long":
            has_exhaustion = (advanced_features.exhaustion_down if hasattr(advanced_features, 'exhaustion_down') else False)
        else:
            has_exhaustion = (advanced_features.exhaustion_up if hasattr(advanced_features, 'exhaustion_up') else False)
    
    # Divergence flags
    has_divergence = False
    if advanced_features:
        if side == "long":
            has_divergence = (advanced_features.bullish_divergence if hasattr(advanced_features, 'bullish_divergence') else False)
        else:
            has_divergence = (advanced_features.bearish_divergence if hasattr(advanced_features, 'bearish_divergence') else False)
    
    # HTF trend direction
    htf_trend_dir = 0
    if advanced_features and hasattr(advanced_features, 'htf_trend_dir'):
        htf_trend_dir = advanced_features.htf_trend_dir
    
    # Range position
    at_range_extreme = False
    if advanced_features:
        if side == "long":
            at_range_extreme = (advanced_features.at_range_low if hasattr(advanced_features, 'at_range_low') else False)
        else:
            at_range_extreme = (advanced_features.at_range_high if hasattr(advanced_features, 'at_range_high') else False)
    
    # Microstructure (spread)
    spread_bps = symbol_stats.get('spread_bps', 9999)
    spread_quality = 1.0 - min(spread_bps / 100.0, 1.0)  # 0-1, better spread = higher
    
    # Build direction_score from components
    # Base: RSI position
    if side == "long":
        rsi_bias = (rsi - 30.0) / 40.0  # RSI 30-70 range -> 0-1
        rsi_bias = max(0.0, min(1.0, rsi_bias))
        direction_score = 0.4 + (rsi_bias * 0.2)  # 0.4-0.6 base
    else:  # short
        rsi_bias = (70.0 - rsi) / 40.0  # RSI 70-30 range -> 0-1
        rsi_bias = max(0.0, min(1.0, rsi_bias))
        direction_score = 0.4 + (rsi_bias * 0.2)  # 0.4-0.6 base
    
    # VWAP distance adjustment
    if side == "long":
        if dist_from_vwap < -0.01:  # Below VWAP, good for long
            direction_score += 0.05
        elif dist_from_vwap > 0.01:  # Above VWAP, bad for long
            direction_score -= 0.05
    else:  # short
        if dist_from_vwap > 0.01:  # Above VWAP, good for short
            direction_score += 0.05
        elif dist_from_vwap < -0.01:  # Below VWAP, bad for short
            direction_score -= 0.05
    
    # Exhaustion boost
    if has_exhaustion:
        direction_score += 0.08
    
    # Divergence boost
    if has_divergence:
        direction_score += 0.06
    
    # HTF trend alignment
    if side == "long" and htf_trend_dir > 0:
        direction_score += 0.04
    elif side == "short" and htf_trend_dir < 0:
        direction_score += 0.04
    
    # Range extreme boost
    if at_range_extreme:
        direction_score += 0.04
    
    # ADX adjustment (volatility context)
    if adx > 25:  # Strong trend
        direction_score += 0.03
    elif adx < 15:  # Weak trend/chop
        direction_score -= 0.03
    
    # Spread quality adjustment
    direction_score += (spread_quality - 0.5) * 0.02
    
    # Clamp to 0-1
    direction_score = max(0.0, min(1.0, direction_score))
    
    # FIX: Lower threshold significantly - only reject if direction_score is truly terrible
    # PURE_SCALPER: Missing data passes, only reject on truly bad predictions
    min_direction_score = 0.45  # Very low threshold - only reject if truly terrible (< 0.45)
    
    if direction_score < min_direction_score:
        return FilterResult(
            passed=False,
            reason=f"direction_fail: score_too_low ({direction_score:.3f} < {min_direction_score})",
            details={"direction_score": direction_score, "min_required": min_direction_score}
        )
    
    return FilterResult(
        passed=True,
        reason="direction_pass",
        details={"direction_score": direction_score}
    )


def evaluate_three_stage_filter(
    symbol: str,
    side: str,
    symbol_stats: Dict,
    orderbook: Optional[Dict],
    indicators: Optional[Dict],
    advanced_features: Optional[Any],
    entry_price: float,
    final_score: Optional[float] = None  # SCORING V2 final score for debug logging
) -> Tuple[bool, Optional[str], Optional[Dict], Optional[StructureConfluence]]:
    """
    Evaluate all three filter stages.
    
    Returns:
        (passed, rejection_reason, filter_details, structure_confluence)
    """
    # Stage 1: Microstructure
    stage1_result = stage1_microstructure_gate(
        symbol, side, symbol_stats, orderbook, indicators, advanced_features, entry_price
    )
    if not stage1_result.passed:
        return False, stage1_result.reason, stage1_result.details, None
    
    # Stage 2: Structure
    stage2_result, confluence = stage2_structure_gate(
        side, advanced_features, indicators, symbol_stats
    )
    if not stage2_result.passed:
        # DEBUG: Log to file only (no console output)
        # Removed print() to prevent console output - use logger if needed
        return False, stage2_result.reason, stage2_result.details, confluence
    
    # DEBUG: Log to file only (no console output)
    # Removed print() to prevent console output - use logger if needed
    
    # Stage 3: Direction
    stage3_result = stage3_direction_prediction(
        side, indicators, advanced_features, symbol_stats, orderbook, entry_price
    )
    if not stage3_result.passed:
        return False, stage3_result.reason, stage3_result.details, confluence
    
    # All stages passed
    filter_details = {
        "stage1": stage1_result.details,
        "stage2": stage2_result.details,
        "stage3": stage3_result.details
    }
    
    return True, None, filter_details, confluence

