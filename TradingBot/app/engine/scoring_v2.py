"""
Scoring v2 - Percentile-based scoring system with enforced rarity.

This module replaces the old linear normalized scoring system with a rank-based
percentile approach that ensures realistic score distributions and enforces
scarcity for elite/unicorn signals.

Key features:
- Score distribution centers around ~50
- Only top 10-15% score above ~75
- Only top 1-2% score above ~90 (unicorns)
- Percentile-based normalization prevents score inflation
- Strict gating before scoring (liquidity, spread, volatility)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from ..config import (
    IDEAL_SPREAD_BPS, MAX_SPREAD_BPS, MIN_24H_VOLUME_USDT,
    BTC_TREND_THRESHOLD
)
from ..indicators import calculate_trend_alignment, calculate_momentum_from_rsi


@dataclass
class RawFactors:
    """Raw factors (0-1 normalized) for a single symbol."""
    momentum_factor: float = 0.0
    trend_factor: float = 0.0
    volatility_factor: float = 0.0
    liquidity_factor: float = 0.0
    spread_factor: float = 0.0  # Inverted: tighter = higher
    volume_factor: float = 0.0
    depth_factor: float = 0.0


@dataclass
class ScoreResult:
    """Result of scoring a signal with v2 system."""
    composite_score: float  # 0-100
    rank_percentile: float  # 0-1 (0.0 = top, 1.0 = bottom)
    tier: str  # "unicorn", "elite", "strong", "average", "weak"
    raw_factors: RawFactors
    percentile_factors: Dict[str, float]  # Percentile-transformed factors
    # PHASE 3: Explainability - detailed breakdown
    factor_contributions: Optional[Dict[str, float]] = field(default=None)  # Contribution of each factor to final score
    weights_used: Optional[Dict[str, float]] = field(default=None)  # Weights used for this score
    
    def explain(self) -> str:
        """Generate human-readable explanation of the score."""
        if not self.factor_contributions or not self.weights_used:
            return f"Score: {self.composite_score:.1f} (Tier: {self.tier}, Rank: {self.rank_percentile*100:.1f}%)"
        
        # Sort factors by contribution
        sorted_factors = sorted(
            self.factor_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        parts = [f"Score: {self.composite_score:.1f} ({self.tier}, top {self.rank_percentile*100:.1f}%)"]
        parts.append("Contributions:")
        for factor, contrib in sorted_factors[:5]:  # Top 5 factors
            weight = self.weights_used.get(factor, 0.0)
            parts.append(f"  {factor}: {contrib:.1f} (weight: {weight:.2f})")
        
        return "\n".join(parts)


class ScoringV2:
    """
    Scoring v2 system with percentile-based normalization and rarity enforcement.
    
    PHASE 3: Evolvable weights and thresholds.
    """
    
    # Default factor weights (used if not provided in __init__)
    DEFAULT_WEIGHTS = {
        'momentum_factor': 0.20,
        'trend_factor': 0.20,
        'volatility_factor': 0.15,
        'liquidity_factor': 0.15,
        'volume_factor': 0.15,
        'spread_factor': 0.10,
        'depth_factor': 0.05
    }
    
    # Default tier thresholds (by rank percentile)
    DEFAULT_TIER_THRESHOLDS = {
        'unicorn': 0.02,    # Top 2%
        'elite': 0.10,      # Top 10%
        'strong': 0.30,     # Top 30%
        'average': 0.60,     # Top 60%
        # Below 60% = "weak"
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        tier_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Scoring v2 system.
        
        Args:
            weights: Evolvable factor weights (if None, uses defaults)
            tier_thresholds: Evolvable tier thresholds (if None, uses defaults)
        """
        # PHASE 3: Evolvable weights
        if weights is not None:
            # Normalize weights to sum to 1.0
            total = sum(weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in weights.items()}
            else:
                self.weights = self.DEFAULT_WEIGHTS.copy()
        else:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        
        # PHASE 3: Evolvable tier thresholds
        if tier_thresholds is not None:
            self.tier_thresholds = tier_thresholds.copy()
        else:
            self.tier_thresholds = self.DEFAULT_TIER_THRESHOLDS.copy()
        
        self._distribution_cache = None
        self._candidate_scores = []  # For building distribution
    
    def compute_raw_factors(
        self,
        symbol: str,
        side: str,
        symbol_stats: Dict,
        indicators: Optional[Dict] = None,
        orderbook: Optional[Dict] = None,
        btc_trend: Optional[float] = None,
        entry_price: float = 0.0
    ) -> RawFactors:
        """
        Compute raw factors (0-1 normalized) for a symbol.
        
        Args:
            symbol: Trading symbol
            side: Position side (long/short)
            symbol_stats: Symbol statistics dict
            indicators: Technical indicators dict
            orderbook: Orderbook data
            btc_trend: BTC trend percentage
            entry_price: Entry price
            
        Returns:
            RawFactors object with normalized factors
        """
        # Extract data
        spread_bps = symbol_stats.get('spread_bps', 9999)
        volume_24h = symbol_stats.get('vol_quote', 0)
        pct_change_24h = symbol_stats.get('pct_change_24h', 0.0)
        
        # Extract indicators
        rsi = indicators.get('rsi') if indicators else None
        ema20 = indicators.get('ema20') if indicators else None
        ema50 = indicators.get('ema50') if indicators else None
        ema100 = indicators.get('ema100') if indicators else None
        atr_pct = indicators.get('atr_pct') if indicators else None
        
        # 1. Momentum factor (0-1)
        momentum_factor = self._compute_momentum_factor(rsi, pct_change_24h, side)
        
        # 2. Trend factor (0-1)
        trend_factor = self._compute_trend_factor(ema20, ema50, ema100, pct_change_24h, btc_trend, side)
        
        # 3. Volatility factor (0-1) - inverted if needed (lower volatility = higher score for some strategies)
        volatility_factor = self._compute_volatility_factor(atr_pct, entry_price, volume_24h)
        
        # 4. Liquidity factor (0-1)
        liquidity_factor = self._compute_liquidity_factor(volume_24h)
        
        # 5. Spread factor (0-1) - inverted (tighter spread = higher score)
        spread_factor = self._compute_spread_factor(spread_bps)
        
        # 6. Volume factor (0-1)
        volume_factor = self._compute_volume_factor(volume_24h)
        
        # 7. Depth factor (0-1)
        depth_factor = self._compute_depth_factor(orderbook, entry_price)
        
        return RawFactors(
            momentum_factor=momentum_factor,
            trend_factor=trend_factor,
            volatility_factor=volatility_factor,
            liquidity_factor=liquidity_factor,
            spread_factor=spread_factor,
            volume_factor=volume_factor,
            depth_factor=depth_factor
        )
    
    def _compute_momentum_factor(self, rsi: Optional[float], pct_change_24h: float, side: str) -> float:
        """Compute momentum factor (0-1)."""
        if rsi is not None:
            # Use RSI-based momentum
            if side == "long":
                # Long: RSI oversold or rising from oversold
                if rsi < 30:
                    return 0.8 + (30 - rsi) / 30.0 * 0.2  # 0.8-1.0
                elif rsi < 50:
                    return 0.5 + (50 - rsi) / 20.0 * 0.3  # 0.5-0.8
                else:
                    return max(0.0, 0.5 - (rsi - 50) / 50.0 * 0.5)  # 0.0-0.5
            else:  # short
                # Short: RSI overbought or falling from overbought
                if rsi > 70:
                    return 0.8 + (rsi - 70) / 30.0 * 0.2  # 0.8-1.0
                elif rsi > 50:
                    return 0.5 + (rsi - 50) / 20.0 * 0.3  # 0.5-0.8
                else:
                    return max(0.0, 0.5 - (50 - rsi) / 50.0 * 0.5)  # 0.0-0.5
        
        # Fallback: Use 24h price change
        momentum_abs = abs(pct_change_24h)
        if momentum_abs >= 3.0:
            return 1.0
        elif momentum_abs >= 2.0:
            return 0.7 + (momentum_abs - 2.0) / 1.0 * 0.3  # 0.7-1.0
        elif momentum_abs >= 1.0:
            return 0.4 + (momentum_abs - 1.0) / 1.0 * 0.3  # 0.4-0.7
        elif momentum_abs >= 0.5:
            return 0.2 + (momentum_abs - 0.5) / 0.5 * 0.2  # 0.2-0.4
        else:
            return momentum_abs / 0.5 * 0.2  # 0.0-0.2
    
    def _compute_trend_factor(
        self,
        ema20: Optional[float],
        ema50: Optional[float],
        ema100: Optional[float],
        pct_change_24h: float,
        btc_trend: Optional[float],
        side: str
    ) -> float:
        """Compute trend factor (0-1)."""
        # Try EMA alignment first
        if ema20 is not None and ema50 is not None and ema100 is not None:
            alignment = calculate_trend_alignment(ema20, ema50, ema100, side)
            return alignment  # Already 0-1
        
        # Fallback: Use 24h trend vs BTC trend
        if btc_trend is not None and abs(btc_trend) > BTC_TREND_THRESHOLD:
            # Check if symbol moves in same direction as BTC
            if (pct_change_24h > 0 and btc_trend > 0) or (pct_change_24h < 0 and btc_trend < 0):
                correlation_strength = min(abs(pct_change_24h) / 3.0, 1.0)
                return 0.5 + (correlation_strength * 0.5)  # 0.5-1.0
            else:
                return 0.3
        
        # No clear trend
        if abs(pct_change_24h) < 0.3:
            return 0.3
        elif abs(pct_change_24h) < 0.8:
            return 0.4
        elif abs(pct_change_24h) < 1.5:
            return 0.6
        else:
            return 0.8
    
    def _compute_volatility_factor(self, atr_pct: Optional[float], entry_price: float, volume_24h: float) -> float:
        """Compute volatility factor (0-1)."""
        if atr_pct is not None:
            # Optimal volatility: 0.5-2.0% ATR
            if 0.5 <= atr_pct <= 2.0:
                return 1.0
            elif atr_pct < 0.5:
                # Too low volatility
                return atr_pct / 0.5 * 0.7  # 0.0-0.7
            elif atr_pct <= 4.0:
                # Moderate volatility
                return 1.0 - (atr_pct - 2.0) / 2.0 * 0.5  # 0.5-1.0
            else:
                # Too high volatility
                return max(0.0, 0.5 - (atr_pct - 4.0) / 4.0 * 0.5)  # 0.0-0.5
        
        # Fallback: Use volume as proxy
        vol_log = math.log10(volume_24h + 1.0)
        return min(vol_log / 8.0, 1.0)
    
    def _compute_liquidity_factor(self, volume_24h: float) -> float:
        """Compute liquidity factor (0-1)."""
        if volume_24h < MIN_24H_VOLUME_USDT:
            return 0.0
        
        vol_log = math.log10(volume_24h + 1.0)
        # Normalize: 6.0 (1M) = 0.5, 8.0 (100M) = 1.0
        if vol_log < 6.0:
            return vol_log / 6.0 * 0.5  # 0.0-0.5
        elif vol_log <= 8.0:
            return 0.5 + (vol_log - 6.0) / 2.0 * 0.5  # 0.5-1.0
        else:
            return 1.0
    
    def _compute_spread_factor(self, spread_bps: float) -> float:
        """Compute spread factor (0-1) - inverted (tighter = higher)."""
        if spread_bps <= IDEAL_SPREAD_BPS:
            return 1.0
        elif spread_bps <= MAX_SPREAD_BPS:
            # Linear decay from IDEAL to MAX
            return 1.0 - (spread_bps - IDEAL_SPREAD_BPS) / (MAX_SPREAD_BPS - IDEAL_SPREAD_BPS)
        else:
            # Below zero for very wide spreads
            return max(0.0, 1.0 - (spread_bps - MAX_SPREAD_BPS) / MAX_SPREAD_BPS)
    
    def _compute_volume_factor(self, volume_24h: float) -> float:
        """Compute volume factor (0-1)."""
        if volume_24h < MIN_24H_VOLUME_USDT:
            return 0.0
        
        vol_log = math.log10(volume_24h + 1.0)
        return min(vol_log / 8.0, 1.0)
    
    def _compute_depth_factor(self, orderbook: Optional[Dict], entry_price: float) -> float:
        """Compute depth factor (0-1)."""
        if not orderbook or entry_price <= 0:
            return 0.5  # Neutral if no data
        
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.5
            
            # Calculate depth at 1% from mid price
            mid_price = (bids[0][0] + asks[0][0]) / 2.0
            depth_threshold = mid_price * 0.01
            
            bid_depth = sum(price * qty for price, qty in bids if mid_price - price <= depth_threshold)
            ask_depth = sum(price * qty for price, qty in asks if price - mid_price <= depth_threshold)
            
            # Normalize depth (1% of mid price * reasonable size)
            target_depth = mid_price * 10000  # Example: $100k depth for $10 asset
            depth_score = min((bid_depth + ask_depth) / target_depth, 1.0)
            
            return depth_score
        except Exception:
            return 0.5  # Neutral on error
    
    def build_distribution(self, candidates: List[Tuple[str, RawFactors]]) -> Dict[str, List[float]]:
        """
        Build distribution of raw factors across all candidates.
        
        Args:
            candidates: List of (symbol, RawFactors) tuples
            
        Returns:
            Dict mapping factor names to lists of values
        """
        distribution = defaultdict(list)
        
        for symbol, factors in candidates:
            distribution['momentum_factor'].append(factors.momentum_factor)
            distribution['trend_factor'].append(factors.trend_factor)
            distribution['volatility_factor'].append(factors.volatility_factor)
            distribution['liquidity_factor'].append(factors.liquidity_factor)
            distribution['spread_factor'].append(factors.spread_factor)
            distribution['volume_factor'].append(factors.volume_factor)
            distribution['depth_factor'].append(factors.depth_factor)
        
        return dict(distribution)
    
    def compute_percentile_factors(
        self,
        raw_factors: RawFactors,
        distribution: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Convert raw factors to percentile factors (0-1).
        
        Args:
            raw_factors: Raw factors for a symbol
            distribution: Distribution of all factors across candidates
            
        Returns:
            Dict mapping factor names to percentile values (0-1)
        """
        percentile_factors = {}
        
        factor_map = {
            'momentum_factor': raw_factors.momentum_factor,
            'trend_factor': raw_factors.trend_factor,
            'volatility_factor': raw_factors.volatility_factor,
            'liquidity_factor': raw_factors.liquidity_factor,
            'spread_factor': raw_factors.spread_factor,
            'volume_factor': raw_factors.volume_factor,
            'depth_factor': raw_factors.depth_factor
        }
        
        for factor_name, raw_value in factor_map.items():
            if factor_name not in distribution or not distribution[factor_name]:
                percentile_factors[factor_name] = 0.5  # Neutral if no distribution
                continue
            
            values = sorted(distribution[factor_name])
            total = len(values)
            
            # Calculate percentile (0.0 = bottom, 1.0 = top)
            rank = sum(1 for v in values if v < raw_value)
            percentile = rank / total if total > 0 else 0.5
            
            percentile_factors[factor_name] = percentile
        
        return percentile_factors
    
    def compute_composite_score(
        self,
        raw_factors: RawFactors,
        percentile_factors: Dict[str, float]
    ) -> float:
        """
        Compute composite score (0-100) from percentile factors.
        
        Args:
            raw_factors: Raw factors (for reference)
            percentile_factors: Percentile-transformed factors
            
        Returns:
            Composite score (0-100)
        """
        # Weighted sum of percentile factors
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor_name, weight in self.weights.items():
            if factor_name in percentile_factors:
                weighted_sum += weight * percentile_factors[factor_name]
                total_weight += weight
        
        # Normalize to 0-100
        if total_weight > 0:
            composite = (weighted_sum / total_weight) * 100.0
        else:
            composite = 50.0  # Neutral
        
        return max(0.0, min(100.0, composite))
    
    def rank_scores(self, scores: List[Tuple[str, float]]) -> Dict[str, Tuple[float, float]]:
        """
        Rank scores and compute percentiles.
        
        Args:
            scores: List of (symbol, composite_score) tuples
            
        Returns:
            Dict mapping symbol to (rank_percentile, composite_score)
        """
        if not scores:
            return {}
        
        # Sort by score descending
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        total = len(sorted_scores)
        
        result = {}
        for rank, (symbol, score) in enumerate(sorted_scores):
            rank_percentile = rank / total if total > 0 else 0.0
            result[symbol] = (rank_percentile, score)
        
        return result
    
    def classify_score(self, composite_score: float, rank_percentile: float) -> str:
        """
        Classify score into tier based on rank percentile.
        
        Args:
            composite_score: Composite score (0-100)
            rank_percentile: Rank percentile (0.0 = top, 1.0 = bottom)
            
        Returns:
            Tier string: "unicorn", "elite", "strong", "average", "weak"
        """
        if rank_percentile <= self.tier_thresholds['unicorn']:
            return "unicorn"
        elif rank_percentile <= self.tier_thresholds['elite']:
            return "elite"
        elif rank_percentile <= self.tier_thresholds['strong']:
            return "strong"
        elif rank_percentile <= self.tier_thresholds['average']:
            return "average"
        else:
            return "weak"
    
    def score_candidates(
        self,
        candidates: List[Tuple[str, RawFactors]]
    ) -> Dict[str, ScoreResult]:
        """
        Score all candidates and return results with rankings.
        
        Args:
            candidates: List of (symbol, RawFactors) tuples
            
        Returns:
            Dict mapping symbol to ScoreResult
        """
        if not candidates:
            return {}
        
        # Build distribution
        distribution = self.build_distribution(candidates)
        
        # Compute composite scores for all candidates
        candidate_scores = []
        results = {}
        
        for symbol, raw_factors in candidates:
            # Convert to percentiles
            percentile_factors = self.compute_percentile_factors(raw_factors, distribution)
            
            # Compute composite score
            composite_score = self.compute_composite_score(raw_factors, percentile_factors)
            
            candidate_scores.append((symbol, composite_score))
            results[symbol] = {
                'composite_score': composite_score,
                'raw_factors': raw_factors,
                'percentile_factors': percentile_factors
            }
        
        # Rank all scores
        rankings = self.rank_scores(candidate_scores)
        
        # CRITICAL: Enforce unicorn rarity - only top 2% can score 90+
        # This prevents "everything is 90+" fantasy and ensures realistic score distribution
        UNICORN_PERCENTILE_THRESHOLD = self.tier_thresholds['unicorn']  # 0.02 (top 2%)
        UNICORN_SCORE_CAP = 90.0  # Hard cap: only unicorns can score 90+
        
        # Classify and build final results
        final_results = {}
        for symbol, (rank_percentile, composite_score) in rankings.items():
            tier = self.classify_score(composite_score, rank_percentile)
            
            # PHASE 3: Calculate factor contributions for explainability
            raw_factors = results[symbol]['raw_factors']
            percentile_factors = results[symbol]['percentile_factors']
            factor_contributions = {}
            for factor_name, weight in self.weights.items():
                if factor_name in percentile_factors:
                    contrib = weight * percentile_factors[factor_name] * 100.0
                    factor_contributions[factor_name] = contrib
            
            # Enforce unicorn rarity: cap score to 89.99 if not in top 2%
            if composite_score >= UNICORN_SCORE_CAP and rank_percentile > UNICORN_PERCENTILE_THRESHOLD:
                # Score would be 90+ but signal is not in top 2% -> cap it
                capped_score = 89.99
                from ..logger import get_logger
                logger = get_logger("ScoringV2")
                logger.debug(
                    f"Unicorn rarity enforced: {symbol} score capped from {composite_score:.2f} to {capped_score:.2f} "
                    f"(rank_percentile={rank_percentile:.4f} > {UNICORN_PERCENTILE_THRESHOLD})"
                )
                composite_score = capped_score
                # Reclassify tier after capping
                tier = self.classify_score(composite_score, rank_percentile)
            
            final_results[symbol] = ScoreResult(
                composite_score=composite_score,
                rank_percentile=rank_percentile,
                tier=tier,
                raw_factors=raw_factors,
                percentile_factors=percentile_factors,
                factor_contributions=factor_contributions,
                weights_used=self.weights.copy()
            )
        
        return final_results

