"""
Freshness-Based Scoring System

Philosophy: "The best trades are fresh momentum moves with confirmation, not extreme moves that have already run."

Three pillars:
1. Freshness (40 points) - How recently did momentum start?
2. Confirmation (40 points) - Is this sustainable or a spike?
3. Execution Quality (20 points) - Can we actually capture this?

Total Score: 0-100
- <70: Reject (stale or low quality)
- 70-85: Moderate momentum (good trades)
- 85-95: Strong momentum (great trades)
- 95+: Extreme exhaustion (rare, should be <5% of signals)
"""

import time
import math
from typing import Dict, Optional, Tuple
from collections import defaultdict


class FreshnessScorer:
    """
    Elegant scoring system based on freshness, confirmation, and execution.
    Naturally filters exhaustion without explicit penalties.
    """
    
    def __init__(self):
        # Track momentum age per symbol
        self.symbol_history = {}  # symbol -> {momentum_start, last_check, last_pct}
        
        # Track performance per symbol (for reliability scoring)
        self.symbol_performance = {}  # symbol -> {trades, wins, win_rate}
        
        # Track volatility history per symbol
        self.volatility_history = defaultdict(list)  # symbol -> [pct_changes]
        self.volatility_window = 100  # Track last 100 observations
    
    def score_signal(
        self,
        symbol: str,
        pct_change_24h: float,
        volume_24h: float,
        spread_bps: float,
        orderbook: Optional[Dict] = None,
        indicators: Optional[Dict] = None,
        latency_ms: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Calculate comprehensive signal score.
        
        Args:
            symbol: Trading symbol
            pct_change_24h: 24h price change percentage
            volume_24h: 24h volume in quote currency
            spread_bps: Bid-ask spread in basis points
            orderbook: Optional orderbook data
            indicators: Optional technical indicators
            latency_ms: Current latency in milliseconds
        
        Returns:
            (total_score, component_breakdown)
        """
        
        # 1. FRESHNESS SCORE (0-40 points)
        momentum_age = self._get_momentum_age(symbol, pct_change_24h)
        freshness_score = self._calculate_freshness(pct_change_24h, momentum_age)
        
        # 2. CONFIRMATION SCORE (0-40 points)
        orderbook_imbalance = self._calculate_orderbook_imbalance(orderbook)
        price_acceleration = self._calculate_acceleration(symbol, pct_change_24h)
        volatility_percentile = self._calculate_volatility_percentile(symbol, pct_change_24h)
        
        confirmation_score = self._calculate_confirmation(
            volume_24h,
            orderbook_imbalance,
            price_acceleration,
            volatility_percentile
        )
        
        # 3. EXECUTION SCORE (0-20 points)
        depth_usd = self._calculate_orderbook_depth(orderbook)
        symbol_reliability = self.symbol_performance.get(symbol, {}).get('win_rate')
        
        execution_score = self._calculate_execution(
            spread_bps,
            depth_usd,
            latency_ms,
            symbol_reliability
        )
        
        # TOTAL SCORE
        total_score = freshness_score + confirmation_score + execution_score
        
        # Component breakdown for analysis
        components = {
            'freshness': round(freshness_score, 2),
            'confirmation': round(confirmation_score, 2),
            'execution': round(execution_score, 2),
            'total': round(total_score, 2),
            'momentum_age_min': round(momentum_age, 2) if momentum_age is not None else None,
            'orderbook_imbalance': round(orderbook_imbalance, 3) if orderbook_imbalance is not None else None,
            'price_acceleration': round(price_acceleration, 3) if price_acceleration is not None else None,
            'volatility_percentile': round(volatility_percentile, 1) if volatility_percentile is not None else None,
        }
        
        return total_score, components
    
    def _calculate_freshness(self, pct_change_24h: float, momentum_age_minutes: Optional[float]) -> float:
        """
        Freshness score based on momentum magnitude and age.
        Peak score at 4-6% moves, drops off for exhaustion.
        
        Returns: 0-40 points
        """
        abs_change = abs(pct_change_24h)
        
        # MAGNITUDE COMPONENT (0-30 points)
        # SCALPER OPTIMIZED: Shifted left to catch moves EARLY (1.5% - 4.0%)
        # Old: Peak at 4-6%. New: Peak at 2.5-5.0%.
        if abs_change < 0.5:
            # Noise: 0-2 points
            magnitude_score = abs_change * 4
        elif abs_change < 2.5:
            # Building FAST: 0.5% - 2.5%
            # 1.5% -> ~15 pts
            # 2.5% -> 25 pts
            magnitude_score = 2 + (abs_change - 0.5) * 11.5
        elif abs_change < 5.0:
            # Sweet spot: 2.5% - 5.0% = 25-30 points
            magnitude_score = 25 + (abs_change - 2.5) * 2.0
        elif abs_change < 8.0:
            # Extension: 5-8% = 30 -> 20 points (Start reducing)
            magnitude_score = 30 - (abs_change - 5.0) * 3.3
        else:
            # EXHAUSTION: 8%+ = Rapid drop
            magnitude_score = 20 - (abs_change - 8.0) * 2.0
            magnitude_score = max(0, magnitude_score)
        
        # AGE COMPONENT (0-10 points)
        # Reward fresh moves, penalize stale momentum
        if momentum_age_minutes is not None:
            if momentum_age_minutes < 5:
                age_score = 10  # Very fresh (<5 min)
            elif momentum_age_minutes < 15:
                age_score = 7   # Fresh (5-15 min)
            elif momentum_age_minutes < 30:
                age_score = 4   # Moderate (15-30 min)
            elif momentum_age_minutes < 60:
                age_score = 2   # Stale (30-60 min)
            else:
                age_score = 0   # Very stale (>60 min)
        else:
            # If we can't track age (e.g., bot just started), give benefit of doubt (Fresh)
            # Was 5 (Neutral), now 7 (Fresh) to help bot "wake up" faster
            age_score = 7
        
        return magnitude_score + age_score
    
    def _calculate_confirmation(
        self,
        volume_24h: float,
        orderbook_imbalance: Optional[float],
        price_acceleration: Optional[float],
        volatility_percentile: Optional[float]
    ) -> float:
        """
        Confirmation score based on volume, orderbook, and momentum quality.
        
        Returns: 0-40 points
        """
        
        # VOLUME CONFIRMATION (0-15 points)
        # Higher volume = more reliable (more generous thresholds)
        volume_log = math.log10(volume_24h + 1)
        if volume_log >= 8.0:  # $100M+
            volume_score = 15
        elif volume_log >= 7.5:  # $30M+
            volume_score = 14
        elif volume_log >= 7.0:  # $10M+
            volume_score = 12
        elif volume_log >= 6.5:  # $3M+
            volume_score = 10
        elif volume_log >= 6.0:  # $1M+
            volume_score = 8
        else:
            volume_score = 5  # Even low volume gets some credit
        
        # ORDERBOOK CONFIRMATION (0-10 points)
        # Strong imbalance in signal direction = confirmation
        if orderbook_imbalance is not None:
            abs_imbalance = abs(orderbook_imbalance)
            if abs_imbalance > 0.3:  # Strong imbalance
                imbalance_score = 10
            elif abs_imbalance > 0.2:  # Moderate
                imbalance_score = 8
            elif abs_imbalance > 0.1:  # Weak
                imbalance_score = 6
            else:
                imbalance_score = 4  # Neutral gets some credit
        else:
            imbalance_score = 5  # Neutral if unavailable
        
        # ACCELERATION CONFIRMATION (0-10 points)
        # Is momentum accelerating or decelerating?
        if price_acceleration is not None:
            if price_acceleration > 0.5:  # Strong acceleration
                accel_score = 10
            elif price_acceleration > 0.2:  # Moderate acceleration
                accel_score = 8
            elif price_acceleration > 0:  # Slight acceleration
                accel_score = 6
            elif price_acceleration > -0.2:  # Steady
                accel_score = 4
            elif price_acceleration > -0.5:  # Slight deceleration
                accel_score = 2
            else:  # Strong deceleration = exhaustion
                accel_score = 2
        else:
            accel_score = 7  # Neutral if unavailable (give benefit of doubt)
        
        # VOLATILITY CONTEXT (0-5 points)
        # Is this move unusual for this symbol?
        if volatility_percentile is not None:
            if volatility_percentile > 90:  # Very unusual (top 10%)
                vol_score = 5
            elif volatility_percentile > 75:  # Unusual (top 25%)
                vol_score = 4
            elif volatility_percentile > 60:  # Above average
                vol_score = 3
            elif volatility_percentile > 40:  # Average
                vol_score = 2
            else:  # Below average
                vol_score = 1
        else:
            vol_score = 2  # Neutral if unavailable
        
        return volume_score + imbalance_score + accel_score + vol_score
    
    def _calculate_execution(
        self,
        spread_bps: float,
        depth_usd: float,
        latency_ms: float,
        symbol_reliability: Optional[float]
    ) -> float:
        """
        Execution score based on market microstructure.
        
        Returns: 0-20 points
        """
        
        # SPREAD QUALITY (0-8 points) - More generous
        if spread_bps < 5:
            spread_score = 8  # Excellent
        elif spread_bps < 10:
            spread_score = 7  # Good
        elif spread_bps < 20:
            spread_score = 6  # Acceptable
        elif spread_bps < 50:
            spread_score = 4  # Fair
        else:
            spread_score = 2  # Wide but tradeable
        
        # DEPTH QUALITY (0-7 points) - More generous
        # Orderbook depth in USD
        if depth_usd > 50000:  # $50k+
            depth_score = 7
        elif depth_usd > 20000:  # $20k+
            depth_score = 6
        elif depth_usd > 10000:  # $10k+
            depth_score = 5
        elif depth_usd > 5000:  # $5k+
            depth_score = 4
        else:
            depth_score = 3  # Even thin books get credit
        
        # LATENCY PENALTY (0-3 points) - More forgiving
        if latency_ms < 50:
            latency_score = 3
        elif latency_ms < 100:
            latency_score = 3  # Still good
        elif latency_ms < 200:
            latency_score = 2
        else:
            latency_score = 1  # Acceptable for most trades
        
        # SYMBOL RELIABILITY (0-2 points) - More generous
        # Historical win rate on this symbol
        if symbol_reliability is not None:
            if symbol_reliability > 0.55:
                reliability_score = 2
            elif symbol_reliability > 0.45:
                reliability_score = 2  # Give benefit of doubt
            else:
                reliability_score = 1
        else:
            reliability_score = 2  # Assume good if unknown
        
        return spread_score + depth_score + latency_score + reliability_score
    
    def _get_momentum_age(self, symbol: str, pct_change: float) -> Optional[float]:
        """
        Track when momentum started for this symbol.
        Returns age in minutes, or None if unknown.
        """
        now = time.time()
        abs_change = abs(pct_change)
        
        if symbol not in self.symbol_history:
            self.symbol_history[symbol] = {
                'momentum_start': now if abs_change > 2.0 else None,
                'last_check': now,
                'last_pct': abs_change
            }
            return None
        
        hist = self.symbol_history[symbol]
        
        # Check if momentum just started (crossed 2% threshold)
        if abs_change > 2.0 and hist['last_pct'] <= 2.0:
            hist['momentum_start'] = now
        
        # Check if momentum ended (dropped below 2%)
        if abs_change <= 2.0:
            hist['momentum_start'] = None
        
        hist['last_check'] = now
        hist['last_pct'] = abs_change
        
        if hist['momentum_start'] is not None:
            age_seconds = now - hist['momentum_start']
            return age_seconds / 60.0  # Return age in minutes
        
        return None
    
    def _calculate_acceleration(self, symbol: str, pct_change: float) -> Optional[float]:
        """
        Calculate if momentum is accelerating or decelerating.
        Returns rate of change in momentum (%/minute).
        """
        if symbol not in self.symbol_history:
            return None
        
        hist = self.symbol_history[symbol]
        last_pct = hist.get('last_pct', 0)
        last_check = hist.get('last_check', time.time())
        
        time_delta = (time.time() - last_check) / 60.0  # minutes
        if time_delta < 0.1:  # Too soon
            return None
        
        pct_delta = abs(pct_change) - last_pct
        acceleration = pct_delta / time_delta  # %/minute
        
        return acceleration
    
    def _calculate_orderbook_imbalance(self, orderbook: Optional[Dict]) -> Optional[float]:
        """
        Calculate bid/ask imbalance from orderbook using multi-level analysis.
        Returns value between -1 (heavy selling pressure) and +1 (heavy buying pressure).
        
        ENHANCED: Based on order flow research - looks at:
        1. Top-of-book imbalance (immediate pressure)
        2. Deeper book imbalance (institutional support/resistance)
        3. Book thickness asymmetry (where is the real liquidity?)
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return None
        
        try:
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            if len(bids) < 3 or len(asks) < 3:
                return None
            
            # LAYER 1: Top-of-book (levels 1-3) - Immediate pressure
            # This is what tape readers watch most closely
            top_bid_vol = sum(float(b[1]) for b in bids[:3] if len(b) >= 2)
            top_ask_vol = sum(float(a[1]) for a in asks[:3] if len(a) >= 2)
            
            # LAYER 2: Mid-book (levels 4-10) - Institutional interest
            mid_bid_vol = sum(float(b[1]) for b in bids[3:10] if len(b) >= 2)
            mid_ask_vol = sum(float(a[1]) for a in asks[3:10] if len(a) >= 2)
            
            # LAYER 3: Deep book (levels 11-20) - Major support/resistance
            deep_bid_vol = sum(float(b[1]) for b in bids[10:20] if len(b) >= 2)
            deep_ask_vol = sum(float(a[1]) for a in asks[10:20] if len(a) >= 2)
            
            # Weighted imbalance: top-of-book matters most for scalping
            # 60% weight on top, 30% on mid, 10% on deep
            top_total = top_bid_vol + top_ask_vol
            mid_total = mid_bid_vol + mid_ask_vol
            deep_total = deep_bid_vol + deep_ask_vol
            
            top_imb = (top_bid_vol - top_ask_vol) / top_total if top_total > 0 else 0
            mid_imb = (mid_bid_vol - mid_ask_vol) / mid_total if mid_total > 0 else 0
            deep_imb = (deep_bid_vol - deep_ask_vol) / deep_total if deep_total > 0 else 0
            
            # Weighted combination
            weighted_imbalance = 0.6 * top_imb + 0.3 * mid_imb + 0.1 * deep_imb
            
            # BONUS: Detect "wall" patterns (large order at specific level)
            # A big bid wall just below = strong support = bullish
            # A big ask wall just above = strong resistance = bearish
            if len(bids) > 0 and len(asks) > 0:
                best_bid_size = float(bids[0][1])
                best_ask_size = float(asks[0][1])
                avg_size = (top_bid_vol + top_ask_vol) / 6 if (top_bid_vol + top_ask_vol) > 0 else 1
                
                # Wall detection: if top level is 3x average, it's a wall
                bid_wall = best_bid_size > (avg_size * 3)
                ask_wall = best_ask_size > (avg_size * 3)
                
                # Adjust imbalance for walls
                if bid_wall and not ask_wall:
                    weighted_imbalance = min(1.0, weighted_imbalance + 0.2)
                elif ask_wall and not bid_wall:
                    weighted_imbalance = max(-1.0, weighted_imbalance - 0.2)
            
            return weighted_imbalance
        except Exception:
            return None
    
    def _calculate_orderbook_depth(self, orderbook: Optional[Dict]) -> float:
        """
        Calculate total orderbook depth in USD from top 5 levels.
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return 0.0
        
        try:
            # Calculate USD value from top 5 levels
            bid_usd = sum(float(b[0]) * float(b[1]) for b in orderbook['bids'][:5] if len(b) >= 2)
            ask_usd = sum(float(a[0]) * float(a[1]) for a in orderbook['asks'][:5] if len(a) >= 2)
            
            total_depth = bid_usd + ask_usd
            return total_depth
        except Exception:
            return 0.0
    
    def _calculate_volatility_percentile(self, symbol: str, pct_change: float) -> Optional[float]:
        """
        Calculate what percentile this move is for this symbol.
        Returns 0-100 (e.g., 90 = top 10% of moves).
        """
        abs_change = abs(pct_change)
        
        # Add to history
        self.volatility_history[symbol].append(abs_change)
        
        # Keep only last N observations
        if len(self.volatility_history[symbol]) > self.volatility_window:
            self.volatility_history[symbol] = self.volatility_history[symbol][-self.volatility_window:]
        
        # Need at least 20 observations for meaningful percentile
        if len(self.volatility_history[symbol]) < 20:
            return None
        
        # Calculate percentile
        history = sorted(self.volatility_history[symbol])
        position = sum(1 for x in history if x < abs_change)
        percentile = (position / len(history)) * 100
        
        return percentile
    
    def update_symbol_performance(self, symbol: str, was_win: bool):
        """
        Update win rate tracking for a symbol.
        Called after trade closes.
        """
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                'trades': 0,
                'wins': 0,
                'win_rate': 0.5  # Start neutral
            }
        
        perf = self.symbol_performance[symbol]
        perf['trades'] += 1
        if was_win:
            perf['wins'] += 1
        
        perf['win_rate'] = perf['wins'] / perf['trades']
    
    def get_score_breakdown_str(self, components: Dict) -> str:
        """
        Format score components for logging.
        """
        return (
            f"Fresh:{components['freshness']:.1f} "
            f"Conf:{components['confirmation']:.1f} "
            f"Exec:{components['execution']:.1f} "
            f"= {components['total']:.1f}"
        )

