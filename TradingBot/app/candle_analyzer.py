"""
Candle Pattern & Local Structure Analyzer
==========================================

Analyzes candlestick patterns and local market structure to enhance signal quality.
Non-invasive: Returns metrics dict that can boost or penalize signal strength.

Features:
- Local high/low detection (last N candles)
- Breakout confirmation (breaking structure)
- Wick analysis (rejection signals)
- Body strength analysis (momentum confirmation)
- Volume + candle correlation
"""

from typing import List, Dict, Optional, Tuple
import math


class CandleAnalyzer:
    """Analyze candlestick patterns and local structure."""
    
    # Candle structure: [timestamp, open, high, low, close, volume]
    TIMESTAMP_IDX = 0
    OPEN_IDX = 1
    HIGH_IDX = 2
    LOW_IDX = 3
    CLOSE_IDX = 4
    VOLUME_IDX = 5
    
    @staticmethod
    def analyze_local_structure(candles: List[List[float]]) -> Dict[str, any]:
        """
        Analyze local market structure and candlestick patterns.
        
        Args:
            candles: List of [timestamp, open, high, low, close, volume]
        
        Returns:
            Dict with analysis results:
            - breaks_local_high: Breaking above recent local high
            - breaks_local_low: Breaking below recent local low
            - upper_wick_ratio: Ratio of upper wick to body (rejection measure)
            - lower_wick_ratio: Ratio of lower wick to body (rejection measure)
            - body_strength: Body size as ratio of total range (momentum measure)
            - close_position: Where close is in the range (0-1, 0=low, 1=high)
            - has_long_upper_wick: Long upper wick (rejection at resistance)
            - has_long_lower_wick: Long lower wick (rejection at support)
            - body_size_pct: Body as % of candle range
            - is_strong_momentum: Large body near extremity
            - local_support: Local support level (from recent lows)
            - local_resistance: Local resistance level (from recent highs)
            - volatility_expansion: Current candle > average volatility
        """
        
        if not candles or len(candles) < 2:
            return CandleAnalyzer._empty_analysis()
        
        current = candles[-1]
        
        # Ensure we have enough history for local structure
        lookback = min(5, len(candles) - 1)  # Last 5 candles (excluding current)
        recent = candles[-lookback-1:-1] if lookback > 0 else []
        
        analysis = {}
        
        # ===== LOCAL STRUCTURE =====
        if recent:
            local_highs = [c[CandleAnalyzer.HIGH_IDX] for c in recent]
            local_lows = [c[CandleAnalyzer.LOW_IDX] for c in recent]
            local_high = max(local_highs)
            local_low = min(local_lows)
        else:
            local_high = current[CandleAnalyzer.HIGH_IDX]
            local_low = current[CandleAnalyzer.LOW_IDX]
        
        current_close = current[CandleAnalyzer.CLOSE_IDX]
        current_high = current[CandleAnalyzer.HIGH_IDX]
        current_low = current[CandleAnalyzer.LOW_IDX]
        current_open = current[CandleAnalyzer.OPEN_IDX]
        
        # Breakout detection
        analysis['breaks_local_high'] = current_close > local_high
        analysis['breaks_local_low'] = current_close < local_low
        analysis['local_support'] = float(local_low)
        analysis['local_resistance'] = float(local_high)
        
        # ===== WICK ANALYSIS =====
        upper_wick = current_high - current_close  # How much above close
        lower_wick = current_close - current_low   # How much below close
        body_size = abs(current_close - current_open)
        total_range = current_high - current_low
        
        # Wick ratios (as percentage of body, clamped at 5.0)
        if body_size > 0:
            analysis['upper_wick_ratio'] = min(upper_wick / body_size, 5.0)
            analysis['lower_wick_ratio'] = min(lower_wick / body_size, 5.0)
        else:
            analysis['upper_wick_ratio'] = 0.0
            analysis['lower_wick_ratio'] = 0.0
        
        # Long wick detection (ratio > 1.5 means wick is 1.5x the body)
        analysis['has_long_upper_wick'] = analysis['upper_wick_ratio'] > 1.5
        analysis['has_long_lower_wick'] = analysis['lower_wick_ratio'] > 1.5
        
        # ===== BODY ANALYSIS =====
        if total_range > 0:
            analysis['body_strength'] = body_size / total_range
            analysis['body_size_pct'] = (body_size / total_range) * 100.0
            # Close position: 0=at low, 1=at high
            analysis['close_position'] = (current_close - current_low) / total_range
        else:
            analysis['body_strength'] = 0.0
            analysis['body_size_pct'] = 0.0
            analysis['close_position'] = 0.5
        
        # Strong momentum: Large body AND close at extreme
        large_body = analysis['body_size_pct'] > 60.0  # >60% of range
        close_at_high = analysis['close_position'] > 0.75
        close_at_low = analysis['close_position'] < 0.25
        analysis['is_strong_momentum'] = large_body and (close_at_high or close_at_low)
        
        # ===== VOLATILITY EXPANSION =====
        if recent and len(recent) > 0:
            recent_ranges = [c[CandleAnalyzer.HIGH_IDX] - c[CandleAnalyzer.LOW_IDX] for c in recent]
            avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
            current_range = total_range
            analysis['volatility_expansion'] = current_range > (avg_range * 1.2) if avg_range > 0 else False
        else:
            analysis['volatility_expansion'] = False
        
        return analysis
    
    @staticmethod
    def _empty_analysis() -> Dict[str, any]:
        """Return neutral analysis when data is insufficient."""
        return {
            'breaks_local_high': False,
            'breaks_local_low': False,
            'upper_wick_ratio': 0.0,
            'lower_wick_ratio': 0.0,
            'body_strength': 0.5,
            'close_position': 0.5,
            'has_long_upper_wick': False,
            'has_long_lower_wick': False,
            'body_size_pct': 50.0,
            'is_strong_momentum': False,
            'local_support': 0.0,
            'local_resistance': 0.0,
            'volatility_expansion': False,
        }
    
    @staticmethod
    def calculate_entry_boost(
        candle_analysis: Dict,
        side: str
    ) -> Tuple[float, str]:
        """
        Calculate signal strength boost based on candle structure.
        
        Args:
            candle_analysis: Analysis dict from analyze_local_structure()
            side: 'long' or 'short'
        
        Returns:
            (boost_value, reason) where boost_value is -0.20 to +0.15
        """
        
        if not candle_analysis:
            return 0.0, "no_analysis"
        
        boost = 0.0
        reason_parts = []
        
        # ===== BREAKOUT CONFIRMATION (Most bullish) =====
        if side == 'long' and candle_analysis.get('breaks_local_high'):
            boost += 0.10
            reason_parts.append("breakout_long")
        elif side == 'short' and candle_analysis.get('breaks_local_low'):
            boost += 0.10
            reason_parts.append("breakout_short")
        
        # ===== STRONG MOMENTUM (Bullish) =====
        if candle_analysis.get('is_strong_momentum'):
            if side == 'long' and candle_analysis.get('close_position', 0) > 0.75:
                boost += 0.05
                reason_parts.append("strong_momentum_long")
            elif side == 'short' and candle_analysis.get('close_position', 0) < 0.25:
                boost += 0.05
                reason_parts.append("strong_momentum_short")
        
        # ===== REJECTION WICKS (Can boost or penalize) =====
        # Long upper wick on LONG = rejection at resistance = bearish
        if side == 'long' and candle_analysis.get('has_long_upper_wick'):
            boost -= 0.08
            reason_parts.append("rejection_wick_long")
        
        # Long lower wick on SHORT = rejection at support = bullish
        if side == 'short' and candle_analysis.get('has_long_lower_wick'):
            boost -= 0.08
            reason_parts.append("rejection_wick_short")
        
        # ===== VOLATILITY EXPANSION (Confirmation) =====
        if candle_analysis.get('volatility_expansion'):
            boost += 0.05
            reason_parts.append("vol_expansion")
        
        # Clamp to reasonable range
        boost = max(-0.20, min(0.15, boost))
        
        reason = "_".join(reason_parts) if reason_parts else "neutral"
        
        return boost, reason
    
    @staticmethod
    def get_rejection_signal_strength(candle_analysis: Dict) -> float:
        """
        Detect rejection signals for mean reversion entries.
        
        Returns: 0.0-1.0 confidence in rejection setup
        """
        if not candle_analysis:
            return 0.0
        
        # Rejection requires: long wick + small body
        has_long_upper_wick = candle_analysis.get('has_long_upper_wick', False)
        has_long_lower_wick = candle_analysis.get('has_long_lower_wick', False)
        body_size_pct = candle_analysis.get('body_size_pct', 50.0)
        
        # Strong rejection: long wick + small body
        if (has_long_upper_wick or has_long_lower_wick) and body_size_pct < 40.0:
            return 0.8
        
        # Moderate rejection: long wick + medium body
        if (has_long_upper_wick or has_long_lower_wick) and body_size_pct < 60.0:
            return 0.5
        
        return 0.0
    
    # ===================================================================
    # PHASE 2: ADVANCED CANDLESTICK PATTERN RECOGNITION
    # ===================================================================
    
    @staticmethod
    def detect_hammer(candles: List[List[float]], current_idx: int = -1) -> Dict[str, any]:
        """
        Detect Hammer pattern (bullish reversal at support).
        
        Characteristics:
        - Small body (< 30% of range)
        - Long lower wick (> 2x body size)
        - Close near top of candle
        - Appears at support levels
        
        Returns: {detected: bool, confidence: 0-1, type: 'hammer'/'inverted_hammer'/'none'}
        """
        if not candles or len(candles) < 1:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        candle = candles[current_idx]
        open_p = candle[CandleAnalyzer.OPEN_IDX]
        high = candle[CandleAnalyzer.HIGH_IDX]
        low = candle[CandleAnalyzer.LOW_IDX]
        close = candle[CandleAnalyzer.CLOSE_IDX]
        
        total_range = high - low
        if total_range <= 0:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        body_size = abs(close - open_p)
        upper_wick = high - close
        lower_wick = close - low
        body_ratio = body_size / total_range
        
        # Hammer: small body, long lower wick, close near top
        is_hammer = (
            body_ratio < 0.30 and  # Small body
            lower_wick > body_size * 2.0 and  # Long lower wick (>2x body)
            (high - close) < (body_size * 0.5)  # Close near top
        )
        
        # Inverted Hammer: small body, long upper wick, close near bottom
        is_inverted_hammer = (
            body_ratio < 0.30 and  # Small body
            upper_wick > body_size * 2.0 and  # Long upper wick (>2x body)
            (close - low) < (body_size * 0.5)  # Close near bottom
        )
        
        if is_hammer:
            confidence = min(lower_wick / (body_size * 3.0), 1.0)  # Confidence based on wick length
            return {'detected': True, 'confidence': confidence, 'type': 'hammer'}
        elif is_inverted_hammer:
            confidence = min(upper_wick / (body_size * 3.0), 1.0)
            return {'detected': True, 'confidence': confidence, 'type': 'inverted_hammer'}
        
        return {'detected': False, 'confidence': 0.0, 'type': 'none'}
    
    @staticmethod
    def detect_engulfing(candles: List[List[float]], current_idx: int = -1) -> Dict[str, any]:
        """
        Detect Engulfing pattern (momentum confirmation).
        
        Characteristics:
        - Current candle body completely engulfs previous candle
        - Same direction (both bullish or bearish)
        - Usually higher volume on current candle
        - Confirms trend continuation
        
        Returns: {detected: bool, confidence: 0-1, type: 'bullish_engulfing'/'bearish_engulfing'/'none'}
        """
        if not candles or len(candles) < 2:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        current = candles[current_idx]
        previous = candles[current_idx - 1]
        
        curr_open = current[CandleAnalyzer.OPEN_IDX]
        curr_close = current[CandleAnalyzer.CLOSE_IDX]
        curr_body_size = abs(curr_close - curr_open)
        
        prev_open = previous[CandleAnalyzer.OPEN_IDX]
        prev_close = previous[CandleAnalyzer.CLOSE_IDX]
        prev_body_min = min(prev_open, prev_close)
        prev_body_max = max(prev_open, prev_close)
        prev_body_size = abs(prev_close - prev_open)
        
        # Bullish engulfing: current body envelops previous
        is_bullish_engulfing = (
            curr_open < prev_body_min and  # Opens below previous low
            curr_close > prev_body_max and  # Closes above previous high
            curr_body_size > prev_body_size  # Larger body
        )
        
        # Bearish engulfing: opposite
        is_bearish_engulfing = (
            curr_open > prev_body_max and  # Opens above previous high
            curr_close < prev_body_min and  # Closes below previous low
            curr_body_size > prev_body_size  # Larger body
        )
        
        if is_bullish_engulfing:
            confidence = min(curr_body_size / max(prev_body_size, 1.0), 1.0)
            return {'detected': True, 'confidence': confidence, 'type': 'bullish_engulfing'}
        elif is_bearish_engulfing:
            confidence = min(curr_body_size / max(prev_body_size, 1.0), 1.0)
            return {'detected': True, 'confidence': confidence, 'type': 'bearish_engulfing'}
        
        return {'detected': False, 'confidence': 0.0, 'type': 'none'}
    
    @staticmethod
    def detect_doji(candles: List[List[float]], current_idx: int = -1) -> Dict[str, any]:
        """
        Detect Doji pattern (indecision/exhaustion).
        
        Characteristics:
        - Tiny body (open ≈ close, < 5% of range)
        - Can have long wicks both sides or one very long
        - Shows market indecision or exhaustion
        - Often appears at resistance/support
        
        Returns: {detected: bool, confidence: 0-1, type: 'doji'/'dragonfly'/'gravestone'/'none'}
        """
        if not candles or len(candles) < 1:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        candle = candles[current_idx]
        open_p = candle[CandleAnalyzer.OPEN_IDX]
        high = candle[CandleAnalyzer.HIGH_IDX]
        low = candle[CandleAnalyzer.LOW_IDX]
        close = candle[CandleAnalyzer.CLOSE_IDX]
        
        total_range = high - low
        if total_range <= 0:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        body_size = abs(close - open_p)
        body_ratio = body_size / total_range
        
        # Basic doji: tiny body (< 5% of range)
        if body_ratio > 0.05:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        # Analyze wicks
        upper_wick = high - max(open_p, close)
        lower_wick = min(open_p, close) - low
        
        # Dragonfly: long lower wick, short upper wick (reversal at bottom)
        is_dragonfly = lower_wick > upper_wick * 2.0
        
        # Gravestone: long upper wick, short lower wick (reversal at top)
        is_gravestone = upper_wick > lower_wick * 2.0
        
        confidence = 0.7  # Doji is always fairly high confidence for indecision
        
        if is_dragonfly:
            return {'detected': True, 'confidence': confidence, 'type': 'dragonfly'}
        elif is_gravestone:
            return {'detected': True, 'confidence': confidence, 'type': 'gravestone'}
        else:
            # Standard doji (wicks both sides similar)
            return {'detected': True, 'confidence': confidence, 'type': 'doji'}
    
    @staticmethod
    def detect_pin_bar(candles: List[List[float]], current_idx: int = -1) -> Dict[str, any]:
        """
        Detect Pin Bar pattern (strong rejection).
        
        Characteristics:
        - Small body on one end
        - Long wick opposite direction
        - Close at extreme (top or bottom)
        - Clear rejection from price level
        - Strong reversal signal
        
        Returns: {detected: bool, confidence: 0-1, type: 'bullish_pin'/'bearish_pin'/'none'}
        """
        if not candles or len(candles) < 1:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        candle = candles[current_idx]
        open_p = candle[CandleAnalyzer.OPEN_IDX]
        high = candle[CandleAnalyzer.HIGH_IDX]
        low = candle[CandleAnalyzer.LOW_IDX]
        close = candle[CandleAnalyzer.CLOSE_IDX]
        
        total_range = high - low
        if total_range <= 0:
            return {'detected': False, 'confidence': 0.0, 'type': 'none'}
        
        body_size = abs(close - open_p)
        body_ratio = body_size / total_range
        
        # Pin bar requires: small body + long wick opposite side
        upper_wick = high - max(open_p, close)
        lower_wick = min(open_p, close) - low
        
        # Bullish pin: long lower wick, small body, close near top
        is_bullish_pin = (
            body_ratio < 0.30 and
            lower_wick > body_size * 2.0 and
            close > open_p and  # Close higher than open (bullish)
            (high - close) < (body_size * 0.5)  # Close near top
        )
        
        # Bearish pin: long upper wick, small body, close near bottom
        is_bearish_pin = (
            body_ratio < 0.30 and
            upper_wick > body_size * 2.0 and
            close < open_p and  # Close lower than open (bearish)
            (close - low) < (body_size * 0.5)  # Close near bottom
        )
        
        if is_bullish_pin:
            confidence = min(lower_wick / (body_size * 3.0), 1.0)
            return {'detected': True, 'confidence': confidence, 'type': 'bullish_pin'}
        elif is_bearish_pin:
            confidence = min(upper_wick / (body_size * 3.0), 1.0)
            return {'detected': True, 'confidence': confidence, 'type': 'bearish_pin'}
        
        return {'detected': False, 'confidence': 0.0, 'type': 'none'}
    
    @staticmethod
    def calculate_pattern_boost(
        candles: List[List[float]],
        side: str,
        local_support: float = 0.0,
        local_resistance: float = 0.0
    ) -> tuple:
        """
        Detect all Phase 2 patterns and calculate combined boost.
        
        Args:
            candles: OHLCV candles list
            side: 'long' or 'short'
            local_support: Support level for pattern context
            local_resistance: Resistance level for pattern context
        
        Returns:
            (boost_value, pattern_reason) where boost is -0.20 to +0.20
        """
        if not candles or len(candles) < 2:
            return 0.0, "no_pattern_data"
        
        boost = 0.0
        patterns_detected = []
        
        # ===== HAMMER / INVERTED HAMMER =====
        hammer = CandleAnalyzer.detect_hammer(candles)
        if hammer['detected']:
            if hammer['type'] == 'hammer':
                # Hammer at support = bullish reversal
                if side == 'long':
                    boost += 0.15 * hammer['confidence']
                    patterns_detected.append(f"hammer({hammer['confidence']:.1%})")
            elif hammer['type'] == 'inverted_hammer':
                # Inverted hammer at resistance = bearish
                if side == 'short':
                    boost += 0.10 * hammer['confidence']
                    patterns_detected.append(f"inv_hammer({hammer['confidence']:.1%})")
        
        # ===== ENGULFING =====
        engulfing = CandleAnalyzer.detect_engulfing(candles)
        if engulfing['detected']:
            if engulfing['type'] == 'bullish_engulfing':
                if side == 'long':
                    boost += 0.10 * engulfing['confidence']
                    patterns_detected.append(f"bull_engulf({engulfing['confidence']:.1%})")
            elif engulfing['type'] == 'bearish_engulfing':
                if side == 'short':
                    boost += 0.10 * engulfing['confidence']
                    patterns_detected.append(f"bear_engulf({engulfing['confidence']:.1%})")
        
        # ===== DOJI (CAUTION SIGNAL) =====
        doji = CandleAnalyzer.detect_doji(candles)
        if doji['detected']:
            # Doji = indecision, reduce confidence
            boost -= 0.08 * doji['confidence']
            patterns_detected.append(f"{doji['type']}({doji['confidence']:.1%})")
        
        # ===== PIN BAR =====
        pin = CandleAnalyzer.detect_pin_bar(candles)
        if pin['detected']:
            if pin['type'] == 'bullish_pin' and side == 'long':
                boost += 0.10 * pin['confidence']
                patterns_detected.append(f"bull_pin({pin['confidence']:.1%})")
            elif pin['type'] == 'bearish_pin' and side == 'short':
                boost += 0.10 * pin['confidence']
                patterns_detected.append(f"bear_pin({pin['confidence']:.1%})")
        
        # Clamp to reasonable range
        boost = max(-0.20, min(0.20, boost))
        
        reason = "_".join(patterns_detected) if patterns_detected else "no_pattern"
        
        return boost, reason
    
    # ===================================================================
    # PHASE 3: MEAN REVERSION SPECIFIC PATTERNS
    # ===================================================================
    
    @staticmethod
    def detect_rejection_zone(candles: List[List[float]], current_idx: int = -1) -> Dict[str, any]:
        """
        Detect Rejection Zone pattern (mean reversion setup).
        
        Characteristics:
        - Wick penetrates support/resistance
        - But closes back inside (rejection)
        - Shows strong rejection from that level
        - Perfect setup for mean reversion
        
        Returns: {detected: bool, confidence: 0-1, zone_type: 'support'/'resistance'/'none'}
        """
        if not candles or len(candles) < 2:
            return {'detected': False, 'confidence': 0.0, 'zone_type': 'none'}
        
        current = candles[current_idx]
        previous = candles[current_idx - 1]
        
        curr_high = current[CandleAnalyzer.HIGH_IDX]
        curr_low = current[CandleAnalyzer.LOW_IDX]
        curr_close = current[CandleAnalyzer.CLOSE_IDX]
        curr_open = current[CandleAnalyzer.OPEN_IDX]
        
        prev_high = previous[CandleAnalyzer.HIGH_IDX]
        prev_low = previous[CandleAnalyzer.LOW_IDX]
        
        body_size = abs(curr_close - curr_open)
        total_range = curr_high - curr_low
        
        if total_range <= 0 or body_size <= 0:
            return {'detected': False, 'confidence': 0.0, 'zone_type': 'none'}
        
        # Get local support/resistance from last 3 candles
        lookback_candles = candles[-3:] if len(candles) >= 3 else candles
        local_highs = [c[CandleAnalyzer.HIGH_IDX] for c in lookback_candles]
        local_lows = [c[CandleAnalyzer.LOW_IDX] for c in lookback_candles]
        local_resistance = max(local_highs)
        local_support = min(local_lows)
        
        # REJECTION AT RESISTANCE: High penetrates but closes below resistance
        # This shows strong rejection from above (bearish for longs, bullish for shorts)
        is_resistance_rejection = (
            curr_high > local_resistance and  # Wick penetrates resistance
            curr_close < local_resistance and  # But closes below
            body_size / total_range < 0.60  # Not a strong continuation (not full engulfing)
        )
        
        # REJECTION AT SUPPORT: Low penetrates but closes above support
        # This shows strong rejection from below (bullish for longs, bearish for shorts)
        is_support_rejection = (
            curr_low < local_support and  # Wick penetrates support
            curr_close > local_support and  # But closes above
            body_size / total_range < 0.60  # Not a strong continuation
        )
        
        if is_resistance_rejection:
            # Strength based on: how far below resistance did it close?
            distance_below = (local_resistance - curr_close) / total_range
            confidence = min(0.9, 0.5 + (distance_below * 0.4))  # 0.5-0.9
            return {'detected': True, 'confidence': confidence, 'zone_type': 'resistance'}
        
        elif is_support_rejection:
            # Strength based on: how far above support did it close?
            distance_above = (curr_close - local_support) / total_range
            confidence = min(0.9, 0.5 + (distance_above * 0.4))  # 0.5-0.9
            return {'detected': True, 'confidence': confidence, 'zone_type': 'support'}
        
        return {'detected': False, 'confidence': 0.0, 'zone_type': 'none'}
    
    @staticmethod
    def detect_bounce_confirmation(candles: List[List[float]], current_idx: int = -1) -> Dict[str, any]:
        """
        Detect Bounce Confirmation pattern (mean reversion confirmation).
        
        Characteristics:
        - Current candle direction opposite from previous
        - Volume > average (confirmation)
        - Moving away from support/resistance zone
        - Confirms rejection zone was legitimate
        
        Returns: {detected: bool, confidence: 0-1, bounce_type: 'bounce_up'/'bounce_down'/'none'}
        """
        if not candles or len(candles) < 2:
            return {'detected': False, 'confidence': 0.0, 'bounce_type': 'none'}
        
        current = candles[current_idx]
        previous = candles[current_idx - 1]
        
        curr_close = current[CandleAnalyzer.CLOSE_IDX]
        curr_open = current[CandleAnalyzer.OPEN_IDX]
        curr_volume = current[CandleAnalyzer.VOLUME_IDX]
        
        prev_close = previous[CandleAnalyzer.CLOSE_IDX]
        prev_open = previous[CandleAnalyzer.OPEN_IDX]
        prev_volume = previous[CandleAnalyzer.VOLUME_IDX]
        
        # Direction of candles
        curr_bullish = curr_close > curr_open
        curr_bearish = curr_close < curr_open
        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open
        
        # Volume confirmation (current > previous)
        volume_confirmed = curr_volume > prev_volume if prev_volume > 0 else curr_volume > 0
        
        # Average volume from last 5 candles
        if len(candles) >= 5:
            recent_volumes = [c[CandleAnalyzer.VOLUME_IDX] for c in candles[-5:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
            volume_strong = curr_volume > avg_volume * 0.9  # Within 90% of avg
        else:
            volume_strong = curr_volume > prev_volume * 0.8  # At least 80% of previous
        
        # Bounce UP: current bullish, previous bearish, volume confirmation
        is_bounce_up = (
            curr_bullish and prev_bearish and volume_strong and
            curr_close > prev_close  # Moving away from low
        )
        
        # Bounce DOWN: current bearish, previous bullish, volume confirmation
        is_bounce_down = (
            curr_bearish and prev_bullish and volume_strong and
            curr_close < prev_close  # Moving away from high
        )
        
        if is_bounce_up:
            # Confidence based on: how strong is the close above previous close?
            if prev_close > 0:
                bounce_strength = (curr_close - prev_close) / prev_close
                confidence = min(0.85, 0.5 + (bounce_strength * 10))  # 0.5-0.85
            else:
                confidence = 0.7
            return {'detected': True, 'confidence': confidence, 'bounce_type': 'bounce_up'}
        
        elif is_bounce_down:
            # Confidence based on: how strong is the close below previous close?
            if prev_close > 0:
                bounce_strength = (prev_close - curr_close) / prev_close
                confidence = min(0.85, 0.5 + (bounce_strength * 10))  # 0.5-0.85
            else:
                confidence = 0.7
            return {'detected': True, 'confidence': confidence, 'bounce_type': 'bounce_down'}
        
        return {'detected': False, 'confidence': 0.0, 'bounce_type': 'none'}
    
    @staticmethod
    def calculate_phase3_boost(
        candles: List[List[float]],
        side: str
    ) -> tuple:
        """
        Detect Phase 3 (Mean Reversion) patterns and calculate boost.
        
        Args:
            candles: OHLCV candles list
            side: 'long' or 'short'
        
        Returns:
            (boost_value, pattern_reason) where boost is -0.15 to +0.25
        """
        if not candles or len(candles) < 2:
            return 0.0, "no_phase3_data"
        
        boost = 0.0
        patterns_detected = []
        
        # ===== REJECTION ZONE (Highest confidence MR setup) =====
        rejection = CandleAnalyzer.detect_rejection_zone(candles)
        if rejection['detected']:
            # Rejection at resistance = bearish = SHORT advantage
            if rejection['zone_type'] == 'resistance' and side == 'short':
                boost += 0.15 * rejection['confidence']
                patterns_detected.append(f"reject_resist({rejection['confidence']:.1%})")
            # Rejection at support = bullish = LONG advantage
            elif rejection['zone_type'] == 'support' and side == 'long':
                boost += 0.15 * rejection['confidence']
                patterns_detected.append(f"reject_support({rejection['confidence']:.1%})")
        
        # ===== BOUNCE CONFIRMATION (Validates rejection zone) =====
        bounce = CandleAnalyzer.detect_bounce_confirmation(candles)
        if bounce['detected']:
            # Bounce UP (after rejection) = bullish confirmation
            if bounce['bounce_type'] == 'bounce_up' and side == 'long':
                boost += 0.10 * bounce['confidence']
                patterns_detected.append(f"bounce_up({bounce['confidence']:.1%})")
            # Bounce DOWN (after rejection) = bearish confirmation
            elif bounce['bounce_type'] == 'bounce_down' and side == 'short':
                boost += 0.10 * bounce['confidence']
                patterns_detected.append(f"bounce_down({bounce['confidence']:.1%})")
        
        # Clamp to reasonable range
        boost = max(-0.15, min(0.25, boost))
        
        reason = "_".join(patterns_detected) if patterns_detected else "no_phase3_pattern"
        
        return boost, reason

