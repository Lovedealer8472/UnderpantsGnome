"""
INDICATOR-BASED TRADE FILTER
=============================

Based on analysis of 675,420 trades, blocks the WORST combinations:

BLOCK LIST (Proven Losers):
- SHORT + RSI > 65 + ADX < 20: Lost $1.3M total, 35-40% WR (no trend to fade)
- Any trade with RSI 60-70 + ADX > 30: 36% WR

BOOST LIST (Proven Winners):
- LONG + RSI 40-60 + ADX < 20: 57.9% WR
- Any trade with RSI 40-50: 49% WR, +3.5% avg

This is a HARD FILTER that runs BEFORE ML scoring.
"""

import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# =============================================================================
# FILTER RULES (Based on 675K trade analysis)
# =============================================================================

# BLOCK: These combinations are PROVEN LOSERS (refined with ADX conditions)
BLOCK_RULES = [
    {
        'name': 'SHORT_RSI_HIGH',
        'description': 'SHORT when RSI > 65 AND ADX < 20 (no trend to fade)',
        'side': 'short',
        'rsi_min': 65,
        'rsi_max': 100,
        'adx_min': None,
        'adx_max': 20,  # Only block when ADX < 20 (no strong trend)
    },
    {
        'name': 'HIGH_RSI_HIGH_ADX',
        'description': 'RSI 60-75 + ADX > 30 (36% WR)',
        'side': None,  # Any side
        'rsi_min': 60,
        'rsi_max': 75,
        'adx_min': 30,
        'adx_max': None,
    },
]

# BOOST: These combinations get score bonus
BOOST_RULES = [
    {
        'name': 'LONG_MID_RSI_LOW_ADX',
        'description': 'LONG + RSI 40-60 + ADX < 20 (57.9% WR)',
        'side': 'long',
        'rsi_min': 40,
        'rsi_max': 60,
        'adx_min': None,
        'adx_max': 20,
        'boost': 15,  # +15 points to score
    },
    {
        'name': 'OPTIMAL_RSI_ZONE',
        'description': 'RSI 40-50 (49% WR, best zone)',
        'side': None,
        'rsi_min': 40,
        'rsi_max': 50,
        'adx_min': None,
        'adx_max': None,
        'boost': 10,  # +10 points
    },
    {
        'name': 'LOW_ADX_RANGING',
        'description': 'ADX < 15 (50.4% WR, ranging market)',
        'side': None,
        'rsi_min': None,
        'rsi_max': None,
        'adx_min': None,
        'adx_max': 15,
        'boost': 8,  # +8 points
    },
]


class IndicatorFilter:
    """
    Filters trades based on proven indicator combinations.
    
    Usage:
        filter = IndicatorFilter()
        
        # Check if trade should be blocked
        blocked, reason = filter.should_block(side='short', rsi=72, adx=25)
        if blocked:
            return  # Don't enter this trade
        
        # Get score boost for good combinations
        boost = filter.get_score_boost(side='long', rsi=45, adx=18)
        final_score = base_score + boost
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats = {
            'blocked': 0,
            'boosted': 0,
            'passed': 0,
            'block_reasons': {}
        }
        logger.info(f"[INDICATOR_FILTER] Initialized (enabled={enabled})")
    
    def should_block(
        self, 
        side: str, 
        rsi: Optional[float] = None, 
        adx: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if this trade should be BLOCKED.
        
        Returns: (should_block: bool, reason: str)
        """
        if not self.enabled:
            return False, ""
        
        if rsi is None and adx is None:
            return False, ""  # No indicator data, can't filter
        
        side_lower = side.lower() if side else ""
        
        for rule in BLOCK_RULES:
            # Check side match
            if rule['side'] and rule['side'] != side_lower:
                continue
            
            # Check RSI range
            rsi_match = True
            if rsi is not None:
                if rule['rsi_min'] is not None and rsi < rule['rsi_min']:
                    rsi_match = False
                if rule['rsi_max'] is not None and rsi > rule['rsi_max']:
                    rsi_match = False
            elif rule['rsi_min'] is not None or rule['rsi_max'] is not None:
                rsi_match = False  # Rule requires RSI but we don't have it
            
            # Check ADX range
            adx_match = True
            if adx is not None:
                if rule['adx_min'] is not None and adx < rule['adx_min']:
                    adx_match = False
                if rule['adx_max'] is not None and adx > rule['adx_max']:
                    adx_match = False
            elif rule['adx_min'] is not None or rule['adx_max'] is not None:
                adx_match = False  # Rule requires ADX but we don't have it
            
            # All conditions must match for block
            if rsi_match and adx_match:
                reason = f"BLOCKED: {rule['name']} - {rule['description']}"
                self.stats['blocked'] += 1
                self.stats['block_reasons'][rule['name']] = \
                    self.stats['block_reasons'].get(rule['name'], 0) + 1
                
                logger.warning(f"[INDICATOR_FILTER] {reason} | side={side} rsi={rsi:.1f} adx={adx:.1f}")
                return True, reason
        
        self.stats['passed'] += 1
        return False, ""
    
    def get_score_boost(
        self, 
        side: str, 
        rsi: Optional[float] = None, 
        adx: Optional[float] = None
    ) -> float:
        """
        Get score boost for good indicator combinations.
        
        Returns: boost amount (0 if no boost applies)
        """
        if not self.enabled:
            return 0.0
        
        if rsi is None and adx is None:
            return 0.0
        
        side_lower = side.lower() if side else ""
        total_boost = 0.0
        
        for rule in BOOST_RULES:
            # Check side match
            if rule['side'] and rule['side'] != side_lower:
                continue
            
            # Check RSI range
            rsi_match = True
            if rsi is not None:
                if rule['rsi_min'] is not None and rsi < rule['rsi_min']:
                    rsi_match = False
                if rule['rsi_max'] is not None and rsi > rule['rsi_max']:
                    rsi_match = False
            elif rule['rsi_min'] is not None or rule['rsi_max'] is not None:
                rsi_match = False
            
            # Check ADX range
            adx_match = True
            if adx is not None:
                if rule['adx_min'] is not None and adx < rule['adx_min']:
                    adx_match = False
                if rule['adx_max'] is not None and adx > rule['adx_max']:
                    adx_match = False
            elif rule['adx_min'] is not None or rule['adx_max'] is not None:
                adx_match = False
            
            if rsi_match and adx_match:
                total_boost += rule['boost']
                self.stats['boosted'] += 1
        
        return total_boost
    
    def evaluate(
        self, 
        side: str, 
        rsi: Optional[float], 
        adx: Optional[float],
        base_score: float
    ) -> Tuple[bool, float, str]:
        """
        Full evaluation: check block and apply boost.
        
        Returns: (should_enter: bool, adjusted_score: float, reason: str)
        """
        # First check if blocked
        blocked, block_reason = self.should_block(side, rsi, adx)
        if blocked:
            return False, 0.0, block_reason
        
        # Apply boost
        boost = self.get_score_boost(side, rsi, adx)
        adjusted_score = base_score + boost
        
        reason = ""
        if boost > 0:
            reason = f"BOOST +{boost:.0f} (score: {base_score:.0f} -> {adjusted_score:.0f})"
            rsi_str = f"{rsi:.1f}" if rsi else "N/A"
            adx_str = f"{adx:.1f}" if adx else "N/A"
            logger.info(f"[INDICATOR_FILTER] {reason} | side={side} rsi={rsi_str} adx={adx_str}")
        
        return True, adjusted_score, reason
    
    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return self.stats.copy()
    
    def get_status_line(self) -> str:
        """Get status for logging."""
        return (
            f"[INDICATOR_FILTER] "
            f"Blocked: {self.stats['blocked']} | "
            f"Boosted: {self.stats['boosted']} | "
            f"Passed: {self.stats['passed']}"
        )


# =============================================================================
# SINGLETON
# =============================================================================

_filter_instance: Optional[IndicatorFilter] = None

def get_indicator_filter() -> IndicatorFilter:
    """Get or create singleton IndicatorFilter."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = IndicatorFilter(enabled=True)
    return _filter_instance


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INDICATOR FILTER TEST")
    print("=" * 70)
    
    f = IndicatorFilter()
    
    # Test BLOCK cases
    print("\n--- BLOCK TESTS ---")
    
    # SHORT + HIGH RSI (should block)
    blocked, reason = f.should_block(side='short', rsi=72, adx=25)
    print(f"SHORT RSI=72 ADX=25: blocked={blocked} ({reason})")
    
    # HIGH RSI + HIGH ADX (should block)
    blocked, reason = f.should_block(side='long', rsi=68, adx=35)
    print(f"LONG RSI=68 ADX=35: blocked={blocked} ({reason})")
    
    # Good trade (should NOT block)
    blocked, reason = f.should_block(side='long', rsi=45, adx=15)
    print(f"LONG RSI=45 ADX=15: blocked={blocked} ({reason})")
    
    # Test BOOST cases
    print("\n--- BOOST TESTS ---")
    
    # LONG + MID RSI + LOW ADX (best combo)
    boost = f.get_score_boost(side='long', rsi=50, adx=18)
    print(f"LONG RSI=50 ADX=18: boost={boost}")
    
    # Optimal RSI zone
    boost = f.get_score_boost(side='short', rsi=45, adx=25)
    print(f"SHORT RSI=45 ADX=25: boost={boost}")
    
    # Full evaluation
    print("\n--- FULL EVALUATION ---")
    
    can_enter, score, reason = f.evaluate(side='long', rsi=48, adx=12, base_score=55)
    print(f"LONG RSI=48 ADX=12 base=55: enter={can_enter} score={score} ({reason})")
    
    can_enter, score, reason = f.evaluate(side='short', rsi=75, adx=20, base_score=65)
    print(f"SHORT RSI=75 ADX=20 base=65: enter={can_enter} score={score} ({reason})")
    
    print(f"\n{f.get_status_line()}")
    print("\n[DONE]")

