"""
Deterministic Ghost Candidate Generator
Fixed distribution for consistent scoring across iterations.
"""

from typing import List, Tuple
from app.engine.scoring_v2 import RawFactors


def get_deterministic_ghost_candidates() -> List[Tuple[str, RawFactors]]:
    """
    Generate fixed ghost candidates for stable percentile scoring.
    Same distribution every time for deterministic backtests.
    
    Returns:
        List of (symbol, RawFactors) tuples
    """
    ghosts = []
    
    # Fixed distribution: 20 ghosts with spread of qualities
    # This ensures percentile scoring works even with single candidate
    
    # 1. The "Unicorn" Ghost (Sets the ceiling) - Top 5%
    ghosts.append(('GHOST_UNICORN', RawFactors(
        momentum_factor=0.95, trend_factor=0.95, volatility_factor=0.90,
        liquidity_factor=0.95, spread_factor=0.95, volume_factor=0.95, depth_factor=0.95
    )))
    
    # 2. The "Elite" Ghosts (High bar) - Top 10-15%
    for i in range(2):
        ghosts.append((f'GHOST_ELITE_{i}', RawFactors(
            momentum_factor=0.80 + i*0.05, trend_factor=0.80 + i*0.05, volatility_factor=0.75,
            liquidity_factor=0.85, spread_factor=0.85, volume_factor=0.85, depth_factor=0.85
        )))
    
    # 3. The "Strong" Ghosts (Mid-high range) - Top 30-40%
    for i in range(5):
        val = 0.60 + (i * 0.04)  # 0.60 to 0.76
        ghosts.append((f'GHOST_STRONG_{i}', RawFactors(
            momentum_factor=val, trend_factor=val, volatility_factor=val*0.9,
            liquidity_factor=val, spread_factor=val, volume_factor=val, depth_factor=val
        )))
    
    # 4. The "Average" Ghosts (Mid range) - Top 50-70%
    for i in range(7):
        val = 0.40 + (i * 0.03)  # 0.40 to 0.58
        ghosts.append((f'GHOST_AVG_{i}', RawFactors(
            momentum_factor=val, trend_factor=val, volatility_factor=val,
            liquidity_factor=val, spread_factor=val, volume_factor=val, depth_factor=val
        )))
    
    # 5. The "Noise" Ghosts (Low quality) - Bottom 30%
    for i in range(5):
        val = 0.15 + (i * 0.03)  # 0.15 to 0.27
        ghosts.append((f'GHOST_NOISE_{i}', RawFactors(
            momentum_factor=val, trend_factor=val, volatility_factor=val,
            liquidity_factor=val, spread_factor=val, volume_factor=val, depth_factor=val
        )))
    
    return ghosts

