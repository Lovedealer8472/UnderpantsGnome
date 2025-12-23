"""
Trading Mode System - Clear boundaries between different trading strategies.

Modes:
1. AGGRESSIVE_SCALPING: Maximum aggression, ultra-low thresholds, all signals
2. MARKSMAN: Selective high-quality signals only, strict filters
3. BALANCED: Default balanced approach
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class TradingMode(Enum):
    """Trading mode enumeration."""
    AGGRESSIVE_SCALPING = "aggressive_scalping"
    MARKSMAN = "marksman"
    BALANCED = "balanced"

@dataclass
class ModeConfig:
    """Configuration for a trading mode."""
    name: str
    mode_type: TradingMode
    
    # Signal thresholds
    min_signal_score: int
    min_signal_strength: float
    hard_min_score: int
    marksman_min_score: Optional[int] = None
    
    # Entry filters
    min_momentum_pct: float
    min_spread_bps: float
    max_spread_bps: float
    min_volume_24h: float
    
    # Position management
    max_concurrent_positions: int
    
    # MARKSMAN mode flag
    enable_marksman: bool = False
    
    # Description
    description: str = ""

# Mode configurations
MODE_CONFIGS = {
    TradingMode.AGGRESSIVE_SCALPING: ModeConfig(
        name="Aggressive Scalping",
        mode_type=TradingMode.AGGRESSIVE_SCALPING,
        description="🔴 HARDCORE ADVISOR MODE: Maximum aggression, ultra-low thresholds, all signal types",
        # Ultra-low thresholds for maximum signal generation
        min_signal_score=40,  # Very low - generate many signals
        min_signal_strength=0.40,  # Very low
        hard_min_score=38,  # Very low floor
        marksman_min_score=None,  # Not used in aggressive mode
        # Relaxed entry filters
        min_momentum_pct=0.3,  # Very low momentum required
        min_spread_bps=0.0,  # Allow any spread
        max_spread_bps=150.0,  # Allow very wide spreads
        min_volume_24h=500_000,  # Very low volume requirement ($500K)
        # Maximum positions
        max_concurrent_positions=20,  # Maximum aggression
        # Enable MARKSMAN (Deprecated - always False)
        enable_marksman=False,
    ),
    
    # MARKSMAN mode removed.
    
    TradingMode.BALANCED: ModeConfig(
        name="Balanced",
        mode_type=TradingMode.BALANCED,
        description="⚖️ Balanced approach, moderate thresholds",
        # ⚠️ FURTHER INCREASED thresholds to improve PF and reduce trades
        min_signal_score=72,  # ⚠️ INCREASED: 72 (was 68) - top 20% of signals only
        min_signal_strength=0.70,  # ⚠️ INCREASED: 0.70 (was 0.65) - much stronger signals required
        hard_min_score=70,  # ⚠️ INCREASED: 70 (was 65) - very strict floor
        marksman_min_score=None,  # Not used
        # Stricter entry filters
        min_momentum_pct=1.5,  # ⚠️ INCREASED: 1.5% (was 1.2%) - require strong momentum
        min_spread_bps=8.0,  # ⚠️ INCREASED: 8 bps (was 5.0) - avoid tight spreads
        max_spread_bps=50.0,  # ⚠️ DECREASED: 50 bps (was 60.0) - tighter max spread
        min_volume_24h=5_000_000,  # ⚠️ INCREASED: $5M (was $3M) - highly liquid symbols only
        # Fewer positions
        max_concurrent_positions=6,  # ⚠️ DECREASED: 6 (was 8) - focus on best opportunities
        # Disable MARKSMAN
        enable_marksman=False,
    ),
}

def get_mode_config(mode: TradingMode) -> ModeConfig:
    """Get configuration for a trading mode."""
    return MODE_CONFIGS[mode]

