"""
Clean Configuration Interface - The ONLY config you should use.

This module provides a clean, minimal interface to the bot's configuration.
It consolidates the 324+ variables in config.py down to ~40 that actually matter.

RULES:
1. ONE variable per concept (no MAX_CONCURRENT_POS_HARD vs MAX_OPEN_POSITIONS confusion)
2. NO bypass mechanisms (UNICORN protocol is banned)
3. Sensible defaults that won't blow up your account
4. Type hints and validation

Usage:
    from app.config_clean import cfg
    
    if cfg.is_live:
        print(f"Trading with max {cfg.max_positions} positions")
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os

# Import the messy config for backward compatibility
from . import config as _legacy


@dataclass
class ExchangeConfig:
    """Exchange connection settings."""
    name: str = "binance_futures"
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # For OKX/Bitget
    testnet: bool = False
    margin_mode: str = "CROSS"  # CROSS or ISOLATED


@dataclass
class RiskConfig:
    """Risk management - THE RULES THAT KEEP YOU ALIVE."""
    # Position limits - ONE SOURCE OF TRUTH
    max_positions: int = 5  # Maximum concurrent positions
    min_position_usd: float = 50.0  # Minimum $50 per position (no dust!)
    max_position_usd: float = 100.0  # Maximum per position
    
    # Risk per trade
    risk_per_trade_pct: float = 2.0  # 2% of equity per trade
    max_account_risk_pct: float = 25.0  # 25% total risk (conservative)
    
    # Circuit breakers
    max_drawdown_pct: float = 10.0  # Stop trading at 10% drawdown
    max_loss_streak: int = 5  # Reduce size after 5 losses
    
    # Fees (realistic!)
    taker_fee_bps: float = 4.0  # 0.04% taker fee
    slippage_bps: float = 10.0  # 10 bps realistic slippage


@dataclass
class EntryConfig:
    """Entry signal requirements."""
    min_score: int = 65  # Minimum signal score (0-100)
    min_strength: float = 0.60  # Minimum strength (0-1)
    max_spread_bps: float = 100.0  # Max spread to enter
    min_volume_24h: float = 1_000_000.0  # $1M minimum volume
    
    # Cooldowns
    cooldown_same_symbol_sec: int = 120  # 2 min before re-entry same symbol
    cooldown_after_exit_sec: int = 30  # 30s after any exit
    max_entries_per_min: int = 3  # Rate limit


@dataclass 
class ExitConfig:
    """Exit strategy - PICK ONE AND STICK WITH IT.
    
    Using simple R-based exits:
    - Stop loss at -1R (initial risk distance)
    - Take profit at +2.5R
    - Trailing stop activates at +1R, trails at 0.5R behind peak
    """
    stop_loss_r: float = -1.0  # Exit at -1R (stop)
    take_profit_r: float = 2.5  # Exit at +2.5R (target)
    
    # Trailing stop
    trail_activation_r: float = 1.0  # Start trailing at +1R
    trail_distance_r: float = 0.5  # Trail 0.5R behind peak
    
    # Time-based
    max_position_age_sec: int = 3600  # 1 hour max hold
    time_exit_if_flat: bool = True  # Exit if <0.3R after max age


@dataclass
class ScanConfig:
    """Universe and scanning settings."""
    symbols_to_scan: int = 100  # How many symbols to scan
    scan_interval_sec: float = 5.0  # How often to scan
    universe_refresh_sec: float = 300.0  # Refresh universe every 5 min


@dataclass
class CleanConfig:
    """
    The ONE config object you need.
    
    This consolidates 324 variables into a clean, validated structure.
    """
    # Mode flags
    is_live: bool = False
    is_replay: bool = False
    
    # Sub-configs
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    
    def __post_init__(self):
        """Validate configuration on creation."""
        self._validate()
    
    def _validate(self):
        """Ensure config is sane. Raises ValueError if not."""
        # Position sizing sanity
        if self.risk.min_position_usd < 10:
            raise ValueError(f"min_position_usd must be >= $10 (got ${self.risk.min_position_usd})")
        
        if self.risk.max_positions > 10 and self.is_live:
            raise ValueError(f"max_positions > 10 in LIVE mode is too risky (got {self.risk.max_positions})")
        
        # Risk sanity
        if self.risk.max_account_risk_pct > 50:
            raise ValueError(f"max_account_risk_pct > 50% is insane (got {self.risk.max_account_risk_pct}%)")
        
        if self.risk.risk_per_trade_pct > 5:
            raise ValueError(f"risk_per_trade_pct > 5% is too aggressive (got {self.risk.risk_per_trade_pct}%)")
        
        # Exit sanity
        if self.exit.take_profit_r < 1.5:
            raise ValueError(f"take_profit_r < 1.5 won't cover costs (got {self.exit.take_profit_r})")
    
    @classmethod
    def from_legacy(cls) -> 'CleanConfig':
        """
        Create CleanConfig from the legacy config.py values.
        This is the bridge between old and new.
        """
        # Determine the ONE TRUE position limit
        # Priority: MAX_CONCURRENT_POS_HARD > MAX_OPEN_POSITIONS > MAX_CONCURRENT_POS
        max_pos = min(
            getattr(_legacy, 'MAX_CONCURRENT_POS_HARD', 15),
            getattr(_legacy, 'MAX_OPEN_POSITIONS', 5),
            15  # Hard cap for sanity
        )
        
        exchange = ExchangeConfig(
            name=_legacy.EXCHANGE,
            api_key=_legacy.BINANCE_API_KEY if _legacy.EXCHANGE == "binance_futures" else "",
            api_secret=_legacy.BINANCE_API_SECRET if _legacy.EXCHANGE == "binance_futures" else "",
            testnet=_legacy.BINANCE_TESTNET if _legacy.EXCHANGE == "binance_futures" else False,
            margin_mode=_legacy.MARGIN_MODE,
        )
        
        risk = RiskConfig(
            max_positions=max_pos,
            min_position_usd=max(_legacy.MIN_POSITION_SIZE, 10.0),  # At least $10
            max_position_usd=_legacy.MAX_POSITION_SIZE,
            risk_per_trade_pct=min(_legacy.RISK_PER_TRADE_PCT, 5.0),  # Cap at 5%
            max_account_risk_pct=min(_legacy.MAX_ACCOUNT_RISK_PCT, 50.0),  # Cap at 50%
            max_drawdown_pct=_legacy.MAX_DRAWDOWN_PCT,
            max_loss_streak=_legacy.MAX_LOSS_STREAK_HARD if _legacy.MAX_LOSS_STREAK_HARD < 100 else 5,
            taker_fee_bps=_legacy.TAKER_FEE_RATE * 10000,
            slippage_bps=_legacy.SLIPPAGE_BPS,
        )
        
        entry = EntryConfig(
            min_score=_legacy.MIN_SIGNAL_SCORE,
            min_strength=_legacy.MIN_SIGNAL_STRENGTH,
            max_spread_bps=_legacy.MAX_SPREAD_BPS,
            min_volume_24h=_legacy.MIN_VOLUME_24H,
            cooldown_same_symbol_sec=_legacy.COOLDOWN_SAME_SYMBOL,
            cooldown_after_exit_sec=_legacy.COOLDOWN_AFTER_EXIT,
            max_entries_per_min=_legacy.MAX_ENTRIES_PER_MIN,
        )
        
        # Consolidate exit config - use sensible defaults
        exit_cfg = ExitConfig(
            stop_loss_r=1.0,  # Default: 1R stop loss
            take_profit_r=2.0,  # Default: 2R take profit
            trail_activation_r=_legacy.TRAIL_ENGINE_START_BUFFER_R,
            trail_distance_r=0.5,  # Sensible default
            max_position_age_sec=_legacy.MAX_POSITION_AGE_SEC,
            time_exit_if_flat=True,
        )
        
        scan = ScanConfig(
            symbols_to_scan=_legacy.SYMBOLS_TO_SCAN,
            scan_interval_sec=_legacy.DISCOVERY_SCAN_INTERVAL_SEC,
            universe_refresh_sec=300.0,
        )
        
        return cls(
            is_live=not _legacy.DRY_RUN,
            is_replay=_legacy.REPLAY_MODE,
            exchange=exchange,
            risk=risk,
            entry=entry,
            exit=exit_cfg,
            scan=scan,
        )
    
    def to_dict(self) -> dict:
        """Export config as dict (for logging/serialization)."""
        return {
            'is_live': self.is_live,
            'is_replay': self.is_replay,
            'exchange': self.exchange.name,
            'risk': {
                'max_positions': self.risk.max_positions,
                'min_position_usd': self.risk.min_position_usd,
                'max_position_usd': self.risk.max_position_usd,
                'risk_per_trade_pct': self.risk.risk_per_trade_pct,
                'max_account_risk_pct': self.risk.max_account_risk_pct,
                'max_drawdown_pct': self.risk.max_drawdown_pct,
            },
            'entry': {
                'min_score': self.entry.min_score,
                'min_strength': self.entry.min_strength,
                'max_spread_bps': self.entry.max_spread_bps,
            },
            'exit': {
                'stop_loss_r': self.exit.stop_loss_r,
                'take_profit_r': self.exit.take_profit_r,
                'trail_activation_r': self.exit.trail_activation_r,
            },
        }


# ----------------------------------------------------------------
# SINGLETON INSTANCE - Use this!
# ----------------------------------------------------------------
# Create singleton from legacy config on import
# This allows gradual migration without breaking existing code

try:
    cfg = CleanConfig.from_legacy()
except ValueError as e:
    # If validation fails, use safe defaults
    import warnings
    warnings.warn(f"Config validation failed: {e}. Using safe defaults.")
    cfg = CleanConfig(
        is_live=False,
        risk=RiskConfig(
            max_positions=5,
            min_position_usd=50.0,
            max_position_usd=100.0,
            risk_per_trade_pct=2.0,
            max_account_risk_pct=25.0,
        ),
    )


# ----------------------------------------------------------------
# CONVENIENCE EXPORTS (for drop-in replacement)
# ----------------------------------------------------------------
# These allow: from config_clean import MAX_POSITIONS
# Instead of reaching into nested objects

MAX_POSITIONS = cfg.risk.max_positions
MIN_POSITION_USD = cfg.risk.min_position_usd
MAX_POSITION_USD = cfg.risk.max_position_usd
RISK_PER_TRADE_PCT = cfg.risk.risk_per_trade_pct
MAX_ACCOUNT_RISK_PCT = cfg.risk.max_account_risk_pct
MAX_DRAWDOWN_PCT = cfg.risk.max_drawdown_pct

MIN_SIGNAL_SCORE = cfg.entry.min_score
MIN_SIGNAL_STRENGTH = cfg.entry.min_strength
MAX_SPREAD_BPS = cfg.entry.max_spread_bps

IS_LIVE = cfg.is_live
IS_DRY_RUN = not cfg.is_live
