"""
Unified Configuration Module - Centralized configuration management.
All configuration values are organized into logical sections for better maintainability.
"""

import os
import json
from typing import Any, Optional, Union, Callable
from pathlib import Path

# CRITICAL: Load .env file FIRST before reading any environment variables
try:
    from dotenv import load_dotenv
    # Try config/.env first, then fall back to root .env
    config_env = Path(__file__).parent.parent / "config" / ".env"
    root_env = Path(__file__).parent.parent / ".env"
    
    if config_env.exists():
        load_dotenv(config_env, override=False)  # override=False to allow system env vars to take precedence
    elif root_env.exists():
        load_dotenv(root_env, override=False)  # override=False to allow system env vars to take precedence
except ImportError:
    # dotenv is optional
    pass

# ----------------------------------------------------------------
# Environment Variable Helper
# ----------------------------------------------------------------

def env(name: str, default: Any = None, cast: Optional[Callable] = None) -> Any:
    """
    Get environment variable with optional type casting.
    
    Args:
        name: Environment variable name
        default: Default value if not set
        cast: Optional type cast function (int, float, str, etc.)
    
    Returns:
        Environment variable value (casted if cast provided) or default
    """
    v = os.getenv(name, default)
    if cast is None:
        return v
    try:
        return cast(v) if v is not None else default
    except (ValueError, TypeError, AttributeError):
        # Type casting failed (e.g., invalid int/float string, None passed to cast)
        return default


# ----------------------------------------------------------------
# DEBUG MODE TOGGLES
# ----------------------------------------------------------------

# Debug flags for development and troubleshooting
DEBUG_MODE = env("DEBUG_MODE", "0") in ("1", "true", "TRUE")
DEBUG_SCANNING = env("DEBUG_SCANNING", "0") in ("1", "true", "TRUE")  # Debug symbol scanning
DEBUG_SIGNALS = env("DEBUG_SIGNALS", "0") in ("1", "true", "TRUE")  # Debug signal generation
DEBUG_ORDERS = env("DEBUG_ORDERS", "0") in ("1", "true", "TRUE")  # Debug order execution
DEBUG_EXITS = env("DEBUG_EXITS", "0") in ("1", "true", "TRUE")  # Debug exit logic
DEBUG_PERFORMANCE = env("DEBUG_PERFORMANCE", "0") in ("1", "true", "TRUE")  # Debug performance metrics
DEBUG_CACHE = env("DEBUG_CACHE", "0") in ("1", "true", "TRUE")  # Debug cache operations
DEBUG_API = env("DEBUG_API", "0") in ("1", "true", "TRUE")  # Debug API calls


# ----------------------------------------------------------------
# CORE FLAGS
# ----------------------------------------------------------------

# Core flags
# NOTE: DRY_RUN removed - bot always runs in LIVE mode on real Binance
DRY_RUN = False  # Always live trading (paper trading removed)
REPLAY_MODE = env("REPLAY_MODE", "0") in ("1","true","TRUE")  # REPLAY MODE: For backtesting only (DO NOT enable in production)
# OPTIMIZATION: Removed USE_WS (WebSocket support not implemented, unused)
USE_RICH_UI = env("USE_RICH_UI", "0") in ("1","true","TRUE")  # Legacy flag, ignored for new Rich UI
# UI_MODE:
#   "v2"   - Rich panel UI (full-screen TUI dashboard)
#   "v4"   - ANSI static header (very lightweight)
#   "rich" - Rich panel UI (like classic v2 but simpler and safer)
#   "btop" - Btop-style UI with real-time graphs and system monitoring
#   "dashboard" - Dashboard mode
#   "static"/"basic"/"simple" - other legacy/simple modes
UI_MODE = env("UI_MODE", "console")  # Default to console/headless - pure logging, no Rich UI

# Exchange Selection (Primary - for single exchange mode)
EXCHANGE = env("EXCHANGE", "binance_futures").lower()  # Supported: binance_futures, bybit, okx, kraken, bitget, mexc_futures

# Credentials - Binance Futures
BINANCE_API_KEY = env("BINANCE_API_KEY", "")
BINANCE_API_SECRET = env("BINANCE_API_SECRET", "")  # Use BINANCE_API_SECRET for consistency
BINANCE_SECRET = BINANCE_API_SECRET  # Alias for backward compatibility
BINANCE_TESTNET = env("BINANCE_TESTNET", "0") in ("1","true","TRUE")  # Use Binance testnet
MARGIN_MODE = env("MARGIN_MODE", "CROSS")  # Default to CROSS for EEA compatibility (was ISOLATED)

# Credentials - Bybit
BYBIT_API_KEY = env("BYBIT_API_KEY", "")
BYBIT_API_SECRET = env("BYBIT_API_SECRET", "")
BYBIT_TESTNET = env("BYBIT_TESTNET", "0") in ("1","true","TRUE")  # Use Bybit testnet

# Credentials - OKX
OKX_API_KEY = env("OKX_API_KEY", "")
OKX_API_SECRET = env("OKX_API_SECRET", "")
OKX_PASSPHRASE = env("OKX_PASSPHRASE", "")  # OKX requires a passphrase
OKX_TESTNET = env("OKX_TESTNET", "0") in ("1","true","TRUE")  # Use OKX testnet

# Credentials - Kraken
KRAKEN_API_KEY = env("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = env("KRAKEN_API_SECRET", "")
KRAKEN_TESTNET = env("KRAKEN_TESTNET", "0") in ("1","true","TRUE")  # Use Kraken testnet

# Credentials - Bitget
BITGET_API_KEY = env("BITGET_API_KEY", "")
BITGET_API_SECRET = env("BITGET_API_SECRET", "")
BITGET_PASSPHRASE = env("BITGET_PASSPHRASE", "")  # Bitget requires a passphrase
BITGET_TESTNET = env("BITGET_TESTNET", "0") in ("1","true","TRUE")  # Use Bitget testnet

# Credentials - MEXC
MEXC_API_KEY = env("MEXC_API_KEY", "")
MEXC_API_SECRET = env("MEXC_API_SECRET", "")
MEXC_TESTNET = env("MEXC_TESTNET", "0") in ("1","true","TRUE")  # Use MEXC testnet

# Validation warnings (non-blocking)
if not DRY_RUN:
    # Exchange-specific API key validation
    if EXCHANGE == "binance_futures" and (not BINANCE_API_KEY or not BINANCE_API_SECRET):
        import warnings
        warnings.warn("BINANCE_API_KEY or BINANCE_API_SECRET not set for Binance Futures", UserWarning)
    elif EXCHANGE == "bybit" and (not BYBIT_API_KEY or not BYBIT_API_SECRET):
        import warnings
        warnings.warn("BYBIT_API_KEY or BYBIT_API_SECRET not set for Bybit", UserWarning)
    elif EXCHANGE == "okx" and (not OKX_API_KEY or not OKX_API_SECRET or not OKX_PASSPHRASE):
        import warnings
        warnings.warn("OKX_API_KEY, OKX_API_SECRET, or OKX_PASSPHRASE not set for OKX", UserWarning)
    elif EXCHANGE == "kraken" and (not KRAKEN_API_KEY or not KRAKEN_API_SECRET):
        import warnings
        warnings.warn("KRAKEN_API_KEY or KRAKEN_API_SECRET not set for Kraken", UserWarning)
    elif EXCHANGE == "bitget" and (not BITGET_API_KEY or not BITGET_API_SECRET or not BITGET_PASSPHRASE):
        import warnings
        warnings.warn("BITGET_API_KEY, BITGET_API_SECRET, or BITGET_PASSPHRASE not set for Bitget", UserWarning)
    elif EXCHANGE == "mexc_futures" and (not MEXC_API_KEY or not MEXC_API_SECRET):
        import warnings
        warnings.warn("MEXC_API_KEY or MEXC_API_SECRET not set for MEXC", UserWarning)

# Trading params
ACCOUNT_BAL        = env("ACCOUNT_BAL","250", float)  # Starting account balance
RISK_PCT           = env("RISK_PCT","0.5", float)/100.0  # Risk per trade: 0.5% (conservative)
LEVERAGE_BASE      = env("LEVERAGE","5", int)  # Legacy base leverage (used if dynamic disabled)

# Dynamic Leverage and Position Sizing (Phase 1: External Review Implementation)
USE_DYNAMIC_LEVERAGE = env("USE_DYNAMIC_LEVERAGE", "1") in ("1","true","TRUE")  # Enable dynamic leverage based on signal strength
MIN_LEVERAGE = env("MIN_LEVERAGE", "3", int)  # AGGRESSIVE: Minimum leverage 3x (was 2x)
MAX_LEVERAGE = env("MAX_LEVERAGE", "10", int)  # AGGRESSIVE: Maximum leverage 10x (was 5x)

USE_DYNAMIC_POSITION_SIZE = env("USE_DYNAMIC_POSITION_SIZE", "1") in ("1","true","TRUE")  # Enable dynamic position sizing
BASE_POSITION_PCT = env("BASE_POSITION_PCT", "2.0", float)  # AGGRESSIVE: Base 2.0% (was 1.2%)
MAX_POSITION_PCT = env("MAX_POSITION_PCT", "4.0", float)  # AGGRESSIVE: Max 4.0% (was 2.5%)
MIN_POSITION_PCT = env("MIN_POSITION_PCT", "1.0", float)  # AGGRESSIVE: Min 1.0% (was 0.5%)

# Risk Management - PURE SCALPER MODE: Simple, tight risk limits
TOTAL_RISK_BUDGET = env("TOTAL_RISK_BUDGET", "2.0", float) / 100.0  # Total risk budget as % of equity (2% = 0.02) - scalper-friendly
# NOTE: Effective max positions are now dynamic: floor(MAX_ACCOUNT_RISK_PCT / RISK_PER_TRADE_PCT),
# but never exceed MAX_CONCURRENT_POS_HARD. The old MAX_OPEN_POSITIONS is kept for backward compatibility.
MAX_OPEN_POSITIONS = env("MAX_OPEN_POSITIONS", "3", int)  # Default 3 positions - unicorns bypass this limit
MAX_CONCURRENT_POS = MAX_OPEN_POSITIONS  # Alias for backward compatibility
MAX_CONCURRENT_POS_MIN = env("MAX_CONCURRENT_POS_MIN","3", int)  # Default to 3 (minimum positions)
MAX_CONCURRENT_POS_MAX = env("MAX_CONCURRENT_POS_MAX","15", int)  # Max allowed

# Dynamic risk-based position cap (Option C: Force Selectivity)
# MAXIMUM AGGRESSION MODE: High risk, high reward
MAX_ACCOUNT_RISK_PCT = env("MAX_ACCOUNT_RISK_PCT", "25.0", float)  # BALANCED: 25% total risk
RISK_PER_TRADE_PCT = env("RISK_PER_TRADE_PCT", "1.0", float)  # ML-optimized: 1.0% per trade (was 5.0% - safer, tighter exits)
MAX_CONCURRENT_POS_HARD = env("MAX_CONCURRENT_POS_HARD", "3", int)  # Default 3 - unicorns bypass this limit

# Per-trade risk bounds
MIN_RISK_PER_TRADE = env("MIN_RISK_PER_TRADE", "0.5", float) / 100.0  # AGGRESSIVE: 0.5% minimum (was 0.2%)
MAX_RISK_PER_TRADE = env("MAX_RISK_PER_TRADE", "2.5", float) / 100.0  # AGGRESSIVE: 2.5% max per trade (was 1.0%)

# Calculate RISK_PCT from TOTAL_RISK_BUDGET / MAX_OPEN_POSITIONS (for backward compatibility)
# This is a default target, but actual sizing uses risk budget dynamically
RISK_PCT = TOTAL_RISK_BUDGET / MAX_OPEN_POSITIONS if MAX_OPEN_POSITIONS > 0 else 0.005  # Default 0.5% if division by zero

# Legacy/fallback risk settings (kept for backward compatibility)
MAX_RISK_PER_TRADE_PCT = env("MAX_RISK_PER_TRADE_PCT", None, float)  # If set, overrides calculated RISK_PCT
if MAX_RISK_PER_TRADE_PCT is not None:
    RISK_PCT = MAX_RISK_PER_TRADE_PCT / 100.0  # Override with explicit setting

CORRELATION_PENALTY_PCT = env("CORRELATION_PENALTY_PCT", "40.0", float) / 100.0  # Reduce size by 40% if 2+ correlated positions
STREAK_PENALTY_PCT = env("STREAK_PENALTY_PCT", "40.0", float) / 100.0  # Reduce size/leverage by 40% if 3+ losses
CORRELATION_THRESHOLD = env("CORRELATION_THRESHOLD", "2", int)  # Number of correlated positions before penalty applies
CORRELATION_BLOCK_THRESHOLD = env("CORRELATION_BLOCK_THRESHOLD", "0.96", float)  # Block entry if correlation > 0.96 with any open position (relaxed from 0.90)
STREAK_THRESHOLD = env("STREAK_THRESHOLD", "3", int)  # Number of consecutive losses before penalty applies
TREND_ALIGNMENT_REQUIRED = env("TREND_ALIGNMENT_REQUIRED", "0") in ("1","true","TRUE")  # Require 5m and 15m trend alignment with trade direction (disabled by default - too restrictive)
MAX_CAPITAL_PER_POS = env("MAX_CAPITAL_PER_POS","0.25", float)  # AGGRESSIVE: Max 25% per position (was 15%)

# Score-Aware Replacement Logic
SCORE_AWARE_REPLACEMENT_ENABLED = env("SCORE_AWARE_REPLACEMENT_ENABLED", "1") in ("1","true","TRUE")  # Enable score-aware replacement
SCORE_REPLACEMENT_MARGIN = env("SCORE_REPLACEMENT_MARGIN", "5", float)  # Reduced margin from 10->5 to free slots for higher scores

# Friction
TAKER_FEE_RATE     = env("TAKER_FEE_RATE","0.0004", float)
ENTRY_FEE_RATE     = env("ENTRY_FEE_RATE","0.0004", float)
FUNDING_RATE_PER8H = env("FUNDING_RATE_PER8H","0.0", float)
# REALISTIC SLIPPAGE: Research shows 10-50 bps in volatile moves, not 2 bps
# Using 10 bps as baseline (conservative estimate for liquid pairs)
SLIPPAGE_BPS       = env("SLIPPAGE_BPS","10", float)

# ===========================
# COST GATE - DISABLED (ML SCORER HANDLES THIS)
# ===========================
# The ML scorer already factors in spread, fees, and market conditions
# when computing win probability. A separate cost gate is redundant.
COST_GATE_ENABLED = env("COST_GATE_ENABLED", "0") in ("1", "true", "TRUE")  # DISABLED - ML is in charge
COST_GATE_MULTIPLIER = env("COST_GATE_MULTIPLIER", "0.8", float)  # Legacy - unused
COST_GATE_MIN_EDGE_BPS = env("COST_GATE_MIN_EDGE_BPS", "3", float)  # Legacy - unused
COST_GATE_SCORE_TO_EDGE_MULT = env("COST_GATE_SCORE_TO_EDGE_MULT", "3.0", float)  # Legacy - unused

# ===========================
# SPREAD EXPLOSION EXIT
# ===========================
# Philosophy: "If liquidity disappears, GET OUT"
SPREAD_EXPLOSION_EXIT_ENABLED = env("SPREAD_EXPLOSION_EXIT_ENABLED", "1") in ("1", "true", "TRUE")
SPREAD_EXPLOSION_MULTIPLIER = env("SPREAD_EXPLOSION_MULTIPLIER", "3.0", float)  # Exit if spread 3x entry spread
SPREAD_EXPLOSION_MAX_BPS = env("SPREAD_EXPLOSION_MAX_BPS", "100", float)  # Hard cap: exit if spread > 100 bps

# ===========================
# FUNDING TIME AVOIDANCE
# ===========================
# Binance funding times: 00:00, 08:00, 16:00 UTC
FUNDING_AVOIDANCE_ENABLED = env("FUNDING_AVOIDANCE_ENABLED", "1") in ("1", "true", "TRUE")
FUNDING_AVOIDANCE_MINUTES = env("FUNDING_AVOIDANCE_MINUTES", "10", int)  # Exit 10 min before funding
FUNDING_AVOIDANCE_MIN_PROFIT_PCT = env("FUNDING_AVOIDANCE_MIN_PROFIT_PCT", "0.05", float)  # Only if profit < 0.05%

# ===========================
# TIME-OF-DAY FILTER (Trading Hours)
# ===========================
# Research shows liquidity and edge vary by time of day
# Best hours: London/NY overlap (13:00-17:00 UTC), Asia open (00:00-04:00 UTC)
# Worst hours: Around funding (±30min), low-liquidity gaps
TIME_FILTER_ENABLED = env("TIME_FILTER_ENABLED", "0") in ("1", "true", "TRUE")  # Disabled by default
# Avoid these hours (UTC) - typically funding hours and low liquidity
TIME_FILTER_AVOID_HOURS = env("TIME_FILTER_AVOID_HOURS", "7,15,23")  # Hours before funding (avoid manipulation)
# Reduce position size during these hours (less confidence)
TIME_FILTER_REDUCED_HOURS = env("TIME_FILTER_REDUCED_HOURS", "3,4,5,11,12,19,20,21")  # Lower liquidity periods
TIME_FILTER_SIZE_REDUCTION = env("TIME_FILTER_SIZE_REDUCTION", "0.5", float)  # 50% size during reduced hours

# ===========================
# REGIME FILTER (Market Conditions)
# ===========================
# Only trade when conditions favor momentum scalping
REGIME_FILTER_ENABLED = env("REGIME_FILTER_ENABLED", "0") in ("1", "true", "TRUE")  # DISABLED: Let ML decide
# CRITICAL: Enable/disable RAPTOR regime override (dynamic threshold lowering)
REGIME_OVERRIDE_ENABLED = env("REGIME_OVERRIDE_ENABLED", "0") in ("1", "true", "TRUE")  # DISABLED: No overrides
# Minimum BTC 24h volatility to trade (avoid dead/ranging markets)
REGIME_MIN_BTC_VOLATILITY_PCT = env("REGIME_MIN_BTC_VOLATILITY_PCT", "1.0", float)
# Maximum BTC 24h volatility (avoid extreme chaos where stops get blown)
REGIME_MAX_BTC_VOLATILITY_PCT = env("REGIME_MAX_BTC_VOLATILITY_PCT", "15.0", float)
# BTC trend strength required (0 = no filter, higher = need stronger trend)
REGIME_MIN_BTC_TREND_STRENGTH = env("REGIME_MIN_BTC_TREND_STRENGTH", "0.5", float)

# ===========================
# QUICK SCALP MODE (Momentum Lock)
# ===========================
# Philosophy: "Ride the green, bail on the turn"
QUICK_SCALP_MODE = env("QUICK_SCALP_MODE", "0") in ("1", "true", "TRUE")  # Disabled by default for LIVE
QS_MIN_PROFIT_TO_TRAIL = env("QS_MIN_PROFIT_TO_TRAIL", "0.15", float)
QS_REVERSAL_THRESHOLD_PCT = env("QS_REVERSAL_THRESHOLD_PCT", "0.08", float)
QS_FLOOR_ABOVE_BE_PCT = env("QS_FLOOR_ABOVE_BE_PCT", "0.12", float)
QS_MAX_HOLD_BARS = env("QS_MAX_HOLD_BARS", "4", int)
QS_STOP_LOSS_R = env("QS_STOP_LOSS_R", "-0.5", float)
QS_PEAK_DECAY_BARS = env("QS_PEAK_DECAY_BARS", "2", int)

# Behavior - Balanced for active trading
CLOSE_FEECHURN_BPS = env("CLOSE_FEECHURN_BPS","4", float)
ENTRY_DELAY_MS     = env("ENTRY_DELAY_MS","50", int)  # Entry delay
STARTUP_DELAY_SEC  = env("STARTUP_DELAY_SEC","30", int)  # Startup delay
COOLDOWN_SEC       = env("COOLDOWN_SEC","15", int)  # 15s global cooldown
COOLDOWN_SAME_SYMBOL = env("COOLDOWN_SAME_SYMBOL","120", int)  # 2 min before re-entering same symbol
COOLDOWN_AFTER_EXIT = env("COOLDOWN_AFTER_EXIT","30", int)  # 30s after exit before new entry
COOLDOWN_DIFF_SYMBOL = env("COOLDOWN_DIFF_SYMBOL","5", int)  # 5s between different symbols
# After flat-ish time_exit (|R| <= 0.25), block re-entries on that symbol for a while
SYMBOL_CHURN_COOLDOWN_SEC = env("SYMBOL_CHURN_COOLDOWN_SEC", "300", int)  # 5 min after flat exit - prevent churn
# Time-based exit: more patient, only kill losers/zombies (increased to maintain 1+ positions)
TIME_EXIT_BARS = env("TIME_EXIT_BARS", "15", int)  # More patient, ~15 candles before time exit (was 10, increased to keep positions longer)

# ATR multiplier for initial stop loss calculation
# DATA-DRIVEN: Analysis shows SL rate drops from 56.5% to 30.2% as ATR increases
# Using 2.5x ATR as minimum stop distance to reduce premature stop-outs
SL_ATR_MULTIPLIER = env("SL_ATR_MULTIPLIER", "2.5", float)  # 2.5x ATR = data-driven buffer (was 1.0)
# NOTE: LIVE always uses full exit engine (partials, trailing, etc.).
# These DRY_* settings must never be consulted in LIVE mode.
STATE_SAVE_SEC     = env("STATE_SAVE_SEC","10", int)
MAX_ENTRIES_PER_MIN= env("MAX_ENTRIES_PER_MIN","5", int)  # 5 entries/min max - reasonable for active trading
MAX_TRADES_PER_DAY = env("MAX_TRADES_PER_DAY", "500", int)  # 500 trades/day max
MAX_TRADES_PER_DAY_REPLAY = env("MAX_TRADES_PER_DAY_REPLAY", "2000", int)  # Replay can do more

# Entry filters
MIN_SPREAD_BPS     = env("MIN_SPREAD_BPS","5", float)  # Minimum spread for entry (5 bps) - relaxed for Binance Futures
MAX_SPREAD_BPS     = env("MAX_SPREAD_BPS","150", float)  # BALANCED: 150 bps
MIN_VOLUME_24H     = env("MIN_VOLUME_24H","1000000", float)  # Minimum $1M 24h volume - relaxed for Binance Futures
MAX_LATENCY_MS     = env("MAX_LATENCY_MS","50", int)  # Ultra-low latency: 50ms (cached data only, no API calls)

# ===========================
# DATA-DRIVEN ENTRY FILTERS (from 4M+ trade backtest)
# ===========================
# Backtested strategies that compound synergistically:
# - High Volatility (ATR >= 1.0%) + Strong Momentum (|1h| >= 3%) + RSI Extreme
# - Result: +0.32R per trade (vs +0.17R baseline) = +88% improvement
#
# These filters work WITH Diamond Hands exit strategy for best results.
#
ENTRY_FILTER_ENABLED = env("ENTRY_FILTER_ENABLED", "0") in ("1", "true", "TRUE")  # DISABLED: ML already considers these features

# Volatility filter: Only trade when market is moving
ENTRY_MIN_ATR_PCT = env("ENTRY_MIN_ATR_PCT", "1.0", float)  # Minimum 1% ATR

# Momentum filter: Only trade strong moves
ENTRY_MIN_MOMENTUM_PCT = env("ENTRY_MIN_MOMENTUM_PCT", "3.0", float)  # Minimum 3% 1h change

# RSI alignment filter: Trade with RSI extreme for direction confirmation
ENTRY_RSI_LONG_MAX = env("ENTRY_RSI_LONG_MAX", "25", float)  # Long only when RSI < 25
ENTRY_RSI_SHORT_MIN = env("ENTRY_RSI_SHORT_MIN", "75", float)  # Short only when RSI > 75

# Allow relaxed mode (less strict filters for more trades)
ENTRY_FILTER_RELAXED = env("ENTRY_FILTER_RELAXED", "0") in ("1", "true", "TRUE")
# Relaxed thresholds (if ENTRY_FILTER_RELAXED=1)
ENTRY_MIN_ATR_PCT_RELAXED = env("ENTRY_MIN_ATR_PCT_RELAXED", "0.5", float)
ENTRY_MIN_MOMENTUM_PCT_RELAXED = env("ENTRY_MIN_MOMENTUM_PCT_RELAXED", "2.0", float)
ENTRY_RSI_LONG_MAX_RELAXED = env("ENTRY_RSI_LONG_MAX_RELAXED", "35", float)
ENTRY_RSI_SHORT_MIN_RELAXED = env("ENTRY_RSI_SHORT_MIN_RELAXED", "65", float)

# ===========================
# ADAPTIVE FILTERS (Auto-adjusts to market conditions)
# ===========================
# Instead of fixed thresholds, uses percentile-based filtering:
# - ATR_PERCENTILE=60 means "trade top 40% volatility"
# - In calm markets: absolute thresholds drop automatically
# - In volatile markets: thresholds rise automatically
#
USE_ADAPTIVE_FILTERS = env("USE_ADAPTIVE_FILTERS", "1") in ("1", "true", "TRUE")  # ENABLED: ML ATR adaptation active
ENTRY_ATR_PERCENTILE = env("ENTRY_ATR_PERCENTILE", "60", float)  # Top 40% volatility
ENTRY_MOMENTUM_PERCENTILE = env("ENTRY_MOMENTUM_PERCENTILE", "70", float)  # Top 30% momentum
ENTRY_RSI_EXTREME_PCT = env("ENTRY_RSI_EXTREME_PCT", "20", float)  # Top/bottom 20% RSI

# ===========================
# SIGNAL STRENGTH THRESHOLDS
# ===========================
# FRESHNESS SCORING SYSTEM (Nov 21, 2025 - Complete Rework)
# Philosophy: "The best trades are fresh momentum moves with confirmation"
# 
# Score Components:
# - Freshness (40 pts): Bell curve peaks at 5%, drops after 7% (natural exhaustion filter)
# - Confirmation (40 pts): Volume, orderbook, acceleration, volatility
# - Execution (20 pts): Spread, depth, latency, symbol reliability
#
# Expected Score Distribution:
# - <70: Reject (50% of signals) - Weak/stale
# - 70-85: Good trades (30% of signals) - Fresh moderate momentum
# - 85-95: Great trades (15% of signals) - Strong fresh momentum
# - 95+: Rare extremes (5% of signals) - Should be very rare
#
# Target: 55-60% WR by naturally filtering exhaustion and requiring confirmation
# WARNING: Higher thresholds to prevent excessive trading and improve PF
# IMPORTANT:
# We support both:
# - MIN_SIGNAL_*: general config knobs (from .env)
# - LIVE_MIN_SIGNAL_*: operator overrides for live runs (set by launcher scripts)
#
# LIVE_* must take precedence, and values may be floats ("62.0") so we parse them robustly.
def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name, None)
        if raw is None:
            return float(default)
        s = str(raw).strip()
        if not s:
            return float(default)
        return float(s)
    except Exception:
        return float(default)

# ===========================
# LEGACY THRESHOLDS - MINIMAL FLOORS ONLY
# ===========================
# ML IS NOW KING! Adaptive filters handle real filtering.
# These are just garbage/sanity checks - set VERY low.
#
MIN_SIGNAL_STRENGTH = _env_float("LIVE_MIN_SIGNAL_STRENGTH", _env_float("MIN_SIGNAL_STRENGTH", 0.05))  # 5% floor
# INTELLIGENT SIGNAL SELECTION v2.0:
# Use LOW collection threshold (50) - let priority queue rank and pick BEST
# Old approach filtered at 70+ BEFORE ranking - missed good signals!
MIN_SIGNAL_SCORE = int(_env_float("LIVE_MIN_SIGNAL_SCORE", _env_float("MIN_SIGNAL_SCORE", 50)))  # COLLECTION threshold - priority queue picks best
HARD_MIN_SCORE = int(_env_float("LIVE_HARD_MIN_SCORE", _env_float("HARD_MIN_SCORE", 50)))  # Absolute floor - adaptive system handles dynamic thresholds

# Operator lock: DISABLED - let adaptive system work
LOCK_ENTRY_FILTERS = (str(os.getenv("LOCK_ENTRY_FILTERS", "0")).strip().lower() in ("1", "true", "yes", "on"))

# LONG BIAS: Score bonus for LONG positions on MAJOR coins (BTC, ETH, SOL, etc.)
# This effectively lowers the threshold from 65 to 60 for high-quality longs
LONG_MAJOR_SCORE_BONUS = _env_float("LONG_MAJOR_SCORE_BONUS", 0.0)

# LONG QUALITY FILTER: Apply stricter requirements for LONGs based on introspection analysis
# LONGs underperform shorts (13.7% vs 34% WR). This filter:
# - Penalizes stale/low-vol LONGs by -10 points (require fresher signals)
# - Small penalty (-5) for off-peak hours (LONGs work best 17-21 UTC)
# Set to "0" or "false" to disable and treat LONGs same as SHORTs
LONG_QUALITY_FILTER_ENABLED = (str(os.getenv("LONG_QUALITY_FILTER_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on"))

# REPLAY MODE: Even in backtesting, prevent excessive trading
REPLAY_MIN_SIGNAL_STRENGTH = env("REPLAY_MIN_SIGNAL_STRENGTH","0.65", float)  # WARNING: 0.65 (was 0.65) - prevent excessive trades even in backtest
# ML SCORER V3: Replay uses same scoring thresholds as live
REPLAY_MIN_SIGNAL_SCORE = env("REPLAY_MIN_SIGNAL_SCORE","32", int)  # ML SCORER: 32 = 42%+ WR trades only
# CRITICAL FIX: REPLAY_HARD_MIN_SCORE must be LOWER than REPLAY_MIN_SIGNAL_SCORE
_replay_hard_min_raw = env("REPLAY_HARD_MIN_SCORE", str(max(1, REPLAY_MIN_SIGNAL_SCORE - 5)), int)  # Default: REPLAY_MIN_SIGNAL_SCORE - 5
REPLAY_HARD_MIN_SCORE = min(_replay_hard_min_raw, REPLAY_MIN_SIGNAL_SCORE - 1)  # Ensure it's always below REPLAY_MIN_SIGNAL_SCORE
SIGNAL_PERCENTILE_THRESHOLD = env("SIGNAL_PERCENTILE_THRESHOLD","0.0", float)  # LIVE MODE: Disabled (0.0 = no percentile filtering) - allows more signals
USE_SIGNAL_PERCENTILE_FILTER = env("USE_SIGNAL_PERCENTILE_FILTER", "0") in ("1","true","TRUE")  # PURE_SCALPER: Disabled by default - percentile filter not used
USE_THREE_STAGE_FILTER = env("USE_THREE_STAGE_FILTER", "1") in ("1","true","TRUE")  # PURE_SCALPER: Enabled by default but softened (microstructure only rejects garbage)
# PHASE 2: Evolvable filter strictness (0.0 = lenient, 1.0 = strict)
# In evolution, this comes from genome. In live bot, use config value.
FILTER_STRICTNESS = env("FILTER_STRICTNESS", "0.5", float)  # Default: moderate strictness

# PHASE 4: Evolvable microstructure thresholds
MAX_SPREAD_PCT = env("MAX_SPREAD_PCT", "0.0025", float)  # Max spread (0.25% default, evolvable)
MIN_DEPTH_USD = env("MIN_DEPTH_USD", "800.0", float)  # Min depth (800 USD default, evolvable)
MAX_VOLATILITY_PCT = env("MAX_VOLATILITY_PCT", "10.0", float)  # BALANCED: 10%
MIN_VOLATILITY_PCT = env("MIN_VOLATILITY_PCT", "0.3", float)  # Min ATR % (0.3% default, evolvable)
SLIPPAGE_BUDGET_BPS = env("SLIPPAGE_BUDGET_BPS", "10.0", float)  # Max acceptable slippage (10 bps default, evolvable)
MIN_VIABLE_SIZE_USD = env("MIN_VIABLE_SIZE_USD", "10.0", float)  # Min order size (10 USD default, evolvable)

SIGNAL_HISTORY_SIZE = env("SIGNAL_HISTORY_SIZE","100", int)  # Track last 100 signals for percentile calculation

# Dynamic Threshold Adjustment (Phase 1: External Review Implementation)
# Dynamic threshold system (keep configurable but OFF by default for PURE_SCALPER)
DYNAMIC_THRESHOLDS_ENABLED = env("DYNAMIC_THRESHOLDS_ENABLED", "0", int) == 1  # PURE_SCALPER: Disabled by default (0 = static thresholds)
THRESHOLD_ADJUSTMENT_WINDOW = env("THRESHOLD_ADJUSTMENT_WINDOW", "50", int)  # Last N trades for win rate calculation
THRESHOLD_ADJUSTMENT_STEP = env("THRESHOLD_ADJUSTMENT_STEP", "2", int)  # Points to adjust (relax/tighten) per adjustment
WIN_RATE_RELAX_THRESHOLD = env("WIN_RATE_RELAX_THRESHOLD", "60.0", float)  # Relax thresholds if win rate > 60%
WIN_RATE_TIGHTEN_THRESHOLD = env("WIN_RATE_TIGHTEN_THRESHOLD", "40.0", float)  # Tighten thresholds if win rate < 40%
# ML SCORER V3: Scores range 25-50 (win probability * 100)
# Dynamic thresholds should stay within this range

# Adaptive Threshold System (Market-aware automatic adjustment)
# Automatically adjusts MIN_SIGNAL_SCORE based on real-time market conditions
# Uses absolute thresholds (not percentiles) to maintain quality standards
USE_ADAPTIVE_THRESHOLDS = env("USE_ADAPTIVE_THRESHOLDS", "0") in ("1", "true", "TRUE")  # Disabled by default
ADAPTIVE_THRESHOLD_MIN = env("ADAPTIVE_THRESHOLD_MIN", "45", int)  # Never below 45 (60% ML prob after quality gate)
ADAPTIVE_THRESHOLD_MAX = env("ADAPTIVE_THRESHOLD_MAX", "75", int)  # Never above 75 (prevents over-restriction)
ADAPTIVE_ADJUSTMENT_INTERVAL = env("ADAPTIVE_ADJUSTMENT_INTERVAL", "300", int)  # Adjust every 5 minutes (300 seconds)
ADAPTIVE_WINDOW_SIZE = env("ADAPTIVE_WINDOW_SIZE", "100", int)  # Analyze last 100 signals for market state
MIN_SCORE_RANGE = (25, 50)  # ML Scorer full range - no artificial clamping
MIN_STRENGTH_RANGE = (0.25, 0.50)  # Aligned with score range (strength = score/100)

# REMOVED: Extended scoring system (archived - unused in trading flow)
# USE_EXTENDED_SCORING - REMOVED
# USE_NEW_SCORING_SYSTEM - REMOVED

# REMOVED: Scoring v2 (archived - only used in analysis tools)
# USE_SCORING_V2 - REMOVED

# Unicorn Found Priority Protocol
UNICORN_SCORE_THRESHOLD = env("UNICORN_SCORE_THRESHOLD", "90", int)  # Score threshold for unicorn signals (>= 90)
UNICORN_PROTOCOL_ENABLED = env("UNICORN_PROTOCOL_ENABLED", "1") in ("1","true","TRUE")  # Enable unicorn priority protocol
UNICORN_POSITION_SIZE_MULTIPLIER = env("UNICORN_POSITION_SIZE_MULTIPLIER", "1.5", float)  # Increase position size by 50% for unicorns
UNICORN_LEVERAGE_MULTIPLIER = env("UNICORN_LEVERAGE_MULTIPLIER", "1.2", float)  # Increase leverage by 20% for unicorns (capped at MAX_LEVERAGE)
UNICORN_BYPASS_COOLDOWN = env("UNICORN_BYPASS_COOLDOWN", "1") in ("1","true","TRUE")  # Bypass cooldowns for unicorns
UNICORN_BYPASS_MAX_POSITIONS = env("UNICORN_BYPASS_MAX_POSITIONS", "0") in ("1","true","TRUE")  # DISABLED: Strictly enforce max positions
UNICORN_EXTRA_POSITION_SLOTS = env("UNICORN_EXTRA_POSITION_SLOTS", "2", int)  # Allow +2 slots for unicorn signals (always get priority)
UNICORN_BYPASS_RATE_LIMIT = env("UNICORN_BYPASS_RATE_LIMIT", "1") in ("1","true","TRUE")  # Bypass rate limits for unicorns
UNICORN_BYPASS_LOSS_STREAK = env("UNICORN_BYPASS_LOSS_STREAK", "1") in ("1","true","TRUE")  # Bypass loss streak protection for unicorns
UNICORN_BYPASS_CORRELATION = env("UNICORN_BYPASS_CORRELATION", "1") in ("1","true","TRUE")  # Allow correlated positions for unicorns

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# EVOLUTION SYSTEM CONFIGURATION
# ----------------------------------------------------------------
# The bot uses evolution-optimized parameters from Phase 2 validation results.
# Parameters are loaded dynamically or set via MIN_SIGNAL_SCORE/STRENGTH below. 

# Symbol scanning and rotation
SYMBOLS_TO_SCAN = env("SYMBOLS_TO_SCAN","300", int)  # Increased to 300 - we have API headroom, catch more opportunities
MAX_ACTIVE_SYMBOLS = env("MAX_ACTIVE_SYMBOLS","1000", int)  # Maximum symbols in active list (expanded to 1000 for maximum opportunity discovery)
STALE_SYMBOL_THRESHOLD_SEC = env("STALE_SYMBOL_THRESHOLD_SEC","60", float)  # Time before symbol considered stale (1 min - faster rotation)
STALE_ROTATION_PCT = env("STALE_ROTATION_PCT","0.50", float)  # Percentage of active list to rotate per cycle (50% - more aggressive)
ROTATION_CHECK_INTERVAL_SEC = env("ROTATION_CHECK_INTERVAL_SEC","30", float)  # How often to check for rotation (30s - faster discovery)

# Universe Configuration
WHITELIST_SYMBOLS = env("WHITELIST_SYMBOLS", "").split(",") if env("WHITELIST_SYMBOLS", "") else []  # Whitelist symbols (empty = all)
WHITELIST_SYMBOLS = [s.strip().upper() for s in WHITELIST_SYMBOLS if s.strip()]  # Clean and normalize
# CRITICAL FIX: Use MIN_VOLUME_24H value, don't override it
# For Binance Futures, we use the configured MIN_VOLUME_24H (5M by default)
MIN_24H_VOLUME_USDT = env("MIN_24H_VOLUME_USDT", None, float)
# Only use MIN_24H_VOLUME_USDT if explicitly set via env var, otherwise use MIN_VOLUME_24H
if MIN_24H_VOLUME_USDT is None:
    MIN_24H_VOLUME_USDT = MIN_VOLUME_24H  # Use MIN_VOLUME_24H value (5M)
# Don't override MIN_VOLUME_24H - use it as the source of truth

# Universe Filtering (New Configs for universe.py)
UNIVERSE_MIN_VOLUME = env("UNIVERSE_MIN_VOLUME", "1000000", float)  # $1M minimum volume for universe inclusion
UNIVERSE_MAX_SPREAD = env("UNIVERSE_MAX_SPREAD", "100", float)  # 100 bps max spread for universe inclusion

# ===========================
# SYMBOL BLACKLIST (Data-Driven)
# ===========================
# These symbols are statistically significant underperformers based on 4M+ trades analysis.
# They have win rates 2+ standard deviations below average (z < -1.96).
# Format: comma-separated list of base symbols (without /USDT suffix)
_BLACKLIST_DEFAULT = "BTC,ETC,ATOM,ADA,BCH,XRP,BNB,AVAX,FLOW"
SYMBOL_BLACKLIST = [s.strip().upper() for s in env("SYMBOL_BLACKLIST", _BLACKLIST_DEFAULT).split(",") if s.strip()]

# ===========================
# MINIMUM ATR FILTER (Skip Low Volatility)
# ===========================
# Analysis shows 56.5% SL rate when ATR < 0.5% vs 30.2% when ATR > 2%
# Low volatility = tight SL relative to noise = more stop outs
MIN_ATR_PCT = env("MIN_ATR_PCT", "0.5", float) / 100.0  # 0.5% minimum ATR to trade

# ===========================
# ===========================
# DIAMOND HANDS EXIT STRATEGY (Data-Driven from 4M+ trades)
# ===========================
# KEY INSIGHT: Time exits with 1-hour hold = +0.27R avg, 72% WR
#              Stop losses killing trades early = -0.43R avg
#
# Strategy: DISABLE stop losses during hold period, let trades run
#
# Analysis results:
#   - 5-15 min holds: 11.7% WR (stop loss noise)
#   - 60+ min holds: 55.9% WR (time to recover)
#   - Total R with stops: -451,044 (losing)
#   - Total R without stops: +505,635 (profitable!)
#
# REPLACED: Hard 60-minute hold time with intelligent PRS-based exits
# NEW APPROACH: Early protection + PRS monitoring + ATR trailing
#
# Phase 1 (0-15 min): Early protection - no trailing stops
EARLY_PROTECTION_MIN = env("EARLY_PROTECTION_MIN", "15", float)  # No trailing for first 15 minutes
EARLY_PROTECTION_ALLOW_EMERGENCY = env("EARLY_PROTECTION_ALLOW_EMERGENCY", "1") in ("1", "true", "TRUE")  # Allow emergency exits if -5%+

# Phase 2 (15+ min): PRS monitoring active (configured below in PRS section)
# Phase 3 (profit > 1R): ATR trailing active (configured below in trailing section)

# DEPRECATED: Hard time limits (replaced by adaptive system)
MIN_HOLD_TIME_SEC = env("MIN_HOLD_TIME_SEC", "0", int)  # DISABLED - no hard minimum (was 3600)
DISABLE_STOP_DURING_HOLD = env("DISABLE_STOP_DURING_HOLD", "0") in ("1", "true", "TRUE")  # DISABLED - let smart systems work
MIN_HOLD_PROFIT_EXCEPTION_PCT = env("MIN_HOLD_PROFIT_EXCEPTION_PCT", "2.0", float) / 100.0  # DEPRECATED
TIME_EXIT_AFTER_HOLD = env("TIME_EXIT_AFTER_HOLD", "0") in ("1", "true", "TRUE")  # DISABLED

# Discovery scanning (Option C: Quality Over Quantity)
# OPTIMIZATION: Reduced from 15s to 10s for more frequent discovery
DISCOVERY_SCAN_INTERVAL_SEC = env("DISCOVERY_SCAN_INTERVAL_SEC","5", float)  # OPTIMIZED: 5s (was 3s) to reduce API load
DISCOVERY_SYMBOLS_PER_CYCLE = env("DISCOVERY_SYMBOLS_PER_CYCLE","50", int)  # Number of discovery symbols to scan per cycle (quantity!)
DISCOVERY_MIN_VOLUME_24H = env("DISCOVERY_MIN_VOLUME_24H","1000000", float)  # Lowered to $1M for maximum breadth
DISCOVERY_MAX_SPREAD_BPS = env("DISCOVERY_MAX_SPREAD_BPS","100", float)  # Relaxed spread threshold for discovery (100 bps)
DISCOVERY_MIN_MOMENTUM_PCT = env("DISCOVERY_MIN_MOMENTUM_PCT","1.0", float)  # Minimum momentum for discovery candidates (1.0%)

# Signal scoring parameters
IDEAL_SPREAD_BPS = env("IDEAL_SPREAD_BPS","20", float)  # Ideal spread for SpreadScore calculation
IDEAL_LATENCY_MS = env("IDEAL_LATENCY_MS","10", int)  # Ideal latency for LatencyScore calculation
MIN_ORDERBOOK_DEPTH_PCT = env("MIN_ORDERBOOK_DEPTH_PCT","0.05", float)  # Minimum orderbook depth (5% of order size) for DepthScore

# ----------------------------------------------------------------
# ADAPTIVE ENTRY CONTROLLER
# ----------------------------------------------------------------
# Dynamically adjusts entry threshold to stay active 24/7 while taking best signals
# Threshold = percentile of recent signals, adjusted for position count & idle time
# ML-OPTIMIZED: Removed USE_ADAPTIVE_ENTRY (now using fixed thresholds from environment)

# Data collection requirement (blocks trading until sufficient market data)
ADAPTIVE_MIN_SIGNALS_REQUIRED = env("ADAPTIVE_MIN_SIGNALS_REQUIRED", "50", int)  # Minimum signals needed before trading starts

# Threshold bounds (safety rails)
ADAPTIVE_HARD_FLOOR = env("ADAPTIVE_HARD_FLOOR", "25", float)   # NEVER go below this (garbage protection)
ADAPTIVE_CEILING = env("ADAPTIVE_CEILING", "70", float)         # Maximum threshold (don't be too picky)

# Percentile targets (what % of signals to consider taking)
# Lower percentile = more selective (e.g., 15 = top 15% only)
ADAPTIVE_IDLE_PERCENTILE = env("ADAPTIVE_IDLE_PERCENTILE", "40", float)    # When no positions: top 40%
ADAPTIVE_NORMAL_PERCENTILE = env("ADAPTIVE_NORMAL_PERCENTILE", "25", float) # With some positions: top 25%
ADAPTIVE_FULL_PERCENTILE = env("ADAPTIVE_FULL_PERCENTILE", "15", float)    # Near max positions: top 15%

# Position targets (influences how aggressive threshold is)
ADAPTIVE_TARGET_POSITIONS = env("ADAPTIVE_TARGET_POSITIONS", "3", int)  # Ideal number of positions

# Activity decay (lower threshold over time to ensure activity)
ADAPTIVE_DECAY_START_MIN = env("ADAPTIVE_DECAY_START_MIN", "5", float)  # Start lowering after 5 min idle
ADAPTIVE_DECAY_RATE = env("ADAPTIVE_DECAY_RATE", "0.5", float)          # Lower by 0.5 per minute
ADAPTIVE_MAX_DECAY = env("ADAPTIVE_MAX_DECAY", "10", float)             # Maximum decay of 10 points

# Legacy thresholds (used as fallback if adaptive disabled)
# ML-OPTIMIZED: Removed IDLE_MIN_SCORE (no longer lowering threshold when idle)

# Position sizing
USE_KELLY_SIZING   = env("USE_KELLY_SIZING","1") in ("1","true","TRUE")  # Use Kelly-adjusted sizing
MIN_POSITION_SIZE  = env("MIN_POSITION_SIZE","10", float)  # Minimum position size in USDT
MAX_POSITION_SIZE  = env("MAX_POSITION_SIZE","50", float)  # Maximum position size in USDT

# Paths
AUDIT_PATH   = env("AUDIT_PATH","trades_audit.csv")
STATE_PATH   = env("STATE_PATH","state.json")
TUNING_LOG   = env("TUNING_LOG","llm_tuning_log.csv")
CONTROL_PATH = env("CONTROL_PATH","control_patch.json")
MODES_PATH   = env("MODES_PATH","modes.json")
ENVELOPE_PATH= env("ENVELOPE_PATH","envelopes.json")

# HTTP
EXCHANGE_TIMEOUT_MS = env("EXCHANGE_TIMEOUT_MS","20000", int)
EXCHANGE_RETRIES    = env("EXCHANGE_RETRIES","3", int)

# ----------------------------------------------------------------
# API GOVERNOR (Precision Clockwork)
# ----------------------------------------------------------------
# Single gateway for all exchange calls:
# - Rate budget (token bucket): prevents exceeding Binance weight/min
# - In-flight budget (semaphore): prevents CCXT throttle queue maxCapacity overflow
API_GOVERNOR_ENABLED = env("API_GOVERNOR_ENABLED", "1") in ("1", "true", "TRUE")
API_TARGET_WEIGHT_PER_MIN = env("API_TARGET_WEIGHT_PER_MIN", "1800", int)  # 75% of 2400/min
API_BURST = env("API_BURST", "60", int)  # Large burst buffer to accommodate heavy init calls (was 10)
API_MAX_INFLIGHT = env("API_MAX_INFLIGHT", "25", int)  # cap concurrent CCXT calls
API_EXIT_TOKEN_RESERVE = env("API_EXIT_TOKEN_RESERVE", "3", int)  # reserve tokens for exits
API_CALL_TIMEOUT_SEC = env("API_CALL_TIMEOUT_SEC", "3.0", float)  # hard timeout per API call

# Magic numbers extracted to constants
MIN_STOP_DISTANCE_PCT = env("MIN_STOP_DISTANCE_PCT", "1.5", float) / 100.0  # ML-optimized: 1.5% minimum stop distance (was 2.0% - tighter)
# Minimum risk per trade (USD) - DRY_RUN only
# If risk_usd < MIN_RISK_USD we skip the trade as "too small"
MIN_RISK_USD = env("MIN_RISK_USD", "0.01", float)  # LOWERED TO 0.01 to allow small test trades
# OPTIMIZATION: Based on backtest analysis - activate trailing earlier to reduce time exits
TRAILING_STOP_ACTIVATION_PCT = env("TRAILING_STOP_ACTIVATION_PCT", "0.5", float)  # Activate at 0.5% profit (reduced from 0.7% to lock profits earlier)
TRAILING_STOP_PCT = env("TRAILING_STOP_PCT", "60.0", float) / 100.0  # Trail stop to 60% of profit (increased from 50% to lock more profit)

# ATR-Based Trailing Stops (Phase 1: External Review Implementation)
USE_ATR_TRAILING_STOP = env("USE_ATR_TRAILING_STOP", "1") in ("1","true","TRUE")  # Enable ATR-based trailing stops
ATR_TRAILING_MULTIPLIER = env("ATR_TRAILING_MULTIPLIER", "2.5", float)  # Default 2.5x ATR behind peak (wider for big moves)

# BINANCE SERVER-SIDE STOP LOSS (runs on exchange, not bot)
# This places a stop-loss order on Binance as a "circuit breaker"
# Benefits: Works even if bot crashes - disaster protection
# DATA-DRIVEN: Analysis shows 56.5% SL rate at 1% vs 30.2% at 2%+
# This should be WIDER than the bot's internal trailing (which handles normal exits)
USE_BINANCE_TRAILING_STOP = env("USE_BINANCE_TRAILING_STOP", "1") in ("1","true","TRUE")  # Enable Binance stop-loss
BINANCE_TRAILING_CALLBACK_RATE = env("BINANCE_TRAILING_CALLBACK_RATE", "2.5", float)  # Stop distance in % (2.5% = circuit breaker)
ATR_TRAILING_MIN_DISTANCE_PCT = env("ATR_TRAILING_MIN_DISTANCE_PCT", "0.5", float) / 100.0  # Minimum 0.5% distance
ATR_TRAILING_SCALPING_MULTIPLIER = env("ATR_TRAILING_SCALPING_MULTIPLIER", "1.5", float)  # Tighter for scalping
ATR_TRAILING_DAY_MULTIPLIER = env("ATR_TRAILING_DAY_MULTIPLIER", "2.0", float)  # Balanced for day trading
ATR_TRAILING_SWING_MULTIPLIER = env("ATR_TRAILING_SWING_MULTIPLIER", "2.5", float)  # Wider for swing trading

# R-Based Trailing Stop Engine (canonical trailing)
# ADAPTIVE STRATEGY: Disabled aggressive R-based trailing, use ATR-based instead
USE_TRAILING_ENGINE = env("USE_TRAILING_ENGINE", "0") in ("1", "true", "TRUE")  # DISABLED - too aggressive
USE_NEW_TRAILING_ENGINE = env("USE_NEW_TRAILING_ENGINE", "0") in ("1", "true", "TRUE")  # DISABLED
# If re-enabled, start at 1R minimum (not 0.3R which causes premature exits)
TRAIL_ENGINE_START_BUFFER_R = env("TRAIL_ENGINE_START_BUFFER_R", "1.0", float)  # Start at 1R minimum (was 0.3R - too early!)
TRAIL_ENGINE_PARTIAL_1_R = env("TRAIL_ENGINE_PARTIAL_1_R", "1.0", float)  # First partial at +1R
TRAIL_ENGINE_PARTIAL_1_SIZE = env("TRAIL_ENGINE_PARTIAL_1_SIZE", "0.25", float)  # Default 25% clip
TRAIL_ENGINE_PARTIAL_1_SL_OFFSET_R = env("TRAIL_ENGINE_PARTIAL_1_SL_OFFSET_R", "0.2", float)  # Keep SL slightly negative until proven
TRAIL_ENGINE_BREAK_EVEN_R = env("TRAIL_ENGINE_BREAK_EVEN_R", "1.5", float)  # Move to BE + buffer at 1.5R
TRAIL_ENGINE_BE_BUFFER_R = env("TRAIL_ENGINE_BE_BUFFER_R", "0.25", float)
TRAIL_ENGINE_PARTIAL_2_R = env("TRAIL_ENGINE_PARTIAL_2_R", "2.0", float)
TRAIL_ENGINE_PARTIAL_2_SIZE = env("TRAIL_ENGINE_PARTIAL_2_SIZE", "0.25", float)
TRAIL_ENGINE_LOCK_R_LEVEL = env("TRAIL_ENGINE_LOCK_R_LEVEL", "2.0", float)
TRAIL_ENGINE_LOCK_AMOUNT_R = env("TRAIL_ENGINE_LOCK_AMOUNT_R", "1.0", float)
TRAIL_ENGINE_RUNNER_START_R = env("TRAIL_ENGINE_RUNNER_START_R", "3.0", float)
TRAIL_ENGINE_RUNNER_TRAIL_DISTANCE_R = env("TRAIL_ENGINE_RUNNER_TRAIL_DISTANCE_R", "1.25", float)
TRAIL_ENGINE_MIN_R_INCREMENT = env("TRAIL_ENGINE_MIN_R_INCREMENT", "0.5", float)
TRAIL_ENGINE_MIN_UPDATE_SECONDS = env("TRAIL_ENGINE_MIN_UPDATE_SECONDS", "20", float)
BTC_TREND_THRESHOLD = env("BTC_TREND_THRESHOLD", "0.3", float)  # BTC trend threshold for correlation
WIDE_SPREAD_EXIT_THRESHOLD_BPS = env("WIDE_SPREAD_EXIT_THRESHOLD_BPS", "50", float)  # Exit if spread > 50 bps
EARLY_STOP_LOSS_PCT = env("EARLY_STOP_LOSS_PCT", "0.8", float)  # Early exit at -0.8% loss
EARLY_STOP_LOSS_MAX_PCT = env("EARLY_STOP_LOSS_MAX_PCT", "1.2", float)  # Maximum early stop loss
EARLY_STOP_MIN_TIME_SEC = env("EARLY_STOP_MIN_TIME_SEC", "60", float)  # Minimum time in position before early stop (60s)
EARLY_STOP_CONFIRMATION_COUNT = env("EARLY_STOP_CONFIRMATION_COUNT", "2", int)  # Require 2 consecutive checks before early stop

# Stale Position Kill-Switch (Prevent Dead Weight)
# Increased to maintain 1+ positions on average - only kill truly dead positions
MAX_POSITION_AGE_SEC = env("MAX_POSITION_AGE_SEC", "3600", int)  # Maximum position age: 60 minutes (3600s, increased from 30min)
STALE_POSITION_PNL_THRESHOLD = env("STALE_POSITION_PNL_THRESHOLD", "0.3", float)  # Auto-close if |PnL| < 0.3% after max age (more lenient)
STALE_POSITION_CHECK_INTERVAL_SEC = env("STALE_POSITION_CHECK_INTERVAL_SEC", "60", int)  # Check every 60 seconds
# Position Recovery Score (PRS) guardrail - minimum age (minutes) before PRS can affect exits
PRS_MIN_AGE_MIN = env("PRS_MIN_AGE_MIN", "15", float)

# Decision log rotation
DECISION_LOG_MAX_MB = env("DECISION_LOG_MAX_MB", "50", float)
DECISION_LOG_MAX_FILES = env("DECISION_LOG_MAX_FILES", "20", int)

# Decision Logging Filters (RECOMMENDATION #5: Reduce logging volume)
LOG_ALL_APPROVALS = env("LOG_ALL_APPROVALS", "1") in ("1","true","TRUE")  # Always log approved signals
LOG_REJECTION_MIN_SCORE = env("LOG_REJECTION_MIN_SCORE", "50", float)  # Only log rejections above this score
LOG_REJECTION_SAMPLE_RATE = env("LOG_REJECTION_SAMPLE_RATE", "0.1", float)  # Sample rate for low-score rejections (0.1 = 10%)

# Extended Stale Position Rules
STALE_90MIN_AGE_SEC = env("STALE_90MIN_AGE_SEC", "5400", int)  # 90 minutes (5400s) - consider closing
STALE_90MIN_PNL_THRESHOLD = env("STALE_90MIN_PNL_THRESHOLD", "0.5", float)  # Close if PnL < 0.5% at 90m
EXTENDED_LEASH_PNL_THRESHOLD = env("EXTENDED_LEASH_PNL_THRESHOLD", "0.8", float)  # If PnL > 0.8%, extend to 120m
EXTENDED_LEASH_AGE_SEC = env("EXTENDED_LEASH_AGE_SEC", "7200", int)  # 120 minutes (7200s) extended leash
STALE_DRAWDOWN_RESUME_THRESHOLD = env("STALE_DRAWDOWN_RESUME_THRESHOLD", "-0.2", float)  # If PnL drops to -0.2% while stale, cut earlier

# Drawdown Circuit Breaker (Live Trading Safety)
MAX_DRAWDOWN_PCT = env("MAX_DRAWDOWN_PCT", "5.0", float)  # Maximum allowed drawdown (5%) - circuit breaker threshold
DRAWDOWN_CIRCUIT_BREAKER_ENABLED = env("DRAWDOWN_CIRCUIT_BREAKER_ENABLED", "1") in ("1","true","TRUE")  # Enable drawdown circuit breaker

# Emergency Position Loss Limit (Prevents catastrophic losses like PIPPIN)
# CRITICAL: Force exit if any single position loses more than this percentage
MAX_POSITION_LOSS_PCT = env("MAX_POSITION_LOSS_PCT", "-8.0", float)  # Default: -8% hard limit per position

# Disable Partial Exits (Prevents dust positions on Binance)
# When enabled, ALL exits are 100% (no partial exits, no scale-outs)
DISABLE_PARTIAL_EXITS = env("DISABLE_PARTIAL_EXITS", "1") in ("1", "true", "TRUE")  # Default: Enabled (no partials)

# Loss streak and risk controls (calibration overrides)
MAX_LOSS_STREAK_HARD = env("MAX_LOSS_STREAK_HARD", "999", int)  # Hard block disabled for calibration - no permanent lockout
LOSS_STREAK_CAUTION_LEVEL = env("LOSS_STREAK_CAUTION_LEVEL", "3", int)  # Log caution from 3 losses onwards (no block, just logging)
LOSS_STREAK_AUTO_RESET_SEC = env("LOSS_STREAK_AUTO_RESET_SEC", "600", int)  # Auto-reset after 10 minutes without new loss (600s)

# Tiered/DD-aware loss streak protections (disabled for calibration run)
TIERED_LOSS_STREAK_ENABLED = env("TIERED_LOSS_STREAK_ENABLED", "0") in ("1","true","TRUE")
LOSS_STREAK_TIER1 = env("LOSS_STREAK_TIER1", "3", int)
LOSS_STREAK_TIER2 = env("LOSS_STREAK_TIER2", "5", int)
LOSS_STREAK_TIER1_RISK_MULTIPLIER = env("LOSS_STREAK_TIER1_RISK_MULTIPLIER", "0.7", float)
LOSS_STREAK_TIER2_RISK_MULTIPLIER = env("LOSS_STREAK_TIER2_RISK_MULTIPLIER", "0.5", float)
LOSS_STREAK_TIER1_POSITION_MULTIPLIER = env("LOSS_STREAK_TIER1_POSITION_MULTIPLIER", "0.8", float)
LOSS_STREAK_TIER2_POSITION_MULTIPLIER = env("LOSS_STREAK_TIER2_POSITION_MULTIPLIER", "0.6", float)
LOSS_STREAK_TIER1_SCORE_BOOST = env("LOSS_STREAK_TIER1_SCORE_BOOST", "0", int)
LOSS_STREAK_TIER2_SCORE_BOOST = env("LOSS_STREAK_TIER2_SCORE_BOOST", "0", int)

DD_AWARE_LOSS_STREAK_ENABLED = env("DD_AWARE_LOSS_STREAK_ENABLED", "0") in ("1","true","TRUE")
DD_AWARE_STREAK_THRESHOLD = env("DD_AWARE_STREAK_THRESHOLD", "3", int)
DD_AWARE_DD_THRESHOLD = env("DD_AWARE_DD_THRESHOLD", "5.0", float)
AUTO_RESET_LOSS_STREAK_ENABLED = env("AUTO_RESET_LOSS_STREAK_ENABLED", "1") in ("1","true","TRUE")
AUTO_RESET_TIME_SEC = env("AUTO_RESET_TIME_SEC", "600", int)
AUTO_RESET_WINNER_PNL_THRESHOLD = env("AUTO_RESET_WINNER_PNL_THRESHOLD", "0", float)
AUTO_RESET_DD_IMPROVEMENT_THRESHOLD = env("AUTO_RESET_DD_IMPROVEMENT_THRESHOLD", "0", float)

# Loss streak state machine (explicit thresholds)
LOSS_STREAK_DEFENSE_LEVEL = env("LOSS_STREAK_DEFENSE_LEVEL", "5", int)  # 5-6 -> defense
LOSS_STREAK_HARD_LEVEL = env("LOSS_STREAK_HARD_LEVEL", "7", int)        # >=7 -> pause
LOSS_STREAK_PAUSE_SEC = env("LOSS_STREAK_PAUSE_SEC", "900", int)        # 15 minutes pause
LOSS_STREAK_DECAY_SECONDS = env("LOSS_STREAK_DECAY_SECONDS", "900", int) # Auto-decay by 1 after 15 minutes
HIGH_SCORE_BYPASS = env("HIGH_SCORE_BYPASS", "90.0", float)              # Unicorn threshold for bypass

# Break-Even Rescue Protocol (BERP)
BERP_ENABLED = env("BERP_ENABLED", "1") in ("1","true","TRUE")  # Enable Break-Even Rescue Protocol
BERP_TRIGGER_AGE_SEC = env("BERP_TRIGGER_AGE_SEC", "3600", int)  # Trigger rescue at 60 minutes (3600s)
BERP_TRIGGER_PNL_THRESHOLD = env("BERP_TRIGGER_PNL_THRESHOLD", "0.3", float)  # Trigger if PnL < +0.3% (soft noise margin)
BERP_RESCUE_DURATION_SEC = env("BERP_RESCUE_DURATION_SEC", "3600", int)  # Rescue duration: 60 minutes (3600s)
BERP_PROFIT_OVERRIDE_PCT = env("BERP_PROFIT_OVERRIDE_PCT", "1.0", float)  # If profit >= 1.0% at 60m, extend hold + trail tight (don't close dead positions)

# ===================================================================
# PHASE 2: ADVANCED CANDLESTICK PATTERN RECOGNITION
# ===================================================================
PHASE2_ENABLED = env("PHASE2_ENABLED", "1") in ("1", "true", "TRUE")  # ENABLED by default for testing
PHASE2_HAMMER_BOOST = env("PHASE2_HAMMER_BOOST", "0.25", float)  # Hammer reversal: +0.25 boost (increased from 0.15)
PHASE2_ENGULFING_BOOST = env("PHASE2_ENGULFING_BOOST", "0.20", float)  # Engulfing confirmation: +0.20 boost (increased from 0.10)
PHASE2_DOJI_PENALTY = env("PHASE2_DOJI_PENALTY", "-0.15", float)  # Doji indecision: -0.15 penalty (increased from -0.08)
PHASE2_PIN_BAR_BOOST = env("PHASE2_PIN_BAR_BOOST", "0.20", float)  # Pin bar rejection: +0.20 boost (increased from 0.10)

# Phase 3 (Mean Reversion): Rejection zones & bounce confirmation
PHASE3_ENABLED = env("PHASE3_ENABLED", "1") in ("1", "true", "TRUE")  # ENABLED by default - mean reversion patterns
PHASE3_REJECTION_ZONE_BOOST = env("PHASE3_REJECTION_ZONE_BOOST", "0.15", float)  # Rejection zone: +0.15 boost (highest MR confidence)
PHASE3_BOUNCE_CONFIRMATION_BOOST = env("PHASE3_BOUNCE_CONFIRMATION_BOOST", "0.10", float)  # Bounce confirmation: +0.10 boost

# R-Based Exit Engine (Score-Aware Exit Management)
USE_R_BASED_EXITS = env("USE_R_BASED_EXITS", "1") in ("1","true","TRUE")  # Enable R-based exit engine
# ML SCORER V3: Exit profiles based on ML score ranges (25-50)
R_EXIT_SCALP_SCORE_MIN = env("R_EXIT_SCALP_SCORE_MIN", "25", int)  # ML: Low quality (25-34) = quick exits
R_EXIT_SCALP_SCORE_MAX = env("R_EXIT_SCALP_SCORE_MAX", "34", int)  # ML: Scalp range max
R_EXIT_STANDARD_SCORE_MIN = env("R_EXIT_STANDARD_SCORE_MIN", "35", int)  # ML: Mid-range (35-44) = balanced exits
R_EXIT_STANDARD_SCORE_MAX = env("R_EXIT_STANDARD_SCORE_MAX", "44", int)  # ML: Standard range max
R_EXIT_RUNNER_SCORE_MIN = env("R_EXIT_RUNNER_SCORE_MIN", "45", int)  # ML: High quality (45+) = let winners run

# Scalp Profile (60-69): Fast exits, no trailing
R_SCALP_TP_R = env("R_SCALP_TP_R", "1.0", float)  # Take profit at 1.0R
R_SCALP_TIME_STOP_BARS = env("R_SCALP_TIME_STOP_BARS", "35", int)  # Time-stop after 35 bars if not hit +/-0.5R
R_SCALP_BOREDOM_RANGE = env("R_SCALP_BOREDOM_RANGE", "0.5", float)  # Exit if stays within +/-0.5R for time-stop period

# Standard Profile (70-85): Partial + trailing
R_STANDARD_PARTIAL_TP_R = env("R_STANDARD_PARTIAL_TP_R", "1.0", float)  # Partial TP at 1.0R
R_STANDARD_PARTIAL_PCT = env("R_STANDARD_PARTIAL_PCT", "50", float) / 100.0  # Close 50% at partial TP
# OPTIMIZATION: Start trailing earlier for standard profile to reduce time exits
R_STANDARD_TRAIL_START_R = env("R_STANDARD_TRAIL_START_R", "1.2", float)  # Start trailing at 1.2R (reduced from 1.5R to activate earlier)
R_STANDARD_TRAIL_ATR_MULT = env("R_STANDARD_TRAIL_ATR_MULT", "0.75", float)  # Trail at 0.75x ATR
R_STANDARD_MAX_R = env("R_STANDARD_MAX_R", "3.0", float)  # Hard cap target around 3R
R_STANDARD_TIME_STOP_BARS = env("R_STANDARD_TIME_STOP_BARS", "50", int)  # Time-stop after 50 bars if not hit +/-0.5R
R_STANDARD_BOREDOM_RANGE = env("R_STANDARD_BOREDOM_RANGE", "0.5", float)  # Exit if stays within +/-0.5R for time-stop period

# Runner Profile (90+): Small partial + aggressive trailing
R_RUNNER_PARTIAL_TP_R = env("R_RUNNER_PARTIAL_TP_R", "1.0", float)  # Small partial at 1.0R
R_RUNNER_PARTIAL_PCT = env("R_RUNNER_PARTIAL_PCT", "25", float) / 100.0  # Close 25% at partial TP
R_RUNNER_BE_MOVE_R = env("R_RUNNER_BE_MOVE_R", "0.25", float)  # Move SL to +0.25R at 1R (or breakeven)
R_RUNNER_TRAIL_START_R = env("R_RUNNER_TRAIL_START_R", "1.5", float)  # Start trailing at 1.5R
R_RUNNER_TRAIL_ATR_MULT_NORMAL = env("R_RUNNER_TRAIL_ATR_MULT_NORMAL", "1.0", float)  # Trail at 1.0x ATR (normal vol)
R_RUNNER_TRAIL_ATR_MULT_HIGH = env("R_RUNNER_TRAIL_ATR_MULT_HIGH", "1.2", float)  # Trail at 1.2x ATR (high vol)
R_RUNNER_MAX_R_NORMAL = env("R_RUNNER_MAX_R_NORMAL", "4.0", float)  # Target 3-4R in normal volatility
R_RUNNER_MAX_R_HIGH = env("R_RUNNER_MAX_R_HIGH", "5.0", float)  # Target 4-5R in high volatility
R_RUNNER_TIME_STOP_BARS = env("R_RUNNER_TIME_STOP_BARS", "100", int)  # Much laxer time-stop (100 bars)
R_RUNNER_BOREDOM_RANGE = env("R_RUNNER_BOREDOM_RANGE", "0.3", float)  # Exit if stays below +0.3R after entry phase

# Bar tracking (for time-stop calculations)
# Since this is a scalping bot, we'll use scan cycles as "bars"
# Each scan cycle counts as 1 bar
R_BAR_SCAN_CYCLE_SEC = env("R_BAR_SCAN_CYCLE_SEC", "30", int)  # Approximate scan cycle time (30 seconds)

# Signal Confirmation Window (SCW)
USE_SIGNAL_CONFIRMATION = env("USE_SIGNAL_CONFIRMATION", "0") in ("1","true","TRUE")  # Disabled for calibration run
SCW_UNICORN_CONFIRM = env("SCW_UNICORN_CONFIRM", "0", int)  # Unicorn (90+): 0 bars confirmation
SCW_STANDARD_CONFIRM = env("SCW_STANDARD_CONFIRM", "2", int)  # Standard (70-89): 2 bars confirmation
SCW_SCALP_CONFIRM = env("SCW_SCALP_CONFIRM", "1", int)  # Scalp (60-69): 1 bar confirmation (Shoot on sight)
SCW_MAX_SPREAD_PCT = env("SCW_MAX_SPREAD_PCT", "0.08", float)  # Max spread for confirmation (0.08%)
SCW_MAX_BODY_ATR_MULT = env("SCW_MAX_BODY_ATR_MULT", "2.0", float)  # Max candle body = 2x ATR
SCW_VOLUME_STABLE_MULT = env("SCW_VOLUME_STABLE_MULT", "1.5", float)  # Volume <= 1.5x volume_ma
SCW_VALIDATION_BARS = env("SCW_VALIDATION_BARS", "10", int)  # Bars to check for validation

# Rank-Based Position Allocation (RPA)
USE_RANK_BASED_ALLOCATION = env("USE_RANK_BASED_ALLOCATION", "1") in ("1","true","TRUE")  # Enable RPA
RPA_MIN_SIZE_MULT = env("RPA_MIN_SIZE_MULT", "0.50", float)  # Minimum size multiplier (0.50 = 50%)
RPA_MAX_SIZE_MULT = env("RPA_MAX_SIZE_MULT", "1.00", float)  # Maximum size multiplier (1.00 = 100%)
RPA_MIN_SIZE_USD = env("RPA_MIN_SIZE_USD", "5", float)  # Minimum position size in USD
RPA_MAX_RISK_BUDGET_PCT = env("RPA_MAX_RISK_BUDGET_PCT", "0.5", float)  # Max position = 50% of total risk budget

# Score-Weighted Trailing Algorithm (SWTA)
USE_SWTA = env("USE_SWTA", "1") in ("1","true","TRUE")  # Enable SWTA
SWTA_BASE_MULTIPLIER = env("SWTA_BASE_MULTIPLIER", "1.6", float)  # More responsive trailing (was 2.0)
SWTA_START_R = env("SWTA_START_R", "1.5", float)  # Start trailing at 1.5R

# Multi-Stage Exit Framework (MSX)
USE_MSX = env("USE_MSX", "1") in ("1","true","TRUE")  # Enable MSX framework
# -------------------------------------------------------------
# MSX EXIT ENGINE SETTINGS
# Stage-1 = initial validation to avoid instant-regret fills.
# For scalping, waiting 3 bars is too slow and causes missed partials/trailing.
# 1 bar ensures MSX activates quickly after the first bar closes,
# which matches scalping behaviour while still filtering bad immediate entries.
MSX_STAGE1_VALIDATION_R = env("MSX_STAGE1_VALIDATION_R", "-0.3", float)  # Exit if -0.3R within validation bars
MSX_STAGE1_VALIDATION_BARS = env("MSX_STAGE1_VALIDATION_BARS", "1", int)
MSX_PARTIAL_SCALP_PCT = env("MSX_PARTIAL_SCALP_PCT", "50", float) / 100.0  # Scalp (60-69): 50% partial
MSX_PARTIAL_STANDARD_PCT = env("MSX_PARTIAL_STANDARD_PCT", "40", float) / 100.0  # Standard (70-89): 40% partial
MSX_PARTIAL_RUNNER_PCT = env("MSX_PARTIAL_RUNNER_PCT", "25", float) / 100.0  # Runner (90+): 25% partial
MSX_UNICORN_BE_R = env("MSX_UNICORN_BE_R", "0.20", float)  # Unicorn BE = entry + 0.20R (more aggressive)
MSX_TIME_STOP_SCALP_BARS = env("MSX_TIME_STOP_SCALP_BARS", "20", int)  # Scalp time-stop (was 30)
MSX_TIME_STOP_STANDARD_BARS = env("MSX_TIME_STOP_STANDARD_BARS", "40", int)  # Standard time-stop (was 60)
MSX_TIME_STOP_RUNNER_BARS = env("MSX_TIME_STOP_RUNNER_BARS", "80", int)  # Runner time-stop (was 120)
MSX_TIME_STOP_MIN_R = env("MSX_TIME_STOP_MIN_R", "0.5", float)  # Time-stop if max_r < 0.5R (was 1.0)

# MSX Stage 1 calibration controls (new)
MSX_STAGE1_ENABLED = env("MSX_STAGE1_ENABLED", "1") in ("1","true","TRUE")
MSX_EARLY_INVALIDATION_R = env("MSX_EARLY_INVALIDATION_R", "-0.5", float)  # Relaxed from -0.3R
MSX_VOL_SPIKE_MULT = env("MSX_VOL_SPIKE_MULT", "2.5", float)  # ATR spike multiplier threshold
MSX_MAX_SPREAD_STAGE1 = env("MSX_MAX_SPREAD_STAGE1", "0.0012", float)  # 12 bps

KELLY_FRACTION = env("KELLY_FRACTION", "0.5", float)  # Use half-Kelly for safety
MAX_KELLY_PCT = env("MAX_KELLY_PCT", "20.0", float) / 100.0  # Cap at 20% Kelly - reduced for live trading safety
WIN_LOSS_RATIO_ASSUMPTION = env("WIN_LOSS_RATIO_ASSUMPTION", "2.0", float)  # Assume 2:1 R:R
MIN_VOLUME_LOG_THRESHOLD = env("MIN_VOLUME_LOG_THRESHOLD", "6", float)  # log10($1M) = 6
MAX_PRICE_SANITY_CHECK = env("MAX_PRICE_SANITY_CHECK", "1e10", float)  # Maximum reasonable price
MAX_SIZE_SANITY_CHECK = env("MAX_SIZE_SANITY_CHECK", "1e6", float)  # Maximum reasonable size

# Magic number constants (extracted for clarity)
RETRY_DELAY_MULTIPLIER = env("RETRY_DELAY_MULTIPLIER", "1.5", float)  # Multiplier for retry delays
CORRELATION_NORMALIZATION_FACTOR = env("CORRELATION_NORMALIZATION_FACTOR", "2.5", float)  # Normalization factor for correlation strength
MID_PRICE_FACTOR = 0.5  # Factor for calculating mid price (0.5 = average of bid/ask)
BTC_TREND_WEAK_THRESHOLD = env("BTC_TREND_WEAK_THRESHOLD", "0.5", float)  # BTC trend threshold for weak trend detection
STRONG_MOMENTUM_THRESHOLD = env("STRONG_MOMENTUM_THRESHOLD", "2.0", float)  # Strong momentum threshold (%)
HIGH_VOLATILITY_MEDIAN_THRESHOLD = env("HIGH_VOLATILITY_MEDIAN_THRESHOLD", "2.5", float)  # High volatility median threshold (%)
HIGH_VOLATILITY_P75_THRESHOLD = env("HIGH_VOLATILITY_P75_THRESHOLD", "3.5", float)  # High volatility 75th percentile threshold (%)

# ----------------------------------------------------------------
# MASTER HINDSIGHT ML SYSTEM
# ----------------------------------------------------------------
# Configuration for the Master Hindsight ML System integration
# Trained on 675K trades with 99%+ accuracy models

# Enable/disable Hindsight ML (DISABLED - faulty 99.54% WR fantasy data)
USE_HINDSIGHT_ML = env("USE_HINDSIGHT_ML", "0") in ("1", "true", "TRUE")

# Hindsight ML model directory (default: latest in master_hindsight_models/)
HINDSIGHT_ML_MODEL_DIR = env("HINDSIGHT_ML_MODEL_DIR", None)

# Hindsight ML score thresholds (0-100 scale)
# These are LOWER than standard thresholds because Hindsight ML is more accurate
HINDSIGHT_MIN_SCORE = int(env("HINDSIGHT_MIN_SCORE", "40"))  # Minimum score to consider (was 35)
HINDSIGHT_HARD_MIN_SCORE = int(env("HINDSIGHT_HARD_MIN_SCORE", "30"))  # Absolute minimum (was 30)

# Hindsight ML confidence thresholds (0-1 scale, from entry_probability)
HINDSIGHT_MIN_CONFIDENCE = float(env("HINDSIGHT_MIN_CONFIDENCE", "0.60"))  # 60% minimum confidence
HINDSIGHT_HIGH_CONFIDENCE = float(env("HINDSIGHT_HIGH_CONFIDENCE", "0.75"))  # 75% = high confidence

# Regime-based filtering
HINDSIGHT_SKIP_DANGEROUS_REGIMES = env("HINDSIGHT_SKIP_DANGEROUS_REGIMES", "1") in ("1", "true", "TRUE")  # Skip regimes 2,4
HINDSIGHT_PREFER_REGIME_0 = env("HINDSIGHT_PREFER_REGIME_0", "1") in ("1", "true", "TRUE")  # Prioritize high-performance market regimes

# Use ML-predicted optimal SL instead of ATR-based SL
HINDSIGHT_USE_OPTIMAL_SL = env("HINDSIGHT_USE_OPTIMAL_SL", "1") in ("1", "true", "TRUE")

# ----------------------------------------------------------------
# LEGACY / BACKWARD COMPATIBILITY
# ----------------------------------------------------------------
# These variables were removed but are still referenced by some scripts/tests.
# We map them to their modern equivalents or provide defaults.
MARKSMAN_MIN_SCORE = MIN_SIGNAL_SCORE
MARKSMAN_MIN_MOMENTUM_PCT = env("MARKSMAN_MIN_MOMENTUM_PCT", "1.0", float)
ENABLE_MARKSMAN = env("ENABLE_MARKSMAN", "0") in ("1", "true", "TRUE")
MARKSMAN_SYMBOLS = [
    "ADA/USDT", "APT/USDT", "ARB/USDT", "ATOM/USDT", "AVAX/USDT", 
    "BCH/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "DOT/USDT", 
    "ETC/USDT", "ETH/USDT", "FIL/USDT", "FTM/USDT", "INJ/USDT", 
    "LINK/USDT", "LTC/USDT", "NEAR/USDT", "OP/USDT", "RUNE/USDT", 
    "SEI/USDT", "SOL/USDT", "SUI/USDT", "TIA/USDT", "UNI/USDT", 
    "WLD/USDT", "XRP/USDT"
]


def safe_read_json(path, default=None):
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_write_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        # Use logger if available, otherwise silent fail
        try:
            from .logger import get_logger
            get_logger().warning(f"JSON write failed {path}: {e}")
        except (ImportError, Exception):
            # Silently fail if logger not available (shouldn't happen)
            pass
