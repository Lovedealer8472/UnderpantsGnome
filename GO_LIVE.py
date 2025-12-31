"""
LIVE TRADING MODE - Adaptive Threshold System
==============================================
Production configuration with adaptive threshold system.
Quality floor: 45 (60% ML probability = profitable)
"""
import asyncio
import sys
import os
import time
import subprocess
import requests
from datetime import datetime, timezone

# ============================================================================
# SAFETY GUARD: PREVENT ACCIDENTAL BACKGROUND LIVE TRADING
# ============================================================================
def _is_interactive() -> bool:
    """Check if running in an interactive terminal."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False

if not _is_interactive():
    print("=" * 60)
    print("SAFETY BLOCK: GO_LIVE.py CANNOT RUN IN BACKGROUND")
    print("=" * 60)
    print("This script trades with REAL MONEY and requires")
    print("interactive supervision. Run DIRECTLY in a terminal:")
    print("  python GO_LIVE.py")
    print("=" * 60)
    sys.exit(1)
# ============================================================================

# Ensure we are in the right directory
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'TradingBot'))

# =============================================================================
# CRITICAL: Set ALL environment variables BEFORE loading .env
# =============================================================================
os.environ["DRY_RUN"] = "0"  # LIVE TRADING ENABLED
os.environ["UI_MODE"] = "console"  # Headless mode - pure logging, no built-in UI
# NOTE: HARD_MIN_SCORE not set - adaptive system handles all threshold logic

# =============================================================================
# LIVE TRADING CONFIGURATION
# =============================================================================
print("=" * 80)
print("LIVE TRADING MODE - Adaptive Threshold System")
print("=" * 80)

# Thresholds (35 = trust ML predictions - 35%+ win probability)
os.environ["MIN_SIGNAL_SCORE"] = "59"  # ML-OPTIMIZED: 675K trades
os.environ["HARD_MIN_SCORE"] = "55"   # ML-OPTIMIZED floor

# Position management
os.environ["MAX_CONCURRENT_POS"] = "3"
os.environ["MAX_OPEN_POSITIONS"] = "3"
os.environ["RISK_PER_TRADE_PCT"] = "2.0"

# Disable adaptive systems (for predictable behavior)
os.environ["USE_ADAPTIVE_REGIME"] = "0"
os.environ["SCORE_AWARE_REPLACEMENT_ENABLED"] = "0"

# Re-enable entry filters for quality (conservative approach)
os.environ["ENTRY_MIN_MOMENTUM_PCT"] = "1.0"  # Require 1% momentum
os.environ["USE_ADAPTIVE_FILTERS"] = "1"     # Enable adaptive filters
os.environ["DISABLE_ALL_FILTERS"] = "0"      # Enable filters

# Enable adaptive threshold system
os.environ["USE_ADAPTIVE_THRESHOLDS"] = "1"
os.environ["ADAPTIVE_THRESHOLD_MIN"] = "45"
os.environ["ADAPTIVE_THRESHOLD_MAX"] = "70"

# Disable percentile filter (replaced by adaptive threshold)
os.environ["USE_SIGNAL_PERCENTILE_FILTER"] = "0"

# Other settings
os.environ["DRY_SIMPLE_EXITS"] = "0"  # Use full exit system for live trading
os.environ["USE_BINANCE_TRAILING_STOP"] = "0"
os.environ["MAX_ENTRIES_PER_MIN"] = "5"  # Conservative for live trading

# NOW load .env (won't override the values we just set)
from dotenv import load_dotenv
load_dotenv(override=False)  # override=False means our env vars above take precedence

# Import config (will read our env vars)
import app.config as config

# Force override config
config.DRY_RUN = False  # LIVE TRADING

# =============================================================================
# Apply configuration overrides
# =============================================================================
# Thresholds (Trust ML predictions)
config.MIN_SIGNAL_SCORE = 59  # ML-OPTIMIZED
config.HARD_MIN_SCORE = 55    # ML-OPTIMIZED floor

# Position management
config.MAX_CONCURRENT_POS = 3
config.MAX_OPEN_POSITIONS = 3
config.MAX_CONCURRENT_POS_HARD = 3  # Hard limit
config.RISK_PER_TRADE_PCT = 2.0

# Disable adaptive systems
if hasattr(config, 'USE_ADAPTIVE_REGIME'):
    config.USE_ADAPTIVE_REGIME = False
config.SCORE_AWARE_REPLACEMENT_ENABLED = False

# Disable entry filters
config.ENTRY_MIN_MOMENTUM_PCT = 0.0
config.ENTRY_FILTER_ENABLED = False
config.ENTRY_MIN_ATR_PCT = 0.0
if hasattr(config, 'USE_DATA_DRIVEN_ENTRY_FILTER'):
    config.USE_DATA_DRIVEN_ENTRY_FILTER = False
if hasattr(config, 'USE_ADAPTIVE_MOMENTUM_FILTER'):
    config.USE_ADAPTIVE_MOMENTUM_FILTER = False
if hasattr(config, 'MIN_INTRABAR_MOMENTUM_PCT'):
    config.MIN_INTRABAR_MOMENTUM_PCT = 0.0

# Disable percentile filter (replaced by adaptive threshold)
config.USE_SIGNAL_PERCENTILE_FILTER = False

# Other settings
config.DRY_SIMPLE_EXITS = False  # Use full exit system
config.USE_BINANCE_TRAILING_STOP = False
config.MAX_ENTRIES_PER_MIN = 5  # Conservative for live trading

# Enable adaptive threshold system
config.USE_ADAPTIVE_THRESHOLDS = True
config.ADAPTIVE_THRESHOLD_MIN = 45
config.ADAPTIVE_THRESHOLD_MAX = 70

# R-based exit profiles (ML-optimized)
config.R_SCALP_TP_R = 0.8
config.R_STANDARD_PARTIAL_TP_R = 2.0
config.R_STANDARD_TRAIL_START_R = 2.0
config.R_STANDARD_MAX_R = 2.5
config.R_RUNNER_MAX_R_NORMAL = 8.0

# Time-based exits (in 5min bars)
config.R_SCALP_TIME_STOP_BARS = 36      # 3 hours
config.R_STANDARD_TIME_STOP_BARS = 72   # 6 hours
config.R_RUNNER_TIME_STOP_BARS = 576    # 48 hours

# Exit score thresholds
config.R_EXIT_SCALP_SCORE_MIN = 35
config.R_EXIT_STANDARD_SCORE_MIN = 40
config.R_EXIT_RUNNER_SCORE_MIN = 45

# Stop-loss configuration
config.SL_ATR_MULTIPLIER = 1.0

# =============================================================================
# Pre-flight checks
# =============================================================================
def get_btc_funding_rate():
    """Fetch current BTC funding rate from Binance."""
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
            timeout=3
        )
        data = resp.json()
        return float(data.get('lastFundingRate', 0)) * 100
    except:
        return None

def get_btc_volatility():
    """Fetch BTC 24h price change as volatility proxy."""
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT",
            timeout=3
        )
        data = resp.json()
        return abs(float(data.get('priceChangePercent', 0)))
    except:
        return None

def pre_flight_checks():
    print("\n" + "="*60)
    print("PRE-FLIGHT CHECKS")
    print("="*60)
    
    # 1. Internet Connectivity
    try:
        requests.get("https://www.google.com", timeout=3)
        print("[OK] Internet: Connected")
    except Exception:
        print("[X] Internet: FAILED")
        sys.exit(1)
        
    # 2. Time Sync (Crucial for Binance)
    try:
        server_time = requests.get("https://api.binance.com/api/v3/time", timeout=3).json()['serverTime']
        local_time = int(time.time() * 1000)
        diff = abs(server_time - local_time)
        if diff > 1000:
            print(f"[WARNING] Time Sync: WARNING (Diff {diff}ms)")
        else:
            print(f"[OK] Time Sync: OK ({diff}ms)")
    except Exception as e:
        print(f"[WARNING] Time Sync: Check failed ({e})")

    # 3. API Keys
    key = os.getenv("BINANCE_API_KEY") or config.BINANCE_API_KEY
    if not key or len(key) < 60:
        print(f"[X] API Key: INVALID (Length: {len(key) if key else 0})")
        sys.exit(1)
    else:
        print(f"[OK] API Key: {key[:4]}...{key[-4:]}")

    # 4. Market Regime Check
    btc_vol = get_btc_volatility()
    funding = get_btc_funding_rate()
    print(f"[OK] BTC Volatility: {btc_vol:.1f}%" if btc_vol else "[WARNING] BTC Vol: Unknown")
    print(f"[OK] Funding Rate: {funding:.4f}%" if funding else "[WARNING] Funding: Unknown")
    
    # 5. Current UTC Time
    now_utc = datetime.now(timezone.utc)
    print(f"[OK] UTC Time: {now_utc.strftime('%H:%M')}")
    
    print("="*60 + "\n")

# =============================================================================
# Configuration summary
# =============================================================================
print("=" * 80)
print("LIVE TRADING MODE - REAL MONEY")
print("=" * 80)
print(f"MIN_SIGNAL_SCORE: {config.MIN_SIGNAL_SCORE} (ML-OPTIMIZED: 675K trades - select top 50% quality)")
print(f"HARD_MIN_SCORE: {config.HARD_MIN_SCORE}")
print(f"MAX_POSITIONS: {config.MAX_CONCURRENT_POS} / {config.MAX_CONCURRENT_POS_HARD} (hard cap)")
print(f"RISK_PER_TRADE: {config.RISK_PER_TRADE_PCT}%")
print(f"MAX_ENTRIES_PER_MIN: {config.MAX_ENTRIES_PER_MIN}")
print(f"ADAPTIVE_THRESHOLDS: ENABLED (floor={config.ADAPTIVE_THRESHOLD_MIN}, cap={config.ADAPTIVE_THRESHOLD_MAX})")
print(f"PERCENTILE_FILTER: DISABLED (using adaptive threshold instead)")
print(f"ENTRY_FILTERS: DISABLED (allow quality signals)")
print("=" * 80)
print(f"R_SCALP_TP: {config.R_SCALP_TP_R}R")
print(f"R_STANDARD_MAX: {config.R_STANDARD_MAX_R}R")
print(f"R_RUNNER_MAX: {config.R_RUNNER_MAX_R_NORMAL}R")
print("=" * 80)
print()

# Run Pre-flight
pre_flight_checks()

# Import and run
from run import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] Bot stopped by user")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        with open("panic.log", "w") as f:
            f.write(f"FATAL CRASH:\n{e}\n\nTraceback:\n{traceback.format_exc()}")

