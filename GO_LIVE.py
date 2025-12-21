import asyncio
import sys
import os
import time
import requests
from datetime import datetime, timezone

# ============================================================================
# SAFETY GUARD: PREVENT BACKGROUND/SUBPROCESS LIVE TRADING
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
# Add TradingBot to path so we can import 'app' as a package
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'TradingBot'))

# CRITICAL: Manually load .env with override=True to ensure keys are loaded
from dotenv import load_dotenv
load_dotenv(override=True)

# =============================================================================
# CRITICAL: Set environment variables BEFORE importing config
# This ensures config.py reads the correct values via env()
# =============================================================================
os.environ.setdefault("USE_SIGNAL_PERCENTILE_FILTER", "1")  # Enable percentile-based filtering
os.environ.setdefault("SIGNAL_PERCENTILE_THRESHOLD", "0.90")  # Top 10% of signals (90th percentile) - Selective cherry picking
os.environ.setdefault("SIGNAL_HISTORY_SIZE", "100")  # Track last 100 signals

# Import config (this will load .env and defaults, but our env vars take precedence)
import app.config as config

# =============================================================================
# RESEARCH-BASED CONFIGURATION (Dec 2025)
# =============================================================================
# Based on deep research into:
# - Order flow analysis & tape reading
# - Liquidation heatmaps & stop-hunt avoidance
# - Funding rate strategies
# - Volume profile & VWAP concepts
# - Market microstructure realities
# - 100K+ trade introspection analysis
# =============================================================================

def get_btc_funding_rate():
    """Fetch current BTC funding rate from Binance."""
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
            timeout=3
        )
        data = resp.json()
        return float(data.get('lastFundingRate', 0)) * 100  # Convert to percentage
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

def check_market_regime():
    """Check if market conditions are favorable for scalping."""
    btc_vol = get_btc_volatility()
    funding = get_btc_funding_rate()
    
    regime_ok = True
    warnings = []
    
    if btc_vol is not None:
        if btc_vol < 1.0:
            warnings.append(f"⚠️ Low volatility ({btc_vol:.1f}%) - ranging market")
        elif btc_vol > 12.0:
            warnings.append(f"⚠️ High volatility ({btc_vol:.1f}%) - chaotic conditions")
    
    if funding is not None:
        if abs(funding) > 0.05:  # >0.05% funding = crowded trade
            direction = "LONGS" if funding > 0 else "SHORTS"
            warnings.append(f"⚠️ Extreme funding ({funding:.3f}%) - {direction} crowded")
    
    return regime_ok, warnings, btc_vol, funding

def pre_flight_checks():
    print("\n" + "="*60)
    print("PRE-FLIGHT CHECKS")
    print("="*60)
    
    # 1. Internet Connectivity
    try:
        requests.get("https://www.google.com", timeout=3)
        print("✅ Internet: Connected")
    except Exception:
        print("❌ Internet: FAILED")
        sys.exit(1)
        
    # 2. Time Sync (Crucial for Binance)
    try:
        server_time = requests.get("https://api.binance.com/api/v3/time", timeout=3).json()['serverTime']
        local_time = int(time.time() * 1000)
        diff = abs(server_time - local_time)
        if diff > 1000:
            print(f"⚠️ Time Sync: WARNING (Diff {diff}ms)")
        else:
            print(f"✅ Time Sync: OK ({diff}ms)")
    except Exception as e:
        print(f"⚠️ Time Sync: Check failed ({e})")

    # 3. API Keys
    key = os.getenv("BINANCE_API_KEY") or config.BINANCE_API_KEY
    if not key or len(key) < 60:
        print(f"❌ API Key: INVALID (Length: {len(key) if key else 0})")
        sys.exit(1)
    else:
        print(f"✅ API Key: {key[:4]}...{key[-4:]}")

    # 4. Market Regime Check
    regime_ok, warnings, btc_vol, funding = check_market_regime()
    print(f"✅ BTC Volatility: {btc_vol:.1f}%" if btc_vol else "⚠️ BTC Vol: Unknown")
    print(f"✅ Funding Rate: {funding:.4f}%" if funding else "⚠️ Funding: Unknown")
    
    for w in warnings:
        print(f"   {w}")
    
    # 5. Current UTC Time (for funding awareness)
    now_utc = datetime.now(timezone.utc)
    next_funding_hour = ((now_utc.hour // 8) + 1) * 8 % 24
    mins_to_funding = ((next_funding_hour - now_utc.hour) % 24) * 60 - now_utc.minute
    if mins_to_funding < 0:
        mins_to_funding += 24 * 60
    print(f"✅ UTC Time: {now_utc.strftime('%H:%M')} (Funding in {mins_to_funding}min)")
    
    print("="*60 + "\n")

# =============================================================================
# BANNER
# =============================================================================
print("\n" + "="*60)
print("🐰 UNIRABBIT SCALPER - LIVE MODE")
print("   Research-Enhanced Configuration (Dec 2025)")
print("="*60)

# =============================================================================
# SECTION 1: MODE SELECTION
# =============================================================================
# Choose your trading style

QUICK_SCALP_ENABLED = False  # True = aggressive momentum lock, False = standard R-based

if QUICK_SCALP_ENABLED:
    config.QUICK_SCALP_MODE = True
    config.QS_MIN_PROFIT_TO_TRAIL = 0.15
    config.QS_REVERSAL_THRESHOLD_PCT = 0.08
    config.QS_FLOOR_ABOVE_BE_PCT = 0.12
    config.QS_MAX_HOLD_BARS = 4
    config.QS_STOP_LOSS_R = -0.5
    config.TIME_EXIT_BARS = 4
else:
    config.QUICK_SCALP_MODE = False

# =============================================================================
# SECTION 2: RESEARCH-BASED SAFETY FEATURES
# =============================================================================

# COST GATE: "If edge < cost, you're dead before you start"
# DISABLED: Formula designed for old scores (60-80), not ML scores (25-50)
config.COST_GATE_ENABLED = False
config.COST_GATE_MULTIPLIER = 0.8      # Edge needs to cover 80% of cost (was 1.0 = too strict)
config.COST_GATE_SCORE_TO_EDGE_MULT = 3.0  # Score 60 = 30 bps edge (was 2.5)
config.COST_GATE_MIN_EDGE_BPS = 3      # Minimum 3 bps edge (was 5)

# SPREAD EXPLOSION: "If liquidity disappears, GET OUT"
config.SPREAD_EXPLOSION_EXIT_ENABLED = True
config.SPREAD_EXPLOSION_MULTIPLIER = 3.0   # Exit if spread 3x entry
config.SPREAD_EXPLOSION_MAX_BPS = 100      # Hard cap

# FUNDING AVOIDANCE: Don't hold small profits into funding
config.FUNDING_AVOIDANCE_ENABLED = True
config.FUNDING_AVOIDANCE_MINUTES = 10      # Exit 10min before funding
config.FUNDING_AVOIDANCE_MIN_PROFIT_PCT = 0.05

# REGIME FILTER: Only trade favorable conditions
config.REGIME_FILTER_ENABLED = True
config.REGIME_MIN_BTC_VOLATILITY_PCT = 1.0   # Need momentum
config.REGIME_MAX_BTC_VOLATILITY_PCT = 15.0  # Avoid chaos

# TIME FILTER: Avoid bad hours (disabled by default)
config.TIME_FILTER_ENABLED = False
config.TIME_FILTER_AVOID_HOURS = "7,15,23"   # Pre-funding manipulation

# =============================================================================
# SECTION 3: REALISTIC EXECUTION MODEL
# =============================================================================
# Research shows retail bots face 10-50bps slippage, not 2bps

config.SLIPPAGE_BPS = 10  # Realistic baseline (was 2 = fantasy)

# =============================================================================
# SECTION 4: EXIT STRATEGIES & STOP-LOSS SETTINGS
# =============================================================================
# From 100K+ trade analysis - optimal TP/SL found via machine learning

# Simple Exit Targets (R-based)
config.DRY_SIMPLE_SL_R = -1.5  # Wider stop (survive noise, avoid stop-hunts)
config.DRY_SIMPLE_TP_R = 2.6   # Balanced TP (actually reachable)

# Exit System Configuration
config.USE_R_BASED_EXITS = True  # R-based exit profiles - ON
config.USE_TRAILING_ENGINE = True  # Enable trailing stop engine
config.USE_NEW_TRAILING_ENGINE = True  # Use new trailing engine
config.USE_ATR_TRAILING_STOP = False  # Legacy ATR trailing - OFF

# Binance Server-Side Stop-Loss (Circuit Breaker)
# This places a stop-loss order on Binance for disaster protection
# Should be WIDER than bot's internal trailing (which handles normal exits)
config.USE_BINANCE_TRAILING_STOP = True  # Enable Binance stop-loss orders
config.BINANCE_TRAILING_CALLBACK_RATE = 3.0  # 3.0% stop distance (circuit breaker - wide enough to not interfere)

# Trailing Stop Settings (R-based engine)
config.TRAIL_ENGINE_START_BUFFER_R = 0.3  # Start trailing at 0.3R (activate early)
config.TRAIL_ENGINE_PARTIAL_1_R = 1.0  # First partial at +1R
config.TRAIL_ENGINE_PARTIAL_1_SIZE = 0.25  # 25% clip at 1R
config.TRAIL_ENGINE_BREAK_EVEN_R = 1.5  # Move to BE at 1.5R
config.TRAIL_ENGINE_BE_BUFFER_R = 0.25  # Buffer above BE

# Stale Position Management (Prevent dead weight)
config.MAX_POSITION_AGE_SEC = 3600  # 1 hour max hold (60 minutes)
config.STALE_POSITION_PNL_THRESHOLD = 0.3  # Auto-close if |PnL| < 0.3% after max age

# Note: weights.json now has skip_probability = 0.35 globally
# This means ~35% of marginal signals will be skipped (introspection-optimal)

# =============================================================================
# SECTION 5: LIVE TRADING CORE
# =============================================================================

config.DRY_RUN = False       # 💰 REAL MONEY MODE
config.REPLAY_MODE = False
config.MARGIN_MODE = "CROSS"
config.EXCHANGE = "binance_futures"

# Startup warmup: Full market scan before allowing trades
# This builds signal history so percentile filter (top 1%) has good data to work with
config.STARTUP_DELAY_SEC = 60  # 60 seconds for full market scan (was 30s)

# =============================================================================
# SECTION 6: RISK MANAGEMENT (SWARM STRATEGY)
# =============================================================================

config.RISK_PER_TRADE_PCT = 5.0       # 5% per trade
config.MAX_OPEN_POSITIONS = 10        # Reduced to prevent dust (was 20)
config.MAX_CONCURRENT_POS = 10
config.MAX_CONCURRENT_POS_HARD = 10
# Position sizing limits - AGGRESSIVE ANTI-DUST SETTINGS
# Increased significantly to prevent 0.0x positions that clog the machine
config.MIN_POSITION_SIZE = 20.0       # Increased to $20 to prevent dust (0.0x positions)
config.MAX_POSITION_SIZE = 50.0
config.TOTAL_RISK_BUDGET = 1.0        # 100% deployment capacity
config.MAX_ACCOUNT_RISK_PCT = 100.0

# Unicorn bypass for exceptional signals
config.UNICORN_BYPASS_MAX_POSITIONS = True

# Dynamic sizing based on signal quality
config.USE_RANK_BASED_ALLOCATION = True
# CRITICAL: No size reduction - prevent dust from forming
config.RPA_MIN_SIZE_MULT = 1.0        # 100% min size for ALL signals (no reduction)
config.RPA_MAX_SIZE_MULT = 1.5        # 150% for strong signals
config.RPA_MIN_SIZE_USD = 20.0        # Must match MIN_POSITION_SIZE to prevent dust

# =============================================================================
# SECTION 7: SIGNAL QUALITY FILTERS
# =============================================================================

# SIGNAL FILTERS - ADAPTIVE PERCENTILE-BASED SYSTEM
# ML outputs 25-50 range (win probability * 100)
# Use percentile filtering to automatically adapt to market conditions
# If many signals score 41+, only top 20% will pass
# If few signals score 41+, more will pass (dynamic threshold)
config.MIN_SIGNAL_SCORE = 40.0        # Base floor - must be 40+ to even be considered
config.MIN_SIGNAL_STRENGTH = 0.40     # Match score/100
config.HARD_MIN_SCORE = 40            # Absolute minimum - FORCE percentile filter to work (must be 40+ to even be considered)
config.MIN_SCORE_RANGE = (25, 50)     # ML Scorer output range - MUST override config.py
config.MIN_STRENGTH_RANGE = (0.25, 0.50)  # Match score range

# ADAPTIVE PERCENTILE FILTERING (Environment variables set above before config import)
# This filters to top X% of recent signals, automatically adjusting threshold
# If 50 signals score 41+, only top 20% (10 signals) will pass
# If only 5 signals score 41+, most/all will pass (adaptive!)
# Also set on config object for any code that reads directly
config.USE_SIGNAL_PERCENTILE_FILTER = True
config.SIGNAL_PERCENTILE_THRESHOLD = 0.90  # Top 10% - Selective cherry picking (top quality signals)
config.SIGNAL_HISTORY_SIZE = 100

# SELECTIVE CHERRY PICKING: Top 10% (0.90) - Only the best signals pass
# - Strict filtering - rejects 90% of signals above the base threshold
# - With 100 signals scoring 40+, only the top 10 will pass
# - Good balance between selectivity and staying active
# Note: SIGNAL_PERCENTILE_THRESHOLD is a float 0.0-1.0 where 0.90 = 90th percentile = top 10%

# CRITICAL: Disable IDLE_RELAX - it was allowing Score 59 garbage through!
config.IDLE_RELAX_ENABLED = False     # NO RELAXING - percentile handles adaptation

# UNLOCK entry filters - allow adaptive percentile system to work
# But don't let regime adapter override - we use percentile instead
os.environ["LOCK_ENTRY_FILTERS"] = "0"  # Allow percentile filtering to work
config.LOCK_ENTRY_FILTERS = False

# Correlation: Disabled for swarm strategy (ride the wave)
config.CORRELATION_BLOCK_THRESHOLD = 0

# =============================================================================
# SECTION 8: MICROSTRUCTURE LIMITS
# =============================================================================

config.MAX_SPREAD_BPS = 300           # Max 3% spread (allow alts)
config.MAX_VOLATILITY_PCT = 15.0      # Max 15% volatility

# =============================================================================
# SECTION 9: EXECUTION THROTTLING
# =============================================================================

config.SYMBOLS_TO_SCAN = 60           # Elite 60 symbols
config.DISCOVERY_SCAN_INTERVAL_SEC = 5.0
config.LOG_LEVEL = "INFO"
config.COOLDOWN_AFTER_EXIT = 60       # 1min cooldown after exit
config.MAX_ENTRIES_PER_MIN = 2        # Prevent runaway

# =============================================================================
# STARTUP OUTPUT
# =============================================================================

print(f"\n📊 CONFIGURATION SUMMARY")
print("-"*40)
print(f"MODE:     {'🚀 Quick Scalp' if QUICK_SCALP_ENABLED else '📈 Standard R-Based'}")
print(f"LIVE:     {'💰 REAL MONEY' if not config.DRY_RUN else '📝 Paper Trading'}")
print(f"MARGIN:   {config.MARGIN_MODE}")
print(f"EXCHANGE: {config.EXCHANGE}")
print("-"*40)
print(f"RISK:     {config.RISK_PER_TRADE_PCT}% per trade")
print(f"MAX POS:  {config.MAX_OPEN_POSITIONS}")
print(f"SIZE:     ${config.MIN_POSITION_SIZE}-${config.MAX_POSITION_SIZE}")
print("-"*40)
print(f"MIN SCORE:    {config.MIN_SIGNAL_SCORE}")
print(f"SKIP PROB:    35% (introspection-optimal)")
print(f"SLIPPAGE:     {config.SLIPPAGE_BPS} bps (realistic)")
print("-"*40)
print("SAFETY FEATURES:")
print(f"  🛡️ Cost Gate:        {config.COST_GATE_ENABLED}")
print(f"  💥 Spread Explosion:  {config.SPREAD_EXPLOSION_EXIT_ENABLED}")
print(f"  ⏰ Funding Avoidance: {config.FUNDING_AVOIDANCE_ENABLED}")
print(f"  📊 Regime Filter:     {config.REGIME_FILTER_ENABLED}")
print(f"  🕐 Time Filter:       {config.TIME_FILTER_ENABLED}")
print("-"*40)
print(f"EXITS: SL={config.DRY_SIMPLE_SL_R}R | TP={config.DRY_SIMPLE_TP_R}R")
print("="*60)
print("⚠️  WARNING: REAL MONEY WILL BE USED")
print("    Monitor Binance Dashboard closely!")
print("="*60 + "\n")

# Run Pre-flight
pre_flight_checks()

# Import and run the bot
import run

if __name__ == "__main__":
    try:
        asyncio.run(run.main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n💀 FATAL: {e}")
        import traceback
        traceback.print_exc()
        with open("panic.log", "w") as f:
            f.write(f"FATAL CRASH:\n{e}\n\nTraceback:\n{traceback.format_exc()}")
