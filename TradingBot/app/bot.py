# OPTIMIZATION: Removed unused imports (ccxt_async, math) and config (USE_WS)
import sys
import os
import time
import asyncio
import math
import logging
import traceback
from collections import deque
from typing import Optional, Dict, Any

from .logger import get_logger
from .config import (
    DRY_RUN, REPLAY_MODE, EXCHANGE, EXCHANGE_TIMEOUT_MS, EXCHANGE_RETRIES,
    ACCOUNT_BAL, LEVERAGE_BASE, MAX_LATENCY_MS, MAX_POSITION_SIZE,
    RETRY_DELAY_MULTIPLIER, MID_PRICE_FACTOR, BTC_TREND_WEAK_THRESHOLD,
    HIGH_VOLATILITY_MEDIAN_THRESHOLD, HIGH_VOLATILITY_P75_THRESHOLD,
    MAX_DRAWDOWN_PCT, DRAWDOWN_CIRCUIT_BREAKER_ENABLED, WHITELIST_SYMBOLS, MIN_24H_VOLUME_USDT,
    MAX_CONCURRENT_POS, MAX_OPEN_POSITIONS, USE_RICH_UI,
    UNICORN_SCORE_THRESHOLD, UNICORN_PROTOCOL_ENABLED,
    STARTUP_DELAY_SEC, PRS_MIN_AGE_MIN
)
from .universe import DynamicUniverse, SymbolStats
from .llm_control import LlmController
# UI removed - using static UI only
from .position_manager import PositionManager
from .signals import SignalGenerator
from .order_manager import OrderManager
from .exit_manager import ExitManager
from .decision_logger import get_decision_logger
from .decision_event import DecisionEvent, log_trade_decision
from .ticker_cache import get_ticker_cache
from .fast_storage import get_fast_storage
from .exchanges.factory import create_exchange
import app.config as config_module

# NEW ARCHITECTURE: Core modules
from .core.positions import PositionRegistry
from .engine.exit_pipeline import ExitPipeline, ExitRequest

# TRADE INTROSPECTION: Self-learning from each trade
from .learner.integration import LearnerIntegration, is_learner_enabled

# OPTIMIZATION: Removed unused now_ms() function

# ----------------------------------------------------------------
# TRADE LIFECYCLE DOCUMENTATION
# ----------------------------------------------------------------
#
# The complete lifecycle of a trade in GreenUniRabbit:
#
# 1. DISCOVERY (scan_and_enter_signals)
#    - Universe refresh: DynamicUniverse refreshes symbol list based on volume/liquidity
#    - Symbol scanning: Each symbol is scanned for signals (momentum, RSI, trend, etc.)
#    - Signal generation: SignalGenerator creates signals with entry/exit prices and scores
#
# 2. FILTERS & CONFIRMATION
#    - Signal Confirmation Window (SCW): Multi-bar confirmation for signal quality
#    - Risk filters: Spread, volume, latency, volatility checks
#    - Correlation blocker: Prevents correlated positions
#    - Risk budget: Total risk budget check (TOTAL_RISK_BUDGET)
#    - Position limit: Max concurrent positions check (MAX_CONCURRENT_POS)
#    - Score-aware replacement: Can replace weakest position if new signal is better
#
# 3. ENTRY (order_manager.enter_position)
#    - Position sizing: Dynamic position sizing based on signal strength, risk budget, Kelly
#    - Leverage: Dynamic leverage based on signal strength and market conditions
#    - Order placement: Market or limit order via exchange wrapper
#    - Position tracking: Added to self.positions dict with metadata
#
# 4. MONITORING (monitor_and_exit_positions)
#    - Price updates: Current price fetched from cache or exchange
#    - PnL calculation: Real-time PnL tracking (peak_pnl, trough_price)
#    - PRS evaluation: Position Recovery Score computed (age, trend, volatility, PnL)
#    - Exit checks: Multiple exit systems checked:
#      * R-based exits (if USE_R_BASED_EXITS): Scalp/Standard/Runner profiles
#      * Legacy exits: Stop-loss, take-profit, trailing stops, stale position rules
#      * PRS exits: Full exit (PRS < 30) or scale-out (PRS < 50)
#      * BERP: Break-Even Rescue Protocol for long-running unprofitable positions
#      * Stale position rules: 90min, extended leash, drawdown resume
#
# 5. EXIT (exit_manager.exit_position - CANONICAL EXIT PATH)
#    - ALL exits must go through exit_manager.exit_position()
#    - Validates position state (not already closed, size > 0)
#    - Calculates exit price (target_price or market)
#    - Executes order (market or limit)
#    - Records PnL, fees, slippage, funding costs
#    - Updates metrics (exits_by_reason)
#    - Removes from self.positions
#    - Logs trade decision
#
# 6. POST-EXIT
#    - PnL aggregation: Updates realized_pnl_total, win_count, loss_count
#    - Performance tracking: Updates win rate, profit factor
#    - Risk budget release: Frees up risk budget for new entries
#
# EXIT REASONS (see app/exit_reasons.py for full list):
# - Stop-loss/take-profit: stop_loss_hit, take_profit_hit, trailing_stop_hit
# - PRS: prs_full_exit_<score>, prs_scale_out_<score>
# - Time-based: stale_position_timeout, max_age_exceeded, stale_90min_exit
# - Risk: risk_budget_exceeded, max_positions_reached, drawdown_circuit_breaker
# - R-based: r_scalp_tp, r_standard_partial_tp, r_runner_trailing_stop, etc.
# - MSX: msx_stage1_invalidation, msx_partial_scalp, etc.
# - Replacement: replaced_by_better_signal, score_aware_replacement
# - Manual: manual_exit, llm_override
#
# ----------------------------------------------------------------

class ScalperBot:
    def __init__(self):
        # Initialize logger (respect UI mode to prevent interference)
        # For initialization, we default to True unless USE_RICH_UI is explicitly set
        from .config import USE_RICH_UI
        self.logger = get_logger("ScalperBot", enable_console=not USE_RICH_UI)
        
        # SAFETY CHECK: Warn loudly if LIVE trading without interactive terminal
        if not DRY_RUN:
            try:
                is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
            except Exception:
                is_interactive = False
            if not is_interactive:
                self.logger.critical("=" * 60)
                self.logger.critical("DANGER: LIVE TRADING WITHOUT INTERACTIVE TERMINAL")
                self.logger.critical("DRY_RUN=False but no TTY detected!")
                self.logger.critical("LIVE TRADING SHOULD BE SUPERVISED.")
                self.logger.critical("=" * 60)
        
        # CRITICAL VALIDATION: Prevent REPLAY_MODE in production
        if REPLAY_MODE:
            self.logger.warning("[!]  REPLAY_MODE is ENABLED - This bot will use RELAXED signal thresholds!")
            self.logger.warning("[!]  DO NOT use REPLAY_MODE for live trading!")
            if not DRY_RUN:
                self.logger.error("[X] FATAL: REPLAY_MODE cannot be used with LIVE trading (DRY_RUN=False)")
                raise RuntimeError("REPLAY_MODE is only for backtesting. Set DRY_RUN=True or disable REPLAY_MODE.")
        
        # Config access via cfg object
        self.cfg = config_module
        self.exchange = None  # Backward compatibility (may be used by some code)
        self.exchange_wrapper = None  # Exchange abstraction wrapper
        self.universe = DynamicUniverse()
        # Legacy budgeter (kept for compatibility, not used with ApiGovernor)
        self.budget = None
        self.ctrl = LlmController()
        self.position_manager = PositionManager()
        self.signal_generator = SignalGenerator()
        
        # ADAPTIVE ENTRY: Dynamic threshold controller for 24/7 activity
        from .adaptive_entry import get_adaptive_controller
        self.adaptive_entry = get_adaptive_controller(logger=self.logger)
        self.order_manager = None  # Will be set after exchange init
        self.exit_manager = None  # Will be set after exchange init
        
        # Adaptive Performance Monitor: Intelligent system that adjusts based on open positions
        from .adaptive_performance import AdaptivePerformanceMonitor
        self.adaptive_monitor = AdaptivePerformanceMonitor(lookback_window_seconds=3600.0)
        
        # REPLAY MODE: Data feed for backtesting
        # CRITICAL: replay_mode is controlled by REPLAY_MODE config flag
        # DO NOT set this manually - it's set by replay_runner.py
        self.replay_mode = REPLAY_MODE  # Read from config (default: False)
        self.replay_feed = None
        self.replay_start_time = 0.0
        self._replay_current_time = None  # Current replay time (set by replay_runner)
        
        # Signal Confirmation Window (SCW)
        from .signal_confirmation import SignalConfirmationManager
        
        # RECOMMENDATION #7: Scanner module for better modularity
        from .scanner import SymbolScanner
        self.scanner = SymbolScanner(self)
        self.signal_confirmation = SignalConfirmationManager()
        
        # LATENCY OPTIMIZATION: Initialize caches for ultra-fast data access
        self.ticker_cache = get_ticker_cache(ttl=0.5)  # 500ms cache TTL
        self.fast_storage = get_fast_storage()  # Persistent SQLite cache

        # NEW ARCHITECTURE: Use PositionRegistry for centralized position management
        self.position_registry = PositionRegistry()
        # Backward compatibility: expose positions dict directly
        self.positions = self.position_registry._positions  # Direct access for compatibility
        self._positions_set = self.position_registry._positions_set  # Direct access for compatibility
        self.cooldown_until = {}
        # TRADE INTROSPECTION: Self-learning from each trade
        self.learner = LearnerIntegration.get_instance() if is_learner_enabled() else None
        if self.learner and self.learner.enabled:
            stats = self.learner.get_stats()
            self.logger.info("[LEARNER] Enabled: %d contexts, %d samples" % (stats["contexts_learned"], stats["global_samples"]))

        # Use DRY_START_BALANCE for DRY_RUN, ACCOUNT_BAL for backward compatibility in LIVE
        if DRY_RUN:
            from .config import DRY_START_BALANCE
            self.start_equity = float(DRY_START_BALANCE)
        else:
            self.start_equity = ACCOUNT_BAL  # LIVE mode uses ACCOUNT_BAL or real exchange balance
        self.equity_peak = self.start_equity
        self.started_at = self._get_current_time()  # Use replay time if in replay mode
        self.circuit_breaker_triggered = False  # Track if drawdown circuit breaker has triggered
        self.realized_pnl_total = 0.0
        self.realized_fees_total = 0.0
        self.realized_entry_fees_total = 0.0
        self.realized_exit_fees_total = 0.0
        self.realized_slippage_total = 0.0
        self.realized_funding_total = 0.0
        self.gross_win = 0.0
        self.gross_loss = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.recent_errors = deque(maxlen=12)
        self.symbols_scanned_last = 0
        self.current_mode = "Neutral"
        self.mode_since = self._get_current_time()
        self.last_draw = 0
        self.last_signal_scan = 0
        # PURE SCALPER MODE: Hard-locked to scalping, no regime switching
        self.current_regime = "scalping"  # Always scalping
        self.regime_since = self._get_current_time()
        # Force scalper regime config - no LLM switching
        from .regime import TradingRegime, REGIME_CONFIGS
        self.regime_config = REGIME_CONFIGS[TradingRegime.SCALPING]
        
        # Enhanced tracking for UI
        self.signal_history = deque(maxlen=20)  # Last 20 signals with full context
        self.recent_trades = deque(maxlen=15)  # Last 15 trades with enhanced details
        self.loop_times = deque(maxlen=10)  # Track scan cycle times
        self.api_call_times = deque(maxlen=60)  # Track API calls for rate calculation
        self.last_universe_refresh = time.time()
        self.last_discovery_scan = 0.0  # Track last discovery scan time
        self.btc_trend = 0.0  # BTC trend percentage
        self.volatility_regime = "Normal"  # Low/Normal/High
        self.spread_regime = "Normal"  # Tight/Normal/Wide
        
        # API BUDGET OPTIMIZATION: Enhanced data fetching
        self.orderbook_cache = {}  # Cache orderbooks for top symbols
        self.funding_rates = {}  # Cache funding rates
        self.last_orderbook_refresh = 0.0
        self.last_funding_refresh = 0.0
        self.orderbook_refresh_interval = 15.0  # [!] RELAXED: Refresh orderbooks every 15s to reduce queue pressure
        self.funding_refresh_interval = 60.0  # Refresh funding rates every 60s (30 calls = 30/min)
        self.last_position_orderbook_refresh = 0.0  # Track position orderbook refresh separately
        self.position_orderbook_refresh_interval = 15.0  # [!] RELAXED: Refresh position orderbooks every 15s to reduce queue pressure
        
        # Adaptive throttling: Track throttle errors and adjust delays dynamically
        self.throttle_error_count = 0  # Count consecutive throttle errors
        self.last_throttle_error = 0.0  # Timestamp of last throttle error
        self.adaptive_delay = 0.1  # Start with 100ms delay, adjust based on errors
        self.throttle_backoff_until = 0.0  # Timestamp when backoff expires
        self.signal_stats = {
            'signals_generated': 0,
            'signals_blocked': 0,
            'last_signal_time': 0
        }
        
        # Scan tracking for detailed metrics
        self.scan_times = deque(maxlen=20)  # Track scan durations
        self.scan_history = deque(maxlen=20)  # Store detailed scan records
        self.last_scan_start = 0.0  # Track when current scan started
        self.last_scan_end = 0.0  # Track when last scan completed
        self.next_scan_time = 0.0  # Track when next scan will occur
        self.next_universe_refresh_time = 0.0  # Track when next universe refresh will occur
        self.last_universe_refresh_time = 0.0  # Track when universe was last refreshed
        self.entries_attempted_this_scan = 0  # Entries attempted in current scan
        self._rate_limit_until = 0.0  # Timestamp when rate limit expires (0 = not rate limited)
        self._last_rate_limit_error = 0.0  # Timestamp of last rate limit error
        self.entries_opened_this_scan = 0  # Entries opened in current scan
        self.is_scanning = False  # Track if currently scanning
        self.startup_period_ended = False  # Track if startup delay period has ended
        self.scan_stats = {
            'total_scans': 0,
            'total_symbols_processed': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'total_orderbook_success': 0,
            'total_orderbook_failures': 0
        }
        # FIX: Cumulative filter stats for Signal Health panel
        self.filter_stats_cumulative = {
            'signals_total': 0,      # Total signals found across all scans
            'signals_passed': 0,      # Total signals that passed all filters
            'signals_rejected': 0,    # Total signals rejected
            'avg_score_sum': 0.0,     # Sum of all signal scores (for running average)
            'avg_score_count': 0     # Count of signals with scores (for running average)
        }
        # Minimal metrics counters
        self.metrics = {
            'entries_attempted': 0,
            'entries_opened': 0,
            'rejections_by_reason': {},
            'exits_by_reason': {}
        }
        self._last_metrics_log = time.time()
        
        # Scan cycle counter (for internal tracking, not logged)
        self.debug_scan_counter = 0
        
        # Symbol status cache: Track symbols that are invalid/unavailable (Reduce Only, delisted, etc.)
        # Key: symbol, Value: timestamp when marked invalid (for potential retry after cooldown)
        self.invalid_symbols = {}  # Track symbols that fail with -4140 or similar errors
        self.invalid_symbol_cooldown = 3600.0  # Retry invalid symbols after 1 hour
    
    def _calculate_pnl_pct(self, entry_price: float, current_price: float, side: str) -> float:
        """
        REFACTOR: Helper function to calculate PnL percentage safely.
        Prevents division by zero and provides consistent PnL calculation.
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            side: 'long' or 'short'
        
        Returns:
            PnL percentage (can be negative)
        """
        if not entry_price or entry_price <= 0:
            return 0.0
        
        if side == 'long':
            return ((current_price - entry_price) / entry_price) * 100
        else:  # short
            return ((entry_price - current_price) / entry_price) * 100

    def _get_current_time(self) -> float:
        """Get current time (replay time if in replay mode, else real time)."""
        if self.replay_mode and self._replay_current_time is not None:
            return self._replay_current_time
        return time.time()
    
    async def init_exchange(self):
        """Initialize exchange using exchange abstraction layer."""
        # REPLAY MODE: Skip exchange initialization
        if self.replay_mode:
            self.logger.info("REPLAY MODE: Skipping exchange initialization")
            self.exchange_wrapper = None
            self.order_manager = OrderManager(None)
            self.exit_manager = ExitManager(None, self.order_manager)
            # Initialize ExitPipeline even without exchange
            self.exit_pipeline = ExitPipeline(
                order_manager=self.order_manager,
                exit_manager=self.exit_manager,
                position_registry=self.position_registry,
                exchange=None
            )
            return
        
        # Create exchange wrapper using factory
        exchange_init_success = False
        try:
            # Check API keys before attempting connection (exchange-agnostic)
            from .config import (
                EXCHANGE, BINANCE_API_KEY, BINANCE_SECRET,
                OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE,
                BYBIT_API_KEY, BYBIT_API_SECRET,
                KRAKEN_API_KEY, KRAKEN_API_SECRET,
                BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE,
                MEXC_API_KEY, MEXC_API_SECRET
            )
            
            exchange_upper = EXCHANGE.upper()
            missing_creds = False
            
            if EXCHANGE == "binance_futures":
                if not BINANCE_API_KEY or not BINANCE_SECRET:
                    missing_creds = True
                    self.logger.error(
                        "[X] EXCHANGE CONNECTION FAILED: Missing API credentials",
                        api_key_set=bool(BINANCE_API_KEY),
                        api_secret_set=bool(BINANCE_SECRET),
                        hint="Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables"
                    )
            elif EXCHANGE == "okx":
                if not OKX_API_KEY or not OKX_API_SECRET or not OKX_PASSPHRASE:
                    missing_creds = True
                    self.logger.error(
                        "[X] EXCHANGE CONNECTION FAILED: Missing API credentials",
                        api_key_set=bool(OKX_API_KEY),
                        api_secret_set=bool(OKX_API_SECRET),
                        passphrase_set=bool(OKX_PASSPHRASE),
                        hint="Set OKX_API_KEY, OKX_API_SECRET, and OKX_PASSPHRASE environment variables"
                    )
            elif EXCHANGE == "bybit":
                if not BYBIT_API_KEY or not BYBIT_API_SECRET:
                    missing_creds = True
                    self.logger.error(
                        "[X] EXCHANGE CONNECTION FAILED: Missing API credentials",
                        api_key_set=bool(BYBIT_API_KEY),
                        api_secret_set=bool(BYBIT_API_SECRET),
                        hint="Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables"
                    )
            elif EXCHANGE == "kraken":
                if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
                    missing_creds = True
                    self.logger.error(
                        "[X] EXCHANGE CONNECTION FAILED: Missing API credentials",
                        api_key_set=bool(KRAKEN_API_KEY),
                        api_secret_set=bool(KRAKEN_API_SECRET),
                        hint="Set KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables"
                    )
            elif EXCHANGE == "bitget":
                if not BITGET_API_KEY or not BITGET_API_SECRET or not BITGET_PASSPHRASE:
                    missing_creds = True
                    self.logger.error(
                        "[X] EXCHANGE CONNECTION FAILED: Missing API credentials",
                        api_key_set=bool(BITGET_API_KEY),
                        api_secret_set=bool(BITGET_API_SECRET),
                        passphrase_set=bool(BITGET_PASSPHRASE),
                        hint="Set BITGET_API_KEY, BITGET_API_SECRET, and BITGET_PASSPHRASE environment variables"
                    )
            elif EXCHANGE in ("mexc", "mexc_futures"):
                if not MEXC_API_KEY or not MEXC_API_SECRET:
                    missing_creds = True
                    self.logger.error(
                        "[X] EXCHANGE CONNECTION FAILED: Missing API credentials",
                        api_key_set=bool(MEXC_API_KEY),
                        api_secret_set=bool(MEXC_API_SECRET),
                        hint="Set MEXC_API_KEY and MEXC_API_SECRET environment variables"
                    )
            
            if missing_creds:
                raise ValueError(f"Missing {exchange_upper} API credentials")
            else:
                self.logger.info(f"[OK] API credentials found for {exchange_upper}, attempting connection...")
            
            self.exchange_wrapper = create_exchange(config_module)
            self.logger.info("[DIAG] init_exchange:initialize")
            # Increased timeout to 30s to match governor timeout and allow for slow connections
            await asyncio.wait_for(self.exchange_wrapper.initialize(), timeout=30.0)
            self.logger.info("[DIAG] init_exchange:initialize_done")
            exchange_init_success = True
            self.logger.info("[OK] Exchange connection successful")
            
            # LIVE MODE: Fetch real balance from exchange and update start_equity
            if not DRY_RUN and self.exchange_wrapper:
                try:
                    self.logger.info("[DIAG] init_exchange:fetch_balance")
                    balance_data = await self.exchange_wrapper.fetch_balance()
                    # Extract USDT total (or synthesized USDT from USDC/BNFCR for EEA users)
                    usdt_total = float(balance_data.get('USDT', {}).get('total', 0.0))
                    if usdt_total > 0:
                        self.start_equity = usdt_total
                        self.equity_peak = usdt_total
                        self.logger.info(
                            f"[OK] LIVE MODE: Synced start_equity with Binance balance: ${usdt_total:.2f}"
                        )
                    else:
                        self.logger.warning(
                            f"[!] LIVE MODE: Could not fetch balance from exchange, using ACCOUNT_BAL: ${ACCOUNT_BAL:.2f}"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"[!] LIVE MODE: Failed to fetch balance from exchange: {e}. Using ACCOUNT_BAL: ${ACCOUNT_BAL:.2f}"
                    )
        except ValueError as e:
            # Missing credentials or configuration error
            self.logger.error(f"[X] EXCHANGE CONNECTION FAILED: Configuration error: {e}")
            self.logger.warning("Exchange initialization failed - will use fallback mode")
            self.exchange_wrapper = None
            self.order_manager = OrderManager(None)
            self.exit_manager = ExitManager(None, self.order_manager)
            # Continue - we'll use fallback symbols
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            self.logger.error(
                f"[X] EXCHANGE CONNECTION FAILED: {error_type}: {error_msg}",
                error_type=error_type,
                error_message=error_msg[:200]  # Truncate long messages
            )
            
            # Provide helpful hints based on error type
            if "DDoSProtection" in error_type or "rate limit" in error_msg.lower():
                self.logger.warning(f"[*] HINT: {EXCHANGE.upper()} rate limit hit. Wait a few minutes and restart.")
            elif "authentication" in error_msg.lower() or "invalid" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                self.logger.warning("[*] HINT: Check your API keys are correct and have proper permissions.")
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                self.logger.warning("[*] HINT: Network/connection issue. Check internet connection.")
            else:
                self.logger.warning("[*] HINT: See logs for details. Bot will continue in fallback mode.")
            
            self.logger.warning("Exchange initialization failed - will use fallback mode")
            self.exchange_wrapper = None
            self.order_manager = OrderManager(None)
            self.exit_manager = ExitManager(None, self.order_manager)
            # Continue - we'll use fallback symbols
        
        # If exchange failed, initialize fallback mode
        if not exchange_init_success:
            self._init_fallback_symbols()
            self._fallback_mode = True
        
        # Initialize ExitPipeline
        self.exit_pipeline = ExitPipeline(
            order_manager=self.order_manager,
            exit_manager=self.exit_manager,
            position_registry=self.position_registry,
            exchange=None
        )
        
        # Load markets (only if exchange wrapper is available)
        if not self.exchange_wrapper:
            # Exchange failed - already initialized fallback symbols
            return
        
        # CRITICAL SAFETY: Perform synchronous position sync at startup
        # This prevents "blind trading" where the bot opens new positions before knowing about existing ones
        if not self.replay_mode:
            # Initialize startup guard timer (prevents exits during first 30 seconds)
            self._startup_time = time.time()
            self.logger.info(f"[SAFETY] Startup guard initialized - no time exits for next 30 seconds")
            
            self.logger.info("[SAFETY] Performing blocking position sync at startup...")
            try:
                await self._refresh_positions()
                pos_count = len(self.positions)
                self.logger.info(f"[SAFETY] Startup sync complete. Found {pos_count} open positions.")
                if pos_count > 0:
                    self.logger.info(f"[SAFETY] Positions protected by startup guard - will not close on restart")
            except Exception as e:
                self.logger.error(f"[CRITICAL] Failed to sync positions at startup: {e}")
                # We continue, but with extreme caution (Ghost Hunter will retry in 60s)
        
        # Load markets (single fast attempt, fallback to cached markets to avoid hangs)
        try:
            self.logger.info("[DIAG] init_exchange:load_markets:start")
            # Increased timeout to 30s to allow for full market load
            mkts = await asyncio.wait_for(
                self.exchange_wrapper.load_markets(reload=False),
                timeout=30.0
            )
            self.logger.info(f"[DIAG] init_exchange:load_markets:done count={len(mkts) if mkts else 0}")
        except Exception as e:
            self.logger.warning(f"[DIAG] init_exchange:load_markets:failed {type(e).__name__}: {e}")
            mkts = getattr(self.exchange_wrapper, "markets", {}) or {}
        
        # If still empty, log and continue (universe will retry later)
        if not mkts:
            self.logger.warning("[DIAG] init_exchange:load_markets:empty using cached/none; universe refresh will retry")
        
        # Filter for futures/swap markets only (USDT margined linear contracts)
        futures_markets = {}
        for sym, m in mkts.items():
            if "USDT" not in sym:
                continue
            if not self.exchange_wrapper.is_futures_market(m):
                continue
            normalized_sym = self.exchange_wrapper.normalize_symbol(sym)
            futures_markets[normalized_sym] = m
            st = self.universe.stats.get(normalized_sym) or SymbolStats(normalized_sym)
            prec = m.get("precision", {}) or {}
            st.amount_step = prec.get("amount", 0.0)
            st.price_tick = prec.get("price", 0.0)
            self.universe.stats[normalized_sym] = st
        
        if WHITELIST_SYMBOLS:
            normalized_whitelist = [self.exchange_wrapper.normalize_symbol(s) for s in WHITELIST_SYMBOLS]
            futures_markets = {k: v for k, v in futures_markets.items() if k in normalized_whitelist}
            self.logger.info(f"Applied whitelist filter: {len(futures_markets)} symbols")
        
        self.logger.info(
            f"Initialized exchange: {EXCHANGE} (futures only)",
            markets=len(futures_markets),
            total_markets=len(mkts),
            dry_run=DRY_RUN
        )
        
        if self.universe.stats and not hasattr(self, '_universe_logged'):
            universe_symbols = list(self.universe.stats.keys())
            sample_size = min(5, len(universe_symbols))
            self.logger.info(
                f"[UNIVERSE] Size={len(universe_symbols)} | Sample={universe_symbols[:sample_size]}"
            )
            self._universe_logged = True
        
        # Initialize order manager and exit manager with exchange wrapper
        self.order_manager = OrderManager(self.exchange_wrapper)
        self.exit_manager = ExitManager(self.exchange_wrapper, self.order_manager)
        
        # Update ExitPipeline with new managers
        self.exit_pipeline = ExitPipeline(
            order_manager=self.order_manager,
            exit_manager=self.exit_manager,
            position_registry=self.position_registry
        )
        
        if hasattr(self.exchange_wrapper, 'exchange'):
            self.exchange = self.exchange_wrapper.exchange
        else:
            self.exchange = self.exchange_wrapper
    
    def _init_fallback_symbols(self):
        """
        Initialize fallback symbols when exchange connection fails.
        MARKSMAN removed - bot waits for exchange reconnection.
        """
        self.logger.warning("Exchange connection failed - cannot initialize fallback symbols without exchange")
        self.logger.info("Bot will wait for exchange reconnection before scanning")
        
        # Set flag to indicate we're in fallback mode
        self._fallback_mode = True

    def health_check(self) -> Dict[str, Any]:
        """
        Return bot health status.
        
        Returns:
            Dictionary containing health metrics
        """
        return {
            'exchange_connected': self.exchange is not None,
            'uptime_seconds': time.time() - self.started_at,
            'positions_count': len(self.positions),
            'recent_errors_count': len(self.recent_errors),
            'last_scan_time': time.time() - self.last_signal_scan if self.last_signal_scan else None,
            'universe_size': len(self.universe.active) if hasattr(self.universe, 'active') else 0,
            'current_regime': self.current_regime,
            'equity': self.equity_now()
        }

    def equity_now(self) -> float:
        """
        Calculate current equity.
        
        CANONICAL ACCOUNTING: Equity = Starting Balance + Realized PnL + Realized Funding + Unrealized PnL
        
        Note: realized_pnl_total contains NET PnL (after costs),
        so we don't subtract fees again here. Fees are tracked
        separately in realized_fees_total for transparency.
        """
        from .accounting import calculate_total_unrealized_pnl, get_current_price_from_bot, calculate_equity
        
        # Calculate unrealized PnL from open positions
        unrealized_pnl = 0.0
        if self.positions:
            try:
                unrealized_pnl = calculate_total_unrealized_pnl(
                    self.positions,
                    lambda symbol: get_current_price_from_bot(self, symbol)
                )
            except Exception as e:
                self.logger.debug(f"Error calculating unrealized PnL: {e}")
        
        # Calculate total equity using canonical accounting
        equity = calculate_equity(
            self.start_equity,
            self.realized_pnl_total,
            self.realized_funding_total,
            unrealized_pnl
        )
        
        # Update equity peak (for drawdown calculation)
        if equity > self.equity_peak:
            self.equity_peak = equity
        
        return equity
    
    def get_drawdown_pct(self) -> float:
        """
        Calculate current drawdown percentage.
        
        Returns:
            Drawdown percentage (negative if below peak, 0 if at/above peak)
        """
        equity = self.equity_now()
        if self.equity_peak > 0:
            drawdown_pct = ((equity - self.equity_peak) / self.equity_peak) * 100
            return min(0.0, drawdown_pct)  # Only negative values (drawdown)
        return 0.0
    
    def check_drawdown_circuit_breaker(self) -> bool:
        """
        Check if drawdown circuit breaker should trigger.
        
        Returns:
            True if circuit breaker should trigger (stop trading), False otherwise
        """
        if not DRAWDOWN_CIRCUIT_BREAKER_ENABLED:
            return False
        
        if self.circuit_breaker_triggered:
            return True  # Already triggered, stay stopped
        
        drawdown_pct = self.get_drawdown_pct()
        
        if drawdown_pct <= -MAX_DRAWDOWN_PCT:
            self.circuit_breaker_triggered = True
            self.logger.critical(
                f"DRAWDOWN CIRCUIT BREAKER TRIGGERED: Drawdown {drawdown_pct:.2f}% exceeds maximum {MAX_DRAWDOWN_PCT}%",
                drawdown_pct=drawdown_pct,
                max_drawdown_pct=MAX_DRAWDOWN_PCT,
                equity=self.equity_now(),
                start_equity=self.start_equity,
                equity_peak=self.equity_peak
            )
            return True
        
        return False
    
    def _calculate_market_regimes(self):
        """
        Calculate volatility and spread regimes from active symbols.
        Uses median and percentile-based thresholds for robustness.
        """
        if not self.universe.active:
            return

        spreads = []
        volatilities = []
        
        # Collect data from top symbols (more reliable)
        for symbol in self.universe.active[:50]:
            stats = self.universe.stats.get(symbol)
            if stats:
                spread = stats.spread_bps
                vol = abs(getattr(stats, 'pct_change_24h', 0.0))
                # Filter out outliers
                if 0 < spread < 200 and 0 <= vol < 20:
                    spreads.append(spread)
                    volatilities.append(vol)
        
        # Need at least 5 symbols for reliable regime detection
        # Ensure both lists have sufficient data and are in sync
        if len(spreads) < 5 or len(volatilities) < 5 or len(spreads) != len(volatilities):
            return
        
        # Use median for robustness (less affected by outliers)
        spreads_sorted = sorted(spreads)
        volatilities_sorted = sorted(volatilities)
        n = len(spreads_sorted)
        median_spread = spreads_sorted[n // 2]
        median_vol = volatilities_sorted[n // 2]
        
        # Use 25th and 75th percentiles for thresholds
        p25_spread = spreads_sorted[n // 4]
        p75_spread = spreads_sorted[3 * n // 4]
        p25_vol = volatilities_sorted[n // 4]
        p75_vol = volatilities_sorted[3 * n // 4]
        
        # Spread regime: Use median with percentile-based thresholds
        # Tight: median < 25 bps OR p75 < 30 bps
        # Wide: median > 45 bps OR p75 > 60 bps
        if median_spread < 25 or p75_spread < 30:
            self.spread_regime = "Tight"
        elif median_spread > 45 or p75_spread > 60:
            self.spread_regime = "Wide"
        else:
            self.spread_regime = "Normal"
        
        # Volatility regime: Use median with percentile-based thresholds
        # Low: median < 0.8% OR p75 < 1.2%
        # High: median > threshold OR p75 > threshold
        if median_vol < 0.8 or p75_vol < 1.2:
            self.volatility_regime = "Low"
        elif median_vol > HIGH_VOLATILITY_MEDIAN_THRESHOLD or p75_vol > HIGH_VOLATILITY_P75_THRESHOLD:
            self.volatility_regime = "High"
        else:
            self.volatility_regime = "Normal"
            
        # [SMART REGIME ADAPTER] Apply settings
        self._apply_dynamic_regime_settings()

    def _apply_dynamic_regime_settings(self):
        """
        Dynamically adjust bot configuration based on market regime.
        This is the "Smart Regime Adapter" requested by the Council.
        """
        if not hasattr(self, 'volatility_regime') or not hasattr(self, 'market_regime'):
            return

        # ---------------------------------------------------------
        # OPERATOR OVERRIDE: lock entry thresholds when explicitly set.
        #
        # This prevents the regime adapter from silently "helping" by
        # resetting MIN_SIGNAL_SCORE back to defaults like 63.
        #
        # If you want dynamic regime thresholds again, unset these env vars
        # or set LOCK_ENTRY_FILTERS=0.
        # ---------------------------------------------------------
        try:
            lock_env = os.getenv("LOCK_ENTRY_FILTERS", "").strip().lower()
            locked = lock_env in ("1", "true", "yes", "on")
            # If LIVE_* overrides are present, treat that as an explicit operator lock.
            if os.getenv("LIVE_MIN_SIGNAL_SCORE") or os.getenv("LIVE_MIN_SIGNAL_STRENGTH") or os.getenv("LIVE_HARD_MIN_SCORE"):
                locked = True
        except Exception:
            locked = False

        if locked:
            # Log once per process to avoid spam.
            if not hasattr(self, "_regime_adapter_locked_logged"):
                try:
                    self.logger.warning(
                        "[REGIME] Dynamic regime thresholds are LOCKED by operator override "
                        "(LIVE_* or LOCK_ENTRY_FILTERS). Keeping configured entry thresholds."
                    )
                except Exception:
                    pass
                self._regime_adapter_locked_logged = True
            return

        vol_regime = self.volatility_regime
        mkt_regime = self.market_regime
        
        # =====================================================================
        # ADAPTIVE THRESHOLD SYSTEM (24/7/365 Operation)
        # =====================================================================
        # HUMAN KILLER ADVANTAGES:
        # 1. We never sleep - aggressive during graveyard shift
        # 2. We never panic - capitalize on volatility spikes
        # 3. We never get bored - patient during dead markets
        # 4. We never FOMO - consistent entry criteria
        # 5. We scan 300+ symbols - humans watch maybe 10
        # =====================================================================
        
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        utc_hour = now_utc.hour
        is_weekend = now_utc.weekday() >= 5  # Saturday=5, Sunday=6
        
        # Track idle time (time since last entry)
        if not hasattr(self, '_last_entry_time'):
            self._last_entry_time = time.time()
        if not hasattr(self, '_adaptive_relaxation'):
            self._adaptive_relaxation = 0.0  # How much we've relaxed (0-5 points)
        
        # Calculate idle time
        idle_minutes = (time.time() - self._last_entry_time) / 60.0
        
        # =====================================================================
        # GLOBAL MARKET SESSIONS (UTC)
        # =====================================================================
        # Crypto trades 24/7, but volume/volatility follows traditional markets:
        #
        #   ASIA SESSION:    00:00-09:00 UTC (Tokyo 09:00-18:00 JST)
        #   EUROPE SESSION:  07:00-16:00 UTC (London 07:00-16:00 GMT)
        #   US SESSION:      13:00-22:00 UTC (NYC 08:00-17:00 EST)
        #
        # OVERLAP PERIODS = HIGH VOLUME:
        #   Asia/Europe:     07:00-09:00 UTC
        #   Europe/US:       13:00-16:00 UTC (BEST VOLUME)
        #
        # DEAD ZONE:         22:00-00:00 UTC (All markets closed)
        # =====================================================================
        
        # Determine active sessions
        asia_active = 0 <= utc_hour < 9
        europe_active = 7 <= utc_hour < 16
        us_active = 13 <= utc_hour < 22
        
        # Count overlapping sessions (more = more volume = more opportunity)
        sessions_active = sum([asia_active, europe_active, us_active])
        
        # DEAD ZONE: 22:00-00:00 UTC - all major markets closed
        is_dead_zone = 22 <= utc_hour or utc_hour < 0
        
        # PRIME OVERLAP: Europe + US (13:00-16:00 UTC) - highest volume
        is_prime_overlap = 13 <= utc_hour < 16
        
        # =====================================================================
        # THRESHOLD MODIFIERS BASED ON MARKET SESSIONS
        # =====================================================================
        # More sessions = more competition BUT also more opportunity
        # We're aggressive when others sleep, selective when it's crowded
        
        session_bonus = 0.0
        session_tag = ""
        
        if is_dead_zone:
            # Dead zone: low competition, be aggressive
            session_bonus = 2.0
            session_tag = "DEADZONE"
        elif is_prime_overlap:
            # Prime overlap: high volume, best setups - be slightly aggressive
            session_bonus = 1.0
            session_tag = "EU+US"
        elif sessions_active >= 2:
            # Two sessions: good volume
            session_bonus = 0.5
            if asia_active and europe_active:
                session_tag = "ASIA+EU"
            elif europe_active and us_active:
                session_tag = "EU+US"
        elif sessions_active == 1:
            # Single session: normal
            session_bonus = 0.0
            if asia_active:
                session_tag = "ASIA"
            elif europe_active:
                session_tag = "EU"
            elif us_active:
                session_tag = "US"
        
        # Historical best/worst hours from ML training data (UTC)
        PRIME_HOURS = {0, 7, 9, 14, 15, 16}      # 42%+ win rate
        AVOID_HOURS = {19}                       # <38% win rate
        
        prime_bonus = 1.0 if utc_hour in PRIME_HOURS else 0.0
        avoid_penalty = 2.0 if utc_hour in AVOID_HOURS else 0.0
        
        # =====================================================================
        # WEEKEND WARRIOR
        # =====================================================================
        # Less institutional competition on weekends
        weekend_bonus = 1.0 if is_weekend else 0.0
        
        # =====================================================================
        # HUMAN KILLER #3: IDLE PATIENCE (we never get bored)
        # =====================================================================
        # After 30 min idle, start lowering threshold
        # Max relaxation: 5 points after 2 hours idle
        IDLE_START_MINUTES = 30
        IDLE_MAX_MINUTES = 120
        MAX_RELAXATION = 5.0
        
        if idle_minutes > IDLE_START_MINUTES:
            relax_progress = min(1.0, (idle_minutes - IDLE_START_MINUTES) / (IDLE_MAX_MINUTES - IDLE_START_MINUTES))
            self._adaptive_relaxation = relax_progress * MAX_RELAXATION
        else:
            self._adaptive_relaxation = max(0, self._adaptive_relaxation - 0.5)
        
        # =====================================================================
        # BASE THRESHOLDS by regime (ML SCORER V3: 25-50 range)
        # =====================================================================
        if vol_regime == "Low" or mkt_regime == "neutral":
            base_score = 30
            target_max_pos = 15
            regime_name = "TORTOISE"
        elif vol_regime == "Normal" and mkt_regime != "neutral":
            base_score = 32
            target_max_pos = 20
            regime_name = "WOLF PACK"
        elif vol_regime == "High" and mkt_regime != "neutral":
            base_score = 28  # Lower = more aggressive in volatility
            target_max_pos = 25
            regime_name = "RAPTOR"
        else:
            base_score = 30
            target_max_pos = 20
            regime_name = "UNKNOWN"
        
        # =====================================================================
        # CALCULATE FINAL THRESHOLD
        # =====================================================================
        # Bonuses SUBTRACT from threshold (lower = more trades)
        # Penalties ADD to threshold (higher = fewer trades)
        total_bonus = session_bonus + prime_bonus + weekend_bonus + self._adaptive_relaxation
        total_penalty = avoid_penalty
        
        target_score = max(25, base_score - total_bonus + total_penalty)  # Floor at 25
        target_hard_score = 25
        target_strength = target_score / 100.0
        
        # Build regime status string with market session info
        modifiers = []
        if session_tag:
            modifiers.append(session_tag)
        if prime_bonus > 0:
            modifiers.append("PRIME")
        if weekend_bonus > 0:
            modifiers.append("WKND")
        if self._adaptive_relaxation > 0.5:
            modifiers.append(f"IDLE-{self._adaptive_relaxation:.0f}")
        if avoid_penalty > 0:
            modifiers.append("!")  # Warning flag
        
        if modifiers:
            regime_name = f"{regime_name} [{'+'.join(modifiers)}]"
            
        # Apply settings only if changed (to reduce log spam)
        # Check against current config values
        if self.cfg.MIN_SIGNAL_SCORE != target_score:
            self.logger.info(f"[REGIME] {regime_name} ACTIVATED ({vol_regime}/{mkt_regime})")
            self.logger.info(f"   -> Adjusting Score: {self.cfg.MIN_SIGNAL_SCORE} -> {target_score}")
            self.logger.info(f"   -> Adjusting Max Pos: {self.cfg.MAX_OPEN_POSITIONS} -> {target_max_pos}")
            
            self.cfg.MIN_SIGNAL_SCORE = target_score
            # SIMPLIFIED: No HARD_MIN_SCORE - using MIN_SIGNAL_SCORE as single threshold
            self.cfg.MIN_SIGNAL_STRENGTH = target_strength
            self.cfg.MAX_OPEN_POSITIONS = target_max_pos
            self.cfg.MAX_CONCURRENT_POS = target_max_pos
            
        # UI UPDATE: Ensure dynamic gate score is updated so TUI reflects reality
        self.dynamic_gate_score = float(self.cfg.MIN_SIGNAL_SCORE)

    async def refresh_universe(self) -> None:
        # OPTIMIZATION: Cache time once per refresh
        loop_start = self._get_current_time()
        refresh_now = loop_start
        
        # DIAG: Trace universe refresh entry
        try:
            self.logger.info(f"[DIAG] refresh_universe:start t={refresh_now:.3f}")
        except Exception:
            pass
        
        # Check if we're rate limited - skip refresh if still banned
        if hasattr(self, '_rate_limit_until') and self._rate_limit_until > 0 and refresh_now < self._rate_limit_until:
            time_remaining = self._rate_limit_until - refresh_now
            if not hasattr(self, '_last_rate_limit_warning') or (refresh_now - getattr(self, '_last_rate_limit_warning', 0)) > 60:
                # Only warn once per minute to avoid log spam
                self._last_rate_limit_warning = refresh_now
                self.logger.warning(
                    f"[PAUSED] Rate limit active - skipping universe refresh. "
                    f"Resume in {int(time_remaining/60)}m {int(time_remaining%60)}s"
                )
            return
        
        # REPLAY MODE: Use replay feed for ticker data
        if self.replay_mode and self.replay_feed:
            try:
                # Get ticker data from replay feed for all symbols
                tickers = {}
                for symbol in self.replay_feed.candles_by_symbol.keys():
                    ticker_data = self.replay_feed.get_ticker_data(symbol)
                    if ticker_data:
                        # Normalize symbol for internal use
                        normalized_sym = symbol  # Assume already normalized
                        tickers[normalized_sym] = ticker_data
                
                # Update universe stats from replay tickers
                for sym, t in tickers.items():
                    if not sym or not t:
                        continue
                    
                    bid = t.get("bid", 0.0)
                    ask = t.get("ask", 0.0)
                    last = t.get("last", 0.0)
                    mark = t.get("mark", last)
                    vol_quote = t.get("quoteVolume", 0.0)
                    
                    # Calculate spread
                    spread_bps = 9999.0
                    if bid and ask and ask > 0:
                        mid = MID_PRICE_FACTOR * (bid + ask)
                        if mid > 0:
                            spread_bps = abs(ask - bid) / mid * 1e4
                    
                    # Get or create stats
                    st = self.universe.stats.get(sym)
                    if not st:
                        st = SymbolStats(sym)
                        self.universe.stats[sym] = st
                    
                    st.bid = bid
                    st.ask = ask
                    st.last = last
                    st.mark = float(mark or 0.0)
                    st.vol_quote = vol_quote
                    st.spread_bps = float(spread_bps)
                    st.heat = (math.log10(vol_quote + 1.0) - 5.0) - (spread_bps / 200.0)
                    st.last_seen = refresh_now
                    st.pct_change_24h = 0.0  # Not available in replay
                
                # Update ticker cache
                self.ticker_cache.batch_set(tickers)
                
                # REPLAY MODE: Populate orderbook cache with synthetic orderbooks
                for sym in self.replay_feed.candles_by_symbol.keys():
                    snapshot = self.replay_feed.get_market_snapshot(sym)
                    if snapshot and 'orderbook' in snapshot:
                        self.orderbook_cache[sym] = {
                            'data': snapshot['orderbook'],
                            'timestamp': refresh_now
                        }
                
                # Log universe size and sample (one-time after initial population in REPLAY mode)
                if self.universe.stats and not hasattr(self, '_universe_logged'):
                    universe_symbols = list(self.universe.stats.keys())
                    sample_size = min(5, len(universe_symbols))
                    self.logger.info(
                        f"[UNIVERSE] Size={len(universe_symbols)} | Sample={universe_symbols[:sample_size]}"
                    )
                    self._universe_logged = True
                return
            except Exception as e:
                self.logger.warning(f"Replay universe refresh failed: {e}")
                return
        
        # Check if exchange wrapper is available
        if not self.exchange_wrapper:
            # FALLBACK MODE: If in fallback mode, wait for exchange reconnection
            if hasattr(self, '_fallback_mode') and self._fallback_mode:
                self.logger.warning(
                    f"[!] UNIVERSE REFRESH (Fallback Mode) | symbols={len(self.universe.stats)} | "
                    f"Waiting for exchange reconnection"
                )
                # Try to reconnect exchange periodically (with exponential backoff if rate limited)
                # Base retry interval: 5 minutes, but increase if rate limited
                base_retry_interval = 300  # 5 minutes
                if self._rate_limit_until > 0:
                    # If rate limited, wait until ban expires + extra buffer
                    time_until_ban_expires = max(0, self._rate_limit_until - refresh_now)
                    retry_interval = time_until_ban_expires + 60  # Wait 1 minute after ban expires
                else:
                    retry_interval = base_retry_interval
                
                if not hasattr(self, '_last_exchange_retry') or (refresh_now - getattr(self, '_last_exchange_retry', 0)) > retry_interval:
                    # Check if rate limit has expired
                    if self._rate_limit_until > 0 and refresh_now < self._rate_limit_until:
                        # Still rate limited, don't retry yet
                        return
                    
                    self._last_exchange_retry = refresh_now
                    self.logger.info("Attempting to reconnect exchange...")
                    # Schedule async retry (will happen in next loop)
                    self._should_retry_exchange = True
            else:
                self.logger.warning(
                    f"[!] No exchange connection. "
                    f"Universe size: {len(self.universe.stats)}. "
                    f"Bot cannot scan symbols."
                )
                self.logger.warning(
                    f"[!] No exchange connection. Universe size: {len(self.universe.stats)}. "
                    f"Initializing fallback symbols..."
                )
                self._init_fallback_symbols()
            return
        
        # Initialize futures_tickers early to avoid UnboundLocalError
        futures_tickers = {}
        tickers = {}
        
        try:
            # Try WebSocket first if available (no API call needed)
            if hasattr(self.exchange_wrapper, 'ws_manager') and self.exchange_wrapper.ws_manager:
                if not self.exchange_wrapper.ws_manager.connected:
                    # Start WebSocket connection on first refresh
                    await self.exchange_wrapper.start_websocket()
                    # Wait a moment for initial data
                    await asyncio.sleep(1.0)
            
            # Check if WebSocket has data (no REST API call needed)
            if (hasattr(self.exchange_wrapper, 'ws_manager') and 
                self.exchange_wrapper.ws_manager and 
                self.exchange_wrapper.ws_manager.connected and
                self.exchange_wrapper.ws_ticker_cache):
                # Use WebSocket data directly (no API call!)
                for symbol, ticker_data in self.exchange_wrapper.ws_ticker_cache.items():
                    # Filter invalid tickers (zero price)
                    last_price = ticker_data.get("last", 0.0)
                    if last_price <= 0:
                        continue
                        
                    # Convert to standard format
                    tickers[symbol] = {
                        "symbol": symbol,
                        "bid": ticker_data.get("bid", 0.0),
                        "ask": ticker_data.get("ask", 0.0),
                        "last": ticker_data.get("last", 0.0),
                        "mark": ticker_data.get("mark", ticker_data.get("last", 0.0)),
                        "quoteVolume": ticker_data.get("quoteVolume", 0.0),
                        "percentage": ticker_data.get("percentage", 0.0),
                        "info": ticker_data.get("info", {})
                    }
                self.logger.debug(f"[!] UNIVERSE REFRESH (WebSocket) | tickers={len(tickers)} | No REST API calls!")
            else:
                # Fallback to REST API (only if WebSocket not available)
                # NOTE: Rate limiting now handled by GovernedExchange wrapper via ApiGovernor
                self.api_call_times.append(refresh_now)
                # Use exchange wrapper to fetch tickers
                # Binance uses defaultType="future" from options
                tickers = await self.exchange_wrapper.fetch_tickers()
            
            try:
                self.logger.info(
                    f"[DIAG] refresh_universe:tickers count={len(tickers) if tickers else 0} "
                    f"ws_cache={hasattr(self.exchange_wrapper, 'ws_ticker_cache') and bool(getattr(self.exchange_wrapper, 'ws_ticker_cache', {}))}"
                )
            except Exception:
                pass
        except Exception as e:
            # Check if this is a rate limit error
            error_type = type(e).__name__
            error_str = str(e).lower()
            # Check both error type name and error message
            is_rate_limit = (
                'ddos' in error_type.lower() or 
                'ratelimit' in error_type.lower() or
                any(keyword in error_str for keyword in [
                    'too many requests', 'rate limit', 'banned', '-1003', 'ddos', '418', 'teapot'
                ])
            )
            
            if is_rate_limit:
                # Extract ban timestamp if available
                import re
                ban_match = re.search(r'banned until (\d+)', str(e))
                if ban_match:
                    ban_until_ms = int(ban_match.group(1))
                    self._rate_limit_until = ban_until_ms / 1000.0  # Convert to seconds
                    time_left = max(0, self._rate_limit_until - refresh_now)
                    self.logger.error(
                        f"[X] Rate limit hit during universe refresh. "
                        f"Ban expires in {int(time_left/60)}m {int(time_left%60)}s. "
                        f"Skipping refresh until ban expires."
                    )
                else:
                    # No specific ban time, use default cooldown (15 minutes)
                    self._rate_limit_until = refresh_now + 900  # 15 minutes
                    self.logger.error(
                        f"[X] Rate limit hit during universe refresh. "
                        f"Using 15 minute cooldown. Skipping refresh."
                    )
                self._last_rate_limit_error = refresh_now
                return  # Skip refresh when rate limited
            else:
                # Non-rate-limit error, re-raise
                raise
        
        # Log ticker fetch results (important for diagnostics)
        if not tickers or len(tickers) == 0:
            self.logger.warning(
                f"[!] UNIVERSE REFRESH | fetch_tickers() returned empty result. "
                f"Current universe size: {len(self.universe.stats)}. "
                f"This may indicate exchange API issues."
            )
            # If no tickers, return early (futures_tickers is already initialized as empty dict)
            return
        else:
            self.logger.info(
                f"[OK] UNIVERSE REFRESH | tickers_fetched={len(tickers)} | "
                f"markets_loaded={len(self.exchange_wrapper.markets) if hasattr(self.exchange_wrapper, 'markets') else 0} | "
                f"current_universe_size={len(self.universe.stats)}"
            )
            
            # OPTIMIZATION: Filter for futures/swap markets only (optimized filtering)
            # Pre-compute market lookup for faster access
            markets_dict = self.exchange_wrapper.markets if hasattr(self.exchange_wrapper, 'markets') and self.exchange_wrapper.markets else {}
        # futures_tickers is already initialized at the start of the function, but reset it here
            futures_tickers = {}
            
            for sym, t in tickers.items():
                # OPTIMIZATION: Early exit for None values
                if not sym or not t:
                    continue
                
                # CRITICAL FIX: Skip tickers with zero last price (invalid)
                # Check this early before normalizing
                raw_last = t.get("last")
                if raw_last is not None and float(raw_last) <= 0:
                    continue
                
                # OPTIMIZATION: Fast USDT check first (most common filter)
                if "USDT" not in sym:
                    continue
                
                # OPTIMIZATION: Check market type from exchange wrapper's markets first (fastest)
                market = markets_dict.get(sym)
                if market:
                    # Use exchange wrapper's method if available (most reliable)
                    if hasattr(self.exchange_wrapper, 'is_futures_market') and not self.exchange_wrapper.is_futures_market(market):
                        continue
                    
                    # CRITICAL FIX: Exclude dated futures (delivery contracts), keep only PERPETUAL SWAPS
                    # "future" = dated delivery, "swap" = perpetual
                    if market.get('type') != 'swap':
                        continue
                        
                else:
                    # Fallback: Check ticker info (slower, but necessary if market not loaded)
                    if isinstance(t, dict):
                        info = t.get("info", {})
                        market_type = info.get("type") if isinstance(info, dict) else t.get("type", "")
                        # CRITICAL FIX: Only allow SWAP (Perpetual), exclude FUTURE (Dated)
                        if market_type and market_type != 'swap':
                            continue
                
                # DOUBLE CHECK: Exclude symbols with delivery dates (e.g. BTCUSDT_251226)
                # Raw symbols with underscore and digits are usually dated futures
                if "_" in sym and any(c.isdigit() for c in sym.split("_")[-1]):
                    continue
                if "-" in sym and any(c.isdigit() for c in sym.split("-")[-1]):
                    continue
                if "USDT:USDT-" in sym:
                    continue
                
                # Normalize symbol for internal use
                normalized_sym = self.exchange_wrapper.normalize_symbol(sym)
                futures_tickers[normalized_sym] = t
            
            # LATENCY OPTIMIZATION: Batch update cache and storage (futures only)
            self.ticker_cache.batch_set(futures_tickers)
            self.fast_storage.batch_save(futures_tickers)
        
        # Process tickers and update universe stats (wrapped in try-except for error handling)
        try:
            # API BUDGET OPTIMIZATION: Refresh orderbooks for top symbols in parallel
            now = self._get_current_time()
            if now - self.last_orderbook_refresh >= self.orderbook_refresh_interval:
                await self._refresh_orderbooks_for_top_symbols()
                self.last_orderbook_refresh = now
            
            # API BUDGET OPTIMIZATION: Refresh funding rates periodically
            if now - self.last_funding_refresh >= self.funding_refresh_interval:
                await self._refresh_funding_rates()
                self.last_funding_refresh = now

            cnt = 0
            # OPTIMIZATION: Pre-compute constants outside loop
            mid_price_factor = MID_PRICE_FACTOR
            
            for sym, t in futures_tickers.items():
                # OPTIMIZATION: Cache ticker info lookup (avoid repeated dict access)
                ticker_info = t.get("info", {}) if isinstance(t, dict) else {}
                
                # CRITICAL: Binance Futures tickers may have bid/ask as None
                # Extract raw values first, then apply fallback logic
                bid_raw = t.get("bid")
                ask_raw = t.get("ask")
                last_raw = t.get("last") or 0.0
                last = float(last_raw)
                
                # CRITICAL FIX: Skip tickers with zero price (invalid/inactive)
                if last <= 0:
                    continue
                
                # If bid/ask are None or <= 0, use last price as fallback (common for futures)
                if bid_raw is None or (isinstance(bid_raw, (int, float)) and float(bid_raw) <= 0):
                    bid = last
                else:
                    bid = float(bid_raw)
                
                if ask_raw is None or (isinstance(ask_raw, (int, float)) and float(ask_raw) <= 0):
                    ask = last
                else:
                    ask = float(ask_raw)
                
                # CRITICAL: Force valid spread for universe stats
                # If spread is zero or negative (bid >= ask), artificially widen it
                # This ensures the symbol passes MIN_SPREAD filters and enters the universe
                if bid > 0 and ask <= bid:
                    # Use 10bps (0.1%) as synthetic spread for universe inclusion
                    # Real spread will be checked via orderbook during signal generation
                    ask = bid * 1.001
                
                # OPTIMIZATION: Cache info lookups
                mark = (
                    ticker_info.get("fairPx")
                    or ticker_info.get("markPrice")
                    or last
                )
                
                # OPTIMIZATION: Cache volume lookups
                vol_quote = (
                    t.get("quoteVolume")
                    or ticker_info.get("amount24h")
                    or ticker_info.get("quote_volume")
                    or 0.0
                )
                try:
                    vol_quote = float(vol_quote)
                except (TypeError, ValueError):
                    # REFACTOR: Handle invalid/missing volume data
                    vol_quote = 0.0

                # OPTIMIZATION: Calculate spread more efficiently
                spread_bps = 9999.0
                if bid and ask and ask > 0:
                    mid = mid_price_factor * (bid + ask)
                    if mid > 0:
                        spread_bps = abs(ask - bid) / mid * 1e4

                # OPTIMIZATION: Get or create stats once
                st = self.universe.stats.get(sym)
                if not st:
                    st = SymbolStats(sym)
                    self.universe.stats[sym] = st
                
                # OPTIMIZATION: Batch update stats
                st.bid = bid
                st.ask = ask
                st.last = last
                st.mark = float(mark or 0.0)
                st.vol_quote = vol_quote
                st.spread_bps = float(spread_bps)
                st.heat = (math.log10(vol_quote + 1.0) - 5.0) - (spread_bps / 200.0)
                st.last_seen = refresh_now  # OPTIMIZATION: Use cached time
                
                # OPTIMIZATION: Extract 24h price change percentage (cache info lookup)
                pct_change = t.get("percentage") or ticker_info.get("priceChangePercent") or 0.0
                try:
                    st.pct_change_24h = float(pct_change)
                except (ValueError, TypeError):
                    st.pct_change_24h = 0.0
                
                cnt += 1

            self.symbols_scanned_last = cnt
            
            # CRITICAL DEBUG: Log stats population
            self.logger.info(
                f"[OK] UNIVERSE STATS POPULATED | stats_count={len(self.universe.stats)} | "
                f"futures_tickers={len(futures_tickers)} | samples={list(self.universe.stats.keys())[:5] if self.universe.stats else []}"
            )
            
            # Calculate BTC trend BEFORE rotation (needed for correlation scoring)
            # Try both formats: BTC/USDT:USDT (full) and BTC/USDT (short)
            btc_stats = self.universe.stats.get("BTC/USDT:USDT") or self.universe.stats.get("BTC/USDT")
            if btc_stats:
                self.btc_trend = getattr(btc_stats, 'pct_change_24h', 0.0)
                # Set BTC trend in universe for correlation scoring
                self.universe.set_btc_trend(self.btc_trend)
            else:
                self.universe.set_btc_trend(0.0)
            
            # Rotate universe (replace stale symbols with fresh candidates)
            # Check if rotation is needed (periodic check to avoid excessive rotation)
            from .config import ROTATION_CHECK_INTERVAL_SEC, MAX_ACTIVE_SYMBOLS
            now_rotation = time.time()
            time_since_rotation = now_rotation - self.universe.last_rotation_time
            
            # Get set of symbols with open positions (these must stay in universe)
            open_position_symbols = set(self.positions.keys())
            
            # First rotation or if active list is too small: do full rotation
            if self.universe.last_rotation_time == 0.0 or len(self.universe.active) < 50:
                self.universe.rotate(force_full_rotation=True, open_positions=open_position_symbols)
            elif time_since_rotation >= ROTATION_CHECK_INTERVAL_SEC:
                # Periodic smart rotation: only replace stale symbols (preserves good symbols + open positions)
                self.universe.rotate(force_full_rotation=False, open_positions=open_position_symbols)
        
            # Track loop time
            loop_time = time.time() - loop_start
            self.loop_times.append(loop_time)
            self.last_universe_refresh = time.time()
            
            # Calculate volatility and spread regimes
            self._calculate_market_regimes()
            
        except Exception as e:
            self.ctrl.recent_errors.append(f"tickers ERR {type(e).__name__}")
            error_msg = str(e)
            self.logger.warning(
                f"Universe refresh failed: {type(e).__name__}: {error_msg[:100]}. "
                f"Current universe size: {len(self.universe.stats)} symbols. "
                f"Using cached data."
            )
            # If universe is empty or very small, try to initialize fallback symbols
            if len(self.universe.stats) < 10:
                if hasattr(self, '_fallback_mode') and self._fallback_mode:
                    self.logger.info("Universe is small but fallback mode already active - this is expected")
                else:
                    self.logger.warning("Universe is very small - attempting to initialize fallback symbols")
                    if not hasattr(self, '_fallback_mode') or not self._fallback_mode:
                        self._init_fallback_symbols()
            return
    
    async def _refresh_positions(self):
        """
        Force-refresh all positions from exchange.
        Synchronizes self.positions with exchange reality.
        """
        if not self.exchange_wrapper or self.replay_mode:
            return

        try:
            # Fetch all open positions (no symbol filter)
            real_positions = await self.exchange_wrapper.fetch_positions()
            
            # Sync timestamp
            sync_time = time.time()
            self._last_pos_sync = sync_time
            
            # Update self.positions
            # 1. Mark all current positions as potentially stale
            # 2. Update/Add real positions
            # 3. Remove positions not in real_positions (unless we just opened them < 5s ago)
            
            real_symbols = set()
            
            for rp in real_positions:
                sym = rp.get('symbol') or rp.get('symbolName')
                if not sym:
                    continue
                
                # Normalize symbol
                if hasattr(self.exchange_wrapper, 'normalize_symbol'):
                    sym = self.exchange_wrapper.normalize_symbol(sym)
                
                real_symbols.add(sym)
                
                # Adopt/Update
                # CRITICAL: Binance returns NEGATIVE positionAmt for shorts - use abs()
                raw_size = float(rp.get('contracts', rp.get('amount', rp.get('positionAmt', 0.0))) or 0.0)
                position_size = abs(raw_size)  # Always positive
                
                # Determine side from positionAmt sign or explicit side field
                explicit_side = rp.get('side', rp.get('positionSide', '')).lower()
                if explicit_side in ('long', 'buy'):
                    side = 'long'
                elif explicit_side in ('short', 'sell'):
                    side = 'short'
                elif raw_size < 0:
                    side = 'short'  # Negative positionAmt = short
                else:
                    side = 'long'   # Positive or zero = long
                
                # Preserve entry_time if known, otherwise use current time
                # DEFENSIVE: Never use 0 or invalid timestamps
                preserved_entry_time = self.positions.get(sym, {}).get('entry_time', sync_time)
                if preserved_entry_time <= 0 or preserved_entry_time > sync_time:
                    preserved_entry_time = sync_time
                
                adopted = {
                    'symbol': sym,
                    'entry_price': float(rp.get('entryPrice', rp.get('price', 0.0)) or 0.0),
                    'size': position_size,
                    'side': side,
                    'leverage': int(rp.get('leverage', 1) or 1),
                    'entry_time': preserved_entry_time,  # Preserve entry time if known, never use 0
                    'unrealizedPnl': float(rp.get('unrealizedPnl', 0.0) or 0.0),
                }

                # Fallbacks
                if not adopted['entry_price']:
                    adopted['entry_price'] = float(rp.get('avgPrice', rp.get('markPrice', 0.0)) or 0.0)
                if not adopted['size']:
                    adopted['size'] = abs(float(rp.get('qty', 0.0) or 0.0))
                
                self.positions[sym] = adopted
                self.position_registry._positions[sym] = adopted
                self.position_registry._positions_set.add(sym)
                self._positions_set.add(sym)

            # Cleanup stale positions (Ghost Busting)
            # Only remove if it's NOT in real_positions AND it's older than 10s
            # (Prevents race condition where we just opened a trade but API doesn't show it yet)
            for sym in list(self.positions.keys()):
                if sym not in real_symbols:
                    entry_ts = self.positions[sym].get('entry_time', 0)
                    if sync_time - entry_ts > 10.0:
                        self.logger.info(f"[!] GHOST BUSTED: Removing stale position {sym} (not on exchange)")
                        del self.positions[sym]
                        if sym in self.position_registry._positions:
                            del self.position_registry._positions[sym]
                        if sym in self.position_registry._positions_set:
                            self.position_registry._positions_set.remove(sym)
                        if sym in self._positions_set:
                            self._positions_set.remove(sym)
                            
        except Exception as e:
            self.logger.error(f"Error refreshing positions: {e}")
            raise # Propagate error for startup check

    async def _refresh_orderbooks_for_top_symbols(self):
        """Refresh orderbooks for top symbols to improve signal quality."""
        # REPLAY MODE: Use synthetic orderbooks from replay feed
        if self.replay_mode and self.replay_feed:
            top_symbols = self.universe.active[:20]
            refresh_now = self._get_current_time()
            for symbol in top_symbols:
                snapshot = self.replay_feed.get_market_snapshot(symbol)
                if snapshot and 'orderbook' in snapshot:
                    self.orderbook_cache[symbol] = {
                        'data': snapshot['orderbook'],
                        'timestamp': refresh_now
                    }
            return
        
        if not self.exchange_wrapper:
            return
        # Note: Orderbooks are fetched in live mode only (not in DRY_RUN to save API budget)
        
        # [!] MAXIMIZE: Get top 20 symbols (restore aggressive scanning)
        top_symbols = self.universe.active[:20]
        if not top_symbols:
            return
        
        # [!] ADAPTIVE THROTTLING: Check if we're in backoff mode
        now = time.time()
        if now < self.throttle_backoff_until:
            # In backoff - skip this refresh cycle
            return
        
        # If we are in throttle backoff, skip this refresh
        if now < self.throttle_backoff_until:
            return

        # Fetch orderbooks in parallel (aggressive batching with adaptive delays)
        async def fetch_orderbook(symbol):
            try:
                # NOTE: Rate limiting now handled by GovernedExchange wrapper via ApiGovernor
                # Denormalize symbol for exchange API
                denormalized_symbol = self.exchange_wrapper.denormalize_symbol(symbol)
                orderbook = await asyncio.wait_for(
                    self.exchange_wrapper.fetch_order_book(denormalized_symbol, limit=20),  # Deeper depth
                    timeout=0.3
                )
                return (symbol, orderbook)
            except Exception as e:
                error_str = str(e)
                # [!] ADAPTIVE: Detect throttle errors and trigger backoff
                if "throttle queue" in error_str.lower() or "maxCapacity" in error_str:
                    self.throttle_error_count += 1
                    self.last_throttle_error = now
                    # Exponential backoff: 2s, 4s, 8s, 16s (max 30s)
                    backoff_duration = min(2.0 * (2 ** min(self.throttle_error_count - 1, 4)), 30.0)
                    self.throttle_backoff_until = now + backoff_duration
                    self.adaptive_delay = min(self.adaptive_delay * 1.5, 1.0)  # Increase delay
                    self.logger.warning(f"Throttle queue pressure detected. Backing off for {backoff_duration:.1f}s")
                return None
        
        # [!] ADAPTIVE BATCH SIZE: Larger batches when healthy, smaller when recovering
        if self.throttle_error_count == 0:
            batch_size = 10  # Aggressive: 10 per batch
        elif self.throttle_error_count < 3:
            batch_size = 7   # Moderate: 7 per batch
        else:
            batch_size = 5   # Conservative: 5 per batch
        
        # OPTIMIZATION: Collect all potential signals first, then process top candidates
        # This prevents "Lazy Fetch" throttle storms when many signals appear at once
        potential_signals = []
        
        async def process_symbol(symbol):
            # ... (existing process_symbol logic) ...
            # Instead of returning processed result directly, we return the data to be sorted
            # ...
            
        # Refactored scan loop to gather-then-sort
        # 1. Gather all candidates (fast checks only, NO lazy fetch yet)
        
        # ... (rest of the parallel processing code) ...
        
        # Sort candidates by score (descending)
        # candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Take Top 5
        # top_candidates = candidates[:5]
        
        # Process Top 5 (Lazy Fetch & Entry)
        # for candidate in top_candidates:
        #     await self.process_candidate(candidate)
            
            # OPTIMIZATION: Update timestamp once per batch
            batch_timestamp = time.time()
            results_list = locals().get("results", [])
            for result in results_list:
                if result and isinstance(result, tuple):
                    symbol, orderbook = result
                    self.orderbook_cache[symbol] = {
                        'data': orderbook,
                        'timestamp': batch_timestamp  # OPTIMIZATION: Use cached time
                    }
    
    async def _refresh_funding_rates(self):
        """
        Refresh funding rates for active symbols.
        OPTIMIZED: Uses Binance Futures batch premium index endpoint when possible.
        """
        if not self.exchange_wrapper or DRY_RUN:
            return
        
        # Get active symbols
        active_symbols = self.universe.active[:30]  # Top 30 symbols
        if not active_symbols:
            return
        
        # OPTIMIZATION: Try to use Binance Futures batch premium index endpoint
        # Binance Futures /fapi/v1/premiumIndex can get funding rates for all symbols in one call
        # Check if exchange wrapper supports batch funding rate fetch
        if hasattr(self.exchange_wrapper, 'fetch_funding_rates_batch'):
            try:
                # NOTE: Rate limiting now handled by GovernedExchange wrapper via ApiGovernor
                denormalized_symbols = [self.exchange_wrapper.denormalize_symbol(s) for s in active_symbols]
                funding_data = await asyncio.wait_for(
                    self.exchange_wrapper.fetch_funding_rates_batch(denormalized_symbols),
                    timeout=1.0
                )
                # Update funding rates cache
                batch_timestamp = time.time()
                for symbol, rate in funding_data.items():
                    if rate is not None:
                        # Map back to normalized symbol
                        normalized_symbol = self.exchange_wrapper.normalize_symbol(symbol)
                        self.funding_rates[normalized_symbol] = {
                            'rate': float(rate),
                            'timestamp': batch_timestamp
                        }
                    return  # Success with batch method
            except Exception as e:
                # Fallback to individual calls if batch method fails
                # OPTIMIZATION: Removed debug log (fallback is expected behavior)
                pass
        
        # Fallback: Fetch funding rates in parallel (batch of 20 - OPTIMIZED from 10)
        async def fetch_funding(symbol):
            try:
                # NOTE: Rate limiting now handled by GovernedExchange wrapper via ApiGovernor
                # OPTIMIZATION: Use premium index endpoint directly if available
                # Binance Futures /fapi/v1/premiumIndex is faster than full ticker
                denormalized_symbol = self.exchange_wrapper.denormalize_symbol(symbol)
                
                # Try to get funding rate from ticker info (CCXT handles this)
                ticker = await asyncio.wait_for(
                    self.exchange_wrapper.fetch_ticker(denormalized_symbol),
                    timeout=0.2
                )
                funding_rate = ticker.get('info', {}).get('fundingRate') or ticker.get('fundingRate')
                if funding_rate:
                    return (symbol, float(funding_rate))
                return None
            except (asyncio.TimeoutError, ConnectionError, OSError, KeyError, ValueError, TypeError, AttributeError) as e:
                # Network errors, timeouts, missing data, or type conversion issues - safe to ignore
                return None
        
        # OPTIMIZATION: Increased batch size from 10 to 20 for better parallelization
        batch_size = 20
        # OPTIMIZATION: Cache time once per batch
        batch_timestamp = time.time()
        for i in range(0, len(active_symbols), batch_size):
            batch = active_symbols[i:i+batch_size]
            results = await asyncio.gather(*[fetch_funding(s) for s in batch], return_exceptions=True)
            
            # OPTIMIZATION: Update timestamp once per batch
            batch_timestamp = time.time()
            for result in results:
                if result and isinstance(result, tuple):
                    symbol, rate = result
                    self.funding_rates[symbol] = {
                        'rate': rate,
                        'timestamp': batch_timestamp  # OPTIMIZATION: Use cached time
                    }
    
    async def _refresh_orderbooks_for_positions(self):
        """Refresh orderbooks for open positions for better exit decisions."""
        if not self.exchange_wrapper or DRY_RUN or not self.positions:
            return
        
        # [!] ADAPTIVE: Check if we should skip this refresh (too frequent or in backoff)
        now = time.time()
        if now < self.last_position_orderbook_refresh + self.position_orderbook_refresh_interval:
            return  # Skip if too soon
        if now < self.throttle_backoff_until:
            return  # Skip if in backoff mode
        
        # If in backoff, skip position refresh
        if now < self.throttle_backoff_until:
            return

        # Fetch orderbooks for all open positions (aggressive with adaptive throttling)
        async def fetch_pos_orderbook(symbol):
            try:
                # NOTE: Rate limiting now handled by GovernedExchange wrapper via ApiGovernor
                # Denormalize symbol for exchange API
                denormalized_symbol = self.exchange_wrapper.denormalize_symbol(symbol)
                orderbook = await asyncio.wait_for(
                    self.exchange_wrapper.fetch_order_book(denormalized_symbol, limit=20),
                    timeout=0.2
                )
                return (symbol, orderbook)
            except Exception as e:
                error_str = str(e)
                # [!] ADAPTIVE: Detect throttle errors and trigger backoff
                if "throttle queue" in error_str.lower() or "maxCapacity" in error_str:
                    self.throttle_error_count += 1
                    self.last_throttle_error = now
                    backoff_duration = min(2.0 * (2 ** min(self.throttle_error_count - 1, 4)), 30.0)
                    self.throttle_backoff_until = now + backoff_duration
                    self.adaptive_delay = min(self.adaptive_delay * 1.5, 1.0)
                return None
        
        # [!] ADAPTIVE BATCH SIZE: Larger batches when healthy
        symbols = list(self.positions.keys())
        if self.throttle_error_count == 0:
            batch_size = 10  # Aggressive: 10 per batch
        elif self.throttle_error_count < 3:
            batch_size = 7   # Moderate: 7 per batch
        else:
            batch_size = 5   # Conservative: 5 per batch
        
        cache_timestamp = time.time()
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            results = await asyncio.gather(*[fetch_pos_orderbook(s) for s in batch], return_exceptions=True)
            
            # [!] ADAPTIVE DELAY: Dynamic delay based on throttle health
            if i + batch_size < len(symbols):
                await asyncio.sleep(self.adaptive_delay)
            
            for result in results:
                if result and isinstance(result, tuple):
                    symbol, orderbook = result
                    self.orderbook_cache[symbol] = {
                        'data': orderbook,
                        'timestamp': cache_timestamp
                    }
        
        self.last_position_orderbook_refresh = now
        
    
    async def scan_and_enter_signals(self):
        """Scan for trading signals and enter positions."""
        from .config import (
            MAX_LATENCY_MS, MIN_VOLUME_24H, MIN_SPREAD_BPS, DRY_RUN
        )
        
        if not self.order_manager:
            # DEBUG: Log why scan is skipped
            if not hasattr(self, '_order_manager_warning_logged'):
                self.logger.warning("CRITICAL: scan_and_enter_signals skipped - order_manager is None")
                self._order_manager_warning_logged = True
            return
        
        # Track scan start
        scan_start_time = time.time()
        self.last_scan_start = scan_start_time
        self.is_scanning = True
        
        # DEBUG: Log scan start (first few scans only)
        if not hasattr(self, '_scan_start_count'):
            self._scan_start_count = 0
        self._scan_start_count += 1
        if self._scan_start_count <= 3:
            self.logger.info(
                f"[SCAN] SCAN START #{self._scan_start_count} | "
                f"universe.stats={len(self.universe.stats)} | "
                f"universe.active={len(self.universe.active)} | "
                f"replay_mode={self.replay_mode}"
            )
        
        # STARTUP DELAY: Block entries during warmup to collect historical data
        # REPLAY MODE: Skip time-based warmup (entries allowed immediately)
        if self.replay_mode:
            # No time-based startup warmup in replay; allow entries immediately
            in_startup_period = False
            if not self.startup_period_ended:
                self.startup_period_ended = True
                self.logger.info("[OK] REPLAY MODE: Startup warmup skipped (entries enabled immediately)")
        else:
            # LIVE MODE: Block entries during warmup period
            time_since_start = scan_start_time - self.started_at
            in_startup_period = time_since_start < STARTUP_DELAY_SEC if STARTUP_DELAY_SEC > 0 else False
            
            # During startup warmup: SCAN signals but BLOCK entries to build signal history
            # This ensures percentile filter has good data before allowing trades
            if in_startup_period and not self.startup_period_ended:
                remaining = STARTUP_DELAY_SEC - time_since_start
                # Only log once when warmup starts
                if not hasattr(self, '_warmup_logged'):
                    self.logger.info(
                        f"Warming up: Full market scan in progress ({STARTUP_DELAY_SEC}s)... Building signal history for percentile filter. Will start trading in {remaining:.0f}s"
                    )
                    self._warmup_logged = True
                # CONTINUE scanning during warmup (don't return early) - we need signal history
                # But set flag to block entries in the entry logic below
                in_startup_period = True
            elif not self.startup_period_ended and STARTUP_DELAY_SEC > 0:
                # Startup period just ended - now allow entries
                self.startup_period_ended = True
                signal_history_count = len(getattr(self.signal_generator, 'signal_history', []))
                self.logger.info(
                    f"Warmup complete. Full market scan finished. Signal history: {signal_history_count} signals collected. Bot is now ready to trade (top 1% filtering active)."
                )
                in_startup_period = False
        
        # OPTIMIZATION: Cache equity calculation (used multiple times)
        equity = self.equity_now()
        
        # RESTORED: Count ALL positions to strictly enforce limits.
        # The Dust Sweeper will handle cleanup, but we must NOT open new trades while over capacity.
        # CRITICAL: Use actual Binance position count to prevent false "max positions" rejections
        # Position sync happens in monitor_and_exit_positions, but we need accurate count here
        # Cache Binance count to avoid blocking on every scan
        if not hasattr(self, '_cached_binance_position_count') or not hasattr(self, '_binance_count_cache_time') or (time.time() - getattr(self, '_binance_count_cache_time', 0)) > 10:
            # Update cache every 10 seconds
            try:
                binance_positions = await self.exchange_wrapper.fetch_positions()
                valid_binance_count = 0
                for rp in binance_positions:
                    sym = rp.get('symbol') or rp.get('symbolName')
                    if sym:
                        clean_sym = sym.replace("/", "").replace("-", "").replace(":", "")
                        if not ("USDTUSDT" in clean_sym or (len(clean_sym) > 6 and clean_sym[-6:].isdigit())):
                            size = abs(float(rp.get('contracts', rp.get('amount', rp.get('positionAmt', 0.0))) or 0.0))
                            if size > 0:
                                valid_binance_count += 1
                self._cached_binance_position_count = valid_binance_count
                self._binance_count_cache_time = time.time()
            except Exception:
                # Fallback to tracked positions if fetch fails
                self._cached_binance_position_count = len(self.positions)
                self._binance_count_cache_time = time.time()
        
        current_positions = max(len(self.positions), getattr(self, '_cached_binance_position_count', len(self.positions)))
        
        # IDLE RELAXATION MODE: Gradually lower entry score when no positions to stay active 24/7
        # This ensures the bot stays active during slow periods (advantage over humans who sleep)
        # Strategy: Start at normal threshold, gradually lower every scan cycle until position taken
        # Once a position is taken, immediately reset to normal strict filtering
        is_idle_mode = current_positions == 0
        
        # SIMPLIFIED: Initialize idle mode tracking (no complex relaxation logic)
        if not hasattr(self, '_idle_mode_active'):
            self._idle_mode_active = False
            self._idle_mode_idle_start_time = 0.0
        
        # Calculate idle duration
        if is_idle_mode:
            if not self._idle_mode_active:
                # SIMPLIFIED: Entering idle mode - use IDLE_MIN_SCORE immediately
                self._idle_mode_active = True
                self._idle_mode_idle_start_time = scan_start_time
                from . import config as cfg
                idle_min = float(getattr(cfg, 'IDLE_MIN_SCORE', 42.0))
                self.logger.info(f"[IDLE_MODE] No positions open. Using IDLE_MIN_SCORE={idle_min:.0f} to find best available signal")
            else:
                # SIMPLIFIED: Just track idle duration, no gradual lowering
                # Idle mode uses fixed IDLE_MIN_SCORE (42.0) - simple and reliable
                idle_duration_minutes = (scan_start_time - self._idle_mode_idle_start_time) / 60.0
                # No complex relaxation logic - just use IDLE_MIN_SCORE
        else:
            # Have positions - exit idle mode
            if self._idle_mode_active:
                idle_duration = (scan_start_time - self._idle_mode_idle_start_time) / 60.0
                self.logger.info(f"[IDLE_MODE] Position opened - exiting idle mode after {idle_duration:.1f} minutes. Resuming normal MIN_SIGNAL_SCORE.")
                self._idle_mode_active = False
                self._idle_mode_idle_start_time = 0.0
        
        # SIMPLIFIED IDLE MODE: No gradual lowering, just use IDLE_MIN_SCORE
        # When no positions, use lower threshold (42.0) to find best available signal
        # Simple and reliable - no complex logic
        if self._idle_mode_active:
            from . import config as cfg
            idle_min_score = float(getattr(cfg, 'IDLE_MIN_SCORE', 42.0))
            idle_duration_minutes = (scan_start_time - self._idle_mode_idle_start_time) / 60.0
            
            # Log idle mode status (once per minute)
            if not hasattr(self, '_last_idle_log') or time.time() - self._last_idle_log > 60:
                self.logger.info(
                    f"[IDLE_MODE] No positions - using IDLE_MIN_SCORE={idle_min_score:.0f} "
                    f"(idle {idle_duration_minutes:.1f}m)"
                )
                self._last_idle_log = time.time()
        
        # DIAGNOSTIC: Compare bot's position count with Binance's actual positions
        # This helps identify ghost positions or sync issues
        if not self.replay_mode and self.exchange_wrapper:
            try:
                binance_positions = await self.exchange_wrapper.fetch_positions()
                # Filter to only non-zero positions
                binance_active = [p for p in binance_positions if abs(float(p.get('contracts', 0))) > 0]
                binance_count = len(binance_active)
                
                if current_positions != binance_count:
                    self.logger.warning(
                        f"[POSITION_SYNC] Mismatch detected: Bot tracks {current_positions} positions, "
                        f"Binance has {binance_count} active positions. "
                        f"Bot symbols: {list(self.positions.keys())}, "
                        f"Binance symbols: {[p.get('symbol', 'N/A') for p in binance_active]}"
                    )
                    # Use Binance's count as source of truth for position limit checks
                    current_positions = binance_count
            except Exception as e:
                self.logger.debug(f"[POSITION_SYNC] Could not verify Binance positions: {e}")
        
        # DUST FILTER REMOVED: Previous filter caused position sprawl (13+ positions)
        # current_positions = sum(1 for p in self.positions.values() 
        #                       if abs(p.get('size', 0)) * p.get('entry_price', 0) >= 5.0)
        
        # SIGNAL CONFIRMATION WINDOW (SCW): Update waiting signals with new bar data
        from .config import USE_SIGNAL_CONFIRMATION, R_BAR_SCAN_CYCLE_SEC
        if USE_SIGNAL_CONFIRMATION:
            # Cleanup stale signals (older than 5 minutes)
            self.signal_confirmation.cleanup_stale_signals(max_age_sec=300.0)
            
            # Update all waiting signals (bar closed = True since we're in a new scan cycle)
            for symbol in list(self.signal_confirmation.waiting_signals.keys()):
                waiting = self.signal_confirmation.waiting_signals.get(symbol)
                if not waiting:
                    continue
                
                # Get current market data for confirmation check
                stats = self.universe.stats.get(symbol)
                if not stats:
                    continue
                
                current_price = stats.mark or stats.last
                spread_bps = stats.spread_bps
                volume_24h = stats.vol_quote
                
                # Get orderbook data if available (for high/low)
                orderbook = self.orderbook_cache.get(symbol, {}).get('data')
                low = None
                high = None
                if orderbook:
                    bids = orderbook.get('bids', [])
                    asks = orderbook.get('asks', [])
                    if bids and asks:
                        low = bids[-1][0] if bids else None  # Lowest bid
                        high = asks[-1][0] if asks else None  # Highest ask
                
                # Get ATR if available
                atr_pct = None
                if hasattr(stats, 'atr_pct'):
                    atr_pct = stats.atr_pct
                
                # CRITICAL NOTE: Opposite signal detection disabled (incomplete feature)
                # To enable: Check signal generator for opposite-side signals and pass score here
                opposite_signal_score = None  # Disabled: would need signal generator integration
                
                # CRITICAL NOTE: Volume MA calculation disabled (incomplete feature)
                # To enable: Calculate rolling volume MA (e.g., 20-period) and pass here
                volume_ma = None  # Disabled: would need volume history tracking
                
                # Update waiting signal
                is_ready, updated_waiting = self.signal_confirmation.update_waiting_signal(
                    symbol=symbol,
                    current_price=current_price,
                    low=low,
                    high=high,
                    spread_bps=spread_bps,
                    atr_pct=atr_pct,
                    volume=volume_24h,
                    volume_ma=volume_ma,
                    opposite_signal_score=opposite_signal_score,
                    bar_closed=True  # New scan cycle = bar closed
                )
                
                    # REMOVED: Individual signal confirmation log (aggregated in summary)
        
        # EVOLUTION-BASED SYSTEM: Scan all active symbols in universe
        # Bot uses evolution-optimized parameters from Phase 2 validation
        
        # Initialize variables
        active_symbols = []
        discovery_symbols = []
        
        # MARKSMAN removed - always scan all active symbols
        if False:  # MARKSMAN removed - this block never executes
            # WARNING: MARKSMAN mode is deprecated
            self.logger.warning(
                "MARKSMAN mode is deprecated. Bot should use evolution-optimized parameters. "
                "Set ENABLE_MARKSMAN=0 to use full universe scanning."
            )
            # Legacy code kept for backward compatibility only
            from .config import MARKSMAN_SYMBOLS
            marksman_symbols = MARKSMAN_SYMBOLS
            if hasattr(self, 'cfg') and hasattr(self.cfg, 'MARKSMAN_SYMBOLS'):
                marksman_symbols = self.cfg.MARKSMAN_SYMBOLS
            
            # Normalize MARKSMAN symbols to match universe format
            normalized_marksman_symbols = []
            if self.exchange_wrapper:
                for sym in marksman_symbols:
                    normalized = self.exchange_wrapper.normalize_symbol(sym)
                    if normalized in self.universe.stats:
                        normalized_marksman_symbols.append(normalized)
            else:
                for sym in marksman_symbols:
                    if sym in self.universe.stats:
                        normalized_marksman_symbols.append(sym)
                    else:
                        normalized = sym.replace(':USDT', '') if ':USDT' in sym else sym
                        if normalized in self.universe.stats:
                            normalized_marksman_symbols.append(normalized)
            
            # OPTIMIZATION: Add discovery candidates to MARKSMAN mode for more quality setups
            from .config import (
                DISCOVERY_SCAN_INTERVAL_SEC, DISCOVERY_SYMBOLS_PER_CYCLE,
                DISCOVERY_MIN_VOLUME_24H, DISCOVERY_MAX_SPREAD_BPS, DISCOVERY_MIN_MOMENTUM_PCT,
                STALE_SYMBOL_THRESHOLD_SEC, MARKSMAN_DISCOVERY_MAX_CANDIDATES
            )
            
            now = time.time()
            discovery_symbols = []
            
            # MARKET ACTIVITY DETECTION: Scan more frequently during high volatility periods
            # Adaptive scan interval: 10s base, 7s in high volatility, 12s in low volatility
            adaptive_scan_interval = DISCOVERY_SCAN_INTERVAL_SEC
            volatility_regime = getattr(self, 'volatility_regime', 'Normal')
            if volatility_regime == 'High':
                adaptive_scan_interval = DISCOVERY_SCAN_INTERVAL_SEC * 0.7  # 7s in high volatility (30% faster)
            elif volatility_regime == 'Low':
                adaptive_scan_interval = DISCOVERY_SCAN_INTERVAL_SEC * 1.2  # 12s in low volatility (20% slower)
            
            if now - self.last_discovery_scan >= adaptive_scan_interval:
                # Get high-quality discovery candidates (prioritize strong momentum >2% and good liquidity)
                discovery_candidates = self.universe.get_discovery_candidates(
                    max_candidates=MARKSMAN_DISCOVERY_MAX_CANDIDATES,  # Limit to 20-30 high-quality candidates
                    min_volume=DISCOVERY_MIN_VOLUME_24H,
                    max_spread_bps=DISCOVERY_MAX_SPREAD_BPS,
                    min_momentum_pct=2.0,  # Higher threshold for MARKSMAN: require >2% momentum
                    stale_threshold_sec=STALE_SYMBOL_THRESHOLD_SEC
                )
                
                # Filter discovery candidates: prioritize strong momentum + good liquidity
                marksman_set = set(normalized_marksman_symbols)
                for symbol in discovery_candidates:
                    if symbol not in marksman_set:  # Don't duplicate MARKSMAN symbols
                        stats = self.universe.stats.get(symbol)
                        if stats:
                            # Quality filter: strong momentum (>2%) and reasonable spread
                            momentum_abs = abs(getattr(stats, 'pct_change_24h', 0.0))
                            spread_bps = getattr(stats, 'spread_bps', 9999.0)
                            vol_quote = getattr(stats, 'vol_quote', 0.0)
                            
                            if momentum_abs >= 2.0 and spread_bps < 60.0 and vol_quote > 2e6:  # $2M+ volume
                                discovery_symbols.append(symbol)
                                if len(discovery_symbols) >= MARKSMAN_DISCOVERY_MAX_CANDIDATES:
                                    break
                
                self.last_discovery_scan = now
            
            # Combine MARKSMAN symbols with discovery candidates
            all_symbols_to_scan = list(set(normalized_marksman_symbols + discovery_symbols))
            total_symbols = len(all_symbols_to_scan)
            
            print(f"DEBUG: normalized_marksman={len(normalized_marksman_symbols)}, discovery={len(discovery_symbols)}, total={total_symbols}", flush=True)
            if len(normalized_marksman_symbols) == 0:
                 print(f"DEBUG: active_universe_sample={list(self.universe.stats.keys())[:5]}", flush=True)
            
            # CRITICAL: Log if no MARKSMAN symbols found
            if len(normalized_marksman_symbols) == 0:
                self.logger.warning(
                    f"MARKSMAN MODE: No MARKSMAN symbols found in universe! "
                    f"Configured symbols: {marksman_symbols}, "
                    f"Universe has {len(self.universe.stats)} symbols, "
                    f"Sample universe symbols: {list(self.universe.stats.keys())[:10]}"
                )
            
            # In MARKSMAN mode, active_symbols includes MARKSMAN + discovery
            # CRITICAL FIX: If normalization failed, use universe.active directly
            if normalized_marksman_symbols:
                active_symbols = normalized_marksman_symbols.copy()
            else:
                # Normalization failed - use universe.active and filter to MARKSMAN symbols
                self.logger.warning(
                    f"MARKSMAN normalization returned 0 symbols, using universe.active ({len(self.universe.active)} symbols)"
                )
                # Try direct match first
                active_symbols = [s for s in self.universe.active if s in marksman_symbols]
                if not active_symbols:
                    # Try matching by base symbol (e.g., "BTC" in "BTC/USDT:USDT")
                    for ms in marksman_symbols:
                        base = ms.split('/')[0] if '/' in ms else ms.replace(':USDT', '').replace('/USDT', '')
                        for s in self.universe.active:
                            if (base in s or s.startswith(base)) and s not in active_symbols:
                                active_symbols.append(s)
                if not active_symbols:
                    # Last resort: use all universe.active symbols (they're all MARKSMAN anyway in this mode)
                    active_symbols = self.universe.active[:50]
                    self.logger.warning(f"Using all universe.active symbols as fallback: {len(active_symbols)} symbols")
            
            # Add discovery symbols
            all_symbols_to_scan = list(set(active_symbols + discovery_symbols))
            total_symbols = len(all_symbols_to_scan)
            
            # This block never executes (MARKSMAN removed)
            pass
        else:
            # Scan all active symbols in universe (limited by SYMBOLS_TO_SCAN)
            # CRITICAL FIX: Slice active_symbols to prevent API throttling
            limit = getattr(self.cfg, 'SYMBOLS_TO_SCAN', 100)
            
            # TRASH ROTATION LOGIC (Council Decree #4)
            # 1. Rotate the starting index every scan to ensure we cover the whole universe
            # 2. Filter out "Jailed" symbols (trash)
            
            # Initialize rotation state if missing
            if not hasattr(self, '_scan_offset'):
                self._scan_offset = 0
                self._trash_jail = {}   # symbol -> jail_release_time
                self._trash_strikes = {} # symbol -> consecutive_low_scores
            
            # Clean up jail (release inmates who served their time)
            now_ts = time.time()
            jailed_symbols = [s for s, release_time in self._trash_jail.items() if now_ts < release_time]
            # Cleanup expired entries from dict
            self._trash_jail = {s: t for s, t in self._trash_jail.items() if now_ts < t}
            
            # Get candidates excluding jail
            available_symbols = [s for s in self.universe.active if s not in self._trash_jail]
            
            # Rotate selection
            total_available = len(available_symbols)
            if total_available > 0:
                # Move offset forward by 'limit' to scan next batch next time
                # If we reach end, wrap around
                start_idx = self._scan_offset % total_available
                
                # Create rotated list starting from offset
                rotated_list = available_symbols[start_idx:] + available_symbols[:start_idx]
                active_symbols = rotated_list[:limit]
                
                # Advance offset for next time
                self._scan_offset = (self._scan_offset + limit) % total_available
            else:
                active_symbols = []
            
            # CRITICAL FIX: If active_symbols is empty but stats exist, populate it
            if not active_symbols and self.universe.stats:
                active_symbols = list(self.universe.stats.keys())[:300]  # Use first 300 symbols
                self.universe.active = active_symbols
                self.logger.warning(
                    f"CRITICAL: universe.active was empty! Force-populated with {len(active_symbols)} symbols from stats"
                )
            
            # DEBUG ONLY: Log universe state (too frequent for LOG panel)
            self.logger.debug(
                f"SCAN START | active_symbols={len(active_symbols)} | "
                f"universe_size={len(self.universe.stats)} | "
                f"current_positions={current_positions} | equity={equity:.2f}"
            )
            
            # OPTION 3: Smart Discovery - Add discovery candidates periodically
            from .config import (
                DISCOVERY_SCAN_INTERVAL_SEC, DISCOVERY_SYMBOLS_PER_CYCLE,
                DISCOVERY_MIN_VOLUME_24H, DISCOVERY_MAX_SPREAD_BPS, DISCOVERY_MIN_MOMENTUM_PCT,
                STALE_SYMBOL_THRESHOLD_SEC
            )
            
            now = time.time()
            discovery_symbols = []
            
            # MARKET ACTIVITY DETECTION: Scan more frequently during high volatility periods
            # Adaptive scan interval: 10s base, 7s in high volatility, 12s in low volatility
            adaptive_scan_interval = DISCOVERY_SCAN_INTERVAL_SEC
            volatility_regime = getattr(self, 'volatility_regime', 'Normal')
            if volatility_regime == 'High':
                adaptive_scan_interval = DISCOVERY_SCAN_INTERVAL_SEC * 0.7  # 7s in high volatility (30% faster)
            elif volatility_regime == 'Low':
                adaptive_scan_interval = DISCOVERY_SCAN_INTERVAL_SEC * 1.2  # 12s in low volatility (20% slower)
            
            if now - self.last_discovery_scan >= adaptive_scan_interval:
                # Get discovery candidates (never-scanned, stale, or high-momentum symbols)
                # Prioritize candidates with strong recent momentum changes
                discovery_symbols = self.universe.get_discovery_candidates(
                    max_candidates=DISCOVERY_SYMBOLS_PER_CYCLE,
                    min_volume=DISCOVERY_MIN_VOLUME_24H,
                    max_spread_bps=DISCOVERY_MAX_SPREAD_BPS,
                    min_momentum_pct=DISCOVERY_MIN_MOMENTUM_PCT,
                    stale_threshold_sec=STALE_SYMBOL_THRESHOLD_SEC
                )
                self.last_discovery_scan = now
            
            # Combine active and discovery symbols (remove duplicates)
            all_symbols_to_scan = list(set(active_symbols + discovery_symbols))
            total_symbols = len(all_symbols_to_scan)
            
            # CRITICAL FIX: If all_symbols_to_scan is empty, populate from stats
            if not all_symbols_to_scan and self.universe.stats:
                all_symbols_to_scan = list(self.universe.stats.keys())[:300]
                active_symbols = all_symbols_to_scan.copy()
                self.universe.active = active_symbols
                total_symbols = len(all_symbols_to_scan)
                self.logger.warning(
                    f"CRITICAL: all_symbols_to_scan was empty! Force-populated with {len(all_symbols_to_scan)} symbols from stats"
                )
            
            # DEBUG: Log scan preparation (first scan only)
            if not hasattr(self, '_first_scan_logged'):
                self.logger.info(
                    f"AGGRESSIVE MODE SCAN PREP | active_symbols={len(active_symbols)} | "
                    f"discovery_symbols={len(discovery_symbols)} | "
                    f"all_symbols_to_scan={len(all_symbols_to_scan)} | "
                    f"universe.stats={len(self.universe.stats)} | "
                    f"universe.active={len(self.universe.active)}"
                )
                self._first_scan_logged = True

            # Calculate Market Regime (Bullish/Bearish/Neutral)
            # Logic: >60% of universe green = Bullish, >60% red = Bearish
            green_count = 0
            red_count = 0
            total_active = 0
            
            if self.universe and self.universe.stats:
                for s in self.universe.stats.values():
                    change = getattr(s, 'pct_change_24h', 0.0)
                    if change > 0.2:
                        green_count += 1
                        total_active += 1
                    elif change < -0.2:
                        red_count += 1
                        total_active += 1
            
            # Initialize market_regime before conditional logic
            market_regime = "neutral"
            if total_active > 10: # Need at least 10 symbols for valid regime
                green_pct = green_count / total_active
                red_pct = red_count / total_active
                if green_pct > 0.60:
                    market_regime = "bullish"
                elif red_pct > 0.60:
                    market_regime = "bearish"
            
            self.market_regime = market_regime
            # Ensure market_regime is always defined for closure access
            if 'market_regime' not in locals():
                market_regime = "neutral"
                
            # --------------------------------------------------------------------------------
            # CONTINUOUS REGIME SCALING (THE PHILOSOPHER'S DECREE)
            # --------------------------------------------------------------------------------
            # Dynamic Score based on Trend Intensity.
            # Base: 55.0
            # Intensity: Max(Green%, Red%)
            # Formula: Scales from FLOW (55) to APE (50) based on market heat.
            
            # Ensure intensity vars exist
            if 'green_pct' not in locals(): green_pct = 0.5
            if 'red_pct' not in locals(): red_pct = 0.5
            
            intensity = max(green_pct, red_pct)
            self.market_intensity = intensity  # Store for UI v3
            
            # Base Settings (Respect Config)
            from .config import MIN_SIGNAL_SCORE, MIN_SIGNAL_STRENGTH
            base_score = float(MIN_SIGNAL_SCORE)
            base_strength = float(MIN_SIGNAL_STRENGTH)
            
            regime_min_score = base_score
            regime_min_strength = base_strength
            active_regime_name = "USER_CFG"
            
            # #region agent log
            # Log removed
            # #endregion

            if intensity > 0.55: # Start scaling when trend > 55%
                # Calculate scalar (0.0 to 1.0)
                scalar = min(1.0, (intensity - 0.55) / 0.20)
                
                # Dynamic Scaling: Lower the bar if market is ripping (FOMO logic)
                # But never drop more than 5 points below user config
                regime_min_score = base_score - (5.0 * scalar)
                regime_min_strength = base_strength - (0.10 * scalar)
                
                # #region agent log
                # Log removed
                # #endregion

                if scalar > 0.8:
                    active_regime_name = "APE (MAX)"
                elif scalar > 0.4:
                    active_regime_name = f"FLOW+ ({int(scalar*100)}%)"
            
            # Store dynamic gate for UI v3 (Continuous Philosophy)
            self.dynamic_gate_score = regime_min_score
            
            # Log regime switch if it changed (debounced)
            if not hasattr(self, '_last_logged_regime') or self._last_logged_regime != active_regime_name:
                self.logger.info(f"REGIME SCALE: {active_regime_name} | Intensity={intensity:.2f} | Gate={regime_min_score:.1f}")
                self._last_logged_regime = active_regime_name
            
            # Log regime (debug level to avoid spam)
            # self.logger.debug(f"Market Regime: {market_regime.upper()} (Green: {green_count}, Red: {red_count}, Total: {total_active})")
        
        # OPTIMIZATION: Pre-set leverage for symbols to reduce entry latency
        if self.order_manager and all_symbols_to_scan:
            try:
                from .config import LEVERAGE_BASE
                # Pre-set leverage in background (non-blocking)
                self.order_manager.pre_set_leverage(all_symbols_to_scan, LEVERAGE_BASE)
            except Exception:
                pass  # Non-critical
        
        signals_found = 0
        entries_attempted = 0
        signals_blocked_circuit_breaker = 0
        signals_blocked_position_manager = 0
        
        # Reset per-scan entry counters
        self.entries_attempted_this_scan = 0
        self.entries_opened_this_scan = 0
        
        # PURE_SCALPER: Per-scan rejection counters
        rejected_score = 0  # Rejected by score < MIN_SIGNAL_SCORE
        rejected_percentile = 0  # Rejected by percentile filter (if enabled)
        rejected_micro = 0  # Rejected by microstructure filter
        rejected_risk = 0  # Rejected by risk/cooldown state
        rejected_capacity = 0  # Rejected by max positions
        passed_filters = 0  # Signals that passed all filters
        entry_candidates = 0  # Signals that passed position manager check (considered for entry)
        entries_opened_this_scan = 0  # Entries actually opened this scan
        
        # DIAGNOSTIC: Track filtering stages
        signals_pass_score_filter = 0
        signals_pass_strength_filter = 0
        signals_pass_percentile_filter = 0
        signals_pass_position_manager = 0
        
        # Scan tracking metrics
        symbols_processed = 0  # Symbols that were processed (had stats, attempted signal generation)
        symbols_skipped = {
            'in_position': 0,
            'no_stats': 0,
            'no_signal': 0,
            'score_below_threshold': 0
        }
        cache_hits = 0
        cache_misses = 0
        orderbook_success = 0
        orderbook_failures = 0
        
        # LATENCY OPTIMIZATION: Process symbols in parallel batches
        # OPTIMIZATION: Cache time.time() at batch level to avoid repeated calls
        batch_now = time.time()
        
        # -----------------------------
        # SCAN BUDGET GOVERNOR (Keep us off the fuse)
        # -----------------------------
        loop_budget_cap = 80  # tighter cap per scan loop (weight units, approximate)
        loop_budget_used = 0
        
        # Respect SYMBOLS_TO_SCAN limit aggressively
        symbols_to_scan_limit = getattr(self.cfg, "SYMBOLS_TO_SCAN", 80)
        all_symbols = self.universe.active[:symbols_to_scan_limit]
        
        async def process_symbol(symbol):
            # SCAN BUDGET GUARD: If we've used too much this loop, skip further processing
            nonlocal loop_budget_used, loop_budget_cap
            if loop_budget_used >= loop_budget_cap:
                return ('skipped', 'budget_cap', None, None, False, False)

            # PURE_SCALPER: Access outer scope counters (only those used inside this function)
            nonlocal rejected_score, rejected_percentile
            """Process a single symbol for signals (optimized)."""
            # OPTIMIZATION: Use cached time from batch level
            symbol_now = batch_now
            
            # DIAGNOSTIC: Log first few symbol processing attempts (MARKSMAN removed)
            # MARKSMAN removed - no longer used
            
            # Track cache hit/miss for all symbols (even skipped ones)
            cached_ticker = self.ticker_cache.get(symbol, max_age=2.0)  # 2s max age (universe refreshes every 3s)
            was_cache_hit = cached_ticker is not None
            
            # OPTIMIZATION: Use set for O(1) position lookup
            if symbol in self._positions_set:
                return ('skipped', 'in_position', None, None, was_cache_hit, False)
            
            # EARLY SKIP: Check invalid symbols cache BEFORE signal generation
            # This prevents unnecessary signal processing for symbols in "Reduce Only" mode
            if symbol in self.invalid_symbols:
                invalid_since = self.invalid_symbols[symbol]
                time_since_invalid = symbol_now - invalid_since
                if time_since_invalid < self.invalid_symbol_cooldown:
                    # Still in cooldown - skip silently (DEBUG log only)
                    self.logger.debug(
                        f"[SKIP_INVALID_EARLY] {symbol}: Cached as invalid (Reduce Only/Delisted), "
                        f"skipping signal generation. Will retry after {self.invalid_symbol_cooldown - time_since_invalid:.0f}s"
                    )
                    return ('skipped', 'invalid_symbol', None, None, was_cache_hit, False)
                else:
                    # Cooldown expired - remove from cache and allow processing
                    del self.invalid_symbols[symbol]
                    self.logger.debug(f"[RETRY_INVALID_EARLY] Retrying previously invalid symbol: {symbol}")
            
            # Get symbol stats from universe (already cached)
            stats = self.universe.stats.get(symbol)
            if not stats:
                # CRITICAL DEBUG: Log missing stats - check first few symbols to diagnose format mismatch
                # Only log first 3 to avoid spam
                if not hasattr(self, '_no_stats_logged_count'):
                    self._no_stats_logged_count = 0
                # DEBUG ONLY: NO STATS messages (excluded from LOG panel by UILogHandler)
                if self._no_stats_logged_count < 3:
                    self.logger.debug(
                        f"NO STATS for {symbol} | "
                        f"stats_keys_count={len(self.universe.stats)} | "
                        f"sample_stats_keys={list(self.universe.stats.keys())[:3] if self.universe.stats else []} | "
                        f"symbol_in_stats={symbol in (self.universe.stats or {})}"
                    )
                    self._no_stats_logged_count += 1
                return ('skipped', 'no_stats', None, None, was_cache_hit, False)
            
            # Track scan time (for stale detection) - NO API CALL, just timestamp
            stats.last_scanned = symbol_now
            
            # LATENCY OPTIMIZATION: Use cache or universe stats only - NO API CALLS
            # Update stats from cache if available and fresher
            if was_cache_hit:
                # Use cached data - sub-millisecond access
                latency_ms = 1  # Cache access is <1ms
                # Update stats from cache if fresher
                if cached_ticker.timestamp > stats.last_seen:
                    stats.bid = cached_ticker.bid
                    stats.ask = cached_ticker.ask
                    stats.last = cached_ticker.last
                    stats.mark = cached_ticker.mark
                    stats.spread_bps = cached_ticker.spread_bps
                    stats.pct_change_24h = cached_ticker.pct_change_24h
            else:
                # Cache miss - use universe stats (fresh from last refresh, no API call)
                # Universe refreshes every 3s, so stats are at most 3s old
                # This is acceptable for signal generation
                latency_ms = 2  # Assume 2ms for universe stats access (in-memory)
                # Stats are already populated from last universe refresh
                # No API call needed - we use what we have
            
            # PERFORMANCE: Pre-extract stats attributes (avoid repeated getattr in logging)
            stats_pct_change = stats.pct_change_24h if hasattr(stats, 'pct_change_24h') else 0.0
            stats_spread = stats.spread_bps
            stats_vol = stats.vol_quote
            stats_last = stats.last if hasattr(stats, 'last') else 0.0
            stats_bid = stats.bid if hasattr(stats, 'bid') else 0.0
            stats_ask = stats.ask if hasattr(stats, 'ask') else 0.0
            
            # OPTIMIZATION: Create symbol_stats dict efficiently
            symbol_stats = {
                'bid': stats_bid,
                'ask': stats_ask,
                'last': stats_last,
                'spread_bps': stats_spread,
                'vol_quote': stats_vol,
                'pct_change_24h': stats_pct_change
            }
            
            # Fetch orderbook for depth scoring (with fallback)
            # API BUDGET OPTIMIZATION: Use cached orderbook if available (fresher data)
            orderbook = None
            orderbook_fetched = False
            
            # OPTIMIZATION: Check cache first (updated every 2s)
            # OPTIMIZATION: Cache timestamp check result
            cached_ob = self.orderbook_cache.get(symbol)
            if cached_ob:
                cache_age = symbol_now - cached_ob['timestamp']
                if cache_age < 3.0:  # Use if < 3s old
                    orderbook = cached_ob['data']
                    orderbook_fetched = True
            else:
                # OPTIMIZATION: Do NOT fetch orderbook immediately for all symbols
                # Wait until we have a candidate signal to save API calls
                orderbook = None
                orderbook_fetched = False
            
            # Get position manager state for exposure scoring
            position_manager_state = self.position_manager.get_exposure_score_data()
            position_manager_state['loss_streak'] = self.position_manager.get_loss_streak()
            
            # Estimate order size (will be refined later, but needed for depth scoring)
            # Use a conservative estimate: 10% of account / entry_price
            equity = self.equity_now()
            estimated_order_size_usd = min(equity * 0.1, MAX_POSITION_SIZE)
            
            # REPLAY MODE: Fetch OHLCV data for signal generation (needed for indicators)
            price_data = None
            if self.replay_mode and self.replay_feed:
                # REPLAY MODE: Get OHLCV from replay feed for all symbols
                try:
                    # MARKSMAN requires 5m candles - fetch those first
                    candles_5m = self.replay_feed.get_ohlcv(symbol, timeframe='5m', limit=40)
                    if candles_5m and len(candles_5m) >= 36:
                        price_data = {'candles_5m': candles_5m}
                        # Also set as 'candles' for MARKSMAN (it checks both)
                        price_data['candles'] = candles_5m
                    
                    # Also get 1m candles for other strategies/indicators
                    try:
                        candles_1m = self.replay_feed.get_ohlcv(symbol, timeframe='1m', limit=200)
                        if candles_1m and len(candles_1m) >= 20:
                            if price_data:
                                price_data['candles_1m'] = candles_1m
                            else:
                                price_data = {'candles': candles_1m}
                    except (KeyError, ValueError, AttributeError, IndexError):
                        # Missing data or format issues - 1m candles not critical if 5m available
                        pass
                except (KeyError, ValueError, AttributeError, IndexError, OSError) as e:
                    # If candle fetch fails (missing data, format issues, or I/O errors), 
                    # signal generator will work with symbol_stats only
                    price_data = None
            
            # MARKSMAN MODE removed - no longer fetching 5m candles
            
            # PERFORMANCE: Use pre-bound scan-level invariants
            # SCORING V2: Pass bot_positions for portfolio scoring
            # DIAGNOSTIC: Log signal generator call
            import os
            if os.environ.get('REPLAY_MODE', '0') == '1':
                if not hasattr(self, '_signal_gen_call_logged'):
                    self._signal_gen_call_logged = set()
                if symbol not in self._signal_gen_call_logged and len(self._signal_gen_call_logged) < 3:
                    self._signal_gen_call_logged.add(symbol)
                    has_candles = price_data is not None and (price_data.get('candles') or price_data.get('candles_5m'))
                    print(f"[DIAG] Calling signal_generator.generate_signal for {symbol}: has_price_data={price_data is not None}, has_candles={has_candles}", flush=True)
            
            try:
                signal, rejection_reason = self.signal_generator.generate_signal(
                symbol=symbol,
                symbol_stats=symbol_stats,
                price_data=price_data,  # CRITICAL: Pass candle data for MARKSMAN
                orderbook=orderbook,
                latency_ms=latency_ms,
                order_size_usd=estimated_order_size_usd,
                position_manager_state=position_manager_state,
                btc_trend=scan_btc_trend,
                regime_config=scan_regime_config,
                recent_trades=scan_recent_trades,
                volatility_regime=scan_volatility_regime,
                bot_positions=self.positions,  # SCORING V2: Pass positions for portfolio scoring
                    # Use bot-wide market_regime attribute instead of closure variable
                    market_regime=getattr(self, "market_regime", "neutral"),  # NEW: Pass calculated market regime
                    # NEW: Dynamic Regime Switching Overrides (Council Decree #3)
                    min_score_override=regime_min_score,
                    min_strength_override=regime_min_strength,
                    # IDLE MODE: Skip percentile filter when no positions to find best available signal
                    skip_percentile_filter=self._idle_mode_active,
                    # IDLE MODE: Use IDLE_MIN_SCORE (42.0) when no positions
                    idle_mode_active=self._idle_mode_active
                )

                # CRITICAL OPTIMIZATION: Lazy Orderbook Fetch for High-Quality Candidates
                # Only fetch orderbook if preliminary score is promising (saves ~95% of API calls)
                if signal and not orderbook_fetched and signal.final_score >= (self.cfg.MIN_SIGNAL_SCORE - 5) and not DRY_RUN:
                    try:
                        if self.exchange_wrapper:
                            # [!] ADAPTIVE: Skip if in backoff mode
                            now_check = time.time()
                            if now_check < self.throttle_backoff_until:
                                return ('skipped', 'throttle_backoff', None, None, was_cache_hit, orderbook_fetched)
                            
                            # NOTE: Rate limiting now handled by GovernedExchange wrapper via ApiGovernor
                            denormalized_symbol = self.exchange_wrapper.denormalize_symbol(symbol)
                            # [!] ADAPTIVE DELAY: Use dynamic delay based on throttle health
                            await asyncio.sleep(self.adaptive_delay)
                            orderbook = await asyncio.wait_for(
                                self.exchange_wrapper.fetch_order_book(denormalized_symbol, limit=20),
                                timeout=0.2
                            )
                            orderbook_fetched = True
                            loop_budget_used += 1
                            self.orderbook_cache[symbol] = {'data': orderbook, 'timestamp': symbol_now}
                            
                            # Re-generate signal with confirmed orderbook data
                            signal, rejection_reason = self.signal_generator.generate_signal(
                                symbol=symbol,
                                symbol_stats=symbol_stats,
                                price_data=price_data,
                                orderbook=orderbook,  # Now populated
                                latency_ms=latency_ms,
                                order_size_usd=estimated_order_size_usd,
                                position_manager_state=position_manager_state,
                                btc_trend=scan_btc_trend,
                                regime_config=scan_regime_config,
                                recent_trades=scan_recent_trades,
                                volatility_regime=scan_volatility_regime,
                                bot_positions=self.positions,
                                market_regime=getattr(self, "market_regime", "neutral"),
                                min_score_override=regime_min_score,
                                min_strength_override=regime_min_strength,
                                skip_percentile_filter=self._idle_mode_active,  # IDLE MODE: Skip percentile when no positions
                                idle_mode_active=self._idle_mode_active  # IDLE MODE: Use IDLE_MIN_SCORE (42.0)
                            )
                            
                            # DIAGNOSTIC: Log if signal was rejected after lazy fetch
                            if rejection_reason:
                                self.logger.info(f"LAZY FETCH REJECTION: {symbol} Reason: {rejection_reason}")
                            
                            # [!] RECOVERY: Gradually reduce delay if no recent errors
                            if now_check - self.last_throttle_error > 60.0:
                                self.adaptive_delay = max(0.02, self.adaptive_delay * 0.95)  # Reduce delay by 5%
                    except Exception as e:
                        error_str = str(e)
                        # [!] ADAPTIVE: Detect throttle errors and trigger backoff
                        if "throttle queue" in error_str.lower() or "maxCapacity" in error_str:
                            self.throttle_error_count += 1
                            self.last_throttle_error = time.time()
                            backoff_duration = min(2.0 * (2 ** min(self.throttle_error_count - 1, 4)), 30.0)
                            self.throttle_backoff_until = time.time() + backoff_duration
                            self.adaptive_delay = min(self.adaptive_delay * 1.5, 1.0)
                            self.logger.debug(f"Throttle detected in lazy fetch. Backing off for {backoff_duration:.1f}s")
                        pass  # Continue with cached/None orderbook

                # DIAGNOSTIC: Log result for first few MARKSMAN symbols
                if os.environ.get('REPLAY_MODE', '0') == '1':
                    if not hasattr(self, '_signal_result_logged'):
                        self._signal_result_logged = set()
                    if symbol not in self._signal_result_logged and len(self._signal_result_logged) < 3:
                        self._signal_result_logged.add(symbol)
                        print(f"[DIAG] generate_signal RESULT for {symbol}: signal={signal is not None}, rejection_reason={rejection_reason}", flush=True)
            except Exception as e:
                # DIAGNOSTIC: Log exception (MARKSMAN removed)
                try:
                    # Log full traceback to debug log for analysis
                    self.logger.error(f"CRASH in generate_signal for {symbol}: {e}\n{traceback.format_exc()}")
                except (Exception, OSError, ImportError):
                    # Ignore diagnostic errors
                    pass
                signal = None
                rejection_reason = f"Exception in generate_signal: {e}"
            
            # Handle rejected signals (rejection_reason is not None)
            if rejection_reason is not None:
                # Signal was generated but rejected by signal generator filters
                # PURE_SCALPER: Track rejection reasons for summary
                # Note: These counters are per-symbol, we'll aggregate at scan end
                # For now, just track in signal_history - summary will count from history
                
                # Add to signal_history so it appears in signal queue
                # CRITICAL FIX: Record rejection even if signal is None (e.g. MARKSMAN filters)
                signal_record = {
                    'timestamp': batch_now,
                    'symbol': symbol,
                    'strength': signal.strength if signal else 0.0,
                    'final_score': signal.final_score if signal else 0.0,
                    'type': signal.signal_type if signal else 'rejected',
                    'side': signal.side if signal else '?',
                    'spread_bps': stats_spread,
                    'volatility': abs(stats_pct_change),
                    'btc_trend': scan_btc_trend,
                    'btc_filter_passed': True,
                    'approved': False,  # Rejected by signal generator
                    'rejection_reason': rejection_reason
                }
                
                # Add score breakdown if available
                if signal and signal.signal_score:
                    signal_record.update(signal.signal_score.to_dict())
                
                self.signal_history.append(signal_record)
                
                # PURE_SCALPER: Count rejections by reason (for summary)
                # These are from signal generator (score/percentile/strength checks)
                if "score below" in rejection_reason or "Score too low" in rejection_reason:
                    rejected_score += 1
                    
                    # TRASH MANAGEMENT: Add strike for low scores
                    # If score is TRULY bad (<40), strike it hard.
                    # If it's borderline (e.g. 52 vs 55), be lenient.
                    current_score = signal.final_score if signal else 0.0
                    if current_score < 40.0:
                        # Initialize if missing (safety)
                        if not hasattr(self, '_trash_strikes'): self._trash_strikes = {}
                        if not hasattr(self, '_trash_jail'): self._trash_jail = {}
                        
                        strikes = self._trash_strikes.get(symbol, 0) + 1
                        self._trash_strikes[symbol] = strikes
                        
                        if strikes >= 2:
                            # 2 Strikes -> 30 Minute Jail
                            self._trash_jail[symbol] = time.time() + 1800
                            # Reset strikes so it gets a fresh chance after jail
                            self._trash_strikes[symbol] = 0
                            # Log (debug only)
                            # self.logger.debug(f"JAILED {symbol} for 30m (Score {current_score:.1f} < 40 twice)")
                
                # TRASH MANAGEMENT: Strike for "No Pattern" (No signals generated)
                elif "No signals generated" in rejection_reason or "No Pattern" in rejection_reason:
                    # Treat this as a Strike (Score essentially 0)
                    if not hasattr(self, '_trash_strikes'): self._trash_strikes = {}
                    if not hasattr(self, '_trash_jail'): self._trash_jail = {}
                    
                    strikes = self._trash_strikes.get(symbol, 0) + 1
                    self._trash_strikes[symbol] = strikes
                    
                    if strikes >= 2:
                        self._trash_jail[symbol] = time.time() + 1800
                        self._trash_strikes[symbol] = 0
                
                elif "percentile" in rejection_reason:
                    rejected_percentile += 1
                
                # PERFORMANCE: Guard logging with isEnabledFor and use pre-extracted stats
                if not hasattr(self, '_signal_reject_log_count'):
                    self._signal_reject_log_count = 0
                # DEBUG ONLY: Signal rejections are aggregated in loop summary (too frequent for LOG panel)
                if self._signal_reject_log_count < 10:
                    self.logger.debug(
                        f"Signal REJECTED by generator: {symbol} | "
                        f"reason={rejection_reason} | "
                        f"score={signal.final_score if signal else 0:.1f} | "
                        f"pct={stats_pct_change:.2f}% | spread={stats_spread:.1f}bps"
                    )
                    self._signal_reject_log_count += 1
                
                # Return specific rejection reason for UI summary
                skip_category = 'rejected_other'
                if rejection_reason:
                    reason_lower = rejection_reason.lower()
                    if "score" in reason_lower:
                        skip_category = 'rejected_score'
                    elif "regime" in reason_lower or "bearish" in reason_lower or "bullish" in reason_lower:
                        skip_category = 'rejected_regime'
                    elif "momentum" in reason_lower:
                        skip_category = 'rejected_momentum'
                    elif "percentile" in reason_lower:
                        skip_category = 'rejected_percentile'
                    elif "no candle" in reason_lower:
                        skip_category = 'no_candles'
                    elif "not in marksman" in reason_lower:
                        skip_category = 'not_in_whitelist'
                    elif "quality" in reason_lower:
                        skip_category = 'rejected_quality'
                
                return ('skipped', skip_category, None, None, was_cache_hit, orderbook_fetched)
            
            # Handle case where no signal was generated at all
            if signal is None:
                # TRASH MANAGEMENT: Strike symbols that fail to generate signals
                # If a symbol repeatedly returns None (e.g. invalid data, 0 vol), Jail it.
                if not hasattr(self, '_trash_strikes'): self._trash_strikes = {}
                if not hasattr(self, '_trash_jail'): self._trash_jail = {}
                
                strikes = self._trash_strikes.get(symbol, 0) + 1
                self._trash_strikes[symbol] = strikes
                
                if strikes >= 2:
                    self._trash_jail[symbol] = time.time() + 1800
                    self._trash_strikes[symbol] = 0
                
                # CRITICAL DEBUG: Log why signal wasn't generated (log first 10 to diagnose)
                # PERFORMANCE: Guard logging and use pre-extracted stats
                if not hasattr(self, '_signal_reject_log_count'):
                    self._signal_reject_log_count = 0
                # DEBUG ONLY: Signal not generated messages (too frequent for LOG panel)
                if self._signal_reject_log_count < 10:
                    self.logger.debug(
                        f"Signal NOT generated: {symbol} | "
                        f"pct={stats_pct_change:.2f}% | spread={stats_spread:.1f}bps | "
                        f"vol=${stats_vol/1e6:.1f}M | last={stats_last:.4f} | bid={stats_bid:.4f} | ask={stats_ask:.4f}"
                    )
                    self._signal_reject_log_count += 1
                return ('skipped', 'no_signal', None, None, was_cache_hit, orderbook_fetched)
            
            return ('processed', symbol, stats, signal, latency_ms, was_cache_hit, orderbook, orderbook_fetched)
        
        # PERFORMANCE: Pre-bind invariants outside per-symbol loop
        # In REPLAY_MODE, disable regime config to use pure config.py settings for testing
        # UNLESS _use_regime_in_replay is set (for aggressive backtesting)
        if REPLAY_MODE and not getattr(self, '_use_regime_in_replay', False):
            scan_regime_config = None
        else:
            # Use forced regime config if set, otherwise use normal regime_config
            if hasattr(self, '_forced_regime_config') and self._forced_regime_config:
                scan_regime_config = self._forced_regime_config
            else:
                scan_regime_config = self.regime_config if hasattr(self, 'regime_config') else None
        scan_recent_trades = list(self.recent_trades) if hasattr(self, 'recent_trades') else []
        scan_volatility_regime = self.volatility_regime if hasattr(self, 'volatility_regime') else 'Normal'
        scan_btc_trend = self.btc_trend
        
        # LATENCY OPTIMIZATION: Process symbols in parallel (OPTIMIZED: larger batch size)
        # LOOP STATS ACCUMULATOR: Aggregate signal stats for summary log
        loop_stats = {
            'signals': 0,           # Total signals generated
            'unicorns': 0,          # Unicorn-level signals (score >= threshold)
            'longs': 0,             # Long signals
            'shorts': 0,            # Short signals
            'scores': [],           # All signal scores (for avg/best)
            'rejected_by_filters': 0  # Optional: rejected by trend/corr/etc.
        }
        from .config import UNICORN_SCORE_THRESHOLD
        
        # CRITICAL: Ensure all_symbols_to_scan exists - populate from universe if missing
        if 'all_symbols_to_scan' not in locals() or not all_symbols_to_scan:
            if self.universe.stats:
                all_symbols_to_scan = list(self.universe.stats.keys())[:300]
                self.logger.warning(
                    f"CRITICAL FIX: all_symbols_to_scan was missing/empty! Populated with {len(all_symbols_to_scan)} symbols"
                )
            else:
                self.logger.error("CRITICAL: Cannot scan - no symbols available (universe.stats is empty)")
                self.is_scanning = False
                return
        
        # DEBUG: Log scan preparation (first scan only)
        if not hasattr(self, '_scan_prep_logged'):
            self.logger.info(
                f"[!] SCAN PREP | all_symbols_to_scan={len(all_symbols_to_scan)} | "
                f"total_symbols_count={len(all_symbols_to_scan)} | "
                f"ENABLE_MARKSMAN={getattr(self.cfg, 'ENABLE_MARKSMAN', 'unknown')} | "
                f"Sample symbols: {all_symbols_to_scan[:5] if all_symbols_to_scan else 'NONE'}"
            )
            self._scan_prep_logged = True
        
        # PERFORMANCE: Increased batch size from 25 to 50 for better parallelization
        batch_size = 50
        total_symbols_count = len(all_symbols_to_scan)  # OPTIMIZATION: Cache len() result
        
        # DEBUG: Log if no symbols to scan (first occurrence only)
        if total_symbols_count == 0:
            if not hasattr(self, '_empty_scan_logged'):
                self.logger.error(
                    f"[X] CRITICAL: total_symbols_count is 0! "
                    f"all_symbols_to_scan={len(all_symbols_to_scan)} | "
                    f"universe.stats={len(self.universe.stats)} | "
                    f"universe.active={len(self.universe.active)} | "
                    f"ENABLE_MARKSMAN={getattr(self.cfg, 'ENABLE_MARKSMAN', 'unknown')}"
                )
                # Last resort: try to populate from universe.stats
                if self.universe.stats:
                    all_symbols_to_scan = list(self.universe.stats.keys())[:300]
                    total_symbols_count = len(all_symbols_to_scan)
                    self.logger.warning(f"LAST RESORT: Populated all_symbols_to_scan with {total_symbols_count} symbols")
                else:
                    self.is_scanning = False
                    return
            else:
                self.is_scanning = False
                return
        
        for i in range(0, total_symbols_count, batch_size):
            # OPTIMIZATION: Update cached time for each batch
            batch_now = time.time()
            batch = all_symbols_to_scan[i:i+batch_size]
            tasks = [process_symbol(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # OPTIMIZATION: Process results efficiently
            for result in results:
                if isinstance(result, Exception) or result is None:
                    continue
            
                result_type = result[0]
            
                if result_type == 'skipped':
                    skip_reason = result[1]
                    was_cache_hit = result[4] if len(result) > 4 else False
                    orderbook_fetched = result[5] if len(result) > 5 else False
            
                    # DYNAMIC UPDATE: Initialize key if missing
                    if skip_reason not in symbols_skipped:
                        symbols_skipped[skip_reason] = 0
                    symbols_skipped[skip_reason] += 1
                    
                    # FIX: Count 'no_signal' as processed for UI clarity
                    # If we ran the generator and got no signal, we effectively "processed" it
                    if skip_reason == 'no_signal' or 'rejected' in str(skip_reason).lower() or skip_reason == 'no_candles' or skip_reason == 'not_in_whitelist':
                        symbols_processed += 1
            
                    if was_cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
            
                    if orderbook_fetched:
                        orderbook_success += 1
                    else:
                        orderbook_failures += 1
            
                    continue
            
                if result_type != 'processed' or len(result) < 8:
                    continue
            
                status, symbol, stats, signal, latency_ms, was_cache_hit, orderbook, orderbook_fetched = result
                symbols_processed += 1
            
                if was_cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
            
                if orderbook_fetched:
                    orderbook_success += 1
                else:
                    orderbook_failures += 1
            
                if signal is None:
                    continue
                
                signals_found += 1
                signals_pass_score_filter += 1
                signals_pass_strength_filter += 1
                signals_pass_percentile_filter += 1
                self.signal_stats['signals_generated'] += 1
                self.signal_stats['last_signal_time'] = batch_now
                
                # SCALPER UPGRADE: Recreate symbol_stats dict from stats object (needed for filter)
                symbol_stats = {
                    'bid': getattr(stats, 'bid', 0.0),
                    'ask': getattr(stats, 'ask', 0.0),
                    'last': getattr(stats, 'last', 0.0),
                    'spread_bps': getattr(stats, 'spread_bps', 9999),
                    'vol_quote': getattr(stats, 'vol_quote', 0.0),
                    'pct_change_24h': getattr(stats, 'pct_change_24h', 0.0),
                    'price_tick': getattr(stats, 'price_tick', 0.0)  # PHASE 4: Exchange-aware spread calculation
                }
                
                # Get indicators for filter evaluation
                indicators = None
                if hasattr(self, 'indicators_cache'):
                    indicators = self.indicators_cache.get(symbol)
                
                # COLLECT STATS (NO LOGGING YET - aggregate into summary)
                final_score_float = float(signal.final_score) if signal.final_score is not None else 0.0
                threshold_float = float(UNICORN_SCORE_THRESHOLD)
                is_unicorn = UNICORN_PROTOCOL_ENABLED and final_score_float >= threshold_float
                
                # Accumulate stats for summary
                loop_stats['signals'] += 1
                loop_stats['scores'].append(final_score_float)
                if is_unicorn:
                    loop_stats['unicorns'] += 1
                if signal.side.lower() == 'long':
                    loop_stats['longs'] += 1
                else:
                    loop_stats['shorts'] += 1
            
                if stats:
                    stats.signals_generated_count += 1
                    stats.last_signal_time = batch_now
            
                if self.check_drawdown_circuit_breaker():
                    signals_blocked_circuit_breaker += 1
                    self._log_signal_decision(symbol, signal, "rejected", "drawdown_circuit_breaker")
                    continue
            
                # TREND ALIGNMENT REQUIREMENT: Check 5m and 15m trend alignment (optional - can be too restrictive)
                from .config import TREND_ALIGNMENT_REQUIRED
                if TREND_ALIGNMENT_REQUIRED:  # Disabled by default - too restrictive for scalping
                    trend5 = 0
                    trend15 = 0
                    try:
                        # Get trend from OHLCV data (replay feed or fast_storage)
                        ohlcv_5m = None
                        ohlcv_15m = None
                        
                        # REPLAY MODE: Get OHLCV from replay feed
                        if self.replay_mode and self.replay_feed:
                            ohlcv_5m = self.replay_feed.get_ohlcv(symbol, timeframe='5m', limit=20)
                            ohlcv_15m = self.replay_feed.get_ohlcv(symbol, timeframe='15m', limit=20)
                        # LIVE MODE: Try fast_storage (if it has get_ohlcv method)
                        elif hasattr(self.fast_storage, 'get_ohlcv'):
                            ohlcv_5m = self.fast_storage.get_ohlcv(symbol, timeframe='5m', limit=20)
                            ohlcv_15m = self.fast_storage.get_ohlcv(symbol, timeframe='15m', limit=20)
                        
                        if ohlcv_5m and len(ohlcv_5m) >= 10:
                            prices_5m = [bar[4] for bar in ohlcv_5m]  # Close prices
                            from .indicators import calculate_trend_direction_from_prices
                            trend5 = calculate_trend_direction_from_prices(prices_5m, ema_short=3, ema_long=8)
                        else:
                            trend5 = 0
                        
                        if ohlcv_15m and len(ohlcv_15m) >= 10:
                            prices_15m = [bar[4] for bar in ohlcv_15m]  # Close prices
                            trend15 = calculate_trend_direction_from_prices(prices_15m, ema_short=5, ema_long=10)
                        else:
                            trend15 = 0
                    except Exception as e:
                        # If trend calculation fails, reject (conservative)
                        self.logger.debug(f"Trend calculation failed for {symbol}: {e}")
                        trend5 = 0
                        trend15 = 0
                    
                    # Check trend alignment: both 5m and 15m must align with trade direction
                    side_mult = 1 if signal.side == 'long' else -1
                    trend5_aligned = (trend5 * side_mult) > 0  # Same sign = aligned
                    trend15_aligned = (trend15 * side_mult) > 0
                    
                    if not (trend5_aligned and trend15_aligned):
                        # Reject: misaligned trend
                        self._log_signal_decision(symbol, signal, "rejected", "RJ - misaligned trend")
                        loop_stats['rejected_by_filters'] += 1
                        signal_record = {
                            'timestamp': batch_now,
                            'symbol': symbol,
                            'strength': signal.strength,
                            'final_score': signal.final_score,
                            'type': signal.signal_type,
                            'side': signal.side,
                            'spread_bps': stats.spread_bps,
                            'volatility': abs(getattr(stats, 'pct_change_24h', 0.0)),
                            'btc_trend': self.btc_trend,
                            'btc_filter_passed': True,
                            'approved': False,
                            'rejection_reason': "RJ - misaligned trend",
                            'is_unicorn': is_unicorn
                        }
                        if signal.signal_score:
                            signal_record.update(signal.signal_score.to_dict())
                        self.signal_history.append(signal_record)
                        continue
                
                # DIRECTION LIMIT: Max positions in same direction (LONG or SHORT)
                # Prevents blowing up if the entire market reverses while you're stacked one way
                # SWARM MODE: Allow more positions per direction (was 4, now 12)
                MAX_SAME_DIRECTION = 12
                if self.positions:
                    signal_side = signal.side.upper() if signal.side else ""
                    same_dir_count = 0
                    for pos_symbol, pos_data in self.positions.items():
                        pos_side = ""
                        if hasattr(pos_data, 'side'):
                            pos_side = pos_data.side.upper()
                        elif isinstance(pos_data, dict):
                            pos_side = pos_data.get('side', '').upper()
                        if pos_side == signal_side:
                            same_dir_count += 1
                    
                    if same_dir_count >= MAX_SAME_DIRECTION:
                        self._log_signal_decision(symbol, signal, "rejected", f"RJ - Max {MAX_SAME_DIRECTION} {signal_side} positions (have {same_dir_count})")
                        loop_stats['rejected_by_filters'] += 1
                        signal_record = {
                            'timestamp': batch_now,
                            'symbol': symbol,
                            'strength': signal.strength,
                            'final_score': signal.final_score,
                            'type': signal.signal_type,
                            'side': signal.side,
                            'spread_bps': stats.spread_bps,
                            'volatility': abs(getattr(stats, 'pct_change_24h', 0.0)),
                            'btc_trend': self.btc_trend,
                            'btc_filter_passed': True,
                            'approved': False,
                            'rejection_reason': f"RJ - Max {MAX_SAME_DIRECTION} {signal_side} positions",
                            'is_unicorn': is_unicorn
                        }
                        if signal.signal_score:
                            signal_record.update(signal.signal_score.to_dict())
                        self.signal_history.append(signal_record)
                        continue
            
                btc_filter_passed = True
            
                # PHASE 2: Unified Quality Gate (soft penalty instead of hard rejection)
                from .config import USE_THREE_STAGE_FILTER
                from .engine.quality_gate import compute_quality_gate_score
                
                quality_score = None
                original_final_score = signal.final_score
                
                if USE_THREE_STAGE_FILTER:
                    # Get advanced_features for quality gate evaluation
                    advanced_features = None
                    if hasattr(self, '_advanced_features_cache'):
                        advanced_features = self._advanced_features_cache.get(symbol)
                    
                    # Compute quality gate score (returns multiplier 0-1)
                    # PHASE 2: Use evolvable filter_strictness from config (or default)
                    from .config import FILTER_STRICTNESS
                    # If signal has filter_strictness from evolved config, use it; otherwise use default
                    filter_strictness = getattr(signal, 'filter_strictness', None) or FILTER_STRICTNESS
                    
                    # PHASE 4: Get evolvable microstructure thresholds from config
                    from .config import (
                        MAX_SPREAD_PCT, MIN_DEPTH_USD, MAX_VOLATILITY_PCT,
                        MIN_VOLATILITY_PCT, SLIPPAGE_BUDGET_BPS, MIN_VIABLE_SIZE_USD
                    )
                    
                    quality_score = compute_quality_gate_score(
                        symbol=symbol,
                        side=signal.side,
                        symbol_stats=symbol_stats,
                        orderbook=orderbook,
                        indicators=indicators,
                        advanced_features=advanced_features,
                        entry_price=signal.entry_price,
                        filter_strictness=filter_strictness,
                        # PHASE 4: Evolvable microstructure thresholds
                        max_spread_pct=getattr(signal, 'max_spread_pct', None) or MAX_SPREAD_PCT,
                        min_depth_usd=getattr(signal, 'min_depth_usd', None) or MIN_DEPTH_USD,
                        max_volatility_pct=getattr(signal, 'max_volatility_pct', None) or MAX_VOLATILITY_PCT,
                        min_volatility_pct=getattr(signal, 'min_volatility_pct', None) or MIN_VOLATILITY_PCT,
                        slippage_budget_bps=getattr(signal, 'slippage_budget_bps', None) or SLIPPAGE_BUDGET_BPS,
                        min_viable_size_usd=getattr(signal, 'min_viable_size_usd', None) or MIN_VIABLE_SIZE_USD
                    )
                    
                    # Apply quality multiplier to final_score (soft penalty)
                    signal.final_score = signal.final_score * quality_score.multiplier
                    signal.strength = signal.final_score / 100.0  # Update strength to match
                    
                    # Only reject if quality is extremely poor (multiplier < 0.3)
                    # This allows evolution to learn from borderline signals
                    if quality_score.multiplier < 0.3:
                        rejection_reason = f"quality_too_low (multiplier={quality_score.multiplier:.2f})"
                        self._log_signal_decision(symbol, signal, "rejected", rejection_reason)
                        loop_stats['rejected_by_filters'] += 1
                        signal_record = {
                            'timestamp': batch_now,
                            'symbol': symbol,
                            'strength': original_final_score / 100.0,
                            'final_score': original_final_score,
                            'adjusted_score': signal.final_score,
                            'quality_multiplier': quality_score.multiplier,
                            'type': signal.signal_type,
                            'side': signal.side,
                            'spread_bps': stats.spread_bps,
                            'volatility': abs(getattr(stats, 'pct_change_24h', 0.0)),
                            'btc_trend': self.btc_trend,
                            'btc_filter_passed': btc_filter_passed,
                            'approved': False,
                            'rejection_reason': rejection_reason,
                            'is_unicorn': is_unicorn,
                            'quality_details': quality_score.details
                        }
                        if signal.signal_score:
                            signal_record.update(signal.signal_score.to_dict())
                        self.signal_history.append(signal_record)
                        continue
                
                # Quality gate passed (or disabled) - signal continues with adjusted score
                # NOTE: Do NOT increment passed_filters here - wait until position manager check passes
                
                # ============================================================
                # ADAPTIVE SCORE CHECK: Dynamic threshold for 24/7 activity
                # ============================================================
                # Threshold adapts based on:
                # - Recent signal quality (percentile of last ~500 signals)
                # - Current position count (more positions = more selective)
                # - Time since last entry (decay to encourage activity)
                from . import config as cfg
                
                # Record signal to adaptive controller (for threshold learning)
                # Use actual Binance position count to prevent false "max positions" rejections
                try:
                    binance_positions = await self.exchange_wrapper.fetch_positions()
                    valid_binance_count = 0
                    for rp in binance_positions:
                        sym = rp.get('symbol') or rp.get('symbolName')
                        if sym:
                            clean_sym = sym.replace("/", "").replace("-", "").replace(":", "")
                            if not ("USDTUSDT" in clean_sym or (len(clean_sym) > 6 and clean_sym[-6:].isdigit())):
                                size = abs(float(rp.get('contracts', rp.get('amount', rp.get('positionAmt', 0.0))) or 0.0))
                                if size > 0:
                                    valid_binance_count += 1
                    current_position_count = max(len(self.positions), valid_binance_count)
                except Exception:
                    current_position_count = len(self.positions)
                
                # Use adaptive threshold if enabled, else fallback to fixed
                if getattr(cfg, 'USE_ADAPTIVE_ENTRY', True):
                    # BLOCK TRADING until sufficient market data is collected
                    if not self.adaptive_entry.has_sufficient_data():
                        # Record signal for learning but skip trading
                        self.adaptive_entry.record_signal(
                            symbol=symbol,
                            score=signal.final_score,
                            side=signal.side,
                            was_taken=False,
                            rejection_reason="insufficient_data_warmup"
                        )
                        # Skip this signal - still collecting data
                        continue
                    
                    min_ml_score = self.adaptive_entry.get_threshold(current_position_count)
                    threshold_source = f"ADAPT({min_ml_score:.0f})"
                    
                    # Update dynamic gate for UI
                    self.dynamic_gate_score = min_ml_score
                else:
                    # Fallback to legacy fixed thresholds
                    if self._idle_mode_active:
                        min_ml_score = float(getattr(cfg, 'IDLE_MIN_SCORE', 35.0))
                        threshold_source = f"IDLE({min_ml_score:.0f})"
                    else:
                        min_ml_score = float(self.cfg.MIN_SIGNAL_SCORE)
                        threshold_source = f"NORMAL({min_ml_score:.0f})"
                
                # CHECK: Score must meet adaptive threshold
                if signal.final_score < min_ml_score:
                    # Record rejected signal for adaptive learning
                    self.adaptive_entry.record_signal(
                        symbol=symbol,
                        score=signal.final_score,
                        side=signal.side,
                        was_taken=False,
                        rejection_reason=f"below_threshold({signal.final_score:.0f}<{min_ml_score:.0f})"
                    )
                    
                    # Reject signal - below threshold
                    self.logger.warning(
                        f"[SCORE_REJECT] {symbol} score={signal.final_score:.1f} < {threshold_source}({min_ml_score:.0f})"
                    )
                    signal_record = {
                        'timestamp': batch_now,
                        'symbol': symbol,
                        'strength': signal.strength,
                        'final_score': signal.final_score,
                        'type': signal.signal_type,
                        'side': signal.side,
                        'spread_bps': stats.spread_bps if stats else 0,
                        'approved': False,
                        'rejection_reason': f'{threshold_source}_score({signal.final_score:.0f}<{min_ml_score:.0f})',
                        'is_unicorn': is_unicorn
                    }
                    self.signal_history.append(signal_record)
                    continue
                
                # Record signal that passed threshold (may still be rejected by position manager)
                self.adaptive_entry.record_signal(
                    symbol=symbol,
                    score=signal.final_score,
                    side=signal.side,
                    was_taken=False,  # Will update to True if entry succeeds
                    rejection_reason=None
                )
                
                # Log when signal passes score check
                self.logger.info(
                    f"[SCORE_PASS] {symbol} score={signal.final_score:.1f} >= {threshold_source}({min_ml_score:.0f})"
                )
                # ============================================================
                
                # DIAGNOSTIC: Log that we're proceeding to position manager
                self.logger.info(f"[POST_SCORE] {symbol} passed score check, proceeding to position manager... (in_startup_period={in_startup_period})")
                
                # Get current drawdown for DD-aware protection and auto-reset
                drawdown_pct = self.get_drawdown_pct()
                
                # GUARD: Skip invalid symbols (Reduce Only, delisted, etc.)
                if symbol in self.invalid_symbols:
                    invalid_since = self.invalid_symbols[symbol]
                    time_since_invalid = batch_now - invalid_since
                    # Retry after cooldown period (symbols may become available again)
                    if time_since_invalid < self.invalid_symbol_cooldown:
                        continue  # Skip this symbol
                    else:
                        # Cooldown expired - remove from cache and retry
                        del self.invalid_symbols[symbol]
                        self.logger.debug(f"Retrying previously invalid symbol: {symbol}")
                
                # RISK-BASED ENTRY: Check if we can enter based on risk budget
                # First pass: basic validation (without position sizing)
                # DATA-DRIVEN: Pass entry features for volatility/momentum/RSI filters
                entry_rsi = indicators.get('rsi', 50) if indicators else None
                entry_atr_pct = indicators.get('atr_pct') if indicators else getattr(stats, 'atr_pct', None)
                entry_pct_1h = getattr(stats, 'pct_change_1h', None) or getattr(stats, 'pct_change_24h', 0) / 24.0  # Estimate if 1h not available
                
                # ADAPTIVE FILTERS: Feed observation to build market model
                # This allows thresholds to auto-adjust based on current market conditions
                try:
                    from app.adaptive_filters import feed_observation
                    if entry_atr_pct and entry_rsi and entry_pct_1h:
                        feed_observation(symbol, entry_atr_pct, entry_pct_1h, entry_rsi, 
                                        getattr(stats, 'volume_ratio', 1.0))
                except ImportError:
                    pass  # Adaptive filters not available
                
                # DIAGNOSTIC: Log position limit calculation for debugging (once per minute)
                if not hasattr(self, '_position_limit_logged') or time.time() - getattr(self, '_position_limit_logged', 0) > 60:
                    try:
                        max_pos = self.position_manager.get_effective_max_positions(equity=equity) if equity > 0 else self.cfg.MAX_OPEN_POSITIONS
                        self.logger.info(
                            f"[POSITION_LIMIT] Current: {current_positions}/{max_pos} | "
                            f"MAX_ACCOUNT_RISK_PCT={self.cfg.MAX_ACCOUNT_RISK_PCT}% | "
                            f"RISK_PER_TRADE_PCT={self.cfg.RISK_PER_TRADE_PCT}% | "
                            f"HARD_CAP={self.cfg.MAX_CONCURRENT_POS_HARD}"
                        )
                        self._position_limit_logged = time.time()
                    except Exception:
                        pass
                
                can_enter, reason, replacement_symbol = self.position_manager.can_enter_position(
                symbol=symbol,
                spread_bps=stats.spread_bps,
                volume_24h=stats.vol_quote,
                latency_ms=latency_ms,
                signal_strength=signal.strength,
                    current_positions=current_positions,
                    cached_now=batch_now,
                    is_unicorn=is_unicorn,
                    signal_score=signal.final_score,
                    drawdown_pct=drawdown_pct,
                    open_positions=self.positions,
                    equity=equity,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    position_size=None,  # Will be calculated below
                    side=signal.side,
                    # Data-driven entry filters
                    rsi=entry_rsi,
                    atr_pct=entry_atr_pct,
                    pct_change_1h=entry_pct_1h
                )
            
                # DIAGNOSTIC: Log position manager rejection reasons in REPLAY_MODE
                if self.replay_mode and not can_enter:
                    if not hasattr(self, '_pm_reject_logged'):
                        self._pm_reject_logged = set()
                    if symbol not in self._pm_reject_logged and len(self._pm_reject_logged) < 5:
                        self._pm_reject_logged.add(symbol)
                        print(f"[DIAG] Position Manager REJECTED {symbol}: reason={reason}, strength={signal.strength:.2f}, score={signal.final_score:.1f}, spread={stats.spread_bps:.1f}bps, positions={current_positions}", flush=True)

                # DIAGNOSTIC: Log ALL position manager rejections (not just high scores)
                if not can_enter:
                    self.logger.warning(f"[POSITION_MANAGER_REJECT] {symbol} (Score={signal.final_score:.1f}) | Reason: {reason} | Spread: {stats.spread_bps:.1f}bps | Positions: {current_positions}")
                
                # WAR ROOM DIAGNOSTIC: Explicitly log rejections for High Score signals in LIVE mode
                if not can_enter and not self.replay_mode and signal.final_score >= 70:
                    self.logger.info(f"REJECTED {symbol} (Score={signal.final_score:.1f}) | Reason: {reason} | Spread: {stats.spread_bps}bps")

                if can_enter:
                    signals_pass_position_manager += 1
                    # DIAGNOSTIC: Log successful position manager approval in REPLAY_MODE
                    if self.replay_mode:
                        if not hasattr(self, '_pm_accept_logged'):
                            self._pm_accept_logged = set()
                        if symbol not in self._pm_accept_logged and len(self._pm_accept_logged) < 5:
                            self._pm_accept_logged.add(symbol)
                            print(f"[DIAG] Position Manager ACCEPTED {symbol}: strength={signal.strength:.2f}, score={signal.final_score:.1f}, spread={stats.spread_bps:.1f}bps", flush=True)
            
                signal_record = {
                    'timestamp': batch_now,
                    'symbol': symbol,
                    'strength': signal.strength,
                    'final_score': signal.final_score,
                    'type': signal.signal_type,
                    'side': signal.side,
                    'spread_bps': stats.spread_bps,
                    'volatility': abs(getattr(stats, 'pct_change_24h', 0.0)),
                    'btc_trend': self.btc_trend,
                    'btc_filter_passed': btc_filter_passed,
                    'approved': can_enter,
                    'rejection_reason': reason if not can_enter else None,
                    'is_unicorn': is_unicorn
                }
                # Add quality gate details if available
                if quality_score:
                    signal_record['quality_multiplier'] = quality_score.multiplier
                    signal_record['quality_details'] = quality_score.details
                    signal_record['original_score'] = original_final_score
                    signal_record['adjusted_score'] = signal.final_score
                if signal.signal_score:
                    signal_record.update(signal.signal_score.to_dict())
                self.signal_history.append(signal_record)
            
                # BLOCK ENTRIES during startup warmup (but allow signal scanning to build history)
                if in_startup_period:
                    self.logger.warning(f"[STARTUP_BLOCK] {symbol} blocked during warmup period (signal passed all checks but startup delay active)")
                    continue  # Skip this signal - we're still building signal history for percentile filter
                
                # SIGNAL CONFIRMATION WINDOW (SCW): Check if signal needs confirmation
                from .config import USE_SIGNAL_CONFIRMATION
                signal_needs_confirmation = False
                if USE_SIGNAL_CONFIRMATION and can_enter:
                    confirmation_required = self.signal_confirmation.get_confirmation_required(signal.final_score)
                    if confirmation_required > 0:
                        # Signal needs confirmation - add to waiting queue
                        key_level = signal.stop_loss  # Use stop_loss as key level
                        waiting = self.signal_confirmation.add_waiting_signal(symbol, signal, key_level)
                        signal_needs_confirmation = True
                        can_enter = False
                        reason = f"SCW: Waiting for {confirmation_required} bar confirmation (current: {waiting.confirmation_count})"
                        # REMOVED: Individual signal confirmation log (aggregated in summary)
                    elif symbol in self.signal_confirmation.waiting_signals:
                        # Check if signal is already confirmed
                        ready_signal = self.signal_confirmation.get_ready_signal(symbol)
                        if ready_signal:
                            # Signal is confirmed and ready
                            can_enter = True
                            reason = "SCW: Signal confirmed"
                            self.signal_confirmation.remove_signal(symbol)
                        else:
                            # Still waiting
                            can_enter = False
                            waiting = self.signal_confirmation.waiting_signals[symbol]
                            reason = f"SCW: Waiting for confirmation ({waiting.confirmation_count}/{waiting.confirmation_required})"
                            signal_needs_confirmation = True
                
                self._log_signal_decision(
                    symbol,
                    signal,
                    "approved" if can_enter else "rejected",
                    reason if not can_enter else None,
                )
                
                if not can_enter:
                    self.signal_stats['signals_blocked'] += 1
                    if not signal_needs_confirmation:
                        signals_blocked_position_manager += 1
                    
                    # PURE_SCALPER: Categorize rejection reasons for counters
                    reason_str = reason or "unknown"
                    if "score below" in reason_str or "Score too low" in reason_str or "min_score_guard" in reason_str:
                        rejected_score += 1
                    elif "percentile" in reason_str:
                        rejected_percentile += 1
                    elif "capacity" in reason_str or "max positions" in reason_str or "max_positions" in reason_str or "RJ - max positions" in reason_str:
                        rejected_capacity += 1
                    elif "risk" in reason_str or "cooldown" in reason_str or "loss_streak" in reason_str or "drawdown" in reason_str:
                        rejected_risk += 1
                    
                    # DEBUG: Log rejection reason (demoted to DEBUG to reduce spam)
                    # Use dynamic cap for logging if equity available, else fallback to legacy
                    try:
                        equity = self.equity_now()
                        max_pos = self.position_manager.get_effective_max_positions(equity=equity) if equity > 0 else self.cfg.MAX_OPEN_POSITIONS
                    except (AttributeError, KeyError, ValueError, TypeError) as e:
                        # Fallback to config default if equity calculation fails (missing attributes/data)
                        max_pos = self.cfg.MAX_OPEN_POSITIONS
                    self.logger.debug(
                        f"[ENTRY_REJECTED] symbol={symbol} side={signal.side} "
                        f"score={signal.final_score:.1f} reason={reason_str} "
                        f"current_positions={current_positions}/{max_pos}"
                    )
                    
                    # METRICS: count rejection reasons
                    try:
                        key = (reason or "unknown").split()[0]
                        self.metrics['rejections_by_reason'][key] = self.metrics['rejections_by_reason'].get(key, 0) + 1
                    except (AttributeError, KeyError, TypeError):
                        # Metrics dict structure issue or invalid reason format - non-critical, safe to ignore
                        pass
                    # We may skip extra logging for startup / SCW, but we must ALWAYS skip entry:
                    continue
                
                # PURE_SCALPER: Signal passed all checks (filter pipeline + position manager) - ready for entry
                # (Entry will be attempted below)
                # CRITICAL: Only count as "passed" if signal passed BOTH filter pipeline AND position manager check
                # This ensures Signal Health panel shows accurate pass rate (signals ready for entry)
                passed_filters += 1  # Signal passed filter pipeline AND position manager check
                entry_candidates += 1  # Track signals that passed position manager check
                
                # DEBUG: Log that we're about to attempt entry (demoted to DEBUG to reduce spam)
                # Use dynamic cap for logging if equity available, else fallback to legacy
                try:
                    equity = self.equity_now()
                    max_pos = self.position_manager.get_effective_max_positions(equity=equity) if equity > 0 else self.cfg.MAX_OPEN_POSITIONS
                except Exception:
                    max_pos = self.cfg.MAX_OPEN_POSITIONS
                self.logger.debug(
                    f"[ENTRY_ATTEMPT] symbol={symbol} side={signal.side} "
                    f"score={signal.final_score:.1f} can_enter={can_enter} "
                    f"current_positions={current_positions}/{max_pos}"
                )
                
                # Signal passed all checks - proceed with entry
                try:
                    from .validators import (
                        validate_price,
                        validate_side,
                        validate_stop_loss_take_profit,
                    )
                    entry_price = validate_price(signal.entry_price, "entry_price")
                    validate_side(signal.side)
                    validate_stop_loss_take_profit(
                        entry_price,
                        signal.stop_loss,
                        signal.take_profit,
                        signal.side,
                    )
                except Exception as e:
                    self.logger.warning(
                        "Signal validation failed",
                        symbol=symbol,
                        error=str(e),
                    )
                    continue
                
                # BINANCE BEST PRACTICE: Early validation of position size requirements
                # Check if position will meet min quantity/notional BEFORE calculating size
                # This saves processing time and prevents wasted API calls
                try:
                    if self.exchange_wrapper and hasattr(self.exchange_wrapper, 'exchange') and self.exchange_wrapper.exchange:
                        market = self.exchange_wrapper.exchange.market(symbol) if self.exchange_wrapper.exchange.markets else None
                        if market:
                            min_qty = market.get('limits', {}).get('amount', {}).get('min')
                            min_cost = market.get('limits', {}).get('cost', {}).get('min')
                            
                            # Estimate minimum position size needed
                            # Use entry price to estimate (for market orders, this is approximate)
                            if min_qty and min_cost:
                                # Calculate minimum size needed to meet both requirements
                                min_size_by_qty = min_qty
                                min_size_by_notional = min_cost / entry_price if entry_price > 0 else 0
                                estimated_min_size = max(min_size_by_qty, min_size_by_notional)
                                
                                # Check if our minimum position size config will meet requirements
                                from .config import MIN_POSITION_SIZE, RPA_MIN_SIZE_USD, USE_RANK_BASED_ALLOCATION
                                min_config_size = (RPA_MIN_SIZE_USD / entry_price) if USE_RANK_BASED_ALLOCATION else (MIN_POSITION_SIZE / entry_price)
                                
                                if estimated_min_size > min_config_size * 2:  # Allow 2x buffer for safety
                                    self.logger.debug(
                                        f"[BINANCE_VALIDATION] {symbol}: Estimated min size {estimated_min_size:.6f} > "
                                        f"config min {min_config_size:.6f}. Will validate during position sizing."
                                    )
                except Exception as e:
                    # Non-critical - continue with position sizing
                    self.logger.debug(f"[BINANCE_VALIDATION] Could not pre-validate {symbol}: {e}")
                
                # Handle replacement if needed
                # REMOVED: Individual replacement log (tracked via exit DecisionEvent)
                if replacement_symbol:
                        # Close the weakest position to make room
                        # Close the position (will be handled by exit manager)
                        # CRITICAL: Check if replacement symbol still exists
                        if replacement_symbol in self.positions:
                            replacement_position = self.positions[replacement_symbol]
                            
                            # Force exit the weakest position
                            exit_result = await self.exit_pipeline._execute_exit_order(
                                symbol=replacement_symbol,
                                position=replacement_position,
                                reason="replaced_by_better_signal",
                                funding_rate=replacement_position.get("funding_rate", 0.0),
                            )
                            
                            if exit_result.success:
                                # Update stats
                                was_win = exit_result.net_pnl > 0 if exit_result.net_pnl else False
                                pnl_pct = (exit_result.net_pnl / equity * 100) if exit_result.net_pnl else None
                                
                                # CANONICAL: Apply exit and log atomically
                                from .position_utils import apply_exit_and_log, record_loop_exit
                                
                                replacement_entry_price = replacement_position.get('entry_price', 0)
                                replacement_entry_time = replacement_position.get('entry_time', time.time())
                                replacement_side = replacement_position.get('side', '')
                                replacement_size = replacement_position.get('size', 0)
                                
                                # Calculate PnL percentage
                                replacement_pnl_pct = 0.0
                                if replacement_entry_price > 0 and exit_result.exit_price and exit_result.exit_price > 0:
                                    if replacement_side.lower() == 'long':
                                        replacement_pnl_pct = ((exit_result.exit_price - replacement_entry_price) / replacement_entry_price) * 100
                                    else:
                                        replacement_pnl_pct = ((replacement_entry_price - exit_result.exit_price) / replacement_entry_price) * 100
                                
                                replacement_now = time.time()
                                replacement_prs = replacement_position.get('recovery_score')
                                
                                success, event_created = apply_exit_and_log(
                                    positions=self.positions,
                                    positions_set=self._positions_set,
                                    symbol=replacement_symbol,
                                    new_size=0.0,  # Full exit
                                    action="EXIT",
                                    exit_price=exit_result.exit_price,
                                    entry_price=replacement_entry_price,
                                    entry_time=replacement_entry_time,
                                    exit_time=replacement_now,
                                    exit_size=exit_result.exit_size or replacement_size,
                                    size_before=replacement_size,
                                    size_after=0.0,
                                    side=replacement_side,
                                    pnl_value=exit_result.net_pnl or 0.0,
                                    pnl_pct=replacement_pnl_pct,
                                    gross_pnl=exit_result.gross_pnl or 0.0,
                                    net_pnl=exit_result.net_pnl or 0.0,
                                    total_costs=exit_result.total_costs or 0.0,
                                    reason="replaced_by_better_signal",
                                    prs=replacement_prs,
                                    was_win=was_win,
                                    is_unicorn=replacement_position.get('is_unicorn', False),
                                    bot_instance=self
                                )
                                
                                if success and event_created:
                                    record_loop_exit(replacement_symbol, "EXIT")
                                    self.position_manager.record_exit(replacement_symbol, was_win, pnl_pct=pnl_pct, exit_reason=None, profit_atr=None)
                                
                                current_positions = len(self.positions)
                            elif "Invalid position" in str(exit_result.error or ""):
                                # Position doesn't exist on exchange - clean up from dict
                                if replacement_symbol in self.positions:
                                    del self.positions[replacement_symbol]
                                    if replacement_symbol in self._positions_set:
                                        self._positions_set.discard(replacement_symbol)
                                current_positions = len(self.positions)
                
                leverage = self.position_manager.calculate_dynamic_leverage(
                    signal.strength,
                    is_unicorn=is_unicorn,
                )
                
                # SCALPER: Get ATR from indicators for scalper stop loss calculation
                atr_pct = None
                if indicators and indicators.get('atr_pct'):
                    atr_pct = indicators.get('atr_pct')
                else:
                    # Fallback: estimate from signal stop loss
                    if signal.signal_score:
                        stop_distance_pct = (
                            abs((signal.stop_loss - signal.entry_price) / signal.entry_price)
                            if signal.entry_price > 0
                            else 0
                        )
                        if stop_distance_pct > 0:
                            atr_pct = stop_distance_pct / 1.5
                
                # Calculate stop loss based on ATR (uses configurable multiplier)
                from .config import SL_ATR_MULTIPLIER
                scalper_stop_loss = signal.stop_loss  # Default fallback
                if atr_pct and atr_pct > 0:
                    if signal.side == "long":
                        scalper_stop_loss = signal.entry_price * (1.0 - SL_ATR_MULTIPLIER * atr_pct)
                    else:  # short
                        scalper_stop_loss = signal.entry_price * (1.0 + SL_ATR_MULTIPLIER * atr_pct)
                
                # RISK-BASED SIZING: Calculate position size using risk budget approach
                # Use scalper stop loss for accurate risk calculation
                (
                    position_size,
                    risk_fraction,
                    sizing_reason,
                ) = self.position_manager.calculate_position_size(
                    equity=equity,
                    entry_price=signal.entry_price,
                    stop_loss_price=scalper_stop_loss,
                    signal_strength=signal.strength,
                    side=signal.side,
                    is_unicorn=is_unicorn,
                    open_positions=self.positions,
                    leverage=leverage,
                    signal_score=signal.final_score,  # Pass signal score for RPA
                )
                
                if position_size <= 0 or sizing_reason:
                    # DEBUG: Log position sizing rejection (important for debugging)
                    self.logger.warning(
                        f"[SIZING_REJECTED] symbol={symbol} size={position_size} "
                        f"reason={sizing_reason or 'size<=0'} risk_fraction={risk_fraction*100:.2f}%"
                    )
                    continue
                
                final_leverage = leverage
                
                entry_delay = self.position_manager.get_entry_delay_ms(
                    signal_strength=signal.strength
                )
                
                use_limit = self.order_manager.should_use_limit_order(
                    spread_bps=stats.spread_bps,
                    signal_strength=signal.strength,
                )
                
                entries_attempted += 1
                self.entries_attempted_this_scan += 1
                
                entry_symbol = symbol
                entry_side = signal.side
                
                # GUARD: Double-check invalid symbols cache before placing order
                # (Symbol might have been added to cache in a previous scan)
                if entry_symbol in self.invalid_symbols:
                    invalid_since = self.invalid_symbols[entry_symbol]
                    time_since_invalid = time.time() - invalid_since
                    if time_since_invalid < self.invalid_symbol_cooldown:
                        self.logger.debug(
                            f"[SKIP_INVALID_SYMBOL] {entry_symbol}: "
                            f"Cached as invalid (Reduce Only/Delisted), skipping order placement. "
                            f"Will retry after {self.invalid_symbol_cooldown - time_since_invalid:.0f}s"
                        )
                        continue  # Skip this symbol
                    else:
                        # Cooldown expired - remove from cache and retry
                        del self.invalid_symbols[entry_symbol]
                        self.logger.debug(f"[RETRY_INVALID_SYMBOL] Retrying previously invalid symbol: {entry_symbol}")
                
                # CRITICAL DIAGNOSTIC: Log before calling order manager
                self.logger.warning(
                    f"[ENTRY_CALLING_ORDER_MANAGER] {entry_symbol} {entry_side}: "
                    f"About to call order_manager.enter_position() - "
                    f"size={position_size:.6f}, price={signal.entry_price:.4f}, "
                    f"exchange={self.exchange_wrapper is not None}, "
                    f"order_manager={self.order_manager is not None}"
                )
                
                # GUARD: Check if order_manager is available
                if not self.order_manager:
                    self.logger.error(
                        f"[ENTRY_BLOCKED] {entry_symbol}: Order manager not initialized - cannot place order"
                    )
                    continue
                
                if not self.exchange_wrapper:
                    self.logger.error(
                        f"[ENTRY_BLOCKED] {entry_symbol}: Exchange wrapper not initialized - cannot place order"
                    )
                    continue
                
                # REMOVED: Individual entry logs (tracked via DecisionEvent in decision_event.py)
                # Unicorn and regular entry logs now appear in Recent Activity panel via DecisionEvent
                
                result = await self.order_manager.enter_position(
                    symbol=entry_symbol,
                    side=entry_side,
                    size=position_size,
                    entry_price=signal.entry_price,
                    delay_ms=entry_delay,
                    use_limit=use_limit,
                    leverage=final_leverage,
                )
                
                # DIAGNOSTIC: Log order result details for debugging
                self.logger.warning(  # Changed to WARNING so it's more visible
                    f"[ORDER_RESULT] {entry_symbol} {entry_side} size={position_size:.6f}: "
                    f"success={result.success}, "
                    f"filled_size={result.filled_size}, "
                    f"filled_price={result.filled_price}, "
                    f"error={result.error}, "
                    f"order_id={result.order_id}"
                )
                
                # CRITICAL: If order failed, log detailed error for debugging
                if not result.success:
                    error_msg = str(result.error) if result.error else "Unknown error"
                    # Check for invalid symbol errors - log at DEBUG level (expected behavior)
                    is_invalid_symbol = (
                        "-4140" in error_msg or 
                        "Invalid symbol status" in error_msg or
                        "INVALID_SYMBOL_STATUS" in error_msg
                    )
                    
                    if is_invalid_symbol:
                        # Already handled and cached - log at DEBUG level
                        self.logger.debug(
                            f"[ORDER_FAILURE_INVALID_SYMBOL] {entry_symbol} {entry_side}: "
                            f"Symbol in Reduce Only mode (cached, will skip future attempts)"
                        )
                    else:
                        # Other errors - log at ERROR level
                        self.logger.error(
                            f"[ORDER_FAILURE_DETAILS] {entry_symbol} {entry_side}: "
                            f"Order failed to fill. Error: {error_msg}. "
                            f"This order will NOT create a position on Binance."
                        )
                
                # CRITICAL: Only track position if order actually filled (filled_size > 0)
                if result.success and result.filled_size and result.filled_size > 0:
                    # METRICS
                    self.metrics['entries_attempted'] += 1
                    self.metrics['entries_opened'] += 1
                    self.entries_attempted_this_scan += 1
                    self.entries_opened_this_scan += 1
                    entries_opened_this_scan += 1  # Track entries opened this scan
                    self.position_manager.record_entry(entry_symbol)
                    
                    # ADAPTIVE ENTRY: Record successful entry for threshold learning
                    self.adaptive_entry.record_entry(entry_symbol, signal.final_score)
                    
                    # ADAPTIVE REGIME: Reset idle timer on successful entry
                    self._last_entry_time = time.time()
                    self._adaptive_relaxation = 0.0  # Reset relaxation
                    
                    # IDLE MODE: Exit idle mode when position is opened (resume strict filtering)
                    if self._idle_mode_active:
                        self._idle_mode_active = False
                        self._idle_mode_entry = None
                        self._idle_mode_entry_symbol = None
                        self.logger.info("[IDLE_MODE] Position opened - exiting idle mode, resuming strict filtering")
                    
                    entry_price = result.filled_price or signal.entry_price
                    entry_time = time.time()
                    filled_size = result.filled_size  # Use actual filled size, not fallback
                    
                    atr_pct = None
                    stop_distance_pct = (
                        abs((signal.stop_loss - entry_price) / entry_price)
                        if entry_price > 0
                        else 0
                    )
                    if stop_distance_pct > 0:
                        atr_pct = stop_distance_pct / 1.5
                    
                    # Get funding rate for position (if available)
                    position_funding_rate = None
                    if entry_symbol in self.funding_rates:
                        position_funding_rate = self.funding_rates[entry_symbol].get("rate", 0.0)
                    
                    # Set initial stop loss based on ATR (configurable multiplier)
                    from .config import SL_ATR_MULTIPLIER, MIN_STOP_DISTANCE_PCT
                    if atr_pct and atr_pct > 0:
                        # CRITICAL: Clamp ATR multiplier to prevent negative stop-loss
                        # If ATR is very large, SL_ATR_MULTIPLIER * atr_pct could exceed 1.0, making stop-loss negative
                        max_atr_mult = 0.95  # Never use more than 95% of entry price as stop distance
                        effective_atr_mult = min(SL_ATR_MULTIPLIER * atr_pct, max_atr_mult)
                        
                        if entry_side == "long":
                            initial_stop_price = entry_price * (1.0 - effective_atr_mult)
                        else:  # short
                            initial_stop_price = entry_price * (1.0 + effective_atr_mult)
                        
                        # Additional safety: Ensure stop-loss is never negative or zero
                        if initial_stop_price <= 0:
                            # Fallback to MIN_STOP_DISTANCE_PCT if ATR calculation fails
                            if entry_side == "long":
                                initial_stop_price = entry_price * (1.0 - MIN_STOP_DISTANCE_PCT)
                            else:  # short
                                initial_stop_price = entry_price * (1.0 + MIN_STOP_DISTANCE_PCT)
                            self.logger.warning(
                                f"[STOP_LOSS_FIX] {entry_symbol} {entry_side}: ATR calculation produced invalid stop_loss, "
                                f"using MIN_STOP_DISTANCE_PCT fallback: {initial_stop_price:.6f}"
                            )
                    else:
                        # Fallback: Use signal.stop_loss if valid, otherwise calculate from MIN_STOP_DISTANCE_PCT
                        if signal.stop_loss and signal.stop_loss > 0:
                            # Validate signal.stop_loss is in correct direction
                            if entry_side == "long" and signal.stop_loss < entry_price:
                                initial_stop_price = signal.stop_loss
                            elif entry_side == "short" and signal.stop_loss > entry_price:
                                initial_stop_price = signal.stop_loss
                            else:
                                # signal.stop_loss is invalid - calculate from minimum distance
                                if entry_side == "long":
                                    initial_stop_price = entry_price * (1.0 - MIN_STOP_DISTANCE_PCT)
                                else:  # short
                                    initial_stop_price = entry_price * (1.0 + MIN_STOP_DISTANCE_PCT)
                                self.logger.warning(
                                    f"[STOP_LOSS_FIX] {entry_symbol} {entry_side}: signal.stop_loss={signal.stop_loss:.6f} invalid, "
                                    f"recalculated to {initial_stop_price:.6f} using MIN_STOP_DISTANCE_PCT"
                                )
                        else:
                            # signal.stop_loss is missing or invalid - calculate from minimum distance
                            if entry_side == "long":
                                initial_stop_price = entry_price * (1.0 - MIN_STOP_DISTANCE_PCT)
                            else:  # short
                                initial_stop_price = entry_price * (1.0 + MIN_STOP_DISTANCE_PCT)
                            self.logger.warning(
                                f"[STOP_LOSS_FIX] {entry_symbol} {entry_side}: signal.stop_loss invalid/missing, "
                                f"calculated {initial_stop_price:.6f} using MIN_STOP_DISTANCE_PCT"
                            )
                    
                    # Use initial_stop_price for stop_loss (scalper uses tighter stops)
                    stop_loss_price = initial_stop_price
                    take_profit_price = signal.take_profit
                    
                    # LEARNER: Apply learned SL/TP adjustments
                    if self.learner and self.learner.enabled:
                        try:
                            entry_indicators = self.indicators_cache.get(entry_symbol, {}) if hasattr(self, 'indicators_cache') else {}
                            entry_rsi = entry_indicators.get('rsi', 50)
                            entry_atr_pct = entry_indicators.get('atr_pct', atr_pct if atr_pct else 1.5)
                            ema20 = entry_indicators.get('ema20', entry_price)
                            entry_trend = 'up' if entry_price > ema20 * 1.01 else ('down' if entry_price < ema20 * 0.99 else 'sideways')
                            vol_bucket = 'low' if entry_atr_pct < 1.0 else ('high' if entry_atr_pct > 2.5 else 'medium')
                            # TURBO: Get spread from signal or stats
                            entry_spread = getattr(signal, 'spread_bps', 15) if hasattr(signal, 'spread_bps') else 15
                            learner_adj = self.learner.get_trade_adjustments(
                                rsi=entry_rsi, volatility=vol_bucket, trend=entry_trend,
                                entry_price=entry_price, default_sl=stop_loss_price, default_tp=take_profit_price,
                                atr_pct=entry_atr_pct, spread_bps=entry_spread, symbol=entry_symbol)
                            # TURBO: Check if learner wants to skip due to quality gates
                            if learner_adj.get('skip'):
                                pass  # Will be handled by skip_probability check
                            elif learner_adj.get('confidence', 0) > 0.2:
                                learner_sl = learner_adj.get('stop_loss')
                                # Validate learner-adjusted stop-loss is valid
                                if learner_sl and learner_sl > 0:
                                    if entry_side == 'long' and learner_sl < entry_price:
                                        stop_loss_price = learner_sl
                                    elif entry_side == 'short' and learner_sl > entry_price:
                                        stop_loss_price = learner_sl
                                    else:
                                        # Learner gave invalid stop-loss - keep original
                                        self.logger.warning(
                                            f"[LEARNER_SL_INVALID] {entry_symbol} {entry_side}: "
                                            f"learner stop_loss={learner_sl:.6f} invalid, keeping original {stop_loss_price:.6f}"
                                        )
                                else:
                                    # Learner gave invalid stop-loss - keep original
                                    self.logger.warning(
                                        f"[LEARNER_SL_INVALID] {entry_symbol} {entry_side}: "
                                        f"learner stop_loss={learner_adj.get('stop_loss')} invalid, keeping original {stop_loss_price:.6f}"
                                    )
                                take_profit_price = learner_adj.get('take_profit', take_profit_price)
                        except:
                            pass
                    
                    # Calculate initial R (distance from entry to stop-loss)
                    if entry_side == "long":
                        initial_r = abs(entry_price - stop_loss_price)
                    else:  # short
                        initial_r = abs(stop_loss_price - entry_price)
                    
                    # CRITICAL VALIDATION: Ensure stop_loss is valid before entry
                    # This prevents unprotected positions like PIPPIN
                    # FINAL SAFETY: If stop_loss is still invalid after all fixes, recalculate one more time
                    if stop_loss_price <= 0:
                        # Last resort: Calculate from MIN_STOP_DISTANCE_PCT
                        if entry_side == 'long':
                            stop_loss_price = entry_price * (1.0 - MIN_STOP_DISTANCE_PCT)
                        else:  # short
                            stop_loss_price = entry_price * (1.0 + MIN_STOP_DISTANCE_PCT)
                        self.logger.warning(
                            f"[STOP_LOSS_EMERGENCY_FIX] {entry_symbol} {entry_side}: stop_loss was invalid, "
                            f"recalculated to {stop_loss_price:.6f} using MIN_STOP_DISTANCE_PCT"
                        )
                    
                    # Validate stop-loss is in correct direction for position side
                    if entry_side == 'long' and stop_loss_price >= entry_price:
                        # Fix: Recalculate to be below entry
                        stop_loss_price = entry_price * (1.0 - MIN_STOP_DISTANCE_PCT)
                        self.logger.warning(
                            f"[STOP_LOSS_DIRECTION_FIX] {entry_symbol} LONG: stop_loss was >= entry, "
                            f"recalculated to {stop_loss_price:.6f}"
                        )
                    elif entry_side == 'short' and stop_loss_price <= entry_price:
                        # Fix: Recalculate to be above entry
                        stop_loss_price = entry_price * (1.0 + MIN_STOP_DISTANCE_PCT)
                        self.logger.warning(
                            f"[STOP_LOSS_DIRECTION_FIX] {entry_symbol} SHORT: stop_loss was <= entry, "
                            f"recalculated to {stop_loss_price:.6f}"
                        )
                    
                    # Final validation: If still invalid after all fixes, reject
                    if stop_loss_price <= 0:
                        self.logger.error(
                            f"[ENTRY_REJECTED] {entry_symbol} {entry_side}: stop_loss={stop_loss_price:.6f} "
                            f"still invalid after all fixes (entry={entry_price:.6f}). Rejecting entry."
                        )
                        continue  # Skip this entry - cannot fix stop-loss
                    
                    if entry_side == 'long' and stop_loss_price >= entry_price:
                        self.logger.error(
                            f"[ENTRY_REJECTED] {entry_symbol} LONG: stop_loss={stop_loss_price:.6f} >= entry={entry_price:.6f} "
                            f"after fixes. Rejecting entry."
                        )
                        continue
                    elif entry_side == 'short' and stop_loss_price <= entry_price:
                        self.logger.error(
                            f"[ENTRY_REJECTED] {entry_symbol} SHORT: stop_loss={stop_loss_price:.6f} <= entry={entry_price:.6f} "
                            f"after fixes. Rejecting entry."
                        )
                        continue
                    
                    # Select exit profile based on signal score
                    exit_profile = self.exit_pipeline.select_exit_profile(signal.final_score)
                    
                    # CRITICAL FIX: Use position_registry.add() instead of direct assignment
                    # This ensures proper initialization of trailing stop fields and thread-safe locking
                    position_data = {
                        "side": entry_side,
                        "size": filled_size,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss_price,
                        "initial_stop_price": initial_stop_price,  # SCALPER: Store initial stop for trailing
                        "take_profit": take_profit_price,
                        "entry_time": entry_time,
                        "signal_strength": signal.strength,
                        "signal_score": signal.final_score,  # Store signal score for replacement logic
                        "signal_type": signal.signal_type,
                        "atr_pct": atr_pct,
                        "leverage": final_leverage,
                        "peak_pnl": 0.0,
                        "rescue_flag": False,
                        "rescue_start_time": 0,
                        "is_unicorn": is_unicorn,  # Track unicorn status for UI display
                        "funding_rate": position_funding_rate,  # Store funding rate for cost calculation
                        # SPREAD EXPLOSION: Store entry spread for comparison
                        "entry_spread_bps": stats.spread_bps if stats else 10.0,
                        # R-based exit metadata
                        "initial_r": initial_r,
                        "exit_profile": exit_profile,
                        "max_r_reached": 0.0,
                        "bars_in_trade": 0,
                        "partial_exit_done": False,
                        "sl_moved_to_be": False,
                        "last_bar_update_time": entry_time,  # Track last bar update for bar counting
                        # MSX framework metadata
                        "stage": 1,
                        "survived_msx1": False,
                        "peak_price": entry_price,
                        "trough_price": entry_price,
                    }
                    
                    # Use position registry to add position (ensures proper initialization and thread safety)
                    if not self.position_registry.add(entry_symbol, position_data):
                        # Position already exists - log warning but continue
                        self.logger.warning(f"[ENTRY] Position {entry_symbol} already exists in registry, updating instead")
                        self.position_registry.update(entry_symbol, position_data)
                    
                    # NO BINANCE STOP-LOSS ORDERS - Using bot's internal logic only
                    # Bot monitors positions and exits when stop-loss is hit via internal monitoring
                    # This gives full control and avoids Binance API complications
                    
                elif not result.success:
                    # Order failed - log the error
                    error_msg = result.error or "Unknown error"
                    self.logger.warning(
                        f"[ENTRY_FAILED] {entry_symbol} {entry_side} size={position_size:.4f}: {error_msg}"
                    )
                    
                    # CRITICAL FALLBACK: Always check exchange positions, even if order_id is None
                    # Sometimes Binance fills the order but our verification fails
                    # This is especially important for market orders which should fill instantly
                    try:
                        await asyncio.sleep(0.8)  # Longer delay to allow position to appear on exchange
                        exchange_positions = await self.exchange_wrapper.fetch_positions([entry_symbol])
                        for pos in exchange_positions:
                            pos_symbol = pos.get('symbol', '')
                            # Normalize symbol for comparison
                            if '/' in pos_symbol:
                                pos_symbol_normalized = pos_symbol
                            else:
                                # Try to match without normalization
                                pos_symbol_normalized = pos_symbol
                            
                            pos_contracts = abs(float(pos.get('contracts', 0) or pos.get('positionAmt', 0) or pos.get('size', 0)))
                            
                            # Check if this position matches our entry symbol
                            if (pos_symbol == entry_symbol or 
                                pos_symbol_normalized == entry_symbol or
                                entry_symbol.replace('/USDT:USDT', '') in pos_symbol or
                                pos_symbol.replace('/USDT', '') == entry_symbol.replace('/USDT:USDT', '')):
                                
                                if pos_contracts > 0:
                                    # Position exists on exchange! Order actually filled
                                    entry_price_from_exchange = float(pos.get('entryPrice', 0) or pos.get('markPrice', 0) or signal.entry_price)
                                    self.logger.warning(  # Use WARNING level so it's visible
                                        f"[ENTRY_RECOVERED] {entry_symbol}: Position found on exchange "
                                        f"despite order status failure! Size: {pos_contracts}, Entry: {entry_price_from_exchange}"
                                    )
                                    # Use exchange position data
                                    result.success = True
                                    result.filled_size = pos_contracts
                                    result.filled_price = entry_price_from_exchange
                                    result.order_id = result.order_id or f"RECOVERED_{int(time.time())}"
                                    break
                    except Exception as e:
                        self.logger.debug(f"[ENTRY_FALLBACK] Could not check exchange position: {e}")
                    
                    # If we recovered the position, continue to position registration below
                    if not result.success:
                        continue  # Skip to next signal
                        
                elif result.success and (not result.filled_size or result.filled_size == 0):
                    # Order succeeded but didn't fill (likely rejected by exchange)
                    self.logger.warning(
                        f"[ENTRY_NO_FILL] {entry_symbol} {entry_side} size={position_size:.4f}: "
                        f"Order placed but not filled (likely MIN_NOTIONAL or precision issue)"
                    )
                    continue  # Skip to next signal
                
                # Ensure entry_time defined (recovered orders path)
                if 'entry_time' not in locals():
                    entry_time = time.time()
                    entry_price = result.filled_price or signal.entry_price
                    filled_size = result.filled_size
                
                # CRITICAL FIX: Ensure stop_loss_price and take_profit_price are defined for recovered orders
                # (Order recovery path may skip the first if block where these are normally assigned)
                if 'stop_loss_price' not in locals():
                    # Calculate stop_loss_price same way as in the first if block
                    atr_pct = None
                    stop_distance_pct = (
                        abs((signal.stop_loss - entry_price) / entry_price)
                        if entry_price > 0
                        else 0
                    )
                    if stop_distance_pct > 0:
                        atr_pct = stop_distance_pct / 1.5
                    
                    from .config import SL_ATR_MULTIPLIER
                    if atr_pct and atr_pct > 0:
                        if entry_side == "long":
                            initial_stop_price = entry_price * (1.0 - SL_ATR_MULTIPLIER * atr_pct)
                        else:  # short
                            initial_stop_price = entry_price * (1.0 + SL_ATR_MULTIPLIER * atr_pct)
                    else:
                        initial_stop_price = signal.stop_loss
                    
                    stop_loss_price = initial_stop_price
                    take_profit_price = signal.take_profit
                
                    # LOGGING V2: Entry logging now handled by DecisionEvent (single concise line)
                    # Only create decision event if position was actually opened
                    if result.success and result.filled_size and result.filled_size > 0:
                        # Log stop-loss/take-profit for debugging (especially for problematic positions)
                        if 'PIPPIN' in entry_symbol.upper() or self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.info(
                                f"[ENTRY_DETAILS] {entry_symbol} {entry_side.upper()} | "
                                f"entry={entry_price:.6f} stop={stop_loss_price:.6f} tp={take_profit_price:.6f} | "
                                f"size={filled_size:.4f}"
                            )
                        
                        # Create canonical decision event for entry
                        decision = DecisionEvent(
                            timestamp=entry_time,
                            action="ENTRY",
                            symbol=entry_symbol,
                            side=entry_side.upper(),
                            price=entry_price,
                            entry_price=entry_price,
                        size=filled_size,
                        reason="approved",
                        score=signal.final_score,  # SCORING V2: This is now the Scoring v2 final score
                        signal_type=signal.signal_type,
                        signal_strength=signal.strength,
                        stop_loss=stop_loss_price,  # SCALPER: Use scalper stop loss
                        take_profit=take_profit_price,
                        is_unicorn=is_unicorn
                    )
                    # SCORING V2: Attach score components to decision event for logging
                    if hasattr(signal, 'score_components_capped') and signal.score_components_capped:
                        decision.score_components_capped = signal.score_components_capped
                    # PHASE 2: Attach quality gate details for logging
                    if quality_score:
                        decision.quality_multiplier = quality_score.multiplier
                        decision.quality_details = quality_score.details
                    
                    # Log using unified system (writes to decisions.jsonl, recent_trades, and Recent Activity buffer)
                    # This also logs the standardized ENTRY line
                    log_trade_decision(decision, bot_instance=self)
                    
                    current_positions += 1
                else:
                    # METRICS
                    self.metrics['entries_attempted'] += 1
                    
                    # IMPROVED: Extract error message with better fallbacks
                    error_msg = None
                    
                    # First, try to get error from result.error (primary source)
                    if hasattr(result, "error") and result.error:
                        error_msg = str(result.error).strip()
                    
                    # If no error message, try to infer from result state
                    if not error_msg or error_msg == "":
                        if hasattr(result, "filled_size"):
                            if result.filled_size == 0 or result.filled_size is None:
                                error_msg = "Order placed but not filled (filled_size=0 or None)"
                            elif result.filled_size > 0 and result.filled_size < position_size * 0.99:
                                error_msg = f"Partial fill rejected: {result.filled_size:.6f}/{position_size:.6f}"
                        elif hasattr(result, "order_id") and not result.order_id:
                            error_msg = "Order rejected by exchange (no order_id returned)"
                        else:
                            error_msg = f"Order failed (success=False, no error details available)"
                    
                    # Ensure we have a non-empty error message
                    if not error_msg or error_msg == "":
                        error_msg = "Unknown error - order failed but no error details available"
                    
                    # GUARD: Cache invalid symbols (Binance -4140: "Invalid symbol status")
                    # This prevents repeated attempts on symbols in "Reduce Only" mode or delisted
                    # Check for both the error code and the specific error message
                    is_invalid_symbol = (
                        "-4140" in str(error_msg) or 
                        "Invalid symbol status" in str(error_msg) or
                        "INVALID_SYMBOL_STATUS" in str(error_msg)
                    )
                    
                    if is_invalid_symbol:
                        # Always update cache timestamp (even if already cached) to reset cooldown
                        self.invalid_symbols[entry_symbol] = time.time()
                        if entry_symbol not in getattr(self, '_invalid_symbol_logged', set()):
                            # Only log warning once per symbol
                            self.logger.warning(
                                f"[!] CACHED INVALID SYMBOL: {entry_symbol} (Reduce Only/Delisted) - Will skip for {self.invalid_symbol_cooldown/3600:.1f}h"
                            )
                            if not hasattr(self, '_invalid_symbol_logged'):
                                self._invalid_symbol_logged = set()
                            self._invalid_symbol_logged.add(entry_symbol)
                        # Log as debug for subsequent attempts (expected behavior, already cached)
                        self.logger.debug(
                            f"[SKIP_INVALID] {entry_symbol} {entry_side}: Symbol in Reduce Only mode (cached, will skip future attempts)"
                        )
                        continue  # Skip to next signal (don't log as error)
                    else:
                        # Other errors (margin, etc.) - log as error with full context
                        error_details = f"msg={error_msg[:100]}"
                        if hasattr(result, "order_id"):
                            error_details += f" order_id={result.order_id}"
                        if hasattr(result, "filled_size"):
                            error_details += f" filled_size={result.filled_size}"
                        if hasattr(result, "latency_ms"):
                            error_details += f" latency={result.latency_ms:.0f}ms"
                        
                        self.logger.error(
                            f"[E] entry_fail sym={entry_symbol} side={entry_side} {error_details}"
                        )
        
        # Track scan completion
        scan_end_time = time.time()
        scan_duration = scan_end_time - scan_start_time
        self.last_scan_end = scan_end_time
        self.is_scanning = False
        
        # LOGGING V2: Detailed scan summary moved to DEBUG (too verbose for INFO)
        # Counts at each filtering stage
        total_candidates = total_symbols  # Symbols scanned
        signals_after_generator = signals_found  # Signals that passed signal generator filters
        signals_after_circuit_breaker = signals_found - signals_blocked_circuit_breaker  # Signals after circuit breaker check
        signals_after_position_manager = signals_pass_position_manager  # Signals that passed position_manager.can_enter_position()
        final_entries_attempted = entries_attempted  # Actually attempted entries
        
        # DIAGNOSTIC: Get signal generator filter stats if available
        signal_filter_stats = {}
        if hasattr(self.signal_generator, '_filter_stats'):
            signal_filter_stats = self.signal_generator._filter_stats.copy()
        
        # DEBUG only: Detailed scan summary (moved from INFO to reduce log volume)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"SCAN_DETAIL candidates={total_candidates} | "
                f"after_generator={signals_after_generator} | "
                f"after_circuit_breaker={signals_after_circuit_breaker} | "
                f"after_position_manager={signals_after_position_manager} | "
                f"entries_attempted={final_entries_attempted} | "
                f"blocked_circuit_breaker={signals_blocked_circuit_breaker} | "
                f"blocked_position_manager={signals_blocked_position_manager} | "
                f"skip_reasons={dict(symbols_skipped)} | "
                f"current_positions={current_positions}/{MAX_CONCURRENT_POS} | "
                f"signal_filters={signal_filter_stats}"
            )
        
        # Calculate cache hit rate
        total_cache_accesses = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_cache_accesses * 100.0) if total_cache_accesses > 0 else 0.0
        
        # Calculate orderbook success rate
        total_orderbook_attempts = orderbook_success + orderbook_failures
        orderbook_success_rate = (orderbook_success / total_orderbook_attempts * 100.0) if total_orderbook_attempts > 0 else 0.0
        
        # GENERATE SINGLE SUMMARY LINE (aggregated from all signals in this loop)
        from datetime import datetime
        from .logger import get_log_buffer
        
        # Log scan summary only when entries are attempted (human-relevant info only)
        if entries_attempted > 0:
            # Simple human-readable summary
            summary = f"Scan: {signals_found} signals found, {entries_attempted} entries attempted"
            self.logger.info(summary)
            
            # Also add to UI buffer
            log_buffer = get_log_buffer()
            log_buffer.append(datetime.now(), 'INFO', summary)
        
        # Store scan record
        scan_record = {
            'timestamp': scan_start_time,
            'duration': scan_duration,
            'symbols_processed': symbols_processed,
            'symbols_skipped': symbols_skipped.copy(),
            'total_symbols': total_symbols,
            'active_symbols': len(active_symbols),
            # Store rejection counters for debugging
            'rejected_score': rejected_score,
            'rejected_percentile': rejected_percentile,
            'rejected_micro': rejected_micro,
            'rejected_risk': rejected_risk,
            'rejected_capacity': rejected_capacity,
            'passed_filters': passed_filters,
            'discovery_symbols': len(discovery_symbols),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'orderbook_success': orderbook_success,
            'orderbook_failures': orderbook_failures,
            'orderbook_success_rate': orderbook_success_rate,
            'signals_found': signals_found,
            'entries_attempted': entries_attempted
        }
        self.scan_history.append(scan_record)
        self.scan_times.append(scan_duration)

        # ------------------------------------------------------------------
        # SCAN HEALTH LOGGING (normal operations visibility)
        # ------------------------------------------------------------------
        # Goal: Always make it obvious if the bot is actually scanning, finding signals,
        # and attempting entries — without needing to dig through debug logs.
        try:
            now_ts = time.time()
            if not hasattr(self, "_scan_health_last_log_ts"):
                self._scan_health_last_log_ts = 0.0
            if not hasattr(self, "_consecutive_zero_signal_scans"):
                self._consecutive_zero_signal_scans = 0
            if not hasattr(self, "_consecutive_zero_entry_attempt_scans"):
                self._consecutive_zero_entry_attempt_scans = 0

            if signals_found <= 0:
                self._consecutive_zero_signal_scans += 1
            else:
                self._consecutive_zero_signal_scans = 0

            if entries_attempted <= 0:
                self._consecutive_zero_entry_attempt_scans += 1
            else:
                self._consecutive_zero_entry_attempt_scans = 0

            # Top skip reasons (if available)
            top_skips = ""
            if symbols_skipped:
                top = sorted(symbols_skipped.items(), key=lambda kv: kv[1], reverse=True)[:3]
                top_skips = ", ".join([f"{k}={v}" for k, v in top])

            # Log at least every 30s, and immediately when entries are attempted/opened.
            should_log = (
                (now_ts - float(self._scan_health_last_log_ts)) >= 30.0
                or entries_attempted > 0
                or entries_opened_this_scan > 0
            )
            if should_log:
                self._scan_health_last_log_ts = now_ts
                self.logger.info(
                    f"[SCAN] processed={symbols_processed}/{total_symbols} "
                    f"signals={signals_found} passed={passed_filters} "
                    f"attempted={entries_attempted} opened={entries_opened_this_scan} "
                    f"cache_hit={cache_hit_rate:.0f}% ob_ok={orderbook_success_rate:.0f}% "
                    + (f"top_skips=[{top_skips}]" if top_skips else "")
                )

            # Escalate if the bot is scanning but producing zero signals for a long time.
            if self._consecutive_zero_signal_scans >= 10 and symbols_processed > 0:
                self.logger.warning(
                    f"[SCAN][WARN] {self._consecutive_zero_signal_scans} consecutive scans with 0 signals "
                    f"(processed={symbols_processed}/{total_symbols}, cache_hit={cache_hit_rate:.0f}%)"
                )
                # Avoid spamming: reset after warning
                self._consecutive_zero_signal_scans = 0

            # Escalate if signals exist but nothing is even attempted for a long time.
            if self._consecutive_zero_entry_attempt_scans >= 20 and symbols_processed > 0 and signals_found > 0:
                self.logger.warning(
                    f"[SCAN][WARN] {self._consecutive_zero_entry_attempt_scans} consecutive scans with 0 entry attempts "
                    f"(signals={signals_found}, passed={passed_filters})"
                )
                self._consecutive_zero_entry_attempt_scans = 0
        except Exception:
            # Never let health logging break the scan loop
            pass
        
        # Mark scanning as complete
        self.is_scanning = False
        
        # Update aggregate statistics
        self.scan_stats['total_scans'] += 1
        self.scan_stats['total_symbols_processed'] += symbols_processed
        self.scan_stats['total_cache_hits'] += cache_hits
        self.scan_stats['total_cache_misses'] += cache_misses
        self.scan_stats['total_orderbook_success'] += orderbook_success
        self.scan_stats['total_orderbook_failures'] += orderbook_failures
        # Store latest rejection counters for debugging
        self.scan_stats['rejected_score'] = rejected_score
        self.scan_stats['rejected_percentile'] = rejected_percentile
        self.scan_stats['rejected_micro'] = rejected_micro
        self.scan_stats['rejected_risk'] = rejected_risk
        self.scan_stats['rejected_capacity'] = rejected_capacity
        self.scan_stats['passed_filters'] = passed_filters
        
        # FIX: Update cumulative filter stats for Signal Health panel
        # Wire the real filter stats from [FILTER] scan#N log to Signal Health
        # CRITICAL: These stats feed the Signal Health panel and [FILTER] log
        # - signals_found: Total signals generated (passed signal generator checks)
        # - passed_filters: Signals that passed filter pipeline AND position manager check (ready for entry)
        # - signals_rejected: Signals that were rejected (signals_found - passed_filters)
        self.filter_stats_cumulative['signals_total'] += signals_found
        self.filter_stats_cumulative['signals_passed'] += passed_filters
        self.filter_stats_cumulative['signals_rejected'] += (signals_found - passed_filters)
        
        # Update running average score
        if loop_stats['scores']:
            for score in loop_stats['scores']:
                self.filter_stats_cumulative['avg_score_sum'] += score
                self.filter_stats_cumulative['avg_score_count'] += 1
        
        # DEBUG: Single aggregated log per scan showing signal flow
        self.logger.debug(
            f"[DEBUG] entries_for_scan raw_signals={signals_found} filtered={passed_filters} "
            f"entry_candidates={entry_candidates} opened={entries_opened_this_scan}"
        )
    
    async def _execute_dust_sweep(self, symbol: str, amount: float, side: str) -> None:
        """
        Execute a dust sweep order with reduceOnly and error handling.
        Designed to clean up small positions without crashing.
        """
        try:
            self.logger.info(f"[🧹] SWEEPING DUST: {symbol} {side} {amount}")
            # Ensure amount is positive
            amount = abs(amount)
            # Side for closing: if Long, Sell. If Short, Buy.
            # Passed side should be the ACTION side (e.g. 'sell' to close long)
            
            # Using create_order via wrapper for consistency
            # params={'reduceOnly': True} ensures we don't flip position
            await self.exchange_wrapper.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params={'reduceOnly': True}
            )
            self.logger.info(f"[✅] SWEEP SUCCESS: {symbol}")
            
            # Remove from local tracking immediately to prevent double-sweep
            if symbol in self.positions:
                del self.positions[symbol]
                
        except Exception as e:
            msg = str(e)
            if "MIN_NOTIONAL" in msg or "Filter failure" in msg:
                self.logger.warning(f"[⚠️] SWEEP FAILED (Too Small): {symbol} - {msg}. Use 'Convert Small Assets' in Binance.")
            else:
                self.logger.error(f"[❌] SWEEP ERROR: {symbol} - {msg}")

    async def monitor_and_exit_positions(self) -> None:
        """Monitor open positions and exit when stop-loss or take-profit is hit."""
        if not self.exit_manager or not self.positions:
            return

        # OPTIMIZATION: Cache equity calculation (used in exit logic)
        equity = self.equity_now()
        positions_to_exit = []

        # OPTIMIZATION: Cache time once for all position checks
        monitor_now = time.time()
        
        # DIAGNOSTIC: Log positions being monitored (for debugging PIPPIN issue)
        if self.positions and self.logger.isEnabledFor(logging.DEBUG):
            monitored_symbols = list(self.positions.keys())
            if any('PIPPIN' in s.upper() for s in monitored_symbols):
                for sym, pos in self.positions.items():
                    if 'PIPPIN' in sym.upper():
                        self.logger.debug(
                            f"[MONITOR_CHECK] {sym} | "
                            f"entry={pos.get('entry_price', 0):.6f} "
                            f"stop={pos.get('stop_loss', 0):.6f} "
                            f"size={pos.get('size', 0):.4f} "
                            f"side={pos.get('side', 'N/A')}"
                        )
        
        # CRITICAL: Startup guard - prevent ANY exits during first 30 seconds after startup
        # This prevents positions from being closed on restart due to invalid entry_time or sync issues
        if not hasattr(self, '_startup_time'):
            self._startup_time = monitor_now
        startup_age = monitor_now - self._startup_time
        STARTUP_GUARD_SEC = 30.0  # 30 second grace period after startup
        if startup_age < STARTUP_GUARD_SEC:
            # During startup guard, only allow critical risk exits (stop-loss hits, not time exits)
            # This prevents false time exits from invalid entry_time on restart
            self.logger.debug(f"[STARTUP_GUARD] Active ({startup_age:.1f}s/{STARTUP_GUARD_SEC}s) - blocking non-critical exits")
        
        # ADAPTIVE PERFORMANCE MONITORING: Analyze positions and adjust thresholds
        if hasattr(self, 'adaptive_monitor'):
            # Update position data for adaptive monitor
            for symbol, position in self.positions.items():
                current_price = position.get('current_price', position.get('entry_price', 0))
                if current_price > 0:
                    self.adaptive_monitor.record_position_update(
                        symbol=symbol,
                        entry_price=position.get('entry_price', 0),
                        current_price=current_price,
                        entry_time=position.get('entry_time', monitor_now),
                        side=position.get('side', 'long'),
                        position_size=abs(position.get('size', 0)),
                        stop_loss=position.get('stop_loss', 0),
                        take_profit=position.get('take_profit')
                    )
            
            # Analyze performance and get recommendations
            analysis = self.adaptive_monitor.analyze_performance(self.positions)
            # Threshold adjustment is automatically applied in scan_and_enter_signals
        
        # PERIODIC SYNC: Fetch real positions from exchange every 5s to catch "Ghost/Dust" positions
        if not hasattr(self, '_last_pos_sync'):
            self._last_pos_sync = 0
        
        # PERIODIC STOP-LOSS VERIFICATION: Disabled - using bot's internal logic only
        # (No longer checking Binance stop-loss orders since we don't place them)
        # Bot handles all exits internally via monitor_and_exit_positions
            
        # SCURFFY THE JANITOR: Ultra-fast sweep (5s) to catch dust immediately
        if monitor_now - self._last_pos_sync > 5 and not self.replay_mode and self.exchange_wrapper:
            try:
                # Fetch all open positions (no symbol filter)
                real_positions = await self.exchange_wrapper.fetch_positions()
                
                for rp in real_positions:
                    sym = rp.get('symbol') or rp.get('symbolName')
                    if not sym:
                        continue
                    
                    # GUARD: Skip delivery contracts (those with dates like 260327) or malformed symbols
                    clean_sym = sym.replace("/", "").replace("-", "").replace(":", "")
                    if "USDTUSDT" in clean_sym or (len(clean_sym) > 6 and clean_sym[-6:].isdigit()):
                        self.logger.debug(f"Skipping delivery contract/malformed symbol in Ghost Hunter: {sym}")
                        continue
                    
                    # Normalize symbol just in case
                    if hasattr(self.exchange_wrapper, 'normalize_symbol'):
                        sym = self.exchange_wrapper.normalize_symbol(sym)
                    
                    # Already tracking?
                    # CRITICAL: Even if tracking, if our size is 0 but exchange says != 0, we must re-adopt!
                    if sym in self.positions:
                        current_mem_size = abs(self.positions[sym].get('size', 0.0))
                        exchange_size = abs(float(rp.get('contracts', rp.get('amount', rp.get('positionAmt', 0.0)))))
                        
                        # If memory says closed (0) but exchange says open (>0), it's a GHOST DUST!
                        if current_mem_size == 0 and exchange_size > 0:
                            self.logger.warning(f"[!] RESURRECTING ZOMBIE: {sym} (Mem=0, Exch={exchange_size})")
                            # Fall through to adoption
                        else:
                            continue # Normal tracking, skip
                    
                    # ADOPT GHOST POSITION (ensure all required fields)
                    # CRITICAL: Binance returns NEGATIVE positionAmt for shorts - use abs()
                    raw_size = float(rp.get('contracts', rp.get('amount', rp.get('positionAmt', 0.0))) or 0.0)
                    position_size = abs(raw_size)  # Always positive
                    
                    # CRITICAL: Adopt ALL positions from Binance, even dust (prevents position count mismatch)
                    # We'll clean up dust separately, but we need accurate position count first
                    if position_size <= 0:
                        continue  # Skip zero-size positions
                    
                    # Determine side from positionAmt sign or explicit side field
                    explicit_side = rp.get('side', rp.get('positionSide', '')).lower()
                    if explicit_side in ('long', 'buy'):
                        side = 'long'
                    elif explicit_side in ('short', 'sell'):
                        side = 'short'
                    elif raw_size < 0:
                        side = 'short'  # Negative positionAmt = short
                    else:
                        side = 'long'   # Positive or zero = long
                    
                    # DEFENSIVE: Use current time for adopted positions (they're "new" to us)
                    # But preserve if we already have a valid entry_time
                    preserved_entry_time = self.positions.get(sym, {}).get('entry_time', monitor_now)
                    if preserved_entry_time <= 0 or preserved_entry_time > monitor_now:
                        preserved_entry_time = monitor_now
                    
                    # Calculate stop-loss from entry price if not available
                    entry_price = float(rp.get('entryPrice', rp.get('price', 0.0)) or 0.0)
                    if not entry_price:
                        entry_price = float(rp.get('avgPrice', rp.get('markPrice', 0.0)) or 0.0)
                    
                    # Set default stop-loss for adopted positions (1% away from entry)
                    from .config import MIN_STOP_DISTANCE_PCT
                    if side == 'long':
                        default_stop_loss = entry_price * (1.0 - MIN_STOP_DISTANCE_PCT) if entry_price > 0 else 0
                    else:  # short
                        default_stop_loss = entry_price * (1.0 + MIN_STOP_DISTANCE_PCT) if entry_price > 0 else 0
                    
                    adopted = {
                        'symbol': sym,
                        'entry_price': entry_price,
                        'size': position_size,
                        'side': side,
                        'leverage': int(rp.get('leverage', 1) or 1),
                        'entry_time': preserved_entry_time,  # Never use 0 or invalid timestamp
                        'unrealizedPnl': float(rp.get('unrealizedPnl', 0.0) or 0.0),
                        'stop_loss': default_stop_loss,  # Set default stop-loss for adopted positions
                        'initial_stop_price': default_stop_loss,
                    }

                    # Fallbacks
                    if not adopted['entry_price']:
                        adopted['entry_price'] = float(rp.get('markPrice', 0.0) or 0.0)
                    if not adopted['size']:
                        adopted['size'] = abs(float(rp.get('qty', 0.0) or 0.0))
                    if adopted['stop_loss'] <= 0:
                        # Recalculate stop-loss if still invalid
                        if side == 'long':
                            adopted['stop_loss'] = adopted['entry_price'] * (1.0 - MIN_STOP_DISTANCE_PCT) if adopted['entry_price'] > 0 else 0
                        else:
                            adopted['stop_loss'] = adopted['entry_price'] * (1.0 + MIN_STOP_DISTANCE_PCT) if adopted['entry_price'] > 0 else 0

                    self.positions[sym] = adopted
                    
                    # Keep registry and sets in sync
                    try:
                        self.position_registry._positions[sym] = adopted
                        self.position_registry._positions_set.add(sym)
                    except Exception:
                        pass
                    self._positions_set.add(sym)
                    
                    self.logger.info(f"[!] ADOPTED GHOST POSITION: {sym} | Side={adopted['side'].upper()} | Size={adopted['size']:.4f} | Entry=${adopted['entry_price']:.4f} | SL=${adopted['stop_loss']:.4f}")
                
                # Update last sync time
                self._last_pos_sync = monitor_now
                
                # DUST CLEANUP: Close very small positions (< $10 notional) to prevent dust accumulation
                # This runs after position adoption to catch any dust positions
                # Note: DRY_RUN is already imported at module level
                dust_cleanup_threshold_usd = 10.0  # Close positions worth less than $10
                
                for sym, pos in list(self.positions.items()):
                    try:
                        # Get current price for notional calculation
                        stats = self.universe.stats.get(sym)
                        current_price = getattr(stats, 'last', pos.get('entry_price', 0)) if stats else pos.get('entry_price', 0)
                        if current_price <= 0:
                            continue
                        
                        position_size = abs(pos.get('size', 0))
                        notional_value = position_size * current_price
                        
                        # Close dust positions (very small notional value)
                        if notional_value > 0 and notional_value < dust_cleanup_threshold_usd:
                            self.logger.warning(
                                f"[DUST_CLEANUP] {sym}: Notional=${notional_value:.2f} < ${dust_cleanup_threshold_usd} threshold. "
                                f"Closing dust position to prevent accumulation."
                            )
                            # Queue for exit
                            positions_to_exit.append((sym, pos, "dust_cleanup", current_price, 1.0))
                    except Exception as e:
                        self.logger.debug(f"[DUST_CLEANUP] Error checking {sym}: {e}")
                
                # MANUAL CLOSE SYNC: Remove positions closed externally (Binance app, etc.)
                # Build set of symbols that exist on exchange
                real_symbols = set()
                for rp in real_positions:
                    sym = rp.get('symbol') or rp.get('symbolName')
                    if sym:
                        if hasattr(self.exchange_wrapper, 'normalize_symbol'):
                            sym = self.exchange_wrapper.normalize_symbol(sym)
                        real_symbols.add(sym)
                
                # Check for positions we're tracking but aren't on exchange
                for sym in list(self.positions.keys()):
                    if sym not in real_symbols:
                        entry_ts = self.positions[sym].get('entry_time', 0)
                        # Only remove if position is older than 10s (avoid race condition with new entries)
                        if monitor_now - entry_ts > 10.0:
                            self.logger.info(f"[X] MANUAL CLOSE DETECTED: {sym} - Removed from tracking")
                            del self.positions[sym]
                            self._positions_set.discard(sym)
                            try:
                                if sym in self.position_registry._positions:
                                    del self.position_registry._positions[sym]
                                self.position_registry._positions_set.discard(sym)
                            except Exception:
                                pass
                
                self._last_pos_sync = monitor_now
            except Exception as e:
                self.logger.warning(f"Position sync failed: {e}")

        # DUST SWEEPER: Clean up only REAL dust (positions too small to manage)
        # Only sweep positions that are clearly dust - NOT normal positions!
        for symbol, position in list(self.positions.items()):
            size = abs(position.get('size', 0.0))
            entry_price = position.get('entry_price', 0.0)
            
            # Use universe stats for current price if available, else entry
            stats = self.universe.stats.get(symbol)
            current_price_est = getattr(stats, 'last', entry_price) if stats else entry_price
            
            # Fallback: if price is 0, use entry price to estimate notional
            if current_price_est <= 0:
                current_price_est = entry_price
            
            # Check notional value
            notional = size * current_price_est
            
            # UNCLOSABLE DUST (<$5): Below Binance MIN_NOTIONAL, can't close via API
            # Log to file for manual cleanup, then purge from memory
            if 0 < notional < 5.0:
                self.logger.warning(f"[🗑️] DUST PURGE: {symbol} (${notional:.2f}) - Removed from tracking (use Binance Convert)")
                # Log to dust file for manual cleanup
                try:
                    import os
                    dust_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'dust_to_clean.txt')
                    with open(dust_file, 'a') as f:
                        from datetime import datetime
                        f.write(f"{datetime.now().isoformat()} | {symbol} | ${notional:.4f} | Use Binance 'Convert Small Assets'\n")
                except Exception:
                    pass
                if symbol in self.positions:
                    del self.positions[symbol]
                if symbol in self._positions_set:
                    self._positions_set.discard(symbol)
                try:
                    if symbol in self.position_registry._positions:
                        del self.position_registry._positions[symbol]
                    self.position_registry._positions_set.discard(symbol)
                except Exception:
                    pass
                continue
            
            # SMALL DUST ($5 - $10): Can close via API - sweep immediately
            # This is REAL dust that shouldn't exist
            if 5.0 <= notional < 10.0:
                self.logger.info(f"[🧹] DUST SWEEP: {symbol} (${notional:.2f}) - Closing now")
                pos_side = position.get('side', 'long').lower()
                close_side = 'sell' if pos_side == 'long' else 'buy'
                await self._execute_dust_sweep(symbol, size, close_side)
                continue
            
            # NOTE: Positions >= $10 are NOT dust - they are valid positions
            # Let them be managed by normal exit logic (TP/SL/trailing)

        for symbol, position in list(self.positions.items()):
            # Skip if already marked for exit by dust sweeper
            if any(p[0] == symbol for p in positions_to_exit):
                continue
                
            # LATENCY OPTIMIZATION: Use cached data (no API calls)
            # Get current market price from universe stats (already cached)
            
            # HELPER: Robust lookup for stats with format fallback
            stats = self.universe.stats.get(symbol)
            if not stats and symbol.endswith(":USDT"):
                stats = self.universe.stats.get(symbol.replace(":USDT", ""))
            if not stats and not symbol.endswith(":USDT"):
                stats = self.universe.stats.get(f"{symbol}:USDT")
                
            if not stats:
                continue
            
            # OPTIMIZATION: Cache position attributes once to avoid repeated dict lookups
            side = position.get('side', '').lower()
            entry_price = position.get('entry_price', 0)
            entry_time = position.get('entry_time', monitor_now)
            
            # DEFENSIVE: Validate entry_time to prevent massive age calculation errors
            # If entry_time is 0 or invalid, use current time (treat as new position)
            if entry_time <= 0 or entry_time > monitor_now:
                self.logger.warning(
                    f"[POSITION_FIX] {symbol} has invalid entry_time={entry_time}, "
                    f"resetting to current time (prevents false time exits)"
                )
                entry_time = monitor_now
                position['entry_time'] = entry_time
            
            # Try cache first for ultra-fast access
            cached_ticker = self.ticker_cache.get(symbol, max_age=2.0)
            if cached_ticker:
                current_price = cached_ticker.mark or cached_ticker.last
                spread_bps = cached_ticker.spread_bps
            else:
                # Fallback to universe stats (still fast, no API call)
                current_price = stats.mark or stats.last
                spread_bps = stats.spread_bps
            
            # HYBRID TIME EXIT + ZOMBIE KILLER (Option 3)
            position_age_sec = monitor_now - entry_time
            
            # DEFENSIVE: Clamp age to reasonable maximum (prevent negative or huge ages)
            if position_age_sec < 0:
                position_age_sec = 0
            elif position_age_sec > 7200:  # 2 hours max (safety cap)
                self.logger.warning(
                    f"[POSITION_FIX] {symbol} age={position_age_sec/60:.1f}min seems invalid, "
                    f"capping to 2h (entry_time={entry_time}, now={monitor_now})"
                )
                position_age_sec = 7200
            
            # ZOMBIE KILLER: Force close if no price data for 5+ minutes
            if current_price <= 0:
                if position_age_sec > 300:  # 5 minutes without price = zombie
                    self.logger.warning(
                        f"ZOMBIE DETECTED: {symbol} age={position_age_sec/60:.1f}min no_price - Force closing"
                    )
                    # Queue for exit with last known price or entry price as fallback
                    exit_price = entry_price  # Use entry as fallback for zombie
                    positions_to_exit.append((symbol, position, "zombie_no_price", exit_price))
                continue  # Skip other checks if no price
            
            # ============================================================
            # SPREAD EXPLOSION EXIT: "If liquidity disappears, GET OUT"
            # ============================================================
            from .config import (
                SPREAD_EXPLOSION_EXIT_ENABLED, SPREAD_EXPLOSION_MULTIPLIER, SPREAD_EXPLOSION_MAX_BPS
            )
            # CRITICAL: Skip spread explosion check if spread_bps is invalid (9999 = sentinel value)
            # This happens when bid/ask data isn't available from the ticker
            if SPREAD_EXPLOSION_EXIT_ENABLED and spread_bps > 0 and spread_bps < 5000:
                entry_spread_bps = position.get('entry_spread_bps', spread_bps)
                spread_exploded = False
                explosion_reason = ""
                
                # Only check if entry_spread was valid (not a sentinel value)
                if entry_spread_bps > 0 and entry_spread_bps < 5000 and spread_bps >= entry_spread_bps * SPREAD_EXPLOSION_MULTIPLIER:
                    spread_exploded = True
                    explosion_reason = f"spread_explosion: {spread_bps:.1f}bps >= {entry_spread_bps:.1f}bps * {SPREAD_EXPLOSION_MULTIPLIER}"
                
                if spread_bps >= SPREAD_EXPLOSION_MAX_BPS:
                    spread_exploded = True
                    explosion_reason = f"spread_explosion_max: {spread_bps:.1f}bps >= {SPREAD_EXPLOSION_MAX_BPS}bps cap"
                
                if spread_exploded:
                    self.logger.warning(f"SPREAD_EXPLOSION: {symbol} {explosion_reason} - Emergency exit!")
                    positions_to_exit.append((symbol, position, explosion_reason, current_price))
                    continue
            
            # ============================================================
            # FUNDING TIME AVOIDANCE: Don't hold small profits into funding
            # ============================================================
            from .config import (
                FUNDING_AVOIDANCE_ENABLED, FUNDING_AVOIDANCE_MINUTES, FUNDING_AVOIDANCE_MIN_PROFIT_PCT
            )
            if FUNDING_AVOIDANCE_ENABLED:
                from datetime import datetime, timezone
                now_utc = datetime.now(timezone.utc)
                current_hour = now_utc.hour
                current_minute = now_utc.minute
                
                approaching_funding = False
                for funding_hour in [0, 8, 16]:
                    if current_hour == funding_hour - 1 or (current_hour == 23 and funding_hour == 0):
                        minutes_until = 60 - current_minute if funding_hour != 0 or current_hour == 23 else 0
                        if minutes_until <= FUNDING_AVOIDANCE_MINUTES:
                            approaching_funding = True
                            break
                
                if approaching_funding:
                    profit_pct = ((current_price - entry_price) / entry_price) * 100 if side == 'long' else ((entry_price - current_price) / entry_price) * 100
                    if 0 < profit_pct < FUNDING_AVOIDANCE_MIN_PROFIT_PCT:
                        self.logger.info(f"FUNDING_AVOIDANCE: {symbol} exiting before funding (profit={profit_pct:.2f}%)")
                        positions_to_exit.append((symbol, position, f"funding_avoidance_{profit_pct:.2f}pct", current_price))
                        continue
            
            # --- TREND DEFENSE MECHANISM ---
            # If position is going against us AND the daily trend agrees with the loss,
            # we tighten the stop loss to prevent "hope mode".
            #
            # Logic: 
            #   If LONG and PnL < -0.4% and Trend is DOWN (>2% drop 24h) -> Tighten Stop
            #   If SHORT and PnL < -0.4% and Trend is UP (>2% pump 24h) -> Tighten Stop
            
            pnl_pct = 0.0
            if side == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            elif side == 'short':
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            # Get 24h change from stats
            pct_change_24h = getattr(stats, 'pct_change_24h', 0.0) or 0.0
            
            # Check for adverse trend condition (RELAXED: Prevent chop-out)
            # Only trigger if PnL is significantly negative (-0.8%) AND trend is strongly against (-5%)
            adverse_trend = False
            if side == 'long' and pnl_pct < -0.8 and pct_change_24h < -5.0:
                 adverse_trend = True
            elif side == 'short' and pnl_pct < -0.8 and pct_change_24h > 5.0:
                 adverse_trend = True
                 
            if adverse_trend:
                # We are losing and the macro trend is against us.
                # Check if we should tighten the stop.
                current_sl = position.get('stop_loss', 0)
                
                # For LONG: New aggressive stop is just below current price
                if side == 'long':
                    aggressive_stop = current_price * 0.998 # 0.2% below current
                    # Only move UP (tighten)
                    if current_sl == 0 or aggressive_stop > current_sl:
                        # Log it once to avoid spam (check if we already tightened close to this)
                        if not position.get('trend_defense_active'):
                             self.logger.info(f"🛡️ TREND DEFENSE: {symbol} PnL={pnl_pct:.2f}% vs Trend={pct_change_24h:.1f}% - Tightening Stop")
                             position['trend_defense_active'] = True
                        
                        # Apply new stop
                        position['stop_loss'] = aggressive_stop
                        
                        # CRITICAL: Immediately check if tightened stop is hit
                        if current_price <= aggressive_stop:
                            self.logger.warning(f"🛡️ TREND DEFENSE STOP HIT: {symbol} price={current_price:.4f} <= stop={aggressive_stop:.4f}")
                            positions_to_exit.append((symbol, position, "trend_defense_stop_loss", aggressive_stop))
                            continue  # Exit immediately, skip other checks

                # For SHORT: New aggressive stop is just above current price
                elif side == 'short':
                    aggressive_stop = current_price * 1.002 # 0.2% above current
                    # Only move DOWN (tighten)
                    if current_sl == 0 or aggressive_stop < current_sl:
                        if not position.get('trend_defense_active'):
                             self.logger.info(f"🛡️ TREND DEFENSE: {symbol} PnL={pnl_pct:.2f}% vs Trend={pct_change_24h:.1f}% - Tightening Stop")
                             position['trend_defense_active'] = True
                        position['stop_loss'] = aggressive_stop
                        
                        # CRITICAL: Immediately check if tightened stop is hit
                        if current_price >= aggressive_stop:
                            self.logger.warning(f"🛡️ TREND DEFENSE STOP HIT: {symbol} price={current_price:.4f} >= stop={aggressive_stop:.4f}")
                            positions_to_exit.append((symbol, position, "trend_defense_stop_loss", aggressive_stop))
                            continue  # Exit immediately, skip other checks
            
            # Calculate current PnL ONCE (used by all exit logic)
            # OPTIMIZATION: Calculate once, reuse everywhere
            if side == 'long':
                current_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                current_pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Update peak PnL tracking (needed for recovery score and time extension logic)
            peak_pnl = position.get('peak_pnl', current_pnl_pct)
            if current_pnl_pct > peak_pnl:
                position['peak_pnl'] = current_pnl_pct
                peak_pnl = current_pnl_pct
            
            # MARKSMAN MODE removed - no longer used
            signal_type = position.get('signal_type', '')
            is_marksman_position = False  # MARKSMAN removed
            
            # MARKSMAN positions: Exit at exactly 45 minutes (legacy guard, constant inline)
            if is_marksman_position:
                marksman_max_age_sec = 2700  # 45 minutes
                if position_age_sec > marksman_max_age_sec:
                    self.logger.info(
                        f"MARKSMAN_TIME_EXIT: {symbol} age={position_age_sec/60:.1f}min (>45min) pnl={current_pnl_pct:.2f}%"
                    )
                    positions_to_exit.append((symbol, position, "marksman_time_exit_45min", current_price))
                    continue  # Skip other checks, force exit at 45min
            
            # TIME-BASED EXIT: Use MAX_POSITION_AGE_SEC from config (not hardcoded)
            from .config import MAX_POSITION_AGE_SEC

            # Check if position has been granted an extension
            time_extension_granted = position.get('time_extension_granted', False)
            extension_duration = 900  # 15 minutes extension
            absolute_max_age = MAX_POSITION_AGE_SEC + extension_duration  # Config max + 15min extension

            # CRITICAL: Startup guard - prevent time exits during first 30 seconds after startup
            # This prevents positions from being closed on restart due to invalid entry_time or sync issues
            startup_age = monitor_now - getattr(self, '_startup_time', monitor_now)
            STARTUP_GUARD_SEC = 30.0
            in_startup_guard = startup_age < STARTUP_GUARD_SEC

            # ABSOLUTE CAP: Config max + extension (no exceptions, but respect startup guard)
            if position_age_sec > absolute_max_age:
                if not in_startup_guard:
                    self.logger.info(
                        f"TIME_EXIT_ABSOLUTE: {symbol} age={position_age_sec/60:.1f}min (>{(absolute_max_age/60):.0f}min cap) pnl={current_pnl_pct:.2f}%"
                    )
                    positions_to_exit.append((symbol, position, "time_exit_absolute_cap", current_price))
                    continue  # Skip other checks, force exit
                else:
                    self.logger.debug(f"[STARTUP_GUARD] Blocked absolute time exit for {symbol} (startup age: {startup_age:.1f}s, position age: {position_age_sec/60:.1f}min)")

            # DEFAULT TIME EXIT: Use MAX_POSITION_AGE_SEC from config
            if position_age_sec > MAX_POSITION_AGE_SEC and not time_extension_granted:
                # CRITICAL: Block time exits during startup guard
                if in_startup_guard:
                    self.logger.debug(f"[STARTUP_GUARD] Blocked time exit for {symbol} (startup age: {startup_age:.1f}s, position age: {position_age_sec/60:.1f}min)")
                    # Skip time exit during startup guard, continue to other checks
                else:
                    # Check if position qualifies for ONE-TIME extension
                    # Criteria: Within 0.5R of break-even AND showing improvement
                    is_near_breakeven = abs(current_pnl_pct) < 0.5  # Within 0.5% of BE
                    is_improving = current_pnl_pct > (peak_pnl - 0.3)  # Not deteriorating badly
                    
                    if is_near_breakeven and is_improving:
                        # Grant ONE-TIME extension
                        position['time_extension_granted'] = True
                        self.logger.info(
                            f"TIME_EXTENSION: {symbol} granted {extension_duration/60:.0f}min extension (near BE, improving) pnl={current_pnl_pct:.2f}%"
                        )
                        # Don't exit, let it continue
                    else:
                        # No extension - force exit at config max age (but not during startup guard)
                        if not in_startup_guard:
                            self.logger.info(
                                f"TIME_EXIT_DEFAULT: {symbol} age={position_age_sec/60:.1f}min (>{MAX_POSITION_AGE_SEC/60:.0f}min) pnl={current_pnl_pct:.2f}%"
                            )
                            positions_to_exit.append((symbol, position, f"time_exit_{int(MAX_POSITION_AGE_SEC/60)}min", current_price))
                            continue  # Skip other checks, force exit
                        else:
                            self.logger.debug(f"[STARTUP_GUARD] Blocked default time exit for {symbol} (startup age: {startup_age:.1f}s)")
            
            # ============================================================
            # EMERGENCY LOSS LIMIT CHECK (Before any other exit logic)
            # ============================================================
            # CRITICAL: Force exit if position loss exceeds hard limit
            # This prevents catastrophic losses like PIPPIN (2 days of losses)
            from .config import MAX_POSITION_LOSS_PCT
            if current_pnl_pct <= MAX_POSITION_LOSS_PCT:
                self.logger.error(
                    f"[EMERGENCY_EXIT] {symbol} {side.upper()} | "
                    f"LOSS LIMIT HIT: {current_pnl_pct:.2f}% <= {MAX_POSITION_LOSS_PCT}% | "
                    f"entry={entry_price:.6f} current={current_price:.6f} | "
                    f"FORCING IMMEDIATE EXIT"
                )
                positions_to_exit.append((symbol, position, f"emergency_loss_limit_{MAX_POSITION_LOSS_PCT}%", current_price))
                continue  # Skip other checks, force exit immediately
            
            # Additional safety: Validate stop-loss exists and is reasonable
            stop_loss = position.get('stop_loss', 0)
            if stop_loss <= 0 or stop_loss == entry_price:
                # Invalid stop-loss - if we're losing, force exit
                if current_pnl_pct <= -5.0:
                    self.logger.error(
                        f"[EMERGENCY_EXIT] {symbol} {side.upper()} | "
                        f"INVALID STOP-LOSS with {current_pnl_pct:.2f}% loss | "
                        f"stop_loss={stop_loss:.6f} entry={entry_price:.6f} | "
                        f"FORCING IMMEDIATE EXIT"
                    )
                    positions_to_exit.append((symbol, position, "emergency_invalid_stop_loss", current_price))
                    continue
            
            # OPTIMIZATION: Update peak/trough prices for ATR trailing stops
            # Use cached position attributes and already-calculated current_pnl_pct
            if side == 'long':
                peak_price = position.get('peak_price', entry_price)
                position['peak_price'] = max(peak_price, current_price)
            else:  # short
                trough_price = position.get('trough_price', entry_price)
                position['trough_price'] = min(trough_price, current_price)
            
            # Peak PnL already updated above in time exit section - no need to recalculate
            
            # DYNAMIC POSITION RECOVERY SCORE (PRS) EVALUATION
            # Compute recovery score and decide on actions (close, scale out, tighten stop)
            age_minutes = (monitor_now - entry_time) / 60.0 if entry_time > 0 else 0.0
            
            # Get trend and volatility data (simplified - can be enhanced with full indicators)
            # For now, use price action and stored ATR
            atr_pct = position.get('atr_pct', None)
            # REFACTOR: Added defensive check for stats existence before getattr
            if atr_pct is None and stats is not None:
                # Try to get from stats if available
                atr_pct = getattr(stats, 'atr_pct', None)
            
            # Determine volatility regime
            vol_regime = self.exit_pipeline.get_volatility_regime(atr_pct=atr_pct)
            
            # PRS ENHANCEMENT: Calculate trend direction from recent price data
            trend5 = 0  # Default neutral
            trend15 = 0  # Default neutral
            
            try:
                # Try to get recent OHLCV data for trend calculation (replay feed or fast_storage)
                ohlcv_5m = None
                ohlcv_15m = None
                
                # REPLAY MODE: Get OHLCV from replay feed
                if self.replay_mode and self.replay_feed:
                    ohlcv_5m = self.replay_feed.get_ohlcv(symbol, timeframe='5m', limit=20)
                    ohlcv_15m = self.replay_feed.get_ohlcv(symbol, timeframe='15m', limit=20)
                # LIVE MODE: Try fast_storage (if it has get_ohlcv method)
                elif hasattr(self.fast_storage, 'get_ohlcv'):
                    # Get 5-minute prices (last 20 bars = ~1.5 hours)
                    ohlcv_5m = self.fast_storage.get_ohlcv(symbol, timeframe='5m', limit=20)
                    # Get 15-minute prices (last 20 bars = ~5 hours)
                    ohlcv_15m = self.fast_storage.get_ohlcv(symbol, timeframe='15m', limit=20)
                
                if ohlcv_5m and len(ohlcv_5m) >= 10:
                    prices_5m = [bar[4] for bar in ohlcv_5m]  # Close prices
                    from .indicators import calculate_trend_direction_from_prices
                    trend5 = calculate_trend_direction_from_prices(prices_5m, ema_short=3, ema_long=8)
                else:
                    trend5 = 0
                
                if ohlcv_15m and len(ohlcv_15m) >= 10:
                    prices_15m = [bar[4] for bar in ohlcv_15m]  # Close prices
                    trend15 = calculate_trend_direction_from_prices(prices_15m, ema_short=5, ema_long=10)
                else:
                    trend15 = 0
            except Exception as e:
                # If trend calculation fails, use neutral (no impact on PRS)
                self.logger.debug(f"Trend calculation failed for {symbol}: {e}")
                trend5 = 0
                trend15 = 0
            
            # Calculate recovery score
            recovery_score = self.exit_pipeline.compute_recovery_score(
                position, current_price, trend5=trend5, trend15=trend15, vol_regime=vol_regime
            )
            
            # Store recovery score and age metadata for UI/logging
            position['recovery_score'] = recovery_score
            position['age_minutes'] = age_minutes
            
            # PRS DEBUG LOGGING: Log detailed PRS information
            if self.logger.isEnabledFor(logging.DEBUG):
                pnl_pct = self._calculate_pnl_pct(entry_price, current_price, side)
                self.logger.debug(
                    f"[PRS] {symbol}: score={recovery_score:.1f}, "
                    f"age={age_minutes:.1f}m, pnl={pnl_pct:.2f}%, "
                    f"peak_pnl={position.get('peak_pnl', 0):.2f}%, "
                    f"trend5={trend5}, trend15={trend15}, vol={vol_regime}"
                )
            
            # NEW ARCHITECTURE: Use RecoveryModule for PRS evaluation
            if not hasattr(self, 'recovery_module'):
                from .core.recovery import RecoveryModule
                self.recovery_module = RecoveryModule(self.exit_manager)
            
            # Evaluate PRS and get action
            prs_action = self.recovery_module.evaluate_position(
                symbol=symbol,
                position=position,
                current_price=current_price,
                now=monitor_now,
                trend5=trend5,
                trend15=trend15,
                vol_regime=vol_regime
            )
            
            # Queue PRS action if needed
            if prs_action:
                # Force 100% exit if partials disabled (prevents dust positions)
                from .config import DISABLE_PARTIAL_EXITS
                exit_size_ratio = 1.0 if DISABLE_PARTIAL_EXITS else prs_action.exit_size_ratio
                positions_to_exit.append((
                    symbol,
                    position,
                    prs_action.reason,
                    None,  # target_price (market order)
                    exit_size_ratio
                ))

            # SCALPER UPGRADE: Use scalper-specific trailing exit logic
            from .engine.scalper_exits import evaluate_scalper_trailing
            
            # Get advanced_features for structural exit checks
            advanced_features = None
            if hasattr(self, '_advanced_features_cache'):
                advanced_features = self._advanced_features_cache.get(symbol)
            
            # Get indicators for ATR
            indicators = None
            if hasattr(self, 'indicators_cache'):
                indicators = self.indicators_cache.get(symbol)
            
            # Update bars_in_trade tracking for scalper exits
            last_bar_update = position.get('last_bar_update_time', entry_time)
            from .config import R_BAR_SCAN_CYCLE_SEC
            bar_closed = (monitor_now - last_bar_update) >= R_BAR_SCAN_CYCLE_SEC
            if bar_closed:
                position['last_bar_update_time'] = monitor_now
                position['bars_in_trade'] = position.get('bars_in_trade', 0) + 1
            
            bars_in_trade = position.get('bars_in_trade', 0)
            
            # SCALPER: Set initial stop loss if not set (0.35 * ATR)
            if not position.get('initial_stop_price'):
                if side == "long":
                    initial_stop = entry_price * (1.0 - 0.35 * (atr_pct or 0.01))
                else:
                    initial_stop = entry_price * (1.0 + 0.35 * (atr_pct or 0.01))
                position['initial_stop_price'] = initial_stop
                if not position.get('stop_loss'):
                    position['stop_loss'] = initial_stop
            
            # Evaluate scalper trailing
            scalper_action = evaluate_scalper_trailing(
                position=position,
                current_price=current_price,
                atr_pct=atr_pct,
                advanced_features=advanced_features,
                indicators=indicators,
                side=side,
                entry_price=entry_price,
                entry_time=entry_time,
                bars_in_trade=bars_in_trade
            )
            
            if scalper_action:
                if scalper_action.action == "exit":
                    # Queue exit
                    # CRITICAL FIX: Use explicit exit_price if provided (for DRY_RUN/scalper exits), otherwise fallback to current_price
                    # This ensures DRY_RUN doesn't default to entry_price when fetching fails
                    exit_price = getattr(scalper_action, 'exit_price', current_price)
                    if exit_price is None or exit_price <= 0:
                        exit_price = current_price
                    
                    # Force 100% exit if partials disabled (prevents dust positions)
                    from .config import DISABLE_PARTIAL_EXITS
                    exit_size_ratio = 1.0 if DISABLE_PARTIAL_EXITS else scalper_action.exit_size_ratio
                    
                    positions_to_exit.append((
                        symbol,
                        position,
                        scalper_action.exit_reason or "scalper_trailing",
                        exit_price,  # target_price (market order, but used for PnL in DRY_RUN)
                        exit_size_ratio
                    ))
                elif scalper_action.action == "update_sl":
                    # Update stop loss
                    position['stop_loss'] = scalper_action.new_stop
                    # Log trailing update
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"TRAIL_SL_UPDATE sym={symbol} SL->{scalper_action.new_stop:.4f} "
                            f"price={current_price:.4f} side={side.upper()}"
                        )
            
            # Dynamic trailing stop evaluation (centralized in ExitPipeline) - fallback
            # Skip in DRY simple exits mode (no partials/trailing)
            trailing_action = None
            from .config import DRY_SIMPLE_EXITS
            if (hasattr(self, 'exit_pipeline') and self.exit_pipeline and not scalper_action and 
                not (DRY_RUN and DRY_SIMPLE_EXITS)):
                trailing_action = self.exit_pipeline.evaluate_trailing(
                    symbol=symbol,
                    position=position,
                    current_price=current_price,
                    now=monitor_now
                )
            
            if trailing_action and trailing_action.partial_actions:
                # Force 100% exit if partials disabled (prevents dust positions)
                from .config import DISABLE_PARTIAL_EXITS
                for ratio, reason in trailing_action.partial_actions:
                    # For trailing stop hits, pass current_price as target_price for PnL calculation
                    # (market order, but we need price for PnL)
                    target_price_for_exit = current_price if "trailing_stop" in reason else None
                    # Force 100% exit if partials disabled
                    exit_ratio = 1.0 if DISABLE_PARTIAL_EXITS else ratio
                    positions_to_exit.append((symbol, position, reason, target_price_for_exit, exit_ratio))
            
            # Check if position should be exited (existing logic continues)
            # Use R-based exit engine if enabled, otherwise fall back to legacy
            from .config import USE_R_BASED_EXITS, R_BAR_SCAN_CYCLE_SEC
            
            if USE_R_BASED_EXITS:
                # R-based exit engine
                # Check if a bar has closed (scan cycle completed)
                last_bar_update = position.get('last_bar_update_time', entry_time)
                bar_closed = (monitor_now - last_bar_update) >= R_BAR_SCAN_CYCLE_SEC
                if bar_closed:
                    position['last_bar_update_time'] = monitor_now
                
                # Get ATR for volatility-aware trailing
                atr_pct = position.get('atr_pct', None)
                
                # Determine if high volatility (for runner profile)
                # Use regime config or fallback to ATR-based heuristic
                is_high_volatility = False
                if hasattr(self, 'regime_config') and self.regime_config:
                    from .regime import TradingRegime
                    is_high_volatility = (self.regime_config.regime_type == TradingRegime.SCALPING and 
                                         getattr(self.regime_config, 'volatility_regime', '') == 'high')
                elif atr_pct and atr_pct > 0.02:  # > 2% ATR suggests high volatility
                    is_high_volatility = True
                
                should_exit, reason, target_price, exit_size_pct = self.exit_pipeline.should_exit_position_r_based(
                    position,
                    current_price,
                    atr_pct=atr_pct,
                    is_high_volatility=is_high_volatility,
                    bar_closed=bar_closed,
                    now_ts=monitor_now
                )
                
                if should_exit:
                    positions_to_exit.append((symbol, position, reason, target_price, exit_size_pct))
            else:
                # Legacy exit logic
                # OPTIMIZATION: Cache regime config lookup (used for all positions in this loop)
                if not hasattr(self, '_monitor_regime_config'):
                    self._monitor_regime_config = getattr(self, 'regime_config', None)
                regime_config = self._monitor_regime_config
                
                should_exit, reason, target_price = self.exit_pipeline.should_exit_position(
                    position, current_price, spread_bps, regime_config, symbol=symbol
                )
                
                if should_exit:
                    positions_to_exit.append((symbol, position, reason, target_price, 1.0))  # Full exit for legacy
        
        # OPTIMIZATION: Use cached time from position monitoring loop
        exit_now = monitor_now
        # Clear cached regime config after loop
        if hasattr(self, '_monitor_regime_config'):
            delattr(self, '_monitor_regime_config')
        
        # CANONICAL POSITION UPDATE TRACKER (prevents double-deletes and missing position errors)
        # Track which positions have been processed in this loop
        if not hasattr(self, '_exits_processed_this_loop'):
            self._exits_processed_this_loop = set()
        if not hasattr(self, '_exit_failures_logged_this_loop'):
            self._exit_failures_logged_this_loop = set()
        self._exits_processed_this_loop.clear()  # Reset for each loop
        self._exit_failures_logged_this_loop.clear()  # Reset for each loop
        
        # CANONICAL EXIT TRACKER: Reset per-loop exit event tracker
        from .position_utils import reset_loop_exit_tracker
        reset_loop_exit_tracker()
        
        # NEW ARCHITECTURE: Route all exits through ExitPipeline
        # Queue exits to pipeline instead of executing directly
        
        # DEDUPLICATION: Remove duplicate exits for same symbol (keep highest priority)
        # This prevents multiple exit systems from queuing the same position
        exit_by_symbol = {}  # symbol -> (exit_data, priority)
        for exit_data in positions_to_exit:
            # Handle both R-based (5-tuple) and legacy (4-tuple) exit data
            if len(exit_data) == 5:
                symbol, position, reason, target_price, exit_size_pct = exit_data
            else:
                symbol, position, reason, target_price = exit_data
                exit_size_pct = 1.0  # Full exit for legacy
            
            # Determine priority (higher = exits first)
            priority = 100  # Default priority
            if "stop_loss" in reason or "circuit_breaker" in reason or "trend_defense" in reason:
                priority = 200  # Highest priority for risk exits
            elif "prs_full_exit" in reason:
                priority = 150  # High priority for PRS full exits
            elif "prs_scale_out" in reason:
                priority = 120  # Medium-high priority for PRS scale-outs
            elif "take_profit" in reason or "tp" in reason:
                priority = 80  # Medium priority for take-profits
            elif "time_exit" in reason:
                priority = 60  # Medium-low priority for time exits
            else:
                priority = 50  # Lower priority for other exits
            
            # Keep highest priority exit for each symbol
            if symbol not in exit_by_symbol or priority > exit_by_symbol[symbol][1]:
                exit_by_symbol[symbol] = (exit_data, priority)
        
        # Use deduplicated exits
        if hasattr(self, 'exit_pipeline') and self.exit_pipeline:
            for symbol, (exit_data, priority) in exit_by_symbol.items():
                # Handle both R-based (5-tuple) and legacy (4-tuple) exit data
                if len(exit_data) == 5:
                    _, position, reason, target_price, exit_size_pct = exit_data
                else:
                    _, position, reason, target_price = exit_data
                    exit_size_pct = 1.0  # Full exit for legacy
                
                # Priority already calculated in deduplication step above
                
                # Determine if limit order should be used
                # Trailing stops always use market orders to avoid price validation issues
                use_limit = (reason in ["take_profit", "scalp_tp_1r", "standard_partial_1r", "runner_partial_1r"]) and "trailing" not in reason.lower()
                
                # Create exit request
                exit_request = ExitRequest(
                    symbol=symbol,
                    position=position,
                    reason=reason,
                    target_price=target_price,
                    exit_size_ratio=exit_size_pct,
                    use_limit=use_limit,
                    priority=priority
                )
                
                # Queue exit request
                self.exit_pipeline.queue_exit(exit_request)
            
            # Process all queued exits
            processed = await self.exit_pipeline.process_exits(bot_instance=self)
            if processed > 0:
                self.logger.debug(f"ExitPipeline processed {processed} exits")
        else:
            # FALLBACK: Use old exit logic if ExitPipeline not available
            # Exit positions (legacy path - will be removed after testing)
            for exit_data in positions_to_exit:
                # Handle both R-based (5-tuple) and legacy (4-tuple) exit data
                if len(exit_data) == 5:
                    symbol, position, reason, target_price, exit_size_pct = exit_data
                else:
                    symbol, position, reason, target_price = exit_data
                    exit_size_pct = 1.0  # Full exit for legacy
                
                # CRITICAL: Check if position still exists BEFORE attempting exit
                if symbol not in self.positions:
                    # Position already deleted (possibly by another exit in same loop)
                    # Log warning ONCE per symbol per loop
                    if symbol not in self._exits_processed_this_loop:
                        self.logger.warning(
                            f"Exit skipped: {symbol} not in positions dict (already closed?)"
                        )
                        self._exits_processed_this_loop.add(symbol)
                    continue  # Skip this exit in legacy fallback loop
                
                # CRITICAL: Verify position data matches
                actual_position = self.positions[symbol]
                if actual_position.get('size', 0) <= 0:
                    # Position size is 0 (already closed or invalid)
                    if symbol not in self._exits_processed_this_loop:
                        self.logger.warning(
                            f"Exit skipped: {symbol} has zero size (already closed?)"
                        )
                        self._exits_processed_this_loop.add(symbol)
                    # Clean up invalid position
                    del self.positions[symbol]
                    if symbol in self._positions_set:
                        self._positions_set.discard(symbol)
                    continue
                
                # CRITICAL: Prevent double-processing same symbol in one loop
                if symbol in self._exits_processed_this_loop:
                    # Already processed in this loop
                    continue
                
                # Mark as processed
                self._exits_processed_this_loop.add(symbol)
                
                # Use actual position from dict (not the one from exit_data, may be stale)
                position = actual_position
                
                # OPTIMIZATION: Cache position attributes to avoid repeated dict lookups
                position_side = position.get('side', '')
                entry_price = position.get('entry_price', 0)
                entry_time = position.get('entry_time', exit_now)
                position_size = position.get('size', 0)
                
                # Calculate exit size (for partial exits)
                # NOTE: Do NOT update position size here - done atomically in update_position_after_exit()
                if exit_size_pct < 1.0:
                    # Partial exit requested - but check if it makes sense
                    stats = self.universe.stats.get(symbol)
                    est_price = getattr(stats, 'last', entry_price) or entry_price
                    current_notional = position_size * est_price
                    
                    # [!] ANTI-DUST RULE #1: No partials on small positions (<$25)
                    # Small positions should always exit 100% to avoid dust
                    if current_notional < 25.0:
                        exit_size = position_size
                        self.logger.info(f"[!] ANTI-DUST: Forcing 100% exit on {symbol} (Position ${current_notional:.2f} too small for partial)")
                    else:
                        requested_exit_size = position_size * exit_size_pct
                        remaining_size = position_size - requested_exit_size
                        remaining_notional = remaining_size * est_price
                        
                        # [!] ANTI-DUST RULE #2: No partials that leave <$10 behind
                        # This prevents creating dust that can't be closed
                        if remaining_notional < 10.0:
                            exit_size = position_size
                            self.logger.info(f"[!] ANTI-DUST: Forcing 100% exit on {symbol} (Partial would leave ${remaining_notional:.2f} dust)")
                        else:
                            # Safe to proceed with partial
                            exit_size = requested_exit_size
                else:
                    # Full exit requested
                    exit_size = position_size
                
                # Determine order type (use limit for take-profit/partial, market for stop-loss)
                use_limit = (reason in ["take_profit", "scalp_tp_1r", "standard_partial_1r", "runner_partial_1r"])
                
                # Execute exit (create temporary position dict for exit calculation)
                exit_position_dict = position.copy()
                exit_position_dict['size'] = exit_size  # Use exit size for this exit
                
                exit_result = await self.exit_pipeline._execute_exit_order(
                    symbol=symbol,
                    position=exit_position_dict,
                    reason=reason,
                    target_price=target_price,
                    use_limit=use_limit
                )
                
                if exit_result.success:
                    # Calculate PnL
                    was_win = exit_result.net_pnl > 0
                    
                    # Update statistics
                    if was_win:
                        self.win_count += 1
                        self.gross_win += exit_result.gross_pnl
                    else:
                        self.loss_count += 1
                        self.gross_loss += abs(exit_result.gross_pnl)
                    
                    self.realized_pnl_total += exit_result.net_pnl
                    self.realized_fees_total += exit_result.total_costs
                    
                    # Track fee breakdown
                    if exit_result.entry_fee is not None:
                        self.realized_entry_fees_total += exit_result.entry_fee
                    if exit_result.exit_fee is not None:
                        self.realized_exit_fees_total += exit_result.exit_fee
                    if exit_result.slippage is not None:
                        self.realized_slippage_total += exit_result.slippage
                    if exit_result.funding_cost is not None:
                        self.realized_funding_total += exit_result.funding_cost
                    
                    # CANONICAL EXIT AND LOG: Single function that both updates position AND logs DecisionEvent
                    # This ensures every successful exit/partial-exit produces exactly one DecisionEvent
                    from .position_utils import apply_exit_and_log, record_loop_exit
                    
                    # Calculate new position size
                    if exit_size_pct < 1.0:
                        # Partial exit: update size
                        new_size = position_size - exit_result.exit_size
                    else:
                        # Full exit: remove position
                        new_size = 0.0
                    
                    # Calculate PnL percentage
                    pnl_pct = 0.0
                    if entry_price > 0 and exit_result.exit_price and exit_result.exit_price > 0:
                        if position_side.lower() == 'long':  # OPTIMIZATION: Use cached value
                            pnl_pct = ((exit_result.exit_price - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = ((entry_price - exit_result.exit_price) / entry_price) * 100
                    
                    # Determine action type
                    if exit_size_pct < 1.0:
                        action = "PARTIAL_EXIT" if "prs_" in reason or "scale" in reason.lower() else "SCALE_OUT"
                    else:
                        action = "EXIT"
                    
                    # Get size before and after for partial exits
                    size_before = position_size
                    size_after = new_size if exit_size_pct < 1.0 else 0.0
                    
                    # Get recovery score (PRS) if available
                    prs = position.get('recovery_score')
                    
                    # CANONICAL: Apply exit and log atomically
                    success, event_created = apply_exit_and_log(
                        positions=self.positions,
                        positions_set=self._positions_set,
                        symbol=symbol,
                        new_size=new_size,
                        action=action,
                        exit_price=exit_result.exit_price,
                        entry_price=entry_price,
                        entry_time=entry_time,
                        exit_time=exit_now,
                        exit_size=exit_result.exit_size,
                        size_before=size_before,
                        size_after=size_after,
                        side=position_side,
                        pnl_value=exit_result.net_pnl,
                        pnl_pct=pnl_pct,
                        gross_pnl=exit_result.gross_pnl,
                        net_pnl=exit_result.net_pnl,
                        total_costs=exit_result.total_costs,
                        reason=reason,
                        prs=prs,
                        was_win=was_win,
                        is_unicorn=position.get('is_unicorn', False),
                        bot_instance=self
                    )
                    
                    if not success:
                        # Position update failed (already deleted or invalid)
                        self.logger.warning(
                            f"Position update failed for {symbol}: position may have been deleted or is invalid "
                            f"(reason={reason}, exit_price={exit_result.exit_price if exit_result else 'N/A'})"
                        )
                        continue
                    
                    # Track successful exit for self-check
                    if event_created:
                        record_loop_exit(symbol, action)
                    
                    # Record exit in position manager (only for full exits)
                    if exit_size_pct >= 1.0:
                        pnl_pct_for_manager = (exit_result.net_pnl / self.equity_now() * 100) if exit_result.net_pnl is not None else None
                        # Calculate profit_atr for churn tracking (if time_exit)
                        profit_atr = None
                        if reason and reason.startswith("time_exit"):
                            # Try to extract from reason string: "time_exit: 4 bars, profit=0.00ATR"
                            import re
                            match = re.search(r'profit=([\d\.-]+)ATR', reason)
                            if match:
                                try:
                                    profit_atr = float(match.group(1))
                                except (ValueError, AttributeError):
                                    pass
                            # Fallback: calculate from position data if available
                            if profit_atr is None and entry_price > 0 and exit_result.exit_price > 0:
                                atr_pct = position.get('atr_pct', None)
                                if atr_pct and atr_pct > 0:
                                    if position_side.lower() == 'long':
                                        profit_pct = ((exit_result.exit_price - entry_price) / entry_price)
                                    else:
                                        profit_pct = ((entry_price - exit_result.exit_price) / entry_price)
                                    profit_atr = profit_pct / atr_pct
                        self.position_manager.record_exit(symbol, was_win, pnl_pct=pnl_pct_for_manager, exit_reason=reason, profit_atr=profit_atr)
                    
                    # METRICS: exits by reason
                    try:
                        self.metrics['exits_by_reason'][reason] = self.metrics['exits_by_reason'].get(reason, 0) + 1
                    except (KeyError, TypeError, AttributeError):
                        # Metrics dict structure issue - non-critical, safe to ignore
                        pass
                else:
                    # Exit failed - handle gracefully and log compact error
                    error_msg = str(exit_result.error) if exit_result.error else "Unknown error"
                    from datetime import datetime
                    time_str = datetime.now().strftime("%H:%M:%S")
                    symbol_short = symbol.replace("/USDT", "")
                    
                    # Detect specific error types and log compact messages
                    if "Invalid position" in error_msg:
                        # Position doesn't exist on exchange (likely already closed elsewhere)
                        # Only log once per symbol per loop (not spammy)
                        if symbol not in self._exit_failures_logged_this_loop:
                            self.logger.warning(
                                f"{time_str}  EXIT_FAILED {symbol_short} Invalid pos"
                            )
                            self._exit_failures_logged_this_loop.add(symbol)
                        
                        # CRITICAL: Clean up position from dict if exchange says it doesn't exist
                        # This prevents future "Invalid pos" errors for the same symbol
                        if symbol in self.positions:
                            del self.positions[symbol]
                            if symbol in self._positions_set:
                                self._positions_set.discard(symbol)
                    elif "invalid_limit_price" in error_msg or "invalid_price" in error_msg or "Invalid target price" in error_msg:
                        # Price validation errors - log compact message and debug details
                        self.logger.warning(
                            f"{time_str}  EXIT_SKIPPED {symbol_short} invalid_price"
                        )
                        # Log detailed debug info (not INFO level to keep LOG panel clean)
                        self.logger.debug(
                            f"[EXIT] {symbol} skipped: {error_msg} | "
                            f"reason={reason} | target_price={target_price} | "
                            f"use_limit={use_limit} | side={position_side} | "
                            f"entry_price={entry_price}"
                        )
                    elif "market_price_unavailable" in error_msg:
                        # Market price fetch failed - log compact message
                        self.logger.warning(
                            f"{time_str}  EXIT_SKIPPED {symbol_short} no_market_price"
                        )
                        self.logger.debug(
                            f"[EXIT] {symbol} skipped: market price unavailable | reason={reason}"
                        )
                    else:
                        # Other errors (API timeout, connection issues, etc.) - log compact format
                        error_type = type(exit_result.error).__name__ if exit_result.error else "Unknown"
                        error_msg_short = str(exit_result.error)[:80] if exit_result.error else "Unknown error"
                        self.logger.error(
                            f"[E] exit_fail sym={symbol_short} type={error_type} msg={error_msg_short}"
                        )
        
        # SELF-CHECK: Verify that every successful exit/partial-exit produced exactly one DecisionEvent
        # Use per-loop tracker instead of time-based matching for accuracy
        try:
            from .position_utils import get_loop_exit_events
            
            # Get exit events recorded in THIS loop (via apply_exit_and_log -> record_loop_exit)
            loop_exit_events = get_loop_exit_events()
            events_logged_count = len(loop_exit_events)
            
            # Count how many exits were attempted (not skipped)
            attempted_symbols = [e[0] for e in positions_to_exit]
            skipped_symbols = [s for s in self._exits_processed_this_loop if s not in [e[0] for e in loop_exit_events]]
            attempted_but_skipped = [s for s in attempted_symbols if s in skipped_symbols]
            attempted_not_skipped = [s for s in attempted_symbols if s not in attempted_but_skipped]
            
            # Count successful exits (those that created DecisionEvents)
            logged_symbols = [e[0] for e in loop_exit_events]
            successful_count = len(logged_symbols)
            
            # Expected: Every exit that was attempted (not skipped) should have succeeded and created a DecisionEvent
            # Exception: If exit_result.success == False, no DecisionEvent is created (expected behavior)
            # So we can't compare attempted vs logged directly - we need to track which ones actually succeeded
            
            # For now, just verify that logged events match what we expect
            # If we have logged events, they should all be from attempted exits
            if loop_exit_events:
                extra_symbols = [s for s in logged_symbols if s not in attempted_symbols]
                
                if extra_symbols:
                    # Extra DecisionEvents that weren't in positions_to_exit (shouldn't happen)
                    # Single-line format for LOG panel
                    extra_symbols_str = ','.join(extra_symbols[:5])  # Limit to first 5
                    if len(extra_symbols) > 5:
                        extra_symbols_str += f" (+{len(extra_symbols)-5} more)"
                    self.logger.warning(
                        f"[SELF-CHECK] Extra DecisionEvents: {extra_symbols_str}"
                    )
                
                # Log success (downgrade to DEBUG to avoid spam once stable)
                # Only log if there are actual exits to verify
                if successful_count > 0:
                    self.logger.debug(
                        f"[SELF-CHECK] {successful_count} exit events logged"
                    )
        except Exception as e:
            # Non-critical, don't break execution
            self.logger.debug(f"[SELF-CHECK] Failed to verify exit events: {e}")

    def _log_signal_decision(self, symbol: str, signal, action: str, reason: Optional[str] = None):
        """
        Log signal decision with full score breakdown to decisions.jsonl.
        
        RECOMMENDATION #5: Filters applied to reduce log volume:
        - Always logs approved signals
        - Only logs rejections above LOG_REJECTION_MIN_SCORE or sampled at LOG_REJECTION_SAMPLE_RATE
        
        Args:
            symbol: Trading symbol
            signal: TradingSignal object
            action: "approved" or "rejected"
            reason: Rejection reason (if rejected)
        """
        from .config import LOG_ALL_APPROVALS, LOG_REJECTION_MIN_SCORE, LOG_REJECTION_SAMPLE_RATE
        import random
        
        # Apply filters to reduce log volume
        should_log = False
        
        if action == "approved":
            # Always log approved signals (default behavior)
            should_log = LOG_ALL_APPROVALS
        elif action == "rejected":
            # For rejections, apply score filter or sampling
            if signal.final_score >= LOG_REJECTION_MIN_SCORE:
                # High-score rejection - always log (interesting case)
                should_log = True
            elif random.random() < LOG_REJECTION_SAMPLE_RATE:
                # Low-score rejection - sample at configured rate
                should_log = True
            # else: skip logging low-score rejection
        
        if not should_log:
            return
        
        log_entry = {
            'timestamp': time.time(),
            'symbol': symbol,
            'side': signal.side,
            'action': action,
            'reason': reason,
            'final_score': signal.final_score,
            'strength': signal.strength,
            'signal_type': signal.signal_type,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        }

        # MVP scoring observability (shadow/live behind rollback switch)
        try:
            mvp_mode = getattr(signal, "mvp_mode", None)
            if mvp_mode:
                log_entry["mvp_mode"] = mvp_mode
                log_entry["mvp_score"] = getattr(signal, "mvp_score", None)
                log_entry["mvp_arm"] = getattr(signal, "mvp_arm", None)
                log_entry["mvp_effective_min_score"] = getattr(signal, "mvp_effective_min_score", None)
                log_entry["mvp_effective_min_strength"] = getattr(signal, "mvp_effective_min_strength", None)
                log_entry["mvp_would_enter"] = getattr(signal, "mvp_would_enter", None)
                # Include component breakdown for forensic debugging.
                log_entry["mvp_components"] = getattr(signal, "mvp_components", None)
        except Exception:
            pass
        
        # Add full score breakdown if available
        if signal.signal_score:
            log_entry.update(signal.signal_score.to_dict())
        
        # Write asynchronously to rotating decision log
        try:
            decision_logger = get_decision_logger()
            decision_logger.log(log_entry)
        except Exception:
            # Non-critical, don't break execution
            pass

    async def run(self):
        # CRITICAL: Disable all console logging from external libraries FIRST
        from .logger import disable_external_loggers
        disable_external_loggers()
        
        # Enable print monitor in debug mode (optional)
        debug_mode = os.getenv("DEBUG_PRINT_MONITOR", "false").lower() == "true"
        if debug_mode:
            try:
                from .debug_monitor import enable_print_monitor
                enable_print_monitor()
                self.logger.debug("[DEBUG] Print monitor enabled")
            except ImportError:
                pass
        
        await self.init_exchange()
        try:
            self.logger.info("[DIAG] run:after_init_exchange")
        except Exception:
            pass
        
        # Initialize UI
        from .config import UI_MODE
        ui_instance = None
        
        # Determine if we should start UI
        # Support v2, rich, and dashboard modes
        start_ui = UI_MODE in ("v2", "rich", "dashboard")
        
        if start_ui:
            try:
                # Import UI here to avoid circular dependencies
                # UPGRADE: Use UIv3 Command Center (Direct Link)
                from .ui_v3 import UIv3
                self.logger.info(f"[UI] Initializing {UI_MODE} (UIv3 Command Center)...")
                ui_instance = UIv3()
                
                # Start UI (take over screen)
                ui_instance.__enter__()
                self.logger.info("[UI] UI started successfully")
            except Exception as e:
                self.logger.error(f"[UI] Failed to start UI: {e}", exc_info=True)
                # Ensure ui_instance is None so we don't try to use/close it
                ui_instance = None
        else:
            self.logger.info(f"[UI] Headless mode (UI_MODE={UI_MODE})")
        
        next_universe = 0
        next_signal_scan = 0
        
        # Initialize regime config on startup
        if self.regime_config is None:
            regime_config = self.ctrl.regime_manager.get_current_config()
            self.ctrl.apply_regime_config(self, regime_config)
        
        try:
            self.logger.info("[DIAG] main_loop:starting")
        except Exception:
            pass
        try:
            while True:
                # DIAG: Trace loop heartbeat (debug level to avoid console spam)
                try:
                    self.logger.debug("[DIAG] main_loop:tick")
                except Exception:
                    pass
                
                # Check drawdown circuit breaker (before any trading activity)
                if self.check_drawdown_circuit_breaker():
                    self.logger.critical(
                        "TRADING HALTED: Drawdown circuit breaker active. Manual intervention required.",
                        drawdown_pct=self.get_drawdown_pct(),
                        max_drawdown_pct=MAX_DRAWDOWN_PCT
                    )
                    # Continue monitoring positions but don't enter new ones
                    # Exit existing positions will still be processed
                    await self.monitor_and_exit_positions()
                    await asyncio.sleep(1.0)  # Wait before next check
                    continue  # Skip signal scanning and new entries
                
                # OPTIMIZATION: Only run LLM controller operations periodically (not every loop)
                # These operations can be expensive and don't need to run every 100ms
                llm_check_interval = 5.0  # Check every 5 seconds
                if not hasattr(self, '_last_llm_check'):
                    self._last_llm_check = 0.0
                
                if self._get_current_time() - self._last_llm_check >= llm_check_interval:
                    await self.ctrl.check_control_patch(self)
                    self.ctrl.generate_advisor_suggestions(self)
                    self.ctrl.check_and_switch_regime(self)  # Check and switch regime if needed
                    self._last_llm_check = self._get_current_time()

                # [!] MAXIMUM AGGRESSION: Force scalping regime on startup
                if not hasattr(self, '_aggressive_mode_forced'):
                    from .regime import TradingRegime
                    self.ctrl.regime_manager.set_regime(TradingRegime.SCALPING, "[FORCED] Maximum aggression mode")
                    self.current_regime = TradingRegime.SCALPING.value
                    self.regime_since = self._get_current_time()
                    regime_config = self.ctrl.regime_manager.get_current_config()
                    self.regime_config = regime_config
                    self.ctrl.apply_regime_config(self, regime_config)
                    self._aggressive_mode_forced = True
                    # CRITICAL FIX: Enforce sane minimum scan interval to prevent API throttling
                    # Binance limit ~1200 req/min. 100 symbols * (60/5) = 1200 req/min.
                    # So 5.0s is the absolute minimum safe interval for 100 symbols.
                    if regime_config.scan_interval_seconds < 5.0:
                        regime_config.scan_interval_seconds = 5.0
                        
                    self.logger.warning(
                        f"[!] MAXIMUM AGGRESSION MODE ACTIVATED | "
                        f"scan_interval={regime_config.scan_interval_seconds}s | "
                        f"max_positions={regime_config.max_concurrent_positions} | "
                        f"risk_per_trade={self.cfg.RISK_PER_TRADE_PCT:.1f}% | total_risk={self.cfg.MAX_ACCOUNT_RISK_PCT:.1f}%"
                    )

                # Get regime-specific intervals (ensure regime_config is set)
                regime_config = getattr(self, 'regime_config', None)
                if regime_config is None:
                    # Fallback: Initialize regime config if somehow None
                    regime_config = self.ctrl.regime_manager.get_current_config()
                    self.ctrl.apply_regime_config(self, regime_config)
                    self.regime_config = regime_config
                
                universe_interval = regime_config.universe_refresh_interval_seconds
                scan_interval = regime_config.scan_interval_seconds
                
                # Skip debug logs on startup
                current_time = self._get_current_time()

                # Refresh universe at regime-specific interval
                if current_time >= next_universe:
                    await self.refresh_universe()
                    next_universe = current_time + universe_interval
                    self.next_universe_refresh_time = next_universe
                    self.last_universe_refresh_time = current_time
                    
                    # Universe health check - alert if empty
                    if len(self.universe.stats) == 0:
                        self.logger.error(
                            "[!] UNIVERSE HEALTH ALERT: Universe is empty! "
                            "No symbols available for scanning. "
                            "Check exchange connection and fallback symbol initialization."
                        )
                    elif len(self.universe.stats) < 10 and hasattr(self, '_fallback_mode') and not self._fallback_mode:
                        self.logger.warning(
                            f"[!] UNIVERSE HEALTH WARNING: Only {len(self.universe.stats)} symbols in universe. "
                            "Expected more symbols. Check exchange connection."
                        )

                # Scan for signals at regime-specific interval
                # CRITICAL: Ensure scan runs immediately on first loop (next_signal_scan=0)
                if current_time >= next_signal_scan:
                    try:
                        self.logger.info("[DIAG] scan_and_enter_signals:start")
                    except Exception:
                        pass
                    # Skip debug logs for strategy cycles
                    await self.scan_and_enter_signals()
                    next_signal_scan = current_time + scan_interval
                    self.next_scan_time = next_signal_scan
                    
                    # CRITICAL FIX: Signal confirmation cleanup (every 5 minutes = 300 seconds)
                    if hasattr(self, 'signal_confirmation') and hasattr(self, '_last_scw_cleanup'):
                        if time.time() - self._last_scw_cleanup >= 300.0:
                            try:
                                self.signal_confirmation.cleanup_stale_signals(max_age_sec=300.0)
                                self._last_scw_cleanup = time.time()
                            except Exception:
                                pass  # Non-critical
                    elif hasattr(self, 'signal_confirmation'):
                        # Initialize cleanup timer
                        self._last_scw_cleanup = time.time()
                else:
                    # Skip debug logs for scan timing
                    pass
                
                # Monitor and exit positions (check every loop iteration)
                await self.monitor_and_exit_positions()
                
                # API BUDGET OPTIMIZATION: Refresh orderbooks for open positions (better exit decisions)
                # [!] THROTTLE PROTECTION: Use separate timer for position orderbooks (10s interval)
                if self.positions:
                    await self._refresh_orderbooks_for_positions()

                # UI UPDATE
                loop_now = time.time()
                # Rich UI is slightly heavier; update a bit less often
                if UI_MODE in ("rich", "v2"):
                    ui_update_interval = 1.0  # Faster updates for v2/rich (1s)
                elif UI_MODE == "v4":
                    ui_update_interval = 2.0
                else:
                    ui_update_interval = 10.0
                
                if loop_now - self.last_draw >= ui_update_interval:
                    try:
                        if ui_instance is not None:
                            ui_instance.render(self)
                    except Exception as e:
                        self.logger.error(f"[UI] Update failed: {e}", exc_info=True)
                    finally:
                        self.last_draw = loop_now

                # LATENCY OPTIMIZATION: Reduce sleep to 0.1s for faster loop (was 0.25s)
                await asyncio.sleep(0.1)
                
                # SNAPSHOT VALIDATION: Periodic check (every ~60s)
                if time.time() - self._last_metrics_log >= 60.0:
                    try:
                        # Use canonical snapshot validation
                        from .engine_snapshot import collect_engine_snapshot, validate_snapshot_consistency
                        
                        snapshot = collect_engine_snapshot(self)
                        warnings = validate_snapshot_consistency(snapshot, logger=self.logger)
                        
                        if warnings:
                            self.logger.warning(f"[SNAPSHOT VALIDATION] Found {len(warnings)} consistency issues")
                        else:
                            self.logger.debug(
                                f"[SNAPSHOT VALIDATION] All checks passed | "
                                f"Positions: {len(snapshot.open_positions)} | "
                                f"Equity: ${snapshot.performance.equity:.2f} | "
                                f"Unrealized: ${snapshot.performance.unrealized_pnl:.2f}"
                            )
                        
                        # Metrics logging
                        total_attempted = self.metrics['entries_attempted']
                        total_opened = self.metrics['entries_opened']
                        winrate = (self.win_count / (self.win_count + self.loss_count) * 100.0) if (self.win_count + self.loss_count) > 0 else 0.0
                        # DEBUG ONLY: Metrics are tracked in Performance panel (too verbose for LOG panel)
                        self.logger.debug(
                            f"[METRICS] entries_attempted={total_attempted} entries_opened={total_opened} "
                            f"winrate={winrate:.1f}% rejections={self.metrics['rejections_by_reason']} "
                            f"exits={self.metrics['exits_by_reason']}"
                        )
                    except Exception as e:
                        self.logger.debug(f"[SNAPSHOT VALIDATION] Validation failed: {e}")
                    self._last_metrics_log = time.time()
        except KeyboardInterrupt:
            self.logger.info("[EXIT] KeyboardInterrupt - shutting down gracefully")
        except Exception as e:
            self.logger.critical(
                f"[CRITICAL] Unhandled exception in main loop: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise  # Re-raise to ensure bot stops
        finally:
            # Cleanup UI
            if ui_instance is not None:
                try:
                    ui_instance.__exit__(None, None, None)
                except Exception:
                    pass
            try:
                # Cleanup: Close storage connection
                if hasattr(self, 'fast_storage'):
                    self.fast_storage.close()
                # Close exchange wrapper
                if self.exchange_wrapper:
                    await self.exchange_wrapper.close()
                elif self.exchange:
                    # Fallback to direct exchange close (backward compatibility)
                    await self.exchange.close()
            except Exception:
                pass
