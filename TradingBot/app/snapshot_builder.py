"""
Engine Snapshot Builder for UI v2

Single source of truth for all UI data.
Collects state directly from engine structures without recomputation.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .accounting import (
    calculate_position_pnl,
    calculate_total_unrealized_pnl,
    get_current_price_from_bot,
    calculate_equity,
    calculate_balance,
)
from .config import MAX_OPEN_POSITIONS, MAX_CONCURRENT_POS, ACCOUNT_BAL, LEVERAGE_BASE
from .decision_event import get_recent_decision_events, DecisionEvent


@dataclass
class GlobalStatus:
    """Global bot status."""
    mode: str  # "DRY_RUN" or "LIVE"
    version: str
    exchange: str
    runtime_seconds: int
    loop_time_ms: float
    universe_size: int
    api_calls_per_min: int
    api_limit_per_min: int
    scan_number: int
    scanning_symbols: List[str] = field(default_factory=list)  # Symbols being scanned (all active symbols)
    exchange_connected: bool = True  # Exchange connection status
    fallback_mode: bool = False  # Whether in fallback mode
    next_scan_time: float = 0.0  # When next scan will occur
    next_universe_refresh_time: float = 0.0  # When next universe refresh will occur
    last_universe_refresh_time: float = 0.0  # When universe was last refreshed


@dataclass
class Performance:
    """Performance metrics."""
    start_balance: float
    balance: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    win_count: int
    loss_count: int
    win_rate: float  # 0-100%
    profit_factor: float
    gross_win: float
    gross_loss: float
    max_drawdown_pct: float
    total_costs: float
    entry_fees: float
    exit_fees: float
    slippage: float
    funding: float


@dataclass
class SignalHealth:
    """Signal generation health metrics."""
    signals_generated: int  # Total signals generated (cumulative)
    signals_approved: int  # Signals that passed all filters
    signals_rejected: int  # Signals that were blocked
    pass_rate: float  # 0-100% (approved / generated)
    avg_signal_score: float  # Average score of all signals (0-100)
    avg_latency_ms: float  # Average signal evaluation latency
    is_scanning: bool  # Is currently scanning?
    symbols_scanned: int  # Symbols scanned in last scan
    last_scan_time: float  # Time of last scan (timestamp)
    entries_attempted_this_scan: int = 0  # Entries attempted in current scan
    entries_opened_this_scan: int = 0  # Entries opened in current scan
    entries_attempted_total: int = 0  # Total entries attempted (cumulative)
    entries_opened_total: int = 0  # Total entries opened (cumulative)
    discovery_candidates: int = 0  # Discovery candidates found


@dataclass
class MLFrameworkStatus:
    """ML Framework status and metrics."""
    signal_scorer_loaded: bool  # Is ML scorer model loaded?
    signal_scorer_error: Optional[str]  # Error message if failed
    signal_scorer_predictions: int  # Total predictions made
    signal_scorer_avg_time_ms: float  # Average prediction time
    parameter_optimizer_available: bool  # Is parameter optimizer available?
    evolution_learner_available: bool  # Is evolution learner available?
    filter_stats: Dict[str, int]  # Unified filter statistics
    filter_pass_rate: float  # Filter pass rate (0-100%)


@dataclass
class RiskSnapshot:
    """Risk and market conditions."""
    open_positions: int
    max_positions: int
    btc_trend: str  # "UP", "DOWN", "FLAT"
    btc_trend_pct: float
    volatility_regime: str  # "Low", "Normal", "High"
    avg_stop_distance_pct: float
    current_regime: str  # "scalping", "day", "swing"
    market_regime: str  # Market regime (Bullish/Bearish/Neutral)
    exit_strategy: str  # "ATR", "Trailing", etc.
    loss_streak: int


@dataclass
class PositionSnapshot:
    """Single position snapshot."""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    pnl_value: float
    pnl_pct: float
    max_pct: Optional[float]
    score: float
    prs: Optional[float]  # Position Recovery Score
    age_str: str  # e.g., "45m", "2h15m"
    sl_pct: Optional[float]
    leverage: int
    size_pct: float
    corr: Optional[float]
    dd_time: Optional[str]  # e.g., "12m", "1h05m"
    is_unicorn: bool
    current_r: Optional[float] = None  # Current R multiple (from trailing engine)
    # Binance market data
    bid_price: Optional[float] = None  # Bid price from exchange
    ask_price: Optional[float] = None  # Ask price from exchange
    spread_bps: Optional[float] = None  # Bid-ask spread in basis points
    vol_24h_quote: Optional[float] = None  # 24h volume in quote currency
    pct_change_24h: Optional[float] = None  # 24h price change %
    stop_loss_price: Optional[float] = None  # Stop loss price level
    take_profit_price: Optional[float] = None  # Take profit price level


@dataclass
class ActivityEvent:
    """Recent activity event."""
    timestamp: datetime
    action: str  # "ENTRY", "EXIT", "PARTIAL_EXIT", "SCALE_OUT", "REJECT"
    symbol: str
    side: Optional[str]
    price: float
    pnl_value: Optional[float]
    pnl_pct: Optional[float]
    score: Optional[float]
    reason: Optional[str]
    is_unicorn: bool


@dataclass
class SignalItem:
    """Signal queue item."""
    symbol: str
    score: float
    side: str
    status: str  # "APPROVED", "REJECTED", "PENDING"
    reason: str

@dataclass
class ScanActivity:
    """Current scanning activity status."""
    is_scanning: bool
    symbols_scanned: int
    last_scan_symbols: List[str]  # Symbols checked in last scan
    last_scan_result: str  # Summary of last scan
    symbols_skipped_reasons: Dict[str, int]  # Skip reasons and counts
    signals_found: int  # Signals found in last scan
    entries_attempted: int  # Entries attempted in last scan


@dataclass
class LLMMessage:
    """LLM advisor message."""
    timestamp: datetime
    message: str
    priority: int


@dataclass
class MLFrameworkStatus:
    """ML Framework status and metrics."""
    signal_scorer_loaded: bool  # Is ML scorer model loaded?
    signal_scorer_error: Optional[str]  # Error message if failed
    signal_scorer_predictions: int  # Total predictions made
    signal_scorer_avg_time_ms: float  # Average prediction time
    parameter_optimizer_available: bool  # Is parameter optimizer available?
    evolution_learner_available: bool  # Is evolution learner available?
    filter_stats: Dict[str, int]  # Unified filter statistics
    filter_pass_rate: float  # Filter pass rate (0-100%)


@dataclass
class EngineSnapshot:
    """Complete engine state snapshot for UI v2."""
    timestamp: datetime
    global_status: GlobalStatus
    performance: Performance
    signal_health: SignalHealth
    risk: RiskSnapshot
    ml_framework: Optional[MLFrameworkStatus] = None  # ML Framework status
    open_positions: List[PositionSnapshot] = field(default_factory=list)
    recent_activity: List[ActivityEvent] = field(default_factory=list)
    signal_queue: List[SignalItem] = field(default_factory=list)
    llm_log: List[LLMMessage] = field(default_factory=list)
    scan_activity: Optional[ScanActivity] = None  # Current scanning activity


def _format_duration(seconds: int) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if minutes > 0:
            return f"{hours}h{minutes:02d}m"
        return f"{hours}h"


def build_engine_snapshot(bot, debug: bool = False) -> EngineSnapshot:
    """
    Build canonical engine snapshot from bot state.
    
    This is the SINGLE SOURCE OF TRUTH for UI v2.
    All data comes directly from engine structures.
    
    Args:
        bot: ScalperBot instance
        debug: Enable debug self-checks
        
    Returns:
        EngineSnapshot with complete engine state
    """
    from .logger import get_logger
    logger = get_logger("SnapshotBuilder")
    
    now = time.time()
    timestamp = datetime.utcfromtimestamp(now)
    
    # ═══════════════════════════════════════════════════════════
    # 1) GLOBAL STATUS
    # ═══════════════════════════════════════════════════════════
    
    from .config import DRY_RUN
    mode = "DRY_RUN" if DRY_RUN else "LIVE"
    version = "1.0"
    exchange = "BINANCE FUTURES"
    
    runtime_seconds = int(now - getattr(bot, "started_at", now))
    
    # Loop time from actual measurements
    loop_times = list(getattr(bot, "loop_times", []) or [])
    loop_time_ms = (sum(loop_times) / len(loop_times) * 1000.0) if loop_times else 0.0
    
    # Universe size
    universe = getattr(bot, "universe", None)
    universe_size = len(getattr(universe, "active", [])) if universe else 0
    
    # Get scanning symbols (all active symbols in universe)
    scanning_symbols = []
    if universe:
        active = getattr(universe, "active", [])
        # Show count instead of truncating to 10 symbols
        active_count = len(active)
        scanning_symbols = [f"All {active_count} symbols"] if active_count > 0 else []
    
    # API calls
    budget = getattr(bot, "budget", None)
    api_calls_per_min = 0
    api_limit_per_min = 300
    if budget:
        try:
            api_calls_per_min = int(float(getattr(budget, "rate_per_min", 0) or 0))
            api_limit_per_min = int(float(getattr(budget, "max_cpm", 300) or 300))
        except (ValueError, TypeError):
            pass
    
    # Scan number
    scan_stats = getattr(bot, "scan_stats", {}) or {}
    scan_number = int(scan_stats.get("total_scans", 0))
    
    # Exchange connection status
    exchange_wrapper = getattr(bot, "exchange_wrapper", None)
    exchange_connected = exchange_wrapper is not None
    fallback_mode = getattr(bot, "_fallback_mode", False)
    
    # Timing information
    next_scan_time = float(getattr(bot, "next_scan_time", 0) or 0)
    next_universe_refresh_time = float(getattr(bot, "next_universe_refresh_time", 0) or 0)
    last_universe_refresh_time = float(getattr(bot, "last_universe_refresh_time", 0) or 0)
    
    global_status = GlobalStatus(
        mode=mode,
        version=version,
        exchange=exchange,
        runtime_seconds=runtime_seconds,
        loop_time_ms=loop_time_ms,
        universe_size=universe_size,
        api_calls_per_min=api_calls_per_min,
        api_limit_per_min=api_limit_per_min,
        scan_number=scan_number,
        scanning_symbols=scanning_symbols or [],
        exchange_connected=exchange_connected,
        fallback_mode=fallback_mode,
        next_scan_time=next_scan_time,
        next_universe_refresh_time=next_universe_refresh_time,
        last_universe_refresh_time=last_universe_refresh_time,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 2) PERFORMANCE (CANONICAL ACCOUNTING)
    # ═══════════════════════════════════════════════════════════
    
    start_balance = getattr(bot, "start_equity", ACCOUNT_BAL)
    realized_pnl = float(getattr(bot, "realized_pnl_total", 0.0) or 0.0)
    realized_funding = float(getattr(bot, "realized_funding_total", 0.0) or 0.0)
    realized_entry_fees = float(getattr(bot, "realized_entry_fees_total", 0.0) or 0.0)
    realized_exit_fees = float(getattr(bot, "realized_exit_fees_total", 0.0) or 0.0)
    realized_slippage = float(getattr(bot, "realized_slippage_total", 0.0) or 0.0)
    
    # Calculate unrealized PnL
    positions_dict = getattr(bot, "positions", {}) or {}
    unrealized_pnl = 0.0
    if positions_dict:
        try:
            unrealized_pnl = calculate_total_unrealized_pnl(
                positions_dict,
                lambda symbol: get_current_price_from_bot(bot, symbol)
            )
        except Exception:
            pass
    
    # Calculate balance and equity using canonical accounting
    balance = calculate_balance(start_balance, realized_pnl, realized_funding)
    equity = calculate_equity(start_balance, realized_pnl, realized_funding, unrealized_pnl)
    
    # Update equity peak
    equity_peak = getattr(bot, "equity_peak", equity)
    if equity > equity_peak:
        equity_peak = equity
        bot.equity_peak = equity_peak
    
    max_drawdown_pct = ((equity - equity_peak) / equity_peak * 100.0) if equity_peak > 0 else 0.0
    max_drawdown_pct = min(0.0, max_drawdown_pct)
    
    # Trade statistics
    win_count = int(getattr(bot, "win_count", 0))
    loss_count = int(getattr(bot, "loss_count", 0))
    total_trades = win_count + loss_count
    win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0
    
    gross_win = float(getattr(bot, "gross_win", 0.0) or 0.0)
    gross_loss = float(getattr(bot, "gross_loss", 0.0) or 0.0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 1.0)
    
    total_costs = realized_entry_fees + realized_exit_fees + realized_slippage + realized_funding
    
    performance = Performance(
        start_balance=start_balance,
        balance=balance,
        equity=equity,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        gross_win=gross_win,
        gross_loss=gross_loss,
        max_drawdown_pct=max_drawdown_pct,
        total_costs=total_costs,
        entry_fees=realized_entry_fees,
        exit_fees=realized_exit_fees,
        slippage=realized_slippage,
        funding=realized_funding,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 3) SIGNAL HEALTH (wired to filter_stats_cumulative)
    # ═══════════════════════════════════════════════════════════
    
    # CRITICAL: Signal Health panel reads from cumulative filter stats
    # These stats are updated in bot.py scan_and_enter_signals() after each scan
    # - signals_total: Cumulative count of signals generated (passed signal generator)
    # - signals_passed: Cumulative count of signals that passed filter pipeline AND position manager
    # - signals_rejected: Cumulative count of rejected signals (signals_total - signals_passed)
    # Source: bot.filter_stats_cumulative (updated in bot.py line 2295-2297)
    filter_stats = getattr(bot, "filter_stats_cumulative", {}) or {}
    signals_generated = filter_stats.get('signals_total', 0) or 0
    signals_approved = filter_stats.get('signals_passed', 0) or 0
    signals_rejected = filter_stats.get('signals_rejected', 0) or 0
    
    # Fallback to signal_history if cumulative stats not available yet (e.g., during startup)
    # NOTE: This fallback is less accurate (only last 100 signals) but provides initial values
    if signals_generated == 0:
        signal_history = list(getattr(bot, "signal_history", []) or [])
        recent_signals = signal_history[-100:] if len(signal_history) > 100 else signal_history
        signals_generated = len(recent_signals)
        signals_approved = sum(1 for s in recent_signals if s.get("approved", False))
        signals_rejected = signals_generated - signals_approved
    
    # Calculate pass rate (0-100%) with defensive division
    pass_rate = (signals_approved / signals_generated * 100.0) if signals_generated > 0 else 0.0
    
    # Average signal score from cumulative stats
    # Source: Updated in bot.py line 2300-2303 (running average of all signal scores)
    avg_score_count = filter_stats.get('avg_score_count', 0) or 0
    avg_score_sum = filter_stats.get('avg_score_sum', 0.0) or 0.0
    avg_signal_score = (avg_score_sum / avg_score_count) if avg_score_count > 0 else 0.0
    
    # Fallback to signal_history if cumulative stats not available (e.g., during startup)
    if avg_signal_score == 0.0:
        signal_history = list(getattr(bot, "signal_history", []) or [])
        recent_signals = signal_history[-100:] if len(signal_history) > 100 else signal_history
        scores = [s.get("final_score", 0) for s in recent_signals if s.get("final_score")]
        avg_signal_score = (sum(scores) / len(scores)) if scores else 0.0
    
    # Average latency (from signal history if available)
    # NOTE: Latency tracking is optional - gracefully handle missing data
    signal_history = list(getattr(bot, "signal_history", []) or [])
    recent_signals = signal_history[-100:] if len(signal_history) > 100 else signal_history
    latencies = [s.get("latency_ms", 0) for s in recent_signals if s.get("latency_ms")]
    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 0.0
    
    # Get scanning status
    is_scanning = bool(getattr(bot, "is_scanning", False))
    last_scan_start = float(getattr(bot, "last_scan_start", 0) or 0)
    last_scan_time = last_scan_start
    
    # Get symbols scanned in last scan
    scan_history = list(getattr(bot, "scan_history", []) or [])
    symbols_scanned = 0
    if scan_history:
        last_scan = scan_history[-1] if scan_history else {}
        symbols_scanned = int(last_scan.get("symbols_processed", 0) or 0)
    
    # Entry statistics
    entries_attempted_this_scan = int(getattr(bot, "entries_attempted_this_scan", 0) or 0)
    entries_opened_this_scan = int(getattr(bot, "entries_opened_this_scan", 0) or 0)
    metrics = getattr(bot, "metrics", {}) or {}
    entries_attempted_total = int(metrics.get("entries_attempted", 0) or 0)
    entries_opened_total = int(metrics.get("entries_opened", 0) or 0)
    
    # Discovery candidates (if available from last scan)
    discovery_candidates = 0
    if scan_history:
        last_scan = scan_history[-1] if scan_history else {}
        discovery_candidates = int(last_scan.get("discovery_candidates", 0) or 0)
    
    # Build SignalHealth dataclass with defensive defaults
    # All fields are guaranteed to have valid values (0 for counts, 0.0 for rates/scores)
    signal_health = SignalHealth(
        signals_generated=signals_generated,
        signals_approved=signals_approved,
        signals_rejected=signals_rejected,
        pass_rate=pass_rate,
        avg_signal_score=avg_signal_score,
        avg_latency_ms=avg_latency_ms,
        is_scanning=is_scanning,
        symbols_scanned=symbols_scanned,
        last_scan_time=last_scan_time,
        entries_attempted_this_scan=entries_attempted_this_scan,
        entries_opened_this_scan=entries_opened_this_scan,
        entries_attempted_total=entries_attempted_total,
        entries_opened_total=entries_opened_total,
        discovery_candidates=discovery_candidates,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 4) RISK SNAPSHOT
    # ═══════════════════════════════════════════════════════════
    
    open_positions_count = len(positions_dict)
    max_positions = MAX_OPEN_POSITIONS
    
    # BTC trend
    btc_trend_raw = getattr(bot, "btc_trend", {})
    btc_trend_pct = 0.0
    if isinstance(btc_trend_raw, dict):
        btc_trend_pct = float(btc_trend_raw.get("trend_pct", 0.0) or 0.0)
    elif isinstance(btc_trend_raw, (list, tuple)) and btc_trend_raw:
        try:
            btc_trend_pct = float(btc_trend_raw[0])
        except (TypeError, ValueError):
            pass
    else:
        try:
            btc_trend_pct = float(btc_trend_raw or 0.0)
        except (TypeError, ValueError):
            pass
    
    if abs(btc_trend_pct) < 0.3:
        btc_trend = "FLAT"
    elif btc_trend_pct > 0:
        btc_trend = "UP"
    else:
        btc_trend = "DOWN"
    
    volatility_regime = getattr(bot, "volatility_regime", "Normal")
    
    # Average stop distance
    avg_stop_distance_pct = 0.0
    stop_distances = []
    for symbol, position in positions_dict.items():
        entry = position.get("entry_price", 0)
        sl = position.get("stop_loss", 0)
        side = position.get("side", "").lower()
        if entry > 0 and sl > 0:
            if side == "long":
                dist_pct = abs((entry - sl) / entry) * 100.0
            else:
                dist_pct = abs((sl - entry) / entry) * 100.0
            stop_distances.append(dist_pct)
    
    if stop_distances:
        avg_stop_distance_pct = sum(stop_distances) / len(stop_distances)
    
    # PURE SCALPER MODE: Always scalping, no regime switching
    current_regime = "scalping"  # Hard-locked to scalping
    
    # NEW: Get market regime from bot
    market_regime = getattr(bot, "market_regime", "neutral")

    from .config import USE_ATR_TRAILING_STOP
    exit_strategy = "ATR" if USE_ATR_TRAILING_STOP else "Trailing"
    
    loss_streak = int(getattr(bot.position_manager, "loss_streak", 0)) if hasattr(bot, "position_manager") else 0
    
    risk = RiskSnapshot(
        open_positions=open_positions_count,
        max_positions=max_positions,
        btc_trend=btc_trend,
        btc_trend_pct=btc_trend_pct,
        volatility_regime=volatility_regime,
        avg_stop_distance_pct=avg_stop_distance_pct,
        current_regime=current_regime,
        market_regime=market_regime,
        exit_strategy=exit_strategy,
        loss_streak=loss_streak,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 4) OPEN POSITIONS
    # ═══════════════════════════════════════════════════════════
    
    # CRITICAL: Open Positions table reads directly from bot.positions
    # bot.positions is a direct reference to position_registry._positions (bot.py line 142)
    # This ensures UI always shows current position state (no stale data)
    # Source: bot.positions (updated in bot.py when positions are opened/closed)
    open_positions = []
    for symbol, position in positions_dict.items():
        if not position:
            continue
        
        # Extract position data
        entry_price = float(position.get("entry_price", 0) or 0)
        size = float(position.get("size", 0) or 0)
        side = (position.get("side", "LONG") or "LONG").upper()
        leverage = int(position.get("leverage", LEVERAGE_BASE) or LEVERAGE_BASE)
        
        # Get current price
        current_price = get_current_price_from_bot(bot, symbol) or entry_price
        
        # Calculate PnL
        pnl_abs, pnl_pct = calculate_position_pnl(position, current_price)
        
        # Signal score - check multiple field names for compatibility
        signal_score = float(
            position.get("signal_score") or 
            position.get("score") or 
            position.get("final_score") or 
            0
        ) or 0
        # If score is in 0-1 range (normalized), convert to 0-100 range
        if signal_score > 0 and signal_score <= 1.0:
            signal_score = signal_score * 100.0
        
        # PRS
        recovery_score = position.get("recovery_score")
        if recovery_score is not None:
            recovery_score = float(recovery_score)
        
        # Age
        entry_time = float(position.get("entry_time", now) or now)
        age_seconds = int(now - entry_time)
        age_str = _format_duration(age_seconds)
        
        # Stop loss %
        stop_loss = position.get("stop_loss")
        sl_pct = None
        if stop_loss and entry_price > 0:
            stop_loss = float(stop_loss)
            if side == "LONG":
                sl_pct = abs((entry_price - stop_loss) / entry_price) * 100.0
            else:
                sl_pct = abs((stop_loss - entry_price) / entry_price) * 100.0
        
        # Max% (MFE)
        max_pct = None
        peak_price = position.get("peak_price")
        trough_price = position.get("trough_price")
        if entry_price > 0:
            if side == "LONG" and peak_price:
                try:
                    max_pct = ((float(peak_price) - entry_price) / entry_price) * 100.0
                except (TypeError, ValueError):
                    pass
            elif side == "SHORT" and trough_price:
                try:
                    max_pct = ((entry_price - float(trough_price)) / entry_price) * 100.0
                except (TypeError, ValueError):
                    pass
        
        # Fallback to stored peak_pnl
        if max_pct is None:
            peak_pnl = position.get("peak_pnl")
            if peak_pnl is not None:
                try:
                    max_pct = float(peak_pnl)
                except (TypeError, ValueError):
                    pass
        
        # Size%
        size_pct = 0.0
        if entry_price > 0 and size > 0 and equity > 0:
            notional_value = abs(entry_price * size)
            size_pct = (notional_value / equity) * 100.0
        
        # BTC correlation
        correlation = None
        if universe and btc_trend_pct != 0.0:
            try:
                stats = getattr(universe, "stats", {})
                symbol_stats = stats.get(symbol)
                if symbol_stats:
                    symbol_pct_change = float(getattr(symbol_stats, "pct_change_24h", 0.0) or 0.0)
                    if abs(btc_trend_pct) > 0.3 and abs(symbol_pct_change) > 0.5:
                        if (btc_trend_pct > 0 and symbol_pct_change > 0) or (btc_trend_pct < 0 and symbol_pct_change < 0):
                            corr_strength = min(abs(symbol_pct_change) / 3.0, 1.0)
                            correlation = 0.5 + (corr_strength * 0.5)
                        else:
                            corr_strength = min(abs(symbol_pct_change) / 3.0, 1.0)
                            correlation = 0.5 - (corr_strength * 0.5)
                        correlation = (correlation - 0.5) * 2.0
            except Exception:
                pass
        
        # Drawdown time
        dd_time_str = None
        peak_pnl_time = position.get("peak_pnl_time")
        if peak_pnl_time:
            try:
                dd_seconds = int(now - float(peak_pnl_time))
                dd_time_str = _format_duration(dd_seconds)
            except (TypeError, ValueError):
                pass
        
        # Unicorn status
        is_unicorn = bool(position.get("is_unicorn", False))
        
        # Current R multiple (from trailing engine)
        current_r = position.get("current_r")
        if current_r is not None:
            try:
                current_r = float(current_r)
            except (TypeError, ValueError):
                current_r = None
        
        # ═══════════════════════════════════════════════════════════
        # BINANCE MARKET DATA (First-hand exchange data)
        # ═══════════════════════════════════════════════════════════
        bid_price = None
        ask_price = None
        spread_bps = None
        vol_24h_quote = None
        pct_change_24h = None
        
        # Try to get market data from ticker cache
        ticker_cache = getattr(bot, "ticker_cache", None)
        if ticker_cache:
            try:
                ticker_data = ticker_cache.get(symbol, max_age=5.0)  # 5s max age
                if ticker_data:
                    bid_price = float(ticker_data.get("bid") or 0.0) or None
                    ask_price = float(ticker_data.get("ask") or 0.0) or None
                    vol_24h_quote = float(ticker_data.get("quoteVolume") or 0.0) or None
                    pct_change_24h = float(ticker_data.get("percentage") or 0.0) or None
            except Exception:
                pass
        
        # Calculate spread in BPS if we have bid/ask
        if bid_price and ask_price and bid_price > 0:
            spread_bps = ((ask_price - bid_price) / bid_price) * 10000.0
        
        # Get stop loss and take profit prices
        stop_loss_price = position.get("stop_loss")
        if stop_loss_price:
            try:
                stop_loss_price = float(stop_loss_price)
            except (TypeError, ValueError):
                stop_loss_price = None
        
        take_profit_price = position.get("take_profit")
        if take_profit_price:
            try:
                take_profit_price = float(take_profit_price) if float(take_profit_price) > 0 else None
            except (TypeError, ValueError):
                take_profit_price = None
        
        open_positions.append(PositionSnapshot(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            pnl_value=pnl_abs,
            pnl_pct=pnl_pct,
            max_pct=max_pct,
            score=signal_score,
            prs=recovery_score,
            age_str=age_str,
            sl_pct=sl_pct,
            leverage=leverage,
            size_pct=size_pct,
            corr=correlation,
            dd_time=dd_time_str,
            is_unicorn=is_unicorn,
            current_r=current_r,
            bid_price=bid_price,
            ask_price=ask_price,
            spread_bps=spread_bps,
            vol_24h_quote=vol_24h_quote,
            pct_change_24h=pct_change_24h,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        ))
    
    # Sort positions by importance
    def position_sort_key(pos):
        priority = 0
        if pos.prs is not None and pos.prs < 50:
            priority += 1000
        elif pos.prs is not None and pos.prs < 70:
            priority += 500
        if pos.pnl_pct < 0:
            priority += 100 - pos.pnl_pct
        if pos.is_unicorn:
            priority += 50
        return -priority
    
    open_positions.sort(key=position_sort_key)
    
    # ═══════════════════════════════════════════════════════════
    # 5) RECENT ACTIVITY (CANONICAL SOURCE)
    # ═══════════════════════════════════════════════════════════
    
    recent_activity = []
    try:
        decision_events = get_recent_decision_events(limit=15)
        
        for decision in decision_events:
            timestamp = datetime.utcfromtimestamp(decision.timestamp)
            
            recent_activity.append(ActivityEvent(
                timestamp=timestamp,
                action=decision.action,
                symbol=decision.symbol,
                side=decision.side,
                price=decision.price or 0.0,
                pnl_value=decision.net_pnl or decision.pnl_value,
                pnl_pct=decision.pnl_pct,
                score=decision.score,
                reason=decision.reason,
                is_unicorn=decision.is_unicorn,
            ))
    except Exception as e:
        logger.debug(f"Failed to get decision events for activity: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # 6) SIGNAL QUEUE
    # ═══════════════════════════════════════════════════════════
    
    # CRITICAL: Signal Queue shows recent rejected signals from signal_history
    # signal_history is populated in bot.py when signals are generated/rejected
    # - Approved signals: added with approved=True (bot.py line 1751)
    # - Rejected signals: added with approved=False (bot.py lines 1355, 1691)
    # Source: bot.signal_history (deque, maxlen=20, updated in bot.py)
    signal_queue = []
    signal_history = list(getattr(bot, "signal_history", []) or [])
    rejected_signals = [s for s in signal_history[-50:] if not s.get("approved", True)]
    
    for signal in rejected_signals[-10:]:
        symbol = signal.get("symbol", "?")
        score = float(signal.get("final_score", 0) or 0)
        side = signal.get("side", "?")
        rejection_reason = signal.get("rejection_reason", "Unknown")
        
        signal_queue.append(SignalItem(
            symbol=symbol,
            score=score,
            side=side,
            status="REJECTED",
            reason=rejection_reason,
        ))
    
    # ═══════════════════════════════════════════════════════════
    # 7) SCAN ACTIVITY
    # ═══════════════════════════════════════════════════════════
    
    # Get current scanning status and last scan details
    is_scanning = bool(getattr(bot, "is_scanning", False))
    scan_history = list(getattr(bot, "scan_history", []) or [])
    last_scan = scan_history[-1] if scan_history else {}
    
    symbols_scanned = int(last_scan.get("symbols_processed", 0) or 0)
    symbols_skipped_reasons = last_scan.get("symbols_skipped", {}) or {}
    signals_found = int(last_scan.get("signals_found", 0) or 0)
    entries_attempted = int(last_scan.get("entries_attempted", 0) or 0)
    
    # Get symbols checked in last scan (from scan history if available)
    last_scan_symbols = []
    total_symbols = int(last_scan.get("total_symbols", 0) or 0)
    # Show active symbols from universe (all symbols, not MARKSMAN subset)
    if universe:
        active = getattr(universe, "active", [])
        last_scan_symbols = [s.replace('/USDT', '')[:8] for s in active[:10]]  # Show first 10
    
    # Build last scan result summary
    last_scan_result = f"Scanned {symbols_scanned}/{total_symbols} symbols"
    if signals_found > 0:
        last_scan_result += f", {signals_found} signal(s) found"
    elif symbols_skipped_reasons:
        skip_reasons = ", ".join([f"{v} {k}" for k, v in symbols_skipped_reasons.items() if v > 0][:2])
        last_scan_result += f" ({skip_reasons})"
    
    scan_activity = ScanActivity(
        is_scanning=is_scanning,
        symbols_scanned=symbols_scanned,
        last_scan_symbols=last_scan_symbols[:10],  # Limit to 10 for display
        last_scan_result=last_scan_result,
        symbols_skipped_reasons=symbols_skipped_reasons,
        signals_found=signals_found,
        entries_attempted=entries_attempted,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 8) LLM LOG
    # ═══════════════════════════════════════════════════════════
    
    llm_log = []
    ctrl = getattr(bot, "ctrl", None)
    if ctrl:
        recent_info = list(getattr(ctrl, "recent_info", []) or [])[-10:]
        for info in recent_info:
            if isinstance(info, dict):
                message = info.get("message", str(info))
                priority = int(info.get("priority", 50))
                info_timestamp = info.get("timestamp", now)
                timestamp = datetime.utcfromtimestamp(float(info_timestamp))
                
                llm_log.append(LLMMessage(
                    timestamp=timestamp,
                    message=message,
                    priority=priority,
                ))
    
    # ═══════════════════════════════════════════════════════════
    # 7) ML FRAMEWORK STATUS
    # ═══════════════════════════════════════════════════════════
    
    ml_framework_status = None
    try:
        from .ml.unified_framework import get_unified_ml_framework
        framework = get_unified_ml_framework()
        status = framework.get_status()
        
        # Get ML scorer details
        scorer = framework.get_signal_scorer()
        signal_scorer_loaded = status['signal_scorer']['loaded']
        signal_scorer_error = status['signal_scorer']['error']
        signal_scorer_predictions = getattr(scorer, 'prediction_count', 0)
        signal_scorer_avg_time_ms = (
            (getattr(scorer, 'total_prediction_time', 0.0) / signal_scorer_predictions * 1000.0)
            if signal_scorer_predictions > 0 else 0.0
        )
        
        # Get filter statistics from UnifiedFilter
        filter_stats = {}
        filter_pass_rate = 0.0
        try:
            position_manager = getattr(bot, 'position_manager', None)
            if position_manager and hasattr(position_manager, 'unified_filter'):
                unified_filter = position_manager.unified_filter
                # Get filter stats from signal generator if available
                signal_gen = getattr(bot, 'signal_generator', None)
                if signal_gen and hasattr(signal_gen, '_filter_stats'):
                    filter_stats = dict(signal_gen._filter_stats)
                    total = filter_stats.get('accepted', 0) + filter_stats.get('rejected_hard_min', 0) + \
                            filter_stats.get('rejected_percentile', 0) + filter_stats.get('rejected', 0)
                    if total > 0:
                        filter_pass_rate = (filter_stats.get('accepted', 0) / total * 100.0)
        except Exception:
            pass
        
        ml_framework_status = MLFrameworkStatus(
            signal_scorer_loaded=signal_scorer_loaded,
            signal_scorer_error=signal_scorer_error,
            signal_scorer_predictions=signal_scorer_predictions,
            signal_scorer_avg_time_ms=signal_scorer_avg_time_ms,
            parameter_optimizer_available=status['parameter_optimizer']['available'],
            evolution_learner_available=status['evolution_learner']['available'],
            filter_stats=filter_stats,
            filter_pass_rate=filter_pass_rate,
        )
    except Exception:
        # ML framework not available - create empty status
        ml_framework_status = MLFrameworkStatus(
            signal_scorer_loaded=False,
            signal_scorer_error="Framework not initialized",
            signal_scorer_predictions=0,
            signal_scorer_avg_time_ms=0.0,
            parameter_optimizer_available=False,
            evolution_learner_available=False,
            filter_stats={},
            filter_pass_rate=0.0,
        )
    
    # ═══════════════════════════════════════════════════════════
    # BUILD SNAPSHOT
    # ═══════════════════════════════════════════════════════════
    
    snapshot = EngineSnapshot(
        timestamp=timestamp,
        global_status=global_status,
        performance=performance,
        signal_health=signal_health,
        risk=risk,
        ml_framework=ml_framework_status,
        open_positions=open_positions,
        recent_activity=recent_activity,
        signal_queue=signal_queue,
        llm_log=llm_log,
        scan_activity=scan_activity,
    )
    
    # ═══════════════════════════════════════════════════════════
    # DEBUG SELF-CHECKS
    # ═══════════════════════════════════════════════════════════
    
    if debug:
        # Check 1: Position PnL sum ≈ Unrealized
        positions_pnl_sum = sum(p.pnl_value for p in snapshot.open_positions)
        diff = abs(positions_pnl_sum - snapshot.performance.unrealized_pnl)
        if diff > 1.0:
            logger.warning(
                f"[SELF-CHECK] PnL mismatch: Position sum=${positions_pnl_sum:.2f}, "
                f"Unrealized=${snapshot.performance.unrealized_pnl:.2f}, Diff=${diff:.2f}"
            )
        
        # Check 2: Equity = Balance + Unrealized
        calculated_equity = snapshot.performance.balance + snapshot.performance.unrealized_pnl
        diff = abs(calculated_equity - snapshot.performance.equity)
        if diff > 0.01:
            logger.warning(
                f"[SELF-CHECK] Equity mismatch: Balance+Unrealized=${calculated_equity:.2f}, "
                f"Equity=${snapshot.performance.equity:.2f}, Diff=${diff:.2f}"
            )
        
        # Check 3: Position count consistency
        if snapshot.risk.open_positions != len(snapshot.open_positions):
            logger.warning(
                f"[SELF-CHECK] Position count mismatch: "
                f"risk.open_positions={snapshot.risk.open_positions}, "
                f"len(open_positions)={len(snapshot.open_positions)}"
            )
        
        # Check 4: Max positions enforcement
        if snapshot.risk.open_positions > snapshot.risk.max_positions:
            logger.warning(
                f"[SELF-CHECK] Position cap exceeded: "
                f"{snapshot.risk.open_positions} > {snapshot.risk.max_positions}"
            )
    
    return snapshot

