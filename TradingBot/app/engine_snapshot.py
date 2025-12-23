"""
Canonical Engine Snapshot Module

Single source of truth for all UI data. This module collects a complete snapshot
of the trading engine state for display in the Rich UI.

All UI panels MUST use this snapshot and NOT recompute any derived values.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from .accounting import (
    calculate_position_pnl,
    calculate_total_unrealized_pnl,
    get_current_price_from_bot,
    calculate_equity,
    calculate_balance,
)
from .config import (
    DRY_RUN,
    MAX_OPEN_POSITIONS,
    MAX_CONCURRENT_POS,
    ACCOUNT_BAL,
    LEVERAGE_BASE,
)


@dataclass
class GlobalStatus:
    """Global status information."""
    mode: str  # "DRY_RUN" or "LIVE"
    version: str
    exchange: str
    runtime_seconds: int
    universe_size: int  # Actual number of symbols scanner is considering
    scan_number: int  # Total scans completed
    loop_latency_ms: float  # Measured loop time
    api_calls_last_min: int  # Actual API calls in last minute
    api_limit_per_min: int  # API rate limit


@dataclass
class PerformanceMetrics:
    """Performance and accounting metrics."""
    start_balance: float
    balance: float  # Balance = start + realized_pnl + funding (no unrealized)
    equity: float  # Equity = balance + unrealized_pnl
    realized_pnl: float  # Net realized PnL (after fees)
    unrealized_pnl: float  # Sum of all position unrealized PnL
    realized_funding: float
    realized_entry_fees: float
    realized_exit_fees: float
    realized_slippage: float
    total_costs: float
    win_count: int
    loss_count: int
    win_rate: float  # 0-100%
    profit_factor: float
    gross_win: float
    gross_loss: float
    max_drawdown_pct: float
    equity_peak: float


@dataclass
class SignalHealth:
    """Signal generation health metrics."""
    signals_generated: int  # Total signals generated (last N scans)
    signals_approved: int  # Signals that passed all filters
    signals_rejected: int  # Signals that were blocked
    pass_rate: float  # 0-100% (approved / generated)
    avg_signal_score: float  # Average score of all signals (0-100)
    avg_latency_ms: float  # Average signal evaluation latency


@dataclass
class MarketSnapshot:
    """Market regime and conditions."""
    volatility_regime: str  # "Low", "Normal", "High"
    spread_regime: str  # "Tight", "Normal", "Wide"
    btc_trend_direction: str  # "UP", "DOWN", "FLAT"
    btc_trend_pct: float  # BTC 24h change %
    avg_stop_distance_pct: float  # Average SL distance across positions
    pnl_momentum: float  # Recent PnL momentum (from recent trades)


@dataclass
class PositionView:
    """Single position view (canonical)."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    current_price: float
    size: float  # Position size (contracts/quantity)
    leverage: int
    pnl_abs: float  # Unrealized PnL (absolute)
    pnl_pct: float  # Unrealized PnL (%)
    max_pct: Optional[float]  # Maximum profit % reached (MFE)
    signal_score: float  # Original signal score (0-100)
    recovery_score: Optional[float]  # PRS (Position Recovery Score)
    age_seconds: int  # Time since entry
    stop_loss: Optional[float]
    stop_loss_pct: Optional[float]  # SL distance as % of entry
    take_profit: Optional[float]
    size_pct: float  # Position size as % of equity (notional / equity * 100)
    correlation: Optional[float]  # BTC correlation (-1 to +1)
    drawdown_time_seconds: Optional[int]  # Time since peak PnL
    is_unicorn: bool
    current_r: Optional[float] = None
    trailing_stop: Optional[float] = None


@dataclass
class RecentEvent:
    """Recent trading event."""
    timestamp: datetime
    event_type: str  # "ENTRY", "EXIT", "REJECT"
    symbol: str
    side: Optional[str]
    price: float
    pnl: Optional[float]  # For exits
    score: Optional[float]  # For entries
    reason: Optional[str]  # For exits/rejects
    is_unicorn: bool
    # Fields with default values must come last
    pnl_pct: Optional[float] = None  # For exits (PnL percentage)
    is_partial: bool = False  # For partial exits


@dataclass
class SignalQueueItem:
    """Signal in the queue."""
    symbol: str
    score: float  # 0-100
    side: str
    status: str  # "APPROVED", "REJECTED", "PENDING"
    rejection_reason: Optional[str]


@dataclass
class LLMMessage:
    """LLM advisory message."""
    timestamp: datetime
    message: str
    priority: int  # 0-100


@dataclass
class EngineSnapshot:
    """Complete engine state snapshot for UI."""
    timestamp: datetime
    
    # Core sections
    global_status: GlobalStatus
    performance: PerformanceMetrics
    signal_health: SignalHealth
    market: MarketSnapshot
    
    # Risk/Config (must come before fields with defaults)
    max_positions: int  # MAX_OPEN_POSITIONS from config
    current_positions: int  # Actual open positions count
    loss_streak: int
    current_regime: str  # "scalping", "day", "swing"
    exit_strategy: str  # "ATR", "Trailing", "Fixed"
    
    # Collections (fields with defaults must come last)
    open_positions: List[PositionView] = field(default_factory=list)
    recent_events: List[RecentEvent] = field(default_factory=list)
    signal_queue: List[SignalQueueItem] = field(default_factory=list)
    llm_messages: List[LLMMessage] = field(default_factory=list)


def collect_engine_snapshot(bot) -> EngineSnapshot:
    """
    Collect complete engine state snapshot.
    
    This is the SINGLE SOURCE OF TRUTH for all UI data.
    All UI panels MUST use this snapshot and NOT recompute values.
    
    Args:
        bot: ScalperBot instance
        
    Returns:
        EngineSnapshot with all engine state
    """
    now = time.time()
    timestamp = datetime.utcfromtimestamp(now)
    
    # ═══════════════════════════════════════════════════════════
    # 1) GLOBAL STATUS
    # ═══════════════════════════════════════════════════════════
    
    mode = "DRY_RUN" if DRY_RUN else "LIVE"
    version = "5.4"
    exchange = "BINANCE FUTURES"
    
    runtime_seconds = int(now - getattr(bot, "started_at", now))
    
    # Universe size = actual symbols scanner is considering
    universe = getattr(bot, "universe", None)
    universe_size = len(getattr(universe, "active", [])) if universe else 0
    
    # Scan number from actual scan stats
    scan_stats = getattr(bot, "scan_stats", {}) or {}
    scan_number = int(scan_stats.get("total_scans", 0))
    
    # Loop latency from actual measurements
    loop_times = list(getattr(bot, "loop_times", []) or [])
    loop_latency_ms = (sum(loop_times) / len(loop_times) * 1000.0) if loop_times else 0.0
    
    # API calls from actual budget tracker
    budget = getattr(bot, "budget", None)
    api_calls_last_min = 0
    api_limit_per_min = 300
    if budget:
        try:
            api_calls_last_min = int(float(getattr(budget, "rate_per_min", 0) or 0))
            api_limit_per_min = int(float(getattr(budget, "max_cpm", 300) or 300))
        except (ValueError, TypeError):
            pass
    
    global_status = GlobalStatus(
        mode=mode,
        version=version,
        exchange=exchange,
        runtime_seconds=runtime_seconds,
        universe_size=universe_size,
        scan_number=scan_number,
        loop_latency_ms=loop_latency_ms,
        api_calls_last_min=api_calls_last_min,
        api_limit_per_min=api_limit_per_min,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 2) PERFORMANCE METRICS (CANONICAL ACCOUNTING)
    # ═══════════════════════════════════════════════════════════
    
    start_balance = getattr(bot, "start_equity", ACCOUNT_BAL)
    realized_pnl = float(getattr(bot, "realized_pnl_total", 0.0) or 0.0)
    realized_funding = float(getattr(bot, "realized_funding_total", 0.0) or 0.0)
    realized_entry_fees = float(getattr(bot, "realized_entry_fees_total", 0.0) or 0.0)
    realized_exit_fees = float(getattr(bot, "realized_exit_fees_total", 0.0) or 0.0)
    realized_slippage = float(getattr(bot, "realized_slippage_total", 0.0) or 0.0)
    
    # Calculate unrealized PnL from open positions
    positions_dict = getattr(bot, "positions", {}) or {}
    unrealized_pnl = 0.0
    if positions_dict:
        try:
            unrealized_pnl = calculate_total_unrealized_pnl(
                positions_dict,
                lambda symbol: get_current_price_from_bot(bot, symbol)
            )
        except Exception:
            pass  # Fallback to 0.0
    
    # Calculate balance and equity using canonical accounting
    balance = calculate_balance(start_balance, realized_pnl, realized_funding)
    equity = calculate_equity(start_balance, realized_pnl, realized_funding, unrealized_pnl)
    
    # Update equity peak
    equity_peak = getattr(bot, "equity_peak", equity)
    if equity > equity_peak:
        equity_peak = equity
        bot.equity_peak = equity_peak
    
    max_drawdown_pct = ((equity - equity_peak) / equity_peak * 100.0) if equity_peak > 0 else 0.0
    max_drawdown_pct = min(0.0, max_drawdown_pct)  # Only negative values
    
    # Trade statistics
    win_count = int(getattr(bot, "win_count", 0))
    loss_count = int(getattr(bot, "loss_count", 0))
    total_trades = win_count + loss_count
    win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0
    
    gross_win = float(getattr(bot, "gross_win", 0.0) or 0.0)
    gross_loss = float(getattr(bot, "gross_loss", 0.0) or 0.0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 1.0)
    
    total_costs = realized_entry_fees + realized_exit_fees + realized_slippage + realized_funding
    
    performance = PerformanceMetrics(
        start_balance=start_balance,
        balance=balance,
        equity=equity,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        realized_funding=realized_funding,
        realized_entry_fees=realized_entry_fees,
        realized_exit_fees=realized_exit_fees,
        realized_slippage=realized_slippage,
        total_costs=total_costs,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        gross_win=gross_win,
        gross_loss=gross_loss,
        max_drawdown_pct=max_drawdown_pct,
        equity_peak=equity_peak,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 3) SIGNAL HEALTH
    # ═══════════════════════════════════════════════════════════
    
    # FIX: Wire Signal Health to real filter stats from [FILTER] scan#N logs
    # Use cumulative filter stats instead of signal_history (which only keeps last 20)
    filter_stats = getattr(bot, "filter_stats_cumulative", {})
    signals_generated = filter_stats.get('signals_total', 0)
    signals_approved = filter_stats.get('signals_passed', 0)
    signals_rejected = filter_stats.get('signals_rejected', 0)
    
    # Fallback to signal_history if cumulative stats not available yet
    if signals_generated == 0:
        signal_history = list(getattr(bot, "signal_history", []) or [])
        recent_signals = signal_history[-100:] if len(signal_history) > 100 else signal_history
        signals_generated = len(recent_signals)
        signals_approved = sum(1 for s in recent_signals if s.get("approved", False))
        signals_rejected = signals_generated - signals_approved
    
    pass_rate = (signals_approved / signals_generated * 100.0) if signals_generated > 0 else 0.0
    
    # Average signal score from cumulative stats
    avg_score_count = filter_stats.get('avg_score_count', 0)
    avg_score_sum = filter_stats.get('avg_score_sum', 0.0)
    avg_signal_score = (avg_score_sum / avg_score_count) if avg_score_count > 0 else 0.0
    
    # Fallback to signal_history if cumulative stats not available
    if avg_signal_score == 0.0:
        signal_history = list(getattr(bot, "signal_history", []) or [])
        recent_signals = signal_history[-100:] if len(signal_history) > 100 else signal_history
        scores = [s.get("final_score", 0) for s in recent_signals if s.get("final_score")]
        avg_signal_score = (sum(scores) / len(scores)) if scores else 0.0
    
    # Average latency (from signal history if available)
    signal_history = list(getattr(bot, "signal_history", []) or [])
    recent_signals = signal_history[-100:] if len(signal_history) > 100 else signal_history
    latencies = [s.get("latency_ms", 0) for s in recent_signals if s.get("latency_ms")]
    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 0.0
    
    signal_health = SignalHealth(
        signals_generated=signals_generated,
        signals_approved=signals_approved,
        signals_rejected=signals_rejected,
        pass_rate=pass_rate,
        avg_signal_score=avg_signal_score,
        avg_latency_ms=avg_latency_ms,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 4) MARKET SNAPSHOT
    # ═══════════════════════════════════════════════════════════
    
    volatility_regime = getattr(bot, "volatility_regime", "Normal")
    spread_regime = getattr(bot, "spread_regime", "Normal")
    
    # BTC trend from bot's calculation
    btc_trend_raw = getattr(bot, "btc_trend", {})
    btc_trend_pct = 0.0
    btc_trend_direction = "FLAT"
    if isinstance(btc_trend_raw, dict):
        btc_trend_pct = float(btc_trend_raw.get("trend_pct", 0.0) or 0.0)
    elif isinstance(btc_trend_raw, (list, tuple)) and btc_trend_raw:
        try:
            btc_trend_pct = float(btc_trend_raw[0])
        except (TypeError, ValueError):
            btc_trend_pct = 0.0
    else:
        try:
            btc_trend_pct = float(btc_trend_raw or 0.0)
        except (TypeError, ValueError):
            btc_trend_pct = 0.0
    
    if abs(btc_trend_pct) < 0.3:
        btc_trend_direction = "FLAT"
    elif btc_trend_pct > 0:
        btc_trend_direction = "UP"
    else:
        btc_trend_direction = "DOWN"
    
    # Average stop distance across positions
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
    
    # PnL momentum from recent trades
    recent_trades = list(getattr(bot, "recent_trades", []) or [])[-10:]
    pnl_momentum = 0.0
    if recent_trades:
        trade_pnls = [t.get("pnl", 0) for t in recent_trades if t.get("pnl") is not None]
        if trade_pnls:
            pnl_momentum = sum(trade_pnls) / len(trade_pnls)
    
    market = MarketSnapshot(
        volatility_regime=volatility_regime,
        spread_regime=spread_regime,
        btc_trend_direction=btc_trend_direction,
        btc_trend_pct=btc_trend_pct,
        avg_stop_distance_pct=avg_stop_distance_pct,
        pnl_momentum=pnl_momentum,
    )
    
    # ═══════════════════════════════════════════════════════════
    # 5) OPEN POSITIONS (CANONICAL)
    # ═══════════════════════════════════════════════════════════
    
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
        
        # Calculate PnL using canonical helper
        pnl_abs, pnl_pct = calculate_position_pnl(position, current_price)
        
        # Signal score (original entry score)
        signal_score = float(position.get("signal_score", position.get("score", 0)) or 0)
        if signal_score <= 1.0:
            signal_score = signal_score * 100.0  # Convert 0-1 to 0-100
        
        # PRS (Position Recovery Score) - use stored value, not recomputed
        recovery_score = position.get("recovery_score")
        if recovery_score is not None:
            recovery_score = float(recovery_score)
        
        # Age
        entry_time = float(position.get("entry_time", now) or now)
        age_seconds = int(now - entry_time)
        
        # Stop loss
        stop_loss = position.get("stop_loss")
        stop_loss_pct = None
        if stop_loss and entry_price > 0:
            stop_loss = float(stop_loss)
            if side == "LONG":
                stop_loss_pct = abs((entry_price - stop_loss) / entry_price) * 100.0
            else:
                stop_loss_pct = abs((stop_loss - entry_price) / entry_price) * 100.0
        
        # Take profit
        take_profit = position.get("take_profit")
        if take_profit:
            take_profit = float(take_profit)
        
        # Max% (MFE - Maximum Favorable Excursion)
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
        
        # Size% = notional value / equity * 100
        size_pct = 0.0
        if entry_price > 0 and size > 0 and equity > 0:
            notional_value = abs(entry_price * size)
            size_pct = (notional_value / equity) * 100.0
        
        # Current R multiple based on initial stop
        current_r = None
        initial_stop_price = position.get("initial_stop_price", position.get("stop_loss"))
        if initial_stop_price is None:
            initial_stop_price = position.get("stop_loss")
        if initial_stop_price is not None and entry_price and initial_stop_price != entry_price:
            risk_unit = abs(entry_price - initial_stop_price)
            if risk_unit > 0:
                if side == "LONG":
                    current_r = (current_price - entry_price) / risk_unit
                else:
                    current_r = (entry_price - current_price) / risk_unit
        
        # BTC correlation
        correlation = None
        if universe and btc_trend_pct != 0.0:
            try:
                stats = getattr(universe, "stats", {})
                symbol_stats = stats.get(symbol)
                if symbol_stats:
                    symbol_pct_change = float(getattr(symbol_stats, "pct_change_24h", 0.0) or 0.0)
                    if abs(btc_trend_pct) > 0.3 and abs(symbol_pct_change) > 0.5:
                        # Same direction = positive correlation
                        if (btc_trend_pct > 0 and symbol_pct_change > 0) or (btc_trend_pct < 0 and symbol_pct_change < 0):
                            corr_strength = min(abs(symbol_pct_change) / 3.0, 1.0)
                            correlation = 0.5 + (corr_strength * 0.5)  # 0.5-1.0
                        else:
                            corr_strength = min(abs(symbol_pct_change) / 3.0, 1.0)
                            correlation = 0.5 - (corr_strength * 0.5)  # 0.0-0.5
                        # Convert to -1.0 to +1.0 range
                        correlation = (correlation - 0.5) * 2.0
            except Exception:
                pass
        
        # Drawdown time (time since peak PnL)
        drawdown_time_seconds = None
        peak_pnl_time = position.get("peak_pnl_time")
        if peak_pnl_time:
            try:
                drawdown_time_seconds = int(now - float(peak_pnl_time))
            except (TypeError, ValueError):
                pass
        
        # Unicorn status
        is_unicorn = bool(position.get("is_unicorn", False))
        
        open_positions.append(PositionView(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            size=size,
            leverage=leverage,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            max_pct=max_pct,
            signal_score=signal_score,
            recovery_score=recovery_score,
            age_seconds=age_seconds,
            stop_loss=stop_loss,
            stop_loss_pct=stop_loss_pct,
            take_profit=take_profit,
            size_pct=size_pct,
            correlation=correlation,
            drawdown_time_seconds=drawdown_time_seconds,
            is_unicorn=is_unicorn,
            current_r=current_r,
            trailing_stop=position.get('stop_loss'),
        ))
    
    # Sort positions by importance (struggling first, then unicorns, then by PnL)
    def position_sort_key(pos):
        priority = 0
        if pos.recovery_score is not None and pos.recovery_score < 50:
            priority += 1000
        elif pos.recovery_score is not None and pos.recovery_score < 70:
            priority += 500
        if pos.pnl_pct < 0:
            priority += 100 - pos.pnl_pct
        if pos.is_unicorn:
            priority += 50
        return -priority
    
    open_positions.sort(key=position_sort_key)
    
    # ═══════════════════════════════════════════════════════════
    # 6) RECENT EVENTS (CANONICAL SOURCE)
    # ═══════════════════════════════════════════════════════════
    
    recent_events = []
    
    # Use canonical DecisionEvent buffer as single source of truth
    try:
        from .decision_event import get_recent_decision_events
        decision_events = get_recent_decision_events(limit=20)
        
        for decision in decision_events:
            # Map DecisionEvent to RecentEvent format
            timestamp = datetime.utcfromtimestamp(decision.timestamp)
            
            # Map action to event_type
            event_type = decision.action
            if decision.action == "PARTIAL_EXIT":
                event_type = "EXIT"  # Display as EXIT for UI consistency
            elif decision.action == "SCALE_OUT":
                event_type = "EXIT"  # Display as EXIT for UI consistency
            
            # Clean symbol (remove /USDT if present)
            symbol = (decision.symbol or "?").replace("/USDT", "")
            
            recent_events.append(RecentEvent(
                timestamp=timestamp,
                event_type=event_type,
                symbol=symbol,
                side=decision.side,
                price=decision.price or 0.0,
                pnl=decision.net_pnl or decision.pnl_value,
                pnl_pct=decision.pnl_pct,
                score=decision.score,
                reason=decision.reason,
                is_unicorn=decision.is_unicorn,
                is_partial=decision.action in ("PARTIAL_EXIT", "SCALE_OUT"),
            ))
    except Exception as e:
        # Fallback to legacy recent_trades if decision events not available
        from .logger import get_logger
        log = get_logger("EngineSnapshot")
        log.debug(f"Failed to use DecisionEvent buffer, falling back to recent_trades: {e}")
        
        trades = list(getattr(bot, "recent_trades", []) or [])[-15:]
        
        for trade in trades:
            trade_type = (trade.get("type", "unknown") or "unknown").upper()
            symbol = (trade.get("symbol", "?") or "?").replace("/USDT", "")
            side = trade.get("side")
            price = float(trade.get("price", trade.get("entry_price", trade.get("exit_price", 0))) or 0)
            pnl = trade.get("pnl")
            if pnl is not None:
                pnl = float(pnl)
            score = trade.get("final_score", trade.get("score"))
            if score is not None:
                score = float(score)
            reason = trade.get("reason")
            is_unicorn = bool(trade.get("is_unicorn", False))
            is_partial = bool(trade.get("is_partial", False))
            
            timestamp = datetime.utcfromtimestamp(float(trade.get("timestamp", now) or now))
            
            # Calculate PnL percentage from trade if available
            pnl_pct = None
            if trade.get('pnl_pct') is not None:
                try:
                    pnl_pct = float(trade['pnl_pct'])
                except (TypeError, ValueError):
                    pass
            
            recent_events.append(RecentEvent(
                timestamp=timestamp,
                event_type=trade_type,
                symbol=symbol,
                side=side,
                price=price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                score=score,
                reason=reason,
                is_unicorn=is_unicorn,
                is_partial=is_partial,
            ))
    
    # Sort by timestamp (newest first)
    recent_events.sort(key=lambda e: e.timestamp, reverse=True)
    
    # ═══════════════════════════════════════════════════════════
    # 7) SIGNAL QUEUE
    # ═══════════════════════════════════════════════════════════
    
    signal_queue = []
    # Get rejected signals from signal history
    rejected_signals = [s for s in signal_history[-50:] if not s.get("approved", True)]
    
    for signal in rejected_signals[-20:]:  # Last 20 rejected
        symbol = signal.get("symbol", "?")
        score = float(signal.get("final_score", 0) or 0)
        side = signal.get("side", "?")
        rejection_reason = signal.get("rejection_reason", "Unknown")
        
        signal_queue.append(SignalQueueItem(
            symbol=symbol,
            score=score,
            side=side,
            status="REJECTED",
            rejection_reason=rejection_reason,
        ))
    
    # ═══════════════════════════════════════════════════════════
    # 8) LLM MESSAGES
    # ═══════════════════════════════════════════════════════════
    
    llm_messages = []
    ctrl = getattr(bot, "ctrl", None)
    if ctrl:
        recent_info = list(getattr(ctrl, "recent_info", []) or [])[-10:]
        for info in recent_info:
            if isinstance(info, dict):
                message = info.get("message", str(info))
                priority = int(info.get("priority", 50))
                info_timestamp = info.get("timestamp", now)
                timestamp = datetime.utcfromtimestamp(float(info_timestamp))
                
                llm_messages.append(LLMMessage(
                    timestamp=timestamp,
                    message=message,
                    priority=priority,
                ))
    
    # ═══════════════════════════════════════════════════════════
    # 9) RISK/CONFIG
    # ═══════════════════════════════════════════════════════════
    
    max_positions = MAX_OPEN_POSITIONS
    current_positions = len(positions_dict)
    loss_streak = int(getattr(bot.position_manager, "loss_streak", 0)) if hasattr(bot, "position_manager") else 0
    current_regime = getattr(bot, "current_regime", "scalping")
    
    from .config import USE_ATR_TRAILING_STOP
    exit_strategy = "ATR" if USE_ATR_TRAILING_STOP else "Trailing"
    
    # ═══════════════════════════════════════════════════════════
    # BUILD SNAPSHOT
    # ═══════════════════════════════════════════════════════════
    
    snapshot = EngineSnapshot(
        timestamp=timestamp,
        global_status=global_status,
        performance=performance,
        signal_health=signal_health,
        market=market,
        open_positions=open_positions,
        recent_events=recent_events,
        signal_queue=signal_queue,
        llm_messages=llm_messages,
        max_positions=max_positions,
        current_positions=current_positions,
        loss_streak=loss_streak,
        current_regime=current_regime,
        exit_strategy=exit_strategy,
    )
    
    return snapshot


def validate_snapshot_consistency(snapshot: EngineSnapshot, logger=None) -> List[str]:
    """
    Validate snapshot consistency and return list of warnings.
    
    Args:
        snapshot: EngineSnapshot to validate
        logger: Optional logger for warnings
        
    Returns:
        List of warning messages (empty if all checks pass)
    """
    warnings = []
    
    # Check 1: Position PnL sum ≈ Unrealized
    positions_pnl_sum = sum(p.pnl_abs for p in snapshot.open_positions)
    diff = abs(positions_pnl_sum - snapshot.performance.unrealized_pnl)
    if diff > 1.0:  # More than $1 difference
        msg = f"PnL mismatch: Position sum=${positions_pnl_sum:.2f}, Unrealized=${snapshot.performance.unrealized_pnl:.2f}, Diff=${diff:.2f}"
        warnings.append(msg)
        if logger:
            logger.warning(f"[SNAPSHOT VALIDATION] {msg}")
    
    # Check 2: Equity = Balance + Unrealized
    calculated_equity = snapshot.performance.balance + snapshot.performance.unrealized_pnl
    diff = abs(calculated_equity - snapshot.performance.equity)
    if diff > 0.01:  # More than 1 cent difference
        msg = f"Equity mismatch: Balance+Unrealized=${calculated_equity:.2f}, Equity=${snapshot.performance.equity:.2f}, Diff=${diff:.2f}"
        warnings.append(msg)
        if logger:
            logger.warning(f"[SNAPSHOT VALIDATION] {msg}")
    
    # Check 3: Position count consistency
    if snapshot.current_positions != len(snapshot.open_positions):
        msg = f"Position count mismatch: current_positions={snapshot.current_positions}, open_positions list={len(snapshot.open_positions)}"
        warnings.append(msg)
        if logger:
            logger.warning(f"[SNAPSHOT VALIDATION] {msg}")
    
    # Check 4: Max positions enforcement
    if snapshot.current_positions > snapshot.max_positions:
        msg = f"Position cap exceeded: {snapshot.current_positions} > {snapshot.max_positions}"
        warnings.append(msg)
        if logger:
            logger.warning(f"[SNAPSHOT VALIDATION] {msg}")
    
    # Check 5: Size% sanity check (sum should be reasonable)
    total_size_pct = sum(p.size_pct for p in snapshot.open_positions)
    if total_size_pct > 500.0:  # More than 500% seems wrong
        msg = f"Total Size% seems high: {total_size_pct:.1f}%"
        warnings.append(msg)
        if logger:
            logger.warning(f"[SNAPSHOT VALIDATION] {msg}")
    
    # Check 6: PRS consistency (if PRS is displayed, it should match stored value)
    # This is validated by using stored PRS in snapshot, not recomputing
    
    return warnings

