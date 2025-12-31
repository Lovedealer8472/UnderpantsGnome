"""
Btop-Style Trading Bot UI
--------------------------
A modern, accurate btop-style terminal interface for monitoring the trading bot.
Features real-time graphs, system stats, position monitoring, and activity feed.

Usage:
    The bot automatically uses this UI when USE_RICH_UI is enabled.
"""

import sys
import time
import shutil
from collections import deque
from typing import Optional, Deque
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create dummy psutil
    class DummyPsutil:
        @staticmethod
        def cpu_percent(interval=None, percpu=False):
            return 0.0 if not percpu else [0.0]
        @staticmethod
        def cpu_count():
            return 1
        @staticmethod
        def virtual_memory():
            class Mem:
                percent = 0.0
                used = 0
                total = 0
            return Mem()
        @staticmethod
        def disk_usage(path):
            class Disk:
                percent = 0.0
                used = 0
                total = 0
            return Disk()
    psutil = DummyPsutil()

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn

from .snapshot_builder import build_engine_snapshot


class GraphBuffer:
    """Circular buffer for graph data."""
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: Deque[float] = deque(maxlen=max_size)
    
    def add(self, value: float):
        self.data.append(value)
    
    def get_bars(self, width: int = 30, min_val: float = 0.0, max_val: float = 100.0) -> str:
        """Convert buffer to bar graph string."""
        if not self.data:
            return "░" * width
        
        # Normalize values to 0-1 range
        if max_val == min_val:
            normalized = [0.5] * len(self.data)
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in self.data]
            # Clamp values to [0, 1]
            normalized = [max(0.0, min(1.0, v)) for v in normalized]
        
        # Sample data points to fit width
        if len(normalized) <= width:
            samples = list(normalized)
            # Pad with empty bars if needed
            while len(samples) < width:
                samples.append(0.0)
        else:
            step = len(normalized) / width
            samples = [normalized[int(i * step)] for i in range(width)]
        
        # Convert to bars
        bars = []
        for val in samples:
            bar_height = int(val * 8)  # 8 levels (0-7)
            bar_height = max(0, min(7, bar_height))  # Clamp to 0-7
            if bar_height >= 7:
                bars.append("█")
            elif bar_height >= 6:
                bars.append("▇")
            elif bar_height >= 5:
                bars.append("▆")
            elif bar_height >= 4:
                bars.append("▅")
            elif bar_height >= 3:
                bars.append("▄")
            elif bar_height >= 2:
                bars.append("▃")
            elif bar_height >= 1:
                bars.append("▂")
            else:
                bars.append("░")
        
        return "".join(bars)


class KeyboardHandler:
    """Non-blocking keyboard input handler."""
    def __init__(self):
        self.input_queue = None
        self.stop_flag = None
        self.thread = None
        if sys.platform == 'win32':
            try:
                import queue
                import threading
                import msvcrt
                self.input_queue = queue.Queue()
                self.stop_flag = threading.Event()
                self.thread = threading.Thread(target=self._listen_windows, daemon=True)
                self.thread.start()
            except Exception:
                pass
        else:
            try:
                import queue
                import threading
                import select
                import tty
                import termios
                self.input_queue = queue.Queue()
                self.stop_flag = threading.Event()
                self.thread = threading.Thread(target=self._listen_unix, daemon=True)
                self.thread.start()
            except Exception:
                pass
    
    def stop(self):
        if self.stop_flag:
            self.stop_flag.set()
    
    def get_key(self) -> Optional[str]:
        if not self.input_queue:
            return None
        try:
            return self.input_queue.get_nowait()
        except Exception:
            return None
    
    def _listen_windows(self):
        try:
            import msvcrt
            while not self.stop_flag.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    self.input_queue.put(key)
                self.stop_flag.wait(0.1)
        except Exception:
            pass
    
    def _listen_unix(self):
        try:
            import select
            import tty
            import termios
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while not self.stop_flag.is_set():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        self.input_queue.put(key)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


class BtopUI:
    """
    Btop-style Trading Bot UI (Unified ML Framework)
    
    Features:
    - Real-time system resource monitoring (CPU, RAM, Disk)
    - Performance graphs (PnL, Equity, API usage)
    - Position monitoring with detailed metrics
    - Signal activity feed
    - Performance statistics
    """
    
    def __init__(self):
        self.console = Console(
            file=sys.__stdout__,
            force_terminal=True,
            legacy_windows=False
        )
        self._live = None
        self._keyboard = KeyboardHandler()
        self._last_render_time = 0.0
        self._term_size = (0, 0)
        self._update_terminal_size()
        
        # Graph buffers for real-time visualization
        self.cpu_graph = GraphBuffer(max_size=50)
        self.mem_graph = GraphBuffer(max_size=50)
        self.pnl_graph = GraphBuffer(max_size=50)
        self.equity_graph = GraphBuffer(max_size=50)
        self.api_graph = GraphBuffer(max_size=50)
        
        # Track min/max for graphs
        self.equity_min = None
        self.equity_max = None
        self.pnl_min = -100.0
        self.pnl_max = 100.0
    
    def _update_terminal_size(self):
        """Update terminal size cache."""
        try:
            term_size = shutil.get_terminal_size()
            current_size = (term_size.columns, term_size.lines)
            
            if current_size == self._term_size:
                return
            
            self._term_size = current_size
            self.term_width = term_size.columns
            self.term_height = term_size.lines
        except Exception:
            self.term_width = 120
            self.term_height = 40
        
        # Calculate panel sizes
        self.header_height = 4
        self.footer_height = 3
        self.main_height = max(10, self.term_height - self.header_height - self.footer_height)
    
    def __enter__(self):
        self._update_terminal_size()
        self._keyboard = KeyboardHandler()
        self._live = Live(
            console=self.console,
            screen=True,
            auto_refresh=False,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._keyboard.stop()
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None
    
    def render(self, bot):
        """Render the UI with current bot state."""
        if not self._live:
            return
        
        # Throttle rendering (max 4 FPS)
        now = time.time()
        if now - self._last_render_time < 0.25:
            return
        
        self._last_render_time = now
        
        # Check for quit key
        key = self._keyboard.get_key()
        if key == 'q':
            raise KeyboardInterrupt("User pressed 'Q'")
        
        # Update terminal size
        self._update_terminal_size()
        
        # Build snapshot
        snapshot = build_engine_snapshot(bot)
        
        # Update graph buffers
        self._update_graphs(snapshot)
        
        # Build layout
        layout = self._build_layout(snapshot)
        
        # Update live display
        self._live.update(layout, refresh=True)
    
    def _update_graphs(self, snapshot):
        """Update graph buffers with current data."""
        # CPU
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_graph.add(cpu_percent)
        
        # Memory
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            self.mem_graph.add(mem.percent)
        
        # PnL
        total_pnl = snapshot.performance.realized_pnl + snapshot.performance.unrealized_pnl
        self.pnl_graph.add(total_pnl)
        if total_pnl < self.pnl_min:
            self.pnl_min = total_pnl
        if total_pnl > self.pnl_max:
            self.pnl_max = total_pnl
        
        # Equity
        equity = snapshot.performance.equity
        if self.equity_min is None:
            self.equity_min = equity
            self.equity_max = equity
        else:
            if equity < self.equity_min:
                self.equity_min = equity
            if equity > self.equity_max:
                self.equity_max = equity
        self.equity_graph.add(equity)
        
        # API usage
        api_pct = (snapshot.global_status.api_calls_per_min / snapshot.global_status.api_limit_per_min * 100.0) if snapshot.global_status.api_limit_per_min > 0 else 0.0
        self.api_graph.add(api_pct)
    
    def _build_layout(self, snapshot) -> Layout:
        """Build the complete layout."""
        root = Layout()
        root.split_column(
            Layout(name="header", size=self.header_height),
            Layout(name="main", size=self.main_height),
            Layout(name="footer", size=self.footer_height)
        )
        
        # Header
        root["header"].update(self._build_header(snapshot))
        
        # Main area: split into left (positions + activity) and right (stats + graphs + ml)
        root["main"].split_row(
            Layout(name="left", ratio=6),
            Layout(name="right", ratio=4)
        )
        
        # Left: positions and activity
        root["main"]["left"].split_column(
            Layout(name="positions", size=None),
            Layout(name="activity", size=None)
        )
        
        # Right: stats, ML status, and graphs
        root["main"]["right"].split_column(
            Layout(name="stats", size=None),
            Layout(name="ml_status", size=None),
            Layout(name="graphs", size=None)
        )
        
        # Update all panels
        root["main"]["left"]["positions"].update(self._build_positions_panel(snapshot))
        root["main"]["left"]["activity"].update(self._build_activity_panel(snapshot))
        root["main"]["right"]["stats"].update(self._build_stats_panel(snapshot))
        root["main"]["right"]["ml_status"].update(self._build_ml_status_panel(snapshot))
        root["main"]["right"]["graphs"].update(self._build_graphs_panel(snapshot))
        root["footer"].update(self._build_footer(snapshot))
        
        return root
    
    def _build_header(self, snapshot) -> Panel:
        """Build btop-style header with system and bot status."""
        try:
            # System stats
            cpu_percent = psutil.cpu_percent(interval=0.1) if PSUTIL_AVAILABLE else 0.0
            cpu_count = psutil.cpu_count() if PSUTIL_AVAILABLE else 1
            mem = psutil.virtual_memory() if PSUTIL_AVAILABLE else None
            disk = psutil.disk_usage('/') if PSUTIL_AVAILABLE else None
            
            # Bot status
            mode = snapshot.global_status.mode
            runtime = snapshot.global_status.runtime_seconds
            hours, rem = divmod(runtime, 3600)
            minutes, seconds = divmod(rem, 60)
            runtime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            # Performance
            total_pnl = snapshot.performance.realized_pnl + snapshot.performance.unrealized_pnl
            equity = snapshot.performance.equity
            win_rate = snapshot.performance.win_rate
            profit_factor = snapshot.performance.profit_factor
            
            # Colors
            cpu_color = "green" if cpu_percent < 50 else "yellow" if cpu_percent < 80 else "red"
            mem_color = "green" if mem and mem.percent < 60 else "yellow" if mem and mem.percent < 80 else "red"
            disk_color = "green" if disk and disk.percent < 70 else "yellow" if disk and disk.percent < 85 else "red"
            pnl_color = "bright_green" if total_pnl >= 0 else "bright_red"
            mode_color = "yellow" if mode == "DRY_RUN" else "bright_green"
            
            # Build header text
            text = Text()
            text.append("🖥️  UniRabbit Trading Bot", style="bold cyan")
            text.append("  |  ", style="dim")
            text.append(f"[{mode_color}]{mode}[/{mode_color}]", style=mode_color)
            text.append(f"  |  Runtime: {runtime_str}", style="blue")
            text.append("\n")
            
            # System stats line
            text.append("CPU: ", style="dim")
            text.append(f"{cpu_percent:5.1f}%", style=cpu_color)
            text.append(f" ({cpu_count} cores)", style="dim")
            text.append("  |  ", style="dim")
            
            if mem:
                text.append("RAM: ", style="dim")
                text.append(f"{mem.percent:5.1f}%", style=mem_color)
                text.append(f" ({mem.used / (1024**3):.1f}G/{mem.total / (1024**3):.1f}G)", style="dim")
                text.append("  |  ", style="dim")
            
            if disk:
                text.append("Disk: ", style="dim")
                text.append(f"{disk.percent:5.1f}%", style=disk_color)
                text.append("  |  ", style="dim")
            
            # Bot metrics
            text.append("Equity: ", style="dim")
            text.append(f"${equity:.2f}", style="bold white")
            text.append("  |  ", style="dim")
            text.append("PnL: ", style="dim")
            text.append(f"{'+' if total_pnl >= 0 else ''}${total_pnl:.2f}", style=f"bold {pnl_color}")
            text.append("  |  ", style="dim")
            text.append(f"WR: {win_rate:.1f}%", style="cyan")
            text.append("  |  ", style="dim")
            text.append(f"PF: {profit_factor:.2f}", style="magenta")
            
            # ML Framework status (brief)
            if snapshot.ml_framework:
                ml = snapshot.ml_framework
                ml_status = "✓" if ml.signal_scorer_loaded else "✗"
                ml_color = "green" if ml.signal_scorer_loaded else "red"
                text.append("  |  ", style="dim")
                text.append("ML: ", style="dim")
                text.append(ml_status, style=ml_color)
                if ml.signal_scorer_predictions > 0:
                    text.append(f" ({ml.signal_scorer_predictions:,})", style="dim")
            
            return Panel(text, border_style="cyan", box=box.ROUNDED, height=4)
        except Exception as e:
            fallback = Text()
            fallback.append("🖥️  UniRabbit Trading Bot", style="bold cyan")
            fallback.append("  |  System stats unavailable", style="red")
            return Panel(fallback, border_style="cyan", box=box.ROUNDED, height=4)
    
    def _build_positions_panel(self, snapshot) -> Panel:
        """Build positions table with Binance market data."""
        table = Table(expand=True, box=box.SIMPLE_HEAD, show_lines=False, show_header=True)
        # Main columns
        table.add_column("SYM", style="bold white", width=8)
        table.add_column("SIDE", justify="center", width=5)
        
        # Market data columns
        table.add_column("CURRENT", justify="right", width=10)
        table.add_column("BID/ASK", justify="center", width=14)
        table.add_column("SPREAD", justify="right", width=7)
        table.add_column("24hVOL", justify="right", width=10)
        table.add_column("CHG24h", justify="right", width=7)
        
        # Position P&L
        table.add_column("PNL $", justify="right", width=9)
        table.add_column("PNL %", justify="right", width=8)
        
        # Levels
        table.add_column("SL", justify="right", width=9)
        table.add_column("TP", justify="right", width=9)
        
        # Risk/Metrics
        table.add_column("SCORE", justify="center", width=6)
        table.add_column("R", justify="right", width=6)
        table.add_column("AGE", justify="right", width=6)
        table.add_column("LEV", justify="center", width=4)
        
        positions = snapshot.open_positions
        
        if not positions:
            # Show empty state
            empty_text = Text("\n\nNO ACTIVE POSITIONS\nSCANNING FOR TARGETS...", justify="center", style="dim")
            return Panel(
                empty_text,
                title="[bold]POSITIONS[/bold]",
                border_style="blue",
                box=box.ROUNDED
            )
        
        for pos in positions[:10]:  # Limit to 10 positions (wider table)
            # Color coding
            pnl_c = "green" if pos.pnl_value >= 0 else "red"
            side_c = "green" if pos.side == "LONG" else "red"
            
            # Score color
            if pos.score >= 70:
                score_c = "bold magenta"
            elif pos.score >= 55:
                score_c = "bold green"
            elif pos.score >= 45:
                score_c = "green"
            elif pos.score >= 30:
                score_c = "yellow"
            else:
                score_c = "red"
            
            # R multiple color
            r_str = f"{pos.current_r:.1f}" if pos.current_r is not None else "-"
            r_color = "green" if pos.current_r and pos.current_r > 1.0 else "yellow" if pos.current_r and pos.current_r > 0 else "red"
            
            # Format market data
            current_price_str = f"${pos.current_price:.4g}" if pos.current_price else "-"
            
            # Bid/Ask display
            if pos.bid_price and pos.ask_price:
                bid_ask_str = f"${pos.bid_price:.4g}/${pos.ask_price:.4g}"
            else:
                bid_ask_str = "-"
            
            # Spread display (BPS)
            if pos.spread_bps is not None:
                spread_str = f"{pos.spread_bps:.1f}bps"
                spread_color = "green" if pos.spread_bps < 5 else "yellow" if pos.spread_bps < 20 else "red"
            else:
                spread_str = "-"
                spread_color = "dim"
            
            # Volume display
            if pos.vol_24h_quote and pos.vol_24h_quote > 0:
                vol_m = pos.vol_24h_quote / 1_000_000
                if vol_m > 1000:
                    vol_str = f"${vol_m/1000:.0f}B"
                else:
                    vol_str = f"${vol_m:.0f}M"
            else:
                vol_str = "-"
            
            # 24h change
            if pos.pct_change_24h is not None:
                chg_str = f"{pos.pct_change_24h:+.1f}%"
                chg_color = "green" if pos.pct_change_24h >= 0 else "red"
            else:
                chg_str = "-"
                chg_color = "dim"
            
            # Stop loss and take profit
            sl_str = f"${pos.stop_loss_price:.4g}" if pos.stop_loss_price else "-"
            tp_str = f"${pos.take_profit_price:.4g}" if pos.take_profit_price else "-"
            
            table.add_row(
                pos.symbol.replace("/USDT", "")[:8],
                Text(pos.side[:1], style=side_c),
                # Market data
                current_price_str,
                bid_ask_str,
                Text(spread_str, style=spread_color),
                vol_str,
                Text(chg_str, style=chg_color),
                # P&L
                Text(f"{pos.pnl_value:+.2f}", style=pnl_c),
                Text(f"{pos.pnl_pct:+.2f}%", style=pnl_c),
                # Levels
                sl_str,
                tp_str,
                # Risk/Metrics
                Text(f"{pos.score:.0f}", style=score_c),
                Text(r_str, style=r_color if pos.current_r is not None else "dim"),
                pos.age_str,
                Text(str(pos.leverage), style="cyan" if pos.leverage > 1 else "dim")
            )
        
        return Panel(table, title=f"[bold]POSITIONS ({len(positions)}) - BINANCE MARKET DATA[/bold]", border_style="blue", box=box.ROUNDED)
    
    def _build_activity_panel(self, snapshot) -> Panel:
        """Build activity feed."""
        content = Text()
        
        activities = snapshot.recent_activity[:10]  # Last 10 activities
        
        if not activities:
            content.append("No recent activity...", style="dim")
        else:
            for activity in activities:
                time_str = activity.timestamp.strftime("%H:%M:%S")
                action = activity.action
                symbol = activity.symbol.replace("/USDT", "")[:8]
                
                # Color coding
                if action == "ENTRY":
                    action_color = "bold green"
                    icon = "▶"
                elif action == "EXIT":
                    action_color = "bold yellow"
                    icon = "◀"
                elif action == "PARTIAL_EXIT":
                    action_color = "yellow"
                    icon = "◁"
                else:
                    action_color = "dim"
                    icon = "•"
                
                content.append(f"{time_str} ", style="dim")
                content.append(f"{icon} ", style=action_color)
                content.append(f"{symbol:<8} ", style="bold white")
                content.append(f"{action:<12} ", style=action_color)
                
                if activity.pnl_value is not None:
                    pnl_color = "green" if activity.pnl_value >= 0 else "red"
                    content.append(f"${activity.pnl_value:+.2f}", style=pnl_color)
                
                if activity.reason:
                    content.append(f" ({activity.reason[:20]})", style="dim")
                
                content.append("\n")
        
        return Panel(content, title="[bold]ACTIVITY[/bold]", border_style="cyan", box=box.ROUNDED)
    
    def _build_stats_panel(self, snapshot) -> Panel:
        """Build performance statistics panel."""
        perf = snapshot.performance
        health = snapshot.signal_health
        risk = snapshot.risk
        ml = snapshot.ml_framework
        
        lines = []
        lines.append("📊 Performance")
        lines.append("")
        lines.append(f"Equity:     ${perf.equity:.2f}")
        lines.append(f"Balance:    ${perf.balance:.2f}")
        lines.append(f"Realized:   ${perf.realized_pnl:+.2f}")
        lines.append(f"Unrealized: ${perf.unrealized_pnl:+.2f}")
        lines.append("")
        lines.append(f"Win Rate:   {perf.win_rate:.1f}%")
        lines.append(f"Profit F:   {perf.profit_factor:.2f}")
        lines.append(f"Wins:       {perf.win_count}")
        lines.append(f"Losses:     {perf.loss_count}")
        lines.append("")
        lines.append("📡 Signals")
        lines.append("")
        lines.append(f"Generated:  {health.signals_generated}")
        lines.append(f"Approved:   {health.signals_approved}")
        lines.append(f"Rejected:   {health.signals_rejected}")
        lines.append(f"Pass Rate:  {health.pass_rate:.1f}%")
        lines.append(f"Avg Score:  {health.avg_signal_score:.1f}")
        
        # ML Framework Status
        if ml:
            lines.append("")
            lines.append("🤖 ML Framework")
            lines.append("")
            scorer_status = "✓" if ml.signal_scorer_loaded else "✗"
            scorer_color = "green" if ml.signal_scorer_loaded else "red"
            lines.append(f"Scorer:     [{scorer_color}]{scorer_status}[/{scorer_color}]")
            if ml.signal_scorer_predictions > 0:
                lines.append(f"Predictions: {ml.signal_scorer_predictions}")
                lines.append(f"Avg Time:    {ml.signal_scorer_avg_time_ms:.2f}ms")
            if ml.signal_scorer_error:
                error_short = ml.signal_scorer_error[:20] + "..." if len(ml.signal_scorer_error) > 20 else ml.signal_scorer_error
                lines.append(f"Error:       {error_short}")
            if ml.filter_pass_rate > 0:
                lines.append(f"Filter Pass: {ml.filter_pass_rate:.1f}%")
        
        lines.append("")
        lines.append("⚡ Risk")
        lines.append("")
        lines.append(f"Positions:  {risk.open_positions}/{risk.max_positions}")
        lines.append(f"Loss Streak: {risk.loss_streak}")
        lines.append(f"BTC Trend:  {risk.btc_trend}")
        lines.append(f"Volatility: {risk.volatility_regime}")
        
        text = "\n".join(lines)
        return Panel(text, title="[bold]STATS[/bold]", border_style="green", box=box.ROUNDED)
    
    def _build_graphs_panel(self, snapshot) -> Panel:
        """Build graphs panel."""
        lines = []
        
        # CPU graph
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_color = "green" if cpu_percent < 50 else "yellow" if cpu_percent < 80 else "red"
            cpu_bars = self.cpu_graph.get_bars(width=30, min_val=0.0, max_val=100.0)
            lines.append(f"CPU [{cpu_percent:5.1f}%]")
            lines.append(f"[{cpu_color}]{cpu_bars}[/{cpu_color}]")
            lines.append("")
        
        # Memory graph
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            mem_color = "green" if mem.percent < 60 else "yellow" if mem.percent < 80 else "red"
            mem_bars = self.mem_graph.get_bars(width=30, min_val=0.0, max_val=100.0)
            lines.append(f"RAM [{mem.percent:5.1f}%]")
            lines.append(f"[{mem_color}]{mem_bars}[/{mem_color}]")
            lines.append("")
        
        # Equity graph
        equity = snapshot.performance.equity
        if self.equity_min is not None and self.equity_max is not None and self.equity_max > self.equity_min:
            equity_bars = self.equity_graph.get_bars(width=30, min_val=self.equity_min, max_val=self.equity_max)
            equity_color = "green" if equity >= snapshot.performance.start_balance else "red"
            lines.append(f"Equity [${equity:.2f}]")
            lines.append(f"[{equity_color}]{equity_bars}[/{equity_color}]")
            lines.append("")
        
        # PnL graph
        total_pnl = snapshot.performance.realized_pnl + snapshot.performance.unrealized_pnl
        if self.pnl_max > self.pnl_min:
            pnl_bars = self.pnl_graph.get_bars(width=30, min_val=self.pnl_min, max_val=self.pnl_max)
            pnl_color = "green" if total_pnl >= 0 else "red"
            lines.append(f"PnL [${total_pnl:+.2f}]")
            lines.append(f"[{pnl_color}]{pnl_bars}[/{pnl_color}]")
            lines.append("")
        
        # API usage graph
        api_pct = (snapshot.global_status.api_calls_per_min / snapshot.global_status.api_limit_per_min * 100.0) if snapshot.global_status.api_limit_per_min > 0 else 0.0
        api_color = "green" if api_pct < 50 else "yellow" if api_pct < 80 else "red"
        api_bars = self.api_graph.get_bars(width=30, min_val=0.0, max_val=100.0)
        lines.append(f"API [{api_pct:5.1f}%]")
        lines.append(f"[{api_color}]{api_bars}[/{api_color}]")
        
        text = "\n".join(lines)
        return Panel(text, title="[bold]GRAPHS[/bold]", border_style="magenta", box=box.ROUNDED)
    
    def _build_ml_status_panel(self, snapshot) -> Panel:
        """Build ML Framework status panel."""
        ml = snapshot.ml_framework
        
        if not ml:
            return Panel(
                "ML Framework: Not Available",
                title="[bold]ML FRAMEWORK[/bold]",
                border_style="yellow",
                box=box.ROUNDED
            )
        
        lines = []
        
        # Signal Scorer Status
        scorer_status = "✓ LOADED" if ml.signal_scorer_loaded else "✗ FAILED"
        scorer_color = "green" if ml.signal_scorer_loaded else "red"
        lines.append(f"Scorer: [{scorer_color}]{scorer_status}[/{scorer_color}]")
        
        if ml.signal_scorer_loaded:
            lines.append(f"Predictions: {ml.signal_scorer_predictions:,}")
            if ml.signal_scorer_avg_time_ms > 0:
                lines.append(f"Avg Time: {ml.signal_scorer_avg_time_ms:.2f}ms")
        else:
            if ml.signal_scorer_error:
                error_short = ml.signal_scorer_error[:30] + "..." if len(ml.signal_scorer_error) > 30 else ml.signal_scorer_error
                lines.append(f"Error: {error_short}")
        
        lines.append("")
        
        # Component Availability
        opt_status = "✓" if ml.parameter_optimizer_available else "✗"
        evo_status = "✓" if ml.evolution_learner_available else "✗"
        lines.append(f"Optimizer: [{('green' if ml.parameter_optimizer_available else 'dim')}]{opt_status}[/]")
        lines.append(f"Evolution: [{('green' if ml.evolution_learner_available else 'dim')}]{evo_status}[/]")
        
        lines.append("")
        
        # Filter Statistics
        if ml.filter_stats:
            lines.append("Filters:")
            accepted = ml.filter_stats.get('accepted', 0)
            rejected = ml.filter_stats.get('rejected_hard_min', 0) + \
                      ml.filter_stats.get('rejected_percentile', 0) + \
                      ml.filter_stats.get('rejected', 0)
            total = accepted + rejected
            if total > 0:
                lines.append(f"  Pass: {accepted} ({ml.filter_pass_rate:.1f}%)")
                lines.append(f"  Fail: {rejected}")
        
        text = "\n".join(lines)
        return Panel(text, title="[bold]ML FRAMEWORK[/bold]", border_style="cyan", box=box.ROUNDED)
    
    def _build_footer(self, snapshot) -> Panel:
        """Build footer with status information."""
        text = Text()
        
        # Bot status
        mode = snapshot.global_status.mode
        mode_color = "yellow" if mode == "DRY_RUN" else "bright_green"
        text.append(f"Mode: ", style="dim")
        text.append(f"{mode}", style=mode_color)
        text.append("  |  ", style="dim")
        
        # Exchange
        exchange = snapshot.global_status.exchange
        text.append(f"Exchange: {exchange}", style="cyan")
        text.append("  |  ", style="dim")
        
        # Universe
        universe_size = snapshot.global_status.universe_size
        text.append(f"Universe: {universe_size} symbols", style="blue")
        text.append("  |  ", style="dim")
        
        # Scans
        scan_number = snapshot.global_status.scan_number
        text.append(f"Scans: {scan_number}", style="yellow")
        text.append("  |  ", style="dim")
        
        # Loop latency
        loop_time = snapshot.global_status.loop_time_ms
        loop_color = "green" if loop_time < 100 else "yellow" if loop_time < 500 else "red"
        text.append(f"Latency: ", style="dim")
        text.append(f"{loop_time:.1f}ms", style=loop_color)
        text.append("  |  ", style="dim")
        
        # API calls
        api_calls = snapshot.global_status.api_calls_per_min
        api_limit = snapshot.global_status.api_limit_per_min
        text.append(f"API: {api_calls}/{api_limit}/min", style="magenta")
        text.append("  |  ", style="dim")
        
        # Help
        text.append("Press 'Q' to quit", style="dim")
        
        return Panel(text, border_style="blue", box=box.ROUNDED, height=3)

