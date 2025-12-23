"""
UI v3 - The Command Center
Direct-Link TUI with Live Scan Feed ("The Tape") and Visual PnL.
"""

import sys
import shutil
import threading
import queue
import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn

from .snapshot_builder import build_engine_snapshot

class KeyboardHandler:
    """Non-blocking keyboard input handler."""
    
    def __init__(self):
        self.input_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread = None
    
    def start(self):
        if sys.platform == 'win32':
            self.thread = threading.Thread(target=self._listen_windows, daemon=True)
        else:
            self.thread = threading.Thread(target=self._listen_unix, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.stop_flag.set()
    
    def get_key(self) -> Optional[str]:
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
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

class UIv3:
    """
    UI v3 - Command Center
    Focus on Motion, Tape, and Visual Status.
    """
    
    def __init__(self):
        self.console = Console(
            file=sys.__stdout__,
            force_terminal=True,
            legacy_windows=False
        )
        self._live = None
        self._keyboard = KeyboardHandler()
        self._last_snapshot = None
        self._last_render_time = 0.0
        self._term_size = (0, 0)  # Cache size to prevent jittery resizing
        self._update_terminal_size()
    
    def _update_terminal_size(self):
        try:
            term_size = shutil.get_terminal_size()
            current_size = (term_size.columns, term_size.lines)
            
            # OPTIMIZATION: Only recalculate if size actually changed
            if current_size == self._term_size:
                return
                
            self._term_size = current_size
            self.term_width = term_size.columns
            self.term_height = term_size.lines
        except Exception:
            self.term_width = 120
            self.term_height = 40
            
        self.header_height = 3
        self.bottom_height = 8
        self.main_height = max(10, self.term_height - self.header_height - self.bottom_height)
    
    def __enter__(self):
        self._update_terminal_size()
        self._keyboard.start()
        self._live = Live(
            console=self.console,
            screen=True,
            auto_refresh=False, # We manually control refresh
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
        if not self._live: return
        
        # JITTER FIX: Smart Throttling
        # Only refresh if content changed significantly or sufficient time passed
        now = time.time()
        
        # 1. Hard Throttle (Max 4 FPS = 250ms)
        if now - self._last_render_time < 0.25:
            return
            
        self._last_render_time = now
        
        # Keyboard check
        key = self._keyboard.get_key()
        if key == 'q':
            raise KeyboardInterrupt("User pressed 'Q'")
        
        # Check terminal size
        self._update_terminal_size()
            
        snapshot = build_engine_snapshot(bot)
        layout = self._build_layout(bot, snapshot)
        
        # 2. Live Update with minimal flicker
        self._live.update(layout, refresh=True)
        
    def _build_layout(self, bot, snapshot) -> Layout:
        root = Layout()
        root.split_column(
            Layout(name="header", size=self.header_height),
            Layout(name="main", size=self.main_height),
            Layout(name="bottom", size=self.bottom_height)
        )
        
        # Header
        root["header"].update(self._build_header(bot, snapshot))
        
        # Main: Split Positions (Left) and Tape (Right)
        root["main"].split_row(
            Layout(name="positions", ratio=6),
            Layout(name="tape", ratio=4)
        )
        
        root["main"]["positions"].update(self._build_positions_panel(snapshot))
        root["main"]["tape"].update(self._build_tape_panel(bot))
        
        # Bottom: Split Logs (Left) and Jail (Right)
        root["bottom"].split_row(
            Layout(name="logs", ratio=8),
            Layout(name="jail", ratio=2)
        )
        
        root["bottom"]["logs"].update(self._build_log_panel())
        root["bottom"]["jail"].update(self._build_jail_panel(bot))
        
        return root

    def _build_header(self, bot, snapshot) -> Panel:
        perf = snapshot.performance
        risk = snapshot.risk
        
        # Calculate PnL Color and Symbol
        pnl = perf.realized_pnl + perf.unrealized_pnl
        pnl_color = "bright_green" if pnl >= 0 else "bright_red"
        pnl_sign = "+" if pnl >= 0 else "-"
        
        # Position count
        num_positions = risk.open_positions
        
        # ADAPTIVE ENTRY: Get current adaptive state
        adaptive = getattr(bot, 'adaptive_entry', None)
        if adaptive:
            adaptive_state = adaptive.get_state(num_positions)
            active_score = adaptive_state.current_threshold
            base_threshold = adaptive_state.base_threshold
            decay_adj = adaptive_state.decay_adjustment
            mins_idle = adaptive_state.minutes_since_last_entry
            avg_signal = adaptive_state.avg_signal_score
        else:
            # Fallback if adaptive not available
            active_score = float(getattr(bot, 'dynamic_gate_score', 35.0))
            base_threshold = active_score
            decay_adj = 0.0
            mins_idle = 0.0
            avg_signal = 0.0
        
        # Rate limit info
        pm = getattr(bot, 'position_manager', None)
        entries_this_min = len(getattr(pm, 'entry_times', [])) if pm else 0
        max_entries = int(getattr(bot.cfg, 'MAX_ENTRIES_PER_MIN', 5))
        
        # Grid layout for header
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        
        # Left: Positions + Rate
        left = Text()
        if num_positions == 0:
            left.append("SEEKING ", style="bold yellow")
            left.append(f"({mins_idle:.0f}m idle)\n", style="dim")
        else:
            left.append("ACTIVE ", style="bold green")
            left.append(f"| {num_positions} positions\n", style="dim")
        left.append(f"{num_positions}/{risk.max_positions} POS", style="bold white")
        left.append(f" | {entries_this_min}/{max_entries} ENT/MIN", style="dim")
        
        # Center: ADAPTIVE Scoring
        center = Text()
        # Color based on how aggressive (lower = more aggressive)
        if active_score <= 30:
            score_style = "bold red"
            mode_label = "HUNT"
        elif active_score <= 40:
            score_style = "bold yellow"
            mode_label = "SEEK"
        elif active_score <= 50:
            score_style = "bold green"
            mode_label = "NORM"
        else:
            score_style = "bold cyan"
            mode_label = "PICK"
        
        center.append(f"GATE: {active_score:.0f}+", style=score_style)
        center.append(f" [{mode_label}]\n", style="dim")
        # Show what's influencing threshold
        if decay_adj > 0:
            center.append(f"base={base_threshold:.0f} ", style="dim")
            center.append(f"-{decay_adj:.0f}decay", style="yellow")
        else:
            center.append(f"avg_sig={avg_signal:.0f}", style="dim")
        
        # Right: PnL Hero
        right = Text("EQUITY: ", style="dim")
        right.append(f"${perf.equity:.2f}\n", style="bold white")
        right.append(f"PnL: ", style="dim")
        right.append(f"{pnl_sign}${abs(pnl):.2f}", style=f"bold {pnl_color}")
        
        grid.add_row(left, center, right)
        
        return Panel(grid, style="blue", box=box.ROUNDED)

    def _build_positions_panel(self, snapshot) -> Panel:
        table = Table(expand=True, box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("SYM", style="bold white")
        table.add_column("SIDE", justify="center")
        table.add_column("PNL $", justify="right")
        table.add_column("PNL %", justify="right")
        table.add_column("SCORE", justify="center")
        table.add_column("AGE", justify="right")
        table.add_column("SIZE", justify="right")
        
        positions = getattr(snapshot, 'open_positions', []) or []
        
        if not positions:
            # Show message if empty
            return Panel(
                Text("\n\nNO ACTIVE POSITIONS\nSCANNING FOR TARGETS...", justify="center", style="dim"),
                title="[bold]ACTIVE POSITIONS[/bold]",
                border_style="dim",
                box=box.ROUNDED
            )
            
        for pos in positions:
            # Color coding
            pnl_c = "green" if pos.pnl_value >= 0 else "red"
            side_c = "green" if pos.side == "LONG" else "red"
            
            # ML Scorer: 25-50 range. 45+ exceptional, 40+ good, 35+ decent
            if pos.score >= 45:
                score_c = "bold magenta"  # Exceptional
            elif pos.score >= 40:
                score_c = "bold green"    # Good
            elif pos.score >= 35:
                score_c = "green"         # Decent
            else:
                score_c = "yellow"        # Marginal
            
            table.add_row(
                pos.symbol.split("/")[0],
                Text(pos.side[:1], style=side_c),
                Text(f"{pos.pnl_value:+.2f}", style=pnl_c),
                Text(f"{pos.pnl_pct:+.2f}%", style=pnl_c),
                Text(f"{pos.score:.0f}", style=score_c),
                pos.age_str,
                f"{pos.size_pct:.1f}%"
            )
            
        return Panel(table, title=f"[bold]ACTIVE POSITIONS ({len(positions)})[/bold]", border_style="blue", box=box.ROUNDED)

    def _build_tape_panel(self, bot) -> Panel:
        # Read directly from bot signal history (Direct Link)
        history = getattr(bot, 'signal_history', [])
        # Get last N items reversed
        tape_len = self.main_height - 2
        recent = list(reversed(history))[:tape_len]
        
        content = Text()
        
        for item in recent:
            sym = item.get('symbol', '?').split("/")[0]
            score = item.get('final_score', 0)
            approved = item.get('approved', False)
            reason = item.get('rejection_reason', '') or ''
            
            # Format
            time_str = time.strftime("%H:%M:%S", time.localtime(item.get('timestamp', 0)))
            
            if approved:
                status = "ENTER"
                style = "bold green"
                reason_display = ">> ORDER SENT"
            else:
                status = "SKIP"
                style = "dim"
                # Simplify reason for new ML system
                reason_lower = reason.lower()
                if "idle" in reason_lower and "score" in reason_lower:
                    reason_display = "Below IDLE threshold"
                elif "normal" in reason_lower and "score" in reason_lower:
                    reason_display = "Below NORMAL threshold"
                elif "rate limit" in reason_lower:
                    reason_display = "Rate limit hit"
                elif "max positions" in reason_lower:
                    reason_display = "Max positions"
                elif "blacklist" in reason_lower:
                    reason_display = "Blacklisted"
                elif "cooldown" in reason_lower:
                    reason_display = "Cooldown active"
                elif "spread" in reason_lower:
                    reason_display = "Spread too wide"
                elif "no signals" in reason_lower or "no pattern" in reason_lower:
                    reason_display = "No pattern"
                else:
                    reason_display = reason[:18] if reason else "Filtered"
            
            # ENHANCED TAPE: Add Volatility and Spread context
            vol = item.get('volatility', 0.0)
            spread = item.get('spread_bps', 0.0)
            
            # Volatility color
            vol_c = "green" if vol > 5.0 else "yellow" if vol > 2.0 else "dim"
            
            # Metrics display: [5.2% | 3bp]
            metrics = Text(f"[{vol:.1f}%|{spread:.0f}bp] ", style=vol_c)
            
            # Color score (ML SCORER V3: 25-50 range)
            if score >= 45:
                score_c = "bold magenta"
                status = "🦄 UNI"  # ML: 45+ = exceptional
            elif score >= 40:
                score_c = "bold cyan"  # ML: 40-44 = very good
            elif score >= 35:
                score_c = "green"  # ML: 35-39 = good
            elif score >= 30:
                score_c = "yellow"  # ML: 30-34 = decent
            else:
                score_c = "dim"
            
            line = Text()
            line.append(f"{time_str} ", style="dim")
            line.append(f"{sym:<6} ", style="bold white")
            line.append(f"{score:>4.1f} ", style=score_c)
            line.append(metrics) # Insert metrics before status/reason
            line.append(f"{status:<5} ", style=style)
            line.append(f"{reason_display}", style="dim")
            line.append("\n")
            
            content.append(line)
            
        return Panel(content, title="[bold]THE TAPE (LIVE SCAN)[/bold]", border_style="cyan", box=box.ROUNDED)

    def _build_log_panel(self) -> Panel:
        # Simple log tail
        from .logger import get_log_buffer
        buf = get_log_buffer()
        lines = buf.get_recent(limit=self.bottom_height - 2)
        
        content = Text()
        for log in lines:
            lvl = log['level']
            msg = log['message']
            c = "red" if lvl in ("ERROR", "CRITICAL") else "yellow" if lvl=="WARNING" else "dim"
            content.append(f"[{lvl[:1]}] {msg}\n", style=c)
            
        return Panel(content, title="[bold]SYSTEM LOG[/bold]", border_style="dim", box=box.ROUNDED)

    def _build_jail_panel(self, bot) -> Panel:
        strikes = len(getattr(bot, '_trash_strikes', {}))
        inmates = len(getattr(bot, '_trash_jail', {}))
        
        # Get last jailed symbol if any
        jail_dict = getattr(bot, '_trash_jail', {})
        last_jailed = list(jail_dict.keys())[-1] if jail_dict else "None"
        
        content = Text()
        content.append(f"STRIKES: ", style="dim")
        content.append(f"{strikes}\n", style="yellow")
        
        content.append(f"JAILED:  ", style="dim")
        content.append(f"{inmates}\n", style="bold red")
        
        content.append(f"LAST:    ", style="dim")
        content.append(f"{last_jailed[:6]}", style="dim white")
        
        return Panel(content, title="[bold]TRASH BIN[/bold]", border_style="dim", box=box.ROUNDED)
