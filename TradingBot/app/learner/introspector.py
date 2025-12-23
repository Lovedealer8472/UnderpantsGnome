"""
Trade Introspector - The main learning engine.

For each trade, asks:
- What went wrong / right?
- What could I have done better?
- How should I adjust for next time?

And learns by updating adaptive weights.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

from .counterfactual import CounterfactualSimulator, CounterfactualAnalysis
from .lessons import Lesson, LessonExtractor, LessonStore, MarketContext
from .weights import AdaptiveWeights, ContextWeights


class TradeIntrospector:
    """
    Main trade introspection engine.
    
    Analyzes each trade with hindsight, extracts lessons,
    and updates adaptive weights for future trades.
    
    Usage:
        introspector = TradeIntrospector()
        
        # After a trade closes
        lesson = introspector.analyze_trade(
            trade=trade_dict,
            ohlcv=market_data,
            entry_bar_idx=100,
        )
        
        # Get adjusted parameters for next trade
        adjustments = introspector.get_adjustments(context_key)
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the introspector.
        
        Args:
            data_dir: Directory for storing lessons and weights
            verbose: Whether to print learning progress
        """
        self.verbose = verbose
        
        if data_dir:
            import os
            os.environ["LEARNER_DATA_DIR"] = data_dir
        
        self.simulator = CounterfactualSimulator()
        self.extractor = LessonExtractor()
        self.lesson_store = LessonStore()
        self.weights = AdaptiveWeights()
        
        # Statistics
        self.trades_analyzed = 0
        self.lessons_with_improvement = 0
        self.total_improvement_potential = 0.0
        
        if self.verbose:
            print(f"[INTROSPECTOR] Initialized with {len(self.lesson_store.lessons)} existing lessons")
            print(f"[INTROSPECTOR] {len(self.weights.weights)} context-specific weight sets loaded")
    
    def analyze_trade(
        self,
        trade: Dict[str, Any],
        ohlcv: np.ndarray,
        entry_bar_idx: int,
        fees_pct: float = 0.1,
    ) -> Lesson:
        """
        Analyze a single trade with full hindsight.
        
        Args:
            trade: Trade dict with entry_price, exit_price, side, sl, tp, etc.
            ohlcv: OHLCV data (shape: N x 5) surrounding the trade
            entry_bar_idx: Index in ohlcv where entry occurred
            fees_pct: Round-trip fees as percentage
        
        Returns:
            Lesson extracted from the analysis
        """
        # Extract market context at entry
        context = self._extract_context(ohlcv, entry_bar_idx, trade)
        
        # Run counterfactual simulations
        analysis = self.simulator.analyze_trade(
            trade=trade,
            ohlcv=ohlcv,
            entry_bar_idx=entry_bar_idx,
            fees_pct=fees_pct,
        )
        
        # Extract lesson
        lesson = self.extractor.extract(trade, analysis, context)
        
        # Store lesson
        self.lesson_store.add(lesson)
        
        # Update weights based on lesson
        self.weights.update_from_lesson(lesson)
        
        # Update statistics
        self.trades_analyzed += 1
        if lesson.improvement > 0.1:
            self.lessons_with_improvement += 1
            self.total_improvement_potential += lesson.improvement
        
        if self.verbose and self.trades_analyzed % 100 == 0:
            self._print_progress()
        
        return lesson
    
    def _extract_context(
        self,
        ohlcv: np.ndarray,
        entry_idx: int,
        trade: Dict[str, Any],
    ) -> MarketContext:
        """Extract market context at the time of entry."""
        # Need at least some bars before entry for indicators
        lookback = min(entry_idx, 50)
        
        if lookback < 14:
            # Not enough data, return default context
            return MarketContext()
        
        closes = ohlcv[entry_idx - lookback:entry_idx + 1, 3]
        highs = ohlcv[entry_idx - lookback:entry_idx + 1, 1]
        lows = ohlcv[entry_idx - lookback:entry_idx + 1, 2]
        volumes = ohlcv[entry_idx - lookback:entry_idx + 1, 4]
        
        # Calculate RSI
        rsi = self._calculate_rsi(closes, 14)
        
        # Determine RSI bucket
        if rsi < 30:
            rsi_bucket = "oversold"
        elif rsi < 45:
            rsi_bucket = "neutral_low"
        elif rsi < 55:
            rsi_bucket = "neutral"
        elif rsi < 70:
            rsi_bucket = "neutral_high"
        else:
            rsi_bucket = "overbought"
        
        # Calculate momentum (% change over different periods)
        momentum_1h = (closes[-1] - closes[-12]) / closes[-12] * 100 if len(closes) >= 12 else 0
        momentum_4h = (closes[-1] - closes[-48]) / closes[-48] * 100 if len(closes) >= 48 else 0
        momentum_24h = trade.get("entry_features", {}).get("pct_change_24h", 0)
        
        # Calculate ATR for volatility
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        atr_pct = atr / closes[-1] * 100
        
        # Volatility bucket
        if atr_pct < 1.0:
            volatility = "low"
        elif atr_pct < 2.5:
            volatility = "medium"
        else:
            volatility = "high"
        
        # Trend (simple: above/below 20-period MA)
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        if closes[-1] > ma20 * 1.01:
            trend = "up"
        elif closes[-1] < ma20 * 0.99:
            trend = "down"
        else:
            trend = "sideways"
        
        # Volume ratio
        avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        return MarketContext(
            trend=trend,
            trend_strength=abs(closes[-1] - ma20) / ma20 if ma20 > 0 else 0,
            rsi=rsi,
            rsi_bucket=rsi_bucket,
            momentum_1h=momentum_1h,
            momentum_4h=momentum_4h,
            momentum_24h=momentum_24h,
            volatility=volatility,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
        )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_adjustments(self, context_key: str) -> Dict[str, Any]:
        """
        Get learned adjustments for a given market context.
        
        Use this before entering a trade to apply learned modifications.
        """
        return self.weights.get_weights(context_key)
    
    def apply_to_trade_setup(
        self,
        context_key: str,
        entry_price: float,
        default_sl: float,
        default_tp: float,
    ) -> Dict[str, Any]:
        """
        Apply learned weights to a trade setup.
        
        Returns adjusted SL, TP, entry delay, and skip recommendation.
        """
        return self.weights.apply_to_signal(
            context_key, entry_price, default_sl, default_tp
        )
    
    def _print_progress(self) -> None:
        """Print learning progress."""
        print(f"\n[INTROSPECTOR] Progress: {self.trades_analyzed} trades analyzed")
        print(f"  - Trades with improvement potential: {self.lessons_with_improvement}")
        print(f"  - Average improvement potential: {self.total_improvement_potential / max(1, self.lessons_with_improvement):.2f}%")
        print(f"  - Context weights learned: {len(self.weights.weights)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get introspection statistics."""
        return {
            "trades_analyzed": self.trades_analyzed,
            "lessons_with_improvement": self.lessons_with_improvement,
            "total_improvement_potential": self.total_improvement_potential,
            "avg_improvement_potential": self.total_improvement_potential / max(1, self.lessons_with_improvement),
            "contexts_learned": len(self.weights.weights),
            "total_lessons": len(self.lesson_store.lessons),
            "weights_summary": self.weights.get_summary(),
        }
    
    def report(self) -> str:
        """Generate comprehensive report."""
        lines = [
            "=" * 70,
            "TRADE INTROSPECTION LEARNING REPORT",
            "=" * 70,
            "",
            f"Trades Analyzed: {self.trades_analyzed}",
            f"Lessons Stored: {len(self.lesson_store.lessons)}",
            f"Trades with Improvement Potential: {self.lessons_with_improvement} ({self.lessons_with_improvement / max(1, self.trades_analyzed) * 100:.1f}%)",
            f"Total Improvement Potential: {self.total_improvement_potential:.1f}%",
            f"Context Weights Learned: {len(self.weights.weights)}",
            "",
        ]
        
        # Context summary
        context_summary = self.lesson_store.get_context_summary()
        if context_summary:
            lines.append("Top Contexts by Trade Count:")
            sorted_contexts = sorted(
                context_summary.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            for key, s in sorted_contexts[:10]:
                lines.append(f"  {key}:")
                lines.append(f"    Trades: {s['count']} | WR: {s['win_rate']:.1%} | Avg Improvement: {s['avg_improvement']:.2f}%")
                if s["skip_rate"] > 0.1:
                    lines.append(f"    Skip Recommended: {s['skip_rate']:.1%}")
                if s["entry_delay_count"] > 0:
                    lines.append(f"    Avg Entry Delay: {s['avg_entry_delay']:.1f} bars")
        
        lines.append("")
        lines.append(self.weights.report())
        
        return "\n".join(lines)


def run_introspection_on_trades(
    trades: List[Dict[str, Any]],
    ohlcv_by_symbol: Dict[str, np.ndarray],
    data_dir: Optional[str] = None,
    verbose: bool = True,
) -> TradeIntrospector:
    """
    Run introspection on a list of historical trades.
    
    Args:
        trades: List of trade dicts (from backtest)
        ohlcv_by_symbol: Dict mapping symbol -> OHLCV array
        data_dir: Directory for storing learner data
        verbose: Print progress
    
    Returns:
        TradeIntrospector with updated weights
    """
    introspector = TradeIntrospector(data_dir=data_dir, verbose=verbose)
    
    for trade in trades:
        symbol = trade.get("symbol", "")
        if symbol not in ohlcv_by_symbol:
            continue
        
        ohlcv = ohlcv_by_symbol[symbol]
        
        # Find entry bar index (assuming entry_time is timestamp)
        entry_time = trade.get("entry_time", 0)
        
        # This is approximate - in real use, you'd match timestamps
        # For now, assume trade has entry_bar_idx or we estimate from position in data
        entry_bar_idx = trade.get("entry_bar_idx", len(ohlcv) // 2)
        
        if entry_bar_idx < 50 or entry_bar_idx >= len(ohlcv) - 50:
            continue  # Need context around the trade
        
        try:
            introspector.analyze_trade(
                trade=trade,
                ohlcv=ohlcv,
                entry_bar_idx=entry_bar_idx,
            )
        except Exception as e:
            if verbose:
                print(f"[INTROSPECTOR] Error analyzing trade: {e}")
    
    if verbose:
        print("\n" + introspector.report())
    
    return introspector
